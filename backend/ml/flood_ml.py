from __future__ import annotations

"""Production-style Flood v1 ML pipeline.

The predictive layer is supervised ML trained from the real India Flood Inventory.
No hand-written risk weights are used for the flood hazard prediction.

Training data:
- India Flood Inventory v3, 1967-2023, HydroSense Lab / IIT Delhi / IMD
- Rainfall feature source defaults to observed rainfall mentioned in the IMD event
  descriptions.  Optional event-level IPED or other precipitation tables can be
  supplied via FLOOD_RAINFALL_SOURCE=csv and FLOOD_RAINFALL_PATH.

The model is XGBoost when available, with Random Forest as a baseline.  The
training pipeline uses a chronological split, randomized hyper-parameter search,
held-out test metrics, permutation importance and best-effort SHAP explanations.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional only during import
    XGBClassifier = None  # type: ignore

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_ROOT = BASE_DIR / "models" / "flood" / "v1"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

SOURCE_URL = "https://zenodo.org/records/16994648/files/India_Flood_Inventory_v3.csv?download=1"
SOURCE_NAME = "India Flood Inventory v3 (1967-2023), HydroSense Lab / IIT Delhi / IMD, latest Zenodo record 10.5281/zenodo.16994648"
SOURCE_MD5 = "fea75a9ff9eba8fb328eaddfacd21d67"
MIN_DISTINCT_YEARS = 8
DEFAULT_LOCAL_PATH = DATA_DIR / "India_Flood_Inventory_v3.csv"
MODEL_PATH = MODEL_ROOT / "model.joblib"
METRICS_PATH = MODEL_ROOT / "metrics.json"
METADATA_PATH = MODEL_ROOT / "metadata.json"
IMPORTANCE_PATH = MODEL_ROOT / "feature_importance.json"
SHAP_PATH = MODEL_ROOT / "shap_importance.json"

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "State",
    "month_sin",
    "month_cos",
    "duration_days",
    "affected_districts",
    "latitude",
    "longitude",
    "rainfall_24h_mm",
    "rainfall_missing",
]
# Main Cause is intentionally NOT a model feature because the supervised target
# can be derived from explicit observed severity words in that field when the
# source Severity column is blank. Excluding it prevents target leakage.
CATEGORICAL_FEATURES = ["State"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def _normalise_text(value: Any, fallback: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _count_list(value: Any) -> int:
    text = _normalise_text(value, "")
    if not text:
        return 0
    return sum(1 for item in re.split(r"[,;]", text) if item.strip() and item.strip().lower() != "none")


def _extract_rainfall_mm(row: pd.Series) -> float:
    """Extract directly observed rainfall mentioned in the IMD event text.

    This avoids inventing rainfall. When event text contains no rainfall value,
    the feature is explicitly missing and is handled by the trained preprocessing.
    """
    blob = " ".join(
        str(row.get(column, "") or "")
        for column in (
            "Description of Casualties/injured",
            "Extent of damage ",
            "Description of Casualties/injured ",
        )
    )
    values: list[float] = []
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:cm|cms|centimeters?)\b", 10.0),
        (r"(\d+(?:\.\d+)?)\s*mm\b", 1.0),
    ]
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, blob.lower()):
            try:
                value = float(match.group(1)) * multiplier
            except ValueError:
                continue
            if 0 < value <= 2500:
                values.append(value)
    return float(max(values)) if values else np.nan


def _prepare_optional_precipitation(raw: pd.DataFrame) -> pd.DataFrame:
    """Optional event-level precipitation enrichment.

    If FLOOD_RAINFALL_SOURCE=csv is configured, the file must contain one row per
    event and either UEI or Start Date plus rainfall_24h_mm. This adapter is
    intentionally generic so an IPED-derived event table can be plugged in later
    without changing the model code.
    """
    source = os.getenv("FLOOD_RAINFALL_SOURCE", "event_text").lower().strip()
    if source != "csv":
        return raw
    path = Path(os.getenv("FLOOD_RAINFALL_PATH", ""))
    if not path.exists():
        raise FileNotFoundError(
            "FLOOD_RAINFALL_SOURCE=csv but FLOOD_RAINFALL_PATH does not exist. "
            "Provide an event-level precipitation table, such as an IPED-derived table."
        )
    rain = pd.read_csv(path)
    required = {"rainfall_24h_mm"}
    if not required.issubset(rain.columns):
        raise ValueError("Rainfall enrichment CSV must contain rainfall_24h_mm")
    if "UEI" in raw.columns and "UEI" in rain.columns:
        rain = rain[["UEI", "rainfall_24h_mm"]].drop_duplicates("UEI")
        return raw.merge(rain, on="UEI", how="left", suffixes=("", "_external"))
    if "Start Date" in raw.columns and "Start Date" in rain.columns:
        left = raw.copy()
        right = rain[["Start Date", "rainfall_24h_mm"]].drop_duplicates("Start Date")
        return left.merge(right, on="Start Date", how="left", suffixes=("", "_external"))
    raise ValueError("Rainfall enrichment CSV must share UEI or Start Date with the IFI dataset")


def _extract_observed_severity_label(row: pd.Series) -> str | float:
    """Extract an explicit observed severity label from the IFI event record.

    The current IFI v3 event CSV contains a Severity column, but many records have
    it blank while the Main Cause field contains explicit phrases such as
    ``moderate``, ``severe`` or ``very severe``. These phrases are part of the
    historical observation, not generated labels. We use them only as the target
    when Severity is unavailable, and never expose Main Cause to the predictor.
    """
    text = _normalise_text(row.get("Severity", ""), "").upper()
    if text and text != "UNKNOWN":
        return text
    cause = _normalise_text(row.get("Main Cause", ""), "").lower()
    if "very severe" in cause:
        return "VERY_SEVERE"
    if "severe" in cause:
        return "SEVERE"
    if "moderate" in cause:
        return "MODERATE"
    return np.nan


def load_event_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or Path(os.getenv("FLOOD_DATASET_PATH", str(DEFAULT_LOCAL_PATH)))
    if not path.exists():
        raise FileNotFoundError(f"Flood dataset not found: {path}")
    df = pd.read_csv(path)
    expected = {"UEI", "Start Date", "Duration(Days)", "Main Cause", "Districts", "State"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Flood event dataset is missing required columns: {sorted(missing)}")
    df = _prepare_optional_precipitation(df)
    return df


def _normalise_target(raw: pd.Series) -> tuple[pd.Series, dict[str, str], str]:
    """Return integer classes, class-name mapping, and target mode."""
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() == 0:
        labels = raw.map(lambda x: _normalise_text(x, "UNKNOWN").upper())
        uniques = sorted(labels.dropna().unique())
        if len(uniques) < 2:
            raise ValueError("Flood dataset target contains fewer than two classes")
        mapping = {label: idx for idx, label in enumerate(uniques)}
        encoded = labels.map(mapping).astype(int)
        names = {str(v): str(k) for k, v in enumerate(uniques)}
        return encoded, names, "categorical"

    clean = numeric.dropna()
    integer_like = np.all(np.isclose(clean, np.round(clean)))
    unique = sorted(clean.unique().tolist())
    if integer_like and len(unique) <= 8:
        mapping = {float(value): idx for idx, value in enumerate(unique)}
        encoded = numeric.map(lambda x: mapping.get(float(x)) if pd.notna(x) else np.nan)
        names = {str(idx): str(int(value) if float(value).is_integer() else value) for value, idx in mapping.items()}
        if len(unique) < 2:
            raise ValueError("Flood dataset target contains fewer than two classes")
        return encoded.astype("Int64"), names, "ordinal_numeric"

    # If the source has a continuous severity score, convert to four classes using
    # quantiles learned ONLY from the training partition later in train_flood_model.
    return numeric, {}, "continuous_needs_binning"


def _bin_continuous_target(train_target: pd.Series, values: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    clean = pd.to_numeric(train_target, errors="coerce").dropna().astype(float)
    cuts = sorted(set(float(x) for x in np.quantile(clean, [0.25, 0.5, 0.75]).tolist()))
    while len(cuts) < 3:
        cuts.append(float(clean.max()))
    edges = [-np.inf, cuts[0], cuts[1], cuts[2], np.inf]
    labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    binned = pd.cut(pd.to_numeric(values, errors="coerce"), bins=edges, labels=labels, include_lowest=True)
    mapping = {str(i): labels[i] for i in range(4)}
    # re-encode the categorical values consistently
    encoded = binned.map({label: i for i, label in enumerate(labels)})
    return encoded.astype("Int64"), mapping


def _resolve_event_year(df: pd.DataFrame) -> pd.Series:
    """Resolve event year robustly from Start Date, date text, or UEI.

    IFI v3 uses dates like ``02-07-1967 00:00``. Some pandas builds can
    infer a single format too aggressively; using ``format="mixed"`` plus the
    stable UEI year prevents a valid official dataset from being rejected with
    ``0 distinct years``.
    """
    raw_date = df["Start Date"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    parsed = pd.to_datetime(raw_date, errors="coerce", format="mixed", dayfirst=True)
    year = pd.to_numeric(parsed.dt.year, errors="coerce")
    year = year.fillna(
        pd.to_numeric(raw_date.str.extract(r"(?P<year>19\d{2}|20\d{2})", expand=False), errors="coerce")
    )
    if "UEI" in df.columns:
        uei_year = pd.to_numeric(
            df["UEI"].astype(str).str.extract(
                r"UEI-[^-]+-FL-(?P<year>19\d{2}|20\d{2})-", expand=False
            ),
            errors="coerce",
        )
        year = year.fillna(uei_year)
    return year


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    start_raw = df["Start Date"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    dates = pd.to_datetime(start_raw, errors="coerce", format="mixed", dayfirst=True)
    resolved_year = _resolve_event_year(df)
    months = dates.dt.month.astype("float64")
    months = months.fillna(7.0)
    angle = 2.0 * math.pi * (months - 1.0) / 12.0

    out = pd.DataFrame(index=df.index)
    out["State"] = df["State"].map(lambda x: _normalise_text(x, "UNKNOWN")).str.upper()
    out["Main Cause"] = df["Main Cause"].map(lambda x: _normalise_text(x, "FLOOD")).str.upper()
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["duration_days"] = pd.to_numeric(df["Duration(Days)"], errors="coerce").clip(lower=1, upper=365)
    out["affected_districts"] = df["Districts"].map(_count_list).clip(lower=0, upper=900)
    out["latitude"] = pd.to_numeric(df.get("Latitude"), errors="coerce") if "Latitude" in df.columns else np.nan
    out["longitude"] = pd.to_numeric(df.get("Longitude"), errors="coerce") if "Longitude" in df.columns else np.nan

    if "rainfall_24h_mm_external" in df.columns:
        external = pd.to_numeric(df["rainfall_24h_mm_external"], errors="coerce")
        text_rain = df.apply(_extract_rainfall_mm, axis=1)
        out["rainfall_24h_mm"] = external.fillna(text_rain)
    elif "rainfall_24h_mm" in df.columns:
        external = pd.to_numeric(df["rainfall_24h_mm"], errors="coerce")
        text_rain = df.apply(_extract_rainfall_mm, axis=1)
        out["rainfall_24h_mm"] = external.fillna(text_rain)
    else:
        out["rainfall_24h_mm"] = df.apply(_extract_rainfall_mm, axis=1)
    out["rainfall_missing"] = out["rainfall_24h_mm"].isna().astype(float)
    out["__year"] = resolved_year
    out["__raw_target"] = df.apply(_extract_observed_severity_label, axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline([( "imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _rf_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", _make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_SEED,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ])


def _xgb_pipeline() -> Pipeline:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for the final Flood v1 model. Install it with: pip install xgboost")
    return Pipeline([
        ("preprocessor", _make_preprocessor()),
        (
            "model",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=450,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                n_jobs=4,
                tree_method="hist",
            ),
        ),
    ])


def _dummy_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", _make_preprocessor()),
        ("model", DummyClassifier(strategy="prior")),
    ])


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in df["__year"].dropna().unique())
    if len(years) < MIN_DISTINCT_YEARS:
        raise ValueError(
            f"Need at least {MIN_DISTINCT_YEARS} distinct years for chronological train/validation/test split; "
            f"found {len(years)}. The IFI v3 source should span 1967-2023. "
            "The cached dataset is likely stale or incomplete; refresh it instead of training on a partial file."
        )
    train_end = max(1, int(len(years) * 0.70))
    val_end = max(train_end + 1, int(len(years) * 0.85))
    train_years = set(years[:train_end])
    val_years = set(years[train_end:val_end])
    test_years = set(years[val_end:])
    train = df[df["__year"].isin(train_years)].copy()
    val = df[df["__year"].isin(val_years)].copy()
    test = df[df["__year"].isin(test_years)].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    return train, val, test


def _classification_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, class_names: dict[str, str]) -> dict[str, Any]:
    pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    labels = sorted(set(int(v) for v in y.tolist()) | set(int(v) for v in pred.tolist()))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
        "classes": [class_names.get(str(i), str(i)) for i in labels],
    }
    if proba is not None and len(labels) >= 2:
        try:
            if len(labels) == 2:
                result["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                result["pr_auc"] = float(average_precision_score(y, proba[:, 1]))
            else:
                result["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
                result["pr_auc"] = float(average_precision_score(pd.get_dummies(y).reindex(columns=range(proba.shape[1]), fill_value=0), proba, average="macro"))
        except Exception:
            result["roc_auc"] = None
            result["pr_auc"] = None
    return result


def _balanced_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    n = len(y)
    k = max(1, len(counts))
    return y.map({cls: n / (k * count) for cls, count in counts.items()}).to_numpy(dtype=float)


def _dataset_source() -> dict[str, str]:
    return {
        "provider": os.getenv("FLOOD_DATASET_PROVIDER", "zenodo"),
        "source_url": SOURCE_URL,
        "precipitation_source": os.getenv("FLOOD_RAINFALL_SOURCE", "event_text"),
        "precipitation_note": (
            "Rainfall values are extracted from observed IMD event descriptions unless "
            "an event-level IPED/precipitation CSV is configured. No synthetic rainfall is generated."
        ),
    }


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset_file(path: Path, *, strict_md5: bool = False) -> tuple[bool, str]:
    """Validate that the cached file is the real multi-year IFI event CSV.

    This deliberately rejects stale/truncated/toy files so the training command can
    never silently train on the old 118-row DFSI excerpt or another partial cache.
    """
    if not path.exists():
        return False, "file does not exist"
    if path.stat().st_size < 500_000:
        return False, f"file too small ({path.stat().st_size} bytes)"
    try:
        header = pd.read_csv(path, nrows=5)
    except Exception as exc:
        return False, f"CSV read failed: {exc}"
    expected = {"UEI", "Start Date", "Duration(Days)", "Main Cause", "Districts", "State"}
    missing = expected - set(header.columns)
    if missing:
        return False, f"missing required columns: {sorted(missing)}"
    try:
        sample = pd.read_csv(path, usecols=["Start Date", "UEI"], low_memory=False)
        resolved = _resolve_event_year(sample)
        years = int(resolved.dropna().astype(int).nunique())
        rows = int(len(sample))
    except Exception as exc:
        return False, f"date/year validation failed: {exc}"
    if years < MIN_DISTINCT_YEARS:
        return False, f"only {years} distinct years detected; expected at least {MIN_DISTINCT_YEARS}"
    if rows < 1000:
        return False, f"only {rows} event rows detected; expected the full IFI inventory"
    if strict_md5:
        actual = _file_md5(path)
        if actual != SOURCE_MD5:
            return False, f"MD5 mismatch: expected {SOURCE_MD5}, got {actual}"
    return True, f"{rows} rows across {years} distinct years"


def _download_dataset() -> Path:
    provider = os.getenv("FLOOD_DATASET_PROVIDER", "zenodo").lower().strip()
    if provider == "local":
        path = Path(os.getenv("FLOOD_DATASET_PATH", str(DEFAULT_LOCAL_PATH)))
        ok, reason = _validate_dataset_file(path, strict_md5=False)
        if not ok:
            raise ValueError(f"FLOOD_DATASET_PROVIDER=local but dataset is invalid: {reason}. Provide the full IFI v3 CSV.")
        return path
    if provider == "kaggle":
        dataset_ref = os.getenv("KAGGLE_DATASET")
        if not dataset_ref:
            raise ValueError("KAGGLE_DATASET must be set when FLOOD_DATASET_PROVIDER=kaggle")
        try:
            import kagglehub  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install kagglehub to use Kaggle ingestion: pip install kagglehub") from exc
        downloaded = Path(kagglehub.dataset_download(dataset_ref))
        requested_file = os.getenv("KAGGLE_FILE")
        if requested_file:
            candidate = downloaded / requested_file
            if not candidate.exists():
                raise FileNotFoundError(f"KAGGLE_FILE was not found in downloaded dataset: {requested_file}")
            return candidate
        csvs = list(downloaded.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset")
        return csvs[0]

    # Zenodo cache: validate before reuse. The previous project could leave a
    # partial/stale CSV on disk, which caused the chronological split to see only
    # a handful of years. If that happens, force a fresh download of the official
    # 1.8 MB IFI v3 file. FLOOD_FORCE_REFRESH=1 can also be used manually.
    force_refresh = os.getenv("FLOOD_FORCE_REFRESH", "0").strip().lower() in {"1", "true", "yes"}
    ok, reason = _validate_dataset_file(DEFAULT_LOCAL_PATH, strict_md5=True) if DEFAULT_LOCAL_PATH.exists() and not force_refresh else (False, "forced refresh")
    if ok:
        return DEFAULT_LOCAL_PATH

    if DEFAULT_LOCAL_PATH.exists():
        try:
            DEFAULT_LOCAL_PATH.unlink()
        except OSError:
            pass

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(SOURCE_URL)
        response.raise_for_status()
        DEFAULT_LOCAL_PATH.write_bytes(response.content)

    ok, reason = _validate_dataset_file(DEFAULT_LOCAL_PATH, strict_md5=True)
    if not ok:
        raise RuntimeError(
            "Downloaded flood dataset failed validation: " + reason + ". "
            "The official IFI v3 file is expected to contain 1967-2023 events. "
            "Delete the cached file and retry with a working internet connection."
        )
    return DEFAULT_LOCAL_PATH


def _top_shap_importance(final_model: Pipeline, X_test: pd.DataFrame) -> list[dict[str, float]]:
    if shap is None:
        return []
    try:
        preprocessor = final_model.named_steps["preprocessor"]
        model = final_model.named_steps["model"]
        transformed = preprocessor.transform(X_test)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out().tolist()
        sample_n = min(300, transformed.shape[0])
        sample = transformed[:sample_n]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            stacked = np.stack([np.abs(np.asarray(v)) for v in values], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)
        rows = [
            {"feature": name, "importance_mean_abs_shap": float(value)}
            for name, value in zip(feature_names, mean_abs)
        ]
        rows.sort(key=lambda x: x["importance_mean_abs_shap"], reverse=True)
        return rows[:20]
    except Exception:
        return []


def train_flood_model() -> dict[str, Any]:
    dataset_path = _download_dataset()
    raw = load_event_dataset(dataset_path)
    frame = build_features(raw)
    frame = frame.dropna(subset=["__year", "__raw_target"]).copy()
    frame["__year"] = frame["__year"].astype(int)

    train, val, test = _chronological_split(frame)
    target_source = "IFI Severity field, falling back to explicit severity words in Main Cause"
    usable_target = frame["__raw_target"].notna()
    frame = frame[usable_target].copy()
    if len(frame) < 500:
        raise ValueError(
            f"Only {len(frame)} events have an observed severity label. The real IFI v3 dataset should provide substantially more. "
            "Verify that the full India_Flood_Inventory_v3.csv was downloaded."
        )
    train, val, test = _chronological_split(frame)
    labels_order = ["MODERATE", "SEVERE", "VERY_SEVERE"]
    present = [label for label in labels_order if label in set(frame["__raw_target"].astype(str).str.upper())]
    if len(present) < 2:
        # If the source contains other categorical labels, keep those rather than fabricating classes.
        present = sorted(set(frame["__raw_target"].astype(str).str.upper()))
    mapping = {label: idx for idx, label in enumerate(present)}
    y_train = train["__raw_target"].astype(str).str.upper().map(mapping)
    y_val = val["__raw_target"].astype(str).str.upper().map(mapping)
    y_test = test["__raw_target"].astype(str).str.upper().map(mapping)
    if y_train.isna().any() or y_val.isna().any() or y_test.isna().any():
        unknown = sorted(set(frame["__raw_target"].astype(str).str.upper()) - set(mapping))
        raise ValueError(f"Unexpected target classes after normalization: {unknown}")
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    class_names = {str(i): label for label, i in mapping.items()}
    target_mode = "observed_severity_text"

    feature_cols = FEATURE_COLUMNS
    X_train = train[feature_cols].reset_index(drop=True)
    X_val = val[feature_cols].reset_index(drop=True)
    X_test = test[feature_cols].reset_index(drop=True)
    y_train = pd.Series(y_train.to_numpy(), name="target")
    y_val = pd.Series(y_val.to_numpy(), name="target")
    y_test = pd.Series(y_test.to_numpy(), name="target")

    classes = sorted(set(y_train.unique().tolist()))
    if len(classes) < 2:
        raise ValueError("Training data contains fewer than two target classes after preprocessing")

    weights = _balanced_weights(y_train)
    candidates: dict[str, Pipeline] = {"RandomForestClassifier": _rf_pipeline()}
    if XGBClassifier is not None:
        candidates["XGBoostClassifier"] = _xgb_pipeline()

    validation_metrics: dict[str, dict[str, Any]] = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train, model__sample_weight=weights)
        validation_metrics[name] = _classification_metrics(pipe, X_val, y_val, class_names)

    best_name = max(validation_metrics, key=lambda name: (validation_metrics[name]["f1_macro"], validation_metrics[name]["accuracy"]))

    selected = candidates[best_name]
    best_params: dict[str, Any] = {}
    if best_name == "XGBoostClassifier":
        param_space = {
            "model__n_estimators": [250, 400, 550],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__min_child_weight": [1, 2, 4],
        }
        search = RandomizedSearchCV(
            selected,
            param_distributions=param_space,
            n_iter=8,
            scoring="f1_macro",
            cv=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train, model__sample_weight=weights)
        selected = search.best_estimator_
        best_params = {k: v for k, v in search.best_params_.items()}
        validation_metrics["XGBoostClassifier_tuned"] = _classification_metrics(selected, X_val, y_val, class_names)
        tuned_name = "XGBoostClassifier_tuned"
        if validation_metrics[tuned_name]["f1_macro"] >= validation_metrics[best_name]["f1_macro"]:
            best_name = tuned_name

    # Baseline dummy model for comparison.
    dummy = _dummy_pipeline()
    dummy.fit(X_train, y_train)
    validation_metrics["DummyClassifier"] = _classification_metrics(dummy, X_val, y_val, class_names)

    # Refit selected model on train + validation. Test is untouched until here.
    trainval = pd.concat([train, val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    trainval_weights = _balanced_weights(y_trainval)
    final_model: Pipeline = selected
    final_model.fit(trainval[feature_cols], y_trainval, model__sample_weight=trainval_weights)

    test_metrics = _classification_metrics(final_model, X_test, y_test, class_names)
    dummy_test = _classification_metrics(dummy, X_test, y_test, class_names)

    permutation = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_SEED,
        scoring="f1_macro",
    )
    importance = [
        {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
        for feature, mean, std in zip(feature_cols, permutation.importances_mean, permutation.importances_std)
    ]
    importance.sort(key=lambda row: row["importance_mean"], reverse=True)
    shap_importance = _top_shap_importance(final_model, X_test)

    metadata = {
        "trained": True,
        "model_version": "flood-v2",
        "algorithm": best_name,
        "baseline_algorithm": "RandomForestClassifier",
        "dataset": SOURCE_NAME,
        "dataset_file": dataset_path.name,
        **_dataset_source(),
        "training_years": [int(train["__year"].min()), int(train["__year"].max())],
        "validation_years": [int(val["__year"].min()), int(val["__year"].max())],
        "test_years": [int(test["__year"].min()), int(test["__year"].max())],
        "rows": {"all": int(len(frame)), "train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "features": feature_cols,
        "target": "Observed flood severity label; uses IFI Severity when populated, otherwise explicit moderate/severe/very-severe wording from Main Cause. Main Cause is excluded from features to prevent leakage.",
        "target_source": target_source,
        "target_mode": target_mode,
        "classes": {str(k): class_names.get(str(k), str(k)) for k in sorted(set(y_trainval.unique().tolist()))},
        "random_seed": RANDOM_SEED,
        "best_params": best_params,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": "Chronological train/validation/test split; test set is held out until final evaluation.",
        "training_note": "Predictive hazard relationships are learned by the ML model. Operational exposure, routing, resources and simulation remain separate deterministic decision-support layers.",
    }
    metrics = {
        "validation_candidates": validation_metrics,
        "selected_model": best_name,
        "test": test_metrics,
        "dummy_test": dummy_test,
        "test_f1_improvement_vs_dummy": float(test_metrics["f1_macro"] - dummy_test["f1_macro"]),
    }

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_columns": feature_cols,
        "class_names": class_names,
        "metadata": metadata,
        "metrics": metrics,
        "feature_importance": importance,
        "shap_importance": shap_importance,
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    SHAP_PATH.write_text(json.dumps(shap_importance, indent=2), encoding="utf-8")

    return metadata | {"metrics": metrics, "feature_importance": importance, "shap_importance": shap_importance}


def model_status() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return {
            "trained": False,
            "model_version": "flood-v2",
            "message": "No trained flood model artifact exists. Run: python -m ml.training.train_flood_model",
        }
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    importance = json.loads(IMPORTANCE_PATH.read_text(encoding="utf-8")) if IMPORTANCE_PATH.exists() else []
    shap_importance = json.loads(SHAP_PATH.read_text(encoding="utf-8")) if SHAP_PATH.exists() else []
    return metadata | {"metrics": metrics, "feature_importance": importance, "shap_importance": shap_importance}


def _load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained flood model. Run: python -m ml.training.train_flood_model")
    return joblib.load(MODEL_PATH)


def _feature_row(
    *,
    state: str,
    rainfall_mm: float,
    month: int,
    duration_days: float,
    affected_districts: int,
    affected_states: int,
    main_cause: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    month_i = max(1, min(12, int(month)))
    angle = 2.0 * math.pi * (month_i - 1) / 12.0
    return pd.DataFrame([{
        "State": _normalise_text(state, "UNKNOWN").upper(),
        "month_sin": math.sin(angle),
        "month_cos": math.cos(angle),
        "duration_days": max(1.0, float(duration_days)),
        "affected_districts": max(0, float(affected_districts)),
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_24h_mm": max(0.0, float(rainfall_mm)),
        "rainfall_missing": 0.0,
    }])


def predict_flood(
    *,
    state: str,
    rainfall_mm: float,
    month: int,
    duration_days: float = 1.0,
    affected_districts: int = 1,
    affected_states: int = 1,
    main_cause: str = "FLOOD",
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    artifact = _load_artifact()
    model: Pipeline = artifact["model"]
    row = _feature_row(
        state=state,
        rainfall_mm=rainfall_mm,
        month=month,
        duration_days=duration_days,
        affected_districts=affected_districts,
        affected_states=affected_states,
        main_cause=main_cause,
        latitude=latitude,
        longitude=longitude,
    )
    proba = model.predict_proba(row)[0]
    model_classes = [int(v) for v in model.named_steps["model"].classes_]
    class_names = artifact.get("class_names", {})
    probabilities = {
        class_names.get(str(cls), str(cls)): float(prob)
        for cls, prob in zip(model_classes, proba)
    }
    predicted_class = model_classes[int(np.argmax(proba))]
    predicted_name = class_names.get(str(predicted_class), str(predicted_class))
    severe_names = {"HIGH", "CRITICAL", "SEVERE", "VERY SEVERE"}
    severe_probability = sum(prob for name, prob in probabilities.items() if str(name).upper() in severe_names)
    if severe_probability == 0 and probabilities:
        ordered = list(probabilities.values())
        severe_probability = ordered[-1]
    top_feature_rows = artifact.get("feature_importance", [])[:8]
    shap_rows = artifact.get("shap_importance", [])[:8]
    return {
        "model_version": artifact["metadata"]["model_version"],
        "algorithm": artifact["metadata"]["algorithm"],
        "prediction": predicted_name,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "severe_probability": float(severe_probability),
        "risk_score": float(severe_probability * 100.0),
        "band": predicted_name,
        "features_used": row.iloc[0].to_dict(),
        "top_features": top_feature_rows,
        "shap_importance": shap_rows,
        "dataset": artifact["metadata"]["dataset"],
    }
