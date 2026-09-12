from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hexadecimal hash for a local file."""
    path = Path(filepath)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_manifest(
    dataset: str,
    provider: str,
    source_url: str,
    license: str,
    record_count: int,
    generated_by: str,
    transformation: str,
    filepath: Path | str | None = None,
    sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and persist a standardized dataset manifest."""
    file_sha = sha256 or (compute_sha256(filepath) if filepath else "")
    now = datetime.now(timezone.utc).isoformat()

    manifest_data = {
        "dataset": dataset,
        "provider": provider,
        "source_url": source_url,
        "retrieved_at": now,
        "license": license,
        "sha256": file_sha,
        "record_count": record_count,
        "generated_by": generated_by,
        "transformation": transformation,
    }
    if extra:
        manifest_data.update(extra)

    manifest_file = MANIFEST_DIR / f"{dataset}.manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def load_manifest(dataset: str) -> dict[str, Any] | None:
    """Load an existing dataset manifest by dataset name."""
    manifest_file = MANIFEST_DIR / f"{dataset}.manifest.json"
    if not manifest_file.exists():
        return None
    try:
        with manifest_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_all_manifests() -> list[dict[str, Any]]:
    """Retrieve all dataset manifests stored in the manifest registry."""
    manifests = []
    for p in sorted(MANIFEST_DIR.glob("*.manifest.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                manifests.append(json.load(f))
        except Exception:
            pass
    return manifests
