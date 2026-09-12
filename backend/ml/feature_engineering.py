from __future__ import annotations

from .flood_ml import FEATURE_COLUMNS, build_features


def rainfall_signal(rainfall_24h_mm: float) -> float:
    """Legacy display helper; not a prediction formula."""
    return max(0.0, float(rainfall_24h_mm))


__all__ = ["FEATURE_COLUMNS", "build_features", "rainfall_signal"]
