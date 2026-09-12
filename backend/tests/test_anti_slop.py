"""Automated Anti-Slop Check.

Verifies that no prototype identifiers from the synthetic Harbor City world
(e.g., Harbor Parkway, Harbor General Hospital, z01, s02, road_01, res_01)
remain in the active production implementation.
"""

from pathlib import Path
import pytest

FORBIDDEN_IDENTIFIERS = [
    "Harbor City",
    "Harbor Parkway",
    "Harbor General Hospital",
    "Downtown Convention Shelter",
    "z01",
    "s02",
    "h01",
    "road_01",
    "res_01",
]

BACKEND_APP_DIR = Path(__file__).resolve().parent.parent / "app"
BACKEND_ML_DIR = Path(__file__).resolve().parent.parent / "ml"

EXCLUDED_PATHS = [
    "legacy",
    "tests",
    "__pycache__",
]


def _is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_PATHS:
            return True
    return False


def test_no_synthetic_harbor_city_in_production_app():
    """Assert that active production code in backend/app does not contain toy identifiers."""
    violations = []

    for file_path in BACKEND_APP_DIR.rglob("*.py"):
        if _is_excluded(file_path):
            continue
        
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_IDENTIFIERS:
            if token in content:
                violations.append(f"{file_path.name}: contains forbidden identifier '{token}'")

    assert not violations, f"Anti-slop check failed with violations:\n" + "\n".join(violations)


def test_no_synthetic_harbor_city_in_production_ml():
    """Assert that active production code in backend/ml does not contain toy identifiers."""
    violations = []

    for file_path in BACKEND_ML_DIR.rglob("*.py"):
        if _is_excluded(file_path):
            continue
        
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_IDENTIFIERS:
            if token in content:
                violations.append(f"{file_path.name}: contains forbidden identifier '{token}'")

    assert not violations, f"Anti-slop check failed with violations in ML:\n" + "\n".join(violations)
