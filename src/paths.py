"""Shared folder locations so every script works no matter where it is run from."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "group_sample_seed44.csv"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"
FE_RESULTS = ROOT / "results" / "feature_engineering"
BASELINE_RESULTS = ROOT / "results" / "baseline"

for folder in (PROCESSED, FEATURES, FE_RESULTS, BASELINE_RESULTS):
    folder.mkdir(parents=True, exist_ok=True)
