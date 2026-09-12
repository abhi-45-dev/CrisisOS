# Flood ML data

The training pipeline does **not** bundle a toy 118-row dataset.

On the first real training run it downloads the full:

- **India Flood Inventory v3 (1967-2023)** from HydroSense Lab / IIT Delhi / IMD
- Zenodo DOI: **10.5281/zenodo.11275211**
- File: `India_Flood_Inventory_v3.csv`

Source:
https://zenodo.org/records/11275211

## Rainfall feature

By default, rainfall is extracted only from rainfall observations explicitly mentioned in the IMD event descriptions inside the flood inventory. No synthetic rainfall is generated.

For a richer historical precipitation feature, an **event-level precipitation table** can be supplied:

```bash
export FLOOD_RAINFALL_SOURCE=csv
export FLOOD_RAINFALL_PATH=/absolute/path/to/event_rainfall.csv
```

The table must contain `rainfall_24h_mm` and either `UEI` or `Start Date`. This is the integration point for an IPED-derived event table or another legitimate historical precipitation source.

The full IPED gridded dataset is large, so it is intentionally not bundled in the project ZIP.

## Cache validation

The training script validates the cached IFI v3 CSV before every real training run. It checks required event columns, event-row count, distinct years, and the official Zenodo MD5 checksum. If a stale/partial CSV is detected, it is deleted and re-downloaded instead of being used.

To force a refresh manually:

```bash
FLOOD_FORCE_REFRESH=1 python3 -m ml.training.train_flood_model
```
