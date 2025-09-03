# Export Notes

This project provides an “Export Data Table” feature for arbor statistics. You can export all currently listed segment IDs to several formats. Each export includes a data table of per‑cell statistics; units are included either in the same file (when supported) or as a companion file.

## How to Export

- In the Morpho‑Typer UI, use the toolbar above the Neuroglancer view:
  - Choose a format: `.csv`, `.pkl` (pickle), `.mat` (MATLAB), `.h5` (HDF5)
  - Click “Export Data Table”
- Programmatic endpoint (same output):
  - `GET /app/export_arbor_stats?segids=<id>&segids=<id>&format=<fmt>`

Notes:
- Missing/unavailable stats are saved as `NaN`.
- The set of columns is the union of stats observed across all exported cells.

## CSV

- File: `arbor_stats_<timestamp>.csv`
- Row layout:
  - Row 1: header — `segmentID` + one column for each stat key
  - Row 2: units — empty for `segmentID`, unit strings for each stat column
  - Row 3+: data rows (one row per segment)

## Pickle (.pkl)

- File: `arbor_stats_<timestamp>.pkl`
- Contents: a Python dict with two keys serialized via pandas/pickle:
  - `table`: pandas DataFrame of the data table
  - `units`: pandas DataFrame with columns `variable, unit`
- Example (Python):
  ```python
  import pandas as pd
  obj = pd.read_pickle('arbor_stats_<timestamp>.pkl')
  df = obj['table']   # data
  units = obj['units']
  ```

## MATLAB (.mat)

- File: `arbor_stats_<timestamp>.mat`
- Variables stored:
  - `S`: N×1 struct array with fields `segmentID` and one field per stat.
    - Values are numeric; non‑numeric entries are saved as `NaN`.
  - `Units`: struct mapping each field name to its unit string (and `segmentID` => `''`).
- Usage (MATLAB):
  ```matlab
  D = load('arbor_stats_<timestamp>.mat');
  T = struct2table(D.S);  % T is a MATLAB table
  U = D.Units;            % struct with units per field
  ```

## HDF5 (.h5)

- File: `arbor_stats_<timestamp>.h5`
- Layout:
  - `/segid_<ID>/` — a group per segment ID, containing one dataset per stat key
  - `/units/` — a group with one string dataset per stat key (and `segmentID`)
  - `/stat_keys` — list of all stat keys (optional helper)
- Example (Python):
  ```python
  import h5py
  with h5py.File('arbor_stats_<timestamp>.h5', 'r') as f:
      keys = list(f['/units'].keys())
      seg = 'segid_123456789'
      stats = {k: f[f'/{seg}/{k}'][()] for k in keys if k != 'segmentID'}
      units = {k: f[f'/units/{k}'][()].decode('utf-8') for k in keys}
  ```

## Requirements

- CSV: no extra dependencies
- Pickle: pandas
- MATLAB (.mat): SciPy (`scipy.io.savemat`)
- HDF5 (.h5): h5py

These are included in `pyproject.toml` (numpy, pandas, scipy, h5py). If running locally via Poetry:

```
poetry lock
poetry install
poetry run ./scripts/run_local_dev.sh
```

## Notes & Limitations

- Non‑numeric stats in MATLAB export are saved as `NaN` to produce a clean numeric table via `struct2table`.
- HDF5 stores units in a dedicated `/units` group for clarity. If you prefer per‑dataset attributes for units, that can be added.
- CSV cannot embed multiple variables; units are provided as a separate companion file with `_units` suffix.
