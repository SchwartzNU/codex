#!/usr/bin/env python3
"""
Export stratification profiles and arbor statistics for all cells labeled
as amacrine (AC) in the configured Google Sheet.

Outputs:
  - arbor stats CSV: table with segmentID + stats columns, plus a unitless
    column "strat_peak_loc" giving the Z location (x-coordinate) of the peak
    stratification. The first data row contains units. Saved as
    ac_arbor_stats_<timestamp>.csv
  - (optional, for debugging) stratification JSON per cell is still written
    with z, histogram, distribution arrays.

This script uses existing utilities:
  - codex.utils.gsheets.seg_ids_and_soma_pos_matching_gsheet_multi
  - codex.utils.plottingFns.simple_skeleton_from_swc

It reads skeletons from <data_root>/<segid>/ and prefers
"skeleton_warped.swc" with fallback to "skeleton.swc". Arbor stats are read
from "arbor_stats.pkl" in the same segment folder; rows are NaN if missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from datetime import datetime
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from codex.utils.gsheets import seg_ids_and_soma_pos_matching_gsheet_multi
from codex.utils.plottingFns import simple_skeleton_from_swc


DEFAULT_GSHEET_ID = "1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk"
DEFAULT_USER_ID = "gregs_eyewire2"
# Default data root for Flatone output; can be overridden via --data-root
DEFAULT_DATA_ROOT = "/Volumes/SchwartzLab/flatone-output/Flatone-Output-New"


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _find_swc(segid: int, data_root: str) -> str | None:
    base = os.path.join(data_root, str(segid))
    swc_warped = os.path.join(base, "skeleton_warped.swc")
    swc_plain = os.path.join(base, "skeleton.swc")
    if os.path.exists(swc_warped):
        return swc_warped
    if os.path.exists(swc_plain):
        return swc_plain
    return None


def _load_strat_profile(segid: int, data_root: str) -> Dict[str, Any] | None:
    """Return dict with keys: z, histogram, distribution for given segid.

    Values are numpy arrays converted to lists for JSON serialization.
    """
    swc = _find_swc(segid, data_root)
    if not swc:
        return None
    try:
        skel = simple_skeleton_from_swc(swc)
        zp = skel.extra.get("z_profile") or {}
        x = np.asarray(zp.get("x", []), float)
        hist = np.asarray(zp.get("histogram", []), float)
        dist = np.asarray(zp.get("distribution", []), float)
        if x.size < 2 or hist.size != x.size or dist.size != x.size:
            return None
        return {
            "segmentID": int(segid),
            "z": x.tolist(),
            "histogram": hist.tolist(),
            "distribution": dist.tolist(),
        }
    except Exception:
        return None


def _compute_strat_peak(segid: int, data_root: str) -> float:
    """Return z-location of the peak stratification (unitless), or NaN if unavailable."""
    rec = _load_strat_profile(segid, data_root)
    if not rec:
        return float("nan")
    try:
        z = np.asarray(rec.get("z", []), float)
        dist = np.asarray(rec.get("distribution", []), float)
        if z.size == 0 or dist.size != z.size:
            return float("nan")
        i = int(np.argmax(dist))
        return float(z[i]) if np.isfinite(z[i]) else float("nan")
    except Exception:
        return float("nan")


def _load_arbor_stats(segid: int, data_root: str) -> Tuple[Dict[str, Any] | None, Dict[str, str]]:
    """Load per-cell arbor stats row and units mapping.

    Returns (row_dict_or_None, units_map_partial)
    """
    base = os.path.join(data_root, str(segid))
    pkl = os.path.join(base, "arbor_stats.pkl")
    if not os.path.exists(pkl):
        return None, {}
    try:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        stats = d.get("stats", {}) if isinstance(d, dict) else {}
        units = d.get("units", {}) if isinstance(d, dict) else {}
        row: Dict[str, Any] = {}
        for k, v in stats.items():
            # Skip array-valued entries; keep scalars only
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and math.isnan(v):
                v = np.nan
            row[k] = v
        # Normalize units to str or empty
        units_norm: Dict[str, str] = {}
        if isinstance(units, dict):
            for uk, uv in units.items():
                units_norm[uk] = "" if (uv is None) else str(uv)
        return row, units_norm
    except Exception:
        return None, {}


def export_ac_data(
    gsheet_id: str,
    user_id: str,
    outdir: str,
    data_root: str,
) -> Tuple[str, str, str]:
    """Fetch AC segids; write strat JSON, strat-peak CSV, and arbor-stats CSV.
    Returns (strat_json_path, strat_peak_csv_path, arbor_csv_path).
    """
    # Fetch seg IDs for AC rows
    segids_list, _ = seg_ids_and_soma_pos_matching_gsheet_multi(
        gsheet_id=gsheet_id,
        user_id=user_id,
        human_cell_type=None,
        machine_cell_type=None,
        cell_class="AC",
    )
    segids = [int(s) for s in segids_list if str(s).isdigit()]
    segids = sorted(set(segids))

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    print(f"Found {len(segids)} AC cells. Starting export...", flush=True)
    t0 = time.time()

    # One-pass processing with progress output
    strat_records: List[Dict[str, Any]] = []
    strat_peak: Dict[int, float] = {}
    stats_per_cell: Dict[int, Dict[str, Any] | None] = {}
    all_keys: set[str] = set()
    units_agg: Dict[str, str] = {}

    for i, sid in enumerate(segids, start=1):
        t_cell = time.time()
        # Stratification profile + peak
        rec = _load_strat_profile(sid, data_root)
        if rec is not None:
            strat_records.append(rec)
            try:
                z = np.asarray(rec.get("z", []), float)
                dist = np.asarray(rec.get("distribution", []), float)
                strat_peak[sid] = float(z[int(np.argmax(dist))]) if (z.size and z.size == dist.size) else float("nan")
            except Exception:
                strat_peak[sid] = float("nan")
        else:
            strat_peak[sid] = float("nan")

        # Arbor stats
        row, units = _load_arbor_stats(sid, data_root)
        stats_per_cell[sid] = row
        if row:
            all_keys.update(row.keys())
        for k, u in units.items():
            if k not in units_agg:
                units_agg[k] = u

        # Progress print
        dt = time.time() - t_cell
        avg = (time.time() - t0) / i
        remaining = avg * (len(segids) - i)
        print(f"  [{i}/{len(segids)}] segid {sid} processed in {dt:.2f}s | avg {avg:.2f}s | ETA {remaining/60:.1f}m", flush=True)

    ts = _timestamp()

    # Write stratification JSON
    strat_path = os.path.join(outdir, f"ac_stratification_profiles_{ts}.json")
    with open(strat_path, "w", encoding="utf-8") as f:
        json.dump({"records": strat_records}, f)

    # Write stratification peak CSV
    strat_peak_path = os.path.join(outdir, f"ac_stratification_peak_{ts}.csv")
    with open(strat_peak_path, "w", encoding="utf-8") as f:
        f.write("segmentID,strat_peak_loc\n")
        for sid in segids:
            val = strat_peak.get(sid, float("nan"))
            f.write(f"{sid},{'' if not np.isfinite(val) else val}\n")

    # Build Arbor stats CSV (with units row after header), include strat_peak_loc
    all_keys.add("strat_peak_loc")
    units_agg.setdefault("strat_peak_loc", "")
    columns = ["segmentID"] + sorted(all_keys)
    records: List[Dict[str, Any]] = []
    for sid in segids:
        base: Dict[str, Any] = {"segmentID": int(sid)}
        row = stats_per_cell.get(sid)
        if not row:
            for k in all_keys:
                base[k] = np.nan
        else:
            for k in all_keys:
                v = row.get(k, np.nan)
                if isinstance(v, float) and math.isnan(v):
                    v = np.nan
                base[k] = v
        base["strat_peak_loc"] = strat_peak.get(sid, float("nan"))
        records.append(base)
    df = pd.DataFrame.from_records(records, columns=columns)
    units_row = [""] + [units_agg.get(k, "") for k in sorted(all_keys)]
    df_units = pd.DataFrame([units_row], columns=columns)
    df_out = pd.concat([df_units, df], ignore_index=True)
    arbor_path = os.path.join(outdir, f"ac_arbor_stats_{ts}.csv")
    df_out.to_csv(arbor_path, index=False)

    total_dt = time.time() - t0
    print(f"Done. Processed {len(segids)} cells in {total_dt/60:.1f} min.")
    print(f"  Strat JSON: {strat_path}")
    print(f"  Strat peak CSV: {strat_peak_path}")
    print(f"  Arbor stats CSV: {arbor_path}")

    return strat_path, strat_peak_path, arbor_path


def main():
    p = argparse.ArgumentParser(description="Export AC stratification and arbor stats from gSheet")
    p.add_argument("--gsheet-id", default=DEFAULT_GSHEET_ID, help="Google Sheet ID")
    p.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for Sheet service")
    p.add_argument("--outdir", default="./exports", help="Output directory")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root folder containing <segid>/ subfolders with SWC and arbor_stats.pkl")
    args = p.parse_args()

    data_root = (args.data_root or DEFAULT_DATA_ROOT).strip()
    strat_path, strat_peak_path, arbor_path = export_ac_data(args.gsheet_id, args.user_id, args.outdir, data_root)
    # export_ac_data already prints detailed progress and summary
    print(f"Completed. Arbor CSV: {arbor_path}; Peak CSV: {strat_peak_path}; JSON: {strat_path}")


if __name__ == "__main__":
    main()
