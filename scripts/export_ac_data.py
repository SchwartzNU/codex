#!/usr/bin/env python3
"""
Export stratification profiles and arbor statistics for all cells labeled
as amacrine (AC) in the configured Google Sheet.

Outputs:
  - stratification JSON: one object per segment with fields
      {"segmentID": int, "z": [...], "histogram": [...], "distribution": [...]}
    saved as ac_stratification_profiles_<timestamp>.json
  - arbor stats CSV: table with segmentID + stats columns
    first data row contains units, saved as ac_arbor_stats_<timestamp>.csv

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
) -> Tuple[str, str]:
    """Fetch AC segids, write stratification JSON and arbor CSV. Returns (strat_path, arbor_path)."""
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

    # 1) Stratification JSON
    strat_records: List[Dict[str, Any]] = []
    for sid in segids:
        rec = _load_strat_profile(sid, data_root)
        if rec is None:
            continue
        strat_records.append(rec)
    ts = _timestamp()
    strat_path = os.path.join(outdir, f"ac_stratification_profiles_{ts}.json")
    with open(strat_path, "w", encoding="utf-8") as f:
        json.dump({"records": strat_records}, f)

    # 2) Arbor stats CSV (with units row after header)
    stats_per_cell: Dict[int, Dict[str, Any] | None] = {}
    all_keys: set[str] = set()
    units_agg: Dict[str, str] = {}
    for sid in segids:
        row, units = _load_arbor_stats(sid, data_root)
        stats_per_cell[sid] = row
        if row:
            all_keys.update(row.keys())
        # collect units mapping (first value wins)
        for k, u in units.items():
            if k not in units_agg:
                units_agg[k] = u

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
        records.append(base)
    df = pd.DataFrame.from_records(records, columns=columns)
    units_row = [""] + [units_agg.get(k, "") for k in sorted(all_keys)]
    df_units = pd.DataFrame([units_row], columns=columns)
    df_out = pd.concat([df_units, df], ignore_index=True)
    arbor_path = os.path.join(outdir, f"ac_arbor_stats_{ts}.csv")
    df_out.to_csv(arbor_path, index=False)

    return strat_path, arbor_path


def main():
    p = argparse.ArgumentParser(description="Export AC stratification and arbor stats from gSheet")
    p.add_argument("--gsheet-id", default=DEFAULT_GSHEET_ID, help="Google Sheet ID")
    p.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for Sheet service")
    p.add_argument("--outdir", default="./exports", help="Output directory")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root folder containing <segid>/ subfolders with SWC and arbor_stats.pkl")
    args = p.parse_args()

    data_root = (args.data_root or DEFAULT_DATA_ROOT).strip()
    strat_path, arbor_path = export_ac_data(args.gsheet_id, args.user_id, args.outdir, data_root)
    print(f"Wrote stratification JSON: {strat_path}")
    print(f"Wrote arbor stats CSV: {arbor_path}")


if __name__ == "__main__":
    main()
