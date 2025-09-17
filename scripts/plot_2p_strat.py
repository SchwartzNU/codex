#!/usr/bin/env python3
"""Plot stratification profiles from 2P data for a chosen cell type."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_TWO_PHOTON_MAT = "morph_data/strat_2P.mat"
DEFAULT_TWO_PHOTON_X = "morph_data/strat_x.csv"


def _normalize_type_name(name: Optional[str]) -> str:
    s = (str(name) if name is not None else "").strip()
    if not s or s.lower() == "nan":
        return "unknown"
    low = s.casefold()
    if low in {"on sac", "on_sac", "onsac", "on sac (starburst)", "on starburst"}:
        return "Starburst"
    return s


def _read_x_axis(csv_path: str) -> np.ndarray:
    values: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric values parsed from {csv_path}")
    return np.asarray(values, dtype=float)


def _load_2p_mat(mat_path: str) -> List[Dict[str, Any]]:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy missing in some envs
        raise RuntimeError("SciPy is required to read .mat files (pip install scipy)") from exc

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "s" not in mat:
        raise ValueError(f"Expected variable 's' in {mat_path}; found keys: {list(mat.keys())}")
    raw = mat["s"]

    def _struct_to_dict(obj: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        fields = getattr(obj, "_fieldnames", None)
        if fields:
            for key in fields:
                data[key] = getattr(obj, key)
        elif isinstance(obj, dict):
            data = obj
        return data

    cells: List[Dict[str, Any]] = []
    iterable = raw.ravel().tolist() if isinstance(raw, np.ndarray) else [raw]
    for elem in iterable:
        rec = _struct_to_dict(elem)
        strat = None
        for key in ("strat_norm", "strat", "stratification", "y"):
            val = rec.get(key)
            if val is not None:
                strat = np.asarray(val, dtype=float).ravel()
                break
        if strat is None:
            try:
                strat = np.asarray(getattr(elem, "strat_norm"), dtype=float).ravel()
            except Exception:
                continue
        cell_type = None
        for key in ("cell_type", "ct", "type", "label", "name"):
            val = rec.get(key)
            if val is None:
                try:
                    val = getattr(elem, key)
                except Exception:
                    val = None
            if val is not None:
                try:
                    cell_type = str(np.array(val).tolist())
                except Exception:
                    cell_type = str(val)
                break
        cells.append({"strat": strat, "cell_type": cell_type or "Unknown"})

    if not cells:
        raise ValueError("No stratification records found in 2P .mat file")
    return cells


def _normalize_prob(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = 0.0
    y[y < 0] = 0.0
    total = float(np.sum(y))
    if total > 0:
        return y / total
    return y


def _interpolate_to_axis(x_ref: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    if y_src.size == x_ref.size:
        return y_src
    x_min, x_max = float(np.min(x_ref)), float(np.max(x_ref))
    x_src = np.linspace(x_min, x_max, num=y_src.size)
    order = np.argsort(x_src)
    return np.interp(x_ref, x_src[order], y_src[order], left=0.0, right=0.0)


def plot_stratification(
    mat_path: str,
    x_csv: str,
    cell_type: str,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> str:
    import matplotlib.pyplot as plt  # imported lazily to avoid backend issues

    x_ref = _read_x_axis(x_csv)
    two_p_cells = _load_2p_mat(mat_path)

    target = cell_type.strip()
    if not target:
        raise ValueError("Cell type must be a non-empty string")
    include_all = target.lower() in {"all", "*"}

    traces: List[np.ndarray] = []
    for cell in two_p_cells:
        strat = np.asarray(cell["strat"], dtype=float).ravel()
        strat = _interpolate_to_axis(x_ref, strat)
        strat = _normalize_prob(strat)
        ctype_norm = _normalize_type_name(cell.get("cell_type", "Unknown"))
        if include_all or ctype_norm.casefold() == target.casefold():
            traces.append(strat)

    if not traces:
        available = sorted({
            _normalize_type_name(c.get("cell_type", "Unknown")) for c in two_p_cells
        })
        raise SystemExit(
            f"No 2P cells found for cell type '{cell_type}'. Available: {', '.join(available)}"
        )

    curves = np.vstack(traces)
    mean_curve = np.mean(curves, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    for trace in curves:
        ax.plot(trace, x_ref, color="0.7", linewidth=0.8, alpha=0.7)
    ax.plot(mean_curve, x_ref, color="black", linewidth=2.0, label="Mean")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(12, color="black", linestyle="--", linewidth=1.0)

    label = "All cell types" if include_all else target
    ax.set_title(f"2P stratification: {label} (n={curves.shape[0]})")
    ax.set_xlabel("Normalized stratification value")
    ax.set_ylabel("Location (µm)")
    ax.legend(loc="upper right")
    ax.grid(False)
    fig.tight_layout()

    if output_path:
        out = output_path
        if not out.lower().endswith(('.png', '.pdf', '.svg')):
            out = f"{out}.png"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        return out

    plt.show()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2P stratification profiles for a cell type")
    parser.add_argument("--two-photon-mat", default=DEFAULT_TWO_PHOTON_MAT, help="Path to strat_2P.mat file")
    parser.add_argument("--two-photon-x", default=DEFAULT_TWO_PHOTON_X, help="Path to strat_x.csv file")
    parser.add_argument("--cell-type", required=True, help="Cell type to plot (use 'all' to include every type)")
    parser.add_argument("--output", default="", help="Output image path (PNG/PDF/SVG). If omitted, shows the plot")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    args = parser.parse_args()

    saved_path = plot_stratification(
        mat_path=args.two_photon_mat,
        x_csv=args.two_photon_x,
        cell_type=args.cell_type,
        output_path=args.output or None,
        dpi=args.dpi,
    )
    if saved_path:
        print(f"Saved stratification plot to {saved_path}")


if __name__ == "__main__":
    main()

