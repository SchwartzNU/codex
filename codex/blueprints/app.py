import json
import os
import re
import base64
import math
import pickle
from datetime import datetime

from flask import (
    Blueprint,
    Response,
    redirect,
    request,
    url_for,
    jsonify
)
from user_agents import parse as parse_ua

from codex.blueprints.base import (
    activity_suffix,
    render_error,
    render_info,
    render_template,
    warning_with_redirect,
)
from codex.configuration import (
    MAX_NEURONS_FOR_DOWNLOAD,
    MAX_NODES_FOR_PATHWAY_ANALYSIS,
    MIN_SYN_THRESHOLD,
)
from codex.data.brain_regions import (
    NEUROPIL_DESCRIPTIONS,
    REGIONS,
    REGIONS_JSON,
)
from codex.data.faq_qa_kb import FAQ_QA_KB
from codex.data.neuron_data_factory import NeuronDataFactory
from codex.data.neuron_data_initializer import NETWORK_GROUP_BY_ATTRIBUTES
from codex.data.neurotransmitters import NEURO_TRANSMITTER_NAMES
from codex.data.sorting import SORT_BY_OPTIONS, sort_search_results
from codex.data.structured_search_filters import (
    OP_PATHWAYS,
    get_advanced_search_data,
    parse_search_query,
)
from codex.data.versions import (
    DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
    DEFAULT_DATA_SNAPSHOT_VERSION,
)
from codex.service.cell_details import cached_cell_details
from codex.service.heatmaps import heatmap_data
from codex.service.motif_search import MotifSearchQuery
from codex.service.network import compile_network_html
from codex.service.search import DEFAULT_PAGE_SIZE, pagination_data
from codex.service.stats import leaderboard_cached, stats_cached
from codex.utils import nglui
from codex.utils.nglui import ew2_config_dict, url_for_ew2_segments, shorten_ew2_state, NGL_EW2_BASE_URL
from codex.utils.formatting import (
    can_be_flywire_root_id,
    display,
    highlight_annotations,
    nanometer_to_flywire_coordinates,
    synapse_table_to_csv_string,
    synapse_table_to_json_dict,
)
from codex.utils.graph_algos import distance_matrix

from codex.utils.pathway_vis import pathway_chart_data_rows
from codex.utils.thumbnails import url_for_skeleton
from codex.utils.gsheets import (
    seg_ids_and_soma_pos_matching_gsheet,
    seg_ids_and_soma_pos_matching_gsheet_multi,
    cell_types_by_class_map,
)
from codex import logger
from codex.utils.arbor_stats import load_swc, arborStatsFromSkeleton
from codex.utils.plottingFns import (
    simple_skeleton_from_swc,
    front_side_plotly,
    strat_profile_plotly,
    load_simple_skeleton_cached,
)
from codex.utils.position_stats import compute_vdri, compute_nnri
from codex.cell_mosaics.coverage import CoverageDensityMapper
import numpy as _np

app = Blueprint("app", __name__, url_prefix="/app")


@app.route("/stats")
def stats():
    filter_string = request.args.get("filter_string", "")
    data_version = request.args.get("data_version", "")
    case_sensitive = request.args.get("case_sensitive", 0, type=int)
    whole_word = request.args.get("whole_word", 0, type=int)

    logger.info(f"Generating stats {activity_suffix(filter_string, data_version)}")
    (
        filtered_root_id_list,
        num_items,
        hint,
        data_stats,
        data_charts,
    ) = stats_cached(
        filter_string=filter_string,
        data_version=data_version,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
    )
    if num_items:
        logger.info(
            f"Stats got {num_items} results {activity_suffix(filter_string, data_version)}"
        )
    else:
        logger.warning(
            f"No stats {activity_suffix(filter_string, data_version)}, sending hint '{hint}'"
        )

    return render_template(
        "stats.html",
        data_stats=data_stats,
        data_charts=data_charts,
        num_items=num_items,
        searched_for_root_id=can_be_flywire_root_id(filter_string),
        # If num results is small enough to pass to browser, pass it to allow copying root IDs to clipboard.
        # Otherwise it will be available as downloadable file.
        root_ids_str=(
            ",".join([str(ddi) for ddi in filtered_root_id_list])
            if len(filtered_root_id_list) <= MAX_NEURONS_FOR_DOWNLOAD
            else []
        ),
        filter_string=filter_string,
        hint=hint,
        data_versions=DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
        data_version=data_version,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        advanced_search_data=get_advanced_search_data(current_query=filter_string),
    )


@app.route("/leaderboard")
def leaderboard():
    query = request.args.get("filter_string", "")
    user_filter = request.args.get("user_filter", "")
    lab_filter = request.args.get("lab_filter", "")

    logger.info(f"Loading Leaderboard, {query=} {user_filter=} {lab_filter=}")
    labeled_cells_caption, leaderboard_data = leaderboard_cached(
        query=query,
        user_filter=user_filter,
        lab_filter=lab_filter,
        data_version=DEFAULT_DATA_SNAPSHOT_VERSION,
    )
    return render_template(
        "leaderboard.html",
        labeled_cells_caption=labeled_cells_caption,
        data_stats=leaderboard_data,
        filter_string=query,
        user_filter=user_filter,
        lab_filter=lab_filter,
    )


@app.route("/explore")
def explore():
    logger.info("Loading Explore page")
    data_version = request.args.get("data_version", "")
    top_values = request.args.get("top_values", type=int, default=6)
    for_attr_name = request.args.get("for_attr_name", "")
    return render_template(
        "explore.html",
        data_versions=DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
        data_version=data_version,
        top_values=top_values,
        categories=NeuronDataFactory.instance()
        .get(data_version)
        .categories(
            top_values=top_values,
            for_attr_name=for_attr_name,
        ),
    )

@app.route("/morphotyper")
def morpho_typer():
    f_type_string = request.args.get("f_type_string", "")
    m_type_string = request.args.get("m_type_string", "")
    cell_class = request.args.get("cell_class", "")
    seg_ids_string = request.args.get("seg_ids_string", "")
    stats_enabled = request.args.get("stats_enabled", "0")
    try:
        stats_enabled = bool(int(stats_enabled))
    except Exception:
        stats_enabled = True
    data_version = "EW2" # Morpho-Typer is only available for EW2 data version for now
    logger.info("Loading Morpho-Typer page")

    # Build mapping from class -> cell types (human) for dependent dropdown
    ct_map = cell_types_by_class_map(
        gsheet_id="1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk",
        user_id="gregs_eyewire2",
    )
    return render_morpho_typer_neuron_list(
        data_version=data_version,
        f_type_string=f_type_string,
        m_type_string=m_type_string,
        cell_class=cell_class,
        seg_ids_string=seg_ids_string,
        cell_types_by_class=ct_map,
        stats_enabled=stats_enabled,
    )

def render_morpho_typer_neuron_list(
    data_version,
    f_type_string,
    m_type_string,
    cell_class,
    seg_ids_string,
    cell_types_by_class,
    stats_enabled
):
    f_type_string = (f_type_string or "").strip()
    m_type_string = (m_type_string or "").strip()
    cell_class = (cell_class or "").strip()
    if f_type_string or m_type_string or cell_class:
        # Apply AND logic across provided filters
        seg_ids, soma_pos = seg_ids_and_soma_pos_matching_gsheet_multi(
            gsheet_id="1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk",
            user_id="gregs_eyewire2",
            human_cell_type=f_type_string if f_type_string else None,
            machine_cell_type=m_type_string if m_type_string else None,
            cell_class=cell_class if cell_class else None,
        )
    elif seg_ids_string:
        seg_ids = [int(sid.strip()) for sid in seg_ids_string.split(",") if sid.strip().isdigit()]
        soma_pos = None
    else:
        seg_ids = []
        soma_pos = None
    
    logger.info(f"Loading Morpho-Typer for {len(seg_ids)} segment IDs: {seg_ids}")
    # Legacy image arrays retained for backward compatibility in template; now unused.
    skeleton_imgs = []
    strat_imgs = []
    # Arbor stats are now loaded on demand (row click) or via export only.
    # Do not preload per-cell stats during search to keep it fast.
    arbor_stats = []
    arbor_stats_units = []
    DATA_ROOT_PATH = "static/data"
    for seg_id in seg_ids:
        # Do not preload heavy Plotly figs; will be fetched per-row via endpoints.
        skeleton_imgs.append(None)
        strat_imgs.append(None)
        # Do not preload arbor stats here. They are fetched per row on click
        # via /app/arbor_stats/<segid> and loaded in bulk only for export.

    # ensure soma_pos is a JSON-serializable list
    soma_pos_list = soma_pos.tolist() if hasattr(soma_pos, 'tolist') else soma_pos

    # Build a comma-separated seg_ids string to populate the input after search
    if seg_ids:
        seg_ids_string_out = ",".join(str(s) for s in seg_ids)
    else:
        seg_ids_string_out = seg_ids_string or ""

    return render_template(
        "morpho_typer.html",
        skeleton_imgs=skeleton_imgs,
        strat_imgs=strat_imgs,
        arbor_stats=arbor_stats,
        units=arbor_stats_units,
        seg_ids=seg_ids,
        soma_pos=soma_pos_list,
        data_version=data_version,
        f_type_string=f_type_string,
        m_type_string=m_type_string,
        cell_class=cell_class,
        seg_ids_string=seg_ids_string_out,
        cell_types_by_class=cell_types_by_class,
        stats_enabled=stats_enabled,
    )


@app.route("/skeleton_plot")
def skeleton_plot():
    """Return Plotly HTML for front/side projections of a given segment ID."""
    segid = request.args.get("segid")
    if not segid:
        return jsonify({"error": "Missing segid"}), 400
    base_dir = os.path.join("static", "data", "EW2", "skeletons", str(segid))
    swc_path = os.path.join(base_dir, "skeleton_warped.swc")
    if not os.path.exists(swc_path):
        # Fallback to non-warped
        swc_path = os.path.join(base_dir, "skeleton.swc")
        if not os.path.exists(swc_path):
            return jsonify({"error": "SWC not found"}), 404
    def _parse_range(name):
        val = request.args.get(name)
        if not val:
            return None
        try:
            a, b = map(float, val.split(","))
            return (a, b)
        except Exception:
            return None

    try:
        mtime = os.path.getmtime(swc_path)
        skel = load_simple_skeleton_cached(swc_path, mtime)
        fig = front_side_plotly(
            skel,
            xlim_xy=_parse_range("xlim_xy"),
            ylim_xy=_parse_range("ylim_xy"),
            xlim_xz=_parse_range("xlim_xz"),
            ylim_xz=_parse_range("ylim_xz"),
        )
        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={"displayModeBar": False, "staticPlot": True},
            default_width="100%",
        )
        return Response(html, mimetype="text/html")
    except Exception as e:
        logger.warning(f"Failed to build skeleton plot for {segid}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/strat_plot")
def strat_plot():
    """Return Plotly HTML for stratification profile + XZ arbor for a given segid."""
    segid = request.args.get("segid")
    if not segid:
        return jsonify({"error": "Missing segid"}), 400
    base_dir = os.path.join("static", "data", "EW2", "skeletons", str(segid))
    swc_path = os.path.join(base_dir, "skeleton_warped.swc")
    if not os.path.exists(swc_path):
        swc_path = os.path.join(base_dir, "skeleton.swc")
        if not os.path.exists(swc_path):
            return jsonify({"error": "SWC not found"}), 404
    def _parse_range(name):
        val = request.args.get(name)
        if not val:
            return None
        try:
            a, b = map(float, val.split(","))
            return (a, b)
        except Exception:
            return None

    try:
        mtime = os.path.getmtime(swc_path)
        skel = load_simple_skeleton_cached(swc_path, mtime)
        # Stratification plot uses a fixed Z range across rows to cover both ON/OFF ChAT
        z_profile_extent = (-20.0, 30.0)
        fig = strat_profile_plotly(
            skel,
            z_profile_extent=z_profile_extent,
            seg_id=None,
            fig_height_px=220,
            max_width_px=560,
        )
        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={"displayModeBar": False, "staticPlot": True},
            default_width="100%",
        )
        return Response(html, mimetype="text/html")
    except Exception as e:
        logger.warning(f"Failed to build strat plot for {segid}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/plot_ranges")
def plot_ranges():
    """Compute global plot ranges across given segids for uniform scaling.

    Returns JSON with keys:
      {
        "xy": {"x": [min, max], "y": [min, max]},
        "xz": {"x": [min, max], "z": [min, max]}
      }
    """
    segids = request.args.getlist("segids")
    if not segids:
        return jsonify({"error": "No segids provided"}), 400
    base = os.path.join("static", "data", "EW2", "skeletons")
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    count = 0
    for sid in segids:
        d = os.path.join(base, str(sid))
        swc = os.path.join(d, "skeleton_warped.swc")
        if not os.path.exists(swc):
            swc = os.path.join(d, "skeleton.swc")
            if not os.path.exists(swc):
                continue
        try:
            coords, _r, _e = load_swc(swc)
            if coords.size == 0:
                continue
            xmin = min(xmin, float(_np.min(coords[:, 0])))
            xmax = max(xmax, float(_np.max(coords[:, 0])))
            ymin = min(ymin, float(_np.min(coords[:, 1])))
            ymax = max(ymax, float(_np.max(coords[:, 1])))
            zmin = min(zmin, float(_np.min(coords[:, 2])))
            zmax = max(zmax, float(_np.max(coords[:, 2])))
            count += 1
        except Exception as e:
            logger.warning(f"plot_ranges: failed reading {sid}: {e}")
            continue
    if count == 0 or not _np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
        return jsonify({"error": "No valid skeletons"}), 404
    def pad(lo, hi, frac=0.02):
        span = hi - lo
        if span <= 0:
            return lo - 1.0, hi + 1.0
        p = span * frac
        return lo - p, hi + p
    xlo, xhi = pad(xmin, xmax)
    ylo, yhi = pad(ymin, ymax)
    zlo, zhi = pad(zmin, zmax)
    return jsonify({
        "xy": {"x": [xlo, xhi], "y": [ylo, yhi]},
        "xz": {"x": [xlo, xhi], "z": [zlo, zhi]},
    })


@app.route("/coverage_plot")
def coverage_plot():
    """Generate a cell coverage mosaic PNG for the given segids.

    Query params: segids repeated or comma-separated list.
    Uses boundary_points from arbor_stats.pkl when available; otherwise uses
    convex hull of SWC nodes projected to XY.
    """
    import io
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as _plt
    from scipy.spatial import ConvexHull as _ConvexHull

    segids_q = request.args.getlist("segids")
    # also accept a comma-separated list in a single segids
    if len(segids_q) == 1 and "," in segids_q[0]:
        segids_q = [s.strip() for s in segids_q[0].split(",") if s.strip()]
    try:
        segids = [int(s) for s in segids_q if str(s).isdigit()]
    except Exception:
        segids = []
    if not segids:
        return jsonify({"error": "No segids provided"}), 400

    DATA_ROOT_PATH = "static/data"
    polys = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for sid in segids:
        base = os.path.join(DATA_ROOT_PATH, "EW2", "skeletons", str(sid))
        stats_path = os.path.join(base, "arbor_stats.pkl")
        poly = None
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "rb") as f:
                    d = pickle.load(f)
                    bp = d.get("stats", {}).get("boundary_points")
                    if bp is not None:
                        arr = _np.asarray(bp, float)
                        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                            poly = arr
            except Exception as e:
                logger.warning(f"coverage_plot: failed to load boundary_points for {sid}: {e}")
        if poly is None:
            # Fallback: convex hull of SWC XY
            swc = os.path.join(base, "skeleton_warped.swc")
            if not os.path.exists(swc):
                swc = os.path.join(base, "skeleton.swc")
            if os.path.exists(swc):
                try:
                    nodes, _r, _e = load_swc(swc)
                    if nodes.size >= 3:
                        pts = _np.asarray(nodes[:, :2], float)
                        hull = _ConvexHull(pts)
                        poly = pts[hull.vertices]
                except Exception as e:
                    logger.warning(f"coverage_plot: SWC fallback failed for {sid}: {e}")
        if poly is not None:
            polys.append(poly)
            xmin = min(xmin, float(poly[:, 0].min()))
            xmax = max(xmax, float(poly[:, 0].max()))
            ymin = min(ymin, float(poly[:, 1].min()))
            ymax = max(ymax, float(poly[:, 1].max()))

    if not polys:
        return jsonify({"error": "No outlines found"}), 404

    # Pad field by 5% of span
    span_x = xmax - xmin
    span_y = ymax - ymin
    pad_x = span_x * 0.05
    pad_y = span_y * 0.05
    field = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    mapper = CoverageDensityMapper(field_bounds=field, resolution=400)
    mapper.add_multiple_polygons(polys)

    # Render
    fig, ax, im = mapper.plot_coverage(plot_cell_outlines=True, colormap="viridis", edgecolor="#333333")
    buf = io.BytesIO()
    fig.tight_layout(pad=0.5)
    fig.savefig(buf, format="png", dpi=110)
    _plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")



# AJAX endpoint to return stats + units for a given segment ID
@app.route("/arbor_stats", defaults={"segid": None})
@app.route("/arbor_stats/<segid>")
def arbor_stats(segid):
    """
    Return JSON {stats: {...}, units: {...}} for the given segid.
    Accepts either ?segid=<id> or path /app/arbor_stats/<id>.
    """
    # allow passing via query-string if not in path
    if segid is None:
        segid = request.args.get("segid")
    if not segid:
        return jsonify({"error": "Missing segid"}), 400

    arbor_stats_path = os.path.join(
        "static", "data", "EW2", "skeletons", str(segid), "arbor_stats.pkl"
    )
    if not os.path.exists(arbor_stats_path):
        return jsonify({"error": "arbor stats file not found"}), 404

    logger.info(f"Reading arbor stats for segment {segid} from {arbor_stats_path}")
    with open(arbor_stats_path, "rb") as f:
        try:
            stats_data = pickle.load(f)
            stats = stats_data.get("stats", {})
            units = stats_data.get("units", {})
        except Exception as e:
            logger.warning(f"Failed to load arbor stats for {segid}: {e}")
            stats = {}
            units = {}

    # Remove any array-valued stats; convert numpy scalars to Python types
    import numpy as _np
    serializable_stats = {}
    for key, val in stats.items():
        if isinstance(val, _np.ndarray):
            # skip array entries entirely
            continue
        if isinstance(val, _np.generic):
            serializable_stats[key] = val.item()
        else:
            serializable_stats[key] = val
    stats = serializable_stats

    # Replace NaN floats with None for valid JSON
    for k, v in stats.items():
        if isinstance(v, float) and math.isnan(v):
            stats[k] = None

    return jsonify({"stats": stats, "units": units})


# AJAX endpoint to compute population-level regularity indices
@app.route("/population_stats", methods=["POST"])
def population_stats():
    """
    Expects JSON body with {"soma_pos": [[x1, y1], [x2, y2], ...]}.
    Returns JSON {"vdri": <float or null>, "nnri": <float or null>}.
    """
    data = request.get_json(force=True)
    soma_pos = data.get("soma_pos")
    import numpy as _np
    import math
    coords = None
    if soma_pos and isinstance(soma_pos, list) and len(soma_pos) > 0:
        # Build numpy array of shape (N,2)
        coords = _np.array(soma_pos, dtype=float)
    else:
        # Fallback: accept segids and estimate soma x,y from SWC (node 0)
        segids = data.get("segids") or []
        try:
            segids = [int(s) for s in segids]
        except Exception:
            segids = []
        if segids:
            est = []
            base_dir = os.path.join("static", "data", "EW2", "skeletons")
            for sid in segids:
                d = os.path.join(base_dir, str(sid))
                swc_path = os.path.join(d, "skeleton_warped.swc")
                if not os.path.exists(swc_path):
                    swc_path = os.path.join(d, "skeleton.swc")
                    if not os.path.exists(swc_path):
                        continue
                try:
                    nodes, _r, _e = load_swc(swc_path)
                    if nodes.size:
                        est.append([float(nodes[0, 0]), float(nodes[0, 1])])
                except Exception as e:
                    logger.warning(f"population_stats: failed to read {sid}: {e}")
            if est:
                coords = _np.array(est, dtype=float)
    if coords is None:
        return jsonify({"error": "Missing positions and segids"}), 400

    # if fewer than 3 cells, both regularity indices are undefined
    #Note: GWS: We will need to create a central bounding box for the VDRI and NNRI calculations to make sense
    #We will also need to comoute the null distributions for these indices
    #Using the convec hulls from arbor_stats.py, we can compute the dendritic overlap regularity index as 
    #in Bae et al. That is a better measure of regularity than the VDRI and NNRI
    if coords.shape[0] < 3:
        vdri = None
        nnri = None
    else:
        vdri = compute_vdri(coords)
        nnri = compute_nnri(coords)

    # Replace NaN with None for JSON
    if isinstance(vdri, float) and math.isnan(vdri):
        vdri = None
    if isinstance(nnri, float) and math.isnan(nnri):
        nnri = None

    return jsonify({"vdri": vdri, "nnri": nnri})


@app.route("/export_arbor_stats")
def export_arbor_stats():
    """
    Export arbor stats for a list of segment IDs as a data table.
    Query params:
      - segids: repeated query param for each segment id
      - format: one of csv, pickle, mat, h5
    Returns a downloadable file with columns: segmentID + per-cell stats keys.
    Missing/unavailable stats are filled with NaN.
    """
    import io
    import numpy as _np
    import pandas as _pd

    segids = request.args.getlist("segids")
    fmt = request.args.get("format", "csv").lower()
    try:
        segids = [int(s) for s in segids if str(s).isdigit()]
    except Exception:
        return jsonify({"error": "Invalid segids"}), 400
    if not segids:
        return jsonify({"error": "No segids provided"}), 400

    DATA_ROOT_PATH = "static/data"
    stats_per_cell = {}
    all_keys = set()
    units_map = {}

    for sid in segids:
        arbor_stats_path = os.path.join(DATA_ROOT_PATH, "EW2", "skeletons", str(sid), "arbor_stats.pkl")
        if os.path.exists(arbor_stats_path):
            logger.info(f"Reading arbor stats for segment {sid} from {arbor_stats_path}")
            with open(arbor_stats_path, "rb") as f:
                try:
                    stats_data = pickle.load(f)
                    stats = stats_data.get("stats", {})
                    units = stats_data.get("units", {})
                    # filter out array-valued stats and convert numpy scalars
                    row = {}
                    for k, v in stats.items():
                        if isinstance(v, _np.ndarray):
                            continue
                        if isinstance(v, _np.generic):
                            v = v.item()
                        row[k] = v
                    stats_per_cell[sid] = row
                    all_keys.update(row.keys())
                    if isinstance(units, dict):
                        for uk, uv in units.items():
                            if uk not in units_map:
                                units_map[uk] = "" if (uv is None) else str(uv)
                except Exception as e:
                    logger.warning(f"Failed to load arbor stats for {sid}: {e}")
                    stats_per_cell[sid] = None
        else:
            stats_per_cell[sid] = None

    # Build DataFrame with NaN for missing
    columns = ["segmentID"] + sorted(all_keys)
    records = []
    for sid in segids:
        base = {"segmentID": sid}
        if stats_per_cell[sid] is None:
            for k in all_keys:
                base[k] = _np.nan
        else:
            for k in all_keys:
                val = stats_per_cell[sid].get(k, _np.nan)
                # Normalize NaN for invalid floats
                if isinstance(val, float) and math.isnan(val):
                    val = _np.nan
                base[k] = val
        records.append(base)
    df = _pd.DataFrame.from_records(records, columns=columns)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"arbor_stats_{ts}"

    if fmt == "csv":
        # Single CSV: header row with column names, second row with units, then data rows
        buf = io.StringIO()
        units_row = [""] + [units_map.get(k, "") for k in sorted(all_keys)]
        df_units_row = _pd.DataFrame([units_row], columns=columns)
        df_out = _pd.concat([df_units_row, df], ignore_index=True)
        df_out.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
        mime = "text/csv"
        filename = f"{base_name}.csv"
    elif fmt in ("pickle", "pkl", "p"):  # pickle DataFrame
        payload = {"table": df}
        # attach units mapping as a DataFrame of variable/unit pairs
        units_entries = [("segmentID", "")] + [(k, units_map.get(k, "")) for k in sorted(all_keys)]
        payload["units"] = _pd.DataFrame(units_entries, columns=["variable", "unit"])
        buf = io.BytesIO()
        _pd.to_pickle(payload, buf)
        data = buf.getvalue()
        mime = "application/octet-stream"
        filename = f"{base_name}.pkl"
    elif fmt in ("mat", ".mat"):
        try:
            from scipy.io import savemat as _savemat
        except Exception:
            return jsonify({"error": "scipy is required for .mat export"}), 400
        # Save as an array of structs (Nx1) so MATLAB can do: T = struct2table(S);
        # Use numeric dtypes: int64 for segmentID, float64 for stats (NaN for missing)
        stat_keys_sorted = sorted(all_keys)
        dt = _np.dtype([('segmentID', 'i8')] + [(k, 'f8') for k in stat_keys_sorted])
        struct_arr = _np.zeros((len(segids), 1), dtype=dt)
        for i, sid in enumerate(segids):
            struct_arr[i, 0]['segmentID'] = int(sid)
            row = stats_per_cell.get(sid)
            for k in stat_keys_sorted:
                v = _np.nan
                if row is not None:
                    v = row.get(k, _np.nan)
                # normalize to numeric where possible, otherwise NaN
                try:
                    if isinstance(v, bool):
                        vnum = float(v)
                    elif v is None:
                        vnum = _np.nan
                    else:
                        # unwrap numpy scalar
                        if isinstance(v, _np.generic):
                            v = v.item()
                        vnum = float(v)
                except Exception:
                    vnum = _np.nan
                struct_arr[i, 0][k] = vnum
        # Units as a struct mapping field -> unit string
        units_struct = {}
        for k in sorted(all_keys):
            uval = units_map.get(k, "")
            if uval is None:
                uval = ""
            units_struct[k] = _np.array(str(uval), dtype=object)
        units_struct["segmentID"] = _np.array("", dtype=object)
        buf = io.BytesIO()
        _savemat(buf, {"S": struct_arr, "Units": units_struct}, do_compression=True)
        data = buf.getvalue()
        mime = "application/octet-stream"
        filename = f"{base_name}.mat"
    elif fmt in ("h5", "hdf5", "hdf"):
        # Write HDF5 with a group per segment id: /segid_{id}/{stat_key}
        try:
            import h5py as _h5
        except Exception as e:
            logger.warning(f"h5py import failed: {e}")
            return jsonify({"error": f"h5py is required for .h5 export (import failed: {e})"}), 400
        import tempfile as _tempfile
        tmp = _tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            with _h5.File(tmp_path, "w") as h5f:
                # store a sorted list of keys at root for reference
                keys_ds = _h5.special_dtype(vlen=str)
                try:
                    h5f.create_dataset("stat_keys", data=_np.array(sorted(all_keys), dtype=keys_ds))
                except Exception:
                    pass
                # write units group
                try:
                    ugrp = h5f.create_group("units")
                    sdt = _h5.string_dtype(encoding='utf-8')
                    for k in sorted(all_keys):
                        ugrp.create_dataset(k, data=_np.array(units_map.get(k, ""), dtype=sdt))
                    ugrp.create_dataset("segmentID", data=_np.array("", dtype=sdt))
                except Exception:
                    pass
                for sid in segids:
                    grp = h5f.create_group(f"segid_{sid}")
                    row = stats_per_cell.get(sid)
                    for k in sorted(all_keys):
                        # default NaN
                        v = _np.nan
                        if row is not None:
                            v = row.get(k, _np.nan)
                        # normalize types for HDF5
                        ds = None
                        try:
                            # booleans -> int
                            if isinstance(v, (bool,)):
                                v = int(v)
                            # numpy scalar -> python
                            if isinstance(v, _np.generic):
                                v = v.item()
                            # try float
                            valf = float(v) if v is not None else _np.nan
                            if _np.isnan(valf):
                                ds = _np.array(_np.nan, dtype=_np.float64)
                            else:
                                ds = _np.array(valf, dtype=_np.float64)
                            grp.create_dataset(k, data=ds)
                        except Exception:
                            # fallback to UTF-8 string
                            try:
                                sdt = _h5.string_dtype(encoding='utf-8')
                                grp.create_dataset(k, data=_np.array(str(v), dtype=sdt))
                            except Exception:
                                # last resort: skip this field
                                continue
            with open(tmp_path, "rb") as f:
                data = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        mime = "application/octet-stream"
        filename = f"{base_name}.h5"
    else:
        return jsonify({"error": "Unsupported format"}), 400

    return Response(
        data,
        headers={
            "Content-Type": mime,
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
        },
    )

def render_neuron_list(
    data_version,
    template_name,
    sorted_search_result_root_ids,
    filter_string,
    case_sensitive,
    whole_word,
    page_number,
    page_size,
    sort_by,
    hint,
    extra_data,
):
    neuron_db = NeuronDataFactory.instance().get(data_version)
    pagination_info, page_ids, page_size, page_size_options = pagination_data(
        items_list=sorted_search_result_root_ids,
        page_number=page_number,
        page_size=page_size,
    )

    display_data = [neuron_db.get_neuron_data(i) for i in page_ids]
    skeleton_thumbnail_urls = {
        nd["root_id"]: (
            url_for_skeleton(nd["root_id"], file_type="png"),
            url_for_skeleton(nd["root_id"], file_type="gif"),
        )
        for nd in display_data
    }
    highlighted_terms = {}
    links = {}
    for nd in display_data:
        # Only highlight from free-form search tokens (and not structured search attributes)
        psq = parse_search_query(filter_string)
        search_terms = psq[1] + [stq["rhs"] for stq in psq[2] or []]
        # highlight all displayed annotations
        terms_to_annotate = set()
        for attr_name in [
            "root_id",
            "label",
            "side",
            "flow",
            "super_class",
            "class",
            "sub_class",
            "cell_type",
            "hemilineage",
            "nt_type",
            "nerve",
            "connectivity_tag",
        ]:
            if nd[attr_name]:
                if isinstance(nd[attr_name], list):
                    terms_to_annotate |= set(nd[attr_name])
                else:
                    terms_to_annotate.add(nd[attr_name])
        highlighted_terms.update(highlight_annotations(search_terms, terms_to_annotate))
        links[nd["root_id"]] = neuron_db.get_links(nd["root_id"])

    return render_template(
        template_name_or_list=template_name,
        display_data=display_data,
        highlighted_terms=highlighted_terms,
        links=links,
        skeleton_thumbnail_urls=skeleton_thumbnail_urls,
        # If num results is small enough to pass to browser, pass it to allow copying root IDs to clipboard.
        # Otherwise it will be available as downloadable file.
        root_ids_str=(
            ",".join([str(ddi) for ddi in sorted_search_result_root_ids])
            if len(sorted_search_result_root_ids) <= MAX_NEURONS_FOR_DOWNLOAD
            else []
        ),
        num_items=len(sorted_search_result_root_ids),
        searched_for_root_id=can_be_flywire_root_id(filter_string),
        pagination_info=pagination_info,
        page_size=page_size,
        page_size_options=page_size_options,
        filter_string=filter_string,
        hint=hint,
        data_versions=DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
        data_version=data_version,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        extra_data=extra_data,
        sort_by=sort_by,
        sort_by_options=SORT_BY_OPTIONS,
        advanced_search_data=get_advanced_search_data(current_query=filter_string),
        multi_val_attrs=neuron_db.multi_val_attrs(sorted_search_result_root_ids),
        non_uniform_labels=neuron_db.non_uniform_values(
            list_attr_key="label",
            page_ids=page_ids,
            all_ids=sorted_search_result_root_ids,
        ),
        non_uniform_cell_types=neuron_db.non_uniform_values(
            list_attr_key="cell_type",
            page_ids=page_ids,
            all_ids=sorted_search_result_root_ids,
        ),
    )


def _search_and_sort():
    filter_string = request.args.get("filter_string", "")
    data_version = request.args.get("data_version", "")
    case_sensitive = request.args.get("case_sensitive", 0, type=int)
    whole_word = request.args.get("whole_word", 0, type=int)
    sort_by = request.args.get("sort_by")
    neuron_db = NeuronDataFactory.instance().get(data_version)
    filtered_root_id_list = neuron_db.search(
        filter_string, case_sensitive=case_sensitive, word_match=whole_word
    )
    return sort_search_results(
        query=filter_string,
        ids=filtered_root_id_list,
        output_sets=neuron_db.output_sets(),
        label_count_getter=lambda x: len(neuron_db.get_neuron_data(x)["label"]),
        nt_type_getter=lambda x: neuron_db.get_neuron_data(x)["nt_type"],
        synapse_neuropil_count_getter=lambda x: len(
            neuron_db.get_neuron_data(x)["input_neuropils"]
        )
        + len(neuron_db.get_neuron_data(x)["output_neuropils"]),
        size_getter=lambda x: neuron_db.get_neuron_data(x)["size_nm"],
        partner_count_getter=lambda x: len(neuron_db.output_sets()[x])
        + len(neuron_db.input_sets()[x]),
        similar_shape_cells_getter=neuron_db.get_similar_shape_cells,
        similar_connectivity_cells_getter=neuron_db.get_similar_connectivity_cells,
        connections_getter=lambda x: neuron_db.cell_connections(x),
        sort_by=sort_by,
    )


@app.route("/search", methods=["GET"])
def search():
    filter_string = request.args.get("filter_string", "")
    page_number = int(request.args.get("page_number", 1))
    page_size = int(request.args.get("page_size", DEFAULT_PAGE_SIZE))
    data_version = request.args.get("data_version", "")
    case_sensitive = request.args.get("case_sensitive", 0, type=int)
    whole_word = request.args.get("whole_word", 0, type=int)
    sort_by = request.args.get("sort_by")

    hint = None
    logger.info(
        f"Loading search page {page_number} {activity_suffix(filter_string, data_version)}"
    )
    sorted_search_result_root_ids, extra_data = _search_and_sort()
    if sorted_search_result_root_ids:
        if len(sorted_search_result_root_ids) == 1:
            if filter_string == str(sorted_search_result_root_ids[0]):
                logger.info("Single cell match, redirecting to cell details")
                return redirect(
                    url_for(
                        "app.cell_details",
                        root_id=sorted_search_result_root_ids[0],
                        data_version=data_version,
                    )
                )

        logger.info(
            f"Loaded {len(sorted_search_result_root_ids)} search results for page {page_number} "
            f"{activity_suffix(filter_string, data_version)}"
        )
    else:
        neuron_db = NeuronDataFactory.instance().get(data_version)
        hint, edist = neuron_db.closest_token(
            filter_string, case_sensitive=case_sensitive
        )
        logger.warning(
            f"No results for '{filter_string}', sending hint '{hint}' {edist=}"
        )

    return render_neuron_list(
        data_version=data_version,
        template_name="search.html",
        sorted_search_result_root_ids=sorted_search_result_root_ids,
        filter_string=filter_string,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        page_number=page_number,
        page_size=page_size,
        sort_by=sort_by,
        hint=hint,
        extra_data=extra_data,
    )


@app.route("/download_search_results")
def download_search_results():
    filter_string = request.args.get("filter_string", "")
    data_version = request.args.get("data_version", "")
    neuron_db = NeuronDataFactory.instance().get(data_version)

    logger.info(
        f"Downloading search results {activity_suffix(filter_string, data_version)}"
    )
    sorted_search_result_root_ids, extra_data = _search_and_sort()
    logger.info(
        f"For download got {len(sorted_search_result_root_ids)} results {activity_suffix(filter_string, data_version)}"
    )

    cols = [
        "root_id",
        "label",
        "name",
        "nt_type",
        "flow",
        "super_class",
        "class",
        "sub_class",
        "cell_type",
        "hemilineage",
        "nerve",
        "connectivity_tag",
        "side",
        "input_synapses",
        "output_synapses",
    ]
    data = [cols]
    for i in sorted_search_result_root_ids:
        data.append(
            [str(neuron_db.get_neuron_data(i)[c]).replace(",", ";") for c in cols]
        )

    fname = f"search_results_{re.sub('[^0-9a-zA-Z]+', '_', filter_string)}.csv"
    return Response(
        "\n".join([",".join(r) for r in data]),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={fname}"},
    )


@app.route("/root_ids_from_search_results")
def root_ids_from_search_results():
    filter_string = request.args.get("filter_string", "")
    data_version = request.args.get("data_version", "")

    logger.info(f"Listing Cell IDs {activity_suffix(filter_string, data_version)}")
    sorted_search_result_root_ids, extra_data = _search_and_sort()
    logger.info(
        f"For list cell ids got {len(sorted_search_result_root_ids)} results {activity_suffix(filter_string, data_version)}"
    )
    fname = f"root_ids_{re.sub('[^0-9a-zA-Z]+', '_', filter_string)}.txt"
    return Response(
        ",".join([str(rid) for rid in sorted_search_result_root_ids]),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={fname}"},
    )


@app.route("/search_results_flywire_url")
def search_results_flywire_url():
    filter_string = request.args.get("filter_string", "")
    data_version = request.args.get("data_version", "")
    request.args.get("case_sensitive", 0, type=int)
    request.args.get("whole_word", 0, type=int)
    NeuronDataFactory.instance().get(data_version)

    logger.info(
        f"Generating URL search results {activity_suffix(filter_string, data_version)}"
    )
    sorted_search_result_root_ids, extra_data = _search_and_sort()
    logger.info(
        f"For URLs got {len(sorted_search_result_root_ids)} results {activity_suffix(filter_string, data_version)}"
    )

    url = nglui.url_for_random_sample(
        sorted_search_result_root_ids,
        version=data_version or DEFAULT_DATA_SNAPSHOT_VERSION,
        sample_size=MAX_NEURONS_FOR_DOWNLOAD,
    )
    logger.info(
        f"Redirecting {len(sorted_search_result_root_ids)} results {activity_suffix(filter_string, data_version)} to FlyWire"
    )
    return ngl_redirect_with_client_check(ngl_url=url)


@app.route("/flywire_url")
def flywire_url():
    root_ids = [int(rid) for rid in request.args.getlist("root_ids")]
    data_version = request.args.get("data_version", "")
    log_request = request.args.get("log_request", default=1, type=int)
    point_to = request.args.get("point_to")
    show_side_panel = request.args.get("show_side_panel", type=int, default=None)

    url = nglui.url_for_root_ids(
        root_ids,
        version=data_version or DEFAULT_DATA_SNAPSHOT_VERSION,
        point_to=point_to,
        show_side_panel=show_side_panel,
    )
    if log_request:
        logger.info(f"Redirecting for {len(root_ids)} root ids to FlyWire, {point_to=}")
    return ngl_redirect_with_client_check(ngl_url=url)


def ngl_redirect_with_client_check(ngl_url):
    ua = parse_ua(str(request.user_agent))
    # iOS (iPhone/iPad) does not render brain meshes or neuropils
    if "iOS" in ua.get_os():
        return warning_with_redirect(
            title="Device not supported",
            message="Neuroglancer (3D neuron rendering) may not be supported on your mobile iOS device. "
            "Compatible platforms: Mac / PC / Android",
            redirect_url=ngl_url,
            redirect_button_text="Proceed anyway",
        )

    min_supported = {
        "Chrome": 51,
        "Edge": 51,
        "Firefox": 46,
        "Safari": 15,
        "Opera": 95,
    }
    # browser_family can contain other parts, e.g. "Mobile Safari", or "Chrome Mobile". Use substring match.
    browser = None
    bfl = ua.browser.family.lower()
    for k in min_supported.keys():
        if k.lower() in bfl:
            browser = k
            break
    if browser in min_supported and ua.browser.version[0] >= min_supported[browser]:
        return redirect(ngl_url, code=302)
    else:
        supported = ", ".join(
            [f"{browser} >= {version}" for browser, version in min_supported.items()]
        )
        return warning_with_redirect(
            title="Browser not supported",
            message=f"Neuroglancer (3D neuron rendering) may not be supported on your browser {ua.get_browser()}. Try: {supported}",
            redirect_url=ngl_url,
            redirect_button_text="Proceed anyway",
        )


@app.route("/cell_coordinates/<path:cell_id>")
def cell_coordinates(cell_id):
    data_version = request.args.get("data_version", "")
    logger.info(f"Loading coordinates for cell {cell_id}, {data_version=}")
    neuron_db = NeuronDataFactory.instance().get(data_version)
    nd = neuron_db.get_neuron_data(cell_id)
    return f"<h2>Supervoxel IDs and coordinates for {cell_id}</h2>" + "<br>".join(
        [
            f"Supervoxel ID: {s}, nanometer coordinates: {c}, FlyWire coordinates: {nanometer_to_flywire_coordinates(c)}"
            for c, s in zip(nd["position"], nd["supervoxel_id"])
        ]
    )


@app.route("/cell_details", methods=["GET", "POST"])
def cell_details():
    data_version = request.args.get("data_version", "")
    reachability_stats = request.args.get("reachability_stats", 0, type=int)
    neuron_db = NeuronDataFactory.instance().get(data_version)

    if request.method == "POST":
        data_version = request.args.get("data_version", "")
        neuron_db = NeuronDataFactory.instance().get(data_version)
        annotation_text = request.form.get("annotation_text")
        annotation_coordinates = request.form.get("annotation_coordinates")
        annotation_cell_id = request.form.get("annotation_cell_id")
        if not annotation_coordinates:
            ndata = neuron_db.get_neuron_data(annotation_cell_id)
            annotation_coordinates = ndata["position"][0] if ndata["position"] else None
        return redirect(
            url_for(
                "app.annotate_cell",
                annotation_cell_id=annotation_cell_id,
                annotation_text=annotation_text,
                annotation_coordinates=annotation_coordinates,
            )
        )

    root_id = None
    cell_names_or_id = request.args.get("cell_names_or_id")
    if not cell_names_or_id:
        cell_names_or_id = request.args.get("root_id")
    if cell_names_or_id:
        if cell_names_or_id == "{random_cell}":
            logger.info("Generated random cell detail page")
            root_id = neuron_db.random_cell_id()
            cell_names_or_id = f"name == {neuron_db.get_neuron_data(root_id)['name']}"
        else:
            logger.info(f"Generating cell detail page from search: '{cell_names_or_id}")
            root_ids = neuron_db.search(search_query=cell_names_or_id)
            if len(root_ids) == 1:
                root_id = root_ids[0]
            else:
                return redirect(url_for("app.search", filter_string=cell_names_or_id))

    if root_id is None:
        logger.info("Generated empty cell detail page")
        return render_template("cell_details.html")
    logger.info(f"Generating neuron info {activity_suffix(root_id, data_version)}")
    dct = cached_cell_details(
        cell_names_or_id=cell_names_or_id,
        root_id=root_id,
        neuron_db=neuron_db,
        data_version=data_version,
        reachability_stats=reachability_stats,
    )
    dct["min_syn_threshold"] = MIN_SYN_THRESHOLD
    if os.environ.get("BRAINCIRCUITS_TOKEN"):
        dct["show_braincircuits"] = 1
        dct["bc_url"] = url_for("app.matching_lines")
    logger.info(
        f"Generated neuron info for {root_id} with {len(dct['cell_attributes']) + len(dct['cell_annotations']) + len(dct['related_cells'])} items"
    )
    return render_template("cell_details.html", **dct)


@app.route("/pathways")
def pathways():
    source = request.args.get("source_cell_id", type=int)
    target = request.args.get("target_cell_id", type=int)
    min_syn_count = request.args.get("min_syn_count", type=int, default=0)
    logger.info(f"Rendering pathways from {source} to {target} with {min_syn_count=}")
    data_version = request.args.get("data_version", "")
    neuron_db = NeuronDataFactory.instance().get(version=data_version)
    for rid in [source, target]:
        if not neuron_db.is_in_dataset(rid):
            return render_error(
                message=f"Cell {rid} is not in the dataset.", title="Cell not found"
            )
    root_ids = [source, target]

    layers, data_rows = pathway_chart_data_rows(
        source=source,
        target=target,
        neuron_db=neuron_db,
        min_syn_count=min_syn_count,
    )
    cons = []
    for data_row in data_rows:
        cons.append([data_row[0], data_row[1], "", data_row[2], ""])
    return compile_network_html(
        center_ids=root_ids,
        contable=cons,
        neuron_db=neuron_db,
        show_regions=0,
        connections_cap=0,
        hide_weights=0,
        log_request=True,
        layers=layers,
        page_title="Pathways",
    )


@app.route("/path_length")
def path_length():
    source_cell_names_or_ids = request.args.get("source_cell_names_or_ids", "")
    target_cell_names_or_ids = request.args.get("target_cell_names_or_ids", "")
    data_version = request.args.get("data_version", "")
    min_syn_count = request.args.get("min_syn_count", type=int, default=0)
    download = request.args.get("download", 0, type=int)

    messages = []

    if source_cell_names_or_ids and target_cell_names_or_ids:
        neuron_db = NeuronDataFactory.instance().get(data_version)

        if source_cell_names_or_ids == target_cell_names_or_ids == "__sample_cells__":
            root_ids_src = neuron_db.search(search_query="gustatory")[:3]
            root_ids_target = neuron_db.search(search_query="motor")[:5]
            source_cell_names_or_ids = ", ".join([str(rid) for rid in root_ids_src])
            target_cell_names_or_ids = ", ".join([str(rid) for rid in root_ids_target])
            logger.info("Generating path lengths table for sample cells")
        else:
            root_ids_src = neuron_db.search(search_query=source_cell_names_or_ids)
            root_ids_target = neuron_db.search(search_query=target_cell_names_or_ids)
            logger.info(
                f"Generating path lengths table for '{source_cell_names_or_ids}' -> '{target_cell_names_or_ids}' "
                f"with {min_syn_count=} and {download=}"
            )
        if not root_ids_src:
            return render_error(
                title="No matching source cells",
                message=f"Could not find any cells matching '{source_cell_names_or_ids}'",
            )
        if not root_ids_target:
            return render_error(
                title="No matching target cells",
                message=f"Could not find any cells matching '{target_cell_names_or_ids}'",
            )

        if len(root_ids_src) > MAX_NODES_FOR_PATHWAY_ANALYSIS:
            messages.append(
                f"{display(len(root_ids_src))} source cells match your query. "
                f"Fetching pathways for the first {MAX_NODES_FOR_PATHWAY_ANALYSIS} sources."
            )
            root_ids_src = root_ids_src[:MAX_NODES_FOR_PATHWAY_ANALYSIS]
        if len(root_ids_target) > MAX_NODES_FOR_PATHWAY_ANALYSIS:
            messages.append(
                f"{display(len(root_ids_target))} target cells match your query. "
                f"Fetching pathways for the first {MAX_NODES_FOR_PATHWAY_ANALYSIS} targets."
            )
            root_ids_target = root_ids_target[:MAX_NODES_FOR_PATHWAY_ANALYSIS]

        matrix = distance_matrix(
            sources=root_ids_src,
            targets=root_ids_target,
            neuron_db=neuron_db,
            min_syn_count=min_syn_count,
        )
        logger.info(
            f"Generated path lengths table of length {len(matrix)}. "
            f"{len(root_ids_src)=}, {len(root_ids_target)=}, {min_syn_count=}, {download=}"
        )
        if len(matrix) <= 1:
            logger.error(
                f"No paths found from {source_cell_names_or_ids} to {target_cell_names_or_ids} with synapse "
                f"threshold {min_syn_count}."
            )
    else:
        matrix = []
        if source_cell_names_or_ids or target_cell_names_or_ids:
            messages.append("Please specify both source and target cell queries.")

    if matrix:
        if download:
            fname = "path_lengths.csv"
            return Response(
                "\n".join([",".join([str(r) for r in row]) for row in matrix]),
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={fname}"},
            )
        else:
            # format matrix with cell info/hyperlinks and pathway hyperlinks
            for i, r in enumerate(matrix[1:]):
                from_root_id = int(r[0])
                for j, val in enumerate(r):
                    if j == 0:
                        r[j] = (
                            f'<a href="{url_for("app.search", filter_string="id == " + str(from_root_id))}">{neuron_db.get_neuron_data(from_root_id)["name"]}</a><br><small>{from_root_id}</small>'
                        )
                    elif val > 0:
                        to_root_id = int(matrix[0][j])
                        if not min_syn_count:
                            q = f"{from_root_id} {OP_PATHWAYS} {to_root_id}"
                            slink = f'<a href="{url_for("app.search", filter_string=q)}" target="_blank" ><i class="fa-solid fa-list"></i> View cells as list</a>'
                        else:
                            slink = ""  # search by pathways is only available for default threshold
                        plink = f'<a href="{url_for("app.pathways", source_cell_id=from_root_id, target_cell_id=to_root_id, min_syn_count=min_syn_count)}" target="_blank" ><i class="fa-solid fa-route"></i> View Pathways chart</a>'
                        r[j] = f"{val} hops <br> <small>{plink} <br> {slink}</small>"
                    elif val == 0:
                        r[j] = ""
                    elif val == -1:
                        r[j] = '<span style="color:grey">no path</span>'

            for j, val in enumerate(matrix[0]):
                if j > 0:
                    matrix[0][
                        j
                    ] = f'<a href="{url_for("app.search", filter_string="id == " + str(val))}">{neuron_db.get_neuron_data(int(val))["name"]}</a><br><small>{val}</small>'

    info_text = (
        "With this tool you can specify one or more source cells + one or more target cells, set a "
        "minimum synapse threshold per connection, and get a matrix with shortest path lengths for all "
        "source/target pairs. From there, you can inspect / visualize the pathways between any pair of "
        f"cells in detail.<br>{FAQ_QA_KB['paths']['a']}"
    )

    return render_template(
        "path_lengths.html",
        source_cell_names_or_ids=source_cell_names_or_ids,
        target_cell_names_or_ids=target_cell_names_or_ids,
        collect_min_syn_count=True,
        min_syn_count=min_syn_count,
        matrix=matrix,
        download_url=url_for(
            "app.path_length",
            download=1,
            source_cell_names_or_ids=source_cell_names_or_ids,
            target_cell_names_or_ids=target_cell_names_or_ids,
        ),
        info_text=info_text,
        messages=messages,
    )


@app.route("/connectivity")
def connectivity():
    data_version = request.args.get("data_version", "")
    nt_type = request.args.get("nt_type", None)
    min_syn_cnt = request.args.get("min_syn_cnt", default=0, type=int)
    connections_cap = request.args.get("cap", default=50, type=int)
    group_by = request.args.get("group_by", default="")
    show_regions = request.args.get("show_regions", default=0, type=int)
    include_partners = request.args.get("include_partners", default=0, type=int)
    hide_weights = request.args.get("hide_weights", default=0, type=int)
    cell_names_or_ids = request.args.get("cell_names_or_ids", "")
    # This flag labels the list of cells with "A, B, C, .." in the order they're specified. Used for mapping motif node
    # names to cell names.
    label_abc = request.args.get("label_abc", type=bool)
    download = request.args.get("download")
    # headless network view (no search box / nav bar etc.)
    headless = request.args.get("headless", default=0, type=int)
    log_request = request.args.get(
        "log_request", default=0 if headless else 1, type=int
    )

    message = None

    group_by_options = {"": "Individual Cells"}
    group_by_options.update({k: display(k) for k in NETWORK_GROUP_BY_ATTRIBUTES})
    group_by_options.update(
        {
            f"{k}_and_side": f"{display(k)} & side"
            for k in NETWORK_GROUP_BY_ATTRIBUTES
            if k != "side"
        }
    )
    if group_by.endswith("_and_side"):
        group_by_attribute_name = group_by[: -len("_and_side")]
        split_groups_by_side = True
    else:
        group_by_attribute_name = group_by
        split_groups_by_side = False

    if not cell_names_or_ids:
        con_doc = FAQ_QA_KB["connectivity"]
        return render_template(
            "connectivity.html",
            cell_names_or_ids=cell_names_or_ids,
            min_syn_cnt=min_syn_cnt,
            nt_type=nt_type,
            network_html=None,
            info_text="With this tool you can specify one or more cells and visualize their connectivity network.<br>"
            f"{con_doc['a']}",
            message=None,
            data_versions=DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
            group_by_options=group_by_options,
            group_by=group_by,
            data_version=data_version,
            show_regions=show_regions,
            hide_weights=hide_weights,
            num_matches=0,
        )
    else:
        neuron_db = NeuronDataFactory.instance().get(data_version)
        node_labels = None
        if cell_names_or_ids == "__sample_cells__":
            root_ids = [
                720575940623725972,
                720575940630057979,
                720575940633300148,
                720575940644300323,
                720575940640176848,
                720575940627796298,
            ]
            root_ids = [r for r in root_ids if neuron_db.is_in_dataset(r)]
            cell_names_or_ids = ", ".join([str(rid) for rid in root_ids])
            logger.info("Generating connectivity network for sample cells")
        else:
            root_ids = neuron_db.search(search_query=cell_names_or_ids)
            if label_abc:
                if not len(root_ids) == 3:
                    raise ValueError(
                        f"Unexpected flag {label_abc=} for {len(root_ids)} matching root IDs"
                    )
                node_labels = {
                    root_ids[i]: ltr for i, ltr in enumerate(["A", "B", "C"])
                }

            if log_request:
                logger.info(
                    ("Downloading " if download else "Generating ")
                    + f"network for '{cell_names_or_ids}'"
                )

        if not root_ids:
            return render_error(
                title="No matching cells found",
                message=f"Could not find any cells matching '{cell_names_or_ids}'",
            )
        elif len(root_ids) == 1:
            # if only one match found, show some connections to it's partners (instead of lonely point)
            include_partners = True

        if len(root_ids) == 1 and not nt_type and not min_syn_cnt:
            # this simplest case (also used in cell details page) can be handled more efficiently
            contable = neuron_db.cell_connections(root_ids[0])
        else:
            contable = neuron_db.connections(
                ids=root_ids,
                nt_type=nt_type,
                induced=include_partners == 0,
                min_syn_count=min_syn_cnt,
            )
        if log_request:
            logger.info(
                f"Generated connections table for {len(root_ids)} cells with {connections_cap=}, {download=} {min_syn_cnt=} {nt_type=}"
            )
        if download:
            if len(contable) > 100000:
                return render_error(
                    message=f"The network generatad for your query is too large to download ({len(contable)} connections). Please refine the query and try again.",
                    title="Selected network is too large for download",
                )
            if download.lower() == "json":
                return Response(
                    json.dumps(
                        synapse_table_to_json_dict(
                            table=contable,
                            neuron_data_fetcher=lambda rid: neuron_db.get_neuron_data(
                                rid
                            ),
                            meta_data={
                                "generated": str(datetime.now()),
                                "data_version": data_version
                                or DEFAULT_DATA_SNAPSHOT_VERSION,
                                "query": cell_names_or_ids,
                                "min_syn_cnt": min_syn_cnt,
                                "nt_types": nt_type,
                                "url": str(request.url),
                            },
                        ),
                        indent=4,
                    ),
                    mimetype="application/json",
                    headers={
                        "Content-disposition": "attachment; filename=connections.json"
                    },
                )
            else:
                return Response(
                    synapse_table_to_csv_string(contable),
                    mimetype="text/csv",
                    headers={
                        "Content-disposition": "attachment; filename=connections.csv"
                    },
                )

        network_html = compile_network_html(
            center_ids=root_ids,
            node_labels=node_labels,
            contable=contable,
            neuron_db=neuron_db,
            show_regions=show_regions,
            group_by_attribute_name=group_by_attribute_name,
            split_groups_by_side=split_groups_by_side,
            connections_cap=connections_cap,
            hide_weights=hide_weights,
            log_request=log_request,
        )
        if headless:
            return network_html
        else:
            return render_template(
                "connectivity.html",
                cell_names_or_ids=cell_names_or_ids,
                min_syn_cnt=min_syn_cnt,
                nt_type=nt_type,
                cap=connections_cap,
                max_cap=200,
                network_html=network_html,
                info_text=None,
                message=message,
                data_versions=DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
                data_version=data_version,
                group_by_options=group_by_options,
                group_by=group_by,
                show_regions=show_regions,
                include_partners=include_partners,
                hide_weights=hide_weights,
                num_matches=len(root_ids),
            )


@app.route("/flywire_neuropil_url")
def flywire_neuropil_url():
    selected = request.args.get("selected")
    segment_ids = [REGIONS[r][0] for r in selected.split(",") if r in REGIONS]
    url = nglui.url_for_neuropils(segment_ids)
    return ngl_redirect_with_client_check(ngl_url=url)


@app.route("/neuropils")
def neuropils():
    landing = False
    selected = request.args.get("selected")
    logger.info(f"Rendering neuropils page with {selected=}")
    if selected:
        selected = selected.strip(",")
        selected_ids = [r for r in selected.split(",") if r]
        if len(selected_ids) > 1:
            caption = ", ".join([NEUROPIL_DESCRIPTIONS[r] for r in selected_ids])
        else:
            caption = NEUROPIL_DESCRIPTIONS[selected_ids[0]]
        caption = display(caption)
        selected = ",".join(selected_ids)
    else:
        selected = ",".join(REGIONS.keys())
        landing = True
        caption = '<i class="fa-solid fa-arrow-down"></i> use links to select regions <i class="fa-solid fa-arrow-down"></i>'

    return render_template(
        "neuropils.html",
        selected=selected,
        REGIONS_JSON=REGIONS_JSON,
        caption=caption,
        landing=landing,
    )


@app.route("/heatmaps")
def heatmaps():
    data_version = request.args.get("data_version", "")
    group_by = request.args.get("group_by")
    count_type = request.args.get("count_type")
    logger.info(f"Rendering heatmaps page with {data_version=} {group_by=}")

    dct = heatmap_data(
        neuron_db=NeuronDataFactory.instance().get(data_version),
        group_by=group_by,
        count_type=count_type,
    )
    dct["data_version"] = data_version

    return render_template("heatmaps.html", **dct)


@app.route("/labeling_log")
def labeling_log():
    root_id = request.args.get("root_id")
    logger.info(f"Loading labling log for {root_id}")
    root_id = int(root_id)
    neuron_db = NeuronDataFactory.instance().get()
    nd = neuron_db.get_neuron_data(root_id)
    if not nd:
        return render_error(
            f"Neuron with ID {root_id} not found in v{DEFAULT_DATA_SNAPSHOT_VERSION} data snapshot."
        )

    labels_data = neuron_db.get_label_data(root_id=root_id)
    labeling_log = [
        f'<small><b>{ld["label"]}</b> - labeled by {ld["user_name"]}'
        + (f' from {ld["user_affiliation"]}' if ld["user_affiliation"] else "")
        + f' on {ld["date_created"]}</small>'
        for ld in sorted(
            labels_data or [], key=lambda x: x["date_created"], reverse=True
        )
    ]

    def format_log(labels):
        return "<br>".join([f"&nbsp; <b>&#x2022;</b> &nbsp; {t}" for t in labels])

    return render_info(
        title=f"Labeling Info & Credits<br><small style='color:teal'>&nbsp;  &nbsp;  &nbsp; {nd['name']}  &#x2022;  {nd['root_id']}</small>",
        message=format_log(labeling_log)
        + f"<br><br>Last synced: <b>{neuron_db.labels_ingestion_timestamp()}</b>",
        back_button=0,
    )


@app.route("/motifs/", methods=["GET", "POST"])
def motifs():
    logger.info(f"Loading motifs search with {request.args}")
    search_results = []

    if request.args:
        motifs_query = MotifSearchQuery.from_form_query(
            request.args, NeuronDataFactory.instance()
        )
        search_results = motifs_query.search()
        logger.info(
            f"Motif search with {motifs_query} found {len(search_results)} matches"
        )
        query = request.args
        show_explainer = False
    else:
        # make a default state so that hitting search finds results (empty query will err)
        query = dict(
            enabledAB=True,
            minSynapseCountAB=1,
            enabledBA=True,
            minSynapseCountBA=1,
            enabledAC=True,
            minSynapseCountAC=1,
            enabledCA=True,
            minSynapseCountCA=1,
            enabledBC=True,
            minSynapseCountBC=1,
            enabledCB=True,
            minSynapseCountCB=1,
        )
        show_explainer = True

    return render_template(
        "motif_search.html",
        regions=list(REGIONS.keys()),
        NEURO_TRANSMITTER_NAMES=NEURO_TRANSMITTER_NAMES,
        query=query,
        results=search_results,
        show_explainer=show_explainer,
    )


@app.route("/neuroglancer_url")
def neuroglancer_url():
    """
    Returns a Neuroglancer state URL for a given seg_id (or list of seg_ids), with the selected one optionally highlighted.
    Only supports the EW2 (stroeh_mouse_retina) dataset.
    Delegates state construction to codex.utils.nglui to keep style elements centralized.
    """
    segids = request.args.getlist("segids")
    selected = request.args.get("selected")
    highlight_color = request.args.get("highlight_color", "#29ff29")

    # Optional view parameters (pass-through if provided)
    position = request.args.get("position")
    cross_section_scale = request.args.get("crossSectionScale")
    projection_orientation = request.args.get("projectionOrientation")
    projection_scale = request.args.get("projectionScale")
    projection_depth = request.args.get("projectionDepth")

    # Parse inputs
    try:
        segids = [int(s) for s in segids if str(s).isdigit()]
    except Exception:
        return {"error": "Invalid segids"}, 400
    try:
        selected_int = int(selected) if selected and str(selected).isdigit() else None
    except Exception:
        selected_int = None

    # Convert JSON-like strings to Python for complex params if supplied
    import json as _json
    def _parse_or_none(val):
        if val is None:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None

    position_val = _parse_or_none(position)
    proj_orient_val = _parse_or_none(projection_orientation)

    # cross_section_scale, projection_scale, projection_depth are floats
    def _to_float(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    cross_section_val = _to_float(cross_section_scale)
    proj_scale_val = _to_float(projection_scale)
    proj_depth_val = _to_float(projection_depth)

    url = url_for_ew2_segments(
        segment_ids=segids,
        selected=selected_int,
        highlight_color=highlight_color,
        position=position_val,
        cross_section_scale=cross_section_val,
        projection_orientation=proj_orient_val,
        projection_scale=proj_scale_val,
        projection_depth=proj_depth_val,
    )
    logger.info(f"Returning Neuroglancer URL: {url}")
    return {"url": url}


@app.route("/neuroglancer_shorten")
def neuroglancer_shorten():
    """
    Build the EW2 Neuroglancer state (same params as /neuroglancer_url) and shorten it
    via the configured jsonStateServer. Returns {"short_url": str}.
    """
    segids = request.args.getlist("segids")
    selected = request.args.get("selected")
    highlight_color = request.args.get("highlight_color", "#29ff29")

    position = request.args.get("position")
    cross_section_scale = request.args.get("crossSectionScale")
    projection_orientation = request.args.get("projectionOrientation")
    projection_scale = request.args.get("projectionScale")
    projection_depth = request.args.get("projectionDepth")

    try:
        segids = [int(s) for s in segids if str(s).isdigit()]
    except Exception:
        return jsonify({"error": "Invalid segids"}), 400
    selected_int = int(selected) if selected and str(selected).isdigit() else None

    import json as _json
    def _parse_or_none(val):
        if val is None:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return None
    position_val = _parse_or_none(position)
    proj_orient_val = _parse_or_none(projection_orientation)
    def _to_float(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    cross_section_val = _to_float(cross_section_scale)
    proj_scale_val = _to_float(projection_scale)
    proj_depth_val = _to_float(projection_depth)

    cfg = ew2_config_dict(
        segment_ids=segids,
        selected=selected_int,
        highlight_color=highlight_color,
        position=position_val,
        cross_section_scale=cross_section_val,
        projection_orientation=proj_orient_val,
        projection_scale=proj_scale_val,
        projection_depth=proj_depth_val,
    )
    short = shorten_ew2_state(cfg)
    if not short:
        return jsonify({"error": "Shorten failed"}), 502
    # Normalize to viewer short URL if server returned a state URL
    if short.startswith("http") and not short.startswith(NGL_EW2_BASE_URL):
        viewer_url = f"{NGL_EW2_BASE_URL}/#!{short}"
    else:
        viewer_url = short
    logger.info(f"Shortened Neuroglancer URL: {viewer_url}")
    return jsonify({"short_url": viewer_url})
