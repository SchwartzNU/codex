import numpy as np
from typing import Sequence, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binned_statistic_2d

Number = int | float

_PLANE_AXES = {
    "xy": (0, 1), "yx": (1, 0),
    "xz": (0, 2), "zx": (2, 0),
    "yz": (1, 2), "zy": (2, 1),
}

_PLANE_NORMAL = {
    "xy": np.array([0, 0, 1.0]),
    "yx": np.array([0, 0, 1.0]),
    "xz": np.array([0, 1.0, 0]),
    "zx": np.array([0, 1.0, 0]),
    "yz": np.array([1.0, 0, 0]),
    "zy": np.array([1.0, 0, 0]),
}

def _project(arr: np.ndarray, ix: int, iy: int, /) -> np.ndarray:
    """Return 2-column slice (arr[:, (ix, iy)])."""
    return arr[:, (ix, iy)].copy()

def _axis_extents(v: np.ndarray):
    gx = (v[:, 0].min(), v[:, 0].max())
    gy = (v[:, 1].min(), v[:, 1].max())
    gz = (v[:, 2].min(), v[:, 2].max())
    dx, dy, dz = np.ptp(v, axis=0)
    return dict(x=gx, y=gy, z=gz), dict(x=dx, y=dy, z=dz)

def _plane_axes(plane: str) -> tuple[str, str]:
    if len(plane) != 2 or any(c not in "xyz" for c in plane.lower()):
        raise ValueError(f"invalid plane spec '{plane}'")
    return plane[0].lower(), plane[1].lower()

def _trapezoid_3d(
    p0_3d: np.ndarray, p1_3d: np.ndarray,
    r0: float, r1: float,
    plane: str,
) -> np.ndarray | None:
    v = p1_3d - p0_3d
    if not np.any(v):
        return None
    n3 = np.cross(v, _PLANE_NORMAL[plane])
    L = np.linalg.norm(n3)
    if L == 0:
        return None
    n3 /= L
    ix, iy = _PLANE_AXES[plane]
    n2 = n3[[ix, iy]]
    p0 = p0_3d[[ix, iy]]
    p1 = p1_3d[[ix, iy]]
    return np.array([
        p0 + n2 * r0,
        p0 - n2 * r0,
        p1 - n2 * r1,
        p1 + n2 * r1,
    ], dtype=float)

def _radii_to_sizes(rr: np.ndarray, xlim, ylim, fig_width: int = 800) -> np.ndarray:
    """Map radii to marker sizes for Plotly scatter."""
    ref_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    return np.clip(rr / ref_range * fig_width * 0.07, 2, 40)

def _nice_bar_length(span: float) -> float:
    """Pick a nice scale bar length (in µm) for a given axis span."""
    if not np.isfinite(span) or span <= 0:
        return 1.0
    target = span * 0.2
    exponent = int(np.floor(np.log10(target)))
    base = 10.0 ** exponent
    for m in (5, 2, 1):
        length = m * base
        if length <= target:
            return length
    return base

def _add_scale_bars(fig, *, row: int, col: int, xr: tuple[float, float], yr: tuple[float, float], length_um: float | None = None):
    """Add small horizontal and vertical scale bars in bottom-left corner of a subplot."""
    x0, x1 = xr
    y0, y1 = yr
    span_x = x1 - x0
    span_y = y1 - y0
    if length_um is None:
        length_um = _nice_bar_length(min(span_x, span_y))
    pad_x = span_x * 0.05
    pad_y = span_y * 0.05
    bx0 = x0 + pad_x
    by0 = y0 + pad_y
    # horizontal bar (X)
    fig.add_shape(type="line", x0=bx0, y0=by0, x1=bx0 + length_um, y1=by0,
                  line=dict(color="black", width=2), row=row, col=col)
    # vertical bar (Y/Z)
    fig.add_shape(type="line", x0=bx0, y0=by0, x1=bx0, y1=by0 + length_um,
                  line=dict(color="black", width=2), row=row, col=col)
    # label centered above horizontal bar
    fig.add_annotation(x=bx0 + length_um * 0.5, y=by0 + span_y * 0.03,
                       text=f"{int(length_um) if length_um >= 1 else length_um} µm",
                       showarrow=False, font=dict(size=10, color="black"),
                       row=row, col=col)

# SVG path for a rotated ellipse (used for ellipse overlays in Plotly)
def _ellipse_path(x0, y0, width, height, angle_deg):
    angle = np.deg2rad(angle_deg)
    rx = width / 2
    ry = height / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    sx = x0 - rx * cos_a
    sy = y0 - rx * sin_a
    ex = x0 + rx * cos_a
    ey = y0 + rx * sin_a
    return (
        f"M {sx},{sy} "
        f"A {rx},{ry} {angle_deg} 1 0 {ex},{ey} "
        f"A {rx},{ry} {angle_deg} 1 0 {sx},{sy}"
    )

def _soma_ellipse2d(soma, plane: str, *, scale: float = 1.0):
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {_PLANE_AXES.keys()}")
    ix, iy = _PLANE_AXES[plane]
    k = 3 - ix - iy
    B = soma.R @ np.diag(1.0 / soma.axes**2) @ soma.R.T
    B_pp = B[[ix, iy]][:, [ix, iy]]
    B_pq = B[[ix, iy], k].reshape(2, 1)
    B_qq = B[k, k]
    Q = B_pp - (B_pq @ B_pq.T) / B_qq
    eigval, eigvec = np.linalg.eigh(Q)
    half_axes = 1.0 / np.sqrt(eigval)
    order = np.argsort(-half_axes)
    width, height = 2 * half_axes[order] * scale
    angle_deg = np.degrees(np.arctan2(eigvec[1, order[0]], eigvec[0, order[0]]))
    centre_xy = soma.center[[ix, iy]] * scale
    return dict(
        x=centre_xy[0],
        y=centre_xy[1],
        width=width,
        height=height,
        angle=angle_deg,
    )

def projection(
    skel,
    mesh=None,
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    bins: int | tuple[int, int] = 800,
    scale: float | Sequence[Number] = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_skel: bool = True,
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    mesh_cmap: str = "Blues",
    skel_cmap: str = "Pastel2",
    vmax_fraction: float = 0.10,
    edge_lw: float = 0.5,
    circle_alpha: float = 0.25,
    cylinder_alpha: float = 0.5,
    highlight_nodes: int | Sequence[int] | None = None,
    highlight_face_alpha: float = 0.5,
    unit: str | None = None,
    draw_soma_mask: bool = True,
    color_by: str = "fixed",
) -> go.Figure:
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")
    ix, iy = _PLANE_AXES[plane]
    if not isinstance(scale, Sequence) or isinstance(scale, str):
        scale = [scale, scale]
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    if unit is None:
        unit = skel.meta.get("unit", "")

    highlight_set = set(map(int, np.atleast_1d(highlight_nodes))) if highlight_nodes is not None else set()
    swc_colors = [
        "gray",    # 0 undefined
        "teal",    # 1 soma
        "orange",  # 2 axon
        "olive",   # 3 dendrite
        "purple",  # 4 apical dendrite
        "goldenrod", # 5 fork point
        "brown"    # 6 end point
    ]
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr = skel.radii[radius_metric] * scl_skel
    xy_mesh = _project(mesh.vertices, ix, iy) * scl_mesh if mesh is not None else np.empty((0, 2), dtype=float)
    def _crop_window(xy):
        keep = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep
    keep_skel = _crop_window(xy_skel)
    idx_keep = np.flatnonzero(keep_skel)
    xy_skel = xy_skel[keep_skel]
    rr = rr[keep_skel]
    if color_by == "ntype" and skel.ntype is not None:
        col_nodes = [swc_colors[nt] for nt in skel.ntype[idx_keep]]
    else:
        col_nodes = "red"
    if mesh is not None and xy_mesh.size:
        keep_mesh = _crop_window(xy_mesh)
        xy_mesh = xy_mesh[keep_mesh]
    heatmap_trace = None
    if mesh is not None and xy_mesh.size:
        bins_arg = bins if isinstance(bins, int) else tuple(bins)
        hist, xedges, yedges, _ = binned_statistic_2d(
            xy_mesh[:, 0], xy_mesh[:, 1], None,
            statistic="count", bins=bins_arg,
        )
        hist = hist.T
        if xlim is None: xlim = (xedges[0], xedges[-1])
        if ylim is None: ylim = (yedges[0], yedges[-1])
        heatmap_trace = go.Heatmap(
            z=hist,
            x=xedges,
            y=yedges,
            colorscale=mesh_cmap,
            zmax=hist.max() * vmax_fraction,
            opacity=1.0,
            showscale=False,
        )
    fig = go.Figure()
    if heatmap_trace is not None:
        fig.add_trace(heatmap_trace)
    
    if draw_skel and xy_skel.size > 0:
        if xlim is not None and ylim is not None:
            xlim = xlim
            ylim = ylim
        else:
            xlim = (xy_skel[:, 0].min(), xy_skel[:, 0].max())
            ylim = (xy_skel[:, 1].min(), xy_skel[:, 1].max())
        marker_sizes = _radii_to_sizes(rr, xlim, ylim)
        fig.add_trace(go.Scatter(
            x=xy_skel[:, 0][1:], y=xy_skel[:, 1][1:],
            mode="markers",
            marker=dict(
                size=marker_sizes[1:],
                color=col_nodes[1:] if isinstance(col_nodes, list) else col_nodes,
                opacity=circle_alpha,
                line=dict(width=0.2, color=col_nodes[1:] if isinstance(col_nodes, list) else col_nodes)
            ),
            showlegend=False,
        ))
        if highlight_set:
            orig_ids = np.flatnonzero(keep_skel)
            hilite_mask = np.isin(orig_ids, list(highlight_set))
            if hilite_mask.any():
                fig.add_trace(go.Scatter(
                    x=xy_skel[hilite_mask, 0], y=xy_skel[hilite_mask, 1],
                    mode="markers",
                    marker=dict(size=marker_sizes[hilite_mask], color="green", opacity=highlight_face_alpha),
                    showlegend=False,
                ))
    if xy_skel.shape[0] > 0:
        fig.add_trace(go.Scatter(
            x=[xy_skel[0, 0]], y=[xy_skel[0, 1]],
            mode="markers",
            marker=dict(size=15, color="black"),
            showlegend=False,
        ))
    # Ellipse/ellipse fallback as a path!
    if (draw_soma_mask and mesh is not None and skel.soma is not None and skel.soma.verts is not None):
        xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy) * scl_mesh
        xy_soma = xy_soma[_crop_window(xy_soma)]
        col_soma = swc_colors[1] if color_by == "ntype" else "pink"
        fig.add_trace(go.Scatter(
            x=xy_soma[:, 0], y=xy_soma[:, 1],
            mode="markers",
            marker=dict(size=3, color=col_soma, opacity=0.5),
            showlegend=False,
        ))
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ellipse_path = _ellipse_path(
            ell["x"], ell["y"], ell["width"], ell["height"], ell["angle"]
        )
        fig.add_shape(
            type="path",
            path=ellipse_path,
            line=dict(color="black", width=2, dash="dash"),
            opacity=0.9,
        )
    elif hasattr(skel, "soma") and skel.soma is not None:
        c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
        centre_col = swc_colors[1] if color_by == "ntype" else "black"
        fig.add_shape(
            type="circle",
            x0=c_xy[0] - skel.soma.equiv_radius * scl_skel,
            y0=c_xy[1] - skel.soma.equiv_radius * scl_skel,
            x1=c_xy[0] + skel.soma.equiv_radius * scl_skel,
            y1=c_xy[1] + skel.soma.equiv_radius * scl_skel,
            line=dict(color=centre_col, width=1, dash="dash"),
            opacity=0.9,
        )
    if draw_skel and skel.edges.size > 0 and draw_edges:
        idx_map = -np.ones(len(keep_skel), int)
        idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())
        ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        xlines = []
        ylines = []
        for n0, n1 in edges_kept:
            i0, i1 = idx_map[[n0, n1]]
            xlines.extend([xy_skel[i0, 0], xy_skel[i1, 0], None])
            ylines.extend([xy_skel[i0, 1], xy_skel[i1, 1], None])
        fig.add_trace(go.Scatter(
            x=xlines, y=ylines, mode="lines",
            line=dict(color="black", width=0.2),
            opacity=cylinder_alpha,
            showlegend=False,
        ))
    if draw_cylinders and draw_skel and skel.edges.size > 0:
        idx_map = -np.ones(len(keep_skel), int)
        idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())
        ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        for n0, n1 in edges_kept:
            i0, i1 = idx_map[[n0, n1]]
            quad = _trapezoid_3d(
                skel.nodes[n0] * scl_skel,
                skel.nodes[n1] * scl_skel,
                rr[i0], rr[i1],
                plane,
            )
            if quad is not None:
                quad_x = np.append(quad[:, 0], quad[0, 0])
                quad_y = np.append(quad[:, 1], quad[0, 1])
                fig.add_trace(go.Scatter(
                    x=quad_x, y=quad_y,
                    fill="toself", mode="lines",
                    line=dict(color="red", width=0.2),
                    fillcolor="rgba(200,0,0,0.5)",
                    opacity=cylinder_alpha,
                    showlegend=False,
                ))
    unit_str = f" ({unit})" if unit else ""
    fig.update_layout(
        xaxis=dict(title=f"{plane[0]}{unit_str}", range=xlim, showgrid=True, zeroline=False, mirror=True, ticks="inside", showline=True, linewidth=2, linecolor='black', constrain='domain'),
        yaxis=dict(title=f"{plane[1]}{unit_str}", range=ylim, scaleanchor="x", scaleratio=1, showgrid=True, zeroline=False, mirror=True, ticks="inside", showline=True, linewidth=2, linecolor='black', constrain='domain'),
        margin=dict(l=10, r=10, b=10, t=30),
        width=1000, height=900,
        template=None,
    )

    
    return fig

def threeviews(
    skel,
    mesh=None,
    *,
    planes: tuple[str, str, str] | list[str] = ["xy", "xz", "zy"],
    scale: float | tuple[float, float] = 1.,
    title: str | None = None,
    figsize: tuple[int, int] = (500, 500),
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    draw_soma_mask: bool = True,
    **plot_kwargs,
):
    planes = list(planes)
    if len(planes) != 3:
        raise ValueError("planes must be a sequence of exactly three plane strings")
    if not isinstance(scale, Sequence) or isinstance(scale, str):
        scale = [scale, scale]
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)
    if title is None:
        title = skel.meta.get("id", None)
    if mesh is not None and mesh.vertices.size:
        v_mesh = mesh.vertices.view(np.ndarray) * scl_mesh
        v_all = np.vstack((v_mesh, skel.nodes * scl_skel))
    else:
        v_all = skel.nodes * scl_skel
    lims, spans = _axis_extents(v_all)
    def _limits(p: str):
        h, v = _plane_axes(p)
        return lims[h], lims[v]
    # Build the subplot grid as in Matplotlib
    # Layout:
    # B .
    # A C
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.18, 0.82],      # control heights
        column_widths=[0.82, 0.18],    # control widths
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=["", "", "", ""],
    )
    panels = {"B": (1, 1), "A": (2, 1), "C": (2, 2)}
    for label, plane in zip(("A", "B", "C"), planes):
        xlim, ylim = _limits(plane)
        subfig = projection(
            skel,
            mesh,
            plane=plane,
            scale=scale,
            xlim=xlim,
            ylim=ylim,
            draw_edges=draw_edges,
            draw_cylinders=draw_cylinders,
            draw_soma_mask=draw_soma_mask,
            **plot_kwargs,
        )
        for trace in subfig.data:
            fig.add_trace(trace, row=panels[label][0], col=panels[label][1])
        if getattr(subfig.layout, "shapes", None):
            for s in subfig.layout.shapes:
                fig.add_shape(s, row=panels[label][0], col=panels[label][1])
        # update subplot axes: always show ticks, show box
        if label=="A":
            axes_title = (subfig.layout.xaxis.title.text, subfig.layout.yaxis.title.text)
            showticklabels = (True, True)
        elif label=="B":
            axes_title = (None, subfig.layout.xaxis.title.text)
            showticklabels = (False, True)
        else:
            axes_title = (subfig.layout.yaxis.title.text, None)
            showticklabels = (True, False)
        fig.update_xaxes(title=axes_title[0],
                         showticklabels=showticklabels[0],
                         showgrid=False, 
                         zeroline=False,    
                         mirror=True, 
                         ticks="outside", 
                         showline=True, 
                         linewidth=2, 
                         linecolor='black',
                         row=panels[label][0], 
                         col=panels[label][1])
        fig.update_yaxes(title=axes_title[1],
                         showticklabels=showticklabels[1],
                         showgrid=False, 
                         zeroline=False, 
                         mirror=True, 
                         ticks="outside", 
                         showline=True, 
                         linewidth=2, 
                         linecolor='black',
                         row=panels[label][0], 
                         col=panels[label][1])

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title=title or "",
        template=None,  
        plot_bgcolor="white",   
        paper_bgcolor="white"   
    )

    return fig


def strat_profile_plotly(
    warped_skeleton,
    *,
    z_profile_extent: tuple[float, float] = (-30, 50),
    seg_id: str | None = None,
    fig_height_px: int = 220,
    max_width_px: int = 600,
):
    """Render ONLY the stratification profile (no side-view skeleton)."""
    colors = ["#d62728", "#1f77b4"]
    zp = warped_skeleton.extra["z_profile"]

    fig = make_subplots(rows=1, cols=1)

    # Distribution line
    fig.add_trace(go.Scatter(
        x=zp["distribution"], y=zp["x"],
        mode="lines", line=dict(color='black', width=2),
        showlegend=False,
    ), row=1, col=1)
    # Histogram bars
    fig.add_trace(go.Bar(
        x=zp["histogram"], y=zp["x"], orientation='h',
        marker=dict(color='gray', opacity=0.5),
        width=(zp["x"][1] - zp["x"][0]) if len(zp["x"]) > 1 else 0.8,
        showlegend=False,
    ), row=1, col=1)
    # Dashed horizontal SAC lines at z=0 and z=12
    for i, yv in enumerate((0, 12)):
        fig.add_hline(y=yv, line_dash='dash', line_color=colors[i], row=1, col=1)

    # Axes
    fig.update_xaxes(
        title_text='dendritic length (µm)', row=1, col=1,
        showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'
    )
    fig.update_yaxes(
        title_text='Z (µm)', row=1, col=1,
        showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black',
        range=z_profile_extent
    )

    fig.update_layout(
        height=fig_height_px, width=None,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=5, r=5, t=10, b=5),
        showlegend=False,
    )

    return fig


# ------------------------
# Lightweight Skeleton API
# ------------------------
class _SimpleSoma:
    def __init__(self, center: np.ndarray, equiv_radius: float):
        self.center = np.asarray(center, dtype=float)
        self.equiv_radius = float(equiv_radius)
        self.verts = None  # no mesh verts by default


class SimpleSkeleton:
    """
    Minimal skeleton wrapper compatible with projection()/threeviews().
    Attributes:
      - nodes: (N,3) float array
      - edges: (M,2) int array (child, parent) with parent >= 0
      - radii: dict with at least one key, e.g. {'r': (N,)}
      - ntype: None or array-like for node types (optional)
      - soma: object with .equiv_radius and .center (optional)
      - meta: dict, may include {'unit': 'µm'}
      - extra: dict for optional extras, e.g. {'z_profile': {...}}
    """
    def __init__(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        radii: np.ndarray,
        *,
        unit: str = "µm",
        ntype: Optional[Sequence[int]] = None,
        soma_center: Optional[Sequence[float]] = None,
        soma_radius: Optional[float] = None,
        z_profile: Optional[dict] = None,
    ):
        self.nodes = np.asarray(nodes, dtype=float)
        self.edges = np.asarray(edges, dtype=int) if len(edges) else np.empty((0, 2), dtype=int)
        self.radii = {"r": np.asarray(radii, dtype=float)}
        self.ntype = None if ntype is None else np.asarray(ntype, dtype=int)
        self.meta = {"unit": unit}
        if soma_center is None:
            soma_center = self.nodes[0] if len(self.nodes) else np.zeros(3)
        if soma_radius is None:
            soma_radius = float(np.median(self.radii["r"])) if len(self.radii["r"]) else 1.0
        self.soma = _SimpleSoma(soma_center, soma_radius)
        self.extra = {}
        if z_profile is not None:
            self.extra["z_profile"] = z_profile

    def recommend_radius(self):
        # Return key and human-friendly label
        return "r", "radius"


def _compute_z_profile(nodes: np.ndarray, edges: np.ndarray, nbins: int = 80) -> dict:
    """
    Approximate dendritic length distribution along Z using segment lengths
    binned by mean Z of each segment.
    Returns dict with keys 'x' (z-axis midpoints), 'histogram' (length per bin),
    and 'distribution' (normalized smooth curve).
    """
    if nodes.size == 0 or edges.size == 0:
        return {"x": np.array([]), "histogram": np.array([]), "distribution": np.array([])}
    # Filter valid edges (parent >= 0)
    valid = edges[:, 1] >= 0
    e = edges[valid]
    p = nodes[e[:, 1]]
    c = nodes[e[:, 0]]
    seg_len = np.linalg.norm(c - p, axis=1)
    z_mean = 0.5 * (c[:, 2] + p[:, 2])
    zmin, zmax = float(np.min(nodes[:, 2])), float(np.max(nodes[:, 2]))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        zmin, zmax = -30.0, 50.0
    hist, edges_z = np.histogram(z_mean, bins=nbins, range=(zmin, zmax), weights=seg_len)
    x = 0.5 * (edges_z[:-1] + edges_z[1:])
    # Simple smoothing for a nicer line
    if hist.size >= 5:
        kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        kernel /= kernel.sum()
        pad = 2
        hpad = np.pad(hist, (pad, pad), mode="edge")
        smooth = np.convolve(hpad, kernel, mode="valid")
    else:
        smooth = hist.astype(float)
    dist = smooth / smooth.max() if smooth.max() > 0 else smooth
    return {"x": x, "histogram": hist, "distribution": dist}


def simple_skeleton_from_swc(swc_path: str) -> SimpleSkeleton:
    """
    Load a minimal skeleton from an SWC file path.
    """
    # Local import to avoid circulars if arbor_stats is used independently
    from .arbor_stats import load_swc as _load_swc
    nodes, radii, edge_list = _load_swc(swc_path)
    # Convert to (child, parent) pairs with parent >= 0 only for drawing
    edges = np.array([(n, p) for (n, p) in edge_list if p >= 0], dtype=int)
    zprof = _compute_z_profile(nodes, edges)
    return SimpleSkeleton(nodes=nodes, edges=edges, radii=radii, z_profile=zprof)


def front_side_plotly(
    skel: SimpleSkeleton,
    *,
    planes: Sequence[str] = ("xy", "xz"),
    fig_height_px: int = 220,
    max_width_px: int = 600,
    title: Optional[str] = None,
    xlim_xy: Optional[tuple[float, float]] = None,
    ylim_xy: Optional[tuple[float, float]] = None,
    xlim_xz: Optional[tuple[float, float]] = None,
    ylim_xz: Optional[tuple[float, float]] = None,
):
    """
    Compose two orthogonal projections (front/side) into a single Plotly figure.
    """
    if len(planes) != 2:
        raise ValueError("planes must have length 2, e.g., ('xy','xz')")
    # Generate each projection first
    # Derive consistent axis limits so units match across subplots
    nodes = skel.nodes
    gx = (float(nodes[:, 0].min()), float(nodes[:, 0].max()))
    gy = (float(nodes[:, 1].min()), float(nodes[:, 1].max()))
    gz = (float(nodes[:, 2].min()), float(nodes[:, 2].max()))
    if xlim_xy is None and xlim_xz is None:
        xlim_xy = xlim_xz = gx
    if ylim_xy is None:
        ylim_xy = gy
    if ylim_xz is None:
        ylim_xz = gz

    # Exception: allow extending side-view Z (y-axis of XZ) so its height
    # matches the front view (XY) given the same µm-per-pixel scaling.
    span_xy_y = float(ylim_xy[1] - ylim_xy[0]) if ylim_xy is not None else 0.0
    span_xz_y = float(ylim_xz[1] - ylim_xz[0]) if ylim_xz is not None else 0.0
    if span_xy_y > 0 and span_xz_y > 0 and span_xz_y < span_xy_y:
        mid = 0.5 * (ylim_xz[0] + ylim_xz[1])
        half = 0.5 * span_xy_y
        ylim_xz = (mid - half, mid + half)

    subfigs = [
        projection(skel, plane="xy", color_by="ntype", skel_cmap="Set2", xlim=xlim_xy, ylim=ylim_xy),
        projection(skel, plane="xz", color_by="ntype", skel_cmap="Set2", xlim=xlim_xz, ylim=ylim_xz),
    ]
    # Estimate per-panel width from ranges
    widths = []
    for sub in subfigs:
        xr = sub.layout.xaxis.range
        yr = sub.layout.yaxis.range
        if xr is None or yr is None:
            widths.append(1.0)
        else:
            w = float(xr[1] - xr[0])
            h = float(yr[1] - yr[0])
            widths.append(max(w, 1e-6) / max(h, 1e-6))
    # Use equal column widths so pixel-per-micron is identical along X across both panels
    col_widths = [0.5, 0.5]
    total_width_px = int(min(max_width_px, sum([fig_height_px * w for w in widths])))
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        horizontal_spacing=0.02,
        column_widths=col_widths,
    )
    for i, sub in enumerate(subfigs, start=1):
        for tr in sub.data:
            fig.add_trace(tr, row=1, col=i)
        if getattr(sub.layout, "shapes", None):
            for sh in sub.layout.shapes:
                fig.add_shape(sh, row=1, col=i)
        # Keep equal aspect; hide axes (ticks, labels, lines)
        fig.update_xaxes(range=sub.layout.xaxis.range, row=1, col=i,
                         constrain="domain",
                         showgrid=False, zeroline=False, showline=False, ticks="",
                         showticklabels=False)
        fig.update_yaxes(range=sub.layout.yaxis.range, row=1, col=i,
                         scaleanchor=f"x{i}", scaleratio=1,
                         constrain="domain",
                         showgrid=False, zeroline=False, showline=False, ticks="",
                         showticklabels=False)
        # Add small scale bars per panel
        xr = sub.layout.xaxis.range
        yr = sub.layout.yaxis.range
        if xr and yr:
            _add_scale_bars(fig, row=1, col=i, xr=(xr[0], xr[1]), yr=(yr[0], yr[1]))

    # Add dashed ON/OFF SAC lines to the XZ subplot (column 2)
    for yv, color in ((0, "#d62728"), (12, "#1f77b4")):
        fig.add_hline(y=yv, line_dash='dash', line_color=color, row=1, col=2)
    fig.update_layout(
        height=fig_height_px,
        width=None,
        title=None,  # do not show title inside plot to save space
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=5, r=5, t=10, b=5),
        showlegend=False,
    )
    return fig
