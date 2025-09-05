import numpy as np
from typing import Sequence
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
        xaxis=dict(title=f"{plane[0]}{unit_str}", range=xlim, showgrid=True, zeroline=False, mirror=True, ticks="inside", showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(title=f"{plane[1]}{unit_str}", range=ylim, scaleanchor="x", scaleratio=1, showgrid=True, zeroline=False, mirror=True, ticks="inside", showline=True, linewidth=2, linecolor='black'),
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
    seg_id: str,
    right_panel_width_px: int = 500,
    fig_height_px: int = 250,
    max_width_px: int = 600,
):
    """
    Plot stratification profile using Plotly. Saves to HTML (returns figure).
    """
    # --- Aspect-scaling logic (keep X:Z = 1:1 in left pane)
    xyz = warped_skeleton.nodes  # (N, 3)
    x_span_um = np.ptp(xyz[:, 0])
    z_span_um = np.ptp(xyz[:, 2])
    left_panel_width_px = fig_height_px * (x_span_um / z_span_um)
    total_width_px = int(left_panel_width_px + right_panel_width_px)

    # ---- CAP the maximum width if too big ----
    if total_width_px > max_width_px:
        scale = max_width_px / total_width_px
        left_panel_width_px = int(left_panel_width_px * scale)
        right_panel_width_px = int(right_panel_width_px * scale)
        total_width_px = max_width_px
    # ------------------------------------------

    left_frac = left_panel_width_px / total_width_px
    right_frac = right_panel_width_px / total_width_px

    # Colors: Matplotlib C3 and C0 (red and blue in default cycle)
    colors = ["#d62728", "#1f77b4"]

    # --- Compose Plotly subplot grid
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0,
        column_widths=[left_frac, right_frac]
    )

    # 1. Arbor (XZ projection) in left panel using projection helper
    proj_fig = projection(
        warped_skeleton, plane="xz",
        color_by="ntype", skel_cmap="Set2"
    )
    # Add all traces from projection() output to left panel
    for trace in proj_fig.data:
        fig.add_trace(trace, row=1, col=1)
    # Copy ellipse/circle overlays (if any) from projection
    if getattr(proj_fig.layout, "shapes", None):
        for shape in proj_fig.layout.shapes:
            fig.add_shape(shape, row=1, col=1)
    # Dashed horizontal lines
    for i, yv in enumerate((0, 12)):
        fig.add_hline(y=yv, line_dash='dash', line_color=colors[i], row=1, col=1)

    # ON/OFF SAC annotation at right edge of left panel
    xlim = (np.min(xyz[:, 0]), np.max(xyz[:, 0]))
    fig.add_annotation(
        text='ON SAC', x=xlim[1], y=0, xref="x", yref="y",
        xanchor='right', yanchor='bottom', font=dict(color=colors[0], size=16),
        showarrow=False, row=1, col=1
    )
    fig.add_annotation(
        text='OFF SAC', x=xlim[1], y=12, xref="x", yref="y",
        xanchor='right', yanchor='bottom', font=dict(color=colors[1], size=16),
        showarrow=False, row=1, col=1
    )

    
    # Axis labels and aspect lock
    fig.update_xaxes(
        title_text='X (µm)', row=1, col=1, scaleanchor="y1", scaleratio=1,
        showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black',
        range=proj_fig.layout.xaxis.range
    )
    fig.update_yaxes(
        title_text='Z (µm)', row=1, col=1,
        showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black',
        range=z_profile_extent
    )

    # Title for left panel
    fig.add_annotation(
        text=f"{seg_id}", xref="x domain", yref="paper",
        x=0.5, y=fig.layout.yaxis.range[1]+1, showarrow=False, font=dict(size=14),
        row=1, col=1
    )

    # 2. Right panel: Z-profile (barh and line)
    zp = warped_skeleton.extra["z_profile"]
    # Line (distribution) — on right panel
    fig.add_trace(go.Scatter(
        x=zp["distribution"], y=zp["x"],
        mode="lines", line=dict(color='black', width=2),
        showlegend=False,
    ), row=1, col=2)
    # Horizontal bar (histogram) — on right panel
    fig.add_trace(go.Bar(
        x=zp["histogram"], y=zp["x"], orientation='h',
        marker=dict(color='gray', opacity=0.5),
        width=(zp["x"][1] - zp["x"][0]) if len(zp["x"]) > 1 else 0.8,
        showlegend=False,
    ), row=1, col=2)
    # Dashed horizontal lines (z=0, z=12)
    for i, yv in enumerate((0, 12)):
        fig.add_hline(y=yv, line_dash='dash', line_color=colors[i], row=1, col=2)

    # Axis styling for right panel
    fig.update_xaxes(
        title_text='dendritic length', row=1, col=2, 
        showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'
    )
    fig.update_yaxes(
        title_text='', row=1, col=2, showticklabels=False, showgrid=False,
        showline=True, linewidth=2, linecolor='black', range=z_profile_extent
    )
    # Title for right panel
    fig.add_annotation(
        text='Z-Profile', xref='x2 domain', yref='paper',
        x=0.5, y=fig.layout.yaxis.range[1]+1, showarrow=False, font=dict(size=14),
        row=1, col=2
    )

    # Final figure layout
    fig.update_layout(
        height=fig_height_px, width=total_width_px,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=5, r=5, t=60, b=5),
        showlegend=False,
    )

    return fig
