import random
import urllib.parse
from nglui import statebuilder
import json
from typing import Optional

from codex.data.brain_regions import REGIONS, COLORS
from codex.data.versions import (
    DEFAULT_DATA_SNAPSHOT_VERSION,
    DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
)

from codex import logger

NGL_FLAT_BASE_URL = "https://ngl.cave-explorer.org"
NGL_EW2_BASE_URL = "https://spelunker.cave-explorer.org"


def url_for_root_ids(
    root_ids, version, point_to="ngl", position=None, show_side_panel=None
):
    if version not in DATA_SNAPSHOT_VERSION_DESCRIPTIONS:
        logger.error(
            f"Invalid version '{version}' passed to 'url_for_root_ids'. Falling back to default."
        )
        version = DEFAULT_DATA_SNAPSHOT_VERSION
    if point_to in ["flywire_prod", "flywire_public"]:
        img_layer = statebuilder.ImageLayerConfig(
            name="EM",
            source="precomputed://gs://microns-seunglab/drosophila_v0/alignment/vector_fixer30_faster_v01/v4/image_stitch_v02",
        )

        seg_layer_name = (
            "Production segmentation"
            if point_to == "flywire_prod"
            else "Public segmentation"
        )
        seg_layer_source = (
            "graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31"
            if point_to == "flywire_prod"
            else "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public"
        )

        seg_layer = statebuilder.SegmentationLayerConfig(
            name=seg_layer_name,
            source=seg_layer_source,
            fixed_ids=root_ids,
        )

        view_options = {
            "layout": "xy-3d",
            "show_slices": False,
            "zoom_3d": 2500,
            "zoom_image": 50,
        }

        if position is not None:
            view_options["position"] = position

        sb = statebuilder.StateBuilder(
            layers=[img_layer, seg_layer],
            resolution=[4, 4, 40],
            view_kws=view_options,
        )

        config = sb.render_state(return_as="dict")
        config["selectedLayer"] = {
            "layer": seg_layer_name,
            "visible": True,
        }
        config["jsonStateServer"] = "https://globalv1.flywire-daf.com/nglstate/post"

        return f"https://ngl.flywire.ai/#!{urllib.parse.quote(json.dumps(config))}"
    else:
        return url_for_cells(
            segment_ids=root_ids, data_version=version, show_side_panel=show_side_panel
        )


def url_for_random_sample(root_ids, version, sample_size=50):
    # make the random subset selections deterministic across executions
    random.seed(420)
    if len(root_ids) > sample_size:
        # make a sorted sample to preserve original order
        root_ids = [
            root_ids[i]
            for i in sorted(random.sample(range(len(root_ids)), sample_size))
        ]
    return url_for_root_ids(root_ids, version=version)


def url_for_cells(segment_ids, data_version, show_side_panel=None):
    if show_side_panel is None:
        show_side_panel = len(segment_ids) > 1
    else:
        show_side_panel = bool(show_side_panel)

    if data_version not in DATA_SNAPSHOT_VERSION_DESCRIPTIONS:
        logger.error(
            f"Invalid version '{data_version}' passed to 'url_for_cells'. Falling back to default."
        )
        data_version = DEFAULT_DATA_SNAPSHOT_VERSION

    config = {
        "dimensions": {"x": [1.6e-8, "m"], "y": [1.6e-8, "m"], "z": [4e-8, "m"]},
        "projectionScale": 30000,
        "layers": [
            {
                "type": "image",
                "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
                "tab": "source",
                "name": "EM",
            },
            {
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "type": "segmentation",
                "objectAlpha": 0.05,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "brain_mesh_v3",
            },
            {
                "type": "segmentation",
                "source": f"precomputed://gs://flywire_v141_m{data_version}",
                "tab": "segments",
                "segments": [
                    str(sid) for sid in segment_ids
                ],  # BEWARE: JSON can't handle big ints
                "name": f"flywire_v141_m{data_version}",
            },
        ],
        "showSlices": False,
        "perspectiveViewBackgroundColor": "#ffffff",
        "showDefaultAnnotations": False,
        "selectedLayer": {
            "visible": show_side_panel,
            "layer": f"flywire_v141_m{data_version}",
        },
        "layout": "3d",
    }

    return f"{NGL_FLAT_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"


def url_for_neuropils(segment_ids=None):
    if segment_ids:
        # exclude "dummy" neuropils, e.g. unassigned, which by convention have negative ids
        segment_ids = [s for s in segment_ids if s >= 0]
    config = {
        "layers": [
            {
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "type": "segmentation",
                "objectAlpha": 0.1,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "brain_mesh_v3",
            },
            {
                "type": "segmentation",
                "mesh": "precomputed://gs://flywire_neuropil_meshes/neuropils/neuropil_mesh_v141_v3",
                "objectAlpha": 1.0,  # workaround for broken transparency on iOS: https://github.com/google/neuroglancer/issues/471
                "tab": "source",
                "segments": segment_ids,
                "segmentColors": {
                    # exclude "dummy" neuropil colors, e.g. unassigned, which by convention have negative ids
                    seg_id: COLORS[key]
                    for key, (seg_id, _) in REGIONS.items()
                    if seg_id >= 0
                },
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "neuropil-regions-surface",
            },
        ],
        "navigation": {
            "pose": {
                "position": {
                    "voxelSize": [4, 4, 40],
                    "voxelCoordinates": [132000, 55390, 512],
                }
            },
            "zoomFactor": 40.875984234132744,
        },
        "showAxisLines": False,
        "perspectiveViewBackgroundColor": "#ffffff",
        "perspectiveZoom": 4000,
        "showSlices": False,
        "gpuMemoryLimit": 2000000000,
        "showDefaultAnnotations": False,
        "selectedLayer": {"layer": "neuropil-regions-surface", "visible": False},
        "layout": "3d",
    }

    return f"{NGL_FLAT_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"


def ew2_config_dict(
    segment_ids,
    selected: Optional[int] = None,
    highlight_color: str = "#29ff29",
    position=None,
    cross_section_scale=None,
    projection_orientation=None,
    projection_scale=None,
    projection_depth=None,
):
    """Return EW2 Neuroglancer state as a Python dict."""
    segment_ids = [int(s) for s in segment_ids]
    if segment_ids:
        selected_str = str(selected if selected is not None else segment_ids[0])
        segment_colors = {
            str(sid): (highlight_color if str(sid) == selected_str else "#ffffff")
            for sid in segment_ids
        }
    else:
        segment_colors = {}

    config = {
        "dimensions": {"x": [1.6e-8, "m"], "y": [1.6e-8, "m"], "z": [4e-8, "m"]},
        "position": [41516.5, 41555.5, 838.5],
        "crossSectionScale": 0.45099710384944064,
        "projectionOrientation": [1, 0, 0, 0],
        "projectionScale": 83820.52470573061,
        "projectionDepth": -75.71853703460232,
        "layers": [
            {
                "type": "image",
                "source": "precomputed://gs://stroeh_sem_mouse_retina/image/v2",
                "tab": "source",
                "name": "img",
            },
            {
                "type": "segmentation",
                "source": {
                    "url": "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina",
                    "state": {
                        "multicut": {"sinks": [], "sources": []},
                        "merge": {"merges": []},
                        "findPath": {},
                    },
                },
                "tab": "segments",
                "annotationColor": "#ffffff",
                "segments": [str(sid) for sid in segment_ids],
                "colorSeed": 225267639,
                "segmentColors": segment_colors,
                "name": "stroeh_mouse_retina",
            },
        ],
        "showSlices": False,
        "gpuMemoryLimit": 4000000000,
        "systemMemoryLimit": 8000000000,
        "selectedLayer": {"layer": "stroeh_mouse_retina"},
        "layout": {"type": "3d", "orthographicProjection": True},
        "jsonStateServer": "https://global.daf-apis.com/nglstate/api/v1/post",
    }

    # Apply view overrides only if provided
    if position is not None:
        config["position"] = position
    if cross_section_scale is not None:
        config["crossSectionScale"] = cross_section_scale
    if projection_orientation is not None:
        config["projectionOrientation"] = projection_orientation
    if projection_scale is not None:
        config["projectionScale"] = projection_scale
    if projection_depth is not None:
        config["projectionDepth"] = projection_depth

    return config


def url_for_ew2_segments(
    segment_ids,
    selected: Optional[int] = None,
    highlight_color: str = "#29ff29",
    position=None,
    cross_section_scale=None,
    projection_orientation=None,
    projection_scale=None,
    projection_depth=None,
):
    """Return full viewer URL that encodes the EW2 state in the hash."""
    config = ew2_config_dict(
        segment_ids=segment_ids,
        selected=selected,
        highlight_color=highlight_color,
        position=position,
        cross_section_scale=cross_section_scale,
        projection_orientation=projection_orientation,
        projection_scale=projection_scale,
        projection_depth=projection_depth,
    )
    return f"{NGL_EW2_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"


def shorten_ew2_state(config: dict) -> Optional[str]:
    """
    POST the provided config to its jsonStateServer and return a shortened share URL
    compatible with the configured viewer.
    Returns None on failure.
    """
    state_server = config.get("jsonStateServer")
    if not state_server:
        return None
    try:
        import requests
        r = requests.post(state_server, json=config, timeout=10)
        # Some servers return JSON, others plain text
        try:
            data = r.json()
        except Exception:
            data = None
        if data:
            # Common patterns: {"url": "https://..."} or {"id": "..."}
            if isinstance(data, dict):
                if "url" in data and isinstance(data["url"], str):
                    return data["url"]
                if "id" in data and isinstance(data["id"], str):
                    # Assume the server returns a redirect url base when given an id at same host
                    base = state_server.rsplit("/", 1)[0]
                    return f"{base}/{data['id']}"
        # Fallback to plain text body
        txt = r.text.strip()
        if txt.startswith("http"):
            return txt
    except Exception:
        return None
    return None
