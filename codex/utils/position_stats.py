import numpy as np
from scipy.spatial import Voronoi, KDTree

# Helper to reconstruct finite Voronoi polygons in 2D
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions by clipping them within a bounding box.
    Source: https://gist.github.com/pv/8036995
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        # compute maximum span along any axis, then double it
        radius = np.ptp(vor.points, axis=0).max() * 2
    # map containing all ridges for a point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # reconstruct regions
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            # compute the missing endpoint at infinity
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            far_point = vor.vertices[v2] + n * radius
            new_vertices.append(far_point.tolist())
            region.append(len(new_vertices) - 1)
        # sort region vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        region = [v for _, v in sorted(zip(angles, region))]
        new_regions.append(region)
    return new_regions, np.asarray(new_vertices)

def compute_vdri(soma_pos):
    """
    Compute the Voronoi Domain Regularity Index (VDRI) for a set of 2D points.
    VDRI = mean(areas) / std(areas), where 'areas' are the areas of finite Voronoi regions.
    
    Parameters:
        soma_pos (np.ndarray): N×2 array of x,y coordinates.
        
    Returns:
        float: VDRI value.
    """
    print(f"compute_vdri: received {soma_pos.shape[0]} points")
    # Build Voronoi diagram
    vor = Voronoi(soma_pos)
    print("compute_vdri: vor.point_region length =", len(vor.point_region))
    # Reconstruct finite polygons and compute areas
    regions, verts = voronoi_finite_polygons_2d(vor)
    areas = []
    for region in regions:
        polygon = verts[region]
        x = polygon[:, 0]
        y = polygon[:, 1]
        # shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(area)
    areas = np.array(areas)
    if areas.size < 2:
        return np.nan
    return areas.mean() / areas.std()

def compute_nnri(soma_pos):
    """
    Compute the Nearest Neighbor Regularity Index (NNRI) for a set of 2D points.
    NNRI = mean(d_nn) / std(d_nn), where d_nn is the distance to each point's nearest neighbor.
    
    Parameters:
        soma_pos (np.ndarray): N×2 array of x,y coordinates.
        
    Returns:
        float: NNRI value.
    """
    # Build KD-tree and query the two closest points (self + nearest neighbor)
    tree = KDTree(soma_pos)
    dists, _ = tree.query(soma_pos, k=2)
    # dists[:, 0] == 0 (distance to self), so take the second column
    nn_dists = dists[:, 1]
    if nn_dists.size < 2:
        return np.nan
    return nn_dists.mean() / nn_dists.std()
