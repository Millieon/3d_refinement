"""
Stage D: Silhouette QA Loop
Measures Chamfer distance between rendered mesh edge maps and reference photo
edge maps. Flags views that exceed the error threshold for re-refinement.
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

ERROR_THRESHOLD_PX = 5.0


def _edge_map(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.Canny(blurred, 80, 160)


def _chamfer_distance(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """
    One-directional Chamfer distance: mean nearest-neighbour distance
    from edge_b points → edge_a points.
    Returns 0.0 if either map is empty.
    """
    pts_a = np.argwhere(edge_a > 0).astype(np.float32)
    pts_b = np.argwhere(edge_b > 0).astype(np.float32)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0

    tree = KDTree(pts_a)
    dists, _ = tree.query(pts_b)
    return float(dists.mean())


def silhouette_qa(
    renders: Dict[str, Path],       # {view_name: render_path}
    reference_photos: List[Path],
) -> dict:
    """
    Computes silhouette error for each view.
    Returns per-view errors, mean error, worst view, and pass/fail.
    """
    per_view = {}

    for i, (view_name, render_path) in enumerate(renders.items()):
        ref_path = reference_photos[i % len(reference_photos)]

        try:
            render_edges = _edge_map(render_path)
            ref_edges = _edge_map(ref_path)

            # Resize to same shape if needed
            if render_edges.shape != ref_edges.shape:
                ref_edges = cv2.resize(ref_edges, (render_edges.shape[1], render_edges.shape[0]))

            error = _chamfer_distance(render_edges, ref_edges)
            per_view[view_name] = error
        except Exception as e:
            log.warning(f"  QA failed for view '{view_name}': {e}")
            per_view[view_name] = float("inf")

    valid_errors = {k: v for k, v in per_view.items() if v != float("inf")}
    mean_error = float(np.mean(list(valid_errors.values()))) if valid_errors else float("inf")
    worst_view = max(valid_errors, key=valid_errors.get) if valid_errors else None

    result = {
        "per_view": per_view,
        "mean_error": mean_error,
        "worst_view": worst_view,
        "passed": mean_error < ERROR_THRESHOLD_PX,
    }

    log.info(f"  QA result: mean={mean_error:.2f}px, passed={result['passed']}")
    return result
