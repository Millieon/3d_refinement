"""
Utility: Canonical View Renderer
Renders a trimesh from 6 standard angles using pyrender.
Falls back to a simple projection if pyrender is not available.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

log = logging.getLogger(__name__)

# (azimuth_deg, elevation_deg) pairs
CANONICAL_VIEWS = {
    "front":         (0,    0),
    "back":          (180,  0),
    "left":          (90,   0),
    "right":         (270,  0),
    "top":           (0,   90),
    "three_quarter": (45,  30),
}

IMG_SIZE = 512


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    Ry = np.array([
        [np.cos(az),  0, np.sin(az)],
        [0,           1, 0         ],
        [-np.sin(az), 0, np.cos(az)],
    ])
    Rx = np.array([
        [1, 0,        0        ],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el),  np.cos(el)],
    ])
    return Rx @ Ry


def _render_with_pyrender(mesh: trimesh.Trimesh, az: float, el: float) -> np.ndarray:
    import pyrender
    import pyrender as pr

    scene = pr.Scene(ambient_light=[0.4, 0.4, 0.4])

    # Centre mesh
    centroid = mesh.centroid
    m = mesh.copy()
    m.vertices -= centroid

    # Scale to unit sphere
    scale = 1.0 / (np.max(np.abs(m.vertices)) + 1e-6)
    m.vertices *= scale

    render_mesh = pr.Mesh.from_trimesh(m, smooth=True)
    scene.add(render_mesh)

    # Camera at distance 2.5 along rotated axis
    R = _rotation_matrix(az, el)
    cam_pos = R @ np.array([0, 0, 2.5])
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = cam_pos

    camera = pr.PerspectiveCamera(yfov=np.radians(45.0))
    scene.add(camera, pose=cam_pose)

    light = pr.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    renderer = pr.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    color, _ = renderer.render(scene)
    renderer.delete()
    return color


def _render_fallback_silhouette(mesh: trimesh.Trimesh, az: float, el: float) -> np.ndarray:
    """Simple orthographic projection silhouette — used when pyrender unavailable."""
    import cv2

    verts = mesh.vertices - mesh.centroid
    scale = 1.0 / (np.max(np.abs(verts)) + 1e-6)
    verts = verts * scale

    R = _rotation_matrix(az, el)
    projected = (R @ verts.T).T

    # Use X, Y as 2D coords
    xs = ((projected[:, 0] + 1) * 0.5 * IMG_SIZE).astype(int)
    ys = ((1 - (projected[:, 1] + 1) * 0.5) * IMG_SIZE).astype(int)

    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    for x, y in zip(xs, ys):
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            img[y, x] = [80, 80, 80]

    return img


def render_canonical_views(mesh: trimesh.Trimesh, output_dir: Path) -> Dict[str, Path]:
    """
    Renders the mesh from all canonical views.
    Returns a dict {view_name: image_path}.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyrender
        render_fn = _render_with_pyrender
        log.info("  Using pyrender for rendering")
    except ImportError:
        render_fn = _render_fallback_silhouette
        log.warning("  pyrender not found — using fallback silhouette renderer")

    paths = {}
    for view_name, (az, el) in CANONICAL_VIEWS.items():
        img = render_fn(mesh, az, el)
        out_path = output_dir / f"{view_name}.png"
        cv2.imwrite(str(out_path), img[:, :, ::-1] if img.ndim == 3 else img)
        paths[view_name] = out_path
        log.info(f"  Rendered: {view_name} → {out_path}")

    return paths
