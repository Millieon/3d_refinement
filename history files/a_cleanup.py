"""
Stage A: Mesh Cleanup
Removes artifacts from raw Hunyuan3D output:
  - Disconnected fragments
  - Non-manifold geometry
  - Uneven polygon density
No open3d dependency — uses trimesh only.
"""

import logging
from pathlib import Path

import trimesh
import trimesh.smoothing

log = logging.getLogger(__name__)

TARGET_FACE_COUNT = 15_000  # good balance for a cat model


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        log.info(f"  Mesh already at {len(mesh.faces)} faces, no decimation needed")
        return mesh
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        log.info(f"  Decimated {len(mesh.faces)} → {len(decimated.faces)} faces")
        return decimated
    except Exception as e:
        log.warning(f"  Decimation failed ({e}), keeping original topology")
        return mesh


def cleanup_mesh(input_path: str, output_path: Path) -> trimesh.Trimesh:
    log.info(f"  Loading mesh: {input_path}")
    loaded = trimesh.load(input_path)

    # .glb files often load as a Scene (multiple meshes + embedded textures)
    if isinstance(loaded, trimesh.Scene):
        log.info(f"  .glb loaded as Scene with {len(loaded.geometry)} geometries — merging")
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    # ── 1. Keep only the largest connected component ──────
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        log.info(f"  Found {len(components)} components, keeping largest")
    mesh = max(components, key=lambda m: len(m.faces))

    # ── 2. Repair ──────────────────────────────────────────
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    # ── 3. Decimate to even polygon density ───────────────
    mesh = _decimate(mesh, TARGET_FACE_COUNT)

    # ── 4. Laplacian smoothing ─────────────────────────────
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=5)
    mesh.fix_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    log.info(f"  Saved clean mesh: {output_path}")
    return mesh
