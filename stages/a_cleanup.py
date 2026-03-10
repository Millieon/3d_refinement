"""
Stage A: Mesh Cleanup
Memory-safe version: decimates immediately after load to avoid OOM kills.
"""

import logging
from pathlib import Path

import trimesh
import trimesh.smoothing

log = logging.getLogger(__name__)

# Decimate to this as soon as the mesh loads — prevents OOM
EARLY_DECIMATE_TARGET = 50_000
# Final target after repair and smoothing
TARGET_FACE_COUNT = 15_000


def _decimate(mesh: trimesh.Trimesh, target_faces: int, label: str = "") -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        log.info(f"  {label}Already at {len(mesh.faces)} faces, skipping")
        return mesh
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        log.info(f"  {label}{len(mesh.faces):,} → {len(decimated.faces):,} faces")
        return decimated
    except Exception as e:
        log.warning(f"  {label}Decimation failed ({e}), keeping as-is")
        return mesh


def cleanup_mesh(input_path: str, output_path: Path) -> trimesh.Trimesh:
    log.info(f"  Loading mesh: {input_path}")
    loaded = trimesh.load(input_path)

    # .glb often loads as a Scene
    if isinstance(loaded, trimesh.Scene):
        log.info(f"  Scene with {len(loaded.geometry)} geometries — merging")
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    log.info(f"  Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # ── STEP 1: Decimate immediately to free memory ────────
    # Do this BEFORE split/repair which can 2-3x memory usage
    if len(mesh.faces) > EARLY_DECIMATE_TARGET:
        log.info(f"  Early decimation to {EARLY_DECIMATE_TARGET:,} faces to free memory...")
        mesh = _decimate(mesh, EARLY_DECIMATE_TARGET, label="Early: ")

    # ── STEP 2: Keep largest connected component ───────────
    log.info("  Finding largest component...")
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        log.info(f"  Found {len(components)} components, keeping largest")
    mesh = max(components, key=lambda m: len(m.faces))

    # ── STEP 3: Repair ────────────────────────────────────
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    # ── STEP 4: Final decimation to target ────────────────
    mesh = _decimate(mesh, TARGET_FACE_COUNT, label="Final: ")

    # ── STEP 5: Laplacian smoothing ───────────────────────
    log.info("  Smoothing...")
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=5)
    mesh.fix_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    log.info(f"  Saved: {output_path} ({len(mesh.faces):,} faces)")
    return mesh
