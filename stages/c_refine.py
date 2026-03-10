"""
Stage C: Texture-Driven Geometry Refinement
  C1 — Displacement mapping (push vertices along normals using a depth-derived map)
  C2 — PBR map baking (albedo, normal, roughness)
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import trimesh

log = logging.getLogger(__name__)


# ── C1: Displacement ─────────────────────────────────────────────────────────

def _generate_displacement_map(reference_photo_path: Path, size: int = 1024) -> np.ndarray:
    """
    Derive a displacement map from a reference photo.
    Uses luminance as a proxy for depth (lighter = raised).
    In a full pipeline, replace this with ControlNet Depth output.
    """
    img = cv2.imread(str(reference_photo_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read reference photo: {reference_photo_path}")

    # Convert to grayscale luminance map
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalise and resize to desired texture resolution
    disp = cv2.resize(gray, (size, size)).astype(np.float32) / 255.0

    # Light Gaussian blur to reduce noise
    disp = cv2.GaussianBlur(disp, (5, 5), 0)
    return disp


def apply_displacement(
    mesh: trimesh.Trimesh,
    reference_photo_path: Path,
    output_path: Path,
    strength: float = 0.02,
    target_view: Optional[str] = None,
) -> trimesh.Trimesh:
    """
    Pushes vertices along their normals based on a displacement map.
    `strength` controls how far vertices are displaced (in mesh units).
    `target_view` is reserved for future per-region masking.
    """
    log.info(f"  Generating displacement map from {reference_photo_path.name}...")
    disp_map = _generate_displacement_map(reference_photo_path)

    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        log.warning("  Mesh has no UV coords — skipping displacement (UV unwrap first)")
        return mesh

    h, w = disp_map.shape
    new_vertices = mesh.vertices.copy()
    normals = mesh.vertex_normals

    for i, (uv, normal) in enumerate(zip(mesh.visual.uv, normals)):
        px = int(uv[0] * w) % w
        py = int((1.0 - uv[1]) * h) % h
        displacement_value = disp_map[py, px]
        new_vertices[i] += normal * displacement_value * strength

    displaced = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces,
        vertex_normals=normals,
    )
    if hasattr(mesh, "visual"):
        displaced.visual = mesh.visual

    output_path.parent.mkdir(parents=True, exist_ok=True)
    displaced.export(str(output_path))
    log.info(f"  Displacement applied → {output_path}")
    return displaced


# ── C2: PBR Baking ───────────────────────────────────────────────────────────

def _bake_albedo(reference_photos: List[Path], size: int = 2048) -> np.ndarray:
    """Average multiple reference photos into a single albedo map."""
    accumulated = np.zeros((size, size, 3), dtype=np.float32)
    for photo_path in reference_photos:
        img = cv2.imread(str(photo_path))
        if img is None:
            continue
        resized = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
        accumulated += resized
    return np.clip(accumulated / max(len(reference_photos), 1), 0, 1)


def _bake_roughness(albedo: np.ndarray) -> np.ndarray:
    """
    Derive roughness from albedo: fur = high roughness, dark/wet areas = lower.
    In production, replace with a dedicated material segmentation model.
    """
    gray = cv2.cvtColor((albedo * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Invert: lighter areas (fur) → high roughness, darker (nose, eyes) → low
    roughness = gray.astype(np.float32) / 255.0
    roughness = 0.4 + 0.4 * roughness  # scale to [0.4, 0.8] range
    return roughness


def _bake_normal_map(albedo: np.ndarray) -> np.ndarray:
    """
    Approximate normal map from luminance gradients.
    In production, use ControlNet Normal or a dedicated normal map baker.
    """
    gray = cv2.cvtColor((albedo * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, kscale=3) / 255.0
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, kscale=3) / 255.0
    grad_z = np.ones_like(grad_x)

    normal = np.stack([grad_x, grad_y, grad_z], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-6)

    # Convert from [-1,1] to [0,1] for storage
    normal_map = (normal * 0.5 + 0.5).clip(0, 1)
    return normal_map


def bake_pbr(
    mesh: trimesh.Trimesh,
    reference_photos: List[Path],
    output_dir: Path,
    texture_size: int = 2048,
) -> dict:
    """
    Bakes albedo, roughness, and normal maps from reference photos.
    Returns paths to each baked map.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("  Baking albedo map...")
    albedo = _bake_albedo(reference_photos, texture_size)
    albedo_path = output_dir / "albedo.png"
    cv2.imwrite(str(albedo_path), (albedo * 255).astype(np.uint8))

    log.info("  Baking roughness map...")
    roughness = _bake_roughness(albedo)
    roughness_path = output_dir / "roughness.png"
    cv2.imwrite(str(roughness_path), (roughness * 255).astype(np.uint8))

    log.info("  Baking normal map...")
    normal = _bake_normal_map(albedo)
    normal_path = output_dir / "normal.png"
    cv2.imwrite(str(normal_path), (normal * 255).astype(np.uint8))

    paths = {
        "albedo": albedo_path,
        "roughness": roughness_path,
        "normal": normal_path,
    }
    log.info(f"  PBR maps saved to {output_dir}")
    return paths
