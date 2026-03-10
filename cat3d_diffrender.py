"""
Cat 3D Refinement via Differentiable Rendering
===============================================
Optimises mesh vertices so renders match your reference photos.

Usage:
    python cat3d_diffrender.py --mesh cat.glb --photos ./cat_photos/ --output ./output/

Requirements:
    pip install torch torchvision pytorch3d trimesh fast-simplification opencv-python scipy
"""

import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh
import trimesh.smoothing
from pathlib import Path

# pytorch3d imports
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
    SoftSilhouetteShader, SoftPhongShader, MeshRenderer,
    PointLights,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

# ── Default camera angles assigned to photos in order ────────────────────────
# Edit these to match your actual photo angles.
# azimuth: 0=front, 90=left, 180=back, 270=right
# elevation: negative = camera slightly above cat (most natural)
DEFAULT_ANGLES = [
    (0,   -10),   # photo 1: front
    (90,  -10),   # photo 2: left side
    (180, -10),   # photo 3: back
    (270, -10),   # photo 4: right side
    (45,  -15),   # photo 5: 3/4 front
    (0,    30),   # photo 6: top down
]

# ── Hyperparameters ───────────────────────────────────────────────────────────
ITERATIONS      = 500
LR              = 3e-3
IMG_SIZE        = 256
W_SILHOUETTE    = 1.0   # match silhouette shape
W_SMOOTH        = 0.5   # prevent spiky mesh
W_NORMAL        = 0.1   # surface normal consistency
W_OFFSET        = 0.05  # stay close to original shape (lower = more aggressive deform)
DECIMATE_TARGET = 5000  # face count for optimisation (lower = faster, less detail)


def load_mesh(glb_path: str) -> trimesh.Trimesh:
    print(f"Loading {glb_path}...")
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        print(f"  Scene with {len(loaded.geometry)} geometries, merging")
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    print(f"  Raw: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Decimate first to avoid OOM
    if len(mesh.faces) > 20000:
        print("  Decimating to 20k for cleanup...")
        try:
            mesh = mesh.simplify_quadric_decimation(20000)
        except Exception as e:
            print(f"  Decimation failed: {e}")

    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: len(m.faces))
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    print(f"  Decimating to {DECIMATE_TARGET} for optimisation...")
    try:
        mesh = mesh.simplify_quadric_decimation(DECIMATE_TARGET)
    except Exception as e:
        print(f"  Final decimation failed: {e}")

    trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=3)
    mesh.fix_normals()
    print(f"  Ready: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


def make_camera(azimuth_deg, elevation_deg, dist=2.2, device="cpu"):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = dist * math.cos(el) * math.sin(az)
    y = dist * math.sin(el)
    z = dist * math.cos(el) * math.cos(az)
    R, T = FoVPerspectiveCameras.look_at_view_transform(
        eye=torch.tensor([[x, y, z]], dtype=torch.float32),
        at=torch.zeros(1, 3),
        up=torch.tensor([[0, 1, 0]], dtype=torch.float32),
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=45, device=device)


def load_silhouette(path: Path, size: int, device) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return torch.tensor(mask / 255.0, dtype=torch.float32, device=device)


def save_progress_image(output_dir: Path, iteration: int,
                        silhouette_renderer, mesh_p3d, cameras, ref_silhouettes, device):
    """Save a side-by-side comparison image every N iterations."""
    out = output_dir / "progress"
    out.mkdir(exist_ok=True)
    imgs = []
    with torch.no_grad():
        for cam, ref_sil in zip(cameras[:3], ref_silhouettes[:3]):
            rendered = silhouette_renderer(mesh_p3d, cameras=cam)
            render_np = rendered[0, :, :, 3].cpu().numpy()
            ref_np = ref_sil.cpu().numpy()
            row = np.hstack([render_np, ref_np]) * 255
            imgs.append(row.astype(np.uint8))
    grid = np.vstack(imgs) if imgs else np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(out / f"iter_{iteration:04d}.png"), grid)


def run(mesh_path: str, photos_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  Warning: running on CPU will be slow. GPU strongly recommended.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load reference photos ─────────────────────────────
    photos_path = Path(photos_dir)
    reference_photos = sorted(photos_path.glob("*.jpg")) + sorted(photos_path.glob("*.png"))
    if not reference_photos:
        raise FileNotFoundError(f"No photos found in {photos_dir}")
    print(f"Reference photos: {[p.name for p in reference_photos]}")

    photo_angles = DEFAULT_ANGLES[:len(reference_photos)]
    print("Camera angle assignments:")
    for p, (az, el) in zip(reference_photos, photo_angles):
        print(f"  {p.name} -> azimuth={az}, elevation={el}")
    print("  (Edit DEFAULT_ANGLES at top of script to adjust)")

    # ── Load + clean mesh ─────────────────────────────────
    mesh = load_mesh(mesh_path)

    # Normalise to unit sphere
    centroid = mesh.centroid.copy()
    verts_np = mesh.vertices - centroid
    scale = float(np.max(np.abs(verts_np)))
    verts_np = verts_np / scale
    faces_np = mesh.faces

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces_np, dtype=torch.int64, device=device)
    deform_verts = nn.Parameter(torch.zeros_like(verts_t))

    # ── Set up renderers ──────────────────────────────────
    silhouette_settings = RasterizationSettings(
        image_size=IMG_SIZE,
        blur_radius=math.log(1 / 1e-4 - 1) * 1e-4,
        faces_per_pixel=50,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=silhouette_settings),
        shader=SoftSilhouetteShader(),
    )

    phong_settings = RasterizationSettings(image_size=IMG_SIZE, blur_radius=0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0, 3, 3]])
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=phong_settings),
        shader=SoftPhongShader(device=device, lights=lights),
    )

    # ── Load reference silhouettes + cameras ──────────────
    ref_silhouettes = [load_silhouette(p, IMG_SIZE, device) for p in reference_photos]
    cameras = [make_camera(az, el, device=device) for az, el in photo_angles]

    # ── Optimisation loop ─────────────────────────────────
    optimizer = torch.optim.Adam([deform_verts], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ITERATIONS, eta_min=1e-4)

    print(f"\nOptimising for {ITERATIONS} iterations...")
    print(f"  Progress images saved to {output_path}/progress/")
    print(f"  {'Iter':>6}  {'Loss':>10}  {'Sil':>10}  {'Smooth':>10}")

    for i in range(ITERATIONS):
        optimizer.zero_grad()

        new_verts = verts_t + deform_verts
        mesh_p3d = Meshes(
            verts=[new_verts],
            faces=[faces_t],
            textures=TexturesVertex(verts_features=torch.ones_like(new_verts)[None] * 0.6),
        )

        # Silhouette loss across all photos
        sil_loss = torch.tensor(0.0, device=device)
        for cam, ref_sil in zip(cameras, ref_silhouettes):
            rendered = silhouette_renderer(mesh_p3d, cameras=cam)
            rendered_sil = rendered[0, :, :, 3]
            sil_loss = sil_loss + torch.mean((rendered_sil - ref_sil) ** 2)
        sil_loss = sil_loss / len(cameras)

        smooth_loss = mesh_laplacian_smoothing(mesh_p3d, method="uniform")
        normal_loss = mesh_normal_consistency(mesh_p3d)
        offset_loss = torch.mean(deform_verts ** 2)

        total_loss = (W_SILHOUETTE * sil_loss
                      + W_SMOOTH * smooth_loss
                      + W_NORMAL * normal_loss
                      + W_OFFSET * offset_loss)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 or i == ITERATIONS - 1:
            print(f"  {i:6d}  {total_loss.item():10.4f}  {sil_loss.item():10.4f}  {smooth_loss.item():10.4f}")
            save_progress_image(output_path, i, silhouette_renderer, mesh_p3d, cameras, ref_silhouettes, device)

    # ── Export refined mesh ───────────────────────────────
    print("\nExporting refined mesh...")
    with torch.no_grad():
        refined_verts = (verts_t + deform_verts).cpu().numpy()

    refined_verts = refined_verts * scale + centroid
    refined_mesh = trimesh.Trimesh(vertices=refined_verts, faces=faces_np)
    refined_mesh.fix_normals()
    trimesh.smoothing.filter_laplacian(refined_mesh, lamb=0.3, iterations=2)
    refined_mesh.fix_normals()

    out_path = output_path / "refined_cat.glb"
    refined_mesh.export(str(out_path))
    print(f"Saved: {out_path}")
    print(f"  {len(refined_mesh.vertices):,} vertices, {len(refined_mesh.faces):,} faces")
    print(f"\nProgress images: {output_path}/progress/")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat 3D Refinement via Differentiable Rendering")
    parser.add_argument("--mesh",   required=True, help="Path to Hunyuan3D .glb file")
    parser.add_argument("--photos", required=True, help="Directory of reference cat photos")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()
    run(args.mesh, args.photos, args.output)
