"""
Audit 3D model against nanobanana-generated views
==================================================
Renders the GLB from the same 8 angles as nanobanana output,
then diffs each render against the corresponding 2D view.

Usage:
    python audit_3d_vs_views.py --mesh cat.glb --views ./views/ --output ./audit_3d/

Views directory should contain:
    front.png, back.png, left.png, right.png,
    top.png, bottom.png, left_45.png, right_45.png

Requirements:
    pip install anthropic opencv-python numpy trimesh pyrender pyglet==1.5.27
"""

import argparse
import base64
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh
import anthropic

# Must be set before pyrender import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# ── Camera angles matching nanobanana's 8 views ──────────────────────────────
VIEW_CAMERAS = {
    "front":    (0,    5),
    "back":     (180,  5),
    "left":     (90,   5),
    "right":    (270,  5),
    "top":      (0,   88),
    "bottom":   (0,  -88),
    "left_45":  (45,   5),
    "right_45": (315,  5),
}

IMG_SIZE = 512

AUDIT_PROMPT = """
You are auditing a 3D model render against a 2D reference image for quality control.

IMAGE 1: Render of the 3D model from the {view} view (white background)
IMAGE 2: Nanobanana-generated 2D reference for the {view} view (white background)

These should match as closely as possible. Compare:
1. Silhouette / overall shape
2. Proportions (head size, body length, limb thickness)
3. Key feature positions (ears, eyes, nose, tail)
4. Surface details visible at this angle

Return ONLY valid JSON:
{{
  "view": "{view}",
  "match_score": 7,
  "silhouette_match": 7,
  "proportion_match": 7,
  "discrepancies": [
    {{
      "feature": "e.g. left ear",
      "issue": "too small compared to reference",
      "severity": 3
    }}
  ],
  "summary": "one sentence"
}}
"""


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh(glb_path: str) -> trimesh.Trimesh:
    print(f"Loading {glb_path}...")
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: len(m.faces))
    mesh.fix_normals()
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


# ── Renderer ──────────────────────────────────────────────────────────────────

def rotation_matrix(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    Ry = np.array([[math.cos(az), 0, math.sin(az)],
                   [0, 1, 0],
                   [-math.sin(az), 0, math.cos(az)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(el), -math.sin(el)],
                   [0, math.sin(el),  math.cos(el)]])
    return Rx @ Ry


def render_view_pyrender(mesh: trimesh.Trimesh, az: float, el: float) -> np.ndarray:
    import pyrender as pr
    scene = pr.Scene(ambient_light=[0.6, 0.6, 0.6], bg_color=[1.0, 1.0, 1.0, 1.0])

    m = mesh.copy()
    m.vertices -= m.centroid
    m.vertices /= (np.max(np.abs(m.vertices)) + 1e-6)
    scene.add(pr.Mesh.from_trimesh(m, smooth=True))

    R = rotation_matrix(az, el)
    cam_pos = R @ np.array([0, 0, 2.5])
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = cam_pos

    scene.add(pr.PerspectiveCamera(yfov=math.radians(45.0)), pose=cam_pose)
    scene.add(pr.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    r = pr.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    color, _ = r.render(scene, flags=pr.RenderFlags.RGBA)
    r.delete()

    rgb   = color[:, :, :3].astype(np.float32)
    alpha = color[:, :, 3:4].astype(np.float32) / 255.0
    white = np.ones_like(rgb) * 255
    return (rgb * alpha + white * (1 - alpha)).astype(np.uint8)


def render_view_fallback(mesh: trimesh.Trimesh, az: float, el: float) -> np.ndarray:
    verts = mesh.vertices - mesh.centroid
    verts /= (np.max(np.abs(verts)) + 1e-6)
    R = rotation_matrix(az, el)
    proj = (R @ verts.T).T
    xs = ((proj[:, 0] + 1) * 0.5 * IMG_SIZE).astype(int)
    ys = ((1 - (proj[:, 1] + 1) * 0.5) * IMG_SIZE).astype(int)
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    for x, y in zip(xs, ys):
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            img[y, x] = [120, 120, 120]
    return img


def render_all_views(mesh: trimesh.Trimesh, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyrender  # noqa: F401
        render_fn = render_view_pyrender
        print("  Using pyrender")
    except ImportError:
        render_fn = render_view_fallback
        print("  pyrender not found — using silhouette fallback")
        print("  (pip install pyrender pyglet==1.5.27 for better renders)")

    paths = {}
    for view_name, (az, el) in VIEW_CAMERAS.items():
        try:
            img = render_fn(mesh, az, el)
        except Exception as e:
            print(f"  render failed for {view_name}: {e}, using fallback")
            img = render_view_fallback(mesh, az, el)

        out_path = output_dir / f"{view_name}.png"
        cv2.imwrite(str(out_path), img[:, :, ::-1] if img.ndim == 3 else img)
        paths[view_name] = out_path
        print(f"  Rendered: {view_name}")

    return paths


# ── Pixel-level silhouette comparison ─────────────────────────────────────────

def silhouette_iou(render_path: Path, ref_path: Path) -> float:
    def to_sil(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        _, mask = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        return mask > 0

    sil_r   = to_sil(render_path)
    sil_ref = to_sil(ref_path)
    intersection = np.logical_and(sil_r, sil_ref).sum()
    union        = np.logical_or(sil_r, sil_ref).sum()
    return float(intersection / union) if union > 0 else 0.0


def save_side_by_side(render_path: Path, ref_path: Path, out_path: Path):
    r   = cv2.resize(cv2.imread(str(render_path)), (IMG_SIZE, IMG_SIZE))
    ref = cv2.resize(cv2.imread(str(ref_path)),    (IMG_SIZE, IMG_SIZE))
    cv2.putText(r,   "3D Render", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(ref, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    cv2.imwrite(str(out_path), np.hstack([r, ref]))


# ── Vision audit per view ─────────────────────────────────────────────────────

def encode_image(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    mt = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), mt


def audit_view_pair(client: anthropic.Anthropic,
                    render_path: Path, ref_path: Path, view_name: str) -> dict:
    b64_r,   mt_r   = encode_image(render_path)
    b64_ref, mt_ref = encode_image(ref_path)

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt_r,   "data": b64_r}},
            {"type": "image", "source": {"type": "base64", "media_type": mt_ref, "data": b64_ref}},
            {"type": "text",  "text": AUDIT_PROMPT.format(view=view_name)}
        ]}]
    )

    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mesh_path: str, views_dir: str, output_dir: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY")

    client      = anthropic.Anthropic(api_key=api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    views_path  = Path(views_dir)

    # Load reference views
    ref_views: Dict[str, Path] = {}
    for name in VIEW_CAMERAS:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = views_path / f"{name}{ext}"
            if p.exists():
                ref_views[name] = p
                break
    print(f"Found {len(ref_views)}/8 reference views: {list(ref_views.keys())}")

    # Render mesh
    mesh = load_mesh(mesh_path)
    print("\nRendering 8 views of 3D model...")
    renders = render_all_views(mesh, output_path / "renders")

    # Side-by-side comparisons
    comparisons_dir = output_path / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)
    for name in VIEW_CAMERAS:
        if name in renders and name in ref_views:
            save_side_by_side(renders[name], ref_views[name], comparisons_dir / f"{name}.png")

    # Silhouette IoU
    print("\nComputing silhouette IoU...")
    iou_scores: Dict[str, float] = {}
    for name in VIEW_CAMERAS:
        if name in renders and name in ref_views:
            iou = silhouette_iou(renders[name], ref_views[name])
            iou_scores[name] = round(iou, 3)
            bar  = "█" * int(iou * 20) + "░" * (20 - int(iou * 20))
            flag = " ← needs work" if iou < 0.6 else ""
            print(f"  {name:10s}  IoU={iou:.2f}  [{bar}]{flag}")

    # Vision audit
    print("\nRunning vision audit (Claude)...")
    vision_results: dict = {}
    all_discrepancies = []

    for name in VIEW_CAMERAS:
        if name not in renders or name not in ref_views:
            print(f"  {name:10s}  skipped (missing render or reference)")
            continue
        print(f"  Auditing {name}...", end=" ", flush=True)
        try:
            result = audit_view_pair(client, renders[name], ref_views[name], name)
            vision_results[name] = result
            all_discrepancies.extend([{**d, "view": name} for d in result.get("discrepancies", [])])
            print(f"match={result['match_score']}/10  "
                  f"sil={result['silhouette_match']}/10  "
                  f"prop={result['proportion_match']}/10")
        except Exception as e:
            print(f"failed: {e}")

    # Rank issues by severity × frequency
    region_freq    = Counter(d["feature"] for d in all_discrepancies)
    ranked_issues  = sorted(all_discrepancies,
                            key=lambda x: x["severity"] * region_freq[x["feature"]],
                            reverse=True)
    seen: set = set()
    top_issues = []
    for d in ranked_issues:
        if d["feature"] not in seen:
            top_issues.append(d)
            seen.add(d["feature"])

    # Report
    mean_iou    = float(np.mean(list(iou_scores.values()))) if iou_scores else 0.0
    match_scores = [v["match_score"] for v in vision_results.values()]
    mean_match  = round(float(np.mean(match_scores)), 1) if match_scores else 0.0
    worst_views = sorted(iou_scores.items(), key=lambda x: x[1])[:3]

    report = {
        "summary": {
            "mean_iou":           round(mean_iou, 3),
            "mean_match_score":   mean_match,
            "views_audited":      len(vision_results),
            "total_discrepancies": len(all_discrepancies),
            "ready_for_use":      mean_iou > 0.65 and mean_match >= 6,
        },
        "per_view_iou":    iou_scores,
        "per_view_vision": vision_results,
        "top_issues":      top_issues[:8],
        "worst_views":     [{"view": v, "iou": s} for v, s in worst_views],
    }

    report_path = output_path / "audit_3d_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print(f"Mean IoU:         {mean_iou:.2f}  (target >0.65)")
    print(f"Mean match score: {mean_match}/10  (target >=6)")
    print(f"Ready for use:    {report['summary']['ready_for_use']}")
    print("=" * 55)
    print("Top issues to fix:")
    for d in top_issues[:5]:
        print(f"  [{d['severity']}/5] {d['view']:10s} {d['feature']}: {d['issue']}")
    print(f"\nComparisons: {comparisons_dir}/")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit 3D model against nanobanana views")
    parser.add_argument("--mesh",   required=True,             help="Path to Hunyuan3D .glb")
    parser.add_argument("--views",  required=True,             help="Directory of nanobanana views")
    parser.add_argument("--output", default="./audit_3d",      help="Output directory")
    args = parser.parse_args()
    run(args.mesh, args.views, args.output)