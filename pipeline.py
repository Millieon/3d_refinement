"""
Cat 3D Refinement Pipeline
==========================
Full chain: raw Hunyuan3D mesh → cleaned → audited → refined → QA'd

Usage:
    python pipeline.py --mesh hunyuan_output.obj --photos ./ref_photos/ --output ./output/
"""

import argparse
import json
import logging
from pathlib import Path

from stages.a_cleanup import cleanup_mesh
from stages.b_audit import vision_audit
from stages.c_refine import apply_displacement, bake_pbr
from stages.d_qa import silhouette_qa
from utils.renderer import render_canonical_views
from utils.report import save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(mesh_path: str, photos_dir: str, output_dir: str, max_iterations: int = 3):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    photos_path = Path(photos_dir)
    reference_photos = sorted(photos_path.glob("*.png")) + sorted(photos_path.glob("*.jpg"))
    if not reference_photos:
        raise FileNotFoundError(f"No reference photos found in {photos_dir}")

    log.info(f"Found {len(reference_photos)} reference photos")
    log.info("=" * 50)

    # ── Stage A: Mesh Cleanup ──────────────────────────────
    log.info("Stage A: Mesh Cleanup")
    mesh = cleanup_mesh(mesh_path, output_path / "stage_a_clean.glb")
    log.info(f"  → {len(mesh.faces)} faces after cleanup")

    # ── Stage B: Vision Audit ──────────────────────────────
    log.info("Stage B: Vision Audit")
    renders = render_canonical_views(mesh, output_path / "renders_b")
    audit_results = vision_audit(renders, reference_photos, output_path / "audit.json")
    log.info(f"  → {len(audit_results['discrepancies'])} discrepancies found, overall score: {audit_results['overall_score']}/10")
    for d in audit_results["discrepancies"]:
        log.info(f"     [{d['severity']}/5] {d['region']}: {d['issue']}")

    # ── Stage C: Texture-Driven Refinement ────────────────
    log.info("Stage C: Geometry + Texture Refinement")
    mesh = apply_displacement(mesh, reference_photos[0], output_path / "stage_c_displaced.glb")
    bake_pbr(mesh, reference_photos, output_path / "textures")
    log.info("  → Displacement applied, PBR maps baked")

    # ── Stage D: QA Loop ──────────────────────────────────
    log.info("Stage D: Silhouette QA Loop")
    for iteration in range(max_iterations):
        renders = render_canonical_views(mesh, output_path / f"renders_iter{iteration}")
        qa = silhouette_qa(renders, reference_photos)

        log.info(f"  Iteration {iteration+1}: mean error = {qa['mean_error']:.2f}px")
        for view, err in qa["per_view"].items():
            flag = " ← worst" if view == qa["worst_view"] else ""
            log.info(f"     {view}: {err:.2f}px{flag}")

        if qa["mean_error"] < 5.0:
            log.info("  → Error below threshold, stopping early")
            break
        elif iteration < max_iterations - 1:
            log.info(f"  → Re-applying targeted displacement on '{qa['worst_view']}' view")
            mesh = apply_displacement(
                mesh, reference_photos[0],
                output_path / f"stage_d_iter{iteration+1}.glb",
                strength=0.01,
                target_view=qa["worst_view"]
            )

    # ── Save final mesh + report ───────────────────────────
    final_mesh_path = output_path / "final_mesh.glb"
    mesh.export(str(final_mesh_path))
    save_report(audit_results, qa, output_path / "pipeline_report.json")

    log.info("=" * 50)
    log.info(f"Pipeline complete. Final mesh: {final_mesh_path}")
    log.info(f"Report: {output_path / 'pipeline_report.json'}")
    return final_mesh_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat 3D Refinement Pipeline")
    parser.add_argument("--mesh", required=True, help="Path to Hunyuan3D output .glb file")
    parser.add_argument("--photos", required=True, help="Directory of reference cat photos")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Max QA loop iterations")
    args = parser.parse_args()

    run_pipeline(args.mesh, args.photos, args.output, args.iterations)
