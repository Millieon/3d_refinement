"""
2D View Auditor for Nanobanana-generated multi-view images
==========================================================
Checks 8 generated views for identity + geometric consistency
before feeding to Hunyuan3D.

Usage:
    python audit_views.py --views ./views/ --output ./audit/

Views directory should contain images named:
    front.png, back.png, left.png, right.png,
    top.png, bottom.png, left_45.png, right_45.png

    (or pass --auto to auto-assign files alphabetically)

Requirements:
    pip install anthropic opencv-python numpy
"""

import argparse
import base64
import json
import os
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import anthropic

# ── Expected view names and their geometric relationships ────────────────────
VIEW_NAMES = ["front", "back", "left", "right", "top", "bottom", "left_45", "right_45"]

MIRROR_PAIRS = [
    ("left", "right"),
    ("left_45", "right_45"),
]

GEOMETRIC_PAIRS = [
    ("front", "left",    "The ear height and body width visible in front should match the body depth in left side"),
    ("front", "right",   "Same as front/left but mirrored"),
    ("front", "top",     "The head width in front should match top view"),
    ("front", "left_45", "The 45-degree view should be an interpolation between front and left"),
    ("back",  "left",    "The rear body shape in back should match left side rear"),
]


def encode_image(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), media_type


def load_views(views_dir: Path, auto: bool) -> Dict[str, Optional[Path]]:
    views: Dict[str, Optional[Path]] = {}
    if auto:
        files = sorted([f for f in views_dir.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        for i, view_name in enumerate(VIEW_NAMES):
            views[view_name] = files[i] if i < len(files) else None
        print("Auto-assigned views:")
        for name, path in views.items():
            print(f"  {name:10s} -> {path.name if path else 'MISSING'}")
    else:
        for name in VIEW_NAMES:
            for ext in [".png", ".jpg", ".jpeg"]:
                p = views_dir / f"{name}{ext}"
                if p.exists():
                    views[name] = p
                    break
            else:
                views[name] = None
    return views


# ── Audit 1: Identity consistency ────────────────────────────────────────────

IDENTITY_PROMPT = """
You are auditing AI-generated multi-view images of an animal for 3D model generation.
These {n} images should all show the SAME animal from different angles.

Check for identity consistency across all views:
1. Fur/coat colour and pattern — do markings match across views?
2. Body proportions — does the animal look the same size/shape?
3. Distinctive features — ears, tail, face shape consistent?
4. Background — all should be plain white

Return ONLY valid JSON:
{{
  "identity_consistent": true,
  "overall_identity_score": 8,
  "issues": [
    {{
      "views_affected": ["front", "left"],
      "issue": "description of inconsistency",
      "severity": 3
    }}
  ],
  "summary": "one sentence overall assessment"
}}
"""

def audit_identity(client: anthropic.Anthropic, views: Dict[str, Optional[Path]], output_path: Path) -> dict:
    print("\nAudit 1: Identity Consistency")
    available = {k: v for k, v in views.items() if v is not None}

    content: List[dict] = []
    for name, path in available.items():
        b64, mt = encode_image(path)
        content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
        content.append({"type": "text", "text": f"[{name} view]"})
    content.append({"type": "text", "text": IDENTITY_PROMPT.format(n=len(available))})

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": content}]
    )

    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    print(f"  Identity score: {result['overall_identity_score']}/10")
    print(f"  Consistent:     {result['identity_consistent']}")
    print(f"  Summary:        {result['summary']}")
    if result.get("issues"):
        print(f"  Issues ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"    [{issue['severity']}/5] {issue['views_affected']}: {issue['issue']}")

    return result


# ── Audit 2: Geometric consistency per pair ───────────────────────────────────

GEOMETRIC_PROMPT = """
You are checking if two animal views are geometrically consistent for 3D reconstruction.

IMAGE 1: {view_a} view
IMAGE 2: {view_b} view

Context: {context}

Check:
1. Do shared visible features (ears, head, body outline, tail) appear at consistent positions/proportions?
2. Are there any features visible in one view that contradict the other?
3. Is the perspective/scale consistent?

Return ONLY valid JSON:
{{
  "consistent": true,
  "consistency_score": 8,
  "issues": [
    {{
      "feature": "e.g. ear position",
      "issue": "description",
      "severity": 3
    }}
  ]
}}
"""

def audit_geometric_pairs(client: anthropic.Anthropic, views: Dict[str, Optional[Path]], output_path: Path) -> dict:
    print("\nAudit 2: Geometric Consistency (view pairs)")
    pair_results = {}

    for view_a, view_b, context in GEOMETRIC_PAIRS:
        if views.get(view_a) is None or views.get(view_b) is None:
            print(f"  Skipping {view_a}/{view_b} (missing)")
            continue

        b64_a, mt_a = encode_image(views[view_a])
        b64_b, mt_b = encode_image(views[view_b])

        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mt_a, "data": b64_a}},
                {"type": "image", "source": {"type": "base64", "media_type": mt_b, "data": b64_b}},
                {"type": "text",  "text": GEOMETRIC_PROMPT.format(view_a=view_a, view_b=view_b, context=context)}
            ]}]
        )

        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        pair_results[f"{view_a}/{view_b}"] = result

        status = "OK  " if result["consistent"] else "FAIL"
        print(f"  {view_a:8s} / {view_b:8s}  [{status}] score={result['consistency_score']}/10", end="")
        if result.get("issues"):
            top = result["issues"][0]
            print(f"  — {top['feature']}: {top['issue']}", end="")
        print()

    return pair_results


# ── Audit 3: Mirror symmetry (pixel-level) ────────────────────────────────────

def audit_mirror_symmetry(views: Dict[str, Optional[Path]], output_path: Path) -> dict:
    print("\nAudit 3: Mirror Symmetry (left vs right)")
    results = {}

    for view_a, view_b in MIRROR_PAIRS:
        if views.get(view_a) is None or views.get(view_b) is None:
            print(f"  Skipping {view_a}/{view_b} (missing)")
            continue

        img_a = cv2.imread(str(views[view_a]), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(views[view_b]), cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            print(f"  Could not read images for {view_a}/{view_b}")
            continue

        size = (256, 256)
        img_a = cv2.resize(img_a, size)
        img_b = cv2.resize(img_b, size)
        img_b_flipped = cv2.flip(img_b, 1)

        _, sil_a = cv2.threshold(img_a, 240, 255, cv2.THRESH_BINARY_INV)
        _, sil_b = cv2.threshold(img_b_flipped, 240, 255, cv2.THRESH_BINARY_INV)
        sil_diff = cv2.absdiff(sil_a, sil_b)
        sil_score = float(100 - (sil_diff.mean() / 255 * 100))

        passed = sil_score > 75
        results[f"{view_a}/{view_b}"] = {
            "silhouette_similarity": round(sil_score, 1),
            "pass": passed,
        }

        # Save diff image
        cv2.imwrite(str(output_path / f"mirror_diff_{view_a}_{view_b}.png"), sil_diff)

        status = "OK  " if passed else "WARN"
        print(f"  {view_a:8s} / {view_b:8s}  [{status}] silhouette similarity={sil_score:.1f}%")

    return results


# ── Audit 4: Background + edge quality ───────────────────────────────────────

def audit_background_quality(views: Dict[str, Optional[Path]], output_path: Path) -> dict:
    print("\nAudit 4: Background & Edge Quality")
    results = {}

    for name, path in views.items():
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue

        h, w = img.shape[:2]
        corners = [
            img[0:20, 0:20],
            img[0:20, w-20:w],
            img[h-20:h, 0:20],
            img[h-20:h, w-20:w],
        ]
        bg_score = float(np.mean([p.mean() for p in corners]))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.sum(edges > 0)) / (h * w) * 1000

        bg_pass = bg_score > 230
        results[name] = {
            "background_whiteness": round(bg_score, 1),
            "background_clean": bg_pass,
            "edge_pixel_density": round(edge_density, 2),
        }

        status = "OK  " if bg_pass else "WARN"
        print(f"  {name:10s}  [{status}] bg={bg_score:.0f}/255  edges={edge_density:.2f}")

    return results


# ── Final report ──────────────────────────────────────────────────────────────

def save_report(identity: dict, geometric: dict, mirror: dict,
                background: dict, views: Dict[str, Optional[Path]], output_path: Path):
    identity_ok = identity.get("identity_consistent", False)
    geo_scores  = [v["consistency_score"] for v in geometric.values()]
    geo_ok      = all(s >= 6 for s in geo_scores) if geo_scores else True
    mirror_ok   = all(v["pass"] for v in mirror.values()) if mirror else True
    bg_ok       = all(v["background_clean"] for v in background.values())
    overall     = identity_ok and geo_ok and mirror_ok and bg_ok

    missing = [name for name, path in views.items() if path is None]

    report = {
        "overall_pass": overall,
        "ready_for_hunyuan3d": overall and len(missing) == 0,
        "missing_views": missing,
        "audits": {
            "identity":         identity,
            "geometric_pairs":  geometric,
            "mirror_symmetry":  mirror,
            "background_quality": background,
        },
        "action_required": []
    }

    if missing:
        report["action_required"].append(f"Regenerate missing views: {missing}")
    if not identity_ok:
        for iss in sorted(identity.get("issues", []), key=lambda x: -x["severity"])[:3]:
            report["action_required"].append(
                f"Identity issue in {iss['views_affected']}: {iss['issue']}")
    for pair, res in geometric.items():
        if res["consistency_score"] < 6:
            for iss in res.get("issues", [])[:2]:
                report["action_required"].append(
                    f"Geometry mismatch ({pair}) — {iss['feature']}: {iss['issue']}")
    for pair, res in mirror.items():
        if not res["pass"]:
            report["action_required"].append(
                f"Mirror inconsistency {pair}: similarity only {res['silhouette_similarity']}%")

    out = output_path / "audit_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print(f"OVERALL: {'PASS — ready for Hunyuan3D' if report['ready_for_hunyuan3d'] else 'FAIL — regenerate views'}")
    print("=" * 55)
    if report["action_required"]:
        print("Action required:")
        for action in report["action_required"]:
            print(f"  • {action}")
    else:
        print("All checks passed. Feed views to Hunyuan3D.")
    print(f"\nFull report: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(views_dir: str, reference: Optional[str], output_dir: str, auto: bool):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    views = load_views(Path(views_dir), auto)
    available = sum(1 for v in views.values() if v is not None)
    print(f"Found {available}/{len(VIEW_NAMES)} views")

    identity   = audit_identity(client, views, output_path)
    geometric  = audit_geometric_pairs(client, views, output_path)
    mirror     = audit_mirror_symmetry(views, output_path)       # fix: pass output_path
    background = audit_background_quality(views, output_path)

    save_report(identity, geometric, mirror, background, views, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit nanobanana-generated multi-view images")
    parser.add_argument("--views",     required=True,        help="Directory containing the 8 view images")
    parser.add_argument("--output",    default="./audit",    help="Output directory for report")
    parser.add_argument("--reference", default=None,         help="Optional: real photo of the animal")
    parser.add_argument("--auto",      action="store_true",  help="Auto-assign files alphabetically")
    args = parser.parse_args()
    run(args.views, args.reference, args.output, args.auto)