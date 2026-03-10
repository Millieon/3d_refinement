"""
Stage B: Vision Audit
Renders the cleaned mesh from 6 canonical angles, then uses Claude
to diff each render against the real reference photos.
Returns a prioritised list of discrepancies.
"""

import base64
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import List

import anthropic

log = logging.getLogger(__name__)

VIEW_NAMES = ["front", "back", "left", "right", "top", "three_quarter"]

AUDIT_PROMPT = """
You are a 3D model quality reviewer specialising in animal anatomy.

I am showing you two images:
  IMAGE 1: A render of a 3D cat model from the {view} view
  IMAGE 2: A reference photo of the real cat from a similar angle

Compare them carefully. Return ONLY valid JSON, no preamble or markdown:
{{
  "view": "{view}",
  "discrepancies": [
    {{
      "region": "e.g. left ear",
      "issue": "concise description of the problem",
      "severity": <integer 1-5, where 5 is most severe>
    }}
  ],
  "overall_score": <integer 1-10, where 10 is perfect match>
}}

Focus on: anatomy proportions, fur pattern accuracy, silhouette shape, facial feature placement.
Ignore lighting differences.
"""


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _audit_single_view(
    client: anthropic.Anthropic,
    render_path: Path,
    reference_path: Path,
    view_name: str,
) -> dict:
    render_b64 = _encode_image(render_path)
    ref_b64 = _encode_image(reference_path)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": render_b64},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": ref_b64},
                    },
                    {
                        "type": "text",
                        "text": AUDIT_PROMPT.format(view=view_name),
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if model adds them
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def vision_audit(
    renders: dict,          # {view_name: Path}
    reference_photos: List[Path],
    output_path: Path,
) -> dict:
    """
    Audits each rendered view against the best matching reference photo.
    Returns aggregated discrepancy list sorted by severity.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Simple heuristic: pair views with reference photos by index (wrap if fewer refs)
    all_discrepancies = []
    view_scores = {}

    for i, (view_name, render_path) in enumerate(renders.items()):
        ref_photo = reference_photos[i % len(reference_photos)]
        log.info(f"  Auditing {view_name} view against {ref_photo.name}...")

        try:
            result = _audit_single_view(client, render_path, ref_photo, view_name)
            all_discrepancies.extend(result.get("discrepancies", []))
            view_scores[view_name] = result.get("overall_score", 5)
        except Exception as e:
            log.warning(f"  Audit failed for {view_name}: {e}")
            view_scores[view_name] = None

    # Deduplicate and rank by severity + frequency
    region_counts = Counter(d["region"] for d in all_discrepancies)
    deduped = {}
    for d in all_discrepancies:
        key = d["region"]
        if key not in deduped or d["severity"] > deduped[key]["severity"]:
            deduped[key] = {**d, "frequency": region_counts[key]}

    ranked = sorted(deduped.values(), key=lambda x: (x["severity"] * x["frequency"]), reverse=True)
    overall = round(sum(s for s in view_scores.values() if s) / max(len(view_scores), 1), 1)

    result = {
        "discrepancies": ranked,
        "view_scores": view_scores,
        "overall_score": overall,
        "priority_regions": [d["region"] for d in ranked if d["severity"] >= 3],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
