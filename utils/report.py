"""Utility: Save pipeline summary report."""

import json
from pathlib import Path


def save_report(audit_results: dict, qa_results: dict, output_path: Path):
    report = {
        "audit": {
            "overall_score": audit_results.get("overall_score"),
            "view_scores": audit_results.get("view_scores"),
            "top_issues": audit_results.get("discrepancies", [])[:5],
            "priority_regions": audit_results.get("priority_regions", []),
        },
        "qa": {
            "final_mean_error_px": qa_results.get("mean_error"),
            "passed_threshold": qa_results.get("passed"),
            "per_view_errors": qa_results.get("per_view"),
            "worst_view": qa_results.get("worst_view"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
