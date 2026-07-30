"""Structured multi-analysis report in JSON and TSV.

FAIR data principles (Wilkinson et al. 2016, Scientific Data 3:160018).
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def build_report(
    sequence_name: str,
    sequence: str,
    model_metrics: list[dict],
    plddt_profile: Optional[list[float]] = None,
    disorder: Optional[dict] = None,
    domains: Optional[dict] = None,
    interface: Optional[dict] = None,
    conservation: Optional[dict] = None,
    ensemble: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    report: dict[str, Any] = {
        "_schema": "AlphaFoldFusion_Report_v2.3",
        "_schema_description": "FAIR-compliant (Wilkinson et al. 2016, Sci Data 3:160018)",
        "_timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence_name": sequence_name,
        "sequence_length": len(sequence.replace(":", "")),
        "is_multimer": ":" in sequence,
        "n_chains": sequence.count(":") + 1,
        "models": model_metrics,
    }
    for key, val in [
        ("plddt_profile", plddt_profile), ("disorder", disorder),
        ("domain_decomposition", domains), ("interface", interface),
        ("conservation", conservation), ("ensemble_variance", ensemble),
    ]:
        if val is not None:
            report[key] = val
    if metadata:
        report["metadata"] = metadata
    return report


def save_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def to_tsv(report: dict) -> str:
    header = ["residue", "pLDDT", "disordered",
              "conservation", "entropy_bits", "gap_fraction", "RMSF_A"]
    lines = ["\t".join(header)]

    plddt = report.get("plddt_profile") or []
    dis = (report.get("disorder") or {}).get("per_residue") or []
    cons = (report.get("conservation") or {}).get("conservation") or []
    ent = (report.get("conservation") or {}).get("entropy") or []
    gf = (report.get("conservation") or {}).get("gap_fraction") or []
    rmsf = (report.get("ensemble_variance") or {}).get("rmsf") or []

    n = max(len(plddt), len(dis), len(cons), len(ent), len(gf), len(rmsf), 1)
    for i in range(n):
        lines.append("\t".join([
            str(i + 1),
            f"{plddt[i]:.1f}" if i < len(plddt) else "",
            str(dis[i]) if i < len(dis) else "",
            f"{cons[i]:.4f}" if i < len(cons) else "",
            f"{ent[i]:.4f}" if i < len(ent) else "",
            f"{gf[i]:.4f}" if i < len(gf) else "",
            f"{rmsf[i]:.3f}" if i < len(rmsf) else "",
        ]))
    return "\n".join(lines)