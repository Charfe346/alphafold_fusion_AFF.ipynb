"""Intrinsic disorder region (IDR) identification from pLDDT scores.

The pLDDT < 50 threshold corresponds to the "very low confidence"
band defined for AlphaFold2 (Jumper et al. 2021, Nature 596:583-589;
Tunyasuvunakool et al. 2021, Nature 596:590-596). The principle that
low pLDDT predicts intrinsic disorder was established by Akdel et al.
(2022) Nat Struct Mol Biol 29:1056-1067, who benchmarked AlphaFold2
pLDDT-based disorder prediction against IUPred2 using AUC-ROC.

In this tool, pLDDT < 50 is used as a disorder threshold and
validated independently against the DisProt database (Piovesan
et al. 2022, Nucleic Acids Res 50:D471-D477) using AUC-ROC and MCC.

The minimum IDR length of 5 residues is a heuristic short-region
filter (not defined by any specific reference).

GFF3 export uses Sequence Ontology term SO:0100002
(intrinsically_unstructured_polypeptide_region).
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from .plddt import plddt_by_chain


def identify_idrs(
    plddt_vals: list[float],
    threshold: float = 50.0,
    min_length: int = 5,
) -> dict:
    """Segment disordered regions from per-residue pLDDT.

    Parameters
    ----------
    plddt_vals : per-residue pLDDT confidence scores
    threshold  : pLDDT < threshold classified as disordered.
                 Default 50.0 = AlphaFold2 "very low confidence" band
                 (Jumper et al. 2021, Nature 596:583; Tunyasuvunakool
                 et al. 2021, Nature 596:590). Low pLDDT predicts
                 disorder (Akdel et al. 2022, NSMB 29:1056).
    min_length : minimum consecutive residues. Default 5 is a
                 heuristic short-region filter (no specific reference).
    """
    n = len(plddt_vals)
    if n == 0:
        return {"regions": [], "fraction_disordered": 0.0,
                "per_residue": [], "n_residues": 0}

    binary = [1 if v < threshold else 0 for v in plddt_vals]

    regions: list[dict] = []
    start = None
    for i in range(n + 1):
        if i < n and binary[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_length:
                    seg = plddt_vals[start:i]
                    regions.append({
                        "start": start + 1,
                        "end": i,
                        "length": length,
                        "mean_plddt": round(float(np.mean(seg)), 1),
                        "min_plddt": round(float(np.min(seg)), 1),
                    })
                start = None

    return {
        "regions": regions,
        "fraction_disordered": round(sum(binary) / n, 3),
        "per_residue": binary,
        "n_residues": n,
        "n_disordered_residues": sum(binary),
        "n_idr_regions": len(regions),
        "threshold_used": threshold,
        "min_length_used": min_length,
        "threshold_reference": "AF2 very low confidence band: Jumper et al. (2021) Nature 596:583; Tunyasuvunakool et al. (2021) Nature 596:590. Disorder principle: Akdel et al. (2022) NSMB 29:1056",
        "min_length_reference": "heuristic short-region filter (>=5 aa)",
    }


def disorder_from_structure(
    txt: str, fmt: str,
    chain: Optional[str] = None,
    threshold: float = 50.0,
    min_length: int = 5,
) -> dict[str, dict]:
    """IDR identification on all chains of a structure file."""
    pc = plddt_by_chain(txt, fmt, chain)
    return {
        ch: identify_idrs(vals, threshold, min_length)
        for ch, vals in pc.items()
    }


def to_gff3(
    disorder_result: dict,
    seqid: str = "query",
    chain: str = "A",
    source: str = "AlphaFoldFusion",
) -> str:
    """Export IDRs as GFF3 (Sequence Ontology SO:0100002)."""
    lines = ["##gff-version 3"]
    for i, r in enumerate(disorder_result.get("regions", []), 1):
        attrs = (f"ID=IDR_{chain}_{i};chain={chain};"
                 f"length={r['length']};"
                 f"mean_plddt={r['mean_plddt']};"
                 f"min_plddt={r['min_plddt']}")
        lines.append(
            f"{seqid}\t{source}\t"
            f"intrinsically_unstructured_polypeptide_region\t"
            f"{r['start']}\t{r['end']}\t"
            f"{r['mean_plddt']}\t.\t.\t{attrs}"
        )
    return "\n".join(lines)