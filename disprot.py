"""DisProt API client and per-residue comparison with pLDDT-based disorder.

DisProt: Piovesan et al. (2022) Nucleic Acids Res 50:D471-D477.
pLDDT<50 disorder baseline: Akdel et al. (2022) Nat Struct Mol Biol 29:1056.
"""

from __future__ import annotations
import json, urllib.request
from typing import Optional
import numpy as np
from .config import log


def fetch_disprot(acc: str, timeout: int = 30) -> Optional[dict]:
    """Fetch a single DisProt entry by UniProt accession."""
    acc = (acc or "").strip().upper()
    if not acc:
        return None
    url = (f"https://disprot.org/api/search?release=current"
           f"&show_ambiguous=false&format=json&acc={acc}")
    import time as _time
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.loads(r.read().decode("utf-8", "ignore"))
            for e in j.get("data", []):
                if e.get("acc", "").upper() == acc:
                    return e
            data = j.get("data", [])
            return data[0] if data else None
        except Exception as exc:
            log.warning("fetch_disprot attempt %d/3: %s",
                        attempt + 1, exc)
            if attempt < 2:
                _time.sleep(1.5 * (attempt + 1))
    return None


def disprot_regions(entry: Optional[dict]) -> list[tuple[int, int]]:
    """Consensus disordered regions (structural state type 'D').

    DisProt organises consensus into sub-categories. Pure structural
    disorder is annotated as type 'D' under 'Structural state'
    (NOT under 'full', which may report transitions 'T').
    """
    if not entry:
        return []
    cons = entry.get("disprot_consensus") or {}
    regs = []
    # Primary source: "Structural state" with type "D" (pure disorder)
    for r in cons.get("Structural state", []):
        if r.get("type") == "D":
            try:
                regs.append((int(r["start"]), int(r["end"])))
            except (TypeError, ValueError, KeyError):
                pass
    return regs


def compare_plddt_vs_disprot(
    plddt_vals: list[float],
    disprot_regs: list[tuple[int, int]],
    threshold: float = 50.0,
) -> dict:
    """Per-residue comparison of pLDDT<threshold vs DisProt consensus.

    Only positions covered by both are scored. Returns MCC, F1,
    precision, recall, confusion matrix, and per-residue arrays.
    """
    n = len(plddt_vals)
    if n == 0:
        return {"error": "No pLDDT values"}

    truth = np.zeros(n, dtype=int)
    for s, e in disprot_regs:
        truth[max(0, s - 1):min(n, e)] = 1

    pred = np.array([1 if v < threshold else 0 for v in plddt_vals], dtype=int)

    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "n_residues": n,
        "threshold": threshold,
        "disprot_disordered": int(truth.sum()),
        "aff_disordered": int(pred.sum()),
        "MCC": round(float(mcc), 3),
        "F1": round(float(f1), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "truth_per_residue": truth.tolist(),
        "pred_per_residue": pred.tolist(),
        "n_disprot_regions": len(disprot_regs),
    }