"""Multi-model ensemble structural variance analysis.

Superposition: Kabsch algorithm for optimal rigid-body alignment
(Kabsch 1976, Acta Cryst A 32:922-923).

AlphaFold/ColabFold generate multiple models per target. This module
superposes those models and reports the per-residue root-mean-square
fluctuation (RMSF) as a descriptor of inter-model structural
variability. Massive sampling of diverse AlphaFold models has been
shown to improve multimer prediction (Wallner 2023, Bioinformatics
39:btad573); here we summarise the variability of the returned
ensemble rather than generating additional samples.

All reported quantities are continuous distributions.
"""

from __future__ import annotations
import os, tempfile
from typing import Optional
import numpy as np
import gemmi
from .structure import is_amino_acid


def _ca_coords(txt: str, fmt: str,
               chain: Optional[str] = None) -> Optional[np.ndarray]:
    """Extract Ca coordinates as (N, 3) array."""
    tmp_path = None
    try:
        suffix = ".cif" if fmt == "cif" else ".pdb"
        tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        tmp.write(txt); tmp.flush(); tmp.close()
        tmp_path = tmp.name
        st_obj = gemmi.read_structure(tmp_path)
        if len(st_obj) == 0:
            return None
        coords = []
        for ch in st_obj[0]:
            if chain and ch.name != chain:
                continue
            if not any(is_amino_acid(r) for r in ch):
                continue
            for res in ch:
                if not is_amino_acid(res):
                    continue
                ca = None
                for at in res:
                    if at.name == "CA":
                        ca = at
                        break
                if ca is not None:
                    coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        return np.array(coords) if coords else None
    except Exception:
        return None
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except OSError: pass


def kabsch_superpose(mobile: np.ndarray,
                     target: np.ndarray) -> np.ndarray:
    """Optimal rigid-body superposition.

    Reference: Kabsch (1976) Acta Cryst A 32:922-923.
    """
    assert mobile.shape == target.shape
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    m, t = mobile - cm, target - ct
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign @ U.T
    return (m @ R.T) + ct


def compute_ensemble_rmsf(
    model_files: list[dict],
    chain: Optional[str] = None,
) -> dict:
    """Per-residue RMSF across AF2 models.

    Returns continuous distributions only, no binary classifications.
    """
    all_coords, model_ids = [], []
    for m in model_files:
        try: txt = open(m["file"]).read()
        except Exception: continue
        c = _ca_coords(txt, m["fmt"], chain)
        if c is not None and len(c) > 0:
            all_coords.append(c)
            model_ids.append(m.get("model_id", "?"))

    if len(all_coords) < 2:
        return {"error": "Requires >= 2 models", "n_models": len(all_coords)}

    n = all_coords[0].shape[0]
    compat = [(c, mid) for c, mid in zip(all_coords, model_ids)
              if c.shape[0] == n]
    if len(compat) < 2:
        return {"error": "Incompatible chain lengths", "n_models": len(compat)}

    coords_list = [c for c, _ in compat]
    used_ids = [mid for _, mid in compat]

    ref = coords_list[0]
    superposed = [ref] + [kabsch_superpose(c, ref) for c in coords_list[1:]]
    stack = np.stack(superposed)

    center = stack.mean(axis=0)
    deviations = stack - center[None, :, :]
    sq_dev = np.sum(deviations ** 2, axis=-1)
    rmsf = np.sqrt(np.mean(sq_dev, axis=0))

    M = len(superposed)
    pairwise = []
    for i in range(M):
        for j in range(i + 1, M):
            diff = superposed[i] - superposed[j]
            rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))
            pairwise.append({"model_i": used_ids[i], "model_j": used_ids[j],
                             "rmsd_A": round(rmsd, 3)})

    return {
        "rmsf": [round(float(v), 3) for v in rmsf],
        "mean_rmsf": round(float(np.mean(rmsf)), 3),
        "median_rmsf": round(float(np.median(rmsf)), 3),
        "std_rmsf": round(float(np.std(rmsf)), 3),
        "max_rmsf": round(float(np.max(rmsf)), 3),
        "n_residues": int(n),
        "n_models": len(superposed),
        "models_used": used_ids,
        "pairwise_rmsd": pairwise,
        "superposition_reference": "Kabsch (1976) Acta Cryst A 32:922",
        "rmsf_reference": "McCammon & Harvey (1987) Dynamics of Proteins and Nucleic Acids",
        "complementarity_reference": "Wallner (2023) Bioinformatics 39:btad573",
    }