"""Inter-chain Cα-Cα contact analysis for multimers.

Contact identification uses Cα-Cα distance with an 8 Angstrom
threshold (Duarte et al. 2012, BMC Bioinformatics 13:334). This is a
contact-map definition, NOT a SASA-buried interface (cf. PISA,
Krissinel & Henrick 2007, J Mol Biol 372:774-797).

Inter-chain PAE decomposition uses the TM-score kernel with
length-dependent d0 = 1.24(L-15)^{1/3} - 1.8
(Zhang & Skolnick 2004, Proteins 57:702-710).
"""

from __future__ import annotations
import os, tempfile
from collections import Counter
from typing import Optional
import numpy as np
import gemmi
from .structure import is_amino_acid


def _chain_data(txt: str, fmt: str) -> dict[str, dict]:
    """Extract per-chain Ca coordinates, B-factors, residue info."""
    tmp_path = None
    try:
        suffix = ".cif" if fmt == "cif" else ".pdb"
        tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        tmp.write(txt); tmp.flush(); tmp.close()
        tmp_path = tmp.name
        st_obj = gemmi.read_structure(tmp_path)
        if len(st_obj) == 0:
            return {}
        chains = {}
        for ch in st_obj[0]:
            if not any(is_amino_acid(r) for r in ch):
                continue
            coords, resnums, bfacs, resnames = [], [], [], []
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
                    try:
                        resnums.append(int(res.seqid.num))
                    except Exception:
                        resnums.append(len(resnums) + 1)
                    bfacs.append(float(ca.b_iso))
                    resnames.append(res.name)
            if coords:
                chains[ch.name] = {
                    "coords": np.array(coords), "resnums": resnums,
                    "bfacs": bfacs, "resnames": resnames}
        return chains
    except Exception:
        return {}
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except OSError: pass


def compute_contacts(
    txt: str, fmt: str,
    chain_a: Optional[str] = None,
    chain_b: Optional[str] = None,
    cutoff: float = 8.0,
) -> dict:
    """Identify inter-chain contacts and compute interface metrics.

    Parameters
    ----------
    cutoff  : Ca-Ca distance threshold (Angstrom).
              Default 8.0 per Duarte et al. (2012) BMC Bioinformatics 13:334.

    Returns all contacts with distances and per-residue pLDDT.
    """
    chains = _chain_data(txt, fmt)
    if len(chains) < 2:
        return {"error": "Requires >= 2 protein chains", "n_contacts": 0}

    names = list(chains.keys())
    if chain_a is None: chain_a = names[0]
    if chain_b is None: chain_b = names[1] if len(names) > 1 else names[0]
    da, db = chains.get(chain_a), chains.get(chain_b)
    if da is None or db is None:
        return {"error": f"Chain {chain_a} or {chain_b} not found"}

    ca, cb = da["coords"], db["coords"]
    diff = ca[:, None, :] - cb[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    idx_a, idx_b = np.where(dists <= cutoff)
    contacts = []
    iface_a, iface_b = set(), set()

    for ia, ib in zip(idx_a, idx_b):
        contacts.append({
            "chain_a": chain_a, "res_a": da["resnums"][ia],
            "resname_a": da["resnames"][ia],
            "chain_b": chain_b, "res_b": db["resnums"][ib],
            "resname_b": db["resnames"][ib],
            "distance_A": round(float(dists[ia, ib]), 2),
            "plddt_a": round(da["bfacs"][ia], 1) if 0 <= da["bfacs"][ia] <= 100 else None,
            "plddt_b": round(db["bfacs"][ib], 1) if 0 <= db["bfacs"][ib] <= 100 else None,
        })
        iface_a.add(ia); iface_b.add(ib)

    iplddt_a = [da["bfacs"][i] for i in iface_a if 0 <= da["bfacs"][i] <= 100]
    iplddt_b = [db["bfacs"][i] for i in iface_b if 0 <= db["bfacs"][i] <= 100]
    all_ip = iplddt_a + iplddt_b

    cc_a, cc_b = Counter(idx_a), Counter(idx_b)
    interface_details_a = [
        {"resnum": da["resnums"][i], "resname": da["resnames"][i],
         "plddt": round(da["bfacs"][i], 1), "n_contacts": cc_a[i]}
        for i in sorted(iface_a)
    ]
    interface_details_b = [
        {"resnum": db["resnums"][i], "resname": db["resnames"][i],
         "plddt": round(db["bfacs"][i], 1), "n_contacts": cc_b[i]}
        for i in sorted(iface_b)
    ]

    return {
        "chain_a": chain_a, "chain_b": chain_b,
        "cutoff_A": cutoff,
        "cutoff_reference": "Duarte et al. (2012) BMC Bioinformatics 13:334",
        "n_contacts": len(contacts),
        "n_interface_a": len(iface_a),
        "n_interface_b": len(iface_b),
        "contacts": contacts,
        "interface_residues_a": sorted(da["resnums"][i] for i in iface_a),
        "interface_residues_b": sorted(db["resnums"][i] for i in iface_b),
        "interface_details_a": interface_details_a,
        "interface_details_b": interface_details_b,
        "mean_plddt_interface": round(float(np.mean(all_ip)), 1) if all_ip else None,
        "mean_plddt_interface_a": round(float(np.mean(iplddt_a)), 1) if iplddt_a else None,
        "mean_plddt_interface_b": round(float(np.mean(iplddt_b)), 1) if iplddt_b else None,
    }


def interchain_pae(
    pae: np.ndarray,
    chain_lengths: list[int],
) -> dict:
    """Decompose PAE into per-block TM-score kernel summaries.

    For each (i, j) chain block we report the mean of the TM-score
    kernel weight w = 1 / (1 + (PAE/d0)^2), with length-dependent
    d0(L) = 1.24(L-15)^{1/3} - 1.8 (Zhang & Skolnick 2004, Proteins
    57:702-710).

    IMPORTANT: `tm_kernel_score` is the MEAN of the TM kernel over the
    block. It is NOT the AlphaFold ipTM/pTM score, which is defined as
    a MAXIMUM over alignment rows of a mean TM-kernel term. We use the
    block mean as a lightweight, symmetric descriptor of inter-chain
    PAE quality; do not interpret it as AlphaFold's (i)pTM.
    """
    boundaries = []
    offset = 0
    for length in chain_lengths:
        boundaries.append((offset, offset + length))
        offset += length

    result = {}
    for i, (si, ei) in enumerate(boundaries):
        for j, (sj, ej) in enumerate(boundaries):
            block = pae[si:ei, sj:ej]
            L = ej - sj
            d0 = max(0.5, 1.24 * max(0, L - 15) ** (1.0 / 3) - 1.8)
            tm_w = 1.0 / (1.0 + (block / d0) ** 2)
            result[f"{i}_{j}"] = {
                "chain_i": i, "chain_j": j,
                "len_i": ei - si, "len_j": ej - sj,
                "mean_pae": round(float(np.mean(block)), 2),
                "median_pae": round(float(np.median(block)), 2),
                "tm_kernel_score": round(float(np.mean(tm_w)), 4),
                "d0_used": round(d0, 3),
                "is_interchain": i != j,
            }
    return result