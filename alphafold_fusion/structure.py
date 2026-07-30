"""Structure file I/O: CIF/PDB conversion, chain detection, format helpers."""

from __future__ import annotations
import os, re, tempfile
from pathlib import Path
from typing import Optional
import numpy as np, gemmi
from Bio.PDB import PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from .config import AA3


def is_amino_acid(res) -> bool:
    try:
        return (res.name or "").strip().upper() in AA3
    except Exception:
        return False


def cif_to_pdb(cif_path: Path, pdb_out: Path) -> bool:
    try:
        pdb_out.write_text(
            gemmi.read_structure(str(cif_path)).make_minimal_pdb())
        return pdb_out.stat().st_size > 0
    except Exception:
        pass
    try:
        io = PDBIO()
        io.set_structure(
            MMCIFParser(QUIET=True).get_structure("S", str(cif_path)))
        io.save(str(pdb_out))
        return pdb_out.stat().st_size > 0
    except Exception:
        return False


def polymer_chains(path) -> list[str]:
    try:
        st_obj = gemmi.read_structure(str(path))
        ch = [(c.name, sum(1 for r in c if is_amino_acid(r)))
              for c in st_obj[0]]
        return [n for n, cnt in sorted(ch, key=lambda x: -x[1]) if cnt > 0]
    except Exception:
        return []


def avg_plddt(path, fmt: str) -> Optional[float]:
    """Mean pLDDT over CA atoms only (one value per residue).

    pLDDT is a per-residue confidence score stored in the B-factor
    column. Averaging over ALL atoms would bias the mean toward large
    side-chain residues and diverge from the CA-based per-residue
    profile used elsewhere (plddt_by_chain). We therefore restrict to
    CA atoms so that avg_plddt equals the mean of the residue-level
    profile.
    """
    vals: list[float] = []
    if fmt == "pdb":
        try:
            with open(path) as f:
                for ln in f:
                    if not ln.startswith("ATOM") or len(ln) < 66:
                        continue
                    # CA atoms only (name field is columns 13-16)
                    if ln[12:16].strip() != "CA":
                        continue
                    try:
                        b = float(ln[60:66])
                        if 0 <= b <= 100:
                            vals.append(b)
                    except ValueError:
                        pass
        except Exception:
            pass
    else:
        try:
            st_obj = gemmi.read_structure(str(path))
            if len(st_obj) > 0:
                for ch in st_obj[0]:
                    for res in ch:
                        if not is_amino_acid(res):
                            continue
                        for at in res:
                            if at.name == "CA":
                                b = float(at.b_iso)
                                if 0 <= b <= 100:
                                    vals.append(b)
                                break
        except Exception:
            pass
    return float(np.mean(vals)) if vals else None


def detect_fmt(txt: str) -> str:
    h = (txt or "")[:2000]
    if "atom_site." in h or h.lstrip().startswith("data"):
        return "cif"
    return "pdb"


def first_model_only(fmt: str, txt: str) -> str:
    if fmt == "pdb" and re.search(r"^MODEL", txt, re.M):
        m = re.search(
            r"^MODEL[^\n]*\n(.*?)(?:\nENDMDL|\Z)", txt, re.S | re.M)
        if m:
            return "\n".join(
                ln for ln in m.group(1).splitlines()
                if ln.startswith(("ATOM", "HETATM", "TER"))
            ) + "\n"
    return txt