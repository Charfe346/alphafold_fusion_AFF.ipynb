"""A3M multiple-sequence-alignment parsing and enrichment pipeline."""

from __future__ import annotations
import gzip, re
from pathlib import Path
from typing import Any, Optional
import numpy as np
from .config import log
from .sequence import parse_fasta


def find_a3m(job_dir: Path) -> list[Path]:
    """Find all .a3m / .a3m.gz files in job_dir (recursive)."""
    if not job_dir.exists():
        log.info("find_a3m: directory does not exist: %s", job_dir)
        return []
    files: list[Path] = []
    # ══════ Patterns élargis ══════
    for pat in ("*.a3m", "*.a3m.gz", "*.aln"):
        files.extend(job_dir.rglob(pat))

    # ColabFold crée parfois un sous-dossier <jobname>/ avec le .a3m dedans
    # rglob couvre déjà ça, mais on vérifie aussi le parent
    if not files and job_dir.parent.exists():
        for pat in ("*.a3m", "*.a3m.gz"):
            files.extend(job_dir.parent.rglob(pat))

    if not files:
        # Log diagnostique : qu'y a-t-il dans le répertoire ?
        try:
            all_files = sorted(job_dir.rglob("*"))
            extensions = set(f.suffix for f in all_files if f.is_file())
            log.info("find_a3m: no .a3m in %s — found %d files, "
                     "extensions: %s", job_dir.name, len(all_files),
                     extensions or "none")
        except Exception:
            pass
        return []

    def _score(p: Path) -> tuple[int, float]:
        n = p.name.lower()
        s = (100 if "uniref" in n else 0) + (50 if "bfd" in n else 0)
        try: return (s, p.stat().st_mtime)
        except Exception: return (s, 0)

    result = sorted(set(files), key=_score, reverse=True)
    log.info("find_a3m: found %d file(s) in %s: %s",
             len(result), job_dir.name,
             [f.name for f in result[:5]])
    return result


def _read_a3m(path: Path) -> str:
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", errors="ignore") as f:
                return f.read()
        with open(path, errors="ignore") as f:
            return f.read()
    except Exception as e:
        log.warning("_read_a3m: cannot read %s: %s", path.name, e)
        return ""


def _parse_a3m(path: Path) -> Optional[dict]:
    if not path or not path.exists():
        return None
    text = _read_a3m(path)
    if not text.strip():
        log.warning("_parse_a3m: empty file %s", path.name)
        return None

    entries: list[tuple[str, str]] = []
    head: Optional[str] = None
    buf: list[str] = []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if head is not None:
                entries.append((head, "".join(buf)))
            head = ln[1:].strip(); buf = []
        elif ln.startswith("#"):
            continue  # skip comment lines
        else:
            buf.append(ln.strip())
    if head is not None:
        entries.append((head, "".join(buf)))

    if not entries:
        log.warning("_parse_a3m: no entries in %s", path.name)
        return None

    log.info("_parse_a3m: %s → %d entries (query + %d hits)",
             path.name, len(entries), len(entries) - 1)

    qh, qa = entries[0]
    hits: list[dict] = []
    for hdr, aln in entries[1:]:
        toks = hdr.split()
        meta: dict[str, Any] = {"score": None, "id_pct": None, "evalue": None}
        if len(toks) >= 4:
            try: meta["score"] = float(toks[1])
            except (TypeError, ValueError): pass
            try:
                v = float(toks[2].rstrip("%"))
                meta["id_pct"] = v if v > 1 else v * 100
            except (TypeError, ValueError): pass
            try: meta["evalue"] = float(toks[3].replace("E", "e"))
            except (TypeError, ValueError): pass
        hits.append({"acc": toks[0] if toks else hdr,
                     "aln": aln, "meta": meta})
    return {"qh": qh, "qa": qa, "hits": hits}


def align_a3m(q: str, t: str) -> tuple[str, str]:
    qi = ti = 0
    ql: list[str] = []
    tl: list[str] = []
    while qi < len(q) or ti < len(t):
        qc = q[qi] if qi < len(q) else None
        tc = t[ti] if ti < len(t) else None
        if qc is not None and qc.islower():
            ql.append(qc); tl.append("-"); qi += 1; continue
        if tc is not None and tc.islower():
            tl.append(tc); ql.append("-"); ti += 1; continue
        ql.append(qc or "-"); tl.append(tc or "-")
        if qc is not None: qi += 1
        if tc is not None: ti += 1
    return "".join(ql), "".join(tl)


def compute_identity(ql: str, tl: str,
                     qlen: int) -> tuple[Optional[float], Optional[float], int]:
    match = aligned = cov = 0
    for q, t in zip(ql, tl):
        if q != "-": cov += 1
        if t != "-":
            aligned += 1
            if q.upper() == t.upper(): match += 1
    ident = (100 * match / aligned) if aligned > 0 else None
    coverage = round(100 * cov / max(1, qlen), 1) if qlen else None
    return ident, coverage, aligned


def build_a3m_data(name: str, job_dir: Path) -> Optional[dict]:
    """Build alignment data from .a3m files in job_dir."""
    a3m_files = find_a3m(job_dir)
    if not a3m_files:
        log.info("build_a3m_data: no .a3m files for '%s' in %s",
                 name, job_dir)
        return None

    parsed = None
    for f in a3m_files[:6]:
        parsed = _parse_a3m(f)
        if parsed and parsed.get("hits"):
            break
        parsed = None  # reset if no hits

    if not parsed:
        log.info("build_a3m_data: .a3m found but no parseable hits "
                 "for '%s'", name)
        return None

    qseq = re.sub(r"[^A-Za-z]", "", parsed["qa"]).upper()
    qlen = len(qseq)
    enriched: list[dict] = []
    for h in parsed["hits"]:
        try: ql, tl = align_a3m(parsed["qa"], h["aln"])
        except Exception: continue
        ident, cov, core = compute_identity(ql, tl, qlen)
        enriched.append({
            "acc": h["acc"],
            "aln": h["aln"],
            "Identity_pct": h["meta"].get("id_pct") or ident,
            "Score": h["meta"].get("score"),
            "Evalue": h["meta"].get("evalue"),
            "Coverage_pct": cov,
            "Core": core,
        })

    if not enriched:
        log.info("build_a3m_data: parsed but 0 enriched hits for '%s'", name)
        return None

    def _sk(x: dict) -> tuple:
        def _v(k, d):
            val = x.get(k)
            if val is None: return d
            if isinstance(val, float) and not np.isfinite(val): return d
            return val
        return (_v("Identity_pct", -1), _v("Score", -1),
                -_v("Evalue", float("inf")), _v("Coverage_pct", -1))
    enriched.sort(key=_sk, reverse=True)

    log.info("build_a3m_data: '%s' → %d hits enriched", name, len(enriched))
    return {"name": f"{name} (AUTO)", "qseq": qseq,
            "qlen": qlen, "hits": enriched}


def parse_text_block(txt: str) -> Optional[dict]:
    entries = parse_fasta(txt)
    if len(entries) < 2:
        return None
    qh, qs = entries[0]
    qseq = re.sub(r"[^A-Za-z]", "", qs).upper()
    hits = [{"acc": re.split(r"\s+", h)[0], "aln": s, "meta": {}}
            for h, s in entries[1:]]
    return {"name": f"Manual {qh}", "qseq": qseq,
            "qlen": len(qseq), "hits": hits}