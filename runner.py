"""ColabFold execution, result harvesting, and analysis."""

from __future__ import annotations
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any, Optional
import numpy as np
from .config import CACHE_DIR, MSA_MODES, RESULTS_DIR
from .structure import avg_plddt, cif_to_pdb


def has_gpu() -> bool:
    """Quick check: is a GPU physically present?"""
    return shutil.which("nvidia-smi") is not None

def jax_backend() -> str:
    """Return JAX's active backend ('gpu', 'cpu', or 'unknown').

    ColabFold silently falls back to CPU when JAX cannot see the GPU
    (e.g. after a NumPy/CUDA desync), yielding catastrophically low
    pLDDT with no visible error. We probe the actual JAX backend—not
    just nvidia-smi—so the UI can warn before a long CPU run.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import jax; print(jax.default_backend())"],
            capture_output=True, text=True, timeout=30,
        )
        out = (r.stdout or "").strip().lower()
        if "gpu" in out:
            return "gpu"
        if "cpu" in out:
            return "cpu"
        return "unknown"
    except Exception:
        return "unknown"

def _parse_ranking(job_dir: Path) -> dict:
    files: list[Path] = []
    for pat in ("ranking_debug.json", "ranking.json"):
        files = list(job_dir.rglob(pat))
        if files: break
    if not files: return {}
    try:
        with open(files[0]) as f:
            d = json.load(f)
        out: dict[str, Any] = {}
        if isinstance(d.get("order"), list): out["order"] = d["order"]
        for k in ("ranking_confidence", "plddts", "iptms", "ptms"):
            if isinstance(d.get(k), dict): out[k] = d[k]
        return out
    except Exception: return {}


def _extract_model_id(p: Path) -> Optional[str]:
    n = p.name
    patterns = [
        # KEEP underscore: "model_1" (matches ColabFold naming)
        (r"model_?(\d+)", lambda m: f"model_{m.group(1)}"),
        (r"ranked_?(\d+)", lambda m: f"model_{int(m.group(1)) + 1}"),
    ]
    for pat, fn in patterns:
        match = re.search(pat, n, re.I)
        if match:
            try: return fn(match)
            except Exception: pass
    return None


def _rank_from_name(name: str) -> Optional[int]:
    for pat in (r"ranked_?(\d+)", r"rank_?(\d+)"):
        m = re.search(pat, name, re.I)
        if m:
            try:
                offset = 1 if "ranked" in pat else 0
                return int(m.group(1)) + offset
            except Exception: pass
    return None

def _model_num(s: str) -> Optional[str]:
    """Extract the numeric model index from any model id / json key."""
    m = re.search(r"model_?(\d+)", s or "", re.I)
    return m.group(1) if m else None


def _lookup(d: Optional[dict], mid: str):
    """Tolerant lookup by model number.

    ColabFold ranking keys look like 'model_1_multimer_v3_pred_0'
    while our mid is 'model_1'. Match on the numeric index to avoid
    the model_1 / model_10 false-positive trap.
    """
    if not d or not mid:
        return None
    if mid in d:
        return d[mid]
    num = _model_num(mid)
    if num is None:
        return None
    for k, v in d.items():
        if _model_num(k) == num:
            return v
    return None


def _is_template(p: Path) -> bool:
    """True if a file lives under a ColabFold 'templates_*/' subdir.

    ColabFold downloads candidate PDB templates into 'templates_NNN/'
    subdirectories when --templates is enabled. These are experimental
    reference structures, NOT predicted models: they carry no model_N
    id, no rank and no pTM/ipTM. Including them in harvest() pollutes
    the model table (empty ranks/scores, template PDB codes shown as
    'best model'). We therefore exclude them everywhere in harvesting.
    """
    return "template" in str(p).lower()


def _scores_from_files(job_dir: Path) -> dict:
    """Read per-model ptm/iptm from ColabFold *_scores_*.json files.

    ColabFold 1.6.x stores scores per model in separate JSON files
    named '*_scores_rank_NNN_..._model_M_...json' instead of a single
    ranking_debug.json. Returns {model_num: {ptm, iptm}}.
    """
    out: dict[str, dict] = {}
    for f in job_dir.rglob("*scores_rank*.json"):
        if _is_template(f):
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        num = _model_num(f.name)   # extrait le numéro de model_M
        if num is None:
            continue
        out[num] = {
            "ptm": d.get("ptm"),
            "iptm": d.get("iptm"),
        }
    return out


def harvest(job_dir: Path) -> tuple[list[dict], dict]:
    rnk = _parse_ranking(job_dir)
    scores = _scores_from_files(job_dir)
    # ══════ FIX: exclude template files (ColabFold templates_*/ subdirs).
    # rglob is recursive and would otherwise pick up downloaded PDB
    # templates as if they were predicted models. ══════
    pdbs = sorted(
        (p for p in job_dir.rglob("*.pdb") if not _is_template(p)),
        key=lambda p: p.stat().st_mtime)
    cifs = sorted(
        (p for p in (list(job_dir.rglob("*.cif"))
                     + list(job_dir.rglob("*.bcif")))
         if not _is_template(p)),
        key=lambda p: p.stat().st_mtime,
    )
    # ════════════════════════════════════════════════════════════════
    if not pdbs and cifs:
        for c in cifs:
            out = c.with_name(c.stem + "_conv.pdb")
            try:
                if cif_to_pdb(c, out): pdbs.append(out)
            except Exception: pass
        pdbs.sort(key=lambda p: p.stat().st_mtime)
    order = rnk.get("order", [])
    rows: list[dict] = []
    for fmt, path in [("pdb", p) for p in pdbs] + [("cif", c) for c in cifs]:
        mid = _extract_model_id(path)
        rpos = None
        if mid and order:
            # match rank order by model number (avoids model_1/model_10 trap)
            for oi, ok in enumerate(order):
                if _model_num(ok) == _model_num(mid):
                    rpos = oi + 1
                    break
        if rpos is None:
            rpos = _rank_from_name(path.name)
        rows.append({
            "model_id": mid or "-", "file": str(path), "fmt": fmt,
            "avg_plddt": avg_plddt(path, fmt), "rank": rpos,
            "ranking_conf": _lookup(rnk.get("ranking_confidence"), mid),
            "ptm": _lookup(rnk.get("ptms"), mid)
                   or (scores.get(_model_num(mid) or "", {}).get("ptm")),
            "iptm": _lookup(rnk.get("iptms"), mid)
                    or (scores.get(_model_num(mid) or "", {}).get("iptm")),
        })
    rows.sort(key=lambda x: (
        x["rank"] if x["rank"] is not None else 9999,
        -(x["avg_plddt"] or 0),
    ))
    return rows, rnk


def analyze(job_dir: Path) -> dict:
    rows, rnk = harvest(job_dir)
    res: dict[str, Any] = {
        "status": "error", "message": "No PDB/CIF", "models": rows,
        "ranking": rnk, "pae_json": None, "pae_png": None, "coverage_png": None,
    }
    if rows:
        res["status"] = "success"; res["message"] = ""
    for tag, key in [("*pae*.json", "pae_json"),
                     ("*predicted_aligned_error*.json", "pae_json"),
                     ("*pae*.png", "pae_png"),
                     ("*coverage*.png", "coverage_png")]:
        if res.get(key):
            continue
        # ══════ FIX: also skip template dirs when locating PAE/coverage ══════
        files = [f for f in job_dir.rglob(tag) if not _is_template(f)]
        if files:
            res[key] = str(sorted(files, key=lambda f: f.stat().st_mtime)[-1])
    return res


def quality_warnings(metrics: dict) -> list[str]:
    w: list[str] = []
    v = metrics.get("avg_plddt")
    if v is not None and v < 70:
        w.append(f"⚠️ Low mean pLDDT ({v:.1f}): proceed with caution.")
    v = metrics.get("iptm")
    if v is not None and v < 0.4:
        w.append(f"⚠️ Low ipTM ({v:.2f}): interfaces likely uncertain.")
    v = metrics.get("ptm")
    if v is not None and v < 0.6:
        w.append(f"⚠️ Low pTM ({v:.2f}): global topology may be uncertain.")
    return w
def count_msa_sequences(job_dir: Path) -> Optional[int]:
    """Count sequences in the deepest .a3m of a job (max over files).

    ColabFold may leave an empty single-sequence .a3m from a failed
    MSA search alongside the real one; we take the MAX depth found.
    Returns None if no .a3m present.
    """
    import gzip, glob
    files = list(job_dir.rglob("*.a3m")) + list(job_dir.rglob("*.a3m.gz"))
    if not files:
        return None
    best = 0
    for f in files:
        try:
            opener = gzip.open if str(f).endswith(".gz") else open
            with opener(f, "rt", errors="ignore") as fh:
                n = sum(1 for line in fh if line.startswith(">"))
            best = max(best, n)
        except Exception:
            continue
    return best


def msa_warning(n_seqs: Optional[int],
                avg_plddt: Optional[float]) -> Optional[str]:
    """Warn when a low pLDDT is caused by an empty MSA, not disorder.

    A single-sequence MSA (n_seqs <= 1) forces ColabFold into
    single-sequence mode, collapsing pLDDT even for well-folded
    proteins. This must NOT be interpreted as intrinsic disorder.
    """
    if n_seqs is not None and n_seqs <= 1 and \
       avg_plddt is not None and avg_plddt < 70:
        return ("⚠️ Empty MSA (1 sequence): prediction ran in "
                "single-sequence mode. The low pLDDT reflects the "
                "MISSING MSA, NOT intrinsic disorder. Re-run with a "
                "valid MSA (check MMseqs2 server / disable cache).")
    return None

def model_name_for_path(path: str, results: dict) -> Optional[str]:
    for n, r in (results or {}).items():
        for m in (r or {}).get("models", []):
            if m.get("file") == path: return n
    return None


def _estimate_timeout(seq_len: int, msa_mode: str, params: dict) -> int:
    base = {"mmseqs2_uniref_env": 3000,
            "mmseqs2_uniref": 1800}.get(msa_mode, 600)
    factor = (
        max(1, params.get("num_models", 1))
        * max(1.0, params.get("num_recycles", 3) / 3.0)
        * (1.5 if "multimer" in str(params.get("model_type", "")) else 1.0)
        * min(4.0, max(1.0, (seq_len or 300) / 300))
    )
    return max(600, min(21600, int(base * factor)))


def _run_subprocess(cmd: list[str], timeout_sec: int = 7200,
                    env: Optional[dict] = None) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, timeout=timeout_sec, check=False,
        )
        return p.returncode == 0, p.stdout or ""
    except subprocess.TimeoutExpired as e:
        return False, (e.stdout or "") + "\n[TIMEOUT]"
    except Exception as e:
        return False, str(e)


def run_colabfold(fasta: str, outdir: str, params: dict,
                  seq_len: int = 0) -> tuple[bool, str]:
    exe = shutil.which("colabfold_batch")
    cmd = [exe or sys.executable]
    if not exe: cmd += ["-m", "colabfold.batch"]
    cmd += [fasta, outdir]
    msa = MSA_MODES.get(
        str(params.get("msa_strategy", "fast")), "mmseqs2_uniref")
    if params.get("model_type"):
        cmd += ["--model-type", params["model_type"]]
    cmd += ["--msa-mode", msa]
    if params.get("pair_mode"):
        cmd += ["--pair-mode", params["pair_mode"]]
    cmd += [
        "--num-models", str(params.get("num_models", 1)),
        "--num-recycle", str(params.get("num_recycles", 3)),
        "--rank", "auto",
        "--jobname", params.get("jobname_prefix", "job"),
        "--recompile-padding", "1",
    ]
    if params.get("use_templates"): cmd += ["--templates"]
    if params.get("use_amber"): cmd += ["--amber"]
    sc = params.get("stop_at_score")
    if sc is not None:
        try: cmd += ["--stop-at-score", f"{float(sc):.3f}"]
        except (TypeError, ValueError): pass
    if shutil.which("nvidia-smi"):
        cmd += ["--disable-unified-memory"]
    if not params.get("reuse_cache", True):
        cmd += ["--overwrite-existing-results"]
    env = {
        **os.environ,
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".95",
    }
    return _run_subprocess(cmd, _estimate_timeout(seq_len, msa, params), env)


def fallback_cpu(fasta: str, outdir: str, prefix: str,
                 multimer: bool, seq_len: int) -> tuple[bool, str]:
    exe = shutil.which("colabfold_batch")
    cmd = [exe or sys.executable]
    if not exe: cmd += ["-m", "colabfold.batch"]
    cmd += [
        fasta, outdir,
        "--model-type",
        "alphafold2_multimer_v3" if multimer else "alphafold2_ptm",
        "--msa-mode", "single_sequence",
        "--num-models", "1", "--num-recycle", "2",
        # single_sequence ne supporte PAS paired
        "--pair-mode", "unpaired",
        "--rank", "auto", "--random-seed", "42",
        "--disable-unified-memory", "--recompile-padding", "1",
        "--jobname", prefix, "--max-msa", "64:64",
        "--disable-cluster-profile", "--stop-at-score", "70",
    ]
    env = {
        **os.environ,
        "JAX_PLUGINS": "disabled", "JAX_PLATFORM_NAME": "cpu",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    return _run_subprocess(
        cmd, _estimate_timeout(seq_len, "single_sequence", {}), env)