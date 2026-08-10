"""Sequence parsing, cleaning, and filesystem helpers."""

from __future__ import annotations
import hashlib, re, secrets, time
from pathlib import Path
from typing import Optional


def clean_sequence(seq: str) -> tuple[str, list[str]]:
    raw = (seq or "").upper()
    warnings = []
    non_std = set(re.findall(r"[XUBZ]", raw))
    if non_std:
        warnings.append(f"Non-standard amino acids removed: {', '.join(sorted(non_std))}")
    cleaned = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY:]", "", raw)
    cleaned = re.sub(r":+", ":", cleaned).strip(":")
    return cleaned, warnings


def is_complex(seq: str) -> bool:
    return ":" in (seq or "")


def total_length(seq: str) -> int:
    return sum(len(p) for p in clean_sequence(seq).split(":") if p)


def safe_basename(name: str, seq: str = "", maxlen: int = 80) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "", name or "seq").strip("._-") or "seq"
    h = hashlib.sha1(f"{name}|{seq}".encode()).hexdigest()[:10]
    return f"{base[:max(8, maxlen - len(h) - 1)]}_{h}"


def parse_fasta(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    head: Optional[str] = None
    buf: list[str] = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            if head and buf:
                entries.append((head, "".join(buf)))
                head = None; buf = []
            continue
        if ln.startswith(">"):
            if head and buf:
                entries.append((head, "".join(buf)))
            head = ln[1:].strip(); buf = []
        else:
            if head is not None:
                buf.append(ln)
            else:
                head = ln
    if head and buf:
        entries.append((head, "".join(buf)))
    return entries


def new_run_dir(results_dir: Path) -> Path:
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    root = results_dir / f"run_{rid}"
    root.mkdir(parents=True, exist_ok=True)
    return root
