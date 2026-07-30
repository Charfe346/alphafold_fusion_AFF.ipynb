"""Per-residue evolutionary conservation from MSA (Shannon entropy).

H(i) = -sum_a f_a(i) log2 f_a(i)
Reference: Shannon (1948) Bell Syst Tech J 27:379-423.

Normalised conservation: C(i) = 1 - H(i) / H_max
where H_max = log2(20) for 20 standard amino acids.
Review: Valdar (2002) Proteins 48:227-241.

Sequence weighting corrects for the phylogenetic redundancy of
ColabFold/MMseqs2 MSAs using position-based weights.
Reference: Henikoff & Henikoff (1994) J Mol Biol 243:574-578.

Weights are scaled so that their mean equals 1 (sum = n_sequences),
i.e. on the same scale as integer sequence counts. An optional
Laplace pseudocount (beta, default 0 = pure Shannon/Valdar) can be
added; when beta > 0 it is consistent with the count-scaled weights
(Durbin et al. 1998, Biological Sequence Analysis, Cambridge Univ.
Press).
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Optional
import numpy as np

_AA20 = set("ACDEFGHIKLMNPQRSTVWY")
_K = 20


def _henikoff_weights(columns: list[list[str]], n_seqs: int) -> np.ndarray:
    """Position-based sequence weights.

    Reference: Henikoff & Henikoff (1994) J Mol Biol 243:574-578.
    For each column, each of the k observed residue types contributes
    weight 1/(k * count(type)) to the sequences carrying it.

    Weights are scaled so that their SUM equals n_seqs (mean weight
    = 1), placing them on the same scale as integer sequence counts.
    This keeps any downstream pseudocount consistent with the
    weighted residue counts and avoids over-smoothing shallow MSAs.
    """
    weights = np.zeros(n_seqs, dtype=float)
    for col in columns:
        present = [c for c in col if c in _AA20]
        if len(present) < 2:
            continue
        counts = Counter(present)
        k = len(counts)
        for i, c in enumerate(col):
            if c in _AA20:
                weights[i] += 1.0 / (k * counts[c])
    total = weights.sum()
    if total <= 0:
        # Fallback: uniform weight of 1 per sequence (sum = n_seqs).
        return np.ones(n_seqs, dtype=float)
    # Scale to sum = n_seqs (mean weight = 1), not to sum = 1.
    return weights * (n_seqs / total)


def _weighted_entropy(column: list[str], seq_weights: np.ndarray,
                      beta: float) -> float:
    """Weighted Shannon entropy of one MSA column (bits).

    beta : total pseudocount mass spread over the 20 amino acids
           (alpha = beta / 20 per residue). beta = 0 gives pure
           weighted Shannon entropy (Shannon 1948; Valdar 2002).
    """
    wcounts = {a: 0.0 for a in _AA20}
    wtotal = 0.0
    for i, c in enumerate(column):
        cu = c.upper()
        if cu in _AA20:
            wcounts[cu] += seq_weights[i]
            wtotal += seq_weights[i]
    if wtotal <= 0:
        return 0.0
    alpha = beta / _K            # per-residue pseudocount
    denom = wtotal + beta        # = wtotal + _K * alpha
    h = 0.0
    for a in _AA20:
        f = (wcounts[a] + alpha) / denom
        if f > 0:
            h -= f * math.log2(f)
    return h


def conservation_profile(
    query_seq: str,
    aligned_hits: list[str],
    beta: float = 0.0,
) -> dict:
    """Per-position conservation from a parsed A3M MSA, with
    Henikoff & Henikoff (1994) sequence weighting.

    Parameters
    ----------
    query_seq    : ungapped query sequence (match-state reference)
    aligned_hits : A3M-aligned homolog rows
    beta         : total pseudocount mass. Default 0.0 = pure
                   Shannon/Valdar (no smoothing). Set beta > 0 for
                   Laplace-style smoothing (Durbin et al. 1998).

    Returns continuous per-residue scores only.
    """
    L = len(query_seq)
    all_seqs = [query_seq] + aligned_hits
    n_total = len(all_seqs)

    # Build per-position columns (upper-case = match state in A3M).
    columns: list[list[str]] = [[query_seq[p]] for p in range(L)]
    for hit in aligned_hits:
        qi = 0
        col_chars = ["-"] * L
        for c in hit:
            if qi >= L:
                break
            if c == "-" or c.isupper():
                col_chars[qi] = c
                qi += 1
        for p in range(L):
            columns[p].append(col_chars[p])

    # Henikoff weights (computed once over all columns), scaled to
    # sum = n_total (mean weight = 1).
    seq_weights = _henikoff_weights(columns, n_total)
    sw_sum = float(np.sum(seq_weights))
    sw_sq = float(np.sum(seq_weights ** 2))
    # Effective number of sequences: Neff = (sum w)^2 / sum(w^2).
    n_effective = (sw_sum ** 2 / sw_sq) if sw_sq > 0 else 0.0

    H_max = math.log2(_K)
    conservation, entropy, gap_fracs, depth = [], [], [], []

    for pos in range(L):
        col = columns[pos]
        h = _weighted_entropy(col, seq_weights, beta)
        entropy.append(round(h, 4))
        conservation.append(round(max(0.0, 1.0 - h / H_max), 4))
        n_gap = sum(1 for c in col if c in "-.")
        gap_fracs.append(round(n_gap / len(col), 4))
        depth.append(len(col) - n_gap)

    return {
        "conservation": conservation,
        "entropy": entropy,
        "gap_fraction": gap_fracs,
        "depth": depth,
        "n_sequences": n_total,
        "n_effective": round(n_effective, 2),
        "n_positions": L,
        "beta_pseudocount": beta,
        "mean_conservation": round(float(np.mean(conservation)), 4),
        "weighting": "Henikoff & Henikoff (1994) position-based",
        "entropy_reference": "Shannon (1948) Bell Syst Tech J 27:379",
        "scoring_reference": "Valdar (2002) Proteins 48:227",
        "weighting_reference": "Henikoff & Henikoff (1994) J Mol Biol 243:574",
        "pseudocount_reference": "Durbin et al. (1998) Biological Sequence Analysis",
    }