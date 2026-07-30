"""Remote API clients for UniProt, InterPro, and AlphaFold DB."""

from __future__ import annotations
import json, re, time, urllib.error, urllib.request
from typing import Any, Optional
import numpy as np, pandas as pd, streamlit as st
from .config import AFDB_API, INTERPRO_APIS, UNIPROT_API, log, EBI_CONTACT_EMAIL

# UniProt accession format (official pattern):
# [OPQ][0-9][A-Z0-9]{3}[0-9] | [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
_ACC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-\d+)?$"
)


def is_uniprot_acc(s: str) -> bool:
    return bool(s and _ACC_RE.match(s.strip().upper()))


def extract_uniprot_acc(s: str) -> Optional[str]:
    if not s:
        return None
    for tok in re.split(r"[|\s,;/]+", str(s).strip()):
        if is_uniprot_acc(tok.upper()):
            return tok.upper()
    m = re.search(r"UniRef\d+_([A-Z0-9]{6,10}(?:-\d+)?)", s, re.I)
    if m and is_uniprot_acc(m.group(1)):
        return m.group(1).upper()
    return None


def guess_acc(seq_name: str,
              afdb_used: Optional[dict[str, str]] = None) -> Optional[str]:
    if afdb_used:
        acc = afdb_used.get(seq_name)
        if acc:
            return acc
    if is_uniprot_acc(seq_name):
        return seq_name
    m = re.search(r"\b([A-Z0-9]{6,10})\b", seq_name, re.I)
    if m and is_uniprot_acc(m.group(1)):
        return m.group(1)
    return None


# ══════════════════════════════════════════════════════════════
# Generic JSON fetcher (for UniProt, AFDB)
# ══════════════════════════════════════════════════════════════
def _fetch_json(url: str, timeout: int = 15,
                retries: int = 3) -> Optional[dict]:
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429 or exc.code >= 500:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════
# UniProt
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_uniprot(acc: str) -> Optional[dict]:
    acc = (acc or "").strip()
    if not acc:
        return None
    return _fetch_json(UNIPROT_API.format(acc=acc), timeout=20)


def uniprot_domains(j: Optional[dict]) -> list[dict]:
    keep = {"Domain", "Repeat", "Region", "Coiled coil", "Zinc finger"}
    segs: list[dict] = []
    for f in (j or {}).get("features", []):
        if f.get("type") not in keep:
            continue
        loc = f.get("location") or {}
        try:
            b = int((loc.get("start") or {}).get("value"))
            e = int((loc.get("end") or {}).get("value"))
        except (TypeError, ValueError):
            continue
        segs.append({
            "start": b, "end": e,
            "label": f.get("description") or f.get("type", "Domain"),
            "type": f.get("type"),
        })
    return sorted(segs, key=lambda s: (s["start"], s["end"]))


# ══════════════════════════════════════════════════════════════
# InterPro — COPIÉ DU CODE FONCTIONNEL (all-in-one)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_interpro(acc: str, timeout: int = 20,
                   retries: int = 3, backoff: float = 1.0) -> Optional[dict]:
    """Fetch InterPro entries for a UniProt accession.

    Uses the EXACT same logic as the working all-in-one version:
    - Try each endpoint in order
    - Per endpoint: retry up to `retries` times
    - On 404: break inner loop, try next endpoint
    - On 408/5xx: sleep and retry same endpoint
    - On success with results: return immediately
    - On success with empty results: try next endpoint
    """
    try_acc = (acc or "").strip()
    if not try_acc:
        return None
    base_acc = try_acc.split("-")[0] if "-" in try_acc else try_acc

    endpoints = [
        f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{base_acc}?page_size=200",
        f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{base_acc}?page_size=200",
        f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/{base_acc}?page_size=200",
    ]
    headers = {"Accept": "application/json"}

    for url in endpoints:
        for attempt in range(max(1, retries)):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = r.read().decode("utf-8", "ignore")
                    j = json.loads(raw)

                # Check for results
                results = j.get("results") or []
                if results:
                    log.info("fetch_interpro: %d results from %s",
                             len(results), url[:80])
                    return {"results": results, "_source_url": url}
                if isinstance(j, list) and j:
                    log.info("fetch_interpro: %d results (list) from %s",
                             len(j), url[:80])
                    return {"results": j, "_source_url": url}

                # Empty results from this endpoint → try next endpoint
                break

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # Not found on this endpoint → try next
                    break
                elif e.code == 408 or e.code == 429 or e.code >= 500:
                    # Timeout / rate-limit / server error → retry
                    log.warning("fetch_interpro: HTTP %d, retry %d/%d",
                                e.code, attempt + 1, retries)
                    time.sleep(backoff * (attempt + 1))
                    continue
                else:
                    # Other HTTP error → try next endpoint
                    break
            except Exception as exc:
                log.warning("fetch_interpro: %s, retry %d/%d",
                            exc, attempt + 1, retries)
                time.sleep(backoff * (attempt + 1))
                continue

    log.warning("fetch_interpro: all endpoints failed for %s", base_acc)
    return None


def interpro_domains(j: Optional[dict]) -> list[dict]:
    """Extract domain segments from InterPro API response.

    Uses the EXACT same logic as the working all-in-one version:
    - Check metadata, then entry, then root-level fields
    - Also check short_name, acc, type_name (not just name/accession/type)
    - Extract from: proteins[].entry_protein_locations,
                     root locations, root entry_protein_locations
    - Deduplicate by (start, end, label)
    """
    segs: list[dict] = []
    if not j:
        return segs

    results = j.get("results") or []
    if not results and isinstance(j, list):
        results = j

    keep_types = {
        "domain", "repeat", "homologous_superfamily",
        "homologous superfamily", "family", "conserved_site",
        "active_site", "binding_site", "ptm", "coiled_coil",
        "coiled-coil",
    }

    for r in results:
        # ── Extract metadata (same priority as working code) ──
        metadata = r.get("metadata") or {}
        entry = r.get("entry") or {}

        if metadata:
            acc_id = metadata.get("accession") or ""
            name = (metadata.get("name")
                    or metadata.get("short_name") or "")
            etype = (metadata.get("type") or "").strip().lower()
        elif entry:
            acc_id = (entry.get("accession")
                      or entry.get("acc") or "")
            name = (entry.get("name")
                    or entry.get("short_name") or "")
            etype = (entry.get("type")
                     or entry.get("type_name") or "").strip().lower()
        else:
            acc_id = (r.get("accession")
                      or r.get("acc") or "")
            name = (r.get("name")
                    or r.get("short_name") or "")
            etype = (r.get("type") or "").strip().lower()

        # ── Type filter ──
        if etype and etype not in keep_types:
            continue

        label = (f"{acc_id} {name}".strip()
                 if (acc_id or name) else "InterPro entry")

        # ── Source 1: proteins[].entry_protein_locations ──
        proteins = r.get("proteins") or []
        for prot in proteins:
            locations = (prot.get("entry_protein_locations")
                         or prot.get("locations") or [])
            for loc in locations:
                fragments = loc.get("fragments") or []
                for fr in fragments:
                    try:
                        beg = int(fr.get("start"))
                        end = int(fr.get("end"))
                        if beg <= end:
                            segs.append({
                                "start": beg, "end": end,
                                "label": label,
                                "type": (etype.title()
                                         if etype else "Domain"),
                            })
                    except (TypeError, ValueError):
                        continue

        # ── Source 2: root-level locations ──
        locations_direct = r.get("locations") or []
        for loc in locations_direct:
            fragments = loc.get("fragments") or []
            for fr in fragments:
                try:
                    beg = int(fr.get("start"))
                    end = int(fr.get("end"))
                    if beg <= end:
                        segs.append({
                            "start": beg, "end": end,
                            "label": label,
                            "type": (etype.title()
                                     if etype else "Domain"),
                        })
                except (TypeError, ValueError):
                    continue

        # ── Source 3: root-level entry_protein_locations ──
        epl = r.get("entry_protein_locations") or []
        for loc in epl:
            fragments = loc.get("fragments") or []
            for fr in fragments:
                try:
                    beg = int(fr.get("start"))
                    end = int(fr.get("end"))
                    if beg <= end:
                        segs.append({
                            "start": beg, "end": end,
                            "label": label,
                            "type": (etype.title()
                                     if etype else "Domain"),
                        })
                except (TypeError, ValueError):
                    continue

    # ── Deduplicate ──
    uniq: list[dict] = []
    seen: set[tuple] = set()
    for s in segs:
        key = (s["start"], s["end"], s["label"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)

    log.info("interpro_domains: %d segments (deduped) from %d results",
             len(uniq), len(results))
    return sorted(uniq, key=lambda x: (x["start"], x["end"]))


# ══════════════════════════════════════════════════════════════
# AlphaFold DB
# ══════════════════════════════════════════════════════════════
def fetch_afdb(name: str, timeout: int = 10) -> Optional[dict]:
    if not name:
        return None
    cands = [t for t in re.split(r"[|\s,;/]+", str(name))
             if is_uniprot_acc(t)]
    if not cands:
        m = re.search(r"\b([A-Z0-9]{6,10}(?:-\d+)?)\b", name)
        if m:
            cands.append(m.group(1))
    for acc in cands:
        try_list = [acc]
        if "-" in acc:
            try_list.append(acc.split("-")[0])
        for a in try_list:
            j = _fetch_json(AFDB_API.format(acc=a), timeout=timeout)
            if isinstance(j, list) and j:
                o = j[0]
                return {
                    "acc": o.get("uniprotAccession", a),
                    "pdb_url": o.get("pdbUrl"),
                    "cif_url": o.get("cifUrl"),
                    "bcif_url": o.get("bcifUrl"),
                }
    return None


def enrich_df_urls(df: pd.DataFrame,
                   col: str = "Accession") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    u_urls: list[Optional[str]] = []
    a_urls: list[Optional[str]] = []
    for v in out[col].astype(str):
        acc = extract_uniprot_acc(v)
        if acc:
            u_urls.append(f"https://www.uniprot.org/uniprotkb/{acc}")
            a_urls.append(
                f"https://alphafold.ebi.ac.uk/entry/{acc.split('-')[0]}")
        else:
            u_urls.append(None)
            a_urls.append(None)
    out["UniProt_URL"] = u_urls
    out["AFDB_URL"] = a_urls
    return out
def blast_identify(seq: str, timeout: int = 1800) -> Optional[dict]:
    """Identify a protein by BLAST against UniProtKB (EBI service).

    Returns the top hit with accession, identity %, description.
    Reference: Madeira et al. (2022) NAR 50:W276-W279.
    """
    import urllib.parse
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", (seq or "").upper())
    if len(seq) < 10:
        return None
    base = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
    try:
        params = {
            "email": EBI_CONTACT_EMAIL,
            "program": "blastp",
            "stype": "protein",
            "database": "uniprotkb",
            "sequence": seq,
        }
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(f"{base}/run", data=data)
        with urllib.request.urlopen(req, timeout=30) as r:
            job_id = r.read().decode().strip()
        log.info("blast_identify: job %s", job_id)

        # Poll (up to `timeout` seconds, 5 s intervals)
        finished = False
        for _ in range(int(timeout / 5)):
            time.sleep(5)
            sreq = urllib.request.Request(f"{base}/status/{job_id}")
            with urllib.request.urlopen(sreq, timeout=15) as r:
                status = r.read().decode().strip()
            if status == "FINISHED":
                finished = True
                break
            if status in ("ERROR", "FAILURE", "NOT_FOUND"):
                log.warning("blast_identify: %s", status)
                return None
        if not finished:
            log.warning("blast_identify: timeout after %ds", timeout)
            return None

        rreq = urllib.request.Request(f"{base}/result/{job_id}/json")
        with urllib.request.urlopen(rreq, timeout=30) as r:
            j = json.loads(r.read().decode("utf-8", "ignore"))

        hits = j.get("hits", [])
        if not hits:
            return None
        top = hits[0]
        hsp = (top.get("hit_hsps") or [{}])[0]
        acc = top.get("hit_acc", "") or top.get("hit_id", "")
        ident = float(hsp.get("hsp_identity", 0))
        cov = None
        try:
            cov = round(100.0 * int(hsp.get("hsp_align_len", 0))
                        / max(1, len(seq)), 1)
        except Exception:
            pass
        return {
            "accession": acc,
            "description": top.get("hit_desc", ""),
            "gene": top.get("hit_uni_gn", ""),
            "organism": top.get("hit_uni_os", ""),
            "identity_pct": round(ident, 1),
            "coverage_pct": cov,
            "length": top.get("hit_len"),
            "is_exact": ident >= 99.0,
        }
    except Exception as exc:
        log.warning("blast_identify: %s", exc)
        return None
def blast_hits(seq: str, timeout: int = 1800,
               max_hits: int = 20) -> Optional[list[dict]]:
    """BLAST against UniProtKB, return top hits with organism info.

    Lets the user pick the correct species/entry rather than forcing
    the single top hit (which may be an ortholog).
    Reference: Madeira et al. (2022) NAR 50:W276.
    """
    import urllib.parse
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", (seq or "").upper())
    if len(seq) < 10:
        return None
    base = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
    try:
        params = {
            "email": EBI_CONTACT_EMAIL, "program": "blastp",
            "stype": "protein", "database": "uniprotkb",
            "sequence": seq,
        }
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(f"{base}/run", data=data)
        with urllib.request.urlopen(req, timeout=30) as r:
            job_id = r.read().decode().strip()

        finished = False
        for _ in range(int(timeout / 5)):
            time.sleep(5)
            sreq = urllib.request.Request(f"{base}/status/{job_id}")
            with urllib.request.urlopen(sreq, timeout=15) as r:
                status = r.read().decode().strip()
            if status == "FINISHED":
                finished = True
                break
            if status in ("ERROR", "FAILURE", "NOT_FOUND"):
                return None
        if not finished:
            return None

        rreq = urllib.request.Request(f"{base}/result/{job_id}/json")
        with urllib.request.urlopen(rreq, timeout=30) as r:
            j = json.loads(r.read().decode("utf-8", "ignore"))

        out = []
        for top in j.get("hits", [])[:max_hits]:
            hsp = (top.get("hit_hsps") or [{}])[0]
            raw_acc = top.get("hit_acc", "") or top.get("hit_id", "")
            acc = extract_uniprot_acc(raw_acc) or raw_acc
            ident = float(hsp.get("hsp_identity", 0))
            out.append({
                "accession": acc,
                "gene": top.get("hit_uni_gn", ""),
                "organism": top.get("hit_uni_os", ""),
                "description": top.get("hit_uni_de", ""),
                "identity_pct": round(ident, 1),
                "length": top.get("hit_len"),
            })
        return out if out else None
    except Exception as exc:
        log.warning("blast_hits: %s", exc)
        return None