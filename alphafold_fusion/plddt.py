"""pLDDT confidence score extraction, binning, and Plotly visualisation."""

from __future__ import annotations
import os, tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional
import gemmi, numpy as np
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from .config import C_HIGH, C_LOW, C_VHIGH, C_VLOW, PLDDT_SCHEME
from .structure import detect_fmt, is_amino_acid


def plddt_by_chain(txt: str, fmt: str = "pdb",
                   chain: Optional[str] = None) -> dict[str, list[float]]:
    fmt = fmt or detect_fmt(txt)
    result: dict[str, list[float]] = {}
    if fmt == "pdb":
        maps: dict[str, OrderedDict] = defaultdict(OrderedDict)
        for ln in (txt or "").splitlines():
            if not ln.startswith("ATOM") or len(ln) < 66:
                continue
            if ln[12:16].strip() != "CA":
                continue
            ch = ln[21:22].strip() or "A"
            if chain and ch != chain:
                continue
            key = (ln[22:26].strip(), ln[26:27])
            try:
                b = float(ln[60:66])
                if 0 <= b <= 100 and key not in maps[ch]:
                    maps[ch][key] = b
            except ValueError:
                pass
        for ch, od in maps.items():
            if od:
                result[ch] = list(od.values())
        return result
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False)
        tmp.write(txt or ""); tmp.flush(); tmp.close()
        tmp_path = tmp.name
        st_obj = gemmi.read_structure(tmp_path)
        if len(st_obj) > 0:
            for ch_obj in st_obj[0]:
                if not any(is_amino_acid(r) for r in ch_obj):
                    continue
                if chain and ch_obj.name != chain:
                    continue
                vals: list[float] = []
                for res in ch_obj:
                    if not is_amino_acid(res):
                        continue
                    ca_b = None
                    for at in res:
                        if at.name == "CA":
                            ca_b = float(at.b_iso)
                            break
                    if ca_b is not None:
                        if 0 <= ca_b <= 100:
                            vals.append(ca_b); continue
                    bs = [float(a.b_iso) for a in res
                          if 0 <= float(getattr(a, "b_iso", -1)) <= 100]
                    if bs:
                        vals.append(sum(bs) / len(bs))
                if vals:
                    result[ch_obj.name] = vals
    except Exception:
        pass
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except OSError: pass
    return result


def classify_plddt(txt: str, fmt: str,
                   only_chain: Optional[str] = None) -> dict:
    out: dict[str, dict[str, list[int]]] = {
        "lt50": {}, "m50": {}, "m70": {}, "ge90": {}, "lt70": {},
    }
    def _add(ch: str, resi: int, v: float) -> None:
        if v < 50:
            out["lt50"].setdefault(ch, []).append(resi)
            out["lt70"].setdefault(ch, []).append(resi)
        elif v < 70:
            out["m50"].setdefault(ch, []).append(resi)
            out["lt70"].setdefault(ch, []).append(resi)
        elif v < 90:
            out["m70"].setdefault(ch, []).append(resi)
        else:
            out["ge90"].setdefault(ch, []).append(resi)
    if fmt == "pdb":
        for ln in (txt or "").splitlines():
            if not ln.startswith("ATOM") or len(ln) < 66:
                continue
            if ln[12:16].strip() != "CA":
                continue
            ch = ln[21:22].strip() or "A"
            if only_chain and ch != only_chain:
                continue
            try: _add(ch, int(ln[22:26]), float(ln[60:66]))
            except ValueError: pass
        return out
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False)
        tmp.write(txt or ""); tmp.close(); tmp_path = tmp.name
        st_obj = gemmi.read_structure(tmp_path)
        if len(st_obj) > 0:
            for ch_obj in st_obj[0]:
                if not any(is_amino_acid(r) for r in ch_obj):
                    continue
                if only_chain and ch_obj.name != only_chain:
                    continue
                for res in ch_obj:
                    if not is_amino_acid(res):
                        continue
                    v = None
                    for at in res:
                        if at.name == "CA":
                            v = float(at.b_iso)
                            break
                    if v is None:
                        bs = [float(a.b_iso) for a in res
                              if 0 <= float(getattr(a, "b_iso", -1)) <= 100]
                        if bs: v = sum(bs) / len(bs)
                    try: resi = int(res.seqid.num)
                    except Exception: continue
                    if v is not None and 0 <= v <= 100:
                        _add(ch_obj.name, resi, v)
    except Exception:
        pass
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except OSError: pass
    return out


def plddt_bins(vals: list[float]) -> dict[str, int]:
    b = {n: 0 for n in PLDDT_SCHEME}
    for v in vals:
        if v < 50: b["Very low (<50)"] += 1
        elif v < 70: b["Low (50-70)"] += 1
        elif v < 90: b["High (70-90)"] += 1
        else: b["Very high (>90)"] += 1
    return b


def plddt_legend() -> str:
    items = "".join(
        f'<div style="display:flex;align-items:center;margin:6px 0">'
        f'<span style="width:14px;height:14px;background:{c["color"]};'
        f'border-radius:2px;margin-right:8px;display:inline-block"></span>'
        f'{n}</div>'
        for n, c in PLDDT_SCHEME.items()
    )
    return (
        '<div style="border-radius:10px;border:1px solid #dce1ea;'
        'max-width:360px"><div style="background:#e9f0ff;padding:10px 12px;'
        'font-weight:700;color:#163dff">Model Confidence</div>'
        f'<div style="padding:10px 12px;background:#fff">{items}</div></div>'
    )


def plddt_figures(txt: str, fmt: str, title: str = "pLDDT",
                  chain: Optional[str] = None):
    pc = plddt_by_chain(txt, fmt, chain)
    vals = [v for arr in pc.values() for v in arr if 0 <= v <= 100]
    counts = plddt_bins(vals)
    names = list(PLDDT_SCHEME.keys())
    colors = [PLDDT_SCHEME[n]["color"] for n in names]
    pie = px.pie(values=[counts[n] for n in names], names=names,
                 title=f"pLDDT — {title}")
    pie.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(colors=colors))
    pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    mean = float(np.mean(vals)) if vals else 0.0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=mean,
        number={"suffix": " pLDDT", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": C_VHIGH},
            "steps": [
                {"range": [0, 50], "color": C_VLOW},
                {"range": [50, 70], "color": C_LOW},
                {"range": [70, 90], "color": C_HIGH},
                {"range": [90, 100], "color": C_VHIGH},
            ],
            "threshold": {"line": {"color": "black", "width": 2},
                          "thickness": 0.75, "value": mean},
        },
        title={"text": "Average pLDDT"},
    ))
    gauge.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return pie, gauge


def show_plddt_panels(txt: str, fmt: str, label: str,
                      chain: Optional[str], prefix: str) -> None:
    if not txt:
        st.info("Empty structure data."); return
    if st.checkbox("Show pLDDT legend", True, key=f"{prefix}_leg"):
        st.markdown(plddt_legend(), unsafe_allow_html=True)
    title = Path(label).name if label else ""
    p, g = plddt_figures(txt, fmt, title, chain)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(p, use_container_width=True)
    with c2: st.plotly_chart(g, use_container_width=True)


def plddt_profile(path: str, fmt: str,
                  chain: Optional[str] = None
                  ) -> tuple[Optional[str], Optional[list[float]]]:
    try: txt = open(path).read()
    except Exception: return None, None
    pc = plddt_by_chain(txt, fmt, chain)
    if not pc: pc = plddt_by_chain(txt, fmt)
    if not pc: return None, None
    cid, vals = max(pc.items(), key=lambda kv: len(kv[1]))
    prof = [v for v in vals if 0 <= v <= 100]
    return (cid, prof) if prof else (None, None)