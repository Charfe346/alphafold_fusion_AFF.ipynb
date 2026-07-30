"""3-D molecular visualisation with py3Dmol and domain overlays."""

from __future__ import annotations
import hashlib, re
from typing import Any, Optional
import py3Dmol
from .config import (
    C_HIGH, C_LOW, C_SPEC_GE70, C_SPEC_GE90, C_SPEC_LT70,
    C_VHIGH, C_VLOW, DOMAIN_COLORS, PY3DMOL_CDN,
)
from .plddt import classify_plddt
from .structure import detect_fmt


def viewer_key(prefix: str, **kw) -> str:
    payload = "|".join(str(v) for v in kw.values())
    return f"{prefix}::{hashlib.md5(payload.encode()).hexdigest()}"


def patch_cdn(html: str) -> str:
    return re.sub(
        r'src="https?://[^"]*3Dmol[^"]*\.js"',
        f'src="{PY3DMOL_CDN}"', html or "", count=1,
    )


def render_3d(txt: str, fmt: str, style: str, scheme: str,
              mono: bool = False, chain: Optional[str] = None,
              dark: bool = False) -> str:
    v = py3Dmol.view(width=1000, height=650)
    v.addModel(txt, "cif" if fmt == "cif" else "pdb")
    v.setBackgroundColor("black" if dark else "white")
    sel_g: dict[str, Any] = {"chain": chain} if (mono and chain) else {}
    rep = (style or "Cartoon").lower()
    if rep not in ("cartoon", "stick", "sphere", "line", "surface"):
        rep = "cartoon"

    def _sty(sel: dict, color: str) -> None:
        m = {
            "cartoon": {"cartoon": {"color": color}},
            "stick":   {"stick": {"color": color, "radius": 0.3}},
            "sphere":  {"sphere": {"color": color, "radius": 1.0}},
            "line":    {"line": {"color": color}},
            "surface": {"surface": {"opacity": 0.85, "color": color}},
        }
        v.setStyle(sel, m[rep])

    _f = fmt if fmt in ("pdb", "cif") else detect_fmt(txt)
    cls = classify_plddt(txt, _f, chain if mono else None)
    has = any(cls.get(k) for k in ("lt50", "m50", "m70", "ge90"))

    eff = scheme
    if scheme in ("AlphaFold (4-color)", "Special (blue/orange)") and not has:
        eff = "pLDDT (B-factor)"

    if eff == "AlphaFold (4-color)":
        for k, c in [("lt50", C_VLOW), ("m50", C_LOW),
                      ("m70", C_HIGH), ("ge90", C_VHIGH)]:
            for ch_name, rl in (cls.get(k) or {}).items():
                if rl:
                    s = dict(sel_g); s["resi"] = rl
                    if not mono: s["chain"] = ch_name
                    _sty(s, c)
    elif eff == "Special (blue/orange)":
        for k, c in [("lt70", C_SPEC_LT70), ("m70", C_SPEC_GE70),
                      ("ge90", C_SPEC_GE90)]:
            for ch_name, rl in (cls.get(k) or {}).items():
                if rl:
                    s = dict(sel_g); s["resi"] = rl
                    if not mono: s["chain"] = ch_name
                    _sty(s, c)
    else:
        cs_map = {
            "pLDDT (B-factor)": {"prop": "b", "gradient": "roygb",
                                  "min": 0, "max": 100},
            "Spectrum": "spectrum",
            "Chain": "chain",
        }
        cs = cs_map.get(eff, {"prop": "b", "gradient": "roygb",
                               "min": 0, "max": 100})
        sty_dict = {"colorscheme": cs}
        if rep == "surface":
            v.setStyle(sel_g, {"surface": {"opacity": 0.85}})
        elif rep == "stick":
            v.setStyle(sel_g, {"stick": {**sty_dict, "radius": 0.3}})
        elif rep == "sphere":
            v.setStyle(sel_g, {"sphere": {**sty_dict, "radius": 1.0}})
        else:
            v.setStyle(sel_g, {rep: sty_dict})

    v.zoomTo(); v.render()
    return v._make_html()


def render_domains_3d(txt: str, fmt: str, segs: list[dict],
                      style: str = "Cartoon",
                      chain: Optional[str] = None) -> str:
    v = py3Dmol.view(width=1000, height=650)
    v.addModel(txt, "cif" if fmt == "cif" else "pdb")
    v.setStyle({}, {"cartoon": {"color": "#DDD"}})
    for i, s in enumerate(segs):
        c = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
        rng = list(range(int(s["start"]), int(s["end"]) + 1))
        sel: dict[str, Any] = {"resi": rng}
        if chain: sel["chain"] = chain
        style_map = {
            "Cartoon": {"cartoon": {"color": c}},
            "Stick":   {"stick": {"color": c, "radius": 0.3}},
            "Sphere":  {"sphere": {"color": c, "radius": 1.0}},
            "Line":    {"line": {"color": c}},
        }
        v.setStyle(sel, style_map.get(style, {"cartoon": {"color": c}}))
    v.setBackgroundColor("white"); v.zoomTo()
    return v._make_html()


def domain_legend(segs: list[dict]) -> str:
    items = "".join(
        f'<div style="display:flex;align-items:center;margin:4px 0">'
        f'<span style="width:14px;height:14px;background:'
        f'{DOMAIN_COLORS[i % len(DOMAIN_COLORS)]};border-radius:2px;'
        f'margin-right:8px;display:inline-block"></span>'
        f'{s.get("label", "?")} ({s["start"]}–{s["end"]})</div>'
        for i, s in enumerate(segs)
    )
    return (
        '<div style="border:1px solid #e5e7eb;border-radius:10px;'
        f'padding:10px;background:#fff">{items}</div>'
    )
def confidence_style(value, kind="plddt"):
    """Return (pct, quality, color) for a metric value.

    kind: 'plddt' (0-100), 'score' (0-1: pTM/ipTM/F1...),
          'mcc' (-1..1), 'fraction' (0-1).
    """
    if value is None:
        return None, None, "#64748B"
    v = float(value)
    if kind == "plddt":
        pct = max(0, min(100, v))
        if v >= 90:  return pct, "Very High", "#059669"
        if v >= 70:  return pct, "High",      "#2563EB"
        if v >= 50:  return pct, "Medium",    "#D97706"
        return pct, "Low", "#DC2626"
    if kind == "score":            # pTM, ipTM, F1, precision, recall
        pct = max(0, min(100, v * 100))
        if v >= 0.80: return pct, "High",   "#059669"
        if v >= 0.60: return pct, "Good",   "#2563EB"
        if v >= 0.40: return pct, "Fair",   "#D97706"
        return pct, "Low", "#DC2626"
    if kind == "mcc":              # -1..1
        pct = max(0, min(100, (v + 1) / 2 * 100))
        if v >= 0.50: return pct, "Strong",   "#059669"
        if v >= 0.30: return pct, "Moderate", "#2563EB"
        if v >= 0.10: return pct, "Weak",     "#D97706"
        return pct, "Poor", "#DC2626"
    # fraction (0-1) — neutral
    pct = max(0, min(100, v * 100))
    return pct, None, "#7C3AED"


def metric_card(label, value, unit="", pct=None,
                quality=None, color="#2563EB"):
    """Scientific metric card with optional confidence bar.

    Use directly, or pair with confidence_style() for auto colors:
        pct, q, c = confidence_style(0.84, "score")
        metric_card("pTM", "0.84", pct=pct, quality=q, color=c)
    """
    bar = ""
    if pct is not None:
        bar = (f'<div style="background:#E2E8F0;border-radius:6px;'
               f'height:8px;margin-top:8px;overflow:hidden">'
               f'<div style="background:{color};width:{pct}%;'
               f'height:100%;border-radius:6px"></div></div>')
    q = (f'<span style="float:right;font-size:.75rem;color:{color};'
         f'font-weight:600">{quality}</span>' if quality else "")
    return (f'<div style="background:#fff;border:1px solid #E2E8F0;'
            f'border-radius:12px;padding:16px 18px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,.06)">'
            f'<div style="font-size:.8rem;color:#64748B;'
            f'font-weight:500">{label}{q}</div>'
            f'<div style="font-size:1.8rem;font-weight:700;'
            f'color:#1E293B;margin-top:4px">{value}'
            f'<span style="font-size:1rem;color:#94A3B8">{unit}'
            f'</span></div>{bar}</div>')
def show_local_png(path, caption: str = "", width_pct: int = 100) -> bool:
    """Display a local PNG via base64 data-URI.

    Streamlit's st.image() serves files through an internal /media/
    endpoint that is NOT relayed behind the Colab proxyPort tunnel,
    producing broken images. Embedding the PNG as a base64 data-URI
    puts the bytes directly in the page, bypassing that endpoint.
    Returns True on success, False if the file is missing/empty.
    """
    import base64
    from pathlib import Path
    import streamlit as st
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return False
        b64 = base64.b64encode(p.read_bytes()).decode()
        html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{width_pct}%;height:auto;border-radius:8px" '
            f'alt="{caption}"/>'
        )
        if caption:
            html += (f'<div style="text-align:center;color:#666;'
                     f'font-size:0.85rem;margin-top:4px">{caption}</div>')
        st.markdown(html, unsafe_allow_html=True)
        return True
    except Exception:
        return False