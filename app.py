"""AlphaFold Fusion — Streamlit entry point."""

from __future__ import annotations
import streamlit as st
from alphafold_fusion import __version__
from alphafold_fusion.config import CSS, RESULTS_DIR, CACHE_DIR
from alphafold_fusion.pages import PAGES, VIEW_MODES
from alphafold_fusion.pages import (
    home, predictions, results, viewer, settings, analysis, validation)

st.set_page_config(page_title="AlphaFold Fusion", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="main-header">AlphaFold Fusion</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center;color:#666;margin-bottom:1rem">'
    f'v{__version__} — Multi-protein structure prediction and analysis</p>',
    unsafe_allow_html=True)

for _k, _v in {"results": {}, "job_dirs": {}, "fasta_paths": {},
                "fasta_text": "", "aln_import": {}, "_afdb_used": {},
                "_blast_acc": {}}.items():
    st.session_state.setdefault(_k, _v)
with st.sidebar:
    # ══════ System status badge ══════
    import subprocess as _sp
    try:
        _b = _sp.run([__import__("sys").executable, "-c",
            "import jax; print(jax.default_backend())"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        _b = "unknown"
    _dot = "#059669" if _b == "gpu" else "#DC2626"
    _txt = "GPU active" if _b == "gpu" else f"CPU ({_b})"
    _n = len([n for n, r in st.session_state.get("results", {}).items()
              if (r or {}).get("status") == "success"])
    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
        f'border-radius:10px;padding:10px 12px;margin-bottom:12px">'
        f'<div style="font-size:.7rem;color:#94A3B8;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.5px">System</div>'
        f'<div style="margin-top:6px;font-size:.85rem;color:#334155">'
        f'<span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{_dot};margin-right:6px"></span>'
        f'{_txt}</div>'
        f'<div style="font-size:.85rem;color:#334155;margin-top:4px">'
        f'<span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:#2563EB;margin-right:6px"></span>'
        f'{_n} prediction(s)</div></div>',
        unsafe_allow_html=True)
    # ═══════════════════════════════════
with st.sidebar:
    st.markdown("## Navigation")
    st.radio("Page", PAGES,
             index=PAGES.index(st.session_state.get("nav_page", PAGES[0])),
             key="nav_page")
    st.markdown("## View mode")
    st.radio("Mode", VIEW_MODES,
             index=(0 if st.session_state.get(
                 "view_mode", VIEW_MODES[0]).startswith("pLDDT") else 1),
             key="view_mode")

_DISPATCH = {
    PAGES[0]: home.render,
    PAGES[1]: predictions.render,
    PAGES[2]: results.render,
    PAGES[3]: viewer.render,
    PAGES[4]: analysis.render,
    PAGES[5]: validation.render,
    PAGES[6]: settings.render,
}
page = st.session_state.get("nav_page", PAGES[0])
_DISPATCH.get(page, home.render)()

st.markdown("---")
st.markdown(
    f'<div style="text-align:center;color:#888;padding:.5rem">'
    f'AlphaFold Fusion v{__version__}</div>',
    unsafe_allow_html=True)