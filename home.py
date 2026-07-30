"""Home page — about information."""

import streamlit as st


def _metric_card(label, value, unit="", pct=None,
                 quality=None, color="#2563EB"):
    """Scientific metric card with optional confidence bar."""
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


def render() -> None:
    st.markdown(
        '<div class="card"><b>🎯 About</b><br/>'
        'AlphaFold-faithful predictions via ColabFold with rich '
        'visualization: pLDDT coloring, PAE heatmaps, UniProt/InterPro '
        'domain overlays, batch multi-protein analysis, and AFDB-first '
        'optimization. Organism-agnostic.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card"><b>📋 Input</b><br/>'
        '<b>Monomer</b>: FASTA sequences (1 protein = 1 prediction).<br/>'
        '<b>Multimer</b>: Side A + Side B proteins forming a complex.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card" style="border-left-color:#e63946">'
        '<b>⚠️ Security &amp; Data Privacy</b><br/>'
        'This tool has no strong authentication. Sequences you submit '
        'are sent to <b>third-party services</b> (ColabFold MMseqs2 '
        'servers, EBI APIs for UniProt/InterPro/BLAST). '
        '<b>Do not submit confidential or proprietary sequences.</b><br/>'
        'When running via a public tunnel, do not share the URL openly, '
        'and stop the session when finished.'
        '</div>',
        unsafe_allow_html=True,
    )