"""3-D Viewer page — interactive exploration with domain overlays.

Domain sources: UniProt features and InterPro entries (database
annotations mapped onto the predicted structure).

Disorder overlay: Akdel et al. (2022) Nat Struct Mol Biol 29:1056.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import streamlit as st
from alphafold_fusion.api import (
    fetch_interpro, fetch_uniprot, guess_acc,
    interpro_domains, uniprot_domains,
)
from alphafold_fusion.config import log
from alphafold_fusion.pae import pae_from_pkl, pae_heatmap
from alphafold_fusion.plddt import plddt_by_chain, show_plddt_panels
from alphafold_fusion.render import (
    domain_legend, patch_cdn, render_3d,
    render_domains_3d, viewer_key,
)
from alphafold_fusion.runner import model_name_for_path
from alphafold_fusion.structure import first_model_only, polymer_chains


def render() -> None:
    ss = st.session_state
    st.markdown('<div class="sub-header">3D Viewer</div>',
                unsafe_allow_html=True)
    if not ss.get("results"):
        st.info("No structures. Run predictions first."); return

    choices: list[tuple[str, dict]] = []
    for n, r in ss.results.items():
        if r.get("status") != "success": continue
        for m in r["models"]:
            label = (f"{n} | {m['model_id']} | "
                     f"pLDDT {m['avg_plddt'] or 0:.1f}")
            choices.append((label, m))
    if not choices: st.warning("No PDB/CIF found."); return

    idx = st.selectbox("Model:", range(len(choices)),
                       format_func=lambda i: choices[i][0], key="v_sel")
    m = choices[idx][1]; p = m["file"]
    fm = "cif" if p.lower().endswith((".cif", ".bcif")) else "pdb"
    try:
        with open(p) as f:
            txt = f.read()
    except Exception as e:
        st.error(f"Cannot read structure file: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sty = st.selectbox("Style",
            ["Cartoon", "Stick", "Sphere", "Line", "Surface"], key="vs")
    with c2:
        sch = st.selectbox("Color",
            ["AlphaFold (4-color)", "Special (blue/orange)",
             "pLDDT (B-factor)", "Spectrum", "Chain"], key="vc")
    with c3: mono = st.checkbox("Monomer", True, key="vm")
    with c4: f1 = st.checkbox("1st model", True, key="vf")

    chs = polymer_chains(p)
    sc = st.selectbox("Chain", chs or ["A"], key="vch") if mono else None
    if f1: txt = first_model_only(fm, txt)

    html = patch_cdn(render_3d(txt, fm, sty, sch, mono, sc))
    stamp = viewer_key("v", p=p, s=sty, c=sch, m=mono, ch=sc)
    st.components.v1.html(f"<!-- {stamp} -->\n{html}", height=680)
    st.download_button("Download structure", txt, Path(p).name,
        "chemical/x-mmcif" if fm == "cif" else "chemical/x-pdb",
        use_container_width=True)
    show_plddt_panels(txt, fm, p, sc if mono else None, "vp")

    _render_domains(txt, fm, sty, mono, sc, p, ss)
    _render_disorder_overlay(txt, fm, mono, sc, ss)
    _render_pae(p, ss)


def _render_domains(txt, fm, sty, mono, sc, path, ss) -> None:
    st.subheader("Domain Annotation")
    all_predicted = [n for n, r in (ss.results or {}).items()
                     if (r or {}).get("status") == "success"]
    sg = model_name_for_path(path, ss.results) or ""
    if all_predicted and len(all_predicted) > 1:
        dp = st.selectbox("Protein:", all_predicted,
                          index=(all_predicted.index(sg)
                                 if sg in all_predicted else 0),
                          key="vdp")
    else:
        dp = sg or (all_predicted[0] if all_predicted else "")

    afdb_used = ss.get("_afdb_used", {})
    aa = guess_acc(dp, afdb_used) if dp else None

    cS, cA = st.columns([1, 3])
    with cS:
        src = st.radio("Source", ["UniProt", "InterPro"],
                       horizontal=True, key="vds")
    with cA:
        ua = st.text_input("UniProt Accession", value=aa or "", key="vua")
        auto = st.checkbox("Auto-generate", True, key="vag")

    do_gen = st.button("Generate", key="vgd")
    doit = do_gen or (auto and (ua or aa))

    if doit:
        acc = ua or aa
        if not acc:
            st.info("Enter UniProt Accession manually."); return
        if src == "UniProt":
            with st.spinner(f"Fetching UniProt domains for {acc}..."):
                j = fetch_uniprot(acc)
                segs = uniprot_domains(j) if j else []
        else:
            with st.spinner(f"Fetching InterPro domains for {acc}..."):
                j = fetch_interpro(acc)
                segs = interpro_domains(j) if j else []
        ss[f"_domains_{dp}"] = segs
        if segs:
            hd = patch_cdn(render_domains_3d(
                txt, fm, segs, sty, sc if mono else None))
            st.components.v1.html(hd, height=680)
            st.markdown(domain_legend(segs), unsafe_allow_html=True)
        else:
            st.info(f"No domain found via {src} ({acc}).")

def _render_disorder_overlay(txt, fm, mono, sc, ss) -> None:
    """Disorder overlay: pLDDT < 50 (AF2 very low confidence band,
    Jumper et al. 2021; Tunyasuvunakool et al. 2021). Disorder
    principle: Akdel et al. (2022) NSMB 29:1056."""
    with st.expander("Disorder Overlay (pLDDT < 50 = AF2 very low "
                     "confidence; disorder principle: Akdel et al. 2022)"):
        from alphafold_fusion.disorder import disorder_from_structure
        import py3Dmol

        dis = disorder_from_structure(txt, fm, sc)
        if not any(d.get("regions") for d in dis.values()):
            st.info("No disordered regions detected (all pLDDT >= 50).")
            return

        for ch_id, d in dis.items():
            if d.get("regions"):
                st.markdown(f"**Chain {ch_id}**: "
                            f"{d['fraction_disordered']:.1%} disordered")

        v = py3Dmol.view(width=1000, height=650)
        v.addModel(txt, "cif" if fm == "cif" else "pdb")
        v.setStyle({}, {"cartoon": {"color": "#BBBBBB"}})
        pc = plddt_by_chain(txt, fm, sc)
        for ch_id, vals in pc.items():
            ordered = [i + 1 for i, val in enumerate(vals) if val >= 50]
            if ordered:
                sel = {"resi": ordered}
                if mono and sc: sel["chain"] = sc
                v.setStyle(sel, {"cartoon": {"color": "#4361ee"}})
        for ch_id, d in dis.items():
            for r in d.get("regions", []):
                rng = list(range(r["start"], r["end"] + 1))
                sel = {"resi": rng}
                if mono and sc: sel["chain"] = sc
                v.setStyle(sel, {"cartoon": {"color": "#FF8C42",
                                             "opacity": 0.85}})
        v.setBackgroundColor("white"); v.zoomTo()
        st.components.v1.html(patch_cdn(v._make_html()), height=680)
def _render_pae(path, ss) -> None:
    sn = model_name_for_path(path, ss.results)
    if not sn: return
    rs = ss.results.get(sn) or {}
    pj = rs.get("pae_json"); pp = rs.get("pae_png")
    if not (pj or pp): return
    with st.expander("PAE Heatmap"):
        if pj:
            f_pae = pae_heatmap(pj, f"PAE — {sn}")
            if f_pae: st.plotly_chart(f_pae, use_container_width=True)
            else: st.info("PAE JSON unreadable.")
        elif pp:
            st.image(pp, use_container_width=True)
