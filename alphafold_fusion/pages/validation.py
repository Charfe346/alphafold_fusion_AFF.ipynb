"""Validation page — compare AFF disorder (pLDDT<50) with DisProt.

DisProt: Piovesan et al. (2022) NAR 50:D471.
pLDDT<50 baseline: Akdel et al. (2022) NSMB 29:1056.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from alphafold_fusion.api import guess_acc
from alphafold_fusion.plddt import plddt_by_chain
from alphafold_fusion.structure import polymer_chains
from alphafold_fusion.disprot import (
    fetch_disprot, disprot_regions, compare_plddt_vs_disprot)


def render() -> None:
    ss = st.session_state
    st.markdown('<div class="sub-header">DisProt Validation</div>',
                unsafe_allow_html=True)
    st.caption("Compare AFF disorder (pLDDT < 50 = AF2 very low confidence "
               "band; disorder principle Akdel et al. 2022 *NSMB* 29:1056) "
               "against experimental DisProt annotations "
               "(Piovesan et al. 2022 *NAR* 50:D471).")

    if not ss.get("results"):
        st.info("No structures loaded. Load a protein first.")
        return

    successful = {n: r for n, r in ss.results.items()
                  if r.get("status") == "success"}
    if not successful:
        st.warning("No successful predictions.")
        return

    seq_name = st.selectbox("Protein:", list(successful.keys()),
                            key="val_seq")
    result = successful[seq_name]
    models = result.get("models", [])
    if not models:
        st.warning("No models.")
        return

    best = sorted(models, key=lambda m: (
        m["rank"] if m["rank"] is not None else 9999,
        -(m["avg_plddt"] or 0)))[0]
    fm = "cif" if best["file"].lower().endswith((".cif", ".bcif")) else "pdb"
    try:
        txt = open(best["file"]).read()
    except Exception as e:
        st.error(f"Cannot read: {e}")
        return

    chs = polymer_chains(best["file"])
    sel_chain = chs[0] if chs else "A"
    pc = plddt_by_chain(txt, fm, sel_chain)
    plddt_vals = [v for arr in pc.values() for v in arr]
    if not plddt_vals:
        st.warning("No pLDDT extracted.")
        return

    afdb_used = ss.get("_afdb_used", {})
    guess = guess_acc(seq_name, afdb_used) or ""
    acc = st.text_input("UniProt accession (for DisProt lookup):",
                        value=guess, key="val_acc").strip().upper()
    thr = st.slider("pLDDT threshold", 30.0, 70.0, 50.0, 1.0,
                    key="val_thr")

    if st.button("Compare with DisProt", key="val_go",
                 use_container_width=True):
        if not acc:
            st.error("Enter a UniProt accession.")
            return
        with st.spinner(f"Querying DisProt for {acc}..."):
            entry = fetch_disprot(acc)
        if not entry:
            st.warning(
                f"'{acc}' not found in DisProt. This protein has no "
                f"experimental disorder annotation. Note: absence from "
                f"DisProt does NOT mean the protein is ordered — it means "
                f"it has not been experimentally characterised.")
            return

        regs = disprot_regions(entry)
        if not regs:
            st.warning(
                f"DisProt entry {entry.get('disprot_id', '?')} has **0 "
                f"consensus 'pure disorder' (type D) regions**. "
                f"The protein may still carry other annotations "
                f"(e.g. disorder-to-order transitions, binding regions) "
                f"that are not counted here. This is why MCC = 0 for "
                f"this entry: there is no pure-disorder ground truth to "
                f"compare against.")
        else:
            st.success(f"DisProt entry {entry.get('disprot_id', '?')} — "
                       f"{len(regs)} consensus disordered region(s), "
                       f"length {entry.get('length', '?')}.")

        comp = compare_plddt_vs_disprot(plddt_vals, regs, thr)
        if "error" in comp:
            st.error(comp["error"]); return

        from alphafold_fusion.render import metric_card, confidence_style
        c1, c2, c3, c4 = st.columns(4)
        for col, name, val, kind in [
            (c1, "MCC", comp["MCC"], "mcc"),
            (c2, "F1", comp["F1"], "score"),
            (c3, "Precision", comp["precision"], "score"),
            (c4, "Recall", comp["recall"], "score"),
        ]:
            pct, q, color = confidence_style(val, kind)
            col.markdown(
                metric_card(name, f"{val:.3f}", pct=pct,
                            quality=q, color=color),
                unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        cc1.markdown(metric_card(
            "DisProt disordered residues",
            f"{comp['disprot_disordered']}", color="#0EA5E9"),
            unsafe_allow_html=True)
        cc2.markdown(metric_card(
            f"AFF disordered (pLDDT<{thr:.0f})",
            f"{comp['aff_disordered']}", color="#7C3AED"),
            unsafe_allow_html=True)

        # Overlay figure
        truth = comp["truth_per_residue"]
        n = comp["n_residues"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=plddt_vals,
            mode="lines", name="pLDDT",
            line=dict(color="#457b9d")))
        fig.add_hline(y=thr, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold {thr:.0f}")
        # DisProt regions shaded
        in_reg = False
        start = 0
        for i in range(n + 1):
            d = truth[i] if i < n else 0
            if d and not in_reg:
                in_reg = True; start = i + 1
            elif not d and in_reg:
                in_reg = False
                fig.add_vrect(x0=start, x1=i,
                              fillcolor="rgba(230,57,70,0.18)",
                              line_width=0,
                              annotation_text="DisProt",
                              annotation_position="top left")
        fig.update_layout(
            title=f"AFF pLDDT vs DisProt disorder — {acc}",
            xaxis_title="Residue", yaxis_title="pLDDT",
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Red shading = experimentally confirmed disorder (DisProt). "
            "Regions below the red line = AFF-predicted disorder. "
            "Only residues present in the structure are compared. "
            "DisProt annotates confirmed disorder only; unshaded regions "
            "are not necessarily ordered — they may be uncharacterised.")

        ss[f"_disprot_comp_{seq_name}"] = {
            k: v for k, v in comp.items()
            if k not in ("truth_per_residue", "pred_per_residue")}