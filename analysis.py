"""Analysis page — integrated post-prediction structural analysis.

Four analysis modules grounded in published methods:

1. Disorder: Akdel et al. (2022) Nat Struct Mol Biol 29:1056
2. Inter-chain contacts: Duarte et al. (2012) BMC Bioinformatics 13:334
3. Ensemble variance: Kabsch (1976) + Wallner (2023)
4. Conservation: Shannon (1948) + Valdar (2002)

Each module reports continuous distributions where applicable.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from alphafold_fusion.config import log
from alphafold_fusion.plddt import plddt_by_chain
from alphafold_fusion.pae import pae_from_pkl
from alphafold_fusion.alignment import build_a3m_data
from alphafold_fusion.structure import polymer_chains


def render() -> None:
    ss = st.session_state
    st.markdown('<div class="sub-header">Post-Prediction Analysis</div>',
                unsafe_allow_html=True)

    if not ss.get("results"):
        st.info("No predictions available. Run predictions first.")
        return

    successful = {n: r for n, r in ss.results.items()
                  if r.get("status") == "success"}
    if not successful:
        st.warning("No successful predictions to analyse.")
        return

    seq_name = st.selectbox("Select prediction:", list(successful.keys()),
                            key="analysis_seq")
    result = successful[seq_name]
    models = result.get("models", [])
    if not models:
        st.warning("No models found.")
        return

    best = sorted(models, key=lambda m: (
        m["rank"] if m["rank"] is not None else 9999,
        -(m["avg_plddt"] or 0)))[0]
    try:
        txt = open(best["file"]).read()
    except Exception as e:
        st.error(f"Cannot read structure: {e}")
        return

    fm = "cif" if best["file"].lower().endswith((".cif", ".bcif")) else "pdb"
    chs = polymer_chains(best["file"])
    sel_chain = chs[0] if chs else "A"
    jd = Path(ss.get("job_dirs", {}).get(seq_name, ""))
    pc = plddt_by_chain(txt, fm, sel_chain)
    plddt_vals = [v for arr in pc.values() for v in arr]
    if plddt_vals:
        ss[f"_plddt_profile_{seq_name}"] = plddt_vals

    # ═══════════════════════════════════════════════════════════════
    # 1. INTRINSIC DISORDER
    # Akdel et al. (2022) Nat Struct Mol Biol 29:1056-1067
    # Piovesan et al. (2022) NAR 50:D471-D477 (DisProt guidelines)
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🌊 Intrinsic Disorder Regions")
    st.caption("pLDDT < 50 = AF2 very low confidence band (Jumper 2021; "
               "Tunyasuvunakool 2021); disorder principle Akdel et al. "
               "(2022) *NSMB* 29:1056 | DisProt: Piovesan et al. (2022) "
               "*NAR* 50:D471")

    if plddt_vals:
        from alphafold_fusion.disorder import (
            disorder_from_structure, to_gff3)

        dis_result = disorder_from_structure(txt, fm, sel_chain)
        for ch_id, dis in dis_result.items():
            st.markdown(f"**Chain {ch_id}**: "
                        f"{dis['fraction_disordered']:.1%} disordered, "
                        f"{dis['n_idr_regions']} region(s)")
            if dis["regions"]:
                st.dataframe(pd.DataFrame(dis["regions"]),
                             use_container_width=True, height=180)

                fig_dis = go.Figure()
                fig_dis.add_trace(go.Scatter(
                    x=list(range(1, dis["n_residues"] + 1)),
                    y=plddt_vals[:dis["n_residues"]],
                    mode="lines", name="pLDDT",
                    line=dict(color="#457b9d")))
                fig_dis.add_hline(
                    y=dis["threshold_used"], line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold {dis['threshold_used']} "
                                    f"(Akdel et al. 2022)")
                for r in dis["regions"]:
                    fig_dis.add_vrect(
                        x0=r["start"], x1=r["end"],
                        fillcolor="rgba(255,127,14,0.2)", line_width=0)
                fig_dis.update_layout(
                    title=f"Disorder — Chain {ch_id}",
                    xaxis_title="Residue", yaxis_title="pLDDT",
                    yaxis_range=[0, 100],
                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_dis, use_container_width=True)

                gff3 = to_gff3(dis, seqid=seq_name, chain=ch_id)
                st.download_button(
                    f"Download GFF3 (Chain {ch_id})", gff3,
                    f"{seq_name}_chain{ch_id}_disorder.gff3",
                    "text/plain", key=f"gff3_{ch_id}")

        ss[f"_disorder_{seq_name}"] = dis_result

    # ═══════════════════════════════════════════════════════════════
    # 2. INTER-CHAIN CONTACT MAP (multimers only)
    # Cα-Cα 8 Å contact map (Duarte et al. 2012 BMC Bioinformatics 13:334)
    # NOTE: contact map, NOT a SASA-based interface definition.
    # ═══════════════════════════════════════════════════════════════
    if len(chs) >= 2:
        st.markdown("🔗 Inter-Chain Contact Map")
        st.caption(
            "Cα-Cα contact map, cutoff 8 Å (Duarte et al. 2012 "
            "*BMC Bioinformatics* 13:334). "
            "This is a contact map, NOT a SASA-based interface.")

        # Load PAE matrix for inter-chain decomposition (optional)
        pae_mat = None
        if jd.exists():
            pae_mat = pae_from_pkl(jd, best["model_id"])

        from alphafold_fusion.interface_analysis import (
            compute_contacts, interchain_pae)

        ci, cj = st.columns(2)
        with ci: ch_a = st.selectbox("Chain A", chs, 0, key="iface_a")
        with cj: ch_b = st.selectbox("Chain B", chs,
                                     min(1, len(chs) - 1), key="iface_b")
        cutoff = st.slider(
            "Contact cutoff (Å) — Duarte et al. 2012 use 8.0",
            4.0, 12.0, 8.0, 0.5, key="iface_cut")

        with st.spinner("Computing contacts..."):
            iface = compute_contacts(txt, fm, ch_a, ch_b, cutoff)

        if "error" not in iface:
            from alphafold_fusion.render import metric_card, confidence_style
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(metric_card("Contacts",
                f"{iface['n_contacts']}", color="#0EA5E9"),
                unsafe_allow_html=True)
            _ip = iface['mean_plddt_interface'] or 0
            _p, _q, _c = confidence_style(_ip, "plddt")
            mc2.markdown(metric_card("Interface mean pLDDT",
                f"{_ip:.1f}", pct=_p, quality=_q, color=_c),
                unsafe_allow_html=True)
            mc3.markdown(metric_card("Contact residues",
                f"{iface['n_interface_a']}+{iface['n_interface_b']}",
                color="#7C3AED"),
                unsafe_allow_html=True)

            if iface["contacts"]:
                st.dataframe(pd.DataFrame(iface["contacts"][:200]),
                             use_container_width=True, height=250)

            if pae_mat is not None:
                chain_lens = []
                for ch_name in chs:
                    pc_ch = plddt_by_chain(txt, fm, ch_name)
                    if ch_name in pc_ch:
                        chain_lens.append(len(pc_ch[ch_name]))
                if chain_lens and sum(chain_lens) == pae_mat.shape[0]:
                    ic_pae = interchain_pae(pae_mat, chain_lens)
                    rows_ic = []
                    for k, v in ic_pae.items():
                        if v["is_interchain"]:
                            rows_ic.append({
                                "Pair": f"{v['chain_i']}→{v['chain_j']}",
                                "Mean PAE (Å)": v["mean_pae"],
                                "TM-kernel score": v["tm_kernel_score"],
                                "d₀ (Å)": v["d0_used"],
                            })
                    if rows_ic:
                        st.markdown("**Inter-chain PAE decomposition** "
                                    "(TM-kernel score = mean TM kernel over "
                                    "block; NOT AlphaFold ipTM):")
                        st.dataframe(pd.DataFrame(rows_ic),
                                     use_container_width=True)

            ss[f"_interface_{seq_name}"] = iface
        else:
            st.warning(iface["error"])

    # ═══════════════════════════════════════════════════════════════
    # 3. ENSEMBLE VARIANCE
    # Kabsch (1976) Acta Cryst A 32:922-923
    # Wallner (2023) Bioinformatics 39:btad573
    # ═══════════════════════════════════════════════════════════════
    if len(models) >= 2:
        st.markdown("### 🎯 Multi-Model Ensemble Variance")
        st.caption("Kabsch (1976) *Acta Cryst A* 32:922 | "
                   "Wallner (2023) *Bioinformatics* 39:btad573")

        from alphafold_fusion.ensemble_variance import compute_ensemble_rmsf

        with st.spinner("Superposing models and computing RMSF..."):
            ens = compute_ensemble_rmsf(models, sel_chain)

        if "error" not in ens:
            from alphafold_fusion.render import metric_card
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.markdown(metric_card("Mean RMSF",
                f"{ens['mean_rmsf']:.2f}", " Å", color="#7C3AED"),
                unsafe_allow_html=True)
            ec2.markdown(metric_card("Median RMSF",
                f"{ens['median_rmsf']:.2f}", " Å", color="#7C3AED"),
                unsafe_allow_html=True)
            ec3.markdown(metric_card("Std RMSF",
                f"{ens['std_rmsf']:.2f}", " Å", color="#8B5CF6"),
                unsafe_allow_html=True)
            ec4.markdown(metric_card("Models Used",
                f"{ens['n_models']}", color="#0EA5E9"),
                unsafe_allow_html=True)

            rmsf_vals = ens["rmsf"]
            fig_rmsf = go.Figure()
            fig_rmsf.add_trace(go.Scatter(
                x=list(range(1, len(rmsf_vals) + 1)), y=rmsf_vals,
                mode="lines", name="RMSF",
                line=dict(color="#2d6a4f")))
            fig_rmsf.update_layout(
                title="Per-Residue RMSF Across Models",
                xaxis_title="Residue", yaxis_title="RMSF (Å)",
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_rmsf, use_container_width=True)

            if ens.get("pairwise_rmsd"):
                st.markdown("**Pairwise RMSD:**")
                st.dataframe(pd.DataFrame(ens["pairwise_rmsd"]),
                             use_container_width=True, height=150)

            if plddt_vals and len(rmsf_vals) == len(plddt_vals):
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(
                    x=plddt_vals, y=rmsf_vals,
                    mode="markers",
                    marker=dict(size=3, opacity=0.4),
                    name="Residues"))
                fig_ov.update_layout(
                    title="pLDDT vs RMSF (complementary metrics, "
                          "Wallner 2023)",
                    xaxis_title="pLDDT", yaxis_title="RMSF (Å)",
                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_ov, use_container_width=True)

            ss[f"_ensemble_{seq_name}"] = ens
        else:
            st.info(ens["error"])

    # ═══════════════════════════════════════════════════════════════
    # 4. EVOLUTIONARY CONSERVATION (Henikoff-weighted)
    # Shannon (1948) Bell Syst Tech J 27:379
    # Valdar (2002) Proteins 48:227
    # Sequence weighting: Henikoff & Henikoff (1994) J Mol Biol 243:574
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🧬 Evolutionary Conservation")
    st.caption("Shannon (1948) *Bell Syst Tech J* 27:379 | "
               "Valdar (2002) *Proteins* 48:227 | "
               "Weighting: Henikoff & Henikoff (1994) *J Mol Biol* 243:574")

    aln_data = (ss.get("aln_import") or {}).get(f"{seq_name} (AUTO)")
    if aln_data is None and jd.exists():
        with st.spinner("Scanning for MSA..."):
            aln_data = build_a3m_data(seq_name, jd)
        if aln_data and aln_data.get("hits"):
            ss.setdefault("aln_import", {})[aln_data["name"]] = aln_data

    if aln_data and aln_data.get("hits") and aln_data.get("qseq"):
        from alphafold_fusion.conservation import conservation_profile

        aligned_seqs = [h.get("aln", "") for h in aln_data["hits"]
                        if h.get("aln")]
        if aligned_seqs:
            with st.spinner("Computing conservation..."):
                cons = conservation_profile(aln_data["qseq"], aligned_seqs)

            from alphafold_fusion.render import metric_card, confidence_style
            cc1, cc2, cc3 = st.columns(3)
            cc1.markdown(metric_card("MSA Depth",
                f"{cons['n_sequences']}", color="#0EA5E9"),
                unsafe_allow_html=True)
            cc2.markdown(metric_card("Effective Depth (Neff)",
                f"{cons.get('n_effective', 0):.1f}", color="#06B6D4"),
                unsafe_allow_html=True)
            _p, _q, _c = confidence_style(cons["mean_conservation"], "fraction")
            cc3.markdown(metric_card("Mean Conservation",
                f"{cons['mean_conservation']:.3f}", pct=_p, color="#7C3AED"),
                unsafe_allow_html=True)

            fig_cons = go.Figure()
            fig_cons.add_trace(go.Scatter(
                x=list(range(1, cons["n_positions"] + 1)),
                y=cons["conservation"],
                mode="lines", name="Conservation",
                line=dict(color="#6a0dad")))
            fig_cons.update_layout(
                title=f"Per-Residue Conservation "
                      f"(MSA depth = {cons['n_sequences']}, "
                      f"Neff = {cons.get('n_effective', 0):.1f})",
                xaxis_title="Residue",
                yaxis_title="Conservation C(i) = 1 - H(i)/H_max",
                yaxis_range=[0, 1.05],
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cons, use_container_width=True)

            ss[f"_conservation_{seq_name}"] = cons
    else:
        st.info("No MSA available. Use MMseqs2 MSA mode (not single_sequence) "
                "to enable conservation analysis.")

    # ═══════════════════════════════════════════════════════════════
    # REPORT EXPORT
    # FAIR: Wilkinson et al. (2016) Scientific Data 3:160018
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### Export Report")
    st.caption("FAIR principles: Wilkinson et al. (2016) *Sci Data* 3:160018")

    if st.button("Generate JSON + TSV Report", key="gen_report",
                 use_container_width=True):
        from alphafold_fusion.report import build_report, to_tsv

        fasta_path = ss.get("fasta_paths", {}).get(seq_name, "")
        seq_str = ""
        if fasta_path:
            try:
                lines = open(fasta_path).readlines()
                seq_str = "".join(
                    l.strip() for l in lines if not l.startswith(">"))
            except Exception:
                pass

        report = build_report(
            sequence_name=seq_name, sequence=seq_str,
            model_metrics=[{
                "model_id": m["model_id"], "rank": m["rank"],
                "avg_plddt": m["avg_plddt"], "ptm": m["ptm"],
                "iptm": m["iptm"], "fmt": m["fmt"],
            } for m in models],
            plddt_profile=ss.get(f"_plddt_profile_{seq_name}"),
            disorder=ss.get(f"_disorder_{seq_name}"),
            domains=ss.get(f"_domains_{seq_name}"),
            interface=ss.get(f"_interface_{seq_name}"),
            conservation=ss.get(f"_conservation_{seq_name}"),
            ensemble=ss.get(f"_ensemble_{seq_name}"),
        )

        report_json = json.dumps(report, indent=2, default=str)
        report_tsv = to_tsv(report)

        c_j, c_t = st.columns(2)
        with c_j:
            st.download_button("JSON Report", report_json,
                               f"{seq_name}_report.json",
                               "application/json",
                               use_container_width=True)
        with c_t:
            st.download_button("TSV (per-residue)", report_tsv,
                               f"{seq_name}_per_residue.tsv",
                               "text/tab-separated-values",
                               use_container_width=True)
        st.success("Report generated.")