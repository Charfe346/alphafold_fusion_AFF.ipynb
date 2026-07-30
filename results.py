"""Results page — model comparison table, inspection, alignment comparator."""

from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alphafold_fusion.alignment import build_a3m_data, find_a3m, parse_text_block
from alphafold_fusion.api import enrich_df_urls
from alphafold_fusion.pae import pae_from_pkl, pae_heatmap
from alphafold_fusion.plddt import plddt_profile, show_plddt_panels
from alphafold_fusion.render import patch_cdn, render_3d, show_local_png
from alphafold_fusion.structure import first_model_only, polymer_chains


def render() -> None:
    ss = st.session_state
    st.markdown('<div class="sub-header">Results</div>',
                unsafe_allow_html=True)
    vmode = ss.get("view_mode", "pLDDT (structures)")
    if vmode.startswith("pLDDT"): render_plddt_mode(ss)
    else: render_identity_mode(ss)


def render_plddt_mode(ss) -> None:
    if not ss.get("results"):
        st.info("No results yet. Run predictions first."); return
    st.markdown("### 🏆 Model Comparator")
    rows = []
    for n, r in ss.results.items():
        if r.get("status") != "success": continue
        for m in r["models"]:
            rows.append({
                "Sequence": n, "Model": m["model_id"],
                "Rank": m["rank"], "Avg_pLDDT": m["avg_plddt"],
                "PTM": m["ptm"], "ipTM": m["iptm"],
                "Fmt": m["fmt"].upper(), "File": m["file"],
            })
    if not rows: st.warning("No models found."); return

    # ══════ Best model summary cards ══════
    from alphafold_fusion.render import metric_card, confidence_style
    best_row = min(rows, key=lambda r: (
        r["Rank"] if r["Rank"] is not None else 9999,
        -(r["Avg_pLDDT"] or 0)))
    _model_label = (f" · {best_row['Model']}"
                    if best_row['Model'] not in ("-", None, "")
                    else " · AFDB")
    st.markdown(f"#### 🏆 Best: {best_row['Sequence']}{_model_label}")
    bc = st.columns(4)
    _p, _q, _c = confidence_style(best_row["Avg_pLDDT"], "plddt")
    bc[0].markdown(metric_card("Avg pLDDT",
        f"{best_row['Avg_pLDDT'] or 0:.1f}", pct=_p,
        quality=_q, color=_c), unsafe_allow_html=True)
    if best_row["PTM"] is not None:
        _p, _q, _c = confidence_style(best_row["PTM"], "score")
        bc[1].markdown(metric_card("pTM",
            f"{best_row['PTM']:.2f}", pct=_p,
            quality=_q, color=_c), unsafe_allow_html=True)
    else:
        bc[1].markdown(metric_card("pTM", "—", color="#94A3B8"),
                       unsafe_allow_html=True)
    if best_row["ipTM"] is not None:
        _p, _q, _c = confidence_style(best_row["ipTM"], "score")
        bc[2].markdown(metric_card("ipTM",
            f"{best_row['ipTM']:.2f}", pct=_p,
            quality=_q, color=_c), unsafe_allow_html=True)
    else:
        bc[2].markdown(metric_card("ipTM", "— (monomer)",
            color="#94A3B8"), unsafe_allow_html=True)
    bc[3].markdown(metric_card("Format",
        f"{best_row['Fmt']}", color="#0EA5E9"),
        unsafe_allow_html=True)
    st.write("")
    # ═══════════════════════════════════════════
    df = pd.DataFrame(rows).sort_values(
        ["Sequence", "Rank", "Avg_pLDDT"],
        ascending=[True, True, False], na_position="last")
    st.dataframe(df, use_container_width=True, height=300)

    st.markdown("### 👁️ Inspect Model")
    idx = st.selectbox(
        "Select:", range(len(df)),
        format_func=lambda i: (
            f"{df.iloc[i]['Sequence']} • {df.iloc[i]['Model']} • "
            f"pLDDT~{df.iloc[i]['Avg_pLDDT'] or 0:.1f}"),
        key="r_sel")
    rec = df.iloc[idx]; p = rec["File"]
    fm = "cif" if p.lower().endswith((".cif", ".bcif")) else "pdb"
    try: txt = open(p).read()
    except Exception as e: st.error(f"Error: {e}"); return

    cA, cB, cC, cD = st.columns(4)
    with cA:
        sty = st.selectbox("Style",
            ["Cartoon", "Stick", "Sphere", "Line", "Surface"], key="rs")
    with cB:
        sch = st.selectbox("Color",
            ["AlphaFold (4-color)", "Special (blue/orange)",
             "pLDDT (B-factor)", "Spectrum", "Chain"], key="rc")
    with cC: mono = st.checkbox("Monomer", True, key="rm")
    with cD: f1 = st.checkbox("1st model", True, key="rf")

    chs = polymer_chains(p)
    sc = st.selectbox("Chain", chs or ["A"], key="rch") if mono else None
    if f1: txt = first_model_only(fm, txt)
    html = patch_cdn(render_3d(txt, fm, sty, sch, mono, sc))
    st.components.v1.html(html, height=680)
    st.download_button("📥 Download", txt, Path(p).name,
        "chemical/x-mmcif" if fm == "cif" else "chemical/x-pdb",
        use_container_width=True)
    show_plddt_panels(txt, fm, p, sc if mono else None, "rp")

    # ── MSA coverage plot (ColabFold) ──
    _cov = (ss.results.get(rec["Sequence"]) or {}).get("coverage_png")
    if _cov and Path(_cov).exists():
        with st.expander("📊 MSA Coverage (ColabFold)", expanded=False):
            if not show_local_png(_cov, "Sequence coverage of the MSA "
                                  "(depth per residue position)."):
                st.info("Coverage image unavailable.")

    if st.button("Show top-3 pLDDT profiles & PAE", key="t3"):
        show_top3(rec["Sequence"], ss)


def show_top3(seq_name: str, ss) -> None:
    rs = ss.results.get(seq_name) or {}
    ms = sorted(rs.get("models") or [],
                key=lambda x: (x["rank"] if x["rank"] is not None else 9999,
                               -(x["avg_plddt"] or 0)))[:3]
    if not ms: st.info("No models to compare."); return
    fig = go.Figure()
    for mr in ms:
        mf = "cif" if mr["file"].lower().endswith((".cif", ".bcif")) else "pdb"
        _, prof = plddt_profile(mr["file"], mf)
        if prof:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(prof) + 1)), y=prof, mode="lines",
                name=f"{mr['model_id']} (r={mr['rank'] or '?'})"))
    if fig.data:
        fig.update_layout(title=f"Per-residue pLDDT — {seq_name}",
                          xaxis_title="Residue", yaxis_title="pLDDT",
                          yaxis_range=[0, 100],
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("pLDDT profiles unavailable.")

    jd = Path(ss.job_dirs.get(seq_name, ""))
    if not jd.exists(): return
    tabs = st.tabs([f"{j+1}. {mr['model_id']}" for j, mr in enumerate(ms)])
    for tab, mr in zip(tabs, ms):
        with tab:
            pm = pae_from_pkl(jd, mr["model_id"])
            if pm is not None:
                f_pae = pae_heatmap(pm, f"PAE — {mr['model_id']}")
                if f_pae: st.plotly_chart(f_pae, use_container_width=True)
            else:
                pp = rs.get("pae_png")
                if not (pp and show_local_png(
                        pp, f"PAE — {mr['model_id']} (ColabFold)")):
                    st.info("No PAE available.")


def render_identity_mode(ss) -> None:
    st.markdown("### 🧬 Alignment Comparator")
    with st.expander("📥 Manual import", expanded=not ss.get("aln_import")):
        ti = st.text_area("Paste block:", height=140, key="ti")
        if st.button("Analyze", key="ab", use_container_width=True):
            d = parse_text_block(ti)
            if d and d.get("hits"):
                ss.setdefault("aln_import", {})[d["name"]] = d
                st.success(f"✅ {d['name']}: {len(d['hits'])} hits")
            else: st.error("Unrecognized format.")

    with st.expander("⚙️ Auto import (.a3m)"):
        jds = list(ss.get("job_dirs", {}).keys())
        if jds:
            sel = st.selectbox("Sequence:", jds, key="as_sel")
            if st.button("Build from .a3m", key="ba",
                         use_container_width=True):
                jd = Path(ss.job_dirs.get(sel, ""))
                if jd.exists():
                    is_afdb = sel in ss.get("_afdb_used", {})
                    if is_afdb:
                        st.info(
                            "ℹ️ **This structure was fetched from AlphaFold DB** "
                            "— no MSA was generated.\n\n"
                            "To get alignment data, re-run this protein "
                            "with **AFDB-first unchecked** so ColabFold "
                            "runs a full MSA search."
                        )
                    else:
                        with st.spinner("Scanning for .a3m files..."):
                            ds = build_a3m_data(sel, jd)
                        if ds and ds.get("hits"):
                            ss.setdefault("aln_import", {})[ds["name"]] = ds
                            from alphafold_fusion.alignment import find_a3m
                            a3m_files = find_a3m(jd)
                            if a3m_files:
                                try:
                                    from alphafold_fusion.alignment import _read_a3m
                                    raw_a3m = _read_a3m(a3m_files[0])
                                    ss.setdefault("_a3m_raw", {})[sel] = raw_a3m
                                except Exception:
                                    pass
                            st.success(
                                f"✅ {ds['name']}: {len(ds['hits'])} hits")
                            try: st.rerun()
                            except Exception: pass
                        else:
                            a3m_files = find_a3m(jd)
                            if a3m_files:
                                st.warning(
                                    f"⚠️ Found {len(a3m_files)} .a3m file(s) "
                                    f"but could not extract hits.\n\n"
                                    f"Files: {', '.join(f.name for f in a3m_files[:5])}\n\n"
                                    f"The .a3m may contain only the query "
                                    f"sequence (no homologs found by MMseqs2)."
                                )
                            else:
                                try:
                                    all_f = [f for f in jd.rglob("*")
                                             if f.is_file()]
                                    exts = sorted(set(
                                        f.suffix for f in all_f)) or ["(none)"]
                                except Exception:
                                    all_f = []; exts = ["?"]
                                st.warning(
                                    f"❌ No .a3m files in `{jd.name}/`\n\n"
                                    f"**{len(all_f)} file(s)** found — "
                                    f"extensions: {', '.join(exts)}\n\n"
                                    f"This can happen if:\n"
                                    f"- The MSA search was skipped "
                                    f"(single_sequence mode)\n"
                                    f"- ColabFold stored MSA data in .pkl "
                                    f"format only\n"
                                    f"- The prediction failed before MSA "
                                    f"generation"
                                )
                else:
                    st.error(f"Directory not found: `{jd}`")
        else:
            st.info("No jobs available.")

    rows = []
    for n, d in (ss.get("aln_import") or {}).items():
        for h in d.get("hits", []):
            rows.append({
                "Sequence": n, "Accession": h.get("acc"),
                "Identity%": h.get("Identity_pct"),
                "Score": h.get("Score"), "Evalue": h.get("Evalue"),
                "Cov%": h.get("Coverage_pct"), "Core": h.get("Core"),
            })
    dfh = (pd.DataFrame(rows) if rows
           else pd.DataFrame(columns=["Sequence", "Accession", "Identity%",
                                      "Score", "Evalue", "Cov%", "Core"]))
    if dfh.empty: st.info("No alignments loaded yet."); return

    c1, c2 = st.columns(2)
    with c1: nrows = st.slider("Top N", 50, 2000, min(300, len(dfh)), 50)
    with c2: sort = st.selectbox("Sort",
        ["Identity%", "Score", "Evalue", "Cov%"], 0)
    asc = (sort == "Evalue")
    dfv = (dfh.sort_values([sort, "Identity%"],
                           ascending=[asc, False], na_position="last")
           .head(nrows).reset_index(drop=True))

    # Download raw .a3m alignment
    a3m_cache = ss.get("_a3m_raw", {})
    if a3m_cache:
        for name, raw in a3m_cache.items():
            st.download_button(
                f"📥 Download MSA (.a3m) — {name}",
                raw, f"{name}_msa.a3m", "text/plain",
                key=f"dl_a3m_{name}")
    if st.checkbox("Enrich UniProt/AFDB URLs", False):
        dfv = enrich_df_urls(dfv)
    try:
        st.dataframe(dfv, use_container_width=True, height=340,
                     column_config={
                         "UniProt_URL": st.column_config.LinkColumn("UniProt"),
                         "AFDB_URL": st.column_config.LinkColumn("AFDB"),
                     })
    except Exception:
        st.dataframe(dfv, use_container_width=True, height=340)