"""Predictions page — input, parameter configuration, job launch.

FIX v2.1.1:
- 'or' changed to 'and' for complex assembly (both sides required)
- AFDB-first blocked for complexes (AFDB has monomers only)
- Added FASTA preview showing exact content sent to ColabFold
- Validation: complex must contain ':' before launch
- Per-side parsing feedback
"""

from __future__ import annotations
import hashlib, re, shutil, urllib.request
from pathlib import Path
import streamlit as st
from alphafold_fusion.config import (
    CACHE_DIR, MSA_LABELS, RECYCLE_PRESETS, RESULTS_DIR,
)
from alphafold_fusion.sequence import (
    is_complex, new_run_dir, parse_fasta, safe_basename, total_length,
)
from alphafold_fusion.api import fetch_afdb
from alphafold_fusion.structure import cif_to_pdb
from alphafold_fusion.runner import (
    analyze, fallback_cpu, has_gpu, quality_warnings, run_colabfold,
)


def render() -> None:
    ss = st.session_state
    st.markdown(
        '<div class="sub-header">Prediction Setup</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio("Mode", ["Monomer", "Multimer"], horizontal=True,
                    key="pmode")
    valid: list[tuple[str, str]] = []

    # ── Direct AFDB fetch by UniProt accession ──
    with st.expander("🔎 Load existing model from AlphaFold DB (by accession)"):
        acc_in = st.text_input(
            "UniProt accession (e.g. Q9UKV8):",
            key="afdb_acc_input").strip().upper()
        if st.button("Fetch from AFDB", key="afdb_fetch_btn"):
            from alphafold_fusion.api import fetch_afdb, is_uniprot_acc
            import urllib.request as _u
            if not is_uniprot_acc(acc_in):
                st.error(f"'{acc_in}' is not a valid UniProt accession.")
            else:
                with st.spinner(f"Querying AFDB for {acc_in}..."):
                    af = fetch_afdb(acc_in)
                if not af:
                    st.error(
                        f"No AlphaFold model found for {acc_in} in AFDB. "
                        f"The protein may be absent (too long / not modelled). "
                        f"You can still try ColabFold below.")
                else:
                    url = (af.get("cif_url") or af.get("pdb_url")
                           or af.get("bcif_url"))
                    run_root = new_run_dir(RESULTS_DIR)
                    ss["run_root"] = str(run_root)
                    od = run_root / f"AFDB_{acc_in}"
                    od.mkdir(parents=True, exist_ok=True)
                    loc = od / f"{acc_in}_AFDB{Path(url).suffix}"
                    try:
                        _u.urlretrieve(url, loc)
                        if loc.suffix.lower() in (".cif", ".bcif"):
                            cif_to_pdb(loc, loc.with_name(loc.stem + "_c.pdb"))
                        r = analyze(od)
                        if r["status"] == "success":
                            ss.setdefault("results", {})[acc_in] = r
                            ss.setdefault("job_dirs", {})[acc_in] = str(od)
                            ss.setdefault("_afdb_used", {})[acc_in] = af["acc"]
                            best = r["models"][0]
                            st.success(
                                f"✅ AFDB model loaded for {acc_in} "
                                f"(pLDDT ~ {best['avg_plddt'] or 0:.1f}, "
                                f"{r['models'][0].get('fmt','').upper()}). "
                                f"Go to Results / 3D Viewer / Analysis.")
                        else:
                            st.error("Downloaded file could not be parsed.")
                    except Exception as e:
                        st.error(f"Download failed: {e}")

    if mode == "Monomer":
        fa = st.text_area(
            "FASTA:", key="fasta_text", height=200,
            placeholder=">P00533\nMRPSGT...\n\n>MyProt\nMADYK...",
        )
        if fa.strip():
            for n, s in parse_fasta(fa):
                cs = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s.upper())
                if cs: valid.append((n, cs))
                else: st.warning(f"'{n}': empty after cleaning")
            if valid: st.success(f"{len(valid)} valid sequence(s)")
# ── Identify protein by BLAST (for headers without accession) ──
            from alphafold_fusion.api import extract_uniprot_acc
            no_acc = [(n, s) for n, s in valid
                      if not extract_uniprot_acc(n)]
            if no_acc:
                st.info(f"{len(no_acc)} sequence(s) have no UniProt "
                        f"accession in the header.")
                if st.button("🔍 Identify protein(s) by sequence (BLAST)",
                             key="blast_id"):
                    from alphafold_fusion.api import blast_identify
                    for n, s in no_acc:
                        with st.spinner(f"BLAST search for '{n}' "
                                        f"(~30-60s)..."):
                            hit = blast_identify(s)
                        if hit:
                            acc = hit["accession"]
                            ident = hit["identity_pct"]
                            ss.setdefault("_afdb_used", {})[n] = acc
                            if hit["is_exact"]:
                                st.success(
                                    f"✅ '{n}' = **{acc}** "
                                    f"({hit['description'][:50]}) "
                                    f"— {ident}% identity (exact match).")
                            else:
                                st.warning(
                                    f"⚠️ '{n}': closest match **{acc}** "
                                    f"({hit['description'][:50]}) "
                                    f"— {ident}% identity (homolog, "
                                    f"not identical).")
                        else:
                            st.error(f"'{n}': no BLAST hit found.")
    else:
        # ═══════════════════════════════════════
        # MULTIMER : Side A + Side B
        # ═══════════════════════════════════════
        cname = st.text_input("Complex name:", "MyComplex", key="cxname")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("#### Side A")
            fa_a = st.text_area(
                "Side A (FASTA):", key="fa_a", height=160,
                placeholder=">TNRC6A\nMKDAYPFEKL...",
            )
        with cB:
            st.markdown("#### Side B")
            fa_b = st.text_area(
                "Side B (FASTA):", key="fa_b", height=160,
                placeholder=">AGO2\nMYSGAGPAL...",
            )

        # ── Parse each side independently ──
        clean_a = []
        if fa_a.strip():
            for n, s in parse_fasta(fa_a.strip()):
                cs = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s.upper())
                if cs:
                    clean_a.append((n, cs))
                    st.caption(f"Side A parsed: **{n}** ({len(cs)} aa)")
            if not clean_a and fa_a.strip():
                st.error("Side A: text found but no valid amino acids extracted.")

        clean_b = []
        if fa_b.strip():
            for n, s in parse_fasta(fa_b.strip()):
                cs = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s.upper())
                if cs:
                    clean_b.append((n, cs))
                    st.caption(f"Side B parsed: **{n}** ({len(cs)} aa)")
            if not clean_b and fa_b.strip():
                st.error("Side B: text found but no valid amino acids extracted.")

        # ═══════════════════════════════════════
        # FIX 1 : 'and' instead of 'or'
        # Both sides MUST have sequences
        # ═══════════════════════════════════════
        if clean_a and clean_b:
            chains_a = [s for _, s in clean_a]
            chains_b = [s for _, s in clean_b]
            all_chains = chains_a + chains_b
            complex_seq = ":".join(all_chains)
            total = sum(len(c) for c in all_chains)

            # Verify colon is present
            if ":" not in complex_seq:
                st.error("Internal error: assembled sequence has no ':'")
            else:
                valid = [(cname.strip() or "Complex", complex_seq)]
                st.success(
                    f"**{len(clean_a)}** chain(s) Side A + "
                    f"**{len(clean_b)}** chain(s) Side B = "
                    f"**{len(all_chains)} chains** ({total} residues)")

                # ═══════════════════════════════════
                # FIX 4 : Show FASTA preview
                # ═══════════════════════════════════
                with st.expander("Preview: what ColabFold will receive"):
                    names_a = [n for n, _ in clean_a]
                    names_b = [n for n, _ in clean_b]
                    st.markdown(
                        f"- **Side A**: {', '.join(names_a)} "
                        f"({sum(len(s) for s in chains_a)} aa)\n"
                        f"- **Side B**: {', '.join(names_b)} "
                        f"({sum(len(s) for s in chains_b)} aa)\n"
                        f"- **Chain separator** `:` at positions: "
                        f"{[i for i, c in enumerate(complex_seq) if c == ':']}\n"
                        f"- **Model type**: alphafold2_multimer_v3\n"
                        f"- **Total**: {len(all_chains)} chains, {total} residues"
                    )
                    st.code(f">{safe_basename(cname or 'Complex', complex_seq)}\n"
                            f"{complex_seq[:60]}...:{complex_seq.split(':')[-1][:30]}..."
                            if len(complex_seq) > 90 else
                            f">{safe_basename(cname or 'Complex', complex_seq)}\n"
                            f"{complex_seq}")

                if total > 2000:
                    st.warning(f"Large complex ({total} residues). "
                               f"A100 GPU recommended.")

        elif fa_a.strip() or fa_b.strip():
            # One side has text but assembly failed
            if clean_a and not clean_b:
                st.error("**Side B is empty or invalid.** "
                         "Both sides are required for multimer prediction.")
            elif clean_b and not clean_a:
                st.error("**Side A is empty or invalid.** "
                         "Both sides are required for multimer prediction.")
            elif not clean_a and not clean_b:
                st.error("Neither side has valid sequences.")

    # ── Parameters ──
    t1, t2 = st.tabs(["Basic", "Advanced"])
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            quality = st.selectbox("Quality", list(RECYCLE_PRESETS), 0)
            msa_strat = st.selectbox("MSA", list(MSA_LABELS), 1)
        with c2:
            nmod = st.slider("Models", 1, 5, 3)
            nrec = st.slider("Min recycles", 1, 20, 6)
        with c3:
            use_tmpl = st.checkbox("PDB Templates", True)
            use_amber = st.checkbox("AMBER Relax", False)
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            mt_ui = st.selectbox("Model type",
                ["auto", "alphafold2_multimer_v3", "alphafold2_ptm"], 0)
            pm_ui = st.selectbox("Pairing",
                ["auto", "paired", "unpaired", "unpaired_paired"], 0)
        with c2:
            stop = st.number_input(
                "Stop at pLDDT (0=off)", 0.0, 100.0, 0.0, 1.0)
            purge = st.checkbox("Purge old results", False)

    strict = st.checkbox("Strict monomer (no pairing)", False)
    no_homo = st.checkbox("No known homologs (skip templates)", False)
    afdb_fast = st.checkbox(
        "AFDB-first (skip ColabFold if available)", False)
    cache = st.checkbox("Reuse cache", True)

    if st.button("Launch", type="primary", use_container_width=True):
        if not valid:
            st.error("No valid sequences. Check your input above.")
            return
# ══════ GPU backend check (warn on silent CPU fallback) ══════
        from alphafold_fusion.runner import jax_backend, has_gpu
        _backend = jax_backend()
        if _backend == "cpu":
            if has_gpu():
                st.error(
                    "⚠️ **GPU present but NOT visible to JAX!**\n\n"
                    "Predictions will run on **CPU** (10-50× slower, "
                    "**severely degraded quality** — pLDDT typically "
                    "<50 even for well-folded proteins).\n\n"
                    "**Fix:** Runtime → Restart session, then re-run the "
                    "installation cell (JAX GPU).\n\n"
                    "You may continue, but CPU results are **not "
                    "reliable**.")
            else:
                st.warning(
                    "⚠️ **No GPU detected** — running on CPU (slow, "
                    "degraded). To enable GPU: Runtime → Change runtime "
                    "type → GPU (T4).")
        elif _backend == "gpu":
            st.caption("✅ GPU active (JAX backend: gpu)")
        # ═══════════════════════════════════════════════════════════
        # ═══════════════════════════════════════
        # FIX 2 : Block launch if multimer has no ':'
        # ═══════════════════════════════════════
        if mode == "Multimer":
            seq_check = valid[0][1] if valid else ""
            if not is_complex(seq_check):
                st.error(
                    "**Cannot launch**: assembled sequence has no chain "
                    "separator. Both Side A and Side B must have valid "
                    "sequences.\n\n"
                    f"Current sequence starts with: `{seq_check[:40]}...`")
                return
            n_ch = seq_check.count(":") + 1
            st.info(f"Launching multimer: **{n_ch} chains**, "
                    f"**{total_length(seq_check)} residues**")

        if purge:
            shutil.rmtree(RESULTS_DIR, ignore_errors=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_root = new_run_dir(RESULTS_DIR)
        ss["run_root"] = str(run_root)
        ss.results = {}; ss.job_dirs = {}; ss.fasta_paths = {}
        prog = st.progress(0); info = st.empty()
        mkey = MSA_LABELS.get(msa_strat, "fast")
        _gpu = has_gpu()

        for i, (name, seq) in enumerate(valid):
            safe = safe_basename(name, seq)
            fa_path = str(Path("/tmp") / f"{safe}.fasta")

            # Write FASTA
            with open(fa_path, "w") as fout:
                fout.write(f">{safe}\n{seq}\n")

            cplx = is_complex(seq)
            slen = total_length(seq)

            # ═══════════════════════════════════════
            # FIX 3 : Verify FASTA content after write
            # ═══════════════════════════════════════
            with open(fa_path) as f:
                written = f.read()
            if cplx and ":" not in written:
                st.error(f"FASTA file for '{name}' has no ':' separator. "
                         f"Something went wrong during file writing.")
                continue

            st.caption(
                f"FASTA: complex={cplx} | "
                f"chains={seq.count(':') + 1 if cplx else 1} | "
                f"{slen} residues | "
                f"model={'multimer_v3' if cplx else 'ptm'}")

            # ═══════════════════════════════════════
            # FIX 5 : AFDB-first ONLY for monomers
            # AFDB has no multimer structures
            # ═══════════════════════════════════════
            if afdb_fast and cplx:
                st.info(f"AFDB-first skipped: '{name}' is a complex "
                        f"(AFDB has monomers only)")
            elif afdb_fast and not cplx:
                if try_afdb(name, seq, safe, run_root, fa_path, ss):
                    prog.progress((i + 1) / len(valid)); continue

            pr = RECYCLE_PRESETS.get(quality, RECYCLE_PRESETS["Classic"])
            nrec_eff = pr["complex"] if cplx else pr["monomer"]
            nrec_eff = max(nrec_eff, nrec)
            if cplx:
                mtype = "alphafold2_multimer_v3" if mt_ui == "auto" else mt_ui
                pmode = "paired" if pm_ui == "auto" else pm_ui
                if mkey == "minimal" and pmode == "paired":
                    pmode = "unpaired"
                # Warn if user forced monomer model on complex
                if mtype == "alphafold2_ptm":
                    st.warning(
                        f"'{mtype}' is a monomer model but input is a "
                        f"complex with {seq.count(':') + 1} chains. "
                        f"Switching to alphafold2_multimer_v3.")
                    mtype = "alphafold2_multimer_v3"
            else:
                mtype = "alphafold2_ptm" if mt_ui == "auto" else mt_ui
                pmode = ("unpaired" if strict
                         else ("unpaired_paired" if pm_ui == "auto" else pm_ui))

            params = {
                "num_models": nmod, "num_recycles": nrec_eff,
                "use_amber": use_amber,
                "use_templates": use_tmpl and not no_homo,
                "model_type": mtype, "pair_mode": pmode,
                "stop_at_score": stop if stop > 0 else None,
                "msa_strategy": mkey, "jobname_prefix": safe,
                "reuse_cache": cache,
            }

            info.text(
                f"Running: {name} | {mtype} | {mkey} | "
                f"{nrec_eff} recycles | {pmode}"
                + (f" | {seq.count(':') + 1} chains | {slen} aa"
                   if cplx else f" | {slen} aa"))

            sh = hashlib.sha1(seq.encode()).hexdigest()[:10]
            ck = "".join([safe, sh, mkey, mtype, pmode,
                          f"{nmod}m", f"{nrec_eff}r",
                          "t" if params["use_templates"] else "n"])
            od = CACHE_DIR / ck if cache else run_root / f"{safe}_{sh}"
            od.mkdir(parents=True, exist_ok=True)

            ok, run_log = run_colabfold(fa_path, str(od), params, slen)
            r = analyze(od)

            if r["status"] != "success":
                if not _gpu:
                    st.info("No GPU — trying CPU fallback...")
                _, run_log = fallback_cpu(fa_path, str(od), safe, cplx, slen)
                r = analyze(od)
                # AFDB fallback ONLY for monomers
                if r["status"] != "success" and not cplx:
                    _try_afdb_fallback(name, safe, od, r)
                if r["status"] != "success" and cplx:
                    st.error(
                        f"Complex '{name}' failed "
                        f"({slen} residues, {seq.count(':') + 1} chains).\n\n"
                        f"Try: fewer models, 'Fast' quality, or A100 GPU.")

            ss.results[name] = r
            ss.job_dirs[name] = str(od)
            ss.fasta_paths[name] = fa_path
            _report(name, r, run_log, job_dir=od)
            prog.progress((i + 1) / len(valid))
        st.success("Done — see Results / 3D Viewer")


def try_afdb(name, seq, safe, run_root, fa_path, ss) -> bool:
    try:
        af = fetch_afdb(name)
        if not af: return False
        url = af.get("pdb_url") or af.get("cif_url") or af.get("bcif_url")
        if not url: return False
        od = run_root / f"{safe}_{hashlib.sha1(seq.encode()).hexdigest()[:8]}"
        od.mkdir(parents=True, exist_ok=True)
        loc = od / (safe + "_AFDB" + Path(url).suffix)
        if not loc.exists(): urllib.request.urlretrieve(url, loc)
        ss.setdefault("_afdb_used", {})[name] = af["acc"]
        if loc.suffix.lower() in (".cif", ".bcif"):
            try: cif_to_pdb(loc, loc.with_name(loc.stem + "_c.pdb"))
            except Exception: pass
        from alphafold_fusion.runner import analyze
        r = analyze(od)
        if r["status"] == "success":
            ss.job_dirs[name] = str(od)
            ss.fasta_paths[name] = fa_path
            ss.results[name] = r
            st.success(f"AFDB structure used for {name}"); return True
    except Exception as e:
        st.info(f"AFDB: {e}")
    return False


def _try_afdb_fallback(name, safe, od, r):
    try:
        af2 = fetch_afdb(name)
        if af2:
            u2 = af2.get("pdb_url") or af2.get("cif_url")
            if u2:
                l2 = od / (safe + "_AFDB2" + Path(u2).suffix)
                if not l2.exists(): urllib.request.urlretrieve(u2, l2)
                if l2.suffix.lower() in (".cif", ".bcif"):
                    try: cif_to_pdb(l2, l2.with_name(l2.stem + "_c.pdb"))
                    except Exception: pass
                from alphafold_fusion.runner import analyze
                r.update(analyze(od))
    except Exception: pass


def _report(name, r, run_log, job_dir=None):
    if r["status"] == "success":
        best = sorted(
            r["models"],
            key=lambda m: (m["rank"] if m["rank"] is not None else 9999,
                           -(m["avg_plddt"] or 0)),
        )
        if best:
            b = best[0]
            st.success(f"{name}: {b['model_id']} | "
                       f"pLDDT~{b['avg_plddt'] or 0:.1f}")
            for w in quality_warnings(b): st.warning(w)
            # ══════ MSA empty check ══════
            if job_dir is not None:
                from alphafold_fusion.runner import (
                    count_msa_sequences, msa_warning)
                n_msa = count_msa_sequences(Path(job_dir))
                mw = msa_warning(n_msa, b.get("avg_plddt"))
                if mw:
                    st.error(mw)
                elif n_msa is not None:
                    st.caption(f"MSA depth: {n_msa} sequences")
        else:
            st.success(f"{name}: structure detected")
    else:
        tail = "\n".join((run_log or "").splitlines()[-20:])
        st.error(f"{name}: {r['message']}\n\n{tail}")