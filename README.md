# 🧬 AlphaFold Fusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/alphafold_fusion_AFF3.ipynb)

> ⚠️ Replace `YOUR-USERNAME/YOUR-REPO` above (and in `CITATION.cff`) with your
> actual GitHub path once this repo is pushed, so the Colab badge resolves.

**Integrated protein structure prediction & analysis platform**, built on
ColabFold/AlphaFold2 and wrapped in a Streamlit interface. Predict a
structure (or load one from the AlphaFold DB), then explore pLDDT
confidence, PAE, intrinsic disorder, evolutionary conservation, inter-chain
contacts, and cross-model variance — all from one dashboard.

## Features

- **Predicts** structures with ColabFold (AlphaFold2, monomer & multimer)
- **Analyses**: pLDDT confidence, Predicted Aligned Error (PAE), intrinsic
  disorder, evolutionary conservation from the MSA, inter-chain contacts,
  ensemble RMSF across recycles
- **Annotates**: UniProt / InterPro domains
- **Validates**: disorder predictions against DisProt experimental ground
  truth (current benchmark: AUC-ROC = 0.80)
- **Exports**: FAIR-compliant JSON/TSV reports
- Interactive 3D structure viewer (py3Dmol)

## Quick start — Google Colab (recommended)

This app was built and validated to run in Google Colab, which provides a
free GPU for the folding step.

1. Open the notebook via the **Open in Colab** badge above
2. `Runtime → Change runtime type → GPU (T4) → Save`
3. `Runtime → Run all` (~10 min on first run — installs JAX, ColabFold, and
   dependencies)
4. Wait for the **public (Cloudflare) link** in the last cell, then open it

If the page doesn't load immediately, wait ~20s and refresh (the tunnel is
initializing). Keep the last cell running while testing.

**Example sequences to try:**

| Type | Input | Time |
|---|---|---|
| Monomer | Ubiquitin — see `examples/ubiquitin_monomer.fasta` | ~5 min |
| Multimer | Two chains separated by `:` (e.g. Barnase + Barstar) | ~15 min |
| Load from AFDB | Accession `P00698` (lysozyme) or `Q9UKV8` (Argonaute 2) | instant |

## Running locally

### 1. Clone and install the base dependencies

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs everything needed for the UI, the 3D viewer, and every
analysis page (domain annotation, conservation, disorder, interface
analysis, DisProt validation) in **browse existing results** mode.

### 2. Structure prediction engine (GPU only, optional)

To run **new** folding jobs — not just browse existing results — you need
JAX and ColabFold. These are hardware-specific (CUDA build of JAX), which is
why they're kept out of `requirements.txt` and installed separately:

```bash
# JAX — pick ONE:
pip install "jax[cuda12]==0.4.38"   # NVIDIA GPU (needs CUDA 12)
# pip install "jax[cpu]==0.4.38"    # CPU only — folding will be very slow; UI/analysis still work

# ColabFold's own dependencies, installed first...
pip install appdirs requests tqdm absl-py dm-tree scipy matplotlib \
            dm-haiku ml-collections immutabledict alphafold-colabfold

# ...then ColabFold itself, with --no-deps so it can't silently downgrade
# the JAX/NumPy versions pinned above
pip install --no-deps "colabfold @ https://codeload.github.com/sokrypton/ColabFold/zip/refs/heads/main"

# Re-pin NumPy: ColabFold's bundled AlphaFold code isn't NumPy-2 compatible
pip install numpy==1.26.4
```

### 3. Launch

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

- **Monomer**: paste a sequence (e.g. the ubiquitin example above)
- **Multimer / complex**: separate chains with `:`, e.g. `SEQ_A:SEQ_B`
- **Load from the AlphaFold Database**: enter a UniProt accession (e.g.
  `P00698`, `Q9UKV8`) to fetch an existing prediction instantly, no folding
  required

⚠️ **Data privacy**: sequences submitted for folding or MSA search are sent
to third-party services (ColabFold's MMseqs2 server, EBI). Do not submit
confidential sequences.

## Repository layout

```
.
├── alphafold_fusion_AFF3.ipynb   # Original Colab notebook (GPU quick start)
├── app.py                        # Streamlit entry point
├── alphafold_fusion/             # Core package
│   ├── config.py                 # Paths, colour schemes, API endpoints
│   ├── sequence.py                # FASTA parsing & sequence utilities
│   ├── api.py                     # UniProt / InterPro / AlphaFold DB clients
│   ├── structure.py                # CIF/PDB parsing helpers
│   ├── plddt.py                    # pLDDT confidence analysis
│   ├── pae.py                      # Predicted Aligned Error
│   ├── render.py                   # 3D structure rendering (py3Dmol)
│   ├── alignment.py                # MSA / A3M handling
│   ├── runner.py                   # ColabFold job orchestration
│   ├── disorder.py                 # Intrinsic disorder (IDR) detection
│   ├── interface_analysis.py       # Inter-chain contacts (complexes)
│   ├── ensemble_variance.py        # Cross-model RMSF
│   ├── conservation.py             # Evolutionary conservation from the MSA
│   ├── disprot.py                  # DisProt ground-truth comparison
│   ├── report.py                   # FAIR JSON/TSV export
│   └── pages/                      # Streamlit multi-page UI (home, predictions,
│                                    # results, viewer, analysis, validation, settings)
├── tests/test_core.py             # 26 unit tests (pure logic, no GPU/network)
├── examples/ubiquitin_monomer.fasta
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

The suite covers the pure-logic modules — sequence parsing, UniProt
accession matching, disorder-region detection, conservation scoring,
DisProt comparison, structure-format detection — and needs no GPU or
network access. A GitHub Actions workflow (`.github/workflows/tests.yml`)
runs it automatically on every push/PR.

## Validation

Intrinsic-disorder predictions (a pLDDT < 50 heuristic) are benchmarked
against DisProt's experimentally curated disorder annotations, reporting a
threshold-free AUC-ROC (current result: **0.80**) alongside MCC/F1 at the
default threshold. A sensitivity analysis over the minimum-IDR-length
parameter (1/3/5/8/10 residues) confirms the AUC-ROC is stable, since it's
computed on continuous pLDDT and is independent of that heuristic.

## Acknowledgments & citations

This project is an analysis/wrapper layer around published tools and public
databases. If you use results produced with it, please cite the underlying
resources:

- **ColabFold** — Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S,
  Steinegger M. "ColabFold: making protein folding accessible to all."
  *Nature Methods* 19, 679–682 (2022). doi:10.1038/s41592-022-01488-1.
  Source code is MIT-licensed.
- **AlphaFold2** — Jumper J, Evans R, Pritzel A, et al. "Highly accurate
  protein structure prediction with AlphaFold." *Nature* 596, 583–589
  (2021). doi:10.1038/s41586-021-03819-z. Source code is Apache-2.0
  licensed; model parameters are CC BY 4.0.
- **UniProt** — The UniProt Consortium. "UniProt: the Universal Protein
  Knowledgebase in 2025." *Nucleic Acids Research* 53, D609–D617 (2025).
- **InterPro** — Blum M, Andreeva A, Florentino LC, et al. "InterPro: the
  protein sequence classification resource in 2025." *Nucleic Acids
  Research* 53, D444–D456 (2025).
- **DisProt** — Nugnes MV, Bouhraoua KEA, Zoubiri M, et al. "DisProt in
  2026: enhancing intrinsically disordered proteins accessibility,
  deposition, and annotation." *Nucleic Acids Research* 54, D383–D392
  (2026).
- **py3Dmol** — David Koes' 3D molecular viewer, used for the in-app
  structure viewer.

Also see `CITATION.cff` for citing this repository itself.

## License

This repository's original code is released under the [MIT License](LICENSE).
It does not redistribute the AlphaFold model parameters or the ColabFold
codebase — both are fetched at install/run time under their own licenses
(see above). You're responsible for complying with the terms of use of the
third-party services this app calls (ColabFold's MMseqs2 server, EBI
UniProt/InterPro, the AlphaFold Database, DisProt).

## Contributing

Issues and pull requests are welcome. Please run `pytest tests/ -v` before
submitting a PR.
