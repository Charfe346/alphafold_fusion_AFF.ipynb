# AlphaFold Fusion — Colab/Streamlit Environment

**ColabFold-centred prediction, artefact harvesting, and integrated structural analysis**

[![(https://colab.research.google.com/drive/1KMWIkIVkcmFlVHsHkTcjYT2UHwEEh_dK](https://colab.research.google.com/drive/1KMWIkIVkcmFlVHsHkTcjYT2UHwEEh_dK)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Publication

> **AlphaFold Fusion: Two Interfaces for Protein-Structure Interpretation
> and Simulation-Ready Exports**
>
> Charfeddine Gharsallah, Thomas Cokelaer, Hervé Lecoeur, Eric Prina,
> Gerald F. Späth
>
> Institut Pasteur, Université Paris Cité, INSERM U1347

## Quick Start

1. Click the **Open in Colab** badge above
2. Set runtime: **Runtime → Change runtime type → GPU (T4 or A100)**
3. Run all cells sequentially
4. Click the generated URL to open the Streamlit interface

## Features

### Prediction
- ColabFold inference (monomers & multimers)
- AFDB-first retrieval mode (skip prediction for known structures)
- Two-panel multimer input (Side A / Side B)
- Configurable MSA, recycles, templates, AMBER relax

### Analysis Modules

| Module | Method | Reference |
|--------|--------|-----------|
| **Coordinate Error** | pLDDT calibration curve | Jumper et al. (2021) *Nature* 596:583, Ext. Data Fig. 1 |
| **Intrinsic Disorder** | pLDDT < 50 threshold + GFF3 export | Akdel et al. (2022) *NSMB* 29:1056 |
| **Domain Decomposition** | Spectral clustering on PAE affinity | Ng et al. (2001) NIPS; von Luxburg (2007) |
| **Interface Analysis** | Cα-Cα contacts (8 Å cutoff) | Duarte et al. (2012) *BMC Bioinformatics* 13:334 |
| **Ensemble Variance** | Kabsch RMSD/RMSF across models | Kabsch (1976); Wallner (2023) |
| **Conservation** | Shannon entropy from MSA | Shannon (1948); Valdar (2002) |

### Visualization
- 3D interactive viewer (py3Dmol)
- pLDDT confidence coloring (4-tier AlphaFold scheme)
- PAE heatmaps
- UniProt & InterPro domain overlays
- PAE-derived domain coloring
- Disorder overlay

### Data Export
- JSON structured reports (FAIR-compliant)
- TSV per-residue profiles
- GFF3 disorder annotations
- PDB/CIF structure files

## Architecture
alphafold_fusion/
├── init.py              # Package metadata
├── config.py                # Paths, colors, API endpoints
├── sequence.py              # FASTA parsing, cleaning
├── api.py                   # UniProt, InterPro, AFDB clients
├── structure.py             # CIF/PDB I/O, chain detection
├── plddt.py                 # pLDDT extraction, visualization
├── pae.py                   # PAE heatmap generation
├── render.py                # 3D visualization (py3Dmol)
├── alignment.py             # A3M MSA parsing
├── runner.py                # ColabFold execution
├── coordinate_error.py      # pLDDT → positional error
├── disorder.py              # IDR identification
├── domain_decomposition.py  # Spectral clustering on PAE
├── interface_analysis.py    # Inter-chain contacts
├── ensemble_variance.py     # Multi-model RMSF
├── conservation.py          # Shannon entropy from MSA
├── report.py                # JSON/TSV export
└── pages/
├── init.py           # Page registry
├── home.py               # About page
├── predictions.py        # Input & job launch
├── results.py            # Model comparator
├── viewer.py             # 3D viewer + domains
├── analysis.py           # Post-prediction analysis
└── settings.py           # System info
app.py                        # Streamlit entry point
## Companion: Web Application

🌐 **Live:** [https://alphafold-fusion-6demeeil3-charfe346s-projects.vercel.app/]

📦 **Code:** [github.com/Charfe346/alphafold-fusion-AFF-](https://github.com/Charfe346/alphafold-fusion-AFF-)

## Requirements

- Google Colab with GPU runtime (T4 minimum, A100 recommended)
- ColabFold (auto-installed by notebook)
- Python 3.10+

## References

- Abramson, J. et al. (2024) Nature, 630, 493–500
- Akdel, M. et al. (2022) Nat. Struct. Mol. Biol., 29, 1056–1067
- Campen, A. et al. (2008) Protein Pept. Lett., 15, 956–963
- Duarte, J.M. et al. (2012) BMC Bioinformatics, 13, 334
- Jumper, J. et al. (2021) Nature, 596, 583–589
- Kabsch, W. (1976) Acta Crystallogr. A, 32, 922–923
- Lin, Z. et al. (2023) Science, 379, 1123–1130
- Mirdita, M. et al. (2022) Nat. Methods, 19, 679–682
- Ng, A.Y. et al. (2001) NIPS, 14, 849–856
- Orengo, C.A. et al. (1997) Structure, 5, 1093–1108
- Piovesan, D. et al. (2022) Nucleic Acids Res., 50, D471–D477
- Shannon, C.E. (1948) Bell Syst. Tech. J., 27, 379–423
- Valdar, W. (2002) Proteins, 48, 227–241
- von Luxburg, U. (2007) Stat. Comput., 17, 395–416
- Wallner, B. (2023) Bioinformatics, 39, btad573
- Wilkinson, M.D. et al. (2016) Sci. Data, 3, 160018
- Zhang, Y. & Skolnick, J. (2004) Proteins, 57, 702–710

## License

MIT License

## Contact

Charfeddine Gharsallah — charfeddine.gharsallah@pasteur.fr
Institut Pasteur, Université Paris Cité, INSERM U1347
