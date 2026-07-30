"""Global configuration: filesystem paths, colour schemes, API endpoints, CSS."""

from __future__ import annotations
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alphafold_fusion")

_BASE = Path("/content") if Path("/content").exists() else Path.cwd()
RESULTS_DIR: Path = _BASE / "alphafold_results"
CACHE_DIR: Path = _BASE / "alphafold_cache"

PLDDT_SCHEME: dict[str, dict] = {
    "Very high (>90)": {"min": 90, "max": 100, "color": "#2166F3"},
    "High (70-90)":    {"min": 70, "max": 90,  "color": "#8FD3FF"},
    "Low (50-70)":     {"min": 50, "max": 70,  "color": "#FFD35A"},
    "Very low (<50)":  {"min": 0,  "max": 50,  "color": "#FF8C42"},
}

C_VHIGH = "#2166F3"
C_HIGH  = "#8FD3FF"
C_LOW   = "#FFD35A"
C_VLOW  = "#FF8C42"
C_SPEC_LT70 = "#FF7F0E"
C_SPEC_GE70 = "#4CA6FF"
C_SPEC_GE90 = "#1F57F7"

DOMAIN_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

UNIPROT_API  = "https://rest.uniprot.org/uniprotkb/{acc}.json"
INTERPRO_APIS: list[str] = [
    "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{acc}?page_size=200",
    "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{acc}?page_size=200",
    "https://www.ebi.ac.uk/interpro/api/protein/uniprot/{acc}?page_size=200",
]
AFDB_API    = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
PY3DMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.0.4/build/3Dmol-min.js"

AA3: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
})

MSA_MODES: dict[str, str] = {
    "full": "mmseqs2_uniref_env", "fast": "mmseqs2_uniref",
    "minimal": "single_sequence",
}
MSA_LABELS: dict[str, str] = {
    "Complete (UniRef+Environ.)": "full",
    "Fast (UniRef)": "fast",
    "Minimal (single sequence)": "minimal",
}
RECYCLE_PRESETS: dict[str, dict[str, int]] = {
    "Fast":           {"monomer": 3,  "complex": 6},
    "Classic":        {"monomer": 6,  "complex": 12},
    "High precision": {"monomer": 12, "complex": 20},
}

CSS = """<style>
.main-header{font-size:3rem;background:linear-gradient(135deg,#4361ee,#3a0ca3);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
text-align:center;margin-bottom:1rem;font-weight:800}
.sub-header{font-size:1.6rem;color:#4361ee;margin:1rem 0 .6rem;
border-bottom:3px solid #4cc9f0}
.card{padding:1rem;border-radius:1rem;background:var(--aff-card-bg,#fff);
color:var(--aff-card-fg,#212529);
box-shadow:0 10px 25px rgba(0,0,0,.08);margin:1rem 0}
.info-card{background:var(--aff-info-bg,#f0f2f6);
color:var(--aff-card-fg,#212529);padding:1rem;
border-radius:1rem;border-left:5px solid #4361ee;margin:1rem 0}
.stButton>button{background:linear-gradient(135deg,#4361ee,#3a0ca3);
color:#fff;border:none;padding:.6rem 1.2rem;border-radius:50px;
font-weight:600;width:100%}

/* Dark mode support */
@media (prefers-color-scheme: dark){
  :root{
    --aff-card-bg:#1e1e2e;
    --aff-card-fg:#e0e0e0;
    --aff-info-bg:#252535;
  }
  .sub-header{color:#8fb3ff;border-bottom-color:#4cc9f0}
  .card{box-shadow:0 10px 25px rgba(0,0,0,.4)}
}
</style>"""
# EBI services require a real contact email (usage policy).
# Set via env var before running (Colab: os.environ["EBI_CONTACT_EMAIL"]=...
# in a cell above this one). Falls back to a placeholder otherwise.
EBI_CONTACT_EMAIL = os.environ.get("EBI_CONTACT_EMAIL", "your-email@example.com")