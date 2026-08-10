%%writefile tests/test_core.py
"""Unit tests for AlphaFold Fusion core functions.

Run with:  pytest tests/ -v
Covers pure functions (no network, no Streamlit).
"""

import sys
sys.path.insert(0, "/content")

# Compatibility shim for NumPy legacy aliases (scipy via xarray).
import numpy as _np
for _a, _t in [("long", int), ("uint", int), ("ulong", int),
               ("longlong", int), ("ulonglong", int), ("unicode", str)]:
    if not hasattr(_np, _a):
        setattr(_np, _a, _t)

import numpy as np
import pytest


def test_clean_sequence():
    from alphafold_fusion.sequence import clean_sequence
    assert clean_sequence("acdef") == "ACDEF"
    assert clean_sequence("ACD-EF*1") == "ACDEF"
    assert clean_sequence("ACD:EFG") == "ACD:EFG"
    assert clean_sequence("") == ""


def test_is_complex():
    from alphafold_fusion.sequence import is_complex
    assert is_complex("ACDE:FGHI") is True
    assert is_complex("ACDEFG") is False


def test_total_length():
    from alphafold_fusion.sequence import total_length
    assert total_length("ACDE:FGH") == 7
    assert total_length("ACDEFG") == 6


def test_parse_fasta_single():
    from alphafold_fusion.sequence import parse_fasta
    result = parse_fasta(">prot1\nACDEF\nGHIKL")
    assert len(result) == 1
    assert result[0][0] == "prot1"
    assert result[0][1] == "ACDEFGHIKL"


def test_parse_fasta_multiple():
    from alphafold_fusion.sequence import parse_fasta
    result = parse_fasta(">p1\nACDE\n>p2\nFGHI")
    assert len(result) == 2
    assert result[1][1] == "FGHI"


def test_safe_basename():
    from alphafold_fusion.sequence import safe_basename
    name = safe_basename("my/prot name!", "ACDEF")
    assert "/" not in name and " " not in name and len(name) > 0


def test_is_uniprot_acc():
    from alphafold_fusion.api import is_uniprot_acc
    assert is_uniprot_acc("Q9UKV8") is True
    assert is_uniprot_acc("P04637") is True
    assert is_uniprot_acc("AGO2") is False
    assert is_uniprot_acc("") is False


def test_extract_uniprot_acc():
    from alphafold_fusion.api import extract_uniprot_acc
    assert extract_uniprot_acc("sp|Q9UKV8|AGO2_HUMAN") == "Q9UKV8"
    assert extract_uniprot_acc("tr|A5D9G9|A5D9G9_BOVIN") == "A5D9G9"
    assert extract_uniprot_acc("just a name") is None
    assert extract_uniprot_acc("UniRef90_Q9UKV8") == "Q9UKV8"


def test_identify_idrs_empty():
    from alphafold_fusion.disorder import identify_idrs
    result = identify_idrs([])
    assert result["n_residues"] == 0
    assert result["regions"] == []


def test_identify_idrs_all_ordered():
    from alphafold_fusion.disorder import identify_idrs
    result = identify_idrs([90.0] * 50)
    assert result["fraction_disordered"] == 0.0
    assert len(result["regions"]) == 0


def test_identify_idrs_all_disordered():
    from alphafold_fusion.disorder import identify_idrs
    result = identify_idrs([30.0] * 50)
    assert result["fraction_disordered"] == 1.0
    assert len(result["regions"]) == 1
    assert result["regions"][0]["start"] == 1
    assert result["regions"][0]["end"] == 50


def test_identify_idrs_min_length():
    from alphafold_fusion.disorder import identify_idrs
    vals = [90.0] * 20 + [30.0] * 3 + [90.0] * 20
    result = identify_idrs(vals, min_length=5)
    assert len(result["regions"]) == 0


def test_identify_idrs_region_detected():
    from alphafold_fusion.disorder import identify_idrs
    vals = [90.0] * 20 + [30.0] * 10 + [90.0] * 20
    result = identify_idrs(vals, min_length=5)
    assert len(result["regions"]) == 1
    assert result["regions"][0]["start"] == 21
    assert result["regions"][0]["end"] == 30
    assert result["regions"][0]["length"] == 10


def test_conservation_identical_column():
    from alphafold_fusion.conservation import conservation_profile
    result = conservation_profile("ACDEF", ["ACDEF"] * 10)
    assert result["n_positions"] == 5
    var = conservation_profile("ACDEF", ["KLMNP", "QRSTV", "WYACD"])
    assert result["mean_conservation"] > var["mean_conservation"]


def test_conservation_variable_column():
    from alphafold_fusion.conservation import conservation_profile
    result = conservation_profile("ACDEF", ["KLMNP", "QRSTV", "WYACD"])
    assert result["n_positions"] == 5
    assert 0.0 <= result["mean_conservation"] <= 1.0


def test_conservation_has_neff():
    from alphafold_fusion.conservation import conservation_profile
    result = conservation_profile("ACDEF", ["ACDEF", "ACDEG"])
    assert "n_effective" in result
    assert result["n_effective"] > 0


def test_disprot_regions_parsing():
    from alphafold_fusion.disprot import disprot_regions
    entry = {"disprot_consensus": {"Structural state": [
        {"start": 10, "end": 20, "type": "D"},
        {"start": 30, "end": 40, "type": "S"},
        {"start": 50, "end": 60, "type": "D"},
    ]}}
    regs = disprot_regions(entry)
    assert (10, 20) in regs
    assert (50, 60) in regs
    assert len(regs) == 2


def test_compare_perfect_match():
    from alphafold_fusion.disprot import compare_plddt_vs_disprot
    plddt = [90.0] * 10 + [30.0] * 10 + [90.0] * 10
    comp = compare_plddt_vs_disprot(plddt, [(11, 20)], threshold=50.0)
    assert comp["MCC"] > 0.9
    assert comp["TP"] == 10
    assert comp["FP"] == 0


def test_compare_no_overlap():
    from alphafold_fusion.disprot import compare_plddt_vs_disprot
    comp = compare_plddt_vs_disprot([90.0] * 30, [(11, 20)], threshold=50.0)
    assert comp["TP"] == 0
    assert comp["FN"] == 10


def test_compare_empty_plddt():
    from alphafold_fusion.disprot import compare_plddt_vs_disprot
    comp = compare_plddt_vs_disprot([], [(1, 5)])
    assert "error" in comp


def test_detect_fmt():
    from alphafold_fusion.structure import detect_fmt
    assert detect_fmt("data_AF-P04637\n_atom_site.x") == "cif"
    assert detect_fmt("ATOM      1  N   MET") == "pdb"


def test_is_amino_acid():
    from alphafold_fusion.structure import is_amino_acid
    class FakeRes:
        def __init__(self, name): self.name = name
    assert is_amino_acid(FakeRes("ALA")) is True
    assert is_amino_acid(FakeRes("HOH")) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
def test_extract_model_id():
    from alphafold_fusion.runner import _extract_model_id
    from pathlib import Path
    assert _extract_model_id(Path("result_model_1_pred_0.pdb")) == "model_1"
    assert _extract_model_id(Path("unrelaxed_model_3_multimer.pdb")) == "model_3"


def test_kabsch_superpose():
    from alphafold_fusion.ensemble_variance import kabsch_superpose
    import numpy as np
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = target + np.array([2.0, 3.0, 4.0])
    aligned = kabsch_superpose(mobile, target)
    assert np.allclose(aligned, target, atol=1e-4)
