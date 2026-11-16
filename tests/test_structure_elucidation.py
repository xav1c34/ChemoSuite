"""
tests/test_structure_elucidation.py — Unit tests for Cross-Modal Structure Elucidation engine.
"""
import pytest
from modules.structure_elucidation import (
    MultiModalConstraints,
    CandidateStructure,
    detect_smiles_substructures,
    estimate_13c_nmr_distribution,
    compute_concordance_score,
    elucidate_candidates,
)


def test_detect_smiles_substructures():
    # Gallic acid: 3,4,5-trihydroxybenzoic acid
    gallic_smiles = "c1c(c(c(cc1C(=O)O)O)O)O"
    feats = detect_smiles_substructures(gallic_smiles)

    assert feats["has_aromatic"] is True
    assert feats["has_carboxyl"] is True
    assert feats["has_carbonyl"] is True
    assert feats["has_phenol_oh"] is True
    assert feats["has_aliphatic"] is False

    # Aliphatic ester: Ethyl acetate
    ester_smiles = "CCOC(=O)C"
    feats_ester = detect_smiles_substructures(ester_smiles)
    assert feats_ester["has_aromatic"] is False
    assert feats_ester["has_carbonyl"] is True
    assert feats_ester["has_aliphatic"] is True


def test_estimate_13c_nmr_distribution():
    # Benzoic acid: 7 carbons total (6 aromatic, 1 carboxyl)
    benzoic_smiles = "c1ccccc1C(=O)O"
    dist = estimate_13c_nmr_distribution(benzoic_smiles)

    assert dist["total_carbons"] == 7
    # 6 / 7 ~ 0.857
    assert 0.80 <= dist["aromatic"] <= 0.90
    # 1 / 7 ~ 0.143
    assert 0.10 <= dist["carboxyl"] <= 0.20
    assert dist["aliphatic"] == 0.0


def test_concordance_scoring_and_ranking():
    # Target constraints: Aromatic polyphenolic carboxylic acid (e.g., Gallic acid derivative)
    constraints = MultiModalConstraints(
        formula="C7H6O5",
        exact_mass=170.0215,
        mass_tol_ppm=1.0,
        ftir_carbonyl=1,
        ftir_carboxyl=1,
        ftir_phenol_oh=1,
        ftir_aromatic_c_c=1,
        ftir_aliphatic_c_h=0,
        nmr_aromatic_pct=0.85,
        nmr_carboxyl_pct=0.15,
    )

    # Candidate 1: Gallic acid (excellent match)
    cand_good = {
        "cid": 370,
        "name": "Gallic acid",
        "smiles": "c1c(c(c(cc1C(=O)O)O)O)O",
        "formula": "C7H6O5",
        "exact_mass": 170.0215,
    }

    # Candidate 2: Aliphatic keto-ester (poor match to aromatic + phenol constraints)
    cand_poor = {
        "cid": 9999,
        "name": "Synthetic Aliphatic Isomer",
        "smiles": "CC(=O)CC(=O)OCC(=O)O",
        "formula": "C7H6O5",
        "exact_mass": 170.0215,
    }

    score_good, bd_good, _ = compute_concordance_score(cand_good, constraints)
    score_poor, bd_poor, _ = compute_concordance_score(cand_poor, constraints)

    assert score_good > score_poor
    assert score_good >= 85.0
    assert bd_good["FTIR Functional Groups"] == 100.0

    # Test elucidation ranking
    results = elucidate_candidates(constraints, custom_candidates=[cand_poor, cand_good])
    assert len(results) == 2
    # First candidate must be Gallic acid
    assert results[0].name == "Gallic acid"
    assert results[0].concordance_score > results[1].concordance_score


def test_multimodal_constraints_and_candidate_structure_view_flow():
    """Verify exact usage pattern from chemo_ml_view.py renders and scores properly."""
    target_formula = "C9H10O3"
    active_ftir_keys = ["1510", "1600", "1720"]
    nmr_arom_val = 66.7
    chns_c_val = 65.0

    constraints = MultiModalConstraints(
        formula=target_formula,
        ftir_active_bands=active_ftir_keys,
        nmr_aromatic_fraction=nmr_arom_val / 100.0,
        nmr_aliphatic_fraction=(100.0 - nmr_arom_val) / 200.0,
        elemental_pct={"C": chns_c_val, "H": 6.0, "O": 28.0},
    )

    # Verify constraints mapping
    assert constraints.formula == "C9H10O3"
    assert constraints.ftir_aromatic_c_c == 1
    assert constraints.ftir_carboxyl == 1
    assert constraints.nmr_aromatic_pct == pytest.approx(0.667, rel=1e-3)
    assert constraints.target_h_c is not None

    candidates = [
        CandidateStructure("2-Methoxy-4-vinylphenol", "COc1cc(C=C)ccc1O", target_formula),
        CandidateStructure("Vanillin ethyl ether", "CCOc1cc(C=O)ccc1OC", target_formula),
        CandidateStructure("Coniferyl aldehyde precursor", "COc1cc(CC=O)ccc1O", target_formula),
        CandidateStructure("Aliphatic ester isomer", "CCCC(=O)OCCCC", target_formula),
    ]

    # Verify elucidate_candidates with (candidates, constraints) argument order
    ranked = elucidate_candidates(candidates, constraints)
    assert len(ranked) == 4
    for i, r in enumerate(ranked, 1):
        assert r.rank == i
        assert isinstance(r.name, str)
        assert isinstance(r.smiles, str)
        assert 0.0 <= r.concordance_score <= 1.0
        assert 0.0 <= r.ms_match_score <= 1.0
        assert 0.0 <= r.ftir_match_score <= 1.0
        assert 0.0 <= r.nmr_match_score <= 1.0
        assert 0.0 <= r.elemental_match_score <= 1.0

    # Ensure best ranked compound has higher concordance score than the lowest ranked
    assert ranked[0].concordance_score >= ranked[-1].concordance_score
