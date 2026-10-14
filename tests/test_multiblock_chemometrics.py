"""
tests/test_multiblock_chemometrics.py — Unit tests for Multi-Block Chemometrics (MB-PLS-DA & SO-PLS).
"""
import numpy as np
import pytest
from modules.multiblock_chemometrics import (
    block_scale,
    MultiBlockPLSDA,
    SequentialOrthogonalizedPLS
)


def test_block_scale_frobenius():
    X1 = np.random.RandomState(42).randn(10, 5) * 100.0  # high variance
    X2 = np.random.RandomState(42).randn(10, 2) * 0.1    # low variance

    blocks = {"MS": X1, "CHNS": X2}
    scaled, factors = block_scale(blocks, method="frobenius")

    # Frobenius norm of each centered & scaled block must equal 1.0
    for name, s_mat in scaled.items():
        norm_val = np.linalg.norm(s_mat, "fro")
        assert np.isclose(norm_val, 1.0, atol=1e-5)


def test_mb_pls_da_training_and_contributions():
    rng = np.random.RandomState(101)
    N = 12

    # Synthesize two distinct classes across 3 modalities
    # Class 0 vs Class 1
    y = np.array(["Soil_A"] * 6 + ["Soil_B"] * 6)

    # Modality 1: FT-ICR (20 features, strong class separation)
    X_ms = rng.randn(N, 20)
    X_ms[:6, :] += 2.0

    # Modality 2: NMR (5 features, moderate separation)
    X_nmr = rng.randn(N, 5)
    X_nmr[:6, :] += 1.0

    # Modality 3: Elemental (3 features, subtle separation)
    X_elem = rng.randn(N, 3)
    X_elem[6:, :] += 0.8

    blocks = {"FTICR": X_ms, "NMR": X_nmr, "Elemental": X_elem}

    model = MultiBlockPLSDA(n_components=2, scale_blocks=True)
    model.fit(blocks, y)

    assert model.super_scores_.shape == (N, 2)
    assert model.r2y_ > 0.7  # strong fit
    assert model.q2_ >= 0.0

    # Check block contributions sum to 100% per component
    for a in range(2):
        total_contrib = sum(model.block_contributions_[b][a] for b in blocks.keys())
        assert np.isclose(total_contrib, 100.0, atol=1e-4)

    # Check B-VIP scores
    for b_name, vip in model.b_vip_.items():
        assert len(vip) == blocks[b_name].shape[1]
        assert np.all(vip >= 0.0)

    # Predictions
    preds = model.predict(blocks)
    assert len(preds) == N
    acc = np.mean(preds == y)
    assert acc >= 0.9  # at least 90% accuracy on synthetic separated data

    # Summary table
    df_summary = model.get_summary_dataframe()
    assert len(df_summary) == 3
    assert "Modality" in df_summary.columns


def test_sequential_orthogonalized_pls():
    rng = np.random.RandomState(42)
    N = 10
    y = np.array(["Type1"] * 5 + ["Type2"] * 5)

    b1 = rng.randn(N, 6)
    b1[:5] += 1.5

    b2 = rng.randn(N, 4)
    b2[:5] += 1.0

    blocks = {"Primary_MS": b1, "Secondary_FTIR": b2}
    sopls = SequentialOrthogonalizedPLS()
    sopls.fit(blocks, y)

    assert "Primary_MS" in sopls.r2_cumulative
    assert "Secondary_FTIR" in sopls.r2_cumulative
    assert sopls.r2_cumulative["Secondary_FTIR"] >= sopls.r2_cumulative["Primary_MS"]
