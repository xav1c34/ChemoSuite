"""
test_cross_correlation.py — Unit tests for cross-modal correlation and 2D heterospectral analysis.
"""
import numpy as np
import pandas as pd
import pytest

from modules.cross_correlation import (
    compute_heterospectral_2d_correlation,
    compute_multiblock_cross_correlation,
    build_correlation_network_edges,
)


def test_heterospectral_2d_correlation():
    # 10 образцов: спектр A (5 полос) и спектр B (4 полосы)
    np.random.seed(42)
    x1 = np.random.randn(10, 5)
    # Создаем сильную искусственную связь между x1[:, 0] и x2[:, 1]
    x2 = np.random.randn(10, 4)
    x2[:, 1] = 0.9 * x1[:, 0] + 0.1 * np.random.randn(10)

    phi = compute_heterospectral_2d_correlation(x1, x2, scale=True)
    assert phi.shape == (5, 4)
    assert np.all(phi >= -1.0) and np.all(phi <= 1.0)
    # Корреляция между первой полосой A и второй B должна быть высокой
    assert phi[0, 1] > 0.8


def test_multiblock_cross_correlation():
    df = pd.DataFrame({
        "Sample_ID": [f"S_{i}" for i in range(10)],
        "Class": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "VK_1": [1.0, 1.2, 1.1, 1.3, 1.0, 3.5, 3.8, 3.6, 3.9, 4.0],
        "FTIR_1710": [0.5, 0.6, 0.55, 0.58, 0.52, 1.8, 1.9, 1.85, 2.0, 2.1],
        "SEC_Mw": [1500, 1600, 1550, 1580, 1520, 4500, 4800, 4600, 4900, 5100],
        "EEM_C1": [10.0, 11.0, 10.5, 10.8, 10.2, 35.0, 38.0, 36.0, 39.0, 41.0],
    })

    res = compute_multiblock_cross_correlation(df, method="pearson")
    assert "correlation_matrix" in res
    assert "p_values" in res
    assert "top_cross_correlations" in res

    corr_df = res["correlation_matrix"]
    assert corr_df.shape == (4, 4)
    # Проверяем, что обнаружены сильные межблочные связи
    top_df = res["top_cross_correlations"]
    assert not top_df.empty
    assert top_df["Abs_Correlation"].iloc[0] > 0.8


def test_correlation_network_edges():
    corr_df = pd.DataFrame(
        [[1.0, 0.85, 0.2],
         [0.85, 1.0, 0.9],
         [0.2, 0.9, 1.0]],
        index=["VK_1", "FTIR_1", "SEC_1"],
        columns=["VK_1", "FTIR_1", "SEC_1"]
    )
    block_map = {"VK_1": "FT-ICR MS", "FTIR_1": "FTIR", "SEC_1": "SEC/GPC"}

    nodes_df, edges_df = build_correlation_network_edges(corr_df, block_map, threshold=0.8, cross_block_only=True)
    assert len(nodes_df) == 3
    assert len(edges_df) == 2  # VK_1-FTIR_1 и FTIR_1-SEC_1
    assert "source" in edges_df.columns and "target" in edges_df.columns
