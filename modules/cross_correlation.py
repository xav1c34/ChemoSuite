"""
cross_correlation.py — Модуль кросс-модального корреляционного анализа (Cross-Omics / 2D Heterospectral Correlation).
Математический аппарат гетероспектральной 2D-корреляции (FT-ICR MS <-> FTIR <-> EEM <-> SEC <-> NMR <-> Thermal <-> Elemental),
расчета межблочных корреляций дескрипторов и построения сетей ко-вариации признаков.
"""
from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_heterospectral_2d_correlation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    scale: bool = True,
) -> np.ndarray:
    """
    Вычисляет синхронную 2D-гетероспектральную корреляционную матрицу (2D-COS)
    между двумя модальностями данных для согласованных N образцов:
    Phi(v1, v2) = (1 / (N - 1)) * X_a_centered^T * X_b_centered
    
    Parameters
    ----------
    matrix_a : np.ndarray
        Матрица первой модальности формы (N, P1)
    matrix_b : np.ndarray
        Матрица второй модальности формы (N, P2)
    scale : bool
        Если True, нормализует до коэффициентов корреляции Пирсона [-1, 1]

    Returns
    -------
    np.ndarray
        Матрица взаимной корреляции формы (P1, P2)
    """
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(f"Число образцов в матрицах должно совпадать: {matrix_a.shape[0]} vs {matrix_b.shape[0]}")

    n_samples = matrix_a.shape[0]
    if n_samples < 2:
        return np.zeros((matrix_a.shape[1], matrix_b.shape[1]))

    # Центрирование
    a_cent = matrix_a - np.mean(matrix_a, axis=0, keepdims=True)
    b_cent = matrix_b - np.mean(matrix_b, axis=0, keepdims=True)

    if scale:
        std_a = np.std(a_cent, axis=0, keepdims=True)
        std_b = np.std(b_cent, axis=0, keepdims=True)
        std_a = np.where(std_a == 0, 1e-12, std_a)
        std_b = np.where(std_b == 0, 1e-12, std_b)
        a_cent = a_cent / std_a
        b_cent = b_cent / std_b

    phi = np.dot(a_cent.T, b_cent) / (n_samples - 1.0)
    return np.clip(phi, -1.0, 1.0) if scale else phi


def compute_multiblock_cross_correlation(
    fused_df: pd.DataFrame,
    sample_id_col: str = "Sample_ID",
    target_col: Optional[str] = "Class",
    method: str = "pearson",
    min_variance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Рассчитывает полную матрицу попарных корреляций всех численных дескрипторов
    мультимодального датасета с расчетом p-значений и принадлежностью к аналитическим блокам.

    Returns
    -------
    Dict с ключами:
        - correlation_matrix: pd.DataFrame
        - p_values: pd.DataFrame
        - block_mapping: Dict[str, str]
        - top_cross_correlations: pd.DataFrame (межблочные сильные связи)
    """
    # Исключаем нечисловые метаданные и метку класса
    ignore_cols = {sample_id_col}
    if target_col and target_col in fused_df.columns:
        ignore_cols.add(target_col)

    numeric_cols = [
        c for c in fused_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(fused_df[c])
    ]

    # Фильтруем константные колонки
    valid_cols = [c for c in numeric_cols if fused_df[c].var() > min_variance]
    sub_df = fused_df[valid_cols].copy()

    # Импорт классификатора блоков из chemo_ml при наличии
    try:
        from modules.chemo_ml import classify_feature_blocks
        block_map = classify_feature_blocks(valid_cols)
    except Exception:
        block_map = {c: "General" for c in valid_cols}

    n_feats = len(valid_cols)
    corr_arr = np.zeros((n_feats, n_feats))
    pval_arr = np.ones((n_feats, n_feats))

    vals = sub_df.values
    for i in range(n_feats):
        for j in range(i, n_feats):
            x = vals[:, i]
            y = vals[:, j]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(mask) >= 3:
                if method == "spearman":
                    r_val, p_val = stats.spearmanr(x[mask], y[mask])
                else:
                    r_val, p_val = stats.pearsonr(x[mask], y[mask])
                r_val = float(r_val) if not np.isnan(r_val) else 0.0
                p_val = float(p_val) if not np.isnan(p_val) else 1.0
            else:
                r_val, p_val = 0.0, 1.0

            corr_arr[i, j] = r_val
            corr_arr[j, i] = r_val
            pval_arr[i, j] = p_val
            pval_arr[j, i] = p_val

    corr_df = pd.DataFrame(corr_arr, index=valid_cols, columns=valid_cols)
    pval_df = pd.DataFrame(pval_arr, index=valid_cols, columns=valid_cols)

    # Поиск топ-межблочных взаимосвязей (разные блоки данных)
    cross_pairs = []
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            feat_a, feat_b = valid_cols[i], valid_cols[j]
            block_a, block_b = block_map.get(feat_a, "Other"), block_map.get(feat_b, "Other")
            if block_a != block_b and block_a != "Other" and block_b != "Other":
                r = float(corr_arr[i, j])
                p = float(pval_arr[i, j])
                if abs(r) >= 0.5:
                    cross_pairs.append({
                        "Feature_A": feat_a,
                        "Block_A": block_a,
                        "Feature_B": feat_b,
                        "Block_B": block_b,
                        "Correlation": round(r, 3),
                        "Abs_Correlation": round(abs(r), 3),
                        "p_value": float(f"{p:.2e}"),
                        "Significant": p < 0.05,
                    })

    cross_pairs_df = pd.DataFrame(cross_pairs)
    if not cross_pairs_df.empty:
        cross_pairs_df = cross_pairs_df.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)

    return {
        "correlation_matrix": corr_df,
        "p_values": pval_df,
        "block_mapping": block_map,
        "top_cross_correlations": cross_pairs_df,
    }


def build_correlation_network_edges(
    corr_df: pd.DataFrame,
    block_map: Dict[str, str],
    threshold: float = 0.7,
    cross_block_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Формирует узлы и рёбра сети для визуализации графа взаимосвязей аналитических модальностей.
    """
    cols = list(corr_df.columns)
    nodes = [{"id": c, "block": block_map.get(c, "Other")} for c in cols]
    nodes_df = pd.DataFrame(nodes)

    edges = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f_i, f_j = cols[i], cols[j]
            b_i, b_j = block_map.get(f_i, "Other"), block_map.get(f_j, "Other")
            if cross_block_only and b_i == b_j:
                continue

            r = float(corr_df.iloc[i, j])
            if abs(r) >= threshold:
                edges.append({
                    "source": f_i,
                    "target": f_j,
                    "weight": round(abs(r), 3),
                    "sign": "Positive" if r > 0 else "Negative",
                    "color": "#2ecc71" if r > 0 else "#e74c3c",
                    "source_block": b_i,
                    "target_block": b_j,
                })

    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df
