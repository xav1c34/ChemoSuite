"""
cross_correlation.py — Модуль кросс-модального корреляционного анализа (Cross-Omics / 2D Heterospectral Correlation).
Математический аппарат гетероспектральной 2D-корреляции (FT-ICR MS <-> FTIR <-> EEM <-> SEC <-> NMR <-> Thermal <-> Elemental),
расчета межблочных корреляций дескрипторов и построения сетей ко-вариации признаков.
"""
from dataclasses import dataclass, field
import io
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


def hilbert_noda_matrix(m: int) -> np.ndarray:
    """
    Матрица преобразования Гильберта–Нода (Noda, 1993/2000) размера (m, m):
    N[j, k] = 0 при j == k, иначе 1 / (pi * (k - j)).
    """
    m = int(m)
    if m <= 0:
        return np.zeros((0, 0))
    idx = np.arange(m)
    diff = idx[None, :] - idx[:, None]  # (k - j)
    with np.errstate(divide="ignore"):
        noda = np.where(diff == 0, 0.0, 1.0 / (np.pi * np.where(diff == 0, 1, diff)))
    return noda


def _dynamic_spectra(matrix: np.ndarray, reference: Union[str, np.ndarray, None] = "mean") -> np.ndarray:
    """Динамические спектры: вычитание референсного спектра (по умолчанию — среднего по возмущению)."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if reference is None:
        return arr.copy()
    if isinstance(reference, str):
        if reference != "mean":
            raise ValueError(f"Неизвестный тип референсного спектра: {reference!r} (ожидается 'mean' или массив).")
        return arr - np.mean(arr, axis=0, keepdims=True)
    ref = np.asarray(reference, dtype=float).reshape(1, -1)
    if ref.shape[1] != arr.shape[1]:
        raise ValueError("Длина референсного спектра не совпадает с числом переменных.")
    return arr - ref


def compute_2d_correlation_spectra(
    spectra_a: np.ndarray,
    spectra_b: Optional[np.ndarray] = None,
    reference: Union[str, np.ndarray, None] = "mean",
) -> Dict[str, np.ndarray]:
    """
    Обобщённая двумерная корреляционная спектроскопия Нода (generalized 2D-COS).

    Строки матриц — m последовательных шагов внешнего возмущения (температура, время,
    концентрация, порядок отбора проб и т.д.), столбцы — спектральные переменные.
    Порядок строк ОБЯЗАН соответствовать порядку возмущения: асинхронный спектр
    без упорядочивания физического смысла не имеет.

    Synchronous:  Phi = Y_a^T Y_b / (m - 1)
    Asynchronous: Psi = Y_a^T N Y_b / (m - 1),  N — матрица Гильберта–Нода.
    Y — динамические (центрированные по референсу) спектры.

    Если spectra_b не задана — гомоспектральный анализ (spectra_b = spectra_a).

    Returns
    -------
    Dict с ключами "synchronous" (P1, P2) и "asynchronous" (P1, P2).
    """
    a = np.asarray(spectra_a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    b = a if spectra_b is None else np.asarray(spectra_b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Число шагов возмущения в матрицах должно совпадать: {a.shape[0]} vs {b.shape[0]}")

    m = a.shape[0]
    if m < 2:
        nan_mat = np.full((a.shape[1], b.shape[1]), np.nan)
        return {"synchronous": nan_mat, "asynchronous": nan_mat.copy()}

    y_a = _dynamic_spectra(a, reference)
    y_b = _dynamic_spectra(b, reference)
    noda = hilbert_noda_matrix(m)

    sync = (y_a.T @ y_b) / (m - 1.0)
    asyn = (y_a.T @ noda @ y_b) / (m - 1.0)
    return {"synchronous": sync, "asynchronous": asyn}


def compute_heterospectral_2d_correlation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    scale: bool = True,
) -> np.ndarray:
    """
    Синхронная 2D-гетероспектральная матрица между двумя модальностями для согласованных N образцов:
    Phi(v1, v2) = (1 / (N - 1)) * X_a_centered^T * X_b_centered

    При scale=True возвращает нормированный синхронный спектр, т.е. матрицу коэффициентов
    корреляции Пирсона (эквивалент np.corrcoef). Переменные с нулевой дисперсией дают NaN
    (корреляция для них не определена). Асинхронный спектр Нода см. compute_2d_correlation_spectra().

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
    matrix_a = np.asarray(matrix_a, dtype=float)
    matrix_b = np.asarray(matrix_b, dtype=float)
    if matrix_a.ndim == 1:
        matrix_a = matrix_a.reshape(-1, 1)
    if matrix_b.ndim == 1:
        matrix_b = matrix_b.reshape(-1, 1)
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(f"Число образцов в матрицах должно совпадать: {matrix_a.shape[0]} vs {matrix_b.shape[0]}")

    n_samples = matrix_a.shape[0]
    if n_samples < 2:
        # Корреляция/ковариация по одному образцу не определена
        return np.full((matrix_a.shape[1], matrix_b.shape[1]), np.nan)

    # Центрирование (синхронный спектр Нода с референсом = среднее)
    a_cent = matrix_a - np.mean(matrix_a, axis=0, keepdims=True)
    b_cent = matrix_b - np.mean(matrix_b, axis=0, keepdims=True)

    phi = np.dot(a_cent.T, b_cent) / (n_samples - 1.0)
    if not scale:
        return phi

    # Нормировка согласованными (ddof=1) стандартными отклонениями: r = cov / (s_a * s_b)
    std_a = np.std(a_cent, axis=0, ddof=1)
    std_b = np.std(b_cent, axis=0, ddof=1)
    denom = np.outer(std_a, std_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 1e-300, phi / np.where(denom > 1e-300, denom, 1.0), np.nan)
    # Обрезка только ошибок округления порядка 1e-15 (|r| <= 1 математически)
    return np.clip(r, -1.0, 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    q-значения Бенджамини–Хохберга (FDR) для одномерного массива p-значений.
    NaN-значения исключаются из семейства гипотез и остаются NaN.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = int(np.sum(valid))
    if m == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    # Монотонизация с конца: q_(i) = min_{k >= i} p_(k) * m / k
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_valid = np.empty(m)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q


def compute_multiblock_cross_correlation(
    fused_df: pd.DataFrame,
    sample_id_col: str = "Sample_ID",
    target_col: Optional[str] = "Class",
    method: str = "pearson",
    min_variance: float = 1e-6,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Рассчитывает полную матрицу попарных корреляций всех численных дескрипторов
    мультимодального датасета с расчетом p-значений и принадлежностью к аналитическим блокам.

    Значимость определяется по q-значениям Бенджамини–Хохберга (FDR), рассчитанным по
    всему семейству проверенных уникальных пар признаков (i < j), а не по нескорректированному p.
    Пары, для которых корреляция не определена (< 3 совместных наблюдений, константа), дают NaN.

    Returns
    -------
    Dict с ключами:
        - correlation_matrix: pd.DataFrame
        - p_values: pd.DataFrame
        - q_values: pd.DataFrame (BH-FDR)
        - block_mapping: Dict[str, str]
        - top_cross_correlations: pd.DataFrame (межблочные сильные связи, с p_value и q_value)
        - n_tests: int (размер семейства гипотез для FDR)
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
    corr_arr = np.full((n_feats, n_feats), np.nan)
    pval_arr = np.full((n_feats, n_feats), np.nan)

    vals = sub_df.values.astype(float)
    for i in range(n_feats):
        for j in range(i, n_feats):
            x = vals[:, i]
            y = vals[:, j]
            mask = ~np.isnan(x) & ~np.isnan(y)
            r_val, p_val = np.nan, np.nan
            if np.sum(mask) >= 3 and np.ptp(x[mask]) > 0 and np.ptp(y[mask]) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if method == "spearman":
                        r_val, p_val = stats.spearmanr(x[mask], y[mask])
                    else:
                        r_val, p_val = stats.pearsonr(x[mask], y[mask])
                r_val = float(r_val)
                p_val = float(p_val)
            # Иначе корреляция не определена -> NaN (а не правдоподобные 0 / 1)

            corr_arr[i, j] = r_val
            corr_arr[j, i] = r_val
            pval_arr[i, j] = p_val
            pval_arr[j, i] = p_val

    # FDR (Benjamini–Hochberg) по всем уникальным парам i < j
    qval_arr = np.full((n_feats, n_feats), np.nan)
    iu = np.triu_indices(n_feats, k=1)
    n_tests = 0
    if len(iu[0]) > 0:
        q_upper = benjamini_hochberg(pval_arr[iu])
        qval_arr[iu] = q_upper
        qval_arr[(iu[1], iu[0])] = q_upper
        n_tests = int(np.sum(~np.isnan(pval_arr[iu])))

    corr_df = pd.DataFrame(corr_arr, index=valid_cols, columns=valid_cols)
    pval_df = pd.DataFrame(pval_arr, index=valid_cols, columns=valid_cols)
    qval_df = pd.DataFrame(qval_arr, index=valid_cols, columns=valid_cols)

    # Поиск топ-межблочных взаимосвязей (разные блоки данных)
    cross_pairs = []
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            feat_a, feat_b = valid_cols[i], valid_cols[j]
            block_a, block_b = block_map.get(feat_a, "Other"), block_map.get(feat_b, "Other")
            if block_a != block_b and block_a != "Other" and block_b != "Other":
                r = float(corr_arr[i, j])
                p = float(pval_arr[i, j])
                q = float(qval_arr[i, j])
                if not np.isnan(r) and abs(r) >= 0.5:
                    cross_pairs.append({
                        "Feature_A": feat_a,
                        "Block_A": block_a,
                        "Feature_B": feat_b,
                        "Block_B": block_b,
                        "Correlation": round(r, 3),
                        "Abs_Correlation": round(abs(r), 3),
                        "p_value": float(f"{p:.2e}"),
                        "q_value": float(f"{q:.2e}"),
                        "Significant": bool(q < alpha),
                    })

    cross_pairs_df = pd.DataFrame(cross_pairs)
    if not cross_pairs_df.empty:
        cross_pairs_df = cross_pairs_df.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)

    return {
        "correlation_matrix": corr_df,
        "p_values": pval_df,
        "q_values": qval_df,
        "block_mapping": block_map,
        "top_cross_correlations": cross_pairs_df,
        "n_tests": n_tests,
        "fdr_alpha": alpha,
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
            if not np.isnan(r) and abs(r) >= threshold:
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
