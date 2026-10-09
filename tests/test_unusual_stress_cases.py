"""
tests/test_unusual_stress_cases.py — Нестандартные и стрессовые тесты ChemoSuite.
Проверка граничных, вырожденных, сингулярных и аномальных состояний всех аналитических ядер.
"""

import os
import io
import pytest
import numpy as np
import pandas as pd
import fticr_core
import eem_core
import chemo_ml
import chemo_pubchem
import report_generator
import project_io


# ==============================================================================
# 1. СТРЕСС-ТЕСТЫ ХЕМОМЕТРИКИ (ADVERSARIAL CHEMOMETRICS)
# ==============================================================================

def test_adversarial_constant_features_scaling():
    """Проверка слияния и скейлинга при нулевой дисперсии (все признаки константны)."""
    # Датафрейм, где половина колонок имеет нулевую дисперсию
    df_const = pd.DataFrame({
        "VK_1": [10.0] * 8,       # FT-ICR константа
        "VK_2": [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
        "FI": [1.5] * 8,          # EEM константа
        "HIX": [10.0, 12.0, 11.0, 9.0, 10.0, 12.0, 11.0, 9.0],
        "A254": [0.0] * 8,        # UV-Vis константа 0
        "SUVA254": [2.5, 3.0, 2.8, 3.2, 2.5, 3.0, 2.8, 3.2],
    })
    # Low-level fusion с блочным скейлингом не должен падать с делением на 0
    X_scaled, names, blocks = chemo_ml.prepare_fused_features(df_const, strategy="low_level", block_scaling=True)
    assert not np.any(np.isinf(X_scaled))
    assert not np.any(np.isnan(X_scaled))
    assert X_scaled.shape == (8, 6)


def test_adversarial_minimal_dataset_plsda():
    """Проверка PLS-DA на минимально возможном объеме данных (по 2 образца в классе)."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    # Берем минимальный срез: по 2 образца из каждого класса
    idx_min = [0, 1, 6, 7]
    df_min = df.iloc[idx_min].reset_index(drop=True)
    y_min = y.iloc[idx_min].reset_index(drop=True)
    X = df_min.drop(columns=["Sample_ID"])
    res = chemo_ml.train_plsda_model(X, y_min.values, n_components=1)
    assert "R2Y" in res
    assert "VIP_df" in res
    assert not res["VIP_df"].empty
    assert not res["VIP_df"]["VIP"].isna().any()


def test_adversarial_collinear_hotelling_ellipse():
    """Проверка 95% эллипса Хотеллинга при коллинеарных проекциях (ранг 1)."""
    x = np.linspace(-5, 5, 20)
    y = x * 2.0  # идеальная коллинеарность
    ell_x, ell_y = chemo_ml.compute_hotelling_ellipse(x, y, n_points=50, confidence=0.95)
    assert len(ell_x) == 50
    assert len(ell_y) == 50
    assert not np.any(np.isnan(ell_x))
    assert not np.any(np.isnan(ell_y))


def test_adversarial_multiclass_splsda_biomarkers():
    """Проверка sPLS-DA при 3 классах и отборе разреженных биомаркеров."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    y_multi = y.astype(str).values.copy()
    # Делаем 3 класса: Class0, Class1, ThirdClass
    y_multi[8:] = "Third_Class"
    X = df.drop(columns=["Sample_ID"])

    res = chemo_ml.train_splsda_model(X, y_multi, n_components=2, keep_x=5)
    assert "Sparse_Loadings_df" in res
    sp_df = res["Sparse_Loadings_df"]
    assert not sp_df.empty
    assert "Is_Selected" in sp_df.columns
    assert "Absolute_Weight" in sp_df.columns
    assert res.get("is_multiclass") is True


# ==============================================================================
# 2. СТРЕСС-ТЕСТЫ FT-ICR MS (ADVERSARIAL MASS SPECTROMETRY)
# ==============================================================================

def test_adversarial_empty_spectrum_kmd():
    """Проверка устойчивости функций FT-ICR при пустом спектре и спектре из одного пика."""
    empty_masses = np.array([])
    km, kmd, nkm = fticr_core.compute_kmd(empty_masses, base_key="CH2")
    assert len(km) == 0
    assert len(kmd) == 0

    single_mass = np.array([300.12345])
    km, kmd, nkm = fticr_core.compute_kmd(single_mass, base_key="CH2")
    assert len(km) == 1
    assert not np.isnan(kmd[0])


def test_adversarial_out_of_bounds_perminova_grid():
    """Проверка расчета 20 ячеек Перминовой на точках далеко за пределами стандартного H/C и O/C."""
    extreme_df = pd.DataFrame({
        "O/C": [3.0, 0.05, 1.8],  # Вне нормальных границ O/C (0.0-1.0)
        "H/C": [5.0, 0.2, 2.9],   # Вне нормальных границ H/C (0.3-2.2)
        "intensity": [100.0, 200.0, 300.0],
    })
    pct_count, pct_weight, grid_df = fticr_core.compute_vk20_grid(extreme_df)
    assert pct_count.shape == (5, 4)
    assert pct_weight.shape == (5, 4)
    assert len(grid_df) == 20
    assert not np.any(np.isnan(pct_count))
    assert not np.any(np.isnan(pct_weight))


def test_adversarial_tmds_with_no_matching_reactions():
    """Проверка TMDS трансформаций при отсутствии реакций в пределах допуска."""
    peaks = pd.DataFrame({
        "mass": [100.0, 500.0, 900.0],
        "intensity": [100.0, 100.0, 100.0]
    })
    # При очень жестком допуске 0.000001 mDa пар между случайными массами быть не должно
    summary, pairs = fticr_core.run_tmds_screening(peaks, top_n=3, tol_mda=0.00001)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(pairs, pd.DataFrame)
    assert pairs.empty


def test_adversarial_extreme_formula_assignment():
    """Проверка приписывания формул при жестких ограничениях (0 совпадений)."""
    peaks = pd.DataFrame({"mass": [9999.12345], "intensity": [1000.0]})
    bounds = {"C": (4, 10), "H": (4, 15), "O": (1, 5), "N": (0, 0), "S": (0, 0)}
    res = fticr_core.fast_formula_assigner(peaks, bounds=bounds, ppm_tolerance=1.0)
    assert res.empty


# ==============================================================================
# 3. СТРЕСС-ТЕСТЫ ОПТИКИ (ADVERSARIAL EEM & UV-VIS)
# ==============================================================================

def test_adversarial_eem_negative_fluorescence_scatter_removal():
    """Проверка удаления рэлеевского рассеяния на матрицах с шумом детектора (<0) и нулями."""
    ex = np.linspace(240, 450, 15)
    em = np.linspace(300, 550, 20)
    # Шум от -5.0 до 10.0
    eem_mat = np.random.uniform(-5.0, 10.0, size=(len(em), len(ex)))

    cleaned = eem_core.remove_scatter_bands(eem_mat, ex=ex, em=em, delta_rayleigh1=15.0)
    assert cleaned.shape == eem_mat.shape
    assert not np.any(np.isnan(cleaned))


def test_adversarial_uv_vis_indices_zero_signal():
    """Проверка расчета оптических дескрипторов УФ-Вид при поглощении ~0 (защита от деления на 0)."""
    wl = np.linspace(200, 700, 501)
    abs_vals = np.zeros_like(wl)
    abs_vals[wl < 260] = 0.5  # только на коротких волнах есть сигнал
    sample = eem_core.UVVisSample(sample_id="Zero_Abs_Test", wl=wl, absorbance=abs_vals, doc=1.0)

    indices = eem_core.calculate_uv_vis_indices(sample)
    assert isinstance(indices, dict)
    assert not np.isinf(indices.get("E2_E3", 0))


# ==============================================================================
# 4. СТРЕСС-ТЕСТЫ ХЕМОИНФОРМАТИКИ (ADVERSARIAL PUBCHEM & CROSS-DB)
# ==============================================================================

def test_adversarial_pubchem_weird_formulas():
    """Проверка очистки и эвристической классификации на заведомо невалидных формулах."""
    assert chemo_pubchem.clean_formula("") == ""
    assert chemo_pubchem.clean_formula("???###$$$") == "???###$$$"
    assert chemo_pubchem.clean_formula("   C12H16O3   ") == "C12H16O3"

    # Несуществующая формула в офлайн режиме
    fake_res = chemo_pubchem.lookup_formula_in_pubchem("C99H999O99", max_records=1, use_online=False)
    assert isinstance(fake_res, list)

    # Генерация таблицы аннотаций со спецсимволами
    df_ann = chemo_pubchem.annotate_formula_table(["C7H8O2", "INVALID_XYZ"], max_top=2)
    assert len(df_ann) == 2
    assert "ChEMBL_URL" in df_ann.columns
    assert "https://www.ebi.ac.uk/chembl/search_results/" in df_ann.iloc[0]["ChEMBL_URL"]
    assert "https://hmdb.ca/unearth/q" in df_ann.iloc[0]["HMDB_URL"]


# ==============================================================================
# 5. СТРЕСС-ТЕСТЫ ПРОЕКТОВ И ЭКСПОРТА (PROJECT IO & EXCEL PASSPORT)
# ==============================================================================

def test_adversarial_corrupted_chemo_project_loading():
    """Проверка защиты от загрузки поврежденного байтового потока вместо .chemo проекта."""
    junk_bytes = b"CORRUPTED_NOT_A_ZIP_HEADER_DATA_12345"
    with pytest.raises(Exception):
        project_io.get_chemo_project_summary(junk_bytes)

    with pytest.raises(Exception):
        project_io.load_chemo_project(junk_bytes)


def test_adversarial_excel_passport_missing_modalities():
    """Проверка генерации отчета Excel, когда доступна только мультиблочная матрица без ML модели."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    df["Class_Target"] = y.values

    # Генерация на русском без model_results
    excel_bytes_ru = report_generator.generate_excel_passport(df, model_results=None, lang="ru")
    assert isinstance(excel_bytes_ru, bytes)
    assert len(excel_bytes_ru) > 1000

    # Генерация на английском
    excel_bytes_en = report_generator.generate_excel_passport(df, model_results=None, lang="en")
    assert isinstance(excel_bytes_en, bytes)
    assert len(excel_bytes_en) > 1000
