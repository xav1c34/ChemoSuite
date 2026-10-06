"""
tests/test_fticr.py — Модульные тесты ядра FT-ICR MS (fticr_core.py).
"""
import numpy as np
import pandas as pd
import pytest

import fticr_core


def test_fast_formula_assigner():
    """Тест приписки брутто-формул по точной массе в ESI(-)."""
    # Ванилиновая кислота C8H8O4: M_neut = 168.042259, ESI(-) [M-H]- = 167.034983
    # Кофейная кислота C9H8O4: M_neut = 180.042259, ESI(-) [M-H]- = 179.034983
    m1 = 8 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276  # 167.034984
    m2 = 9 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276  # 179.034984

    peaks_df = pd.DataFrame({
        "mass": [m1, m2, 300.12345],  # 3-я масса заведомо без формулы при жестких границах
        "intensity": [50000.0, 80000.0, 1000.0],
    })

    bounds = {"C": (4, 15), "H": (4, 25), "O": (1, 10), "N": (0, 0), "S": (0, 0)}
    res = fticr_core.fast_formula_assigner(
        peaks_df=peaks_df,
        bounds=bounds,
        max_hc=2.5,
        max_oc=1.2,
        ppm_tolerance=2.0,
        ion_mode="ESI(-)",
        max_charge=1,
    )

    assert not res.empty
    assert len(res) >= 2
    formulas = list(res["Formula"].values)
    assert "C8H8O4" in formulas
    assert "C9H8O4" in formulas
    assert all(abs(err) <= 2.0 for err in res["error_ppm"])


def test_calculate_descriptors():
    """Тест расчета химических индексов (H/C, O/C, DBE, AI, NOSC)."""
    df = pd.DataFrame({
        "C": [10, 20],
        "H": [10, 30],
        "O": [4, 6],
        "N": [1, 0],
        "S": [0, 1],
    })

    res = fticr_core.calculate_descriptors(df, lang="ru")

    assert "H/C" in res.columns
    assert "O/C" in res.columns
    assert "DBE" in res.columns
    assert "AI" in res.columns
    assert "NOSC" in res.columns
    assert "Hetero_Class" in res.columns
    assert "Compound_Class" in res.columns

    # C10H10O4N1: H/C = 1.0, O/C = 0.4, DBE = 1 + 10 - 5 + 0.5 = 6.5
    assert np.isclose(res.loc[0, "H/C"], 1.0)
    assert np.isclose(res.loc[0, "O/C"], 0.4)
    assert np.isclose(res.loc[0, "DBE"], 6.5)
    assert res.loc[0, "Hetero_Class"] == "CHON"
    assert res.loc[1, "Hetero_Class"] == "CHOS"


def test_compute_kmd():
    """Тест вычисления дефекта массы Кендрика (KMD)."""
    masses = np.array([200.05, 214.06565, 228.0813])  # гомологическая серия CH2
    km, kmd, nkm = fticr_core.compute_kmd(masses, base_key="CH2")

    assert len(km) == 3
    assert len(kmd) == 3
    assert len(nkm) == 3
    # Разница в KMD между членами истинной серии CH2 должна быть близка к нулю
    diff_kmd = np.diff(kmd)
    assert np.all(np.abs(diff_kmd) < 0.005)


def test_compute_vk20_grid():
    """Тест сетки хемотипирования 20 ячеек Ван-Кревелена Перминовой."""
    # 4 точки в разных ячейках
    df = pd.DataFrame({
        "O/C": [0.1, 0.4, 0.6, 0.9],
        "H/C": [0.4, 0.8, 1.2, 1.6],
        "intensity": [100.0, 200.0, 300.0, 400.0],
    })

    pct_count, pct_weight, grid_df = fticr_core.compute_vk20_grid(df)

    assert pct_count.shape == (5, 4)
    assert pct_weight.shape == (5, 4)
    assert len(grid_df) == 20
    assert np.isclose(np.sum(pct_count), 100.0)
    assert np.isclose(np.sum(pct_weight), 100.0)


def test_spectral_algebra():
    """Тест спектральной алгебры (A-B, A+B, A ∩ B)."""
    df_a = pd.DataFrame({"mass": [100.0, 200.0, 300.0], "intensity": [10, 20, 30]})
    df_b = pd.DataFrame({"mass": [200.0, 300.0, 400.0], "intensity": [25, 35, 45]})

    diff_ab = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A - B", ppm_tol=5.0)
    assert len(diff_ab) == 1
    assert np.isclose(diff_ab.iloc[0]["mass"], 100.0)

    inter = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A ∩ B", ppm_tol=5.0)
    assert len(inter) == 2
    assert set(np.round(inter["mass"].values, 1)) == {200.0, 300.0}

    union = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A + B", ppm_tol=5.0)
    assert len(union) == 4


def test_tmds_screening():
    """Тест скрининга массовых разностей TMDS."""
    # Создаем пару с разницей массы в CH2 (14.01565 Да)
    m1 = 200.00000
    m2 = m1 + 14.01565
    peaks_df = pd.DataFrame({"mass": [m1, m2], "intensity": [100.0, 200.0]})

    summary, pairs = fticr_core.run_tmds_screening(peaks_df, top_n=10, tol_mda=2.0)
    assert not summary.empty
    ch2_hit = summary[summary["Transformation"].str.contains("CH2")]
    assert not ch2_hit.empty
    assert ch2_hit.iloc[0]["Count"] == 1
    assert len(pairs) == 1
