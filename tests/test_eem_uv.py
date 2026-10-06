"""
tests/test_eem_uv.py — Модульные тесты оптической спектроскопии (eem_core.py).
"""
import numpy as np
import pandas as pd
import pytest

import eem_core


def test_eem_parsing_and_orientation():
    """Тест парсинга EEM матрицы и автоматической ориентации осей (Em — строки, Ex — столбцы)."""
    ex_vals = np.array([250, 300, 350, 400])
    em_vals = np.array([300, 350, 400, 450, 500])
    raw_data = np.random.uniform(10, 100, (len(em_vals), len(ex_vals)))

    df_eem = pd.DataFrame(raw_data, index=em_vals, columns=ex_vals)
    df_eem.index.name = "Em"

    sample = eem_core.parse_eem_dataframe(df_eem, sample_id="Test_Sample")
    assert sample.sample_id == "Test_Sample"
    assert len(sample.ex) == len(ex_vals)
    assert len(sample.em) == len(em_vals)
    assert sample.data.shape == (len(em_vals), len(ex_vals))


def test_eem_scatter_removal():
    """Тест удаления рэлеевского рассеяния 1-го и 2-го порядков."""
    ex_vals = np.arange(250, 400, 20)
    em_vals = np.arange(250, 550, 10)
    EX, EM = np.meshgrid(ex_vals, em_vals)

    # Искусственное рэлеевское рассеяние
    data = 100.0 + 5000.0 * np.exp(-0.5 * ((EM - EX) / 8.0) ** 2)

    cleaned = eem_core.remove_scatter_bands(data, ex_vals, em_vals, delta_rayleigh1=15.0, delta_rayleigh2=15.0)
    assert cleaned.shape == data.shape
    # В зоне пика рассеяния значение должно существенно снизиться за счет интерполяции
    rayleigh_zone = (EM >= EX - 10) & (EM <= EX + 10)
    assert np.mean(cleaned[rayleigh_zone]) < np.mean(data[rayleigh_zone])


def test_eem_spectral_indices():
    """Тест расчета оптических индексов флуоресценции (FI, HIX)."""
    ex_vals = np.arange(250, 450, 5)
    em_vals = np.arange(300, 600, 5)
    # Создаем флуоресцентную матрицу с максимумом при Ex 370, Em 450
    EM, EX = np.meshgrid(em_vals, ex_vals, indexing="ij")
    data = 1000.0 * np.exp(-0.5 * (((EX - 370) / 30) ** 2 + ((EM - 450) / 40) ** 2))

    sample = eem_core.EEMSample(sample_id="Index_Test", ex=ex_vals, em=em_vals, data=data)
    indices = eem_core.calculate_spectral_indices(sample)

    assert "FI" in indices
    assert "HIX" in indices
    assert not np.isnan(indices["FI"])
    assert not np.isnan(indices["HIX"])
    assert indices["FI"] > 0
    assert indices["HIX"] > 0


def test_uv_vis_indices_and_derivatives():
    """Тест расчета дескрипторов УФ-Вид, производных и уравнения Перминовой."""
    wl = np.linspace(200, 700, 501)
    # Экспоненциальный спад + гауссово плечо лигнина при 280 нм
    a_exp = 0.6 * np.exp(-0.015 * (wl - 254.0))
    a_lignin = 0.08 * np.exp(-0.5 * ((wl - 280.0) / 10.0) ** 2)
    absorbance = a_exp + a_lignin

    sample = eem_core.UVVisSample(sample_id="UV_Test", wl=wl, absorbance=absorbance, doc=5.0)

    # 1. Расчет оптических индексов
    indices = eem_core.calculate_uv_vis_indices(sample, doc=5.0, pathlength_cm=1.0)
    assert "A254" in indices
    assert "A280" in indices
    assert "E2_E3" in indices
    assert "S_R" in indices
    assert "Mw_est" in indices
    assert "SUVA254" in indices
    assert not np.isnan(indices["A254"])
    assert not np.isnan(indices["E2_E3"])
    assert not np.isnan(indices["Mw_est"])

    # 2. Оценка молекулярной массы (уравнение Перминовой: Mw = 3450 - 390 * E2/E3)
    e2_e3 = indices["E2_E3"]
    expected_mw = 3450.0 - 390.0 * e2_e3
    assert np.isclose(indices["Mw_est"], np.clip(expected_mw, 400.0, 15000.0), atol=2.0)

    # 3. Производная спектроскопия (Савицкий — Голей)
    derivs = eem_core.compute_uv_derivatives(wl, absorbance)
    assert "d1_a" in derivs
    assert "d2_a" in derivs
    assert "lignin_min_wl" in derivs
    assert "lignin_d2_val" in derivs
    # Минимум второй производной должен лежать в районе 280 нм
    assert 270.0 <= derivs["lignin_min_wl"] <= 290.0
    assert derivs["lignin_d2_val"] < 0.0


def test_inner_filter_effect_correction():
    """Тест коррекции эффекта внутреннего фильтра (IFE) для EEM."""
    ex = np.array([250.0, 300.0, 350.0])
    em = np.array([350.0, 400.0, 450.0])
    eem_data = np.ones((len(em), len(ex))) * 100.0

    uv_wl = np.linspace(200, 500, 301)
    uv_abs = np.linspace(0.5, 0.1, 301)  # поглощение убывает с длиной волны

    eem_corr, cf_matrix, warning = eem_core.correct_inner_filter_effect(
        eem=eem_data, ex=ex, em=em, uv_wl=uv_wl, uv_a=uv_abs, pathlength_cm=1.0
    )

    # Скорректированная флуоресценция должна быть строго больше наблюдаемой (F_corr > F_obs)
    assert np.all(eem_corr >= eem_data)
    assert warning is None
    assert cf_matrix.shape == eem_data.shape


def test_link_uv_vis_to_eem():
    """Тест связывания спектров УФ-Вид с матрицами EEM."""
    ex = np.array([250.0, 300.0])
    em = np.array([350.0, 400.0])
    eem_sample = eem_core.EEMSample(sample_id="Sample_01", ex=ex, em=em, data=np.ones((2, 2)))

    uv_wl = np.linspace(200, 500, 301)
    uv_abs = 0.5 * np.exp(-0.01 * (uv_wl - 200))
    uv_sample = eem_core.UVVisSample(sample_id="Sample_01_uv", wl=uv_wl, absorbance=uv_abs, doc=4.0)

    logs = eem_core.link_uv_vis_to_eem([eem_sample], [uv_sample], doc_map={"Sample_01": 4.0}, apply_ife=True)

    assert len(logs) == 1
    assert eem_sample.a254 is not None
    assert eem_sample.doc == 4.0
    assert logs[0]["IFE_Applied"] is True
