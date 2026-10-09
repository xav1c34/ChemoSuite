"""
tests/test_ftir.py — Модульные тесты математического ядра колебательной спектроскопии (ftir_core.py).
"""
import numpy as np
import pandas as pd
import pytest

import ftir_core


def test_parse_ftir_file_absorbance_and_transmittance():
    """Тест парсинга текстовых файлов FTIR в шкалах Absorbance и Transmittance (%T)."""
    # 1. Текстовый спектр в единицах оптической плотности A с метаданными и запятыми
    raw_abs_text = """# Bruker OPUS Export FTIR
# Date: 2026-10-09
# Sample: Baikal_Water_01
4000.0, 0.05
3400.5, 0.45
2920.0, 0.25
1720.0, 0.15
1600.0, 0.20
1030.0, 0.35
650.0,  0.08
"""
    sample_a = ftir_core.parse_ftir_file(raw_abs_text, sample_id="Test_Abs")
    assert sample_a.sample_id == "Test_Abs"
    assert sample_a.signal_type == "absorbance"
    assert len(sample_a.wavenumbers) == 7
    # Волновые числа должны быть отсортированы по возрастанию
    assert np.all(np.diff(sample_a.wavenumbers) > 0)
    assert pytest.approx(sample_a.wavenumbers[0], 0.1) == 650.0
    assert pytest.approx(sample_a.wavenumbers[-1], 0.1) == 4000.0

    # 2. Текстовый спектр в единицах пропускания %T (должен авто-конвертироваться в A)
    raw_t_text = """Wavenumber\t%T
650.0\t85.0
1030.0\t45.0
1510.0\t35.0
1600.0\t30.0
1720.0\t25.0
2920.0\t55.0
4000.0\t90.0
"""
    sample_t = ftir_core.parse_ftir_file(raw_t_text, sample_id="Test_Trans")
    assert sample_t.signal_type == "transmittance"
    assert len(sample_t.absorbance) == 7
    # При 25% пропускания A = 2 - log10(25) ≈ 0.602
    idx_1720 = int(np.argmin(np.abs(sample_t.wavenumbers - 1720.0)))
    assert pytest.approx(sample_t.absorbance[idx_1720], 0.05) == 0.602


def test_baseline_als_and_subtraction():
    """Тест коррекции базовой линии AsLS (Asymmetric Least Squares)."""
    wn = np.linspace(650.0, 4000.0, 1000)
    # Истинный фоновый дрейф + два пика
    true_bg = 0.2 + 0.5 * ((wn - 650.0) / 3350.0) ** 2
    peak1 = 0.8 * np.exp(-0.5 * ((wn - 1600.0) / 25.0) ** 2)
    peak2 = 0.6 * np.exp(-0.5 * ((wn - 2920.0) / 30.0) ** 2)
    signal = true_bg + peak1 + peak2

    sample = ftir_core.FTIRSample(
        sample_id="Synthetic_Baseline_Test",
        wavenumbers=wn,
        absorbance=signal,
        raw_signal=signal.copy()
    )

    sample_corr, base = ftir_core.subtract_baseline(sample, method="als", lam=1e5, p=0.001)
    assert sample_corr.baseline is not None
    assert sample_corr.corrected_absorbance is not None
    # Базовая линия должна лежать преимущественно ниже или на уровне сигнала
    assert np.mean(sample_corr.absorbance >= base - 1e-3) > 0.95
    # На краях после вычитания фон близок к 0
    assert sample_corr.corrected_absorbance[0] < 0.05
    assert sample_corr.corrected_absorbance[-1] < 0.05
    # Пики должны сохранить свою структуру
    idx_peak1 = int(np.argmin(np.abs(wn - 1600.0)))
    assert sample_corr.corrected_absorbance[idx_peak1] > 0.70


def test_smoothing_and_normalization():
    """Тест фильтра Савицкого — Голея и спектральных нормировок."""
    x = np.linspace(1000, 2000, 500)
    y = np.sin(x / 50.0) + np.random.normal(0, 0.05, 500)

    # 1. Сглаживание
    smoothed = ftir_core.smooth_savitzky_golay(y, window_length=15, polyorder=2)
    assert len(smoothed) == len(y)
    # Шум должен уменьшиться
    assert np.std(np.diff(smoothed)) < np.std(np.diff(y))

    # 2. Нормировка SNV
    y_snv = ftir_core.normalize_spectrum(smoothed, method="snv")
    assert pytest.approx(float(np.mean(y_snv)), abs=1e-5) == 0.0
    assert pytest.approx(float(np.std(y_snv)), abs=1e-5) == 1.0

    # 3. Нормировка MinMax
    y_mm = ftir_core.normalize_spectrum(smoothed, method="minmax")
    assert pytest.approx(float(np.min(y_mm)), abs=1e-5) == 0.0
    assert pytest.approx(float(np.max(y_mm)), abs=1e-5) == 1.0


def test_calculate_ftir_indices_distinction():
    """Тест расчета хемометрических индексов для природного РОВ и шлам-лигнина."""
    wn = np.linspace(650.0, 4000.0, 1000)

    # Создаем спектр чистого РОВ (много алифатики и углеводов, мало лигнина)
    abs_pristine = (
        0.50 * np.exp(-0.5 * ((wn - 2920.0) / 30.0) ** 2) +
        0.20 * np.exp(-0.5 * ((wn - 1720.0) / 25.0) ** 2) +
        0.25 * np.exp(-0.5 * ((wn - 1600.0) / 25.0) ** 2) +
        0.05 * np.exp(-0.5 * ((wn - 1510.0) / 15.0) ** 2) +
        0.60 * np.exp(-0.5 * ((wn - 1030.0) / 45.0) ** 2)
    )

    # Создаем спектр шлам-лигнина (мощный пик 1510 см^-1, карбоксилы 1720, ароматика 1600)
    abs_lignin = (
        0.20 * np.exp(-0.5 * ((wn - 2920.0) / 30.0) ** 2) +
        0.65 * np.exp(-0.5 * ((wn - 1720.0) / 25.0) ** 2) +
        0.70 * np.exp(-0.5 * ((wn - 1600.0) / 25.0) ** 2) +
        0.60 * np.exp(-0.5 * ((wn - 1510.0) / 15.0) ** 2) +
        0.25 * np.exp(-0.5 * ((wn - 1030.0) / 45.0) ** 2)
    )

    ind_pristine = ftir_core.calculate_ftir_indices(wn, abs_pristine, sample_id="Pristine")
    ind_lignin = ftir_core.calculate_ftir_indices(wn, abs_lignin, sample_id="Lignin")

    # Проверка ключевых различий
    # 1. Лигниновый индекс I_1510 / I_1600 у шлам-лигнина существенно выше
    assert ind_lignin["FTIR_Lignin_Idx"] > 0.70
    assert ind_pristine["FTIR_Lignin_Idx"] < 0.30

    # 2. Индекс ароматичности I_1600 / I_2920 выше у шлам-лигнина
    assert ind_lignin["FTIR_Aromaticity_Idx"] > ind_pristine["FTIR_Aromaticity_Idx"]

    # 3. Полисахаридный индекс выше у природного РОВ
    assert ind_pristine["FTIR_Polysaccharide_Idx"] > ind_lignin["FTIR_Polysaccharide_Idx"]


def test_deconvolve_carbonyl_region():
    """Тест математической деконволюции диапазона 1500–1800 см^-1."""
    wn = np.linspace(1500.0, 1800.0, 301)
    # Создаем синтетическую суперпозицию 5 известных гауссиан
    p1 = 0.50 * np.exp(-0.5 * ((wn - 1510.0) / 12.0) ** 2)  # Лигнин
    p2 = 0.60 * np.exp(-0.5 * ((wn - 1605.0) / 15.0) ** 2)  # Ароматика
    p3 = 0.35 * np.exp(-0.5 * ((wn - 1655.0) / 14.0) ** 2)  # Хиноиды
    p4 = 0.55 * np.exp(-0.5 * ((wn - 1715.0) / 16.0) ** 2)  # Карбоксилы
    p5 = 0.20 * np.exp(-0.5 * ((wn - 1740.0) / 10.0) ** 2)  # Сложные эфиры
    y_composite = p1 + p2 + p3 + p4 + p5

    res = ftir_core.deconvolve_carbonyl_region(wn, y_composite, n_peaks=5)

    assert res["r2"] > 0.98
    assert res["rmse"] < 0.05
    assert len(res["peaks_df"]) == 5
    assert "Relative_Area_pct" in res["peaks_df"].columns
    # Сумма относительных площадей должна составлять 100%
    assert pytest.approx(float(res["peaks_df"]["Relative_Area_pct"].sum()), 0.5) == 100.0


def test_synthetic_ftir_dataset_generation():
    """Тест генерации эталонного бенчмарка FTIR для двух классов."""
    samples_dict, df_desc = ftir_core.generate_synthetic_ftir_dataset(n_samples=10, random_state=42)

    assert len(samples_dict) == 10
    assert len(df_desc) == 10
    assert "FTIR_Lignin_Idx" in df_desc.columns
    assert "Sample_Type" in df_desc.columns

    # Разделение по типам
    lignin_rows = df_desc[df_desc["Sample_Type"] == "Sludge Lignin"]
    pristine_rows = df_desc[df_desc["Sample_Type"] == "Pristine Water"]

    assert len(lignin_rows) == 5
    assert len(pristine_rows) == 5

    # Средний Lignin_Idx должен достоверно разделять классы (p < 0.001)
    mean_lig = lignin_rows["FTIR_Lignin_Idx"].mean()
    mean_pris = pristine_rows["FTIR_Lignin_Idx"].mean()
    assert mean_lig > mean_pris * 2.0


def test_find_and_annotate_peaks():
    """Тест автоматического поиска пиков и аннотации характеристических полос."""
    wn = np.linspace(650.0, 4000.0, 1676)
    # Создаем спектр с четкими пиками при 1510, 1605, 1718, 2924 и 3390 см^-1
    y = (
        0.55 * np.exp(-0.5 * ((wn - 3390.0) / 60.0) ** 2) +
        0.45 * np.exp(-0.5 * ((wn - 2924.0) / 15.0) ** 2) +
        0.60 * np.exp(-0.5 * ((wn - 1718.0) / 14.0) ** 2) +
        0.65 * np.exp(-0.5 * ((wn - 1605.0) / 15.0) ** 2) +
        0.58 * np.exp(-0.5 * ((wn - 1510.0) / 12.0) ** 2) +
        0.40 * np.exp(-0.5 * ((wn - 1035.0) / 25.0) ** 2)
    )

    df_peaks_ru = ftir_core.find_and_annotate_peaks(wn, y, prominence=0.1, lang="ru")
    assert not df_peaks_ru.empty
    assert len(df_peaks_ru) >= 5

    # Проверка обнаружения маркера лигнина 1510 см^-1
    lignin_peaks = df_peaks_ru[df_peaks_ru["Functional_Group"] == "guaiacyl lignin C=C"]
    assert len(lignin_peaks) == 1
    assert pytest.approx(float(lignin_peaks.iloc[0]["Wavenumber_cm1"]), 2.0) == 1510.0
    assert "Шлам-лигнин" in lignin_peaks.iloc[0]["Compound_Class"]

    # Проверка карбонильной полосы 1718 см^-1
    carb_peaks = df_peaks_ru[df_peaks_ru["Functional_Group"] == "carboxylic C=O stretching"]
    assert len(carb_peaks) == 1
    assert pytest.approx(float(carb_peaks.iloc[0]["Wavenumber_cm1"]), 2.0) == 1718.0

    # Проверка аннотации на английском языке
    df_peaks_en = ftir_core.find_and_annotate_peaks(wn, y, prominence=0.1, lang="en")
    assert df_peaks_en.iloc[0]["Identification_Status"] == "Verified"


def test_spectral_library_matching():
    """Тест спектрального сопоставления с эталонной библиотекой (HQI и Cosine Similarity)."""
    # 1. Генерируем тестовый спектр, близкий к сульфатному шлам-лигнину
    wn_grid = np.linspace(650.0, 4000.0, 1676)
    _, kraft_abs, _ = ftir_core.get_reference_spectrum("Kraft_Lignin_BCBK", wn_grid=wn_grid)
    # Добавим небольшой шум
    noisy_kraft = kraft_abs + np.random.normal(0, 0.01, size=len(wn_grid))

    match_df = ftir_core.match_spectrum_against_library(wn_grid, noisy_kraft, top_n=3, lang="ru")
    assert not match_df.empty
    # Топ-1 совпадение должно быть именно крафт-лигнином
    assert match_df.iloc[0]["Standard_ID"] == "Kraft_Lignin_BCBK"
    assert match_df.iloc[0]["HQI_pct"] > 95.0
    assert match_df.iloc[0]["Cosine_Similarity"] > 0.97
    assert match_df.iloc[0]["Rank"] == 1

    # 2. Проверяем эталон целлюлозы
    _, cell_abs, _ = ftir_core.get_reference_spectrum("Cellulose_Standard", wn_grid=wn_grid)
    match_cell = ftir_core.match_spectrum_against_library(wn_grid, cell_abs, top_n=2, lang="en")
    assert match_cell.iloc[0]["Standard_ID"] == "Cellulose_Standard"
    assert match_cell.iloc[0]["HQI_pct"] > 99.0

