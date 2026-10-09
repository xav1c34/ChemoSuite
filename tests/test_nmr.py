"""
tests/test_nmr.py — Комплексный набор модульных тестов для ядра ЯМР-спектроскопии (13C / 1H NMR).
Проверяет:
1. Парсинг табличных форматов (CSV, TXT, XY) и JCAMP-DX (.jdx).
2. Вычитание базовой линии (AsLS, линейная, сплайн).
3. Интегрирование по каноническим регионам IHSS и расчет хемометрических индексов (fa, HB/HI, A/OA).
4. Деконволюцию сложной ароматико-фенольной зоны (108–165 ppm) на гауссовы компоненты.
5. Автопоиск и отнесение пиков (Peak Picking & Functional Group Assignment).
6. Идентификацию по библиотеке стандартов IHSS и шлам-лигнина (HQI %).
7. Пакетную обработку и генерацию эталонного датасета demo_data/nmr.
8. Интеграцию с Data Fusion (chemo_ml), экспортом в Excel (report_generator) и проектом (.chemo).
"""
import io
import os
import numpy as np
import pandas as pd
import pytest

import nmr_core
import chemo_ml
import project_io
import report_generator


def test_parse_nmr_file_table_and_jcamp():
    """Тест парсинга текстовых таблиц и международного формата JCAMP-DX."""
    # 1. Текстовый CSV с европейскими запятыми
    csv_content = """# Bruker 13C CP/MAS NMR Export
Chemical Shift (ppm);Intensity
210,5;12,4
175,2;85,6
148,0;42,1
130,1;98,3
56,2;75,4
30,0;110,2
15,1;35,0
"""
    sample = nmr_core.parse_nmr_file(csv_content, sample_id="Test_Sample_CSV")
    assert sample.sample_id == "Test_Sample_CSV"
    assert sample.nucleus == "13C"
    assert len(sample.chemical_shifts) == 7
    # Проверка упорядоченности по возрастанию ppm
    assert np.all(np.diff(sample.chemical_shifts) > 0)
    assert pytest.approx(sample.chemical_shifts[0], 0.01) == 15.1
    assert pytest.approx(sample.chemical_shifts[-1], 0.01) == 210.5

    # 2. Формат IUPAC JCAMP-DX
    jcamp_content = """##TITLE=13C CP/MAS NMR of Baikal NOM
##JCAMP-DX=5.01
##DATA TYPE=NMR SPECTRUM
##.OBSERVE NUCLEUS=13C
##XUNITS=PPM
##YUNITS=ARBITRARY UNITS
##FIRSTX=220.0
##LASTX=0.0
##NPOINTS=5
##PEAK TABLE=(XY..XY)
205.0, 10.5
174.5, 45.2
130.0, 60.1
56.0, 30.2
25.0, 80.0
##END=
"""
    s_jcamp = nmr_core.parse_nmr_file(jcamp_content, sample_id="Test_JCAMP")
    assert s_jcamp.nucleus == "13C"
    assert len(s_jcamp.chemical_shifts) == 5
    assert np.all(np.diff(s_jcamp.chemical_shifts) > 0)
    assert pytest.approx(s_jcamp.chemical_shifts[0], 0.01) == 25.0
    assert pytest.approx(s_jcamp.chemical_shifts[-1], 0.01) == 205.0


def test_baseline_als_and_normalization():
    """Тест AsLS коррекции базовой линии и нормализации спектра ЯМР."""
    x = np.linspace(0.0, 220.0, 221)
    # Пик на 130 ppm с линейным трендом фона
    background = 0.5 * x + 10.0
    signal = 50.0 * np.exp(-((x - 130.0) ** 2) / 50.0)
    y_raw = background + signal

    sample = nmr_core.NMRSample(sample_id="Base_Test", chemical_shifts=x, intensity=y_raw)
    sample_corr, base_curve = nmr_core.subtract_baseline(sample, method="asls", lam=1e5, p=0.01)

    assert len(base_curve) == len(x)
    assert sample_corr.corrected_intensity is not None
    # Базовая линия должна убрать фон на краях (на 10 и 210 ppm сигнал должен быть близок к 0)
    assert sample_corr.corrected_intensity[10] < 5.0
    assert sample_corr.corrected_intensity[210] < 5.0
    # Пик на 130 ppm должен сохраниться
    peak_idx = np.argmin(np.abs(x - 130.0))
    assert sample_corr.corrected_intensity[peak_idx] > 35.0

    # Проверка нормализации по площади (Area = 100%)
    norm_y = nmr_core.normalize_nmr_spectrum(sample_corr.corrected_intensity, method="area", chemical_shifts=x)
    tot_area = nmr_core.simpson(norm_y, x=x)
    assert pytest.approx(tot_area, rel=0.05) == 100.0


def test_calculate_13c_nmr_descriptors_distinction():
    """Тест расчета дескрипторов 13C ЯМР и дискриминации чистого NOM от лигнина."""
    ppm_grid = np.linspace(0.0, 220.0, 1101)

    # Модельный спектр чистого РОВ (высокие сахара O-alkyl 72 ppm, низкая ароматичность)
    nom_curve = (
        0.3 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=30.0, wid=8.0) +
        0.5 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=72.0, wid=10.0) +
        0.2 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=130.0, wid=8.0) +
        0.2 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=175.0, wid=6.0)
    )
    desc_nom = nmr_core.calculate_13c_nmr_descriptors(ppm_grid, nom_curve, sample_id="Baikal_NOM")

    # Модельный спектр лигнина (высокая ароматичность 130 ppm, фенолы 148 ppm, метоксилы 56 ppm)
    lignin_curve = (
        0.1 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=30.0, wid=6.0) +
        0.4 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=56.0, wid=3.0) +
        0.6 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=130.0, wid=10.0) +
        0.4 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=148.0, wid=4.0) +
        0.2 * nmr_core.gaussian(ppm_grid, amp=1.0, cen=175.0, wid=5.0)
    )
    desc_lignin = nmr_core.calculate_13c_nmr_descriptors(ppm_grid, lignin_curve, sample_id="Sludge_Lignin")

    # 1. Сумма долей 5 главных областей должна составлять 100%
    sum_nom = (desc_nom["NMR_C_aliph_pct"] + desc_nom["NMR_C_O_alkyl_pct"] +
               desc_nom["NMR_C_aromatic_pct"] + desc_nom["NMR_C_carboxyl_pct"] +
               desc_nom["NMR_C_carbonyl_pct"])
    assert pytest.approx(sum_nom, 0.5) == 100.0

    # 2. Ароматичность лигнина должна быть существенно выше чистого NOM
    assert desc_lignin["NMR_fa_aromaticity"] > desc_nom["NMR_fa_aromaticity"]
    assert desc_lignin["NMR_fa_aromaticity"] > 40.0

    # 3. Фенольный углерод и метоксилы лигнина должны быть выше
    assert desc_lignin["NMR_C_phenolic_pct"] > desc_nom["NMR_C_phenolic_pct"]
    assert desc_lignin["NMR_C_methoxy_pct"] > desc_nom["NMR_C_methoxy_pct"]
    assert desc_lignin["NMR_lignin_signature"] > desc_nom["NMR_lignin_signature"]

    # 4. Отношение сахаров (C_carbohydrate) в чистом NOM должно быть выше
    assert desc_nom["NMR_C_carbohydrate_pct"] > desc_lignin["NMR_C_carbohydrate_pct"]


def test_deconvolve_aromatic_region():
    """Тест деконволюции ароматико-фенольной зоны 108–165 ppm на гауссовы компоненты."""
    x = np.linspace(100.0, 180.0, 401)
    # Создаем 3 четкие полосы: 115, 130, 148 ppm
    y = (
        0.4 * nmr_core.gaussian(x, amp=1.0, cen=115.0, wid=3.5) +
        0.8 * nmr_core.gaussian(x, amp=1.0, cen=130.0, wid=4.5) +
        0.5 * nmr_core.gaussian(x, amp=1.0, cen=148.0, wid=3.5)
    )
    res = nmr_core.deconvolve_aromatic_region(x, y, n_peaks=3)
    assert res["success"] is True
    assert res["r2"] > 0.98
    assert res["rmse"] < 0.05
    assert len(res["components"]) == 3
    centers = [c["center_ppm"] for c in res["components"]]
    assert any(abs(c - 115.0) < 3.0 for c in centers)
    assert any(abs(c - 130.0) < 3.0 for c in centers)
    assert any(abs(c - 148.0) < 3.0 for c in centers)


def test_find_and_annotate_nmr_peaks():
    """Тест поиска и отнесения пиков 13C ЯМР к функциональным группам."""
    x = np.linspace(0.0, 220.0, 1101)
    # Сигнал с известными резонансами: 56.1 (метоксил), 148.0 (гваяцил фенол), 174.5 (карбоксил)
    y = (
        0.7 * nmr_core.gaussian(x, amp=1.0, cen=56.1, wid=2.0) +
        0.8 * nmr_core.gaussian(x, amp=1.0, cen=148.0, wid=3.0) +
        0.6 * nmr_core.gaussian(x, amp=1.0, cen=174.5, wid=3.0)
    )
    df_peaks = nmr_core.find_and_annotate_nmr_peaks(x, y, prominence=0.1, distance_ppm=5.0, lang="ru")
    assert not df_peaks.empty
    assert len(df_peaks) == 3
    # Проверка обнаружения метоксила
    methoxy_row = df_peaks[df_peaks["Chemical_Shift_ppm"].between(54.0, 58.0)]
    assert not methoxy_row.empty
    assert "Метоксильн" in methoxy_row.iloc[0]["Assignment"] or "-OCH3" in methoxy_row.iloc[0]["Assignment"]


def test_spectral_library_matching():
    """Тест поиска по библиотеке эталонов IHSS и шлам-лигнина."""
    ppm_grid = np.linspace(0.0, 220.0, 1101)
    # Генерируем тестовый спектр шлам-лигнина
    test_lignin = nmr_core.synthesize_reference_13c_curve("BAIKAL_SLUDGE", ppm_grid)
    hits_df = nmr_core.match_spectrum_against_library(ppm_grid, test_lignin, top_n=3, lang="ru")
    assert not hits_df.empty
    top_hit = hits_df.iloc[0]
    assert top_hit["Reference_ID"] == "BAIKAL_SLUDGE"
    assert top_hit["HQI_Similarity_Pct"] > 99.0


def test_synthetic_nmr_dataset_generation_and_batch_processing(tmp_path):
    """Тест генерации демо-датасета 12 образцов и сохранения в demo_data/nmr."""
    demo_dir = os.path.join(str(tmp_path), "demo_nmr")
    samples_dict, df_desc = nmr_core.generate_synthetic_nmr_dataset(n_samples_per_class=6, save_dir=demo_dir)

    assert len(samples_dict) == 12
    assert len(df_desc) == 12
    assert "Class_Target" in df_desc.columns
    assert "NMR_fa_aromaticity" in df_desc.columns

    # Проверка созданных файлов на диске
    files = os.listdir(demo_dir)
    assert len(files) == 13  # 12 CSV спектров + 1 дескрипторная матрица
    assert "Baikal_Pristine_01.csv" in files
    assert "Sludge_Lignin_01.csv" in files
    assert "nmr_descriptors_matrix.csv" in files

    # Пакетная обработка через batch_process_nmr_samples
    df_batch = nmr_core.batch_process_nmr_samples(list(samples_dict.values()))
    assert len(df_batch) == 12
    assert "Sample_ID" in df_batch.columns
    assert "NMR_hb_hi_index" in df_batch.columns


def test_chemo_ml_and_project_io_nmr_integration():
    """Тест классификации признаков NMR_ в chemo_ml и сериализации в .chemo проекте."""
    # 1. chemo_ml.classify_feature_blocks
    cols = ["NMR_fa_aromaticity", "NMR_C_phenolic_pct", "FTIR_Lignin_Idx", "VK_1_Lipids", "A254"]
    b_map = chemo_ml.classify_feature_blocks(cols)
    assert b_map["NMR_fa_aromaticity"] == "NMR"
    assert b_map["NMR_C_phenolic_pct"] == "NMR"
    assert b_map["FTIR_Lignin_Idx"] == "FTIR"
    assert b_map["VK_1_Lipids"] == "FT-ICR MS"
    assert b_map["A254"] == "UV-Vis"

    # 2. Сериализация и восстановление в project_io
    ppm_grid = np.linspace(0.0, 220.0, 100)
    samples_dict, df_desc = nmr_core.generate_synthetic_nmr_dataset(n_samples_per_class=2)
    session_data = {
        "spectra_db": {},
        "nmr_ml_descriptors": df_desc,
        "fused_data": df_desc,
        "fused_source": "Test_NMR_Session",
        "lang": "ru"
    }

    proj_bytes = project_io.save_chemo_project(session_data, project_name="NMR_Test_Project")
    assert len(proj_bytes) > 500

    summary = project_io.get_chemo_project_summary(proj_bytes)
    assert summary["project_name"] == "NMR_Test_Project"
    assert summary["summary"]["has_nmr"] is True

    restored = project_io.load_chemo_project(proj_bytes)
    assert "nmr_ml_descriptors" in restored
    rest_nmr = restored["nmr_ml_descriptors"]
    assert isinstance(rest_nmr, pd.DataFrame)
    assert len(rest_nmr) == 4
    assert "NMR_fa_aromaticity" in rest_nmr.columns


def test_report_generator_nmr_sheet():
    """Тест генерации многостраничного паспорта Excel с отдельным листом NMR Descriptors."""
    samples_dict, df_desc = nmr_core.generate_synthetic_nmr_dataset(n_samples_per_class=2)
    excel_bytes = report_generator.generate_excel_passport(
        fused_df=df_desc,
        model_results=None,
        metadata={"project_name": "NMR Excel Test"},
        lang="ru"
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 2000

    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Passport & Summary" in sheet_names
    assert "Fused Feature Matrix" in sheet_names
    assert "NMR Descriptors" in sheet_names
