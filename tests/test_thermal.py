"""
tests/test_thermal.py — Модульные и интеграционные тесты для Модуля 6:
Термический анализ и пиролиз (Py-GC-MS & TGA/DTG/DSC).
"""
import io
import os
import numpy as np
import pandas as pd
import pytest

from modules import thermal_core
from modules import chemo_ml
from modules import project_io
from modules import report_generator


def test_match_py_compound():
    """Проверка сопоставления с эталонной библиотекой пиролиза по RT."""
    # Тест точного попадания на Гваякол (~10.5 мин)
    match_g = thermal_core.match_py_compound(10.52, window=0.5, lang="ru")
    assert "Гваякол" in match_g["name"]
    assert match_g["class"] == "G_Lignin"
    assert match_g["cas"] == "90-05-1"
    assert match_g["similarity"] > 0.9

    # Тест попадания на Сирингол (~18.2 мин)
    match_s = thermal_core.match_py_compound(18.18, window=0.5, lang="en")
    assert "Syringol" in match_s["name"]
    assert match_s["class"] == "S_Lignin"

    # Тест попадания на Левоглюкозан (~23.5 мин)
    match_carb = thermal_core.match_py_compound(23.48, window=0.5, lang="ru")
    assert "Левоглюкозан" in match_carb["name"]
    assert match_carb["class"] == "Carbohydrate"

    # Тест неидентифицированного пика
    match_unknown = thermal_core.match_py_compound(55.0, window=0.5, lang="en")
    assert match_unknown["class"] == "Unclassified"
    assert match_unknown["similarity"] == 0.0


def test_parse_py_gcms_raw_tic():
    """Проверка парсинга сырой хроматограммы TIC и автопоиска пиков."""
    rt_grid = np.linspace(5.0, 30.0, 500)
    # Создаем хроматограмму с двумя явными пиками: Гваякол (10.5 мин) и Сирингол (18.2 мин)
    tic = np.zeros_like(rt_grid) + 10.0
    tic += 500.0 * np.exp(-((rt_grid - 10.5) ** 2) / (2 * 0.05 ** 2))
    tic += 300.0 * np.exp(-((rt_grid - 18.2) ** 2) / (2 * 0.05 ** 2))

    df_tic = pd.DataFrame({"Retention_Time": rt_grid, "TIC_Intensity": tic})
    csv_bytes = df_tic.to_csv(index=False).encode("utf-8")

    sample = thermal_core.parse_py_gcms_file(csv_bytes, sample_id="Test_Py_Sample", rt_window=0.5)
    assert sample.sample_id == "Test_Py_Sample"
    assert len(sample.peaks) >= 2

    # Проверяем наличие идентифицированных классов
    classes = [p.chemical_class for p in sample.peaks]
    assert "G_Lignin" in classes
    assert "S_Lignin" in classes

    # Проверяем расчет индексов
    idx = sample.indices
    assert idx["Py_G_pct"] > 0
    assert idx["Py_S_pct"] > 0
    assert idx["Py_S_G_ratio"] > 0


def test_parse_py_gcms_peak_table():
    """Проверка парсинга таблицы интегрированных пиков."""
    table_csv = """Retention_Time,Peak_Area,Compound_Name
10.50,1500.0,Guaiacol
18.20,750.0,Syringol
7.20,250.0,Phenol
5.40,500.0,Furfural
22.80,300.0,Vanillin
24.50,150.0,Acetovanillone
"""
    sample = thermal_core.parse_py_gcms_file(table_csv, sample_id="Table_Sample")
    assert len(sample.peaks) == 6
    idx = sample.indices

    # S/G = 750 / (1500 + 300 + 150) = 750 / 1950 ≈ 0.385
    assert 0.30 <= idx["Py_S_G_ratio"] <= 0.45
    # (Ac/Al)_G = 150 / 300 = 0.50
    assert abs(idx["Py_Ac_Al_G"] - 0.50) < 0.05
    # Lignin/Carbohydrate ratio > 0
    assert idx["Py_L_C_ratio"] > 1.0


def test_tga_dtg_parsing_and_indices():
    """Проверка парсинга термогравиметрии и расчета параметров деструкции."""
    t_grid = np.linspace(25.0, 800.0, 776)
    # Моделируем потерю массы: 10% влага (100°C), 30% алифатика (300°C), 40% ароматика (450°C), остаток 20%
    m = 100.0 - 10.0 / (1.0 + np.exp(-(t_grid - 100.0) / 20.0)) \
              - 30.0 / (1.0 + np.exp(-(t_grid - 300.0) / 30.0)) \
              - 40.0 / (1.0 + np.exp(-(t_grid - 450.0) / 40.0))

    df_tga = pd.DataFrame({"Temperature": t_grid, "Weight_pct": m})
    csv_bytes = df_tga.to_csv(index=False).encode("utf-8")

    sample = thermal_core.parse_tga_file(csv_bytes, sample_id="Test_TGA")
    assert sample.sample_id == "Test_TGA"
    assert len(sample.dtg) == len(t_grid)
    assert np.all(sample.dtg >= 0.0)

    idx = sample.indices
    # Проверяем потери массы по зонам
    assert 7.0 <= idx["TGA_Moisture_Loss_pct"] <= 12.0
    assert 25.0 <= idx["TGA_Aliphatic_Loss_pct"] <= 35.0
    assert 30.0 <= idx["TGA_Aromatic_Loss_pct"] <= 45.0
    # Индекс термостабильности TI = Δm_arom / Δm_aliph ≈ 40 / 30 ≈ 1.33
    assert 1.0 <= idx["TGA_Thermostability_Index"] <= 1.6
    # Пик DTG должен быть в области 280-500 °C
    assert 280.0 <= idx["DTG_T_max1"] <= 500.0


def test_synthetic_thermal_dataset_and_class_distinction():
    """Проверка генерации синтетического бенчмарка и различимости классов."""
    samples, df_matrix = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=6, random_state=42)
    assert len(samples) == 12
    assert len(df_matrix) == 12
    assert "Sample_ID" in df_matrix.columns
    assert "Class_Target" in df_matrix.columns

    # Проверяем наличие ключевых дескрипторов
    expected_cols = [
        "Py_S_G_ratio", "Py_L_C_ratio", "Py_G_pct", "TGA_Thermostability_Index",
        "TGA_Moisture_Loss_pct", "TGA_Char_Yield_600C", "DTG_T_max1"
    ]
    for col in expected_cols:
        assert col in df_matrix.columns

    # Проверяем химически осмысленное разделение классов
    nom_rows = df_matrix[df_matrix["Class_Target"] == "Baikal_Pristine"]
    lignin_rows = df_matrix[df_matrix["Class_Target"] == "Sludge_Lignin"]

    # 1. Шлам-лигнин имеет намного более высокое отношение Лигнин/Углеводы (L/C)
    assert lignin_rows["Py_L_C_ratio"].mean() > nom_rows["Py_L_C_ratio"].mean() * 2.0

    # 2. Шлам-лигнин имеет более высокий индекс термостабильности TI
    assert lignin_rows["TGA_Thermostability_Index"].mean() > nom_rows["TGA_Thermostability_Index"].mean()

    # 3. Шлам-лигнин имеет более высокий коксовый остаток (Char Yield)
    assert lignin_rows["TGA_Char_Yield_600C"].mean() > nom_rows["TGA_Char_Yield_600C"].mean()

    # 4. Фоновое РОВ имеет более высокую потерю гигроскопической влаги
    assert nom_rows["TGA_Moisture_Loss_pct"].mean() > lignin_rows["TGA_Moisture_Loss_pct"].mean()


def test_chemo_ml_thermal_integration():
    """Проверка классификации блока 'Thermal' и обучения PLS-DA модели на мультимодальных данных."""
    # 1. Проверка классификации колонок
    cols = ["Py_S_G_ratio", "Py_L_C_ratio", "TGA_Thermostability_Index", "DTG_T_max1", "TGA_Char_Yield_600C"]
    block_map = chemo_ml.classify_feature_blocks(cols)
    for c in cols:
        assert block_map[c] == "Thermal"

    # 2. Обучение модели на термическом датасете
    _, df_matrix = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=6, random_state=42)
    y_labels = df_matrix["Class_Target"]
    feature_df = df_matrix.drop(columns=["Class_Target"])

    pls_res = chemo_ml.train_plsda_model(feature_df, y_labels, n_components=2)
    assert pls_res["Q2"] > 70.0
    assert pls_res["Balanced_Accuracy"] > 90.0

    # Проверяем расчет VIP-score
    vip_df = pls_res["VIP_df"]
    assert not vip_df.empty
    assert "Block" in vip_df.columns
    assert any(vip_df["Block"] == "Thermal")


def test_project_io_and_report_generator_thermal():
    """Проверка сериализации в .chemo проект и экспорта в аналитический паспорт Excel."""
    _, df_matrix = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=6, random_state=42)

    session_data = {
        "spectra_db": {},
        "eem_ml_descriptors": None,
        "uv_ml_descriptors": None,
        "ftir_ml_descriptors": None,
        "nmr_ml_descriptors": None,
        "thermal_ml_descriptors": df_matrix,
        "fused_data": df_matrix,
        "fused_source": "Thermal Integration Test",
        "pls_results": None,
        "lang": "en",
    }

    # 1. Сохранение проекта
    chemo_bytes = project_io.save_chemo_project(session_data, project_name="Thermal_Test_Project")
    assert isinstance(chemo_bytes, bytes)
    assert len(chemo_bytes) > 0

    # 2. Проверка сводки
    summary = project_io.get_chemo_project_summary(chemo_bytes)
    assert summary["project_name"] == "Thermal_Test_Project"
    assert summary["summary"]["has_thermal"] is True

    # 3. Восстановление проекта
    restored = project_io.load_chemo_project(chemo_bytes)
    assert restored["thermal_ml_descriptors"] is not None
    assert len(restored["thermal_ml_descriptors"]) == 12

    # 4. Генерация паспорта Excel
    excel_bytes = report_generator.generate_excel_passport(
        df_matrix,
        model_results=None,
        metadata={"project_name": "Thermal_Excel_Test"},
        lang="en"
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Проверяем наличие листа 'Thermal Descriptors' в Excel
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Thermal Descriptors" in sheet_names


def test_adversarial_thermal_degraded_inputs():
    """Стресс-тестирование граничных и поврежденных входных данных."""
    # 1. Пустой файл
    empty_py = thermal_core.parse_py_gcms_file("", sample_id="Empty_Py")
    assert empty_py.sample_id == "Empty_Py"
    assert len(empty_py.peaks) == 0
    assert empty_py.indices["Py_S_G_ratio"] == 0.0

    empty_tga = thermal_core.parse_tga_file("", sample_id="Empty_TGA")
    assert empty_tga.sample_id == "Empty_TGA"
    assert len(empty_tga.temperature) == 0
    assert empty_tga.indices["TGA_Thermostability_Index"] == 0.0

    # 2. Текст без числовых данных
    corrupt_txt = "Header only\nSome comments\nNo numbers here\n"
    res_py = thermal_core.parse_py_gcms_file(corrupt_txt)
    assert len(res_py.peaks) == 0

    res_tga = thermal_core.parse_tga_file(corrupt_txt)
    assert len(res_tga.temperature) == 0

    # 3. Изотермические данные (одинаковая температура)
    iso_csv = "Temperature,Weight\n200.0,100.0\n200.0,99.0\n200.0,98.0\n"
    iso_tga = thermal_core.parse_tga_file(iso_csv)
    assert iso_tga.indices["TGA_Thermostability_Index"] >= 0.0


def test_py_gcms_peak_attributes_and_ui_compatibility():
    """Проверка доступности свойств и атрибутов PyGCMSPeak для интерфейса Streamlit."""
    peak = thermal_core.PyGCMSPeak(
        retention_time=12.5,
        area=500.0,
        height=100.0,
        compound_name="Guaiacol",
        chemical_class="G_Lignin",
    )
    # 1. Проверка алиасов свойств
    assert peak.rt == 12.5
    assert peak.peak_area == 500.0
    assert peak.compound_class == "Guaiacyl"
    assert peak.name_en == "Guaiacol"
    assert peak.name_ru == "Guaiacol"
    assert peak.monomer_type == "G_Lignin"

    # 2. Проверка работы с синтетическим датасетом
    samples, _ = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=2, random_state=42)
    s = list(samples.values())[0]
    assert len(s.py_gcms.peaks) > 0
    for p in s.py_gcms.peaks:
        assert hasattr(p, "compound_class")
        assert hasattr(p, "rt")
        assert hasattr(p, "peak_area")
        assert hasattr(p, "area_pct")
        assert p.compound_class in ["Guaiacyl", "Syringyl", "p-Hydroxyphenyl", "Carbohydrate", "Nitrogen", "Aliphatic", "Other"]
