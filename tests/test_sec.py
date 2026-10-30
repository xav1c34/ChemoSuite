"""
test_sec.py — Тестовый сьют для Модуля 7: Гель-проникающая и эксклюзионная хроматография (SEC / GPC).
Проверяет:
1. Построение и валидацию калибровочных кривых log10(MW) = f(V);
2. Парсинг сырых хроматограмм элюирования и базовую линию ALS;
3. Расчет молекулярно-массовых средних (Mn, Mw, Mz, Mp, PDI / Đ) и долей фракций I–IV;
4. Геохимическую валидность синтетического датасета (Байкал vs Шлам-лигнин);
5. Интеграцию в ChemoSuite ML (классификация блока 'SEC/GPC', Data Fusion, VIP);
6. Сериализацию проекта .chemo и экспорт в паспорт Excel;
7. Стресс-тестирование на поврежденных/граничных данных.
"""
import io
import os
import numpy as np
import pandas as pd
import pytest

import sec_core
import chemo_ml
import project_io
import report_generator


def test_fit_sec_calibration():
    """Проверка построения и точности калибровочной кривой по стандартам PSS."""
    standards_v = [10.25, 9.40, 8.55, 7.75, 7.15, 6.45, 5.85]
    standards_mw = [1100.0, 3600.0, 8200.0, 17800.0, 32000.0, 65400.0, 122000.0]

    calib = sec_core.fit_sec_calibration(standards_v, standards_mw, degree=1)
    assert calib.r_squared > 0.99
    assert calib.coefficients[1] < 0  # С увеличением объема молекулярная масса падает

    # Проверка интерполяции
    mw_test = calib.v_to_mw(8.55)
    assert abs(mw_test - 8200.0) / 8200.0 < 0.20


def test_parse_sec_raw_chromatogram():
    """Проверка парсинга сырой хроматограммы, базовой линии ALS и дифференциального ММР."""
    v_grid = np.linspace(5.0, 11.5, 651)
    # Пик при V = 8.5 мл
    sig = 100.0 * np.exp(-((v_grid - 8.5) ** 2) / (2 * 0.4 ** 2)) + 5.0

    df_test = pd.DataFrame({"Volume_mL": v_grid, "UV254": sig})
    csv_bytes = df_test.to_csv(index=False).encode("utf-8")

    sample = sec_core.parse_sec_file(csv_bytes, sample_id="Test_SEC_01", flow_rate_ml_min=1.0)
    assert sample.sample_id == "Test_SEC_01"
    assert len(sample.elution_volume) == 651
    assert len(sample.signal_corrected) == 651
    assert len(sample.mwd_diff) == 651
    assert len(sample.mwd_cum) == 651
    assert len(sample.peaks) >= 1

    # Пик должен быть около 8.5 мл
    p0 = sample.peaks[0]
    assert abs(p0.elution_volume - 8.5) < 0.15
    assert p0.peak_mw > 0.0


def test_calculate_sec_indices_and_fractions():
    """Проверка расчета молекулярно-массовых индексов и распределения по фракциям."""
    v_grid = np.linspace(5.0, 11.5, 651)
    # Имитируем смесь: 30% HMW (V ~ 6.5 мл), 50% MMW (V ~ 8.8 мл), 20% LMW (V ~ 10.5 мл)
    sig = (
        30.0 * np.exp(-((v_grid - 6.5) ** 2) / (2 * 0.3 ** 2)) +
        50.0 * np.exp(-((v_grid - 8.8) ** 2) / (2 * 0.4 ** 2)) +
        20.0 * np.exp(-((v_grid - 10.5) ** 2) / (2 * 0.3 ** 2))
    )

    indices = sec_core.calculate_sec_indices(v_grid, sig)
    assert "SEC_Mn" in indices
    assert "SEC_Mw" in indices
    assert "SEC_Mz" in indices
    assert "SEC_PDI" in indices
    assert "SEC_Fraction_I_pct" in indices
    assert "SEC_Fraction_IV_pct" in indices

    # Математическое неравенство для полидисперсных систем: Mn <= Mw <= Mz
    assert indices["SEC_Mn"] <= indices["SEC_Mw"]
    assert indices["SEC_Mw"] <= indices["SEC_Mz"]
    assert indices["SEC_PDI"] >= 1.0

    # Доли фракций считаются внутри калиброванного окна (без экстраполяции) и в сумме дают ~100%;
    # фракция IV (< 1 кДа) вне диапазона стандартов PSS (самый лёгкий 1.1 кДа) -> NaN, а не 0.
    frac_sum = np.nansum([
        indices["SEC_Fraction_I_pct"],
        indices["SEC_Fraction_II_pct"],
        indices["SEC_Fraction_III_pct"],
        indices["SEC_Fraction_IV_pct"],
    ])
    assert abs(frac_sum - 100.0) < 1.0
    assert np.isnan(indices["SEC_Fraction_IV_pct"])
    # Пик при V = 10.5 мл элюирует позже самого лёгкого стандарта (10.25 мл) и исключён из интегрирования
    assert indices["SEC_Below_Cal_Range_pct"] > 10.0
    assert abs(indices["SEC_Excluded_Area_pct"] - indices["SEC_Below_Cal_Range_pct"] - indices["SEC_Above_Cal_Range_pct"]) < 0.05


def test_synthetic_sec_dataset_and_class_distinction():
    """Проверка генерации синтетического бенчмарка и геохимического разделения классов."""
    samples, df_matrix = sec_core.generate_synthetic_sec_dataset(n_samples_per_class=6, random_state=42)
    assert len(samples) == 12
    assert len(df_matrix) == 12

    nom_rows = df_matrix[df_matrix["Class_Target"] == "Baikal_Pristine"]
    lignin_rows = df_matrix[df_matrix["Class_Target"] == "Sludge_Lignin"]

    # 1. Шлам-лигнин имеет существенно более высокую среднемассовую молекулярную массу Mw
    assert lignin_rows["SEC_Mw"].mean() > nom_rows["SEC_Mw"].mean() * 1.5

    # 2. Шлам-лигнин обогащен высокомолекулярной фракцией I (> 20 кДа)
    assert lignin_rows["SEC_Fraction_I_pct"].mean() > nom_rows["SEC_Fraction_I_pct"].mean() * 2.0

    # 3. Шлам-лигнин обогащён HMW (I + II), а у фона Байкала больше сигнала элюирует позже самого
    #    лёгкого стандарта PSS (< 1.1 кДа). HMW/LMW для PSS не вычисляется (фракция < 1 кДа вне калибровки).
    assert lignin_rows["SEC_HMW_pct"].mean() > nom_rows["SEC_HMW_pct"].mean() * 1.4
    assert nom_rows["SEC_Below_Cal_Range_pct"].mean() > lignin_rows["SEC_Below_Cal_Range_pct"].mean() * 2.0
    assert df_matrix["SEC_HMW_LMW_ratio"].isna().all()

    # 4. Фоновое РОВ Байкала обогащено средне-низкомолекулярными фракциями III и IV
    assert nom_rows["SEC_Fraction_III_pct"].mean() > lignin_rows["SEC_Fraction_III_pct"].mean()


def test_chemo_ml_sec_integration():
    """Проверка классификации признаков 'SEC/GPC' и обучения PLS-DA модели на дескрипторах SEC."""
    cols = ["SEC_Mn", "SEC_Mw", "SEC_Mz", "SEC_PDI", "SEC_Fraction_I_pct", "SEC_HMW_LMW_ratio"]
    mapping = chemo_ml.classify_feature_blocks(cols)
    for c in cols:
        assert mapping[c] == "SEC/GPC"

    # Обучение модели на синтетической матрице
    _, df_matrix = sec_core.generate_synthetic_sec_dataset(n_samples_per_class=6, random_state=42)
    y_labels = df_matrix["Class_Target"]
    features = df_matrix.drop(columns=["Class_Target"])

    pls_res = chemo_ml.train_plsda_model(features, y_labels, n_components=2)
    assert pls_res["Q2"] > 70.0
    assert pls_res["Accuracy"] >= 90.0

    # Проверка блока в VIP-скорах
    vip_df = pls_res["VIP_df"]
    assert any(vip_df["Block"] == "SEC/GPC")


def test_project_io_and_report_generator_sec():
    """Проверка сериализации в .chemo проект и генерации листа 'SEC-GPC Descriptors' в Excel."""
    _, df_matrix = sec_core.generate_synthetic_sec_dataset(n_samples_per_class=4, random_state=42)

    session_data = {
        "spectra_db": {},
        "fused_data": df_matrix,
        "sec_ml_descriptors": df_matrix,
        "pls_results": None,
    }

    # 1. Сохранение и распаковка проекта
    proj_bytes = project_io.save_chemo_project(session_data, project_name="SEC_Test_Project")
    loaded = project_io.load_chemo_project(proj_bytes)
    assert "sec_ml_descriptors" in loaded
    assert len(loaded["sec_ml_descriptors"]) == 8

    # 2. Генерация паспорта Excel
    excel_bytes = report_generator.generate_excel_passport(
        fused_df=df_matrix,
        model_results=None,
        metadata={"project_name": "SEC_Excel_Report"},
        lang="ru"
    )
    assert len(excel_bytes) > 2000

    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert "SEC-GPC Descriptors" in wb.sheetnames


def test_adversarial_sec_degraded_inputs():
    """Стресс-тестирование граничных, пустых и зашумленных данных SEC."""
    # 1. Пустой файл
    s_empty = sec_core.parse_sec_file("", sample_id="Empty_SEC")
    assert s_empty.sample_id == "Empty_SEC"
    assert len(s_empty.elution_volume) == 0
    # Нечего вычислять -> NaN, а не правдоподобные константы
    assert np.isnan(s_empty.indices["SEC_Mw"])

    # 2. Данные без сигнала (все нули)
    v_arr = np.linspace(5.0, 11.0, 100)
    df_zeros = pd.DataFrame({"Volume_mL": v_arr, "Signal": np.zeros(100)})
    s_zeros = sec_core.parse_sec_file(df_zeros.to_csv(index=False), sample_id="Zeros_SEC")
    assert np.isnan(s_zeros.indices["SEC_PDI"])

    # 3. Шум вокруг нуля
    df_noise = pd.DataFrame({"Volume_mL": v_arr, "Signal": np.random.normal(0, 0.001, size=100)})
    s_noise = sec_core.parse_sec_file(df_noise.to_csv(index=False), sample_id="Noise_SEC")
    assert np.isnan(s_noise.indices["SEC_Mw"]) or s_noise.indices["SEC_Mw"] >= 0.0
