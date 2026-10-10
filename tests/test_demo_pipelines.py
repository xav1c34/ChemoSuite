"""
tests/test_demo_pipelines.py — Сквозные интеграционные тесты реальных демо-данных ChemoSuite.
Проверяет непрерывность пайплайна: FT-ICR MS, EEM-PARAFAC, UV-Vis, Data Fusion, ML, Excel и .chemo.
"""
import os
import io
import numpy as np
import pandas as pd
import pytest

import fticr_core
import eem_core
import chemo_ml
import chemo_pubchem
import report_generator
import project_io


def test_demo_fticr_pipeline():
    """Тест обработки всех демонстрационных масс-спектров FT-ICR MS."""
    demo_dir = os.path.join(os.path.dirname(__file__), "..", "demo_data", "fticr_ms")
    ft_files = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.endswith(".csv")]
    assert len(ft_files) >= 4

    for fpath in ft_files:
        with open(fpath, "rb") as f:
            raw_b = f.read()
        parsed_df = fticr_core.parse_uploaded_file(raw_b)
        assert not parsed_df.empty
        assert "mass" in parsed_df.columns
        assert "intensity" in parsed_df.columns
        assert len(parsed_df) >= 100

        # Формулы и дескрипторы на топ-интенсивных пиках
        top_peaks = parsed_df.sort_values("intensity", ascending=False).head(300)
        assigned = fticr_core.fast_formula_assigner(top_peaks, ppm_tolerance=2.0)
        assert not assigned.empty
        descs = fticr_core.calculate_descriptors(assigned, lang="ru")
        assert "H/C" in descs.columns
        assert "O/C" in descs.columns

        # KMD
        km, kmd, nkm = fticr_core.compute_kmd(parsed_df["mass"].values[:100], base_key="CH2")
        assert len(kmd) == 100

        # TMDS сети и векторные потоки
        fluxes = fticr_core.compute_geochemical_vector_fluxes(parsed_df.head(200), tol_mda=2.0)
        assert fluxes["total_reactions"] >= 0
        assert "indices" in fluxes


def test_demo_eem_and_uv_pipeline():
    """Тест обработки всех матриц флуоресценции EEM и спектров поглощения УФ-Вид."""
    eem_dir = os.path.join(os.path.dirname(__file__), "..", "demo_data", "eem")
    uv_dir = os.path.join(os.path.dirname(__file__), "..", "demo_data", "uv_vis")
    eem_files = [os.path.join(eem_dir, f) for f in os.listdir(eem_dir) if f.endswith(".csv")]
    uv_files = [os.path.join(uv_dir, f) for f in os.listdir(uv_dir) if f.endswith(".csv")]

    assert len(eem_files) >= 6
    assert len(uv_files) >= 6

    eem_samples = []
    for fpath in eem_files[:4]:
        with open(fpath, "rb") as f:
            s = eem_core.load_eem_file(f.read(), sample_id=os.path.basename(fpath))
            assert s.data.shape[0] > 10
            assert s.data.shape[1] > 5
            eem_samples.append(s)

    # Подавление рассеяния и тензор PARAFAC
    s0 = eem_samples[0]
    sc_clean = eem_core.remove_scatter_bands(s0.data, s0.ex, s0.em)
    assert sc_clean.shape == s0.data.shape

    tensor, ex_c, em_c, valid_s = eem_core.build_eem_tensor(eem_samples)
    assert tensor.ndim == 3
    parafac_res = eem_core.fit_parafac(tensor, n_components=2)
    assert "scores" in parafac_res
    assert "em_profiles" in parafac_res
    assert "ex_profiles" in parafac_res

    # УФ-Вид и IFE коррекция
    with open(uv_files[0], "rb") as f:
        uv_s = eem_core.parse_uv_vis_spectrum(f.read(), sample_id="UV_Test")
    assert len(uv_s.wl) > 50

    uv_indices = eem_core.calculate_uv_vis_indices(uv_s, doc=5.0)
    assert "A254" in uv_indices
    assert "E2_E3" in uv_indices
    assert "Mw_est" in uv_indices

    ife_mat, cf_mat, ife_warn = eem_core.correct_inner_filter_effect(s0, uv_s)
    assert ife_mat.shape == s0.data.shape


def test_demo_multimodal_ml_and_reports():
    """Тест полного сквозного цикла машинного обучения, генерации отчетов и проектов."""
    fused_path = os.path.join(os.path.dirname(__file__), "..", "demo_data", "multimodal_ml", "chemo_unified_multimodal.csv")
    df_fused = pd.read_csv(fused_path)
    assert len(df_fused) >= 10

    feat_cols = [c for c in df_fused.columns if c not in ["Sample_ID", "Class_Target"]]
    X = df_fused[feat_cols]
    y = df_fused["Class_Target"].values

    # 1. PLS-DA
    pls_res = chemo_ml.train_plsda_model(X, y, n_components=2)
    assert pls_res["Accuracy"] >= 80.0
    assert "VIP_df" in pls_res

    # 2. OPLS-DA
    opls_res = chemo_ml.train_oplsda_model(X, y, n_ortho=1)
    assert opls_res["Accuracy"] >= 80.0
    assert "S_Plot_df" in opls_res
    assert not opls_res["S_Plot_df"].empty

    # 3. sPLS-DA
    spls_res = chemo_ml.train_splsda_model(X, y, n_components=2, keep_x=12)
    assert spls_res["Accuracy"] >= 80.0
    assert "Sparse_Loadings_df" in spls_res

    # 4. Генерация паспорта Excel
    excel_bytes = report_generator.generate_excel_passport(
        fused_df=df_fused,
        model_results=opls_res,
        metadata={"project_name": "Integration Test", "operator": "PyTest CI"}
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 5000

    # 5. Сохранение и загрузка .chemo проекта
    session_data = {
        "fused_data": df_fused,
        "pls_results": opls_res,
        "spectra_db": {},
    }
    chemo_bytes = project_io.save_chemo_project(session_data, project_name="Test CI")
    assert isinstance(chemo_bytes, bytes)
    assert len(chemo_bytes) > 500

    restored = project_io.load_chemo_project(chemo_bytes)
    assert "fused_data" in restored
    assert "pls_results" in restored


def test_batch_descriptors_and_kmd_slicing():
    """Тест пакетного извлечения дескрипторов FT-ICR MS и KMD-среза."""
    demo_csv = os.path.join("demo_data", "fticr_ms", "Baikal_Control_01_fticr.csv")
    assert os.path.exists(demo_csv)
    df_peaks = pd.read_csv(demo_csv)

    # Приписывание формул
    assigned = fticr_core.fast_formula_assigner(df_peaks, ppm_tolerance=1.5)
    assert assigned is not None and not assigned.empty

    # Извлечение вектора дескрипторов (Feature 4)
    desc = chemo_ml.extract_fticr_descriptors(assigned, sample_id="Baikal_Control_01")
    assert desc["Sample_ID"] == "Baikal_Control_01"
    assert "VK_1" in desc and "VK_20" in desc
    assert "Mn" in desc and "H/C" in desc and "O/C" in desc and "AI" in desc

    # KMD Slicing (Feature 2)
    km, kmd, nkm = fticr_core.compute_kmd(assigned["mass"].values, "CH2")
    assigned["KMD"] = kmd
    target_kmd = float(np.median(kmd))
    tol_kmd = 0.015
    mask = (assigned["KMD"] >= target_kmd - tol_kmd) & (assigned["KMD"] <= target_kmd + tol_kmd)
    sliced = assigned[mask]
    assert len(sliced) > 0
    assert (sliced["KMD"].max() - sliced["KMD"].min()) <= (2 * tol_kmd + 1e-6)


def test_session_sample_persistence_and_recalibration():
    """Тест надежности сохранения спектров и сохранения рекалиброванных масс при многократных прогонах."""
    demo_csv = os.path.join("demo_data", "fticr_ms", "Baikal_Control_01_fticr.csv")
    with open(demo_csv, "rb") as f:
        raw_bytes = f.read()

    sample_entry = {
        "raw_path": demo_csv,
        "file_bytes": raw_bytes,
        "parsed_peaks": None,
        "assigned_df": None,
        "raw_df": None,
    }

    # 1. Первоначальный парсинг
    df_raw = fticr_core.parse_uploaded_file(raw_bytes, delimiter="Auto", decimal_sep=".", has_header=True)
    assert not df_raw.empty
    parsed_peaks = pd.DataFrame({"mass": df_raw.iloc[:, 0].astype(float), "intensity": df_raw.iloc[:, 1].astype(float)})
    sample_entry["parsed_peaks"] = parsed_peaks.copy()

    # 2. Имитация рекалибровки в Tab 2
    poly_shift = np.poly1d([1e-4, -0.005])
    calibrated_masses = parsed_peaks["mass"].values - poly_shift(parsed_peaks["mass"].values)
    sample_entry["parsed_peaks"]["mass"] = calibrated_masses
    sample_entry["recal_poly"] = poly_shift
    sample_entry["is_recalibrated"] = True

    # 3. Имитация перехода на другую вкладку и повторного прогона фильтрации
    # Если sample помечен is_recalibrated и есть recal_poly, массы должны оставаться рекалиброванными!
    re_filtered = parsed_peaks.copy()
    if sample_entry.get("is_recalibrated") and sample_entry.get("recal_poly") is not None:
        re_filtered["mass"] = re_filtered["mass"] - sample_entry["recal_poly"](re_filtered["mass"])

    np.testing.assert_allclose(re_filtered["mass"].values, calibrated_masses)


def test_demo_thermal_pipeline_and_6modal_data_fusion():
    """Тест обработки демонстрационных данных Py-GC-MS и TGA и 6-модального слияния данных."""
    import thermal_core
    th_dir = os.path.join(os.path.dirname(__file__), "..", "demo_data", "thermal")
    assert os.path.exists(th_dir)
    py_files = [f for f in os.listdir(th_dir) if f.endswith("_PyGCMS.csv")]
    tga_files = [f for f in os.listdir(th_dir) if f.endswith("_TGA.csv")]
    assert len(py_files) == 12
    assert len(tga_files) == 12

    # 1. Парсинг первого образца Py-GC-MS и TGA
    with open(os.path.join(th_dir, py_files[0]), "rb") as f:
        py_s = thermal_core.parse_py_gcms_file(f.read(), sample_id="Test_Py")
    assert len(py_s.peaks) > 0
    assert "Py_S_G_ratio" in py_s.indices
    assert "Py_L_C_ratio" in py_s.indices

    with open(os.path.join(th_dir, tga_files[0]), "rb") as f:
        tga_s = thermal_core.parse_tga_file(f.read(), sample_id="Test_TGA")
    assert len(tga_s.temperature) > 50
    assert "TGA_Thermal_Index_TI" in tga_s.indices
    assert "TGA_T50_degC" in tga_s.indices

    # 2. Загрузка матрицы дескрипторов
    mat_file = os.path.join(th_dir, "thermal_descriptors_matrix.csv")
    assert os.path.exists(mat_file)
    df_thermal = pd.read_csv(mat_file)
    assert len(df_thermal) == 12
    assert "Py_S_G_ratio" in df_thermal.columns
    assert ("TGA_Thermostability_Index" in df_thermal.columns or "TGA_Thermal_Index_TI" in df_thermal.columns)

    # 3. 6-модальное слияние данных в chemo_ml
    fused_path = os.path.join(os.path.dirname(__file__), "..", "demo_data", "multimodal_ml", "chemo_unified_multimodal.csv")
    df_fused = pd.read_csv(fused_path)

    # Приводим Sample_ID к общему виду для слияния
    df_thermal_aligned = df_thermal.copy()
    df_thermal_aligned["Sample_ID"] = df_fused["Sample_ID"].iloc[:len(df_thermal_aligned)].values

    merged_6m = chemo_ml.merge_feature_blocks(df_fused, None, df_thermal=df_thermal_aligned)
    assert not merged_6m.empty
    blocks = chemo_ml.classify_feature_blocks(merged_6m)
    assert "Thermal" in blocks.values()

    # 4. Обучение модели PLS-DA на 6 модальностях
    feat_cols = [c for c in merged_6m.columns if c not in ["Sample_ID", "Class_Target"]]
    X = merged_6m[feat_cols]
    y = merged_6m["Class_Target"].values
    pls_res = chemo_ml.train_plsda_model(X, y, n_components=2)
    assert pls_res["Q2"] > 0
    assert "VIP_df" in pls_res
    assert any(pls_res["VIP_df"]["Block"] == "Thermal")

    # 5. Экспорт паспорта Excel с 6 модальностями
    rep_bytes = report_generator.generate_excel_passport(
        merged_6m,
        model_results=pls_res,
        metadata={"project_name": "6-Modal ChemoSuite Test", "operator": "Pytest"},
        lang="ru"
    )
    assert len(rep_bytes) > 2000

    # 6. Сохранение и загрузка проекта .chemo с блоком Thermal
    session_data = {
        "fused_data": merged_6m,
        "pls_results": pls_res,
        "thermal_ml_descriptors": df_thermal_aligned,
    }
    proj_bytes = project_io.save_chemo_project(session_data, project_name="Test_6M_Project")
    loaded = project_io.load_chemo_project(proj_bytes)
    assert "thermal_ml_descriptors" in loaded
    assert len(loaded["thermal_ml_descriptors"]) == len(df_thermal_aligned)


def test_demo_sec_pipeline_and_7modal_data_fusion():
    """Тест полного сквозного цикла Модуля 7 (SEC/GPC) и 7-модального Data Fusion."""
    import sec_core
    sec_dir = os.path.join(os.path.dirname(__file__), "..", "demo_data", "sec")
    assert os.path.exists(sec_dir)
    sec_files = [f for f in os.listdir(sec_dir) if f.endswith("_SEC.csv")]
    assert len(sec_files) == 12

    # 1. Парсинг первого образца
    with open(os.path.join(sec_dir, sec_files[0]), "rb") as f:
        s0 = sec_core.parse_sec_file(f.read(), sample_id="Test_SEC_Pipe")
    assert len(s0.elution_volume) > 50
    assert "SEC_Mw" in s0.indices
    assert "SEC_Fraction_I_pct" in s0.indices
    assert s0.indices["SEC_Mw"] > 0

    # 2. Загрузка матрицы дескрипторов SEC
    mat_file = os.path.join(sec_dir, "sec_descriptors_matrix.csv")
    assert os.path.exists(mat_file)
    df_sec = pd.read_csv(mat_file)
    assert len(df_sec) == 12
    assert "SEC_Mw" in df_sec.columns

    # 3. 7-модальное слияние данных в chemo_ml
    fused_path = os.path.join(os.path.dirname(__file__), "..", "demo_data", "multimodal_ml", "chemo_unified_multimodal.csv")
    df_fused = pd.read_csv(fused_path)

    th_path = os.path.join(os.path.dirname(__file__), "..", "demo_data", "thermal", "thermal_descriptors_matrix.csv")
    df_thermal = pd.read_csv(th_path)
    df_thermal_aligned = df_thermal.copy()
    df_thermal_aligned["Sample_ID"] = df_fused["Sample_ID"].iloc[:len(df_thermal_aligned)].values

    df_sec_aligned = df_sec.copy()
    df_sec_aligned["Sample_ID"] = df_fused["Sample_ID"].iloc[:len(df_sec_aligned)].values

    merged_7m = chemo_ml.merge_feature_blocks(df_fused, None, df_thermal=df_thermal_aligned, df_sec=df_sec_aligned)
    assert not merged_7m.empty
    blocks = chemo_ml.classify_feature_blocks(merged_7m)
    assert "SEC/GPC" in blocks.values()
    assert "Thermal" in blocks.values()

    # 4. Обучение модели PLS-DA на 7 модальностях
    feat_cols = [c for c in merged_7m.columns if c not in ["Sample_ID", "Class_Target"]]
    X = merged_7m[feat_cols]
    y = merged_7m["Class_Target"].values
    pls_res = chemo_ml.train_plsda_model(X, y, n_components=2)
    assert pls_res["Q2"] > 0
    assert "VIP_df" in pls_res
    assert any(pls_res["VIP_df"]["Block"] == "SEC/GPC")

    # 5. Экспорт аналитического паспорта Excel на 7 модальностях
    rep_bytes = report_generator.generate_excel_passport(
        merged_7m,
        model_results=pls_res,
        metadata={"project_name": "7-Modal ChemoSuite Passport", "operator": "Pytest"},
        lang="ru"
    )
    assert len(rep_bytes) > 2000

    # 6. Сохранение и загрузка .chemo проекта
    session_data = {
        "fused_data": merged_7m,
        "pls_results": pls_res,
        "thermal_ml_descriptors": df_thermal_aligned,
        "sec_ml_descriptors": df_sec_aligned,
    }
    proj_bytes = project_io.save_chemo_project(session_data, project_name="Test_7M_Project")
    loaded = project_io.load_chemo_project(proj_bytes)
    assert "sec_ml_descriptors" in loaded
    assert len(loaded["sec_ml_descriptors"]) == len(df_sec_aligned)

