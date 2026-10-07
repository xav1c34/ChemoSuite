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
