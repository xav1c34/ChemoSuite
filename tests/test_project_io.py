"""
Tests for ChemoSuite Project I/O (.chemo format)
================================================
Проверка сохранения, сжатия, валидации и полного восстановления
аналитической сессии ChemoSuite (.chemo / ZIP).
"""

import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from project_io import (
    save_chemo_project,
    load_chemo_project,
    get_chemo_project_summary,
    PROJECT_FORMAT_VERSION,
)
from chemo_ml import generate_multimodal_benchmark, train_plsda_model, train_oplsda_model, run_permutation_test


def test_save_and_load_empty_project():
    """Проверка сериализации и загрузки пустой / начальной сессии."""
    empty_session = {
        "spectra_db": {},
        "eem_ml_descriptors": None,
        "uv_ml_descriptors": None,
        "fused_data": None,
        "fused_source": "Empty",
        "pls_results": None,
        "pls_sample_ids": None,
        "pls_y_input": None,
        "perm_results": None,
        "lang": "ru",
    }

    chemo_bytes = save_chemo_project(empty_session, project_name="Empty_Test")
    assert isinstance(chemo_bytes, bytes)
    assert len(chemo_bytes) > 0

    # Проверка сводки
    summary = get_chemo_project_summary(chemo_bytes)
    assert summary["project_name"] == "Empty_Test"
    assert summary["format_version"] == PROJECT_FORMAT_VERSION
    assert summary["summary"]["n_spectra"] == 0
    assert summary["summary"]["has_trained_model"] is False

    # Загрузка
    restored = load_chemo_project(chemo_bytes)
    assert restored["spectra_db"] == {}
    assert restored["eem_ml_descriptors"] is None
    assert restored["fused_data"] is None
    assert restored["pls_results"] is None


def test_save_and_load_full_multimodal_project():
    """Проверка полного сохранения и восстановления сессии со всеми 3 модальностями и моделью."""
    # 1. Синтезируем данные
    fused_df, y_labels = generate_multimodal_benchmark()
    pls_res = train_plsda_model(fused_df, y_labels, n_components=3)
    perm_res = run_permutation_test(fused_df, y_labels, n_components=3, n_permutations=5)

    spectra_db = {
        "sample_1.csv": {
            "raw_path": "user_spectra/sample_1.csv",
            "file_bytes": b"m/z,Intensity\n200.1,500\n300.2,1000\n",
            "raw_df": pd.DataFrame({"m/z": [200.1, 300.2], "Intensity": [500.0, 1000.0]}),
            "parsed_peaks": pd.DataFrame({"mz": [200.1, 300.2], "I": [500.0, 1000.0]}),
            "assigned_df": pd.DataFrame({"Formula": ["C10H10O4", "C15H15O6"], "m/z": [200.1, 300.2]}),
        }
    }

    eem_df = pd.DataFrame({"Sample_ID": ["S1", "S2"], "FI_370": [1.2, 1.4], "HIX": [0.8, 0.9]})
    uv_df = pd.DataFrame({"Sample_ID": ["S1", "S2"], "A254": [0.35, 0.42], "E2_E3": [4.1, 3.8]})

    session_data = {
        "spectra_db": spectra_db,
        "eem_ml_descriptors": eem_df,
        "uv_ml_descriptors": uv_df,
        "fused_data": fused_df,
        "fused_source": "Trimodal Integration Test",
        "pls_results": pls_res,
        "pls_sample_ids": fused_df["Sample_ID"].values,
        "pls_y_input": y_labels.values,
        "perm_results": perm_res,
        "lang": "en",
    }

    # 2. Сериализация в .chemo
    chemo_bytes = save_chemo_project(
        session_data,
        project_name="Trimodal_Baikal_Project",
        notes="Тестовый проект с полным набором данных",
    )

    # 3. Проверка метаданных
    meta = get_chemo_project_summary(chemo_bytes)
    assert meta["project_name"] == "Trimodal_Baikal_Project"
    assert meta["notes"] == "Тестовый проект с полным набором данных"
    assert meta["summary"]["n_spectra"] == 1
    assert meta["summary"]["has_eem"] is True
    assert meta["summary"]["has_uv"] is True
    assert meta["summary"]["has_fused"] is True
    assert meta["summary"]["has_trained_model"] is True
    assert meta["summary"]["model_type"] == "PLS-DA"
    assert meta["summary"]["has_permutation"] is True

    # 4. Десериализация
    restored = load_chemo_project(chemo_bytes)

    # Проверка базы спектров
    assert "sample_1.csv" in restored["spectra_db"]
    entry = restored["spectra_db"]["sample_1.csv"]
    assert entry["file_bytes"] == b"m/z,Intensity\n200.1,500\n300.2,1000\n"
    assert_frame_equal(entry["raw_df"], spectra_db["sample_1.csv"]["raw_df"])
    assert_frame_equal(entry["parsed_peaks"], spectra_db["sample_1.csv"]["parsed_peaks"])
    assert_frame_equal(entry["assigned_df"], spectra_db["sample_1.csv"]["assigned_df"])

    # Проверка дескрипторов
    assert_frame_equal(restored["eem_ml_descriptors"], eem_df)
    assert_frame_equal(restored["uv_ml_descriptors"], uv_df)
    assert_frame_equal(restored["fused_data"], fused_df)
    assert restored["fused_source"] == "Trimodal Integration Test"

    # Проверка модели
    rest_pls = restored["pls_results"]
    assert rest_pls["model_type"] == "PLS-DA"
    assert np.isclose(rest_pls["R2X"], pls_res["R2X"])
    assert np.isclose(rest_pls["R2Y"], pls_res["R2Y"])
    assert np.isclose(rest_pls["Accuracy"], pls_res["Accuracy"])
    assert_frame_equal(rest_pls["VIP_df"], pls_res["VIP_df"])
    assert isinstance(rest_pls["scores_t1"], np.ndarray)
    assert np.allclose(rest_pls["scores_t1"], pls_res["scores_t1"])

    # Проверка пермутации
    rest_perm = restored["perm_results"]
    assert np.isclose(rest_perm["q2_orig"], perm_res["q2_orig"])
    assert np.isclose(rest_perm["p_val_q2"], perm_res["p_val_q2"])
    assert len(rest_perm["perm_q2"]) == len(perm_res["perm_q2"])


def test_save_and_load_oplsda_project():
    """Проверка сериализации сессии с OPLS-DA моделью и S-plot."""
    fused_df, y_labels = generate_multimodal_benchmark()
    opls_res = train_oplsda_model(fused_df, y_labels, n_ortho=1)

    session_data = {
        "spectra_db": {},
        "eem_ml_descriptors": None,
        "uv_ml_descriptors": None,
        "fused_data": fused_df,
        "fused_source": "OPLS Benchmark",
        "pls_results": opls_res,
        "pls_sample_ids": fused_df["Sample_ID"].values,
        "pls_y_input": y_labels.values,
        "perm_results": None,
    }

    chemo_bytes = save_chemo_project(session_data, project_name="OPLS_Session")
    restored = load_chemo_project(chemo_bytes)

    rest_opls = restored["pls_results"]
    assert rest_opls["model_type"] == "OPLS-DA"
    assert "s_plot_df" in rest_opls
    assert_frame_equal(rest_opls["s_plot_df"], opls_res["s_plot_df"])
    assert np.isclose(rest_opls["R2X_pred"], opls_res["R2X_pred"])
    assert np.isclose(rest_opls["R2X_ortho"], opls_res["R2X_ortho"])


def test_invalid_project_handling():
    """Проверка устойчивости к невалидным или поврежденным данным."""
    with pytest.raises(Exception):
        load_chemo_project(b"invalid binary content that is not zip")

    # ZIP без metadata.json
    empty_zip = io.BytesIO()
    with zipfile.ZipFile(empty_zip, mode="w") as zf:
        zf.writestr("test.txt", "hello")

    with pytest.raises(ValueError):
        load_chemo_project(empty_zip.getvalue())
