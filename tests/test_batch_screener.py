"""
tests/test_batch_screener.py — Тесты автономного пакетного конвейера скрининга директорий ChemoSuite 3.0.
"""
import io
import os
from pathlib import Path
import zipfile
import pandas as pd
import pytest

import batch_screener


def test_detect_file_modality_by_name():
    """Проверка автодетекции модальностей по именам файлов."""
    assert batch_screener.detect_file_modality(b"", "Sample_01_fticr.csv")["modality"] == "FT-ICR MS"
    assert batch_screener.detect_file_modality(b"", "Baikal_eem.csv")["modality"] == "EEM-PARAFAC"
    assert batch_screener.detect_file_modality(b"", "Lake_uv.csv")["modality"] == "UV-Vis"
    assert batch_screener.detect_file_modality(b"", "Soil_ftir.csv")["modality"] == "FTIR"
    assert batch_screener.detect_file_modality(b"", "Spec_nmr.jdx")["modality"] == "NMR"
    assert batch_screener.detect_file_modality(b"", "TGA_sample.txt")["modality"] == "Thermal"
    assert batch_screener.detect_file_modality(b"", "Polymer_sec.csv")["modality"] == "SEC/GPC"
    assert batch_screener.detect_file_modality(b"", "CHN_elemental.csv")["modality"] == "Elemental"


def test_detect_file_modality_by_content():
    """Проверка распознавания метода по структуре заголовков файлов."""
    fticr_header = b"m/z,intensity,resol,s/n\n355.1234,50000,450000,12.5\n"
    assert batch_screener.detect_file_modality(fticr_header, "data1.csv")["modality"] == "FT-ICR MS"

    ftir_header = b"Wavenumber,Absorbance\n4000,0.01\n3998,0.02\n"
    assert batch_screener.detect_file_modality(ftir_header, "spec.txt")["modality"] == "FTIR"

    tga_header = b"Temperature (C),Weight (%)\n25.0,100.0\n100.0,98.2\n"
    assert batch_screener.detect_file_modality(tga_header, "curve.csv")["modality"] == "Thermal"

    nmr_jdx = b"##TITLE= 13C CPMAS\n##JCAMP-DX= 5.01\n##DATA TYPE= NMR SPECTRUM\n"
    assert batch_screener.detect_file_modality(nmr_jdx, "unknown.dx")["modality"] == "NMR"


def test_extract_sample_id():
    """Проверка очистки имен образцов."""
    assert batch_screener.extract_sample_id_from_filename("Baikal_Control_01_fticr.csv") == "Baikal_Control_01"
    assert batch_screener.extract_sample_id_from_filename("Lignin_Impact_02_eem.csv") == "Lignin_Impact_02"
    assert batch_screener.extract_sample_id_from_filename("Sample_A_ftir.csv") == "Sample_A"


def test_batch_screen_directory_fticr():
    """Тест пакетного скрининга реальной папки demo_data/fticr_ms."""
    base_dir = Path(__file__).resolve().parent.parent / "demo_data" / "fticr_ms"
    if not base_dir.exists():
        pytest.skip("Директория demo_data/fticr_ms не найдена.")

    res = batch_screener.batch_screen_directory(str(base_dir), recursive=False)
    summary_df = res["summary_df"]
    fused_df = res["fused_df"]
    stats = res["statistics"]

    assert not summary_df.empty
    assert stats["total_files"] >= 4
    assert stats["successful"] > 0
    assert "Detected_Modality" in summary_df.columns
    assert "FT-ICR MS" in stats["modalities_found"]

    assert not fused_df.empty
    assert "Sample_ID" in fused_df.columns
    assert "Class_Target" in fused_df.columns


def test_batch_screen_directory_ftir():
    """Тест пакетного скрининга папки demo_data/ftir."""
    base_dir = Path(__file__).resolve().parent.parent / "demo_data" / "ftir"
    if not base_dir.exists():
        pytest.skip("Директория demo_data/ftir не найдена.")

    res = batch_screener.batch_screen_directory(str(base_dir), recursive=False)
    summary_df = res["summary_df"]
    fused_df = res["fused_df"]

    assert not summary_df.empty
    assert len(summary_df) >= 6
    assert not fused_df.empty
    assert "Sample_ID" in fused_df.columns


def test_export_batch_results_zip():
    """Тест генерации сводного ZIP-архива пакетного скрининга."""
    fused_mock = pd.DataFrame([
        {"Sample_ID": "S1", "Class_Target": "Baikal", "FTIR_Arom": 0.45, "EEM_HIX": 3.2},
        {"Sample_ID": "S2", "Class_Target": "Lignin", "FTIR_Arom": 0.85, "EEM_HIX": 8.1},
    ])
    summary_mock = pd.DataFrame([
        {"Filename": "S1.csv", "Sample_ID": "S1", "Detected_Modality": "FTIR", "Status": "success", "Points_Detected": 100, "Descriptors_Count": 5, "Notes": "OK"},
        {"Filename": "S2.csv", "Sample_ID": "S2", "Detected_Modality": "FTIR", "Status": "success", "Points_Detected": 100, "Descriptors_Count": 5, "Notes": "OK"},
    ])

    batch_res = {
        "fused_df": fused_mock,
        "summary_df": summary_mock,
        "statistics": {"total_files": 2, "successful": 2, "failed": 0, "unique_samples": 2, "modalities_found": ["FTIR"]},
    }

    zip_bytes = batch_screener.export_batch_results_zip(batch_res, metadata={"project_name": "Test Batch Screening"})
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 2000

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert "ChemoSuite_Batch_Fused_Matrix.csv" in namelist
        assert "ChemoSuite_Batch_Screening_Summary.csv" in namelist
        assert "ChemoSuite_Batch_Ecological_Passport.html" in namelist
        assert "ChemoSuite_Batch_Analytical_Passport.xlsx" in namelist


def test_batch_screen_directory_elemental():
    """Тест пакетного скрининга папки demo_data/elemental."""
    base_dir = Path(__file__).resolve().parent.parent / "demo_data" / "elemental"
    if not base_dir.exists():
        pytest.skip("Директория demo_data/elemental не найдена.")

    res = batch_screener.batch_screen_directory(str(base_dir), recursive=False)
    summary_df = res["summary_df"]
    fused_df = res["fused_df"]

    assert not summary_df.empty
    assert "Elemental" in res["statistics"]["modalities_found"]
    assert not fused_df.empty
    assert len(fused_df) >= 12
    assert "Elem_H_C" in fused_df.columns
    assert "Elem_NOSC" in fused_df.columns

