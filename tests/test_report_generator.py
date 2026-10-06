"""
tests/test_report_generator.py — Модульные тесты генератора аналитических паспортов Excel (.xlsx).
"""
import io
import openpyxl
import pandas as pd
import pytest

import chemo_ml
import report_generator


def test_generate_excel_passport_plsda():
    """Тест генерации многостраничного паспорта для PLS-DA модели."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    df["Class_Target"] = y.values
    X = df.drop(columns=["Sample_ID", "Class_Target"])

    res = chemo_ml.train_plsda_model(X, y.values, n_components=2)
    meta = {"project_name": "Baikal Test Benchmark", "operator": "Dr. Chemist"}

    excel_bytes = report_generator.generate_excel_passport(df, model_results=res, metadata=meta)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 5000

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Passport & Summary" in sheet_names
    assert "Fused Feature Matrix" in sheet_names
    assert "FT-ICR MS Descriptors" in sheet_names
    assert "EEM Descriptors" in sheet_names
    assert "UV-Vis Descriptors" in sheet_names
    assert "Biomarkers & VIP" in sheet_names

    # Проверка листа сводки
    ws_sum = wb["Passport & Summary"]
    assert "CHEMOSUITE UNIFIED ANALYTICAL PASSPORT" in str(ws_sum["A1"].value)
    assert ws_sum["B5"].value == "Baikal Test Benchmark"


def test_generate_excel_passport_oplsda():
    """Тест генерации паспорта для OPLS-DA модели с вкладкой S-Plot."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    df["Class_Target"] = y.values
    X = df.drop(columns=["Sample_ID", "Class_Target"])

    res = chemo_ml.train_oplsda_model(X, y.values, n_ortho=1)
    excel_bytes = report_generator.generate_excel_passport(df, model_results=res)

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert "OPLS-DA S-Plot" in wb.sheetnames

    # Проверка содержимого S-Plot
    ws_s = wb["OPLS-DA S-Plot"]
    headers = [cell.value for cell in ws_s[1]]
    assert "Descriptor" in headers
    assert "p1_cov" in headers
    assert "p_corr" in headers
    assert "VIP" in headers


def test_generate_excel_passport_no_model():
    """Тест генерации паспорта без обученной модели (сырая матрица)."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    df["Class_Target"] = y.values

    excel_bytes = report_generator.generate_excel_passport(df, model_results=None)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 3000

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert "Passport & Summary" in wb.sheetnames
    assert "Fused Feature Matrix" in wb.sheetnames
    assert "Biomarkers & VIP" not in wb.sheetnames
