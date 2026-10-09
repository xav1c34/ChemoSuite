"""
report_generator.py — Генератор комплексного многостраничного аналитического паспорта ChemoSuite (.xlsx).
Включает вкладки:
1. Паспорт и сводка (Metadata & Model Summary)
2. Полная матрица признаков (Fused Feature Matrix)
3. FT-ICR MS дескрипторы (20 ячеек Перминовой, VK-1..20, молекулярные индексы)
4. EEM-PARAFAC дескрипторы (флуорофоры C1..C3, оптические индексы)
5. UV-Vis дескрипторы (поглощение, наклоны спектра, производные, MW)
6. Биомаркеры и VIP-ранжирование (Biomarkers & VIP Ranking)
7. S-Plot (для OPLS-DA: p1_cov, p_corr)
"""

import io
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

import chemo_ml


def _style_header_cell(cell, fill_hex="0B2545", font_color="FFFFFF", font_size=11, bold=True):
    cell.fill = PatternFill(start_color=fill_hex, end_color=fill_hex, fill_type="solid")
    cell.font = Font(name="Calibri", size=font_size, bold=bold, color=font_color)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _style_data_cell(cell, bold=False, num_format=None, align="left", bg_color=None):
    if bg_color:
        cell.fill = PatternFill(start_color=bg_color, end_color=bg_color, fill_type="solid")
    cell.font = Font(name="Calibri", size=10, bold=bold)
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if num_format:
        cell.number_format = num_format


def _apply_thin_borders(ws, min_row, max_row, min_col, max_col):
    thin = Side(border_style="thin", color="D9D9D9")
    border = Border(top=thin, left=thin, right=thin, bottom=thin)
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            if not cell.border or cell.border.top.style is None:
                cell.border = border


def _autofit_columns(ws, min_width=12, max_width=45):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if "\n" in val_str:
                val_str = max(val_str.split("\n"), key=len)
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 3, max_width))


def generate_excel_passport(
    fused_df: pd.DataFrame,
    model_results: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    lang: str = "ru",
) -> bytes:
    """
    Генерирует многостраничный файл Excel (.xlsx) с паспортом анализа и возвращает байты.
    Поддерживает языки отчета: "ru" и "en".
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Удаляем дефолтный пустой лист

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = metadata or {}
    project_name = meta.get("project_name", "ChemoSuite Analytical Session")
    operator_name = meta.get("operator", "ChemoSuite Laboratory Engine")

    feature_cols = [c for c in fused_df.columns if c not in ["Sample_ID", "Class_Target"]]
    block_map = chemo_ml.classify_feature_blocks(feature_cols)

    ft_cols = [c for c in feature_cols if block_map.get(c) == "FT-ICR MS"]
    eem_cols = [c for c in feature_cols if block_map.get(c) == "EEM-PARAFAC"]
    uv_cols = [c for c in feature_cols if block_map.get(c) == "UV-Vis"]
    other_cols = [c for c in feature_cols if block_map.get(c) == "Other"]

    is_en = (lang == "en")

    # =========================================================================
    # ЛИСТ 1: Паспорт и сводка (Passport & Summary)
    # =========================================================================
    ws_sum = wb.create_sheet(title="Passport & Summary")
    ws_sum.views.sheetView[0].showGridLines = True

    # Главный баннер
    ws_sum.merge_cells("A1:G2")
    banner_cell = ws_sum["A1"]
    banner_cell.value = "CHEMOSUITE UNIFIED ANALYTICAL PASSPORT & REPORT"
    banner_cell.fill = PatternFill(start_color="0B2545", end_color="0B2545", fill_type="solid")
    banner_cell.font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    banner_cell.alignment = Alignment(horizontal="center", vertical="center")

    ws_sum["A4"] = "1. GENERAL ANALYSIS METADATA" if is_en else "1. ОБЩИЕ МЕТАДАННЫЕ АНАЛИЗА"
    ws_sum["A4"].font = Font(name="Calibri", size=12, bold=True, color="134074")

    meta_info = [
        ("Project / Session:" if is_en else "Проект / Сессия:", project_name),
        ("Generation Date:" if is_en else "Дата генерации:", now_str),
        ("Operator / Engine:" if is_en else "Оператор / Платформа:", operator_name),
        ("Total Samples:" if is_en else "Количество образцов (Samples):", len(fused_df)),
        ("Total Features:" if is_en else "Всего дескрипторов (Features):", len(feature_cols)),
        ("FT-ICR MS Descriptors:" if is_en else "Дескрипторов FT-ICR MS:", len(ft_cols)),
        ("EEM-PARAFAC Descriptors:" if is_en else "Дескрипторов EEM-PARAFAC:", len(eem_cols)),
        ("UV-Vis Descriptors:" if is_en else "Дескрипторов UV-Vis:", len(uv_cols)),
    ]

    for idx, (lbl, val) in enumerate(meta_info, start=5):
        ws_sum.cell(row=idx, column=1, value=lbl)
        ws_sum.cell(row=idx, column=1).font = Font(name="Calibri", size=10, bold=True)
        ws_sum.cell(row=idx, column=2, value=val)
        _style_data_cell(ws_sum.cell(row=idx, column=2))

    # Секция хемометрической модели
    curr_row = 15
    ws_sum.cell(row=curr_row, column=1, value="2. CHEMOMETRIC MODEL SUMMARY" if is_en else "2. СВОДКА ХЕМОМЕТРИЧЕСКОЙ МОДЕЛИ")
    ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=12, bold=True, color="134074")
    curr_row += 1

    if model_results:
        m_type = model_results.get("model_type", "PLS-DA")
        acc = model_results.get("Balanced_Accuracy", model_results.get("Accuracy", 0.0))
        r2x = model_results.get("R2X", 0.0)
        r2x_p = model_results.get("R2X_pred", None)
        r2x_o = model_results.get("R2X_ortho", None)
        r2y = model_results.get("R2Y", 0.0)
        q2 = model_results.get("Q2", 0.0)

        model_info = [
            ("Model Architecture:" if is_en else "Архитектура модели:", m_type),
            ("R²X (Explained X-variance):" if is_en else "R²X (Объясненная дисперсия X):", f"{r2x:.2f}%"),
            ("R²Y (Explained Class Y-variance):" if is_en else "R²Y (Объясненная дисперсия классов Y):", f"{r2y:.2f}%"),
            ("Q² (Predictive Ability CV):" if is_en else "Q² (Прогностическая способность CV):", f"{q2:.2f}%"),
            ("Balanced Accuracy:" if is_en else "Сбалансированная точность (Accuracy):", f"{acc:.2f}%"),
        ]
        if r2x_p is not None and r2x_o is not None:
            model_info.append(("R²X_pred (Predictive Variance):" if is_en else "R²X_pred (Предиктивная дисперсия):", f"{r2x_p:.2f}%"))
            model_info.append(("R²X_ortho (Orthogonal Noise):" if is_en else "R²X_ortho (Ортогональный шум):", f"{r2x_o:.2f}%"))

        for lbl, val in model_info:
            ws_sum.cell(row=curr_row, column=1, value=lbl)
            ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=10, bold=True)
            ws_sum.cell(row=curr_row, column=2, value=val)
            _style_data_cell(ws_sum.cell(row=curr_row, column=2))
            curr_row += 1

        # Вклад аналитических блоков
        b_contrib = model_results.get("Block_Contributions", {})
        if b_contrib:
            curr_row += 1
            ws_sum.cell(row=curr_row, column=1, value="3. RELATIVE ANALYTICAL BLOCK CONTRIBUTION (VIP²)" if is_en else "3. ОТНОСИТЕЛЬНЫЙ ВКЛАД АНАЛИТИЧЕСКИХ БЛОКОВ (VIP²)")
            ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=12, bold=True, color="134074")
            curr_row += 1

            ws_sum.cell(row=curr_row, column=1, value="Analytical Block" if is_en else "Аналитический блок")
            ws_sum.cell(row=curr_row, column=2, value="Contribution Share (%)" if is_en else "Доля вклада (%)")
            _style_header_cell(ws_sum.cell(row=curr_row, column=1), fill_hex="134074")
            _style_header_cell(ws_sum.cell(row=curr_row, column=2), fill_hex="134074")
            curr_row += 1

            for b_name, b_pct in b_contrib.items():
                ws_sum.cell(row=curr_row, column=1, value=b_name)
                _style_data_cell(ws_sum.cell(row=curr_row, column=1), bold=True)
                ws_sum.cell(row=curr_row, column=2, value=b_pct / 100.0)
                _style_data_cell(ws_sum.cell(row=curr_row, column=2), num_format="0.0%", align="right")
                curr_row += 1
    else:
        ws_sum.cell(row=curr_row, column=1, value="Machine learning model has not been trained yet." if is_en else "Модель машинного обучения еще не была обучена.")
        ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=10, italic=True)

    _apply_thin_borders(ws_sum, 4, curr_row, 1, 2)
    _autofit_columns(ws_sum)

    # =========================================================================
    # Вспомогательная функция для наполнения табличных листов
    # =========================================================================
    def populate_table_sheet(ws_obj, df_source: pd.DataFrame, header_fill="134074"):
        ws_obj.views.sheetView[0].showGridLines = True
        cols = list(df_source.columns)

        # Заголовки
        for c_idx, col_name in enumerate(cols, start=1):
            cell = ws_obj.cell(row=1, column=c_idx, value=col_name)
            _style_header_cell(cell, fill_hex=header_fill)

        # Данные
        for r_idx, row_vals in enumerate(df_source.itertuples(index=False), start=2):
            for c_idx, val in enumerate(row_vals, start=1):
                cell = ws_obj.cell(row=r_idx, column=c_idx)
                if pd.isna(val):
                    cell.value = ""
                    _style_data_cell(cell)
                elif isinstance(val, (int, np.integer)):
                    cell.value = int(val)
                    _style_data_cell(cell, num_format="0", align="right")
                elif isinstance(val, (float, np.floating)):
                    cell.value = float(val)
                    _style_data_cell(cell, num_format="0.000", align="right")
                else:
                    cell.value = str(val)
                    _style_data_cell(cell, align="left")

        _apply_thin_borders(ws_obj, 1, len(df_source) + 1, 1, len(cols))
        _autofit_columns(ws_obj)

    # =========================================================================
    # ЛИСТ 2: Полная матрица признаков (Fused Feature Matrix)
    # =========================================================================
    ws_fused = wb.create_sheet(title="Fused Feature Matrix")
    populate_table_sheet(ws_fused, fused_df, header_fill="0B2545")

    # =========================================================================
    # ЛИСТ 3: FT-ICR MS дескрипторы (FT-ICR MS Descriptors)
    # =========================================================================
    ft_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + ft_cols
    if len(ft_cols) > 0:
        ws_ft = wb.create_sheet(title="FT-ICR MS Descriptors")
        populate_table_sheet(ws_ft, fused_df[ft_export_cols], header_fill="0B5394")

    # =========================================================================
    # ЛИСТ 4: EEM-PARAFAC дескрипторы (EEM Descriptors)
    # =========================================================================
    eem_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + eem_cols
    if len(eem_cols) > 0:
        ws_eem = wb.create_sheet(title="EEM Descriptors")
        populate_table_sheet(ws_eem, fused_df[eem_export_cols], header_fill="E69138")

    # =========================================================================
    # ЛИСТ 5: UV-Vis дескрипторы (UV-Vis Descriptors)
    # =========================================================================
    uv_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + uv_cols
    if len(uv_cols) > 0:
        ws_uv = wb.create_sheet(title="UV-Vis Descriptors")
        populate_table_sheet(ws_uv, fused_df[uv_export_cols], header_fill="2CA02C")

    # =========================================================================
    # ЛИСТ 6: Биомаркеры и VIP-ранжирование (Biomarkers & VIP)
    # =========================================================================
    if model_results and "VIP_df" in model_results:
        vip_df = model_results["VIP_df"].copy()
        if not vip_df.empty:
            ws_vip = wb.create_sheet(title="Biomarkers & VIP")
            ws_vip.views.sheetView[0].showGridLines = True

            headers = ["Descriptor", "VIP Score", "Analytical Block", "Status (VIP > 1.0)"]
            for c_idx, h in enumerate(headers, start=1):
                cell = ws_vip.cell(row=1, column=c_idx, value=h)
                _style_header_cell(cell, fill_hex="134074")

            for r_idx, row in enumerate(vip_df.itertuples(index=False), start=2):
                desc = getattr(row, "Descriptor")
                vip_v = float(getattr(row, "VIP"))
                block = getattr(row, "Block", "Other")
                is_sig = (vip_v >= 1.0)

                bg_col = "E6F4EA" if is_sig else None  # Светло-зеленый для значимых маркеров

                c1 = ws_vip.cell(row=r_idx, column=1, value=desc)
                _style_data_cell(c1, bold=is_sig, bg_color=bg_col)

                c2 = ws_vip.cell(row=r_idx, column=2, value=vip_v)
                _style_data_cell(c2, num_format="0.000", align="right", bold=is_sig, bg_color=bg_col)

                c3 = ws_vip.cell(row=r_idx, column=3, value=block)
                _style_data_cell(c3, align="center", bg_color=bg_col)

                c4 = ws_vip.cell(row=r_idx, column=4, value="Significant Biomarker" if is_sig else "Sub-threshold")
                _style_data_cell(c4, align="center", bold=is_sig, bg_color=bg_col)

            _apply_thin_borders(ws_vip, 1, len(vip_df) + 1, 1, 4)
            _autofit_columns(ws_vip)

    # =========================================================================
    # ЛИСТ 7: S-Plot (для OPLS-DA моделей)
    # =========================================================================
    if model_results and "S_Plot_df" in model_results:
        s_df = model_results["S_Plot_df"].copy()
        if not s_df.empty:
            ws_s = wb.create_sheet(title="OPLS-DA S-Plot")
            populate_table_sheet(ws_s, s_df, header_fill="4A154B")

    # Сохраняем в память
    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)
    return out_buf.getvalue()
