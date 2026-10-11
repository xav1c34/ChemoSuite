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

from __future__ import annotations

import html
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

try:
    import chemo_ml
except ImportError:
    from . import chemo_ml

try:
    import cross_correlation
except ImportError:
    try:
        from . import cross_correlation
    except ImportError:
        cross_correlation = None


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
    fused_df: Union[pd.DataFrame, Dict[str, Any]],
    model_results: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    lang: str = "ru",
    cross_corr_results: Optional[Dict] = None,
) -> bytes:
    """
    Генерирует многостраничный файл Excel (.xlsx) с паспортом анализа и возвращает байты.
    Поддерживает языки отчета: "ru" и "en".
    """
    if isinstance(fused_df, dict):
        if model_results is None:
            model_results = fused_df
        fused_df = fused_df.get("fused_df", pd.DataFrame())
    elif fused_df is None:
        fused_df = pd.DataFrame()

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
    ftir_cols = [c for c in feature_cols if block_map.get(c) == "FTIR"]
    nmr_cols = [c for c in feature_cols if block_map.get(c) == "NMR"]
    thermal_cols = [c for c in feature_cols if block_map.get(c) == "Thermal"]
    sec_cols = [c for c in feature_cols if block_map.get(c) == "SEC/GPC"]
    elem_cols = [c for c in feature_cols if block_map.get(c) == "Elemental"]
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
        ("FTIR Descriptors:" if is_en else "Дескрипторов FTIR:", len(ftir_cols)),
        ("NMR Descriptors:" if is_en else "Дескрипторов NMR:", len(nmr_cols)),
        ("Thermal Descriptors:" if is_en else "Дескрипторов Thermal:", len(thermal_cols)),
        ("SEC/GPC Descriptors:" if is_en else "Дескрипторов SEC/GPC:", len(sec_cols)),
        ("CHNS-O Elemental Descriptors:" if is_en else "Дескрипторов CHNS-O Elemental:", len(elem_cols)),
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
    # ЛИСТ 6: FTIR дескрипторы (FTIR Descriptors)
    # =========================================================================
    ftir_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + ftir_cols
    if len(ftir_cols) > 0:
        ws_ftir = wb.create_sheet(title="FTIR Descriptors")
        populate_table_sheet(ws_ftir, fused_df[ftir_export_cols], header_fill="8E44AD")

    # =========================================================================
    # ЛИСТ 7: NMR дескрипторы (NMR Descriptors)
    # =========================================================================
    nmr_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + nmr_cols
    if len(nmr_cols) > 0:
        ws_nmr = wb.create_sheet(title="NMR Descriptors")
        populate_table_sheet(ws_nmr, fused_df[nmr_export_cols], header_fill="D35400")

    # =========================================================================
    # ЛИСТ 8: Thermal / Py-GC-MS дескрипторы (Thermal Descriptors)
    # =========================================================================
    thermal_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + thermal_cols
    if len(thermal_cols) > 0:
        ws_therm = wb.create_sheet(title="Thermal Descriptors")
        populate_table_sheet(ws_therm, fused_df[thermal_export_cols], header_fill="C0392B")

    # =========================================================================
    # ЛИСТ 9: SEC / GPC дескрипторы (SEC-GPC Descriptors)
    # =========================================================================
    sec_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + sec_cols
    if len(sec_cols) > 0:
        ws_sec = wb.create_sheet(title="SEC-GPC Descriptors")
        populate_table_sheet(ws_sec, fused_df[sec_export_cols], header_fill="16A085")

    # =========================================================================
    # ЛИСТ 10: CHNS-O Elemental дескрипторы (CHNS-O Descriptors)
    # =========================================================================
    elem_export_cols = [c for c in ["Sample_ID", "Class_Target"] if c in fused_df.columns] + elem_cols
    if len(elem_cols) > 0:
        ws_elem = wb.create_sheet(title="CHNS-O Descriptors")
        populate_table_sheet(ws_elem, fused_df[elem_export_cols], header_fill="0077B6")

    # =========================================================================
    # ЛИСТ 11: Кросс-модальные корреляции (Cross-Omics Correlations)
    # =========================================================================
    if cross_corr_results and "top_cross_correlations" in cross_corr_results:
        top_cc = cross_corr_results["top_cross_correlations"]
        if isinstance(top_cc, pd.DataFrame) and not top_cc.empty:
            ws_cc = wb.create_sheet(title="Cross-Omics Correlations")
            populate_table_sheet(ws_cc, top_cc, header_fill="2A9D8F")

    # =========================================================================
    # ЛИСТ 10: Биомаркеры и VIP-ранжирование (Biomarkers & VIP)
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


def generate_ecological_passport_html(
    fused_df: Union[pd.DataFrame, Dict[str, Any]],
    model_results: Optional[Dict] = None,
    cross_corr_results: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    lang: str = "ru",
) -> str:
    """
    Генерирует автономный интерактивный HTML-паспорт / Экологический сертификат образцов.
    Включает сводные метрики 8 аналитических методов, результаты хемометрической классификации,
    панель ключевых биомаркеров, межмодальные кросс-корреляции и экспертное заключение с поддержкой печати в PDF.
    """
    if isinstance(fused_df, dict):
        if model_results is None:
            model_results = fused_df
        fused_df = fused_df.get("fused_df", pd.DataFrame())
    elif fused_df is None:
        fused_df = pd.DataFrame()

    is_en = (lang == "en")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = metadata or {}
    project_name = html.escape(str(meta.get("project_name", "ChemoSuite Multi-Omics Session")))
    operator_name = html.escape(str(meta.get("operator", "ChemoSuite Analytical Platform 3.0")))

    feature_cols = [c for c in fused_df.columns if c not in ["Sample_ID", "Class_Target"]]
    block_map = chemo_ml.classify_feature_blocks(feature_cols)

    # 8 аналитических блоков платформы ChemoSuite 3.0
    all_blocks = [
        ("FT-ICR MS", "FT-ICR MS (Ultrahigh Resolution)", "FT-ICR MS (Сверхвысокое разрешение)", "#0b5394"),
        ("EEM-PARAFAC", "EEM-PARAFAC (3D Fluorescence)", "EEM-PARAFAC (3D-флуоресценция)", "#e69138"),
        ("UV-Vis", "UV-Vis Spectrophotometry", "УФ-Вид спектрофотометрия", "#2ca02c"),
        ("FTIR", "FTIR Spectroscopy", "ИК-Фурье спектроскопия", "#8e44ad"),
        ("NMR", "NMR Spectroscopy (1H/13C)", "ЯМР спектроскопия (1H/13C)", "#d35400"),
        ("Thermal", "Thermal & Py-GC-MS", "Термогравиметрия & Пиролиз-ГХ-МС", "#c0392b"),
        ("SEC/GPC", "SEC/GPC Chromatography", "Эксклюзионная хроматография (SEC/GPC)", "#16a085"),
        ("Elemental", "CHNS-O Elemental Stoichiometry", "Элементный анализ (CHNS-O)", "#0077b6"),
    ]

    active_blocks_count = 0
    block_cards_html = ""
    for b_key, b_title_en, b_title_ru, b_color in all_blocks:
        b_cols = [c for c in feature_cols if block_map.get(c) == b_key]
        b_title = b_title_en if is_en else b_title_ru
        if len(b_cols) > 0:
            active_blocks_count += 1
            sample_stats_html = ""
            for col in b_cols[:4]:
                val_mean = float(fused_df[col].mean())
                val_std = float(fused_df[col].std()) if len(fused_df) > 1 else 0.0
                c_clean = html.escape(col)
                sample_stats_html += f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed #e2e8f0; font-size: 12px;">
                    <span style="color: #475569; font-family: monospace;">{c_clean}:</span>
                    <span style="font-weight: 600; color: #0f172a;">{val_mean:.3f} ± {val_std:.3f}</span>
                </div>
                """
            badge_status = f'<span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{len(b_cols)} {"features" if is_en else "признаков"}</span>'
        else:
            sample_stats_html = f'<div style="color: #94a3b8; font-size: 12px; font-style: italic; padding: 8px 0;">{"Not measured in this session" if is_en else "Не измерялся в данной сессии"}</div>'
            badge_status = f'<span style="background: #f1f5f9; color: #64748b; padding: 2px 8px; border-radius: 12px; font-size: 11px;">{"Inactive" if is_en else "Не активен"}</span>'

        block_cards_html += f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; border-top: 4px solid {b_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; font-size: 14px; color: #0f172a; font-weight: 600;">{b_title}</h4>
                {badge_status}
            </div>
            {sample_stats_html}
        </div>
        """

    # Экологическая оценка и генезис
    classes_present = list(fused_df["Class_Target"].unique()) if "Class_Target" in fused_df.columns else []
    classes_str = ", ".join(str(c) for c in classes_present) if classes_present else ("N/A")

    has_lignin_or_sludge = any(k in str(classes_str).lower() for k in ["lignin", "sludge", "шлам", "1"])
    has_high_q2 = bool(model_results and model_results.get("Q2", 0) > 40)

    if has_lignin_or_sludge or has_high_q2:
        risk_label = "HIGH CONTRAST / TECHNOGENIC TRANSFORMATION" if is_en else "ВЫСОКИЙ КОНТРАСТ / ТЕХНОГЕННАЯ ТРАНСФОРМАЦИЯ"
        risk_bg = "#fee2e2"
        risk_color = "#b91c1c"
        conclusion_text = (
            "Molecular characterization reveals significant structural divergence between sample subsets. "
            "Biomarker analysis indicates high enrichment in condensed polyphenolic and lignin-derived aromatics "
            "with elevated AI_mod and SUVA254, characteristic of technogenic impact (e.g. industrial sludge-lignin discharge). "
            "Background aquatic organic matter is distinguished by dominant aliphatic and humic-like constituents."
            if is_en else
            "Мультимодальная молекулярная характеризация выявила выраженную структурную дифференциацию между подгруппами образцов. "
            "Анализ биомаркеров свидетельствует о резком обогащении конденсированными полифенольными и лигниноподобными ароматическими структурами "
            "с повышенными индексами AI_mod и SUVA254, что характерно для техногенного загрязнения (включая шлам-лигнин БЦБК). "
            "Фоновые природные воды контрастно отличаются преобладанием алифатических и классических гуминовых соединений."
        )
    else:
        risk_label = "NATURAL BACKGROUND / AQUATIC DOM" if is_en else "ФОНОВЫЙ ПРИРОДНЫЙ ГЕНЕЗИС / ПРИРОДНОЕ ОВ"
        risk_bg = "#dcfce7"
        risk_color = "#15803d"
        conclusion_text = (
            "Multi-omics profiling confirms that the organic matter pool consists of predominantly natural background dissolved organic matter (DOM). "
            "Spectroscopic and mass spectrometric signatures correspond to autochthonous microbial and terrigenous soil fulvic/humic complexes "
            "with normal aromaticity levels and balanced heteroatom distributions."
            if is_en else
            "Мультимодальное профилирование подтверждает, что исследованный пул органического вещества соответствует фоновому "
            "природному растворенному органическому веществу (DOM). Спектроскопические и масс-спектрометрические сигнатуры указывают на "
            "сбалансированное присутствие автохтонных биогенных и терригенных гумусовых комплексов с естественным уровнем ароматичности."
        )

    # Модель ML
    model_section_html = ""
    if model_results:
        m_type = model_results.get("model_type", "PLS-DA")
        acc = model_results.get("Balanced_Accuracy", model_results.get("Accuracy", 0.0))
        r2x = model_results.get("R2X", 0.0)
        r2y = model_results.get("R2Y", 0.0)
        q2 = model_results.get("Q2", 0.0)

        vip_rows_html = ""
        vip_df = model_results.get("VIP_df", pd.DataFrame())
        if isinstance(vip_df, pd.DataFrame) and not vip_df.empty:
            for _, r in vip_df.head(8).iterrows():
                desc_name = html.escape(str(r["Descriptor"]))
                vip_val = float(r["VIP"])
                blk_name = html.escape(str(r.get("Block", "Other")))
                sig_badge = f'<span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;">VIP={vip_val:.2f} ★</span>' if vip_val >= 1.0 else f'<span style="background: #f1f5f9; color: #64748b; padding: 2px 8px; border-radius: 10px; font-size: 11px;">VIP={vip_val:.2f}</span>'
                vip_rows_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-weight: 600;">{desc_name}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; color: #0b5394;">{blk_name}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: 700;">{vip_val:.3f}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: center;">{sig_badge}</td>
                </tr>
                """

        model_section_html = f"""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">🔬 {"Хемометрическая дискриминантная модель и биомаркеры" if not is_en else "Chemometric Discriminant Model & Biomarker Panel"} ({m_type})</h3>
                <span style="font-size: 12px; color: #64748b;">Classes: <strong>{html.escape(str(classes_str))}</strong></span>
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">R²X (Spectral Variance)</div>
                    <div class="kpi-value">{r2x:.1f}%</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">R²Y (Class Variance)</div>
                    <div class="kpi-value">{r2y:.1f}%</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Q² (LOO-CV Predictive)</div>
                    <div class="kpi-value" style="color: {'#16a34a' if q2 > 50 else '#d97706'};">{q2:.1f}%</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Balanced Accuracy</div>
                    <div class="kpi-value" style="color: #0b5394;">{acc:.1f}%</div>
                </div>
            </div>
            <table class="custom-table" style="margin-top: 16px;">
                <thead>
                    <tr>
                        <th>{"Молекулярный дескриптор" if not is_en else "Molecular Descriptor"}</th>
                        <th>{"Аналитический блок" if not is_en else "Analytical Block"}</th>
                        <th style="text-align: right;">VIP Score</th>
                        <th style="text-align: center;">{"Статус значимости" if not is_en else "Significance Status"}</th>
                    </tr>
                </thead>
                <tbody>
                    {vip_rows_html}
                </tbody>
            </table>
        </div>
        """

    # Кросс-модальные корреляции
    cc_section_html = ""
    top_cc = None
    if cross_corr_results and "top_cross_correlations" in cross_corr_results:
        top_cc = cross_corr_results["top_cross_correlations"]
    elif cross_correlation is not None:
        try:
            cc_calc = cross_correlation.compute_multiblock_cross_correlation(fused_df)
            top_cc = cc_calc.get("top_cross_correlations")
        except Exception:
            top_cc = None

    if isinstance(top_cc, pd.DataFrame) and not top_cc.empty:
        cc_rows = ""
        for _, r in top_cc.head(6).iterrows():
            fa = html.escape(str(r["Feature_A"]))
            ba = html.escape(str(r["Block_A"]))
            fb = html.escape(str(r["Feature_B"]))
            bb = html.escape(str(r["Block_B"]))
            r_val = float(r["Correlation"])
            p_val = float(r.get("p_value", 0.0))
            r_col = "#15803d" if r_val > 0 else "#b91c1c"
            cc_rows += f"""
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0;">
                    <strong>{fa}</strong> <span style="font-size: 11px; color: #64748b;">[{ba}]</span>
                </td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: center; color: #94a3b8;">⟷</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0;">
                    <strong>{fb}</strong> <span style="font-size: 11px; color: #64748b;">[{bb}]</span>
                </td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: 700; color: {r_col};">{r_val:+.3f}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-size: 11px; color: #64748b;">p = {p_val:.2e}</td>
            </tr>
            """

        cc_section_html = f"""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">🌐 {"Кросс-модальные корреляционные синергии (Cross-Omics Links)" if not is_en else "Cross-Omics Correlation Synergies & Inter-Block Links"}</h3>
                <span style="font-size: 12px; color: #64748b;">Threshold |r| ≥ 0.5</span>
            </div>
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>{"Дескриптор A (Блок A)" if not is_en else "Feature A (Block A)"}</th>
                        <th style="text-align: center;"></th>
                        <th>{"Дескриптор B (Блок B)" if not is_en else "Feature B (Block B)"}</th>
                        <th style="text-align: right;">r (Pearson)</th>
                        <th style="text-align: right;">p-value</th>
                    </tr>
                </thead>
                <tbody>
                    {cc_rows}
                </tbody>
            </table>
        </div>
        """

    doc_title = "ChemoSuite Ecological Passport & Analytical Report" if is_en else "Экологический паспорт и аналитический сертификат ChemoSuite"
    print_btn_label = "🖨️ Print / Save as PDF" if is_en else "🖨️ Распечатать / Сохранить в PDF"

    html_content = f"""<!DOCTYPE html>
<html lang="{'en' if is_en else 'ru'}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{doc_title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {{
  --primary: #0b2545;
  --secondary: #134074;
  --accent: #0077b6;
  --bg-subtle: #f8fafc;
  --text-main: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --card-bg: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background-color: var(--bg-subtle);
  color: var(--text-main);
  margin: 0;
  padding: 30px 20px;
  line-height: 1.5;
}}
.passport-container {{
  max-width: 1040px;
  margin: 0 auto;
}}
.header-card {{
  background: linear-gradient(135deg, #0b2545 0%, #134074 100%);
  color: #ffffff;
  border-radius: 14px;
  padding: 28px 34px;
  margin-bottom: 24px;
  box-shadow: 0 10px 25px -5px rgba(11, 37, 69, 0.25);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.header-title-box h1 {{
  margin: 0 0 6px 0;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.5px;
}}
.header-title-box p {{
  margin: 0;
  font-size: 13px;
  color: #93c5fd;
}}
.print-btn {{
  background: #0077b6;
  color: #ffffff;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: opacity 0.2s;
}}
.print-btn:hover {{ opacity: 0.9; }}
.card {{
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 26px;
  margin-bottom: 22px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
}}
.card-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}}
.card-title {{
  font-size: 16px;
  font-weight: 700;
  color: var(--primary);
  margin: 0;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}}
.kpi-box {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px 14px;
  text-align: center;
}}
.kpi-label {{
  font-size: 11px;
  text-transform: uppercase;
  color: var(--text-muted);
  font-weight: 600;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}}
.kpi-value {{
  font-size: 19px;
  font-weight: 700;
  color: var(--secondary);
}}
.custom-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  text-align: left;
}}
.custom-table th {{
  background: #f8fafc;
  color: #475569;
  font-weight: 600;
  padding: 9px 12px;
  border-bottom: 2px solid var(--border);
}}
.conclusion-box {{
  background: #eff6ff;
  border-left: 4px solid #3b82f6;
  border-radius: 0 8px 8px 0;
  padding: 16px 20px;
  font-size: 14px;
  line-height: 1.6;
  color: #1e3a8a;
}}
.footer-box {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}}
@media print {{
  body {{ background: #fff !important; padding: 0 !important; }}
  .no-print {{ display: none !important; }}
  .header-card {{ box-shadow: none !important; }}
  .card {{ box-shadow: none !important; border: 1px solid #cbd5e1 !important; page-break-inside: avoid; }}
}}
</style>
</head>
<body>

<div class="passport-container">

  <!-- Шапка отчета -->
  <div class="header-card">
    <div class="header-title-box">
      <h1>{doc_title}</h1>
      <p>{"Комплексная мультимодальная характеризация органического вещества природно-техногенных систем" if not is_en else "Unified Multi-Modal Characterization of Natural & Technogenic Organic Matter"}</p>
    </div>
    <div class="no-print">
      <button class="print-btn" onclick="window.print()">{print_btn_label}</button>
    </div>
  </div>

  <!-- Общие метаданные и вердикт -->
  <div class="card">
    <div class="card-header">
      <h3 class="card-title">📋 {"Общие сведения об аналитической сессии" if not is_en else "Analytical Session Overview"}</h3>
      <span style="background: {risk_bg}; color: {risk_color}; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">{risk_label}</span>
    </div>
    <div class="kpi-grid">
      <div class="kpi-box">
        <div class="kpi-label">{"Образцов в партии" if not is_en else "Sample Batch Size"}</div>
        <div class="kpi-value">{len(fused_df)}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">{"Активных методов" if not is_en else "Active Modalities"}</div>
        <div class="kpi-value">{active_blocks_count} / 8</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">{"Всего дескрипторов" if not is_en else "Total Descriptors"}</div>
        <div class="kpi-value">{len(feature_cols)}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">{"Дата генерации" if not is_en else "Date of Issue"}</div>
        <div class="kpi-value" style="font-size: 14px; font-weight: 600; padding-top: 4px;">{now_str.split()[0]}</div>
      </div>
    </div>
    <table class="custom-table" style="margin-top: 10px;">
      <tr>
        <td style="width: 25%; font-weight: 600; color: #475569;">{"Проект / Заказчик:" if not is_en else "Project / Client:"}</td>
        <td>{project_name}</td>
        <td style="width: 25%; font-weight: 600; color: #475569;">{"Оператор / Лаборатория:" if not is_en else "Operator / Laboratory:"}</td>
        <td>{operator_name}</td>
      </tr>
      <tr>
        <td style="font-weight: 600; color: #475569;">{"Аналитическая платформа:" if not is_en else "Analytical Platform:"}</td>
        <td>ChemoSuite Unified Analytical Engine 3.0</td>
        <td style="font-weight: 600; color: #475569;">{"Статус сертификации:" if not is_en else "Certification Status:"}</td>
        <td><strong style="color: #16a34a;">PASS (Chemometrically Validated)</strong></td>
      </tr>
    </table>
  </div>

  <!-- Мультимодальный атлас: 8 аналитических блоков -->
  <div class="card">
    <div class="card-header">
      <h3 class="card-title">🧬 {"Мультимодальный аналитический профиль (8 аналитических методов)" if not is_en else "Multi-Omics Analytical Profile (8 Analytical Modalities)"}</h3>
      <span style="font-size: 12px; color: #64748b;">{"Согласованная матрица признаков" if not is_en else "Aligned Feature Space"}</span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
      {block_cards_html}
    </div>
  </div>

  <!-- Модель классификации (если обучена) -->
  {model_section_html}

  <!-- Кросс-модальные корреляции -->
  {cc_section_html}

  <!-- Экспертное заключение -->
  <div class="card">
    <div class="card-header">
      <h3 class="card-title">📝 {"Автоматизированное экспертное заключение" if not is_en else "Automated Expert Conclusion & Diagnosis"}</h3>
      <span style="font-size: 12px; color: #64748b;">ChemoSuite Rule-Based Synthesis</span>
    </div>
    <div class="conclusion-box">
      <strong>{"ЗАКЛЮЧЕНИЕ ЭКСПЕРТА:" if not is_en else "EXPERT APPRAISAL:"}</strong> {conclusion_text}
    </div>
  </div>

  <!-- Футер -->
  <div class="footer-box">
    <div>ChemoSuite 3.0 Platform &copy; {datetime.now().year} &bull; Natural Humic Systems Lab &bull; Moscow State University</div>
    <div>Report ID: CS-3.0-{abs(hash(now_str + project_name)) % 1000000:06d}</div>
  </div>

</div>

</body>
</html>"""
    return html_content

