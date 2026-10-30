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
import re
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


# ==============================================================================
# Защита от формульных инъекций в Excel/CSV (OWASP «CSV / Formula Injection»)
# ==============================================================================
# Строки, начинающиеся с = + - @ (а также TAB / CR), Excel и LibreOffice могут
# интерпретировать как формулы (например, =HYPERLINK(...), =cmd|...). Такие строки
# экранируются ведущим апострофом. Обычные числовые литералы («-28.5», «+3») не
# изменяются, чтобы не портить легитимные данные.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def sanitize_excel_value(value: Any) -> Any:
    """Возвращает значение, безопасное для ячейки Excel (строки-формулы получают префикс «'»)."""
    if not isinstance(value, str) or not value:
        return value
    if value[0] in _FORMULA_PREFIXES and not _PLAIN_NUMBER_RE.match(value):
        return "'" + value
    return value


def sanitize_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Копия DataFrame, безопасная для экспорта в Excel/CSV: экранируются строковые ячейки,
    заголовки столбцов и строковый индекс. Числовые столбцы не изменяются.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    for i in range(out.shape[1]):
        s = out.iloc[:, i]
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            out.isetitem(i, s.map(sanitize_excel_value))
    if not isinstance(out.columns, pd.MultiIndex):
        out.columns = pd.Index([sanitize_excel_value(c) for c in out.columns], name=out.columns.name)
    if not isinstance(out.index, pd.MultiIndex) and (
        out.index.dtype == object or pd.api.types.is_string_dtype(out.index.dtype)
    ):
        out.index = pd.Index([sanitize_excel_value(v) for v in out.index], name=out.index.name)
    return out


# ==============================================================================
# Форматирование метрик: отсутствующее / NaN значение -> «—» (никаких выдуманных 0.0)
# ==============================================================================
_MISSING = "—"


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_pct(value: Any, nd: int = 1) -> str:
    """Метрика уже в процентах (как возвращает chemo_ml) -> «85.3%» или «—»."""
    f = _num(value)
    return f"{f:.{nd}f}%" if np.isfinite(f) else _MISSING


def _fmt_num(value: Any, nd: int = 3) -> str:
    f = _num(value)
    return f"{f:.{nd}f}" if np.isfinite(f) else _MISSING


def _fmt_p(value: Any) -> str:
    f = _num(value)
    if not np.isfinite(f):
        return _MISSING
    return f"{f:.3f}" if f >= 0.001 else f"{f:.1e}"


def _first_metric(d: Optional[Dict], keys: Tuple[str, ...]) -> Tuple[float, Optional[str]]:
    """Первое конечное значение из перечня ключей (NaN, None, если ни одного нет)."""
    if not isinstance(d, dict):
        return float("nan"), None
    for k in keys:
        if k in d:
            f = _num(d.get(k))
            if np.isfinite(f):
                return f, k
    return float("nan"), None


def _resolve_permutation_p(model_results: Optional[Dict], permutation_results: Optional[Dict]) -> float:
    """p-value перестановочного теста Q² (chemo_ml.permutation_test_plsda: ключ p_val_q2)."""
    for src in (permutation_results, model_results):
        if isinstance(src, dict) and "p_val_q2" in src:
            f = _num(src.get("p_val_q2"))
            if np.isfinite(f):
                return f
    return float("nan")


# ==============================================================================
# Статус валидации модели и автоматическая сводка по данным (без «экспертных» выводов)
# ==============================================================================
# Пороги: Q² > 0.5 — «хорошая» прогностическая способность (Eriksson L. et al.,
# Multi- and Megavariate Data Analysis, Umetrics, 2006); значимость Q² проверяется
# перестановочным тестом, а малые выборки дают сильно завышенные/нестабильные оценки
# CV (Westerhuis J.A. et al., Metabolomics 4 (2008) 81–89).
Q2_GOOD_PCT = 50.0
PERMUTATION_ALPHA = 0.05
SMALL_N_SAMPLES = 30
SMALL_N_PER_CLASS = 10


def assess_model_validation(
    model_results: Optional[Dict],
    permutation_results: Optional[Dict] = None,
    lang: str = "ru",
) -> Dict[str, Any]:
    """
    Статус валидации классификационной модели по Q² (CV) и p-value перестановочного теста.

    status: "no_model" | "unverified" (нет Q² или нет перестановочного теста) |
            "not_validated" (Q² ≤ 0 или p ≥ 0.05) | "weak" (p < 0.05, Q² ≤ 50%) |
            "validated" (p < 0.05, Q² > 50%).
    """
    is_en = (lang == "en")
    q2, _ = _first_metric(model_results, ("Q2",))
    p_val = _resolve_permutation_p(model_results, permutation_results)

    if not model_results:
        status = "no_model"
        label = "NO MODEL: DESCRIPTIVE STATISTICS ONLY" if is_en else "МОДЕЛЬ НЕ ОБУЧЕНА: ТОЛЬКО ОПИСАТЕЛЬНАЯ СТАТИСТИКА"
        color, bg = "#475569", "#f1f5f9"
    elif not np.isfinite(q2):
        status = "unverified"
        label = "UNVERIFIED: Q² NOT AVAILABLE" if is_en else "НЕ ПОДТВЕРЖДЕНО: Q² НЕ РАССЧИТАН"
        color, bg = "#b45309", "#fef3c7"
    elif q2 <= 0.0:
        status = "not_validated"
        label = "NOT VALIDATED: Q² ≤ 0 (NO PREDICTIVE ABILITY)" if is_en else "НЕ ВАЛИДИРОВАНО: Q² ≤ 0 (НЕТ ПРОГНОСТИЧЕСКОЙ СПОСОБНОСТИ)"
        color, bg = "#b91c1c", "#fee2e2"
    elif not np.isfinite(p_val):
        status = "unverified"
        label = "UNVERIFIED: PERMUTATION TEST NOT RUN" if is_en else "НЕ ПОДТВЕРЖДЕНО: ПЕРЕСТАНОВОЧНЫЙ ТЕСТ НЕ ВЫПОЛНЕН"
        color, bg = "#b45309", "#fef3c7"
    elif p_val >= PERMUTATION_ALPHA:
        status = "not_validated"
        label = (f"NOT VALIDATED: PERMUTATION p = {_fmt_p(p_val)} ≥ {PERMUTATION_ALPHA}" if is_en
                 else f"НЕ ВАЛИДИРОВАНО: ПЕРЕСТАНОВОЧНЫЙ p = {_fmt_p(p_val)} ≥ {PERMUTATION_ALPHA}")
        color, bg = "#b91c1c", "#fee2e2"
    elif q2 <= Q2_GOOD_PCT:
        status = "weak"
        label = (f"WEAK: p = {_fmt_p(p_val)}, Q² ≤ {Q2_GOOD_PCT:.0f}%" if is_en
                 else f"СЛАБАЯ МОДЕЛЬ: p = {_fmt_p(p_val)}, Q² ≤ {Q2_GOOD_PCT:.0f}%")
        color, bg = "#b45309", "#fef3c7"
    else:
        status = "validated"
        label = (f"VALIDATED: Q² > {Q2_GOOD_PCT:.0f}%, PERMUTATION p = {_fmt_p(p_val)}" if is_en
                 else f"ВАЛИДИРОВАНО: Q² > {Q2_GOOD_PCT:.0f}%, ПЕРЕСТАНОВОЧНЫЙ p = {_fmt_p(p_val)}")
        color, bg = "#15803d", "#dcfce7"

    return {"status": status, "label": label, "color": color, "bg": bg, "q2": q2, "p_value": p_val}


def _class_counts(fused_df: pd.DataFrame) -> Dict[str, int]:
    if not isinstance(fused_df, pd.DataFrame) or "Class_Target" not in fused_df.columns:
        return {}
    vc = fused_df["Class_Target"].dropna().astype(str).value_counts(sort=False)
    return {str(k): int(v) for k, v in vc.items()}


def _top_markers_with_direction(
    fused_df: pd.DataFrame, vip_df: Any, max_markers: int = 5, vip_threshold: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Дескрипторы с VIP ≥ 1 (по убыванию VIP) и направление различия: в каком классе
    среднее значение (по исходной, немасштабированной матрице) максимально / минимально.
    """
    if not isinstance(vip_df, pd.DataFrame) or vip_df.empty or "Descriptor" not in vip_df.columns or "VIP" not in vip_df.columns:
        return []
    v = vip_df.copy()
    v["VIP"] = pd.to_numeric(v["VIP"], errors="coerce")
    v = v[v["VIP"] >= vip_threshold].sort_values("VIP", ascending=False).head(max_markers)
    has_cls = isinstance(fused_df, pd.DataFrame) and "Class_Target" in fused_df.columns
    out = []
    for _, r in v.iterrows():
        desc = str(r["Descriptor"])
        item = {"descriptor": desc, "vip": float(r["VIP"]), "block": str(r.get("Block", "")),
                "high_class": None, "low_class": None, "high_mean": float("nan"), "low_mean": float("nan"),
                "n_classes": 0}
        if has_cls and desc in fused_df.columns:
            vals = pd.to_numeric(fused_df[desc], errors="coerce")
            means = vals.groupby(fused_df["Class_Target"].astype(str)).mean().dropna()
            if len(means) >= 2 and float(means.max()) != float(means.min()):
                item.update(high_class=str(means.idxmax()), low_class=str(means.idxmin()),
                            high_mean=float(means.max()), low_mean=float(means.min()),
                            n_classes=int(len(means)))
        out.append(item)
    return out


def build_data_driven_summary(
    fused_df: pd.DataFrame,
    model_results: Optional[Dict] = None,
    permutation_results: Optional[Dict] = None,
    active_blocks: Optional[List[str]] = None,
    lang: str = "ru",
) -> Dict[str, Any]:
    """
    Автоматическая сводка, построенная только по данным сессии: дизайн выборки,
    метрики модели (Q², R²Y, точность CV / обучения, p-value перестановок), предупреждение
    о малой выборке, ведущие VIP-маркеры с направлением и вклад аналитических блоков.
    Никаких заранее заготовленных выводов о генезисе/источнике загрязнения не делается.

    Возвращает dict: validation (см. assess_model_validation), items (список строк, plain text),
    small_n (bool), markers (list), html (готовый <ul> для паспорта).
    """
    is_en = (lang == "en")
    df = fused_df if isinstance(fused_df, pd.DataFrame) else pd.DataFrame()
    n = int(len(df))
    feature_cols = [c for c in df.columns if c not in ["Sample_ID", "Class_Target"]]
    counts = _class_counts(df)
    validation = assess_model_validation(model_results, permutation_results, lang=lang)
    items: List[str] = []

    blocks_txt = ", ".join(active_blocks) if active_blocks else _MISSING
    cls_txt = ", ".join(f"{k} (n={v})" for k, v in counts.items()) if counts else _MISSING
    items.append(
        f"Data: {n} samples, {len(feature_cols)} descriptors; modalities ({len(active_blocks or [])}): {blocks_txt}. Classes: {cls_txt}."
        if is_en else
        f"Данные: образцов — {n}, дескрипторов — {len(feature_cols)}; методы ({len(active_blocks or [])}): {blocks_txt}. Классы: {cls_txt}."
    )

    min_class = min(counts.values()) if counts else n
    small_n = bool(n < SMALL_N_SAMPLES or (counts and min_class < SMALL_N_PER_CLASS))
    markers: List[Dict[str, Any]] = []

    if not model_results:
        items.append(
            "No supervised model was trained: the report contains descriptive statistics only; "
            "no class discrimination, source attribution or genesis is inferred."
            if is_en else
            "Модель с учителем не обучалась: отчёт содержит только описательную статистику; "
            "выводы о различии классов, источнике или генезисе ОВ не делаются."
        )
    else:
        m_type = str(model_results.get("model_type", "PLS-DA"))
        cv_scheme = model_results.get("CV_Scheme")
        cv_txt = f", {cv_scheme}" if cv_scheme else ""
        q2 = validation["q2"]
        r2y, _ = _first_metric(model_results, ("R2Y",))
        acc_cv, acc_cv_key = _first_metric(model_results, ("Balanced_Accuracy_CV", "Accuracy_CV"))
        acc_tr, acc_tr_key = _first_metric(model_results, ("Balanced_Accuracy", "Accuracy"))
        if acc_cv_key is not None:
            acc_name = ("balanced accuracy (CV)" if acc_cv_key == "Balanced_Accuracy_CV" else "accuracy (CV)") if is_en else \
                ("сбалансированная точность (CV)" if acc_cv_key == "Balanced_Accuracy_CV" else "точность (CV)")
            acc_txt = f"{acc_name} = {_fmt_pct(acc_cv)}"
        elif acc_tr_key is not None:
            acc_txt = (f"training (resubstitution) accuracy = {_fmt_pct(acc_tr)} — optimistic, not a validation metric"
                       if is_en else
                       f"точность на обучающей выборке (ресубституция) = {_fmt_pct(acc_tr)} — завышенная оценка, не метрика валидации")
        else:
            acc_txt = (f"accuracy = {_MISSING}" if is_en else f"точность = {_MISSING}")
        items.append(
            f"{m_type}: Q² = {_fmt_pct(q2)} (cross-validated{cv_txt}), R²Y = {_fmt_pct(r2y)}, {acc_txt}."
            if is_en else
            f"{m_type}: Q² = {_fmt_pct(q2)} (кросс-валидация{cv_txt}), R²Y = {_fmt_pct(r2y)}, {acc_txt}."
        )
        p_val = validation["p_value"]
        if np.isfinite(p_val):
            items.append(f"Permutation test of Q²: p = {_fmt_p(p_val)}." if is_en else f"Перестановочный тест Q²: p = {_fmt_p(p_val)}.")
        else:
            items.append(
                "Permutation test was not run: the statistical significance of Q² is not established."
                if is_en else
                "Перестановочный тест не выполнялся: статистическая значимость Q² не установлена."
            )
        items.append((f"Validation status: {validation['label']}." if is_en else f"Статус валидации: {validation['label']}."))

        markers = _top_markers_with_direction(df, model_results.get("VIP_df", model_results.get("vip_df")))
        if markers:
            parts = []
            for mk in markers:
                if mk["high_class"] is not None:
                    if mk["n_classes"] == 2:
                        d_txt = (f"higher in class {mk['high_class']} ({mk['high_mean']:.3g} vs {mk['low_mean']:.3g})" if is_en
                                 else f"выше в классе {mk['high_class']} ({mk['high_mean']:.3g} против {mk['low_mean']:.3g})")
                    else:
                        d_txt = (f"highest in {mk['high_class']} ({mk['high_mean']:.3g}), lowest in {mk['low_class']} ({mk['low_mean']:.3g})" if is_en
                                 else f"максимум в {mk['high_class']} ({mk['high_mean']:.3g}), минимум в {mk['low_class']} ({mk['low_mean']:.3g})")
                else:
                    d_txt = "direction n/a" if is_en else "направление не определено"
                parts.append(f"{mk['descriptor']} [VIP {mk['vip']:.2f}; {d_txt}]")
            items.append(("Top markers (VIP ≥ 1): " if is_en else "Ведущие маркеры (VIP ≥ 1): ") + "; ".join(parts) + ".")
        else:
            items.append("No descriptor reached VIP ≥ 1." if is_en else "Ни один дескриптор не достиг VIP ≥ 1.")

        contrib = model_results.get("Block_Contributions") or {}
        contrib_f = sorted(((str(k), _num(v)) for k, v in contrib.items() if np.isfinite(_num(v))), key=lambda kv: -kv[1])
        if contrib_f:
            c_txt = ", ".join(f"{k} {v:.1f}%" for k, v in contrib_f[:3])
            items.append((f"Largest block contributions (share): {c_txt}." if is_en else f"Наибольший вклад блоков (доля): {c_txt}."))

    if small_n:
        items.append(
            f"Caution: small sample set (n = {n}, smallest class n = {min_class}); cross-validated metrics have high "
            "variance and multivariate models overfit easily (Westerhuis et al., 2008). Treat results as exploratory."
            if is_en else
            f"Внимание: малая выборка (n = {n}, наименьший класс n = {min_class}); оценки кросс-валидации имеют большую "
            "дисперсию, многомерные модели легко переобучаются (Westerhuis et al., 2008). Результаты носят разведочный характер."
        )
    items.append(
        "This summary is computed automatically from the session data; it is not an expert interpretation "
        "of the origin or source of the organic matter."
        if is_en else
        "Сводка рассчитана автоматически по данным сессии и не является экспертной интерпретацией "
        "происхождения или источника органического вещества."
    )

    html_list = "<ul style=\"margin: 6px 0 0 0; padding-left: 20px;\">" + "".join(
        f"<li>{html.escape(t)}</li>" for t in items
    ) + "</ul>"
    return {"validation": validation, "items": items, "small_n": small_n, "markers": markers, "html": html_list}


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
    permutation_results: Optional[Dict] = None,
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
        ws_sum.cell(row=idx, column=2, value=sanitize_excel_value(val))
        _style_data_cell(ws_sum.cell(row=idx, column=2))

    # Секция хемометрической модели
    curr_row = 15
    ws_sum.cell(row=curr_row, column=1, value="2. CHEMOMETRIC MODEL SUMMARY" if is_en else "2. СВОДКА ХЕМОМЕТРИЧЕСКОЙ МОДЕЛИ")
    ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=12, bold=True, color="134074")
    curr_row += 1

    if model_results:
        m_type = str(model_results.get("model_type", "PLS-DA"))
        cv_scheme = model_results.get("CV_Scheme")
        # Отсутствующие метрики показываются как «—», а не подставляются нулями
        r2x = model_results.get("R2X")
        r2x_p = model_results.get("R2X_pred", None)
        r2x_o = model_results.get("R2X_ortho", None)
        r2y = model_results.get("R2Y")
        q2 = model_results.get("Q2")
        acc_cv, acc_cv_key = _first_metric(model_results, ("Balanced_Accuracy_CV", "Accuracy_CV"))
        acc_tr, acc_tr_key = _first_metric(model_results, ("Balanced_Accuracy", "Accuracy"))
        p_perm = _resolve_permutation_p(model_results, permutation_results)
        validation = assess_model_validation(model_results, permutation_results, lang=lang)

        q2_lbl = (f"Q² (Predictive Ability, CV{': ' + str(cv_scheme) if cv_scheme else ''}):" if is_en
                  else f"Q² (Прогностическая способность, CV{': ' + str(cv_scheme) if cv_scheme else ''}):")
        if acc_cv_key == "Accuracy_CV":
            acc_cv_lbl = "Accuracy (CV):" if is_en else "Точность (CV):"
        else:
            acc_cv_lbl = "Balanced Accuracy (CV):" if is_en else "Сбалансированная точность (CV):"
        if acc_tr_key == "Accuracy":
            acc_tr_lbl = "Accuracy (train, resubstitution):" if is_en else "Точность (обучение, ресубституция):"
        else:
            acc_tr_lbl = "Balanced Accuracy (train, resubstitution):" if is_en else "Сбалансированная точность (обучение, ресубституция):"

        model_info = [
            ("Model Architecture:" if is_en else "Архитектура модели:", m_type),
            ("R²X (Explained X-variance):" if is_en else "R²X (Объясненная дисперсия X):", _fmt_pct(r2x, 2)),
            ("R²Y (Explained Class Y-variance):" if is_en else "R²Y (Объясненная дисперсия классов Y):", _fmt_pct(r2y, 2)),
            (q2_lbl, _fmt_pct(q2, 2)),
            (acc_cv_lbl, _fmt_pct(acc_cv, 2)),
            (acc_tr_lbl, _fmt_pct(acc_tr, 2)),
            ("Permutation test p(Q²):" if is_en else "Перестановочный тест p(Q²):",
             _fmt_p(p_perm) if np.isfinite(p_perm) else (f"{_MISSING} (not run)" if is_en else f"{_MISSING} (не выполнялся)")),
            ("Validation status:" if is_en else "Статус валидации:", validation["label"]),
        ]
        if r2x_p is not None and r2x_o is not None:
            model_info.append(("R²X_pred (Predictive Variance):" if is_en else "R²X_pred (Предиктивная дисперсия):", _fmt_pct(r2x_p, 2)))
            model_info.append(("R²X_ortho (Orthogonal Noise):" if is_en else "R²X_ortho (Ортогональный шум):", _fmt_pct(r2x_o, 2)))

        for lbl, val in model_info:
            ws_sum.cell(row=curr_row, column=1, value=lbl)
            ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=10, bold=True)
            ws_sum.cell(row=curr_row, column=2, value=sanitize_excel_value(val))
            _style_data_cell(ws_sum.cell(row=curr_row, column=2))
            curr_row += 1

        # Вклад аналитических блоков
        b_contrib = model_results.get("Block_Contributions", {})
        if b_contrib:
            curr_row += 1
            # Нейтральная подпись: для PLS-DA это доля ΣVIP², для MB-PLS — доля объяснённой Y,
            # для SO-PLS — инкрементальный прирост R²Y (см. chemo_ml)
            ws_sum.cell(row=curr_row, column=1, value="3. RELATIVE ANALYTICAL BLOCK CONTRIBUTION (SHARE)" if is_en else "3. ОТНОСИТЕЛЬНЫЙ ВКЛАД АНАЛИТИЧЕСКИХ БЛОКОВ (ДОЛЯ)")
            ws_sum.cell(row=curr_row, column=1).font = Font(name="Calibri", size=12, bold=True, color="134074")
            curr_row += 1

            ws_sum.cell(row=curr_row, column=1, value="Analytical Block" if is_en else "Аналитический блок")
            ws_sum.cell(row=curr_row, column=2, value="Block contribution (share)" if is_en else "Вклад блока (доля)")
            _style_header_cell(ws_sum.cell(row=curr_row, column=1), fill_hex="134074")
            _style_header_cell(ws_sum.cell(row=curr_row, column=2), fill_hex="134074")
            curr_row += 1

            for b_name, b_pct in b_contrib.items():
                ws_sum.cell(row=curr_row, column=1, value=sanitize_excel_value(str(b_name)))
                _style_data_cell(ws_sum.cell(row=curr_row, column=1), bold=True)
                b_val = _num(b_pct)
                if np.isfinite(b_val):
                    ws_sum.cell(row=curr_row, column=2, value=b_val / 100.0)
                    _style_data_cell(ws_sum.cell(row=curr_row, column=2), num_format="0.0%", align="right")
                else:
                    ws_sum.cell(row=curr_row, column=2, value=_MISSING)
                    _style_data_cell(ws_sum.cell(row=curr_row, column=2), align="right")
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
            cell = ws_obj.cell(row=1, column=c_idx, value=sanitize_excel_value(str(col_name)))
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
                    if np.isfinite(val):
                        cell.value = float(val)
                        _style_data_cell(cell, num_format="0.000", align="right")
                    else:
                        # ±inf не является допустимым числом в .xlsx — пишем текстом
                        cell.value = str(float(val))
                        _style_data_cell(cell, align="right")
                else:
                    # Защита от формульных инъекций (=, +, -, @)
                    cell.value = sanitize_excel_value(str(val))
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
                desc = sanitize_excel_value(str(getattr(row, "Descriptor")))
                vip_v = _num(getattr(row, "VIP"))
                block = sanitize_excel_value(str(getattr(row, "Block", "Other")))
                is_sig = bool(np.isfinite(vip_v) and vip_v >= 1.0)

                bg_col = "E6F4EA" if is_sig else None  # Светло-зеленый для значимых маркеров

                c1 = ws_vip.cell(row=r_idx, column=1, value=desc)
                _style_data_cell(c1, bold=is_sig, bg_color=bg_col)

                c2 = ws_vip.cell(row=r_idx, column=2, value=vip_v if np.isfinite(vip_v) else _MISSING)
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
    permutation_results: Optional[Dict] = None,
) -> str:
    """
    Генерирует автономный интерактивный HTML-паспорт образцов.
    Включает сводные метрики 8 аналитических методов, результаты хемометрической классификации,
    панель ключевых биомаркеров, межмодальные кросс-корреляции и автоматическую сводку по данным
    (метрики валидации модели, предупреждение о малой выборке, VIP-маркеры с направлением различий,
    вклад блоков) с поддержкой печати в PDF. Заготовленные «экспертные» выводы о генезисе не формируются.

    permutation_results: результат chemo_ml.permutation_test_plsda (ключ p_val_q2), необязательно.
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
    active_block_names: List[str] = []
    block_cards_html = ""
    for b_key, b_title_en, b_title_ru, b_color in all_blocks:
        b_cols = [c for c in feature_cols if block_map.get(c) == b_key]
        b_title = b_title_en if is_en else b_title_ru
        if len(b_cols) > 0:
            active_blocks_count += 1
            active_block_names.append(b_key)
            sample_stats_html = ""
            for col in b_cols[:4]:
                col_vals = pd.to_numeric(fused_df[col], errors="coerce")
                val_mean = float(col_vals.mean())
                val_std = float(col_vals.std()) if col_vals.notna().sum() > 1 else float("nan")
                c_clean = html.escape(str(col))
                sample_stats_html += f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed #e2e8f0; font-size: 12px;">
                    <span style="color: #475569; font-family: monospace;">{c_clean}:</span>
                    <span style="font-weight: 600; color: #0f172a;">{_fmt_num(val_mean)} ± {_fmt_num(val_std)}</span>
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

    # Автоматическая сводка по данным (вместо заготовленного «экспертного» заключения):
    # статус определяется только метриками валидации модели, а не названиями классов.
    classes_present = list(fused_df["Class_Target"].dropna().unique()) if "Class_Target" in fused_df.columns else []
    classes_str = ", ".join(str(c) for c in classes_present) if classes_present else ("N/A")

    summary = build_data_driven_summary(
        fused_df, model_results=model_results, permutation_results=permutation_results,
        active_blocks=active_block_names, lang=lang,
    )
    validation = summary["validation"]
    risk_label = html.escape(validation["label"])
    risk_bg = validation["bg"]
    risk_color = validation["color"]
    validation_status_html = f'<strong style="color: {risk_color};">{risk_label}</strong>'
    conclusion_text = summary["html"]

    # Модель ML
    model_section_html = ""
    if model_results:
        m_type = html.escape(str(model_results.get("model_type", "PLS-DA")))
        cv_scheme = model_results.get("CV_Scheme")
        acc_cv, acc_cv_key = _first_metric(model_results, ("Balanced_Accuracy_CV", "Accuracy_CV"))
        acc_tr, acc_tr_key = _first_metric(model_results, ("Balanced_Accuracy", "Accuracy"))
        if acc_cv_key is not None:
            acc = acc_cv
            acc_label = "Balanced Accuracy (CV)" if acc_cv_key == "Balanced_Accuracy_CV" else "Accuracy (CV)"
        else:
            acc = acc_tr
            acc_label = "Balanced Accuracy (train)" if acc_tr_key != "Accuracy" else "Accuracy (train)"
        r2x = model_results.get("R2X")
        r2y = model_results.get("R2Y")
        q2 = _num(model_results.get("Q2"))
        q2_label = f"Q² ({html.escape(str(cv_scheme))})" if cv_scheme else "Q² (CV)"

        vip_rows_html = ""
        vip_df = model_results.get("VIP_df", pd.DataFrame())
        if isinstance(vip_df, pd.DataFrame) and not vip_df.empty:
            for _, r in vip_df.head(8).iterrows():
                desc_name = html.escape(str(r["Descriptor"]))
                vip_val = _num(r["VIP"])
                blk_name = html.escape(str(r.get("Block", "Other")))
                sig_badge = f'<span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;">VIP={vip_val:.2f} ★</span>' if (np.isfinite(vip_val) and vip_val >= 1.0) else f'<span style="background: #f1f5f9; color: #64748b; padding: 2px 8px; border-radius: 10px; font-size: 11px;">VIP={_fmt_num(vip_val, 2)}</span>'
                vip_rows_html += f"""
                <tr>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-weight: 600;">{desc_name}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; color: #0b5394;">{blk_name}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: 700;">{_fmt_num(vip_val)}</td>
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
                    <div class="kpi-value">{_fmt_pct(r2x)}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">R²Y (Class Variance)</div>
                    <div class="kpi-value">{_fmt_pct(r2y)}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">{q2_label}</div>
                    <div class="kpi-value" style="color: {'#16a34a' if (np.isfinite(q2) and q2 > Q2_GOOD_PCT) else '#d97706'};">{_fmt_pct(q2)}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">{acc_label}</div>
                    <div class="kpi-value" style="color: #0b5394;">{_fmt_pct(acc)}</div>
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
            r_val = _num(r["Correlation"])
            # Предпочтительно q-value (поправка Бенджамини–Хохберга на множественные сравнения),
            # иначе p-value; отсутствующее значение -> «—» (не 0.0)
            if "q_value" in r.index and np.isfinite(_num(r.get("q_value"))):
                sig_name, sig_val = "q", _num(r.get("q_value"))
            else:
                sig_name, sig_val = "p", _num(r.get("p_value"))
            sig_txt = f"{sig_name} = {sig_val:.2e}" if np.isfinite(sig_val) else f"{sig_name} = {_MISSING}"
            r_txt = f"{r_val:+.3f}" if np.isfinite(r_val) else _MISSING
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
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: 700; color: {r_col};">{r_txt}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-size: 11px; color: #64748b;">{sig_txt}</td>
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
                        <th style="text-align: right;">q-value (BH-FDR) / p-value</th>
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
        <td style="font-weight: 600; color: #475569;">{"Статус валидации модели:" if not is_en else "Model Validation Status:"}</td>
        <td>{validation_status_html}</td>
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

  <!-- Автоматическая сводка по данным (не экспертное заключение) -->
  <div class="card">
    <div class="card-header">
      <h3 class="card-title">📝 {"Автоматическая сводка результатов" if not is_en else "Automated Data Summary"}</h3>
      <span style="font-size: 12px; color: #64748b;">{"Рассчитано по данным сессии" if not is_en else "Computed from session data"}</span>
    </div>
    <div class="conclusion-box">
      <strong>{"СВОДКА ПО ДАННЫМ:" if not is_en else "DATA SUMMARY:"}</strong> {conclusion_text}
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

