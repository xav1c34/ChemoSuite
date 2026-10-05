# app.py — ChemoSuite Unified Analytical Platform
import inspect
import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Попытка импорта специализированных библиотек
try:
    import nomspectra as ns
    from nomspectra import Spectrum
    NOMSPECTRA_INSTALLED = True
except ImportError:
    NOMSPECTRA_INSTALLED = False

try:
    import eem_core
    EEM_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import eem_core
        EEM_CORE_AVAILABLE = True
    except ImportError:
        EEM_CORE_AVAILABLE = False

# ==============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ И СТИЛИЗАЦИЯ
# ==============================================================================
st.set_page_config(
    page_title="ChemoSuite Platform | FT-ICR MS & EEM-PARAFAC",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .platform-header {
        background: linear-gradient(90deg, #0b2545 0%, #134074 100%);
        padding: 16px 22px;
        border-radius: 8px;
        color: white;
        margin-bottom: 18px;
    }
    .platform-header h2 {
        color: #eef4f8;
        font-size: 24px;
        margin: 0;
        padding-bottom: 4px;
    }
    .platform-header p {
        color: #8da9c4;
        margin: 0;
        font-size: 13px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2e7bcf;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STORAGE_DIR = "user_spectra"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ==============================================================================
# СЛОВАРЬ ЛОКАЛИЗАЦИИ FT-ICR MS (I18N)
# ==============================================================================
T = {
    "ru": {
        "title": "🔬 Спектрометрия NOM сверхвысокого разрешения (FT-ICR MS)",
        "lang_label": "🌐 Язык / Language",
        "sidebar_mgr": "📁 Менеджер масс-спектров",
        "nomspectra_missing": "Библиотека nomspectra не найдена в окружении. Используется встроенное векторное ядро.",
        "uploader_label": "Загрузить спектры (файлы или перетащите папку):",
        "folder_expander": "📂 Импорт из локальной папки на диске",
        "folder_input": "Путь к папке со спектрами:",
        "folder_help": "Укажите директорию для сканирования спектральных файлов",
        "folder_btn": "📥 Сканировать и загрузить все спектры",
        "folder_success": "Успешно импортировано спектров: {n}",
        "folder_err": "Указанная папка не найдена или путь пуст.",
        "search_label": "🔍 Поиск спектра:",
        "search_ph": "Введите часть названия...",
        "found_n": "Найдено: {found} из {total}",
        "search_empty": "По запросу ничего не найдено.",
        "active_sample": "Активный образец:",
        "del_sample": "🗑️ Удалить выбранный образец",
        "parse_expander": "Параметры чтения и парсинга",
        "delimiter": "Разделитель (Delimiter)",
        "decimal_sep": "Десятичный знак",
        "has_header": "Файл содержит заголовок",
        "preview_caption": "Предпросмотр данных (первые 5 строк):",
        "col_mz": "Колонка m/z (масса):",
        "col_int": "Колонка Intensity (интенсивность):",
        "same_col_warn": "Внимание: выбрана одна и та же колонка для массы и интенсивности!",
        "filtering_header": "Фильтрация и шкалирование пиков",
        "norm_mode_label": "Режим нормализации интенсивности:",
        "norm_base": "Базовый пик (Max = 100%)",
        "norm_tic": "Сумма / TIC (Сумма = 100%)",
        "norm_raw": "Исходная интенсивность (Raw)",
        "mz_range": "Диапазон m/z (Да):",
        "cutoff_int": "Порог отсечения шума (мин. интенсивность):",
        "loaded_peaks_success": "Загружено и отфильтровано пиков: {n:,}",
        "no_spectra_info": "Нет загруженных спектров. Перетащите файлы выше или импортируйте папку.",
        "tab_stick": "📈 Масс-спектр (Stick Plot)",
        "tab_recal": "🎯 Рекалибровка m/z",
        "tab_assign": "🧬 Приписывание формул",
        "tab_vk": "🗺️ Диаграммы и проекции",
        "tab_kmd": "🔍 Анализ Кендрика (KMD)",
        "tab_vk20": "🗂️ Хемотипирование 20 ячеек",
        "tab_cmp": "⚖️ Сравнение и алгебра спектров",
        "tab_tmds": "🔗 Сети трансформаций (TMDS)",
        "tab_desc": "📊 Сводные характеристики",
        "download_png": "💾 Скачать PNG (300 DPI)",
        "download_svg": "💾 Скачать SVG (Вектор)",
        "nav_caption": "🔍 Навигация: колесико мыши — масштаб (Zoom), зажатая левая кнопка — рамка зума / сдвиг (Pan). Двойной клик — сброс.",
        "stick_signals": "Отображено сигналов: {n:,}",
        "annotate_top": "Подписать топ-5 пиков",
        "peak_trace": "Пики спектра",
        "top5_trace": "Топ-5 пиков",
        "mz_axis": "m/z (Дальтон)",
        "rel_int_axis": "Интенсивность",
        "recal_cal_set": "Набор калибрантов (внутренний стандарт):",
        "recal_tol": "Окно поиска реперов (ppm):",
        "recal_poly": "Порядок полинома коррекции:",
        "recal_ion": "Режим ионизации калибрантов:",
        "recal_few_peaks": "Найдено реперных пиков: {n} (необходимо минимум 3 для построения полинома). Попробуйте расширить окно поиска ppm.",
        "recal_found": "Найдено калибровочных реперов в спектре: **{n}**",
        "recal_runge_warn": "⚠️ Диапазон обнаруженных калибрантов ({cmin:.1f}–{cmax:.1f} Да) покрывает **{cov:.1f}%** диапазона масс спектра (< 60%). Степень полинома принудительно снижена до 1 (линейная регрессия) для предотвращения эффекта Рунге и краевого улета погрешности на массах > 500 Да.",
        "recal_btn": "🚀 Применить рекалиброванные массы для приписывания формул",
        "recal_success": "Шкала m/z успешно рекалибрована! Скорректировано пиков: {n:,}. Ошибка снижена с {before:.2f} ppm до {after:.2f} ppm. Перейдите во вкладку «Приписывание формул».",
        "recal_scatter_before": "Реперы (до коррекции)",
        "recal_scatter_after": "Реперы (после коррекции)",
        "recal_title_before": "Систематический дрейф ppm до рекалибровки",
        "recal_title_after": "Погрешность после полиномиальной рекалибровки",
        "ppm_axis": "Погрешность массы (ppm)",
        "res_ppm_axis": "Остаточная погрешность (ppm)",
        "assign_expander": "⚙️ Настройки химического пространства и фильтрации",
        "ion_mode": "Режим ионизации:",
        "max_charge": "Максимальный заряд (z):",
        "ppm_tol": "Допуск погрешности (ppm):",
        "c_lim": "Лимит углерода (C):",
        "h_lim": "Лимит водорода (H):",
        "o_lim": "Лимит кислорода (O):",
        "n_lim": "Лимит азота (N):",
        "s_lim": "Лимит серы (S):",
        "max_oc": "Максимум O/C:",
        "max_hc": "Максимум H/C:",
        "iso_filter_label": "Валидация по изотопу 13C (+1.00335 Да)",
        "iso_strict_label": "Отсекать формулы без подтверждения 13C (строгий фильтр)",
        "assign_btn": "🚀 Запустить приписывание формул",
        "assign_spinner": "Выполняется идентификация брутто-формул и расчёт молекулярных индексов...",
        "metric_total": "Всего пиков",
        "metric_assigned": "Приписано формул",
        "metric_rate": "Эффективность (Rate)",
        "metric_err": "Средняя |ppm| ошибка",
        "assign_tbl_title": "Таблица идентифицированных формул:",
        "dl_csv_formulas": "📥 Скачать результат идентификации (CSV)",
        "no_formulas_warn": "В заданных границах элементов и ppm-погрешности формул не найдено.",
        "need_assign_first": "⚠️ Сначала выполните приписывание формул во вкладке 3.",
        "proj_mode": "Тип проекции диаграммы:",
        "proj_vk": "Диаграмма Ван-Кревелена (H/C vs O/C)",
        "proj_dbe_c": "Конденсированность DBE vs C (BE vs n)",
        "proj_custom": "Пользовательская проекция (X vs Y)",
        "pow_exp": "Сжатие шкалы интенсивности (Pow γ):",
        "hetero_dist_title": "Распределение по классам гетероатомов",
        "comp_dist_title": "Распределение по структурным пулам",
        "class_col": "Класс",
        "pool_col": "Класс/Пул",
        "count_col": "Количество",
        "share_col": "Доля (%)",
        "oc_axis": "O/C (Кислород / Углерод)",
        "hc_axis": "H/C (Водород / Углерод)",
        "kmd_base_group": "Базовая функциональная группа KMD:",
        "kmd_color_mode": "Цветовая дифференциация серий:",
        "kmd_pt_size": "Размер точек:",
        "kmd_even": "Четный NKM",
        "kmd_odd": "Нечетный NKM",
        "kmd_parity_mode": "Четность NKM (Радикалы / Азот)",
        "kmd_int_mode": "Интенсивность",
        "kmd_o_mode": "Число атомов кислорода (O)",
        "kmd_dbe_mode": "Индекс ненасыщенности (DBE)",
        "kmd_hetero_mode": "Классы гетероатомов",
        "kmd_x_axis": "Номинальная масса Кендрика NKM ({base})",
        "kmd_y_axis": "Дефект массы Кендрика KMD [0, 1) ({base})",
        "vk20_basis": "Базис расчета заселенности:",
        "vk20_opt_count": "Относительное число формул (%)",
        "vk20_opt_weight": "Взвешенная интенсивность (%)",
        "vk20_overlay": "Наложить границы 20 ячеек на диаграмму Ван-Кревелена",
        "vk20_heatmap_title": "Тепловая карта заселенности 20 ячеек ({mode})",
        "vk20_dl_csv": "📥 Скачать вектор дескрипторов 20 ячеек (CSV)",
        "cmp_subtab_view": "Сравнение образцов (A vs B)",
        "cmp_subtab_sub": "Вычитание бланка / фона (int_sub)",
        "cmp_subtab_algebra": "Алгебра спектров и диаграмма Венна",
        "cmp_need_two": "Для сравнительного анализа необходимо загрузить как минимум два спектра в боковой панели.",
        "cmp_spec_a": "Образец А (Базовый):",
        "cmp_spec_b": "Образец Б (Сравниваемый / Бланк):",
        "cmp_tol": "Допуск совмещения (ppm):",
        "cmp_common": "Общих (A ∩ B)",
        "cmp_jaccard": "Индекс Жаккара: {val:.3f}",
        "cmp_cosine": "Косинусное сходство",
        "cmp_common_label": "Общие A ∩ B ({n:,})",
        "cmp_unique_a": "Уникальные для {name} ({n:,})",
        "cmp_unique_b": "Уникальные для {name} ({n:,})",
        "cmp_mirror_title": "Зеркальный спектр совмещения (Head-to-Tail Stick Plot)",
        "sub_factor_label": "Коэффициент вычитания фона (k):",
        "sub_btn": "🧹 Вычесть бланк и сохранить очищенный спектр",
        "sub_success": "Бланк успешно вычтен! Создан спектр: {name} (пиков: {n:,}).",
        "alg_op_label": "Множественная операция:",
        "alg_and": "Пересечение A ∩ B (Общие пики)",
        "alg_or": "Объединение A ∪ B (Все пики)",
        "alg_sub_a_b": "Разность A \\ B (Уникальные для A)",
        "alg_sub_b_a": "Разность B \\ A (Уникальные для B)",
        "alg_xor": "Симметрическая разность A ⊕ B (Не пересекающиеся)",
        "alg_btn": "💾 Применить операцию и сохранить спектр в базу",
        "alg_success": "Операция выполнена! Новый спектр сохранен: {name} ({n:,} пиков).",
        "venn_title": "Диаграмма Венна перекрытия спектров (ppm допуск: {tol})",
        "tmds_header": "Скрининг характеристических разностей масс (TMDS)",
        "tmds_top_label": "Количество наиболее интенсивных пиков для анализа:",
        "tmds_tol_label": "Допуск разности масс (mDa):",
        "tmds_trans_title": "Частота обнаружения биогеохимических трансформаций",
        "tmds_trans_col": "Трансформация",
        "tmds_delta_col": "Δm (Да)",
        "tmds_count_col": "Число связей",
        "tmds_share_col": "Доля от всех пар (%)",
        "tmds_dl_csv": "📥 Скачать пары связей TMDS (CSV)",
        "desc_num_avg": "Среднечисленные значения (Number-averaged parameters)",
        "desc_mn": "Среднечисленная масса (Mn)",
        "desc_mw": "Mw (Взвешенная масса)",
        "desc_dbe": "Среднечисленный DBE",
        "desc_dbe_o": "Среднечисленный DBE - O",
        "desc_ai": "Среднечисленный AI (Кох)",
        "desc_weighted_header": "Сравнение со средневзвешенными значениями по интенсивности (Mw-weighted)",
        "desc_stats_header": "Полная сводная статистика дескрипторов",
        "stat_mean": "Среднее",
        "stat_std": "Стд. откл.",
        "stat_min": "Мин.",
        "stat_med": "Медиана",
        "stat_max": "Макс.",
    },
    "en": {
        "title": "🔬 Ultra-High Resolution Mass Spectrometry (FT-ICR MS)",
        "lang_label": "🌐 Language / Язык",
        "sidebar_mgr": "📁 Spectrum Manager",
        "nomspectra_missing": "nomspectra library not found in environment. Built-in high-performance engine is active.",
        "uploader_label": "Upload spectra (files or drag & drop folder):",
        "folder_expander": "📂 Import from local disk directory",
        "folder_input": "Path to spectra directory:",
        "folder_help": "Specify directory path to recursively scan for spectral files",
        "folder_btn": "📥 Scan and import all spectra",
        "folder_success": "Successfully imported spectra: {n}",
        "folder_err": "Specified directory not found or path is empty.",
        "search_label": "🔍 Search spectrum:",
        "search_ph": "Type spectrum name...",
        "found_n": "Found: {found} of {total}",
        "search_empty": "No matching spectra found.",
        "active_sample": "Active sample:",
        "del_sample": "🗑️ Delete selected sample",
        "parse_expander": "Parsing & format parameters",
        "delimiter": "Delimiter",
        "decimal_sep": "Decimal separator",
        "has_header": "File contains column header",
        "preview_caption": "Data preview (first 5 rows):",
        "col_mz": "m/z column (mass):",
        "col_int": "Intensity column:",
        "same_col_warn": "Warning: The same column is selected for both mass and intensity!",
        "filtering_header": "Peak filtering & scaling",
        "norm_mode_label": "Intensity normalization mode:",
        "norm_base": "Base Peak (Max = 100%)",
        "norm_tic": "Sum / TIC (Sum = 100%)",
        "norm_raw": "Raw Intensity",
        "mz_range": "m/z range (Da):",
        "cutoff_int": "Noise cutoff (min intensity):",
        "loaded_peaks_success": "Peaks loaded & filtered: {n:,}",
        "no_spectra_info": "No spectra loaded. Drag & drop files above or import a local folder.",
        "tab_stick": "📈 Mass Spectrum (Stick Plot)",
        "tab_recal": "🎯 m/z Recalibration",
        "tab_assign": "🧬 Formula Assignment",
        "tab_vk": "🗺️ Plots & Projections",
        "tab_kmd": "🔍 Kendrick Analysis (KMD)",
        "tab_vk20": "🗂️ 20-Grid Chemotyping",
        "tab_cmp": "⚖️ Sample Comparison & Algebra",
        "tab_tmds": "🔗 Reaction Networks (TMDS)",
        "tab_desc": "📊 Summary Descriptors",
        "download_png": "💾 Download PNG (300 DPI)",
        "download_svg": "💾 Download SVG (Vector)",
        "nav_caption": "🔍 Navigation: Mouse wheel — Zoom, Click & drag — Box Zoom / Pan. Double click — Reset view.",
        "stick_signals": "Signals displayed: {n:,}",
        "annotate_top": "Annotate top-5 peaks",
        "peak_trace": "Spectral peaks",
        "top5_trace": "Top-5 peaks",
        "mz_axis": "m/z (Dalton)",
        "rel_int_axis": "Intensity",
        "recal_cal_set": "Calibrant reference library:",
        "recal_tol": "Search window (ppm):",
        "recal_poly": "Polynomial regression degree:",
        "recal_ion": "Calibrant ionization mode:",
        "recal_few_peaks": "Calibrants found: {n} (at least 3 required for curve fitting). Try increasing the ppm window.",
        "recal_found": "Internal calibrants matched in spectrum: **{n}**",
        "recal_runge_warn": "⚠️ Calibrant span ({cmin:.1f}–{cmax:.1f} Da) covers **{cov:.1f}%** of spectrum span (< 60%). Polynomial degree forced to 1 (linear) to prevent Runge's phenomenon at m/z > 500 Da.",
        "recal_btn": "🚀 Apply recalibrated m/z to spectrum",
        "recal_success": "m/z scale successfully recalibrated! Corrected peaks: {n:,}. Mean error reduced from {before:.2f} ppm to {after:.2f} ppm. Proceed to 'Formula Assignment'.",
        "recal_scatter_before": "Calibrants (before correction)",
        "recal_scatter_after": "Calibrants (after correction)",
        "recal_title_before": "Systematic ppm drift before recalibration",
        "recal_title_after": "Residual error after polynomial recalibration",
        "ppm_axis": "Mass error (ppm)",
        "res_ppm_axis": "Residual error (ppm)",
        "assign_expander": "⚙️ Chemical search space & stoichiometric boundaries",
        "ion_mode": "Ionization mode:",
        "max_charge": "Maximum Charge (z):",
        "ppm_tol": "Tolerance window (ppm):",
        "c_lim": "Carbon limit (C):",
        "h_lim": "Hydrogen limit (H):",
        "o_lim": "Oxygen limit (O):",
        "n_lim": "Nitrogen limit (N):",
        "s_lim": "Sulfur limit (S):",
        "max_oc": "Maximum O/C:",
        "max_hc": "Maximum H/C:",
        "iso_filter_label": "Validate with 13C isotope (+1.00335 Da)",
        "iso_strict_label": "Discard formulas without 13C confirmation (strict filter)",
        "assign_btn": "🚀 Run Formula Assignment",
        "assign_spinner": "Assigning elemental formulas and calculating molecular indices...",
        "metric_total": "Total peaks",
        "metric_assigned": "Formulas assigned",
        "metric_rate": "Assignment rate",
        "metric_err": "Mean |ppm| error",
        "assign_tbl_title": "Identified elemental formulas table:",
        "dl_csv_formulas": "📥 Download identification results (CSV)",
        "no_formulas_warn": "No formulas found within the given elemental boundaries and ppm tolerance.",
        "need_assign_first": "⚠️ Please run formula assignment in Tab 3 first.",
        "proj_mode": "Projection diagram type:",
        "proj_vk": "Van Krevelen Diagram (H/C vs O/C)",
        "proj_dbe_c": "Aromaticity DBE vs C (BE vs n)",
        "proj_custom": "Custom 2D Projection (X vs Y)",
        "pow_exp": "Intensity scale compression (Pow γ):",
        "hetero_dist_title": "Distribution by heteroatom classes",
        "comp_dist_title": "Distribution by biochemical structural pools",
        "class_col": "Class",
        "pool_col": "Pool/Class",
        "count_col": "Count",
        "share_col": "Share (%)",
        "oc_axis": "O/C (Oxygen / Carbon)",
        "hc_axis": "H/C (Hydrogen / Carbon)",
        "kmd_base_group": "KMD functional base group:",
        "kmd_color_mode": "Series color mapping:",
        "kmd_pt_size": "Marker size:",
        "kmd_even": "Even NKM",
        "kmd_odd": "Odd NKM",
        "kmd_parity_mode": "NKM parity (Radicals / Nitrogen rule)",
        "kmd_int_mode": "Intensity",
        "kmd_o_mode": "Oxygen atoms count (O)",
        "kmd_dbe_mode": "Double Bond Equivalent (DBE)",
        "kmd_hetero_mode": "Heteroatom classes",
        "kmd_x_axis": "Nominal Kendrick Mass NKM ({base})",
        "kmd_y_axis": "Kendrick Mass Defect KMD [0, 1) ({base})",
        "vk20_basis": "Density metric basis:",
        "vk20_opt_count": "Relative formula count (%)",
        "vk20_opt_weight": "Intensity-weighted abundance (%)",
        "vk20_overlay": "Overlay 20-grid boundaries onto Van Krevelen plot",
        "vk20_heatmap_title": "Perminova 20-Grid Density Heatmap ({mode})",
        "vk20_dl_csv": "📥 Download 20-Grid feature vector (CSV)",
        "cmp_subtab_view": "Sample Comparison (A vs B)",
        "cmp_subtab_sub": "Blank Subtraction (int_sub)",
        "cmp_subtab_algebra": "Spectral Algebra & Venn Diagram",
        "cmp_need_two": "At least two spectra must be loaded in the sidebar for comparative analysis.",
        "cmp_spec_a": "Sample A (Base):",
        "cmp_spec_b": "Sample B (Comparison / Blank):",
        "cmp_tol": "Alignment tolerance (ppm):",
        "cmp_common": "Common (A ∩ B)",
        "cmp_jaccard": "Jaccard Index: {val:.3f}",
        "cmp_cosine": "Cosine Similarity",
        "cmp_common_label": "Common A ∩ B ({n:,})",
        "cmp_unique_a": "Unique to {name} ({n:,})",
        "cmp_unique_b": "Unique to {name} ({n:,})",
        "cmp_mirror_title": "Head-to-Tail Mirror Spectrum Comparison",
        "sub_factor_label": "Blank scaling factor (k):",
        "sub_btn": "🧹 Subtract blank and save clean spectrum",
        "sub_success": "Blank subtracted successfully! Created spectrum: {name} (peaks: {n:,}).",
        "alg_op_label": "Set operation:",
        "alg_and": "Intersection A ∩ B (Common peaks)",
        "alg_or": "Union A ∪ B (All peaks merged)",
        "alg_sub_a_b": "Difference A \\ B (Unique to A)",
        "alg_sub_b_a": "Difference B \\ A (Unique to B)",
        "alg_xor": "Symmetric Difference A ⊕ B (Exclusive peaks)",
        "alg_btn": "💾 Apply operation and save spectrum",
        "alg_success": "Operation successful! New spectrum saved: {name} ({n:,} peaks).",
        "venn_title": "Venn Diagram of Spectral Overlap (ppm tol: {tol})",
        "tmds_header": "Targeted Mass Difference Screening (TMDS)",
        "tmds_top_label": "Top abundant peaks to analyze:",
        "tmds_tol_label": "Mass difference tolerance (mDa):",
        "tmds_trans_title": "Biogeochemical Transformations Frequency",
        "tmds_trans_col": "Transformation",
        "tmds_delta_col": "Δm (Da)",
        "tmds_count_col": "Connections count",
        "tmds_share_col": "Share of all pairs (%)",
        "tmds_dl_csv": "📥 Download TMDS connected pairs (CSV)",
        "desc_num_avg": "Number-averaged parameters (Mn)",
        "desc_mn": "Number-averaged mass (Mn)",
        "desc_mw": "Mw (Weight-averaged mass)",
        "desc_dbe": "Number-averaged DBE",
        "desc_dbe_o": "Number-averaged DBE - O",
        "desc_ai": "Number-averaged AI (Koch)",
        "desc_weighted_header": "Comparison with Intensity-Weighted Descriptors (Mw-weighted)",
        "desc_stats_header": "Complete Descriptive Statistics",
        "stat_mean": "Mean",
        "stat_std": "Std. Dev.",
        "stat_min": "Min",
        "stat_med": "Median",
        "stat_max": "Max",
    },
}

# ==============================================================================
# КОНСТАНТЫ И БАЗОВЫЕ СПРАВОЧНИКИ
# ==============================================================================
EXACT_MASSES = {
    "C": 12.000000,
    "H": 1.007825,
    "O": 15.994915,
    "N": 14.003074,
    "S": 31.972071,
}
C13_DIFF = 1.003355
H_ION_MASS = 1.007276

KMD_BASES = {
    "CH2": {"nom": 14.00000, "exact": 14.015650, "label": "CH2"},
    "COO": {"nom": 44.00000, "exact": 43.989829, "label": "COO"},
    "O": {"nom": 16.00000, "exact": 15.994915, "label": "O"},
    "H2": {"nom": 2.00000, "exact": 2.015650, "label": "H2"},
}

TMDS_LIBRARY = [
    {"name": "CH2 (Alkylation / Homology)", "delta": 14.015650},
    {"name": "O (Oxidation / Hydroxylation)", "delta": 15.994915},
    {"name": "H2O (Hydration / Dehydration)", "delta": 18.010565},
    {"name": "H2 (Hydrogenation / Dehydrogenation)", "delta": 2.015650},
    {"name": "CO2 (Carboxylation / Decarboxylation)", "delta": 43.989829},
    {"name": "CO (Carbonylation)", "delta": 27.994915},
    {"name": "NH3 (Amination / Deamination)", "delta": 17.026549},
    {"name": "SO3 (Sulfonation)", "delta": 79.956815},
]

# ==============================================================================
# ХЕЛПЕРЫ И ВЫЧИСЛИТЕЛЬНЫЕ ФУНКЦИИ FT-ICR MS
# ==============================================================================
def st_df(data, **kwargs):
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)


def st_plotly(fig, **kwargs):
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    try:
        return st.plotly_chart(fig, width="stretch", config=config, **kwargs)
    except TypeError:
        return st.plotly_chart(
            fig, use_container_width=True, config=config, **kwargs
        )


def get_plot_download_buttons(fig, base_filename: str, lang: str):
    col1, col2, _ = st.columns([1.8, 1.8, 4.4])
    with col1:
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label=T[lang]["download_png"],
            data=png_buf.getvalue(),
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"dl_png_{base_filename}",
        )
    with col2:
        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        st.download_button(
            label=T[lang]["download_svg"],
            data=svg_buf.getvalue(),
            file_name=f"{base_filename}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{base_filename}",
        )


def calculate_descriptors(df: pd.DataFrame, lang: str = "ru") -> pd.DataFrame:
    res = df.copy()
    c = res["C"].astype(float)
    h = res["H"].astype(float)
    o = res["O"].astype(float)
    n = res["N"].astype(float) if "N" in res.columns else 0.0
    s = res["S"].astype(float) if "S" in res.columns else 0.0

    res["H/C"] = np.where(c > 0, h / c, np.nan)
    res["O/C"] = np.where(c > 0, o / c, np.nan)
    res["DBE"] = 1.0 + c - 0.5 * h + 0.5 * n
    res["DBE-O"] = res["DBE"] - o

    num_ai = 1.0 + c - o - s - 0.5 * h
    den_ai = c - o - s - n
    ai = np.where((den_ai > 0) & (num_ai > 0), num_ai / den_ai, 0.0)
    res["AI"] = np.clip(ai, 0.0, 1.0)
    res["NOSC"] = np.where(
        c > 0, 4.0 - (4.0 * c + h - 3.0 * n - 2.0 * o - 2.0 * s) / c, np.nan
    )

    def get_hetero_class(row):
        has_n = row["N"] > 0
        has_s = row["S"] > 0
        if has_n and has_s:
            return "CHONS"
        if has_n:
            return "CHON"
        if has_s:
            return "CHOS"
        return "CHO"

    res["Hetero_Class"] = res.apply(get_hetero_class, axis=1)

    def get_compound_class(row):
        hc = row["H/C"]
        oc = row["O/C"]
        ai_val = row["AI"]
        n_val = row["N"]
        if pd.isna(hc) or pd.isna(oc):
            return "Undefined" if lang == "en" else "Не определено"
        if n_val > 0 and (0.3 < hc < 0.8) and (oc < 0.4):
            return (
                "CHON pool (0.3 < H/C < 0.8, O/C < 0.4)"
                if lang == "en"
                else "Пул CHON (0.3 < H/C < 0.8, O/C < 0.4)"
            )
        if ai_val >= 0.5 or (hc < 0.7 and oc <= 0.67):
            return (
                "Condensed tannins / Aromatics"
                if lang == "en"
                else "Конденсированные таннины / Ароматика"
            )
        if 0.7 <= hc < 1.5 and 0.1 <= oc <= 0.67:
            return (
                "Lignin-like / CRAM" if lang == "en" else "Лигнины / CRAM"
            )
        if 0.5 <= hc < 1.5 and 0.67 < oc <= 1.0:
            return (
                "Hydrolyzable tannins"
                if lang == "en"
                else "Гидролизуемые таннины"
            )
        if 1.5 <= hc <= 2.0 and oc <= 0.3:
            return "Lipids" if lang == "en" else "Липиды"
        if 1.5 <= hc <= 2.0 and 0.3 < oc <= 0.67:
            return (
                "Peptides / Proteins / Aliphatics"
                if lang == "en"
                else "Пептиды / Белки / Алифатика"
            )
        if 1.5 <= hc <= 2.0 and 0.67 < oc <= 1.0:
            return "Carbohydrates" if lang == "en" else "Углеводы"
        return (
            "Other / Unclassified" if lang == "en" else "Прочие компоненты"
        )

    res["Compound_Class"] = res.apply(get_compound_class, axis=1)
    return res


def parse_uploaded_file(
    file_bytes: bytes, delimiter: str, decimal_sep: str, has_header: bool
) -> pd.DataFrame:
    sep_map = {
        "Auto": None,
        "Авто (автоопределение)": None,
        "Comma (,)": ",",
        "Запятая (,)": ",",
        "Semicolon (;)": ";",
        "Точка с запятой (;)": ";",
        "Tab (\\t)": "\t",
        "Табуляция (\\t)": "\t",
        "Space": r"\s+",
        "Пробел": r"\s+",
    }
    actual_sep = sep_map.get(delimiter, ",")
    engine = "python" if (actual_sep is None or actual_sep == r"\s+") else "c"

    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(
            bio,
            sep=actual_sep,
            decimal=decimal_sep,
            header=0 if has_header else None,
            engine=engine,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
    except Exception:
        bio.seek(0)
        df = pd.read_csv(
            bio,
            sep=r"\s+",
            decimal=decimal_sep,
            header=0 if has_header else None,
            engine="python",
            on_bad_lines="skip",
        )

    if not has_header:
        df.columns = [f"Col_{i+1}" for i in range(len(df.columns))]

    return df


def fast_formula_assigner(
    peaks_df: pd.DataFrame,
    bounds: Dict[str, Tuple[int, int]],
    max_hc: float,
    max_oc: float,
    ppm_tolerance: float,
    ion_mode: str,
    max_charge: int = 1,
    iso_check: bool = False,
    iso_strict: bool = False,
) -> pd.DataFrame:
    c_min, c_max = bounds.get("C", (4, 120))
    h_min, h_max = bounds.get("H", (4, 200))
    o_min, o_max = bounds.get("O", (1, 60))
    n_min, n_max = bounds.get("N", (0, 2))
    s_min, s_max = bounds.get("S", (0, 1))

    peaks_m = peaks_df["mass"].values
    peaks_int = peaks_df["intensity"].values
    peaks_norm = (
        peaks_df["norm_intensity"].values
        if "norm_intensity" in peaks_df.columns
        else peaks_int
    )

    min_mz = float(peaks_m.min())
    max_mz = float(peaks_m.max())

    charges = [1, 2] if max_charge >= 2 else [1]
    all_assigned_rows = []

    for z in charges:
        if "ESI(-)" in ion_mode:
            ion_shift = -z * H_ION_MASS
        elif "ESI(+)" in ion_mode:
            ion_shift = z * H_ION_MASS
        else:
            ion_shift = 0.0

        c_list, h_list, o_list, n_list, s_list = [], [], [], [], []

        for n in range(n_min, n_max + 1):
            for s in range(s_min, s_max + 1):
                for c in range(c_min, c_max + 1):
                    cur_o_max = min(o_max, int(max_oc * c))
                    for o in range(o_min, cur_o_max + 1):
                        base_neut = (
                            c * EXACT_MASSES["C"]
                            + o * EXACT_MASSES["O"]
                            + n * EXACT_MASSES["N"]
                            + s * EXACT_MASSES["S"]
                        )
                        base_mz = (base_neut + ion_shift) / z
                        if base_mz > max_mz + 2.0:
                            continue

                        h_low = max(
                            h_min,
                            int(np.ceil(0.2 * c)),
                            int(np.ceil(2 * (c - o - 9) + n)),
                            int(
                                np.ceil(
                                    ((min_mz - 2.0) * z - base_neut - ion_shift)
                                    / EXACT_MASSES["H"]
                                )
                            ),
                        )
                        h_high = min(
                            h_max,
                            int(max_hc * c),
                            int(2 * c + n + 2),
                            int(2 * (11 + c - o) + n),
                            int(
                                np.floor(
                                    ((max_mz + 2.0) * z - base_neut - ion_shift)
                                    / EXACT_MASSES["H"]
                                )
                            ),
                        )

                        if h_low > h_high:
                            continue

                        for h in range(h_low, h_high + 1):
                            if (h + n) % 2 != 0:
                                continue
                            dbe = 1.0 + c - 0.5 * h + 0.5 * n
                            dbe_o = dbe - o
                            if dbe < 0 or dbe_o < -10 or dbe_o > 10:
                                continue

                            c_list.append(c)
                            h_list.append(h)
                            o_list.append(o)
                            n_list.append(n)
                            s_list.append(s)

        if not c_list:
            continue

        c_arr = np.array(c_list, dtype=np.int16)
        h_arr = np.array(h_list, dtype=np.int16)
        o_arr = np.array(o_list, dtype=np.int16)
        n_arr = np.array(n_list, dtype=np.int8)
        s_arr = np.array(s_list, dtype=np.int8)

        cand_masses = (
            c_arr * EXACT_MASSES["C"]
            + h_arr * EXACT_MASSES["H"]
            + o_arr * EXACT_MASSES["O"]
            + n_arr * EXACT_MASSES["N"]
            + s_arr * EXACT_MASSES["S"]
            + ion_shift
        ) / z

        sort_idx = np.argsort(cand_masses)
        cand_masses = cand_masses[sort_idx]
        c_arr = c_arr[sort_idx]
        h_arr = h_arr[sort_idx]
        o_arr = o_arr[sort_idx]
        n_arr = n_arr[sort_idx]
        s_arr = s_arr[sort_idx]

        delta = peaks_m * (ppm_tolerance * 1e-6)
        left_idx = np.searchsorted(cand_masses, peaks_m - delta)
        right_idx = np.searchsorted(cand_masses, peaks_m + delta)

        for i in range(len(peaks_m)):
            l, r = left_idx[i], right_idx[i]
            if r > l:
                best_j = None
                best_score = (1e9, 99)
                for j in range(l, r):
                    calc_m = cand_masses[j]
                    err_ppm = abs((peaks_m[i] - calc_m) / calc_m) * 1e6
                    hetero_penalty = n_arr[j] + s_arr[j]
                    score = (err_ppm, hetero_penalty)
                    if score < best_score:
                        best_score = score
                        best_j = j

                if best_j is not None:
                    calc_m = cand_masses[best_j]
                    err_ppm = ((peaks_m[i] - calc_m) / calc_m) * 1e6
                    c_val = int(c_arr[best_j])
                    h_val = int(h_arr[best_j])
                    o_val = int(o_arr[best_j])
                    n_val = int(n_arr[best_j])
                    s_val = int(s_arr[best_j])

                    iso_confirmed = False
                    if iso_check:
                        expected_c13_m = peaks_m[i] + (C13_DIFF / z)
                        iso_tol = expected_c13_m * (ppm_tolerance * 2.0 * 1e-6)
                        idx_c13 = np.where(
                            (peaks_m >= expected_c13_m - iso_tol)
                            & (peaks_m <= expected_c13_m + iso_tol)
                        )[0]
                        if len(idx_c13) > 0:
                            iso_confirmed = True

                    if (
                        iso_strict
                        and iso_check
                        and (not iso_confirmed)
                        and (c_val > 15)
                    ):
                        continue

                    all_assigned_rows.append(
                        {
                            "mass": peaks_m[i],
                            "intensity": peaks_int[i],
                            "norm_intensity": peaks_norm[i],
                            "calc_mass": calc_m,
                            "error_ppm": err_ppm,
                            "z": z,
                            "C": c_val,
                            "H": h_val,
                            "O": o_val,
                            "N": n_val,
                            "S": s_val,
                            "Formula": f"C{c_val}H{h_val}O{o_val}"
                            + (f"N{n_val}" if n_val > 0 else "")
                            + (f"S{s_val}" if s_val > 0 else ""),
                            "Iso_13C_Confirmed": iso_confirmed
                            if iso_check
                            else True,
                        }
                    )

    if not all_assigned_rows:
        return pd.DataFrame()

    res_df = pd.DataFrame(all_assigned_rows)
    res_df = (
        res_df.sort_values(by="error_ppm", key=abs)
        .drop_duplicates(subset=["mass"])
        .sort_values("mass")
        .reset_index(drop=True)
    )
    return res_df


def run_formula_assignment(
    peaks_df: pd.DataFrame,
    bounds: Dict[str, Tuple[int, int]],
    max_hc: float,
    max_oc: float,
    ppm_tolerance: float,
    ion_mode: str,
    max_charge: int = 1,
    iso_check: bool = False,
    iso_strict: bool = False,
    lang: str = "ru",
) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as tmp:
        tmp_path = tmp.name
        peaks_df.to_csv(tmp_path, sep="\t", index=False)

    assigned_df = pd.DataFrame()

    try:
        if NOMSPECTRA_INSTALLED and max_charge == 1 and not iso_check:
            try:
                spec = Spectrum(tmp_path)
                target_method = getattr(
                    spec, "assign_formulas", None
                ) or getattr(spec, "assign", None)
                if target_method is not None:
                    sig = inspect.signature(target_method).parameters
                    kwargs = {}
                    if "error" in sig:
                        kwargs["error"] = ppm_tolerance
                    elif "ppm" in sig:
                        kwargs["ppm"] = ppm_tolerance
                    elif "tolerance" in sig:
                        kwargs["tolerance"] = ppm_tolerance

                    for elem, (low, high) in bounds.items():
                        if elem in sig:
                            kwargs[elem] = (low, high)

                    if "elements" in sig:
                        kwargs["elements"] = bounds
                    if "hc_limits" in sig:
                        kwargs["hc_limits"] = (0.2, max_hc)
                    if "oc_limits" in sig:
                        kwargs["oc_limits"] = (0.0, max_oc)
                    if "mode" in sig:
                        kwargs["mode"] = (
                            "neg"
                            if "ESI(-)" in ion_mode
                            else ("pos" if "ESI(+)" in ion_mode else "neutral")
                        )

                    target_method(**kwargs)

                    for attr in [
                        "data",
                        "df",
                        "assigned",
                        "assigned_data",
                        "peaks",
                    ]:
                        if hasattr(spec, attr):
                            val = getattr(spec, attr)
                            if isinstance(val, pd.DataFrame) and not val.empty:
                                assigned_df = val.copy()
                                break
            except Exception:
                assigned_df = pd.DataFrame()

        if assigned_df.empty or "C" not in assigned_df.columns:
            assigned_df = fast_formula_assigner(
                peaks_df=peaks_df,
                bounds=bounds,
                max_hc=max_hc,
                max_oc=max_oc,
                ppm_tolerance=ppm_tolerance,
                ion_mode=ion_mode,
                max_charge=max_charge,
                iso_check=iso_check,
                iso_strict=iso_strict,
            )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not assigned_df.empty and "C" in assigned_df.columns:
        assigned_df = calculate_descriptors(assigned_df, lang=lang)

    return assigned_df


def compute_kmd(
    masses: np.ndarray, base_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nom = KMD_BASES[base_key]["nom"]
    exact = KMD_BASES[base_key]["exact"]
    km = masses * (nom / exact)
    kmd = km - np.floor(km)
    nkm = np.floor(km).astype(int)
    return km, nkm, kmd


def compute_vk20_grid(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oc_bins = [0.00, 0.25, 0.50, 0.75, 1.000001]
    hc_bins = [0.20, 0.60, 1.00, 1.40, 1.80, 2.200001]

    sub = df[
        (df["O/C"] >= 0.0)
        & (df["O/C"] <= 1.0)
        & (df["H/C"] >= 0.2)
        & (df["H/C"] <= 2.2)
    ].copy()
    if sub.empty:
        return np.zeros((5, 4)), np.zeros((5, 4)), pd.DataFrame()

    c_idx = np.clip(np.digitize(sub["O/C"], oc_bins) - 1, 0, 3)
    r_idx = np.clip(np.digitize(sub["H/C"], hc_bins) - 1, 0, 4)

    grid_count = np.zeros((5, 4), dtype=float)
    grid_weight = np.zeros((5, 4), dtype=float)

    total_count = len(sub)
    total_int = (
        sub["intensity"].sum()
        if "intensity" in sub.columns
        else float(total_count)
    )

    np.add.at(grid_count, (r_idx, c_idx), 1.0)
    if "intensity" in sub.columns:
        np.add.at(grid_weight, (r_idx, c_idx), sub["intensity"].values)
    else:
        grid_weight = grid_count.copy()

    pct_count = (
        (grid_count / total_count) * 100.0 if total_count > 0 else grid_count
    )
    pct_weight = (
        (grid_weight / total_int) * 100.0 if total_int > 0 else grid_weight
    )

    features = []
    for r in range(5):
        for c in range(4):
            vk_num = 1 + r * 4 + c
            features.append(
                {
                    "Zone": f"VK_{vk_num}",
                    "Count": int(grid_count[r, c]),
                    "Relative_Count_pct": round(pct_count[r, c], 3),
                    "Weighted_Intensity_pct": round(pct_weight[r, c], 3),
                }
            )
    return pct_count, pct_weight, pd.DataFrame(features)


def align_two_spectra_fast(
    df_a: pd.DataFrame, df_b: pd.DataFrame, ppm_tol: float = 1.5
):
    a = df_a.sort_values("mass").reset_index(drop=True)
    b = df_b.sort_values("mass").reset_index(drop=True)

    ma = a["mass"].values
    mb = b["mass"].values

    delta = ma * (ppm_tol * 1e-6)
    left = np.searchsorted(mb, ma - delta)
    right = np.searchsorted(mb, ma + delta)

    matched_a = []
    matched_b = []
    b_used = set()

    for i in range(len(ma)):
        l, r = left[i], right[i]
        if r > l:
            best_j = None
            min_d = 1e9
            for j in range(l, r):
                if j in b_used:
                    continue
                diff = abs(ma[i] - mb[j])
                if diff < min_d:
                    min_d = diff
                    best_j = j
            if best_j is not None:
                b_used.add(best_j)
                matched_a.append(i)
                matched_b.append(best_j)

    n_a = len(a)
    n_b = len(b)
    n_common = len(matched_a)
    n_union = n_a + n_b - n_common
    jaccard = n_common / n_union if n_union > 0 else 0.0

    if n_common > 0:
        ia = a.loc[matched_a, "intensity"].values
        ib = b.loc[matched_b, "intensity"].values
        cos_sim = np.dot(ia, ib) / (
            np.linalg.norm(ia) * np.linalg.norm(ib) + 1e-12
        )
    else:
        cos_sim = 0.0

    return {
        "df_a": a,
        "df_b": b,
        "matched_a": matched_a,
        "matched_b": matched_b,
        "n_a": n_a,
        "n_b": n_b,
        "n_common": n_common,
        "jaccard": jaccard,
        "cos_sim": cos_sim,
    }


def perform_spectral_algebra(
    df_a: pd.DataFrame, df_b: pd.DataFrame, operation: str, ppm_tol: float = 1.5
) -> pd.DataFrame:
    align = align_two_spectra_fast(df_a, df_b, ppm_tol=ppm_tol)
    a = align["df_a"]
    b = align["df_b"]
    m_a_set = set(align["matched_a"])
    m_b_set = set(align["matched_b"])

    if operation == "and":
        return a.iloc[align["matched_a"]].copy().reset_index(drop=True)
    elif operation == "sub_a_b":
        mask = [i not in m_a_set for i in range(len(a))]
        return a[mask].copy().reset_index(drop=True)
    elif operation == "sub_b_a":
        mask = [j not in m_b_set for j in range(len(b))]
        return b[mask].copy().reset_index(drop=True)
    elif operation == "xor":
        a_uniq = a[[i not in m_a_set for i in range(len(a))]]
        b_uniq = b[[j not in m_b_set for j in range(len(b))]]
        return (
            pd.concat([a_uniq, b_uniq], ignore_index=True)
            .sort_values("mass")
            .reset_index(drop=True)
        )
    elif operation == "or":
        common_rows = []
        for i_a, i_b in zip(align["matched_a"], align["matched_b"]):
            common_rows.append(
                {
                    "mass": (a.loc[i_a, "mass"] + b.loc[i_b, "mass"]) / 2.0,
                    "intensity": a.loc[i_a, "intensity"]
                    + b.loc[i_b, "intensity"],
                }
            )
        a_uniq = a[[i not in m_a_set for i in range(len(a))]]
        b_uniq = b[[j not in m_b_set for j in range(len(b))]]
        res = pd.concat(
            [
                pd.DataFrame(common_rows),
                a_uniq[["mass", "intensity"]],
                b_uniq[["mass", "intensity"]],
            ],
            ignore_index=True,
        )
        return res.sort_values("mass").reset_index(drop=True)
    return pd.DataFrame()


def run_tmds_screening(
    peaks_df: pd.DataFrame, top_n: int = 1500, tol_mda: float = 2.0
):
    sub = (
        peaks_df.sort_values("intensity", ascending=False)
        .head(top_n)
        .sort_values("mass")
        .reset_index(drop=True)
    )
    masses = sub["mass"].values
    n = len(masses)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame()

    diff_matrix = np.abs(masses[:, None] - masses[None, :])
    i_upper, j_upper = np.triu_indices(n, k=1)
    diffs = diff_matrix[i_upper, j_upper]

    tol_da = tol_mda / 1000.0
    total_pairs = len(diffs)
    summary_rows = []
    pair_rows = []

    for item in TMDS_LIBRARY:
        delta_theor = item["delta"]
        mask = np.abs(diffs - delta_theor) <= tol_da
        hit_count = int(np.sum(mask))
        share = (hit_count / total_pairs * 100.0) if total_pairs > 0 else 0.0

        summary_rows.append(
            {
                "Transformation": item["name"],
                "Delta_m": delta_theor,
                "Count": hit_count,
                "Share_pct": round(share, 3),
            }
        )

        if hit_count > 0:
            hit_i = i_upper[mask]
            hit_j = j_upper[mask]
            for idx_a, idx_b in zip(hit_i[:300], hit_j[:300]):
                pair_rows.append(
                    {
                        "Transformation": item["name"],
                        "Mass_1": masses[idx_a],
                        "Mass_2": masses[idx_b],
                        "Delta_obs": abs(masses[idx_a] - masses[idx_b]),
                        "Error_mDa": (
                            abs(masses[idx_a] - masses[idx_b]) - delta_theor
                        )
                        * 1000.0,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def get_calibrant_library(series_name: str, ion_mode: str) -> pd.DataFrame:
    calibrants = []
    if "FA" in series_name or "Жирные" in series_name or "Fatty" in series_name:
        for n in range(12, 34):
            m_neut = (
                n * EXACT_MASSES["C"]
                + 2 * n * EXACT_MASSES["H"]
                + 2 * EXACT_MASSES["O"]
            )
            m_ion = (
                m_neut - H_ION_MASS
                if "ESI(-)" in ion_mode
                else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)
            )
            calibrants.append(
                {"name": f"FA {n}:0 (C{n}H{2*n}O2)", "m_theor": m_ion}
            )
    elif "CHO" in series_name:
        for n in range(14, 32):
            h_count = 2 * n - 8
            m_neut = (
                n * EXACT_MASSES["C"]
                + h_count * EXACT_MASSES["H"]
                + 7 * EXACT_MASSES["O"]
            )
            m_ion = (
                m_neut - H_ION_MASS
                if "ESI(-)" in ion_mode
                else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)
            )
            calibrants.append(
                {"name": f"CHO C{n}H{h_count}O7", "m_theor": m_ion}
            )
    return pd.DataFrame(calibrants)


# ==============================================================================
# ИНИЦИАЛИЗАЦИЯ ХРАНИЛИЩА SESSION_STATE
# ==============================================================================
if "spectra_db" not in st.session_state:
    st.session_state["spectra_db"] = {}
    for fname in os.listdir(STORAGE_DIR):
        if fname.lower().endswith((".csv", ".txt", ".tsv", ".xy")):
            fpath = os.path.join(STORAGE_DIR, fname)
            try:
                with open(fpath, "rb") as f:
                    content = f.read()
                st.session_state["spectra_db"][fname] = {
                    "raw_path": fpath,
                    "file_bytes": content,
                    "parsed_peaks": None,
                    "assigned_df": None,
                    "raw_df": None,
                }
            except Exception:
                pass

if "eem_ml_descriptors" not in st.session_state:
    st.session_state["eem_ml_descriptors"] = None

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ: ВЫБОР МОДУЛЯ И ПАРАМЕТРОВ
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚗️ Платформа ChemoSuite")
    active_module = st.radio(
        "Аналитический модуль / Analytical Module:",
        [
            "🧪 NOM-Spectra FT-ICR MS",
            "💡 EEM-PARAFAC (Флуоресценция)",
            "🧬 ChemoSuite ML (Data Fusion)",
        ],
        index=0,
    )
    st.markdown("---")

    lang_choice = st.selectbox(
        "🌐 Язык / Language", ["Русский", "English"], index=0
    )
    lang = "ru" if lang_choice == "Русский" else "en"
    st.session_state["lang"] = lang

# Фирменный баннер платформы
st.markdown(
    """
    <div class="platform-header">
        <h2>⚗️ ChemoSuite: Анализ РОВ и Шлам-лигнина</h2>
        <p>Кафедра аналитической химии & Лаборатория природных гуминовых систем химфака МГУ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# МОДУЛЬ 1: NOM-SPECTRA FT-ICR MS STUDIO
# ==============================================================================
if active_module == "🧪 NOM-Spectra FT-ICR MS":
    with st.sidebar:
        st.title(T[lang]["sidebar_mgr"])
        if not NOMSPECTRA_INSTALLED:
            st.info(T[lang]["nomspectra_missing"])

        uploaded_files = st.file_uploader(
            T[lang]["uploader_label"],
            type=["csv", "txt", "tsv", "xy"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state["spectra_db"]:
                    save_path = os.path.join(STORAGE_DIR, uf.name)
                    b_content = uf.getvalue()
                    with open(save_path, "wb") as f:
                        f.write(b_content)
                    st.session_state["spectra_db"][uf.name] = {
                        "raw_path": save_path,
                        "file_bytes": b_content,
                        "parsed_peaks": None,
                        "assigned_df": None,
                        "raw_df": None,
                    }

        with st.expander(T[lang]["folder_expander"], expanded=False):
            local_folder = st.text_input(
                T[lang]["folder_input"],
                placeholder="C:/data/spectra"
                if lang == "en"
                else "например, C:/data/spectra",
                help=T[lang]["folder_help"],
            )
            if st.button(T[lang]["folder_btn"]):
                if local_folder and os.path.isdir(local_folder):
                    added_count = 0
                    for root, _, files in os.walk(local_folder):
                        for file in files:
                            if file.lower().endswith(
                                (".csv", ".txt", ".tsv", ".xy")
                            ):
                                if file not in st.session_state["spectra_db"]:
                                    src_path = os.path.join(root, file)
                                    dst_path = os.path.join(STORAGE_DIR, file)
                                    try:
                                        with open(src_path, "rb") as sf:
                                            content = sf.read()
                                        with open(dst_path, "wb") as df:
                                            df.write(content)
                                        st.session_state["spectra_db"][file] = {
                                            "raw_path": dst_path,
                                            "file_bytes": content,
                                            "parsed_peaks": None,
                                            "assigned_df": None,
                                            "raw_df": None,
                                        }
                                        added_count += 1
                                    except Exception:
                                        pass
                    st.success(T[lang]["folder_success"].format(n=added_count))
                    st.rerun()
                else:
                    st.error(T[lang]["folder_err"])

        all_spectra = list(st.session_state["spectra_db"].keys())

        if all_spectra:
            st.markdown("---")
            search_query = (
                st.text_input(
                    T[lang]["search_label"],
                    placeholder=T[lang]["search_ph"],
                )
                .strip()
                .lower()
            )

            if search_query:
                available_spectra = [
                    s for s in all_spectra if search_query in s.lower()
                ]
                st.caption(
                    T[lang]["found_n"].format(
                        found=len(available_spectra), total=len(all_spectra)
                    )
                )
            else:
                available_spectra = all_spectra

            if not available_spectra:
                st.warning(T[lang]["search_empty"])
                active_spectrum_name = None
                current_sample = None
            else:
                active_spectrum_name = st.selectbox(
                    T[lang]["active_sample"],
                    options=available_spectra,
                    index=0,
                    key="active_spectrum_selector",
                )
                current_sample = st.session_state["spectra_db"][
                    active_spectrum_name
                ]

                if st.button(T[lang]["del_sample"], type="secondary"):
                    if os.path.exists(current_sample["raw_path"]):
                        try:
                            os.remove(current_sample["raw_path"])
                        except Exception:
                            pass
                    del st.session_state["spectra_db"][active_spectrum_name]
                    st.rerun()

                with st.expander(T[lang]["parse_expander"], expanded=False):
                    delim_opts = [
                        "Auto" if lang == "en" else "Авто (автоопределение)",
                        "Tab (\\t)" if lang == "en" else "Табуляция (\\t)",
                        "Comma (,)" if lang == "en" else "Запятая (,)",
                        "Semicolon (;)"
                        if lang == "en"
                        else "Точка с запятой (;)",
                        "Space" if lang == "en" else "Пробел",
                    ]
                    delimiter = st.selectbox(
                        T[lang]["delimiter"],
                        delim_opts,
                        index=0,
                        key=f"delim_{active_spectrum_name}",
                    )
                    decimal_sep = st.selectbox(
                        T[lang]["decimal_sep"],
                        [".", ","],
                        index=0,
                        key=f"dec_{active_spectrum_name}",
                    )
                    has_header = st.checkbox(
                        T[lang]["has_header"],
                        value=True,
                        key=f"head_{active_spectrum_name}",
                    )

                try:
                    raw_df = parse_uploaded_file(
                        current_sample["file_bytes"],
                        delimiter,
                        decimal_sep,
                        has_header,
                    )
                    current_sample["raw_df"] = raw_df

                    if raw_df.empty:
                        st.error(
                            "Error: File is empty or failed to parse rows."
                            if lang == "en"
                            else "Ошибка: Файл пуст или не удалось распознать строки."
                        )
                        st.stop()

                    st.write(T[lang]["preview_caption"])
                    st_df(raw_df.head(5))

                    col_names = list(raw_df.columns)
                    def_mz_idx = 0
                    def_int_idx = 1 if len(col_names) > 1 else 0
                    for idx, cname in enumerate(col_names):
                        cn_low = str(cname).lower()
                        if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]):
                            def_mz_idx = idx
                        elif any(
                            k in cn_low for k in ["int", "i", "count", "abund"]
                        ):
                            def_int_idx = idx

                    col_mz = st.selectbox(
                        T[lang]["col_mz"],
                        col_names,
                        index=def_mz_idx,
                        key=f"mzcol_{active_spectrum_name}",
                    )
                    col_int = st.selectbox(
                        T[lang]["col_int"],
                        col_names,
                        index=def_int_idx,
                        key=f"intcol_{active_spectrum_name}",
                    )

                    if col_mz == col_int:
                        st.warning(T[lang]["same_col_warn"])

                    st.markdown("---")
                    st.subheader(T[lang]["filtering_header"])

                    norm_choice = st.selectbox(
                        T[lang]["norm_mode_label"],
                        [
                            T[lang]["norm_base"],
                            T[lang]["norm_tic"],
                            T[lang]["norm_raw"],
                        ],
                        index=0,
                        key=f"norm_mode_{active_spectrum_name}",
                    )

                    clean_mass = pd.to_numeric(raw_df[col_mz], errors="coerce")
                    clean_int = pd.to_numeric(raw_df[col_int], errors="coerce")
                    valid_mask = clean_mass.notna() & clean_int.notna()

                    valid_df = pd.DataFrame(
                        {
                            "mass": clean_mass[valid_mask].astype(float),
                            "intensity": clean_int[valid_mask].astype(float),
                        }
                    )
                    valid_df = valid_df[valid_df["mass"] > 0]

                    if valid_df.empty:
                        st.error(
                            "Error: No valid numeric data found in selected columns."
                            if lang == "en"
                            else "Ошибка: В выбранных колонках нет корректных числовых данных."
                        )
                        st.stop()

                    min_m_data = float(valid_df["mass"].min())
                    max_m_data = float(valid_df["mass"].max())

                    mz_range = st.slider(
                        T[lang]["mz_range"],
                        min_value=max(50.0, float(np.floor(min_m_data))),
                        max_value=min(2500.0, float(np.ceil(max_m_data))),
                        value=(
                            max(100.0, float(np.floor(min_m_data))),
                            min(1200.0, float(np.ceil(max_m_data))),
                        ),
                        step=10.0,
                        key=f"mzrange_{active_spectrum_name}",
                    )

                    max_int_data = float(valid_df["intensity"].max())
                    cutoff_intensity = st.number_input(
                        T[lang]["cutoff_int"],
                        min_value=0.0,
                        max_value=max_int_data,
                        value=0.0,
                        step=max_int_data * 0.001 if max_int_data > 0 else 1.0,
                        format="%.2e",
                        key=f"cutoff_{active_spectrum_name}",
                    )

                    filtered_df = valid_df[
                        (valid_df["mass"] >= mz_range[0])
                        & (valid_df["mass"] <= mz_range[1])
                        & (valid_df["intensity"] >= cutoff_intensity)
                    ].sort_values("mass").reset_index(drop=True)

                    if not filtered_df.empty:
                        if norm_choice == T[lang]["norm_base"]:
                            max_i = filtered_df["intensity"].max()
                            filtered_df["norm_intensity"] = (
                                filtered_df["intensity"] / max_i
                            ) * 100.0
                        elif norm_choice == T[lang]["norm_tic"]:
                            sum_i = filtered_df["intensity"].sum()
                            filtered_df["norm_intensity"] = (
                                filtered_df["intensity"] / sum_i
                            ) * 100.0
                        else:
                            filtered_df["norm_intensity"] = filtered_df[
                                "intensity"
                            ]
                    else:
                        filtered_df["norm_intensity"] = []

                    current_sample["parsed_peaks"] = filtered_df
                    st.success(
                        T[lang]["loaded_peaks_success"].format(
                            n=len(filtered_df)
                        )
                    )

                except KeyError as e:
                    st.error(f"Mapping error: {e}")
                    st.stop()
                except Exception as err:
                    st.error(f"Error parsing file: {err}")
                    st.stop()
        else:
            active_spectrum_name = None
            current_sample = None
            st.info(T[lang]["no_spectra_info"])

    st.title(T[lang]["title"])

    tabs = st.tabs(
        [
            T[lang]["tab_stick"],
            T[lang]["tab_recal"],
            T[lang]["tab_assign"],
            T[lang]["tab_vk"],
            T[lang]["tab_kmd"],
            T[lang]["tab_vk20"],
            T[lang]["tab_cmp"],
            T[lang]["tab_tmds"],
            T[lang]["tab_desc"],
        ]
    )

    # 1. Stick Plot
    with tabs[0]:
        if (
            current_sample is None
            or current_sample["parsed_peaks"] is None
            or current_sample["parsed_peaks"].empty
        ):
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_stick']}: {active_spectrum_name}")

            col_ctrl1, col_ctrl2 = st.columns([3, 1])
            with col_ctrl1:
                st.caption(T[lang]["stick_signals"].format(n=len(peaks_df)))
            with col_ctrl2:
                annotate_top = st.checkbox(
                    T[lang]["annotate_top"],
                    value=True,
                    key=f"ann_{active_spectrum_name}",
                )

            m_vals = peaks_df["mass"].values
            i_vals = peaks_df["norm_intensity"].values

            x_stick = np.empty(len(m_vals) * 3)
            y_stick = np.empty(len(m_vals) * 3)
            x_stick[0::3] = m_vals
            x_stick[1::3] = m_vals
            x_stick[2::3] = None
            y_stick[0::3] = 0
            y_stick[1::3] = i_vals
            y_stick[2::3] = None

            fig_stick = go.Figure()
            fig_stick.add_trace(
                go.Scattergl(
                    x=x_stick,
                    y=y_stick,
                    mode="lines",
                    line=dict(color="#0b5394", width=1.1),
                    hoverinfo="skip",
                    name=T[lang]["peak_trace"],
                )
            )

            if annotate_top and len(peaks_df) > 0:
                top5 = peaks_df.nlargest(5, "norm_intensity")
                fig_stick.add_trace(
                    go.Scattergl(
                        x=top5["mass"],
                        y=top5["norm_intensity"],
                        mode="markers+text",
                        text=[f"{val:.4f}" for val in top5["mass"]],
                        textposition="top center",
                        marker=dict(color="#b45f06", size=7),
                        textfont=dict(
                            color="#b45f06", size=10, family="sans-serif"
                        ),
                        name=T[lang]["top5_trace"],
                        hovertemplate="<b>m/z</b>: %{x:.4f}<br><b>Int</b>: %{y:.1f}<extra></extra>",
                    )
                )

            y_max_plot = (
                float(peaks_df["norm_intensity"].max() * 1.15)
                if not peaks_df.empty
                else 100.0
            )
            fig_stick.update_layout(
                xaxis=dict(
                    title=T[lang]["mz_axis"],
                    gridcolor="#f1f3f5",
                    zerolinecolor="#ced4da",
                ),
                yaxis=dict(
                    title=T[lang]["rel_int_axis"],
                    range=[0, y_max_plot],
                    gridcolor="#f1f3f5",
                    zerolinecolor="#ced4da",
                ),
                plot_bgcolor="white",
                height=500,
                margin=dict(l=45, r=30, t=30, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            st.caption(T[lang]["nav_caption"])
            st_plotly(fig_stick)

    # 2. Рекалибровка m/z
    with tabs[1]:
        if (
            current_sample is None
            or current_sample["parsed_peaks"] is None
            or current_sample["parsed_peaks"].empty
        ):
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_recal']}: {active_spectrum_name}")

            cal_opts = [
                "Fatty Acids (C12–C33 saturated FA)"
                if lang == "en"
                else "Жирные кислоты (C12–C33 насыщенные ЖК)",
                "CHO Homologues (C_n H_{2n-8} O7)"
                if lang == "en"
                else "Гомологи CHO (C_n H_{2n-8} O7)",
            ]

            c_r1, c_r2, c_r3 = st.columns([2, 1.5, 1.5])
            with c_r1:
                calib_type = st.selectbox(
                    T[lang]["recal_cal_set"],
                    cal_opts,
                    index=0,
                    key=f"calibtype_{active_spectrum_name}",
                )
            with c_r2:
                search_tol = st.number_input(
                    T[lang]["recal_tol"],
                    min_value=2.0,
                    max_value=25.0,
                    value=8.0,
                    step=0.5,
                    key=f"stol_{active_spectrum_name}",
                )
            with c_r3:
                poly_order = st.selectbox(
                    T[lang]["recal_poly"],
                    [1, 2],
                    index=1,
                    key=f"polyord_{active_spectrum_name}",
                )

            calib_ion_mode = st.selectbox(
                T[lang]["recal_ion"],
                [
                    "ESI(-) [M - H]⁻",
                    "ESI(+) [M + H]⁺",
                    "Neutral masses [M]"
                    if lang == "en"
                    else "Нейтральные массы [M]",
                ],
                index=0,
                key=f"calib_ion_{active_spectrum_name}",
            )
            calib_lib = get_calibrant_library(calib_type, calib_ion_mode)

            exp_masses = peaks_df["mass"].values
            matched_calibs = []

            for _, row in calib_lib.iterrows():
                m_th = row["m_theor"]
                delta_lim = m_th * (search_tol * 1e-6)
                candidates = exp_masses[
                    (exp_masses >= m_th - delta_lim)
                    & (exp_masses <= m_th + delta_lim)
                ]
                if len(candidates) > 0:
                    best_m = candidates[np.argmin(np.abs(candidates - m_th))]
                    err_ppm = ((best_m - m_th) / m_th) * 1e6
                    delta_m = best_m - m_th
                    matched_calibs.append(
                        {
                            "Calibrant": row["name"],
                            "m_theor": m_th,
                            "m_exp": best_m,
                            "delta_m": delta_m,
                            "error_ppm": err_ppm,
                        }
                    )

            if len(matched_calibs) < 3:
                st.warning(
                    T[lang]["recal_few_peaks"].format(n=len(matched_calibs))
                )
            else:
                calib_df = pd.DataFrame(matched_calibs)
                st.write(T[lang]["recal_found"].format(n=len(calib_df)))

                spec_min = float(peaks_df["mass"].min())
                spec_max = float(peaks_df["mass"].max())
                spec_span = max(1.0, spec_max - spec_min)

                calib_min = float(calib_df["m_exp"].min())
                calib_max = float(calib_df["m_exp"].max())
                calib_span = max(0.0, calib_max - calib_min)
                coverage = calib_span / spec_span

                actual_poly_order = poly_order
                if poly_order > 1 and coverage < 0.60:
                    actual_poly_order = 1
                    st.warning(
                        T[lang]["recal_runge_warn"].format(
                            cmin=calib_min,
                            cmax=calib_max,
                            cov=coverage * 100,
                        )
                    )

                coeffs = np.polyfit(
                    calib_df["m_exp"].values,
                    calib_df["delta_m"].values,
                    deg=actual_poly_order,
                )
                poly_fn = np.poly1d(coeffs)

                corrected_exp = calib_df["m_exp"] - poly_fn(calib_df["m_exp"])
                residual_ppm = (
                    (corrected_exp - calib_df["m_theor"]) / calib_df["m_theor"]
                ) * 1e6

                fig_rec, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(13, 5), dpi=150
                )
                ax1.scatter(
                    calib_df["m_exp"],
                    calib_df["error_ppm"],
                    color="#D62728",
                    s=35,
                    label=T[lang]["recal_scatter_before"],
                )
                m_grid = np.linspace(
                    peaks_df["mass"].min(), peaks_df["mass"].max(), 200
                )
                grid_ppm_drift = (poly_fn(m_grid) / m_grid) * 1e6
                poly_lbl = (
                    f"Degree {actual_poly_order} poly"
                    if lang == "en"
                    else f"Полином степени {actual_poly_order}"
                )
                ax1.plot(
                    m_grid,
                    grid_ppm_drift,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=poly_lbl,
                )
                ax1.set_xlabel("m/z", fontsize=10)
                ax1.set_ylabel(T[lang]["ppm_axis"], fontsize=10)
                ax1.set_title(T[lang]["recal_title_before"], fontsize=11)
                ax1.grid(True, linestyle="--", alpha=0.4)
                ax1.legend()

                ax2.scatter(
                    calib_df["m_exp"],
                    residual_ppm,
                    color="#2CA02C",
                    s=35,
                    label=T[lang]["recal_scatter_after"],
                )
                ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
                ax2.set_xlabel("m/z", fontsize=10)
                ax2.set_ylabel(T[lang]["res_ppm_axis"], fontsize=10)
                ax2.set_title(T[lang]["recal_title_after"], fontsize=11)
                ax2.grid(True, linestyle="--", alpha=0.4)
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig_rec)
                plt.close(fig_rec)

                if st.button(
                    T[lang]["recal_btn"],
                    type="primary",
                    key=f"btn_recal_{active_spectrum_name}",
                ):
                    m_orig = peaks_df["mass"].values
                    m_recalibrated = m_orig - poly_fn(m_orig)
                    current_sample["parsed_peaks"]["mass"] = m_recalibrated
                    current_sample["assigned_df"] = None
                    st.success(
                        T[lang]["recal_success"].format(
                            n=len(m_recalibrated),
                            before=calib_df["error_ppm"].abs().mean(),
                            after=residual_ppm.abs().mean(),
                        )
                    )

    # 3. Приписывание формул
    with tabs[2]:
        if (
            current_sample is None
            or current_sample["parsed_peaks"] is None
            or current_sample["parsed_peaks"].empty
        ):
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_assign']}: {active_spectrum_name}")

            with st.expander(T[lang]["assign_expander"], expanded=True):
                col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
                with col_m1:
                    ion_mode = st.selectbox(
                        T[lang]["ion_mode"],
                        [
                            "ESI(-) [M - H]⁻",
                            "ESI(+) [M + H]⁺",
                            "Neutral masses [M]"
                            if lang == "en"
                            else "Нейтральные массы [M]",
                        ],
                        index=0,
                        key=f"ionmode_{active_spectrum_name}",
                    )
                with col_m2:
                    max_charge_val = st.selectbox(
                        T[lang]["max_charge"],
                        [1, 2],
                        index=0,
                        key=f"zmax_{active_spectrum_name}",
                    )
                with col_m3:
                    ppm_tol = st.number_input(
                        T[lang]["ppm_tol"],
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key=f"ppmtol_{active_spectrum_name}",
                    )

                col_el1, col_el2, col_el3 = st.columns(3)
                with col_el1:
                    c_bounds = st.slider(
                        T[lang]["c_lim"],
                        4,
                        120,
                        (4, 120),
                        key=f"cb_{active_spectrum_name}",
                    )
                    h_bounds = st.slider(
                        T[lang]["h_lim"],
                        4,
                        200,
                        (4, 200),
                        key=f"hb_{active_spectrum_name}",
                    )
                with col_el2:
                    o_bounds = st.slider(
                        T[lang]["o_lim"],
                        1,
                        60,
                        (1, 60),
                        key=f"ob_{active_spectrum_name}",
                    )
                    n_bounds = st.slider(
                        T[lang]["n_lim"],
                        0,
                        3,
                        (0, 2),
                        key=f"nb_{active_spectrum_name}",
                    )
                with col_el3:
                    s_bounds = st.slider(
                        T[lang]["s_lim"],
                        0,
                        2,
                        (0, 1),
                        key=f"sb_{active_spectrum_name}",
                    )
                    max_oc_val = st.slider(
                        T[lang]["max_oc"],
                        0.1,
                        1.0,
                        1.0,
                        step=0.05,
                        key=f"moc_{active_spectrum_name}",
                    )
                    max_hc_val = st.slider(
                        T[lang]["max_hc"],
                        0.2,
                        2.2,
                        2.0,
                        step=0.05,
                        key=f"mhc_{active_spectrum_name}",
                    )

                col_iso1, col_iso2 = st.columns(2)
                with col_iso1:
                    iso_chk = st.checkbox(
                        T[lang]["iso_filter_label"],
                        value=False,
                        key=f"isochk_{active_spectrum_name}",
                    )
                with col_iso2:
                    iso_strict_chk = st.checkbox(
                        T[lang]["iso_strict_label"],
                        value=False,
                        disabled=not iso_chk,
                        key=f"isostrict_{active_spectrum_name}",
                    )

            if st.button(
                T[lang]["assign_btn"],
                type="primary",
                key=f"btn_assign_{active_spectrum_name}",
            ):
                elem_bounds = {
                    "C": c_bounds,
                    "H": h_bounds,
                    "O": o_bounds,
                    "N": n_bounds,
                    "S": s_bounds,
                }
                with st.spinner(T[lang]["assign_spinner"]):
                    assigned_res = run_formula_assignment(
                        peaks_df=peaks_df,
                        bounds=elem_bounds,
                        max_hc=max_hc_val,
                        max_oc=max_oc_val,
                        ppm_tolerance=ppm_tol,
                        ion_mode=ion_mode,
                        max_charge=max_charge_val,
                        iso_check=iso_chk,
                        iso_strict=iso_strict_chk,
                        lang=lang,
                    )
                    current_sample["assigned_df"] = assigned_res

            assigned_data = current_sample["assigned_df"]
            if assigned_data is not None and not assigned_data.empty:
                total_n = len(peaks_df)
                assigned_n = len(assigned_data)
                rate = (
                    (assigned_n / total_n * 100.0) if total_n > 0 else 0.0
                )
                mean_ppm = (
                    assigned_data["error_ppm"].abs().mean()
                    if "error_ppm" in assigned_data.columns
                    else 0.0
                )

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(T[lang]["metric_total"], f"{total_n:,}")
                m2.metric(T[lang]["metric_assigned"], f"{assigned_n:,}")
                m3.metric(T[lang]["metric_rate"], f"{rate:.1f}%")
                m4.metric(T[lang]["metric_err"], f"{mean_ppm:.3f}")

                st.write(T[lang]["assign_tbl_title"])
                st_df(assigned_data)

                csv_data = assigned_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=T[lang]["dl_csv_formulas"],
                    data=csv_data,
                    file_name=f"{active_spectrum_name}_formulas.csv",
                    mime="text/csv",
                    key=f"dl_csv_formulas_{active_spectrum_name}",
                )
            elif assigned_data is not None and assigned_data.empty:
                st.warning(T[lang]["no_formulas_warn"])

    # 4. Диаграммы и проекции
    with tabs[3]:
        assigned_data = (
            current_sample.get("assigned_df") if current_sample else None
        )
        if assigned_data is None or assigned_data.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            st.subheader(f"{T[lang]['tab_vk']}: {active_spectrum_name}")
            proj_type = st.radio(
                T[lang]["proj_mode"],
                [
                    T[lang]["proj_vk"],
                    T[lang]["proj_dbe_c"],
                    T[lang]["proj_custom"],
                ],
                horizontal=True,
                key=f"proj_type_{active_spectrum_name}",
            )
            pow_exp = st.slider(
                T[lang]["pow_exp"],
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"pow_{active_spectrum_name}",
            )

            palette = {
                "CHO": "#0020C2",
                "CHON": "#FF7F0E",
                "CHOS": "#2CA02C",
                "CHONS": "#D62728",
            }
            draw_order = ["CHO", "CHON", "CHOS", "CHONS"]

            if proj_type == T[lang]["proj_vk"]:
                fig_inter = go.Figure()
                for cls in draw_order:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    if sub.empty:
                        continue
                    scaled_sizes = (
                        2.0 + 5.0 * (sub["norm_intensity"] / 100.0) ** pow_exp
                    )
                    hover_text = [
                        f"<b>{row.get('Formula', '')}</b><br>m/z: {row['mass']:.4f}<br>Err: {row.get('error_ppm', 0):.2f} ppm<br>DBE: {row.get('DBE', 0):.1f}<br>AI: {row.get('AI', 0):.2f}<br>Int: {row.get('norm_intensity', 0):.1f}"
                        for _, row in sub.iterrows()
                    ]
                    fig_inter.add_trace(
                        go.Scattergl(
                            x=sub["O/C"],
                            y=sub["H/C"],
                            mode="markers",
                            name=f"{cls} ({len(sub):,})",
                            marker=dict(
                                color=palette[cls],
                                size=scaled_sizes,
                                opacity=0.65,
                            ),
                            text=hover_text,
                            hoverinfo="text",
                        )
                    )
                fig_inter.update_layout(
                    xaxis=dict(
                        title=T[lang]["oc_axis"],
                        range=[0.0, 1.0],
                        gridcolor="#f1f3f5",
                    ),
                    yaxis=dict(
                        title=T[lang]["hc_axis"],
                        range=[0.2, 2.2],
                        gridcolor="#f1f3f5",
                    ),
                    plot_bgcolor="white",
                    height=560,
                )
                st.caption(T[lang]["nav_caption"])
                st_plotly(fig_inter)

            elif proj_type == T[lang]["proj_dbe_c"]:
                fig_inter = go.Figure()
                for cls in draw_order:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    if sub.empty:
                        continue
                    scaled_sizes = (
                        2.0 + 5.0 * (sub["norm_intensity"] / 100.0) ** pow_exp
                    )
                    fig_inter.add_trace(
                        go.Scattergl(
                            x=sub["C"],
                            y=sub["DBE"],
                            mode="markers",
                            name=f"{cls} ({len(sub):,})",
                            marker=dict(
                                color=palette[cls],
                                size=scaled_sizes,
                                opacity=0.65,
                            ),
                            text=[
                                f"{row.get('Formula', '')}<br>AI: {row.get('AI', 0):.2f}"
                                for _, row in sub.iterrows()
                            ],
                            hoverinfo="text",
                        )
                    )
                c_line = np.linspace(4, 60, 100)
                fig_inter.add_trace(
                    go.Scatter(
                        x=c_line,
                        y=c_line * 0.9,
                        mode="lines",
                        name="Planar limit (AI=0.67)",
                        line=dict(color="red", dash="dot"),
                    )
                )
                fig_inter.add_trace(
                    go.Scatter(
                        x=c_line,
                        y=c_line * 0.5,
                        mode="lines",
                        name="Aromatic (AI=0.5)",
                        line=dict(color="gray", dash="dash"),
                    )
                )
                fig_inter.update_layout(
                    xaxis=dict(
                        title="Carbon atoms (C)",
                        range=[0, assigned_data["C"].max() + 5],
                    ),
                    yaxis=dict(
                        title="Double Bond Equivalent (DBE)",
                        range=[0, assigned_data["DBE"].max() + 3],
                    ),
                    plot_bgcolor="white",
                    height=560,
                )
                st.caption(T[lang]["nav_caption"])
                st_plotly(fig_inter)

            else:
                avail_cols = [
                    "mass",
                    "intensity",
                    "norm_intensity",
                    "C",
                    "H",
                    "O",
                    "N",
                    "S",
                    "H/C",
                    "O/C",
                    "DBE",
                    "DBE-O",
                    "AI",
                    "NOSC",
                    "error_ppm",
                ]
                c_sel1, c_sel2, c_sel3 = st.columns(3)
                with c_sel1:
                    cx = st.selectbox(
                        "Ось X / X Axis:",
                        avail_cols,
                        index=avail_cols.index("mass"),
                    )
                with c_sel2:
                    cy = st.selectbox(
                        "Ось Y / Y Axis:",
                        avail_cols,
                        index=avail_cols.index("DBE"),
                    )
                with c_sel3:
                    cc = st.selectbox(
                        "Цвет / Color map:",
                        avail_cols,
                        index=avail_cols.index("AI"),
                    )

                fig_cust = px.scatter(
                    assigned_data,
                    x=cx,
                    y=cy,
                    color=cc,
                    color_continuous_scale="Viridis",
                    hover_data=["Formula", "mass", "AI"],
                    opacity=0.7,
                    render_mode="webgl",
                )
                fig_cust.update_layout(height=560, plot_bgcolor="white")
                st_plotly(fig_cust)

    # 5. KMD
    with tabs[4]:
        st.subheader(
            f"{T[lang]['tab_kmd']}: {active_spectrum_name if active_spectrum_name else ''}"
        )
        assigned_df = (
            current_sample.get("assigned_df") if current_sample else None
        )
        peaks_df = current_sample.get("parsed_peaks") if current_sample else None

        if assigned_df is not None and not assigned_df.empty:
            work_df = assigned_df.copy()
            has_formulas = True
        elif peaks_df is not None and not peaks_df.empty:
            work_df = peaks_df.copy()
            has_formulas = False
        else:
            st.info(T[lang]["no_spectra_info"])
            work_df = None

        if work_df is not None:
            col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
            with col_b1:
                base_choice = st.selectbox(
                    T[lang]["kmd_base_group"],
                    list(KMD_BASES.keys()),
                    index=0,
                    key=f"kmd_base_{active_spectrum_name}",
                )
            with col_b2:
                color_options = [
                    T[lang]["kmd_parity_mode"],
                    T[lang]["kmd_int_mode"],
                ]
                if has_formulas:
                    color_options.extend(
                        [
                            T[lang]["kmd_o_mode"],
                            T[lang]["kmd_dbe_mode"],
                            T[lang]["kmd_hetero_mode"],
                        ]
                    )
                color_mode = st.selectbox(
                    T[lang]["kmd_color_mode"],
                    color_options,
                    index=0,
                    key=f"kmd_color_{active_spectrum_name}",
                )
            with col_b3:
                point_sz = st.slider(
                    T[lang]["kmd_pt_size"],
                    1,
                    8,
                    3,
                    key=f"kmd_pt_{active_spectrum_name}",
                )

            km, nkm, kmd = compute_kmd(work_df["mass"].values, base_choice)
            work_df["KM"] = km
            work_df["NKM"] = nkm
            work_df["KMD"] = kmd
            base_label = KMD_BASES[base_choice]["label"]

            fig_kmd_inter = go.Figure()
            if color_mode == T[lang]["kmd_parity_mode"]:
                even_mask = work_df["NKM"] % 2 == 0
                fig_kmd_inter.add_trace(
                    go.Scattergl(
                        x=work_df.loc[even_mask, "NKM"],
                        y=work_df.loc[even_mask, "KMD"],
                        mode="markers",
                        name=T[lang]["kmd_even"],
                        marker=dict(color="#0020C2", size=point_sz, opacity=0.6),
                    )
                )
                fig_kmd_inter.add_trace(
                    go.Scattergl(
                        x=work_df.loc[~even_mask, "NKM"],
                        y=work_df.loc[~even_mask, "KMD"],
                        mode="markers",
                        name=T[lang]["kmd_odd"],
                        marker=dict(color="#FF7F0E", size=point_sz, opacity=0.7),
                    )
                )
            elif color_mode == T[lang]["kmd_hetero_mode"] and has_formulas:
                palette = {
                    "CHO": "#0020C2",
                    "CHON": "#FF7F0E",
                    "CHOS": "#2CA02C",
                    "CHONS": "#D62728",
                }
                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = work_df[work_df["Hetero_Class"] == cls]
                    if not sub.empty:
                        fig_kmd_inter.add_trace(
                            go.Scattergl(
                                x=sub["NKM"],
                                y=sub["KMD"],
                                mode="markers",
                                name=f"{cls} ({len(sub):,})",
                                marker=dict(
                                    color=palette[cls],
                                    size=point_sz,
                                    opacity=0.65,
                                ),
                            )
                        )
            else:
                c_vals = (
                    work_df["O"]
                    if (color_mode == T[lang]["kmd_o_mode"] and has_formulas)
                    else (
                        work_df["DBE"]
                        if (
                            color_mode == T[lang]["kmd_dbe_mode"]
                            and has_formulas
                        )
                        else work_df["norm_intensity"]
                    )
                )
                c_scale = "Viridis" if "O" in color_mode else "Cividis"
                fig_kmd_inter.add_trace(
                    go.Scattergl(
                        x=work_df["NKM"],
                        y=work_df["KMD"],
                        mode="markers",
                        marker=dict(
                            color=c_vals,
                            colorscale=c_scale,
                            size=point_sz,
                            opacity=0.65,
                        ),
                    )
                )

            fig_kmd_inter.update_layout(
                xaxis=dict(
                    title=T[lang]["kmd_x_axis"].format(base=base_label),
                    gridcolor="#f1f3f5",
                ),
                yaxis=dict(
                    title=T[lang]["kmd_y_axis"].format(base=base_label),
                    range=[-0.02, 1.02],
                    gridcolor="#f1f3f5",
                ),
                plot_bgcolor="white",
                height=540,
            )
            st.caption(T[lang]["nav_caption"])
            st_plotly(fig_kmd_inter)

    # 6. Хемотипирование 20 ячеек
    with tabs[5]:
        st.subheader(
            f"{T[lang]['tab_vk20']}: {active_spectrum_name if active_spectrum_name else ''}"
        )
        assigned_df = (
            current_sample.get("assigned_df") if current_sample else None
        )
        if assigned_df is None or assigned_df.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            pct_count, pct_weight, feat_df = compute_vk20_grid(assigned_df)
            metric_mode = st.radio(
                T[lang]["vk20_basis"],
                [T[lang]["vk20_opt_count"], T[lang]["vk20_opt_weight"]],
                horizontal=True,
                key=f"vk20_mode_{active_spectrum_name}",
            )
            active_matrix = (
                pct_count
                if metric_mode == T[lang]["vk20_opt_count"]
                else pct_weight
            )

            fig_hm, ax_hm = plt.subplots(figsize=(9, 6), dpi=150)
            im = ax_hm.imshow(
                active_matrix, origin="lower", cmap="YlOrRd", aspect="auto"
            )
            ax_hm.set_xticks(range(4))
            ax_hm.set_xticklabels(
                [
                    "C1 [0.00-0.25)",
                    "C2 [0.25-0.50)",
                    "C3 [0.50-0.75)",
                    "C4 [0.75-1.00]",
                ],
                fontsize=9,
            )
            ax_hm.set_yticks(range(5))
            ax_hm.set_yticklabels(
                [
                    "R1 [0.20-0.60)",
                    "R2 [0.60-1.00)",
                    "R3 [1.00-1.40)",
                    "R4 [1.40-1.80)",
                    "R5 [1.80-2.20]",
                ],
                fontsize=9,
            )
            ax_hm.set_xlabel(
                T[lang]["oc_axis"], fontsize=10, fontweight="medium"
            )
            ax_hm.set_ylabel(
                T[lang]["hc_axis"], fontsize=10, fontweight="medium"
            )
            ax_hm.set_title(
                T[lang]["vk20_heatmap_title"].format(mode=metric_mode),
                fontsize=11,
            )

            for r in range(5):
                for c in range(4):
                    vk_num = 1 + r * 4 + c
                    val = active_matrix[r, c]
                    txt_color = (
                        "white"
                        if val > (active_matrix.max() * 0.65)
                        else "black"
                    )
                    ax_hm.text(
                        c,
                        r,
                        f"VK_{vk_num}\n{val:.1f}%",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=9,
                        fontweight="bold",
                    )

            plt.colorbar(im, ax=ax_hm, pad=0.02, label="Density (%)")
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

            st.write(feat_df)
            st.download_button(
                label=T[lang]["vk20_dl_csv"],
                data=feat_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{active_spectrum_name}_vk20.csv",
                mime="text/csv",
                key=f"dl_vk20_{active_spectrum_name}",
            )

    # 7. Сравнение и алгебра
    with tabs[6]:
        st.subheader(T[lang]["tab_cmp"])
        if len(all_spectra) < 2:
            st.info(T[lang]["cmp_need_two"])
        else:
            cmp_tab_choice = st.radio(
                "Раздел / Mode:",
                [
                    T[lang]["cmp_subtab_view"],
                    T[lang]["cmp_subtab_sub"],
                    T[lang]["cmp_subtab_algebra"],
                ],
                horizontal=True,
                key="cmp_internal_mode",
            )
            col_s1, col_s2, col_s3 = st.columns([2, 2, 1.5])
            with col_s1:
                name_a = st.selectbox(
                    T[lang]["cmp_spec_a"],
                    all_spectra,
                    index=0,
                    key="cmp_spec_a",
                )
            with col_s2:
                name_b = st.selectbox(
                    T[lang]["cmp_spec_b"],
                    all_spectra,
                    index=1 if len(all_spectra) > 1 else 0,
                    key="cmp_spec_b",
                )
            with col_s3:
                tol_comp = st.number_input(
                    T[lang]["cmp_tol"],
                    min_value=0.1,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key="cmp_ppm_tol",
                )

            sample_a = st.session_state["spectra_db"][name_a]
            sample_b = st.session_state["spectra_db"][name_b]
            peaks_a = sample_a.get("parsed_peaks")
            peaks_b = sample_b.get("parsed_peaks")

            if (
                peaks_a is None
                or peaks_a.empty
                or peaks_b is None
                or peaks_b.empty
            ):
                st.warning("Один из выбранных образцов еще не отфильтрован.")
            elif name_a == name_b:
                st.warning("Выбран один и тот же образец!")
            else:
                align_res = align_two_spectra_fast(
                    peaks_a, peaks_b, ppm_tol=tol_comp
                )

                if cmp_tab_choice == T[lang]["cmp_subtab_view"]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Пиков в {name_a}", f"{align_res['n_a']:,}")
                    m2.metric(f"Пиков в {name_b}", f"{align_res['n_b']:,}")
                    m3.metric(
                        T[lang]["cmp_common"],
                        f"{align_res['n_common']:,}",
                        delta=T[lang]["cmp_jaccard"].format(
                            val=align_res["jaccard"]
                        ),
                    )
                    m4.metric(
                        T[lang]["cmp_cosine"], f"{align_res['cos_sim']:.4f}"
                    )

                elif cmp_tab_choice == T[lang]["cmp_subtab_sub"]:
                    sub_factor = st.slider(
                        T[lang]["sub_factor_label"],
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.05,
                    )
                    if st.button(T[lang]["sub_btn"], type="primary"):
                        sub_peaks = peaks_a.copy()
                        matched_dict = dict(
                            zip(align_res["matched_a"], align_res["matched_b"])
                        )
                        new_ints = [
                            max(
                                0.0,
                                sub_peaks.loc[i, "intensity"]
                                - sub_factor
                                * peaks_b.loc[matched_dict[i], "intensity"],
                            )
                            if i in matched_dict
                            else sub_peaks.loc[i, "intensity"]
                            for i in range(len(sub_peaks))
                        ]
                        sub_peaks["intensity"] = new_ints
                        clean_sub_df = sub_peaks[
                            sub_peaks["intensity"] > 0
                        ].copy().reset_index(drop=True)
                        if not clean_sub_df.empty:
                            clean_sub_df["norm_intensity"] = (
                                clean_sub_df["intensity"]
                                / clean_sub_df["intensity"].max()
                            ) * 100.0

                        new_sample_name = f"{name_a}_sub_{name_b}.csv"
                        st.session_state["spectra_db"][new_sample_name] = {
                            "raw_path": os.path.join(
                                STORAGE_DIR, new_sample_name
                            ),
                            "file_bytes": clean_sub_df.to_csv(
                                sep="\t", index=False
                            ).encode("utf-8"),
                            "parsed_peaks": clean_sub_df,
                            "assigned_df": None,
                            "raw_df": clean_sub_df,
                        }
                        st.success(
                            T[lang]["sub_success"].format(
                                name=new_sample_name, n=len(clean_sub_df)
                            )
                        )
                        st.rerun()

                else:
                    op_choice = st.selectbox(
                        T[lang]["alg_op_label"],
                        [
                            ("and", T[lang]["alg_and"]),
                            ("or", T[lang]["alg_or"]),
                            (
                                "sub_a_b",
                                T[lang]["alg_sub_a_b"].format(
                                    a=name_a, b=name_b
                                ),
                            ),
                            (
                                "sub_b_a",
                                T[lang]["alg_sub_b_a"].format(
                                    a=name_a, b=name_b
                                ),
                            ),
                            ("xor", T[lang]["alg_xor"]),
                        ],
                        format_func=lambda x: x[1],
                    )

                    fig_v, ax_v = plt.subplots(figsize=(8, 4), dpi=150)
                    c_a = patches.Circle(
                        (0.35, 0.5),
                        0.3,
                        facecolor="#0020C2",
                        alpha=0.4,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                    c_b = patches.Circle(
                        (0.65, 0.5),
                        0.3,
                        facecolor="#2CA02C",
                        alpha=0.4,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                    ax_v.add_patch(c_a)
                    ax_v.add_patch(c_b)
                    ax_v.text(
                        0.22,
                        0.5,
                        f"{name_a}\n\n{align_res['n_a'] - align_res['n_common']:,}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax_v.text(
                        0.78,
                        0.5,
                        f"{name_b}\n\n{align_res['n_b'] - align_res['n_common']:,}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax_v.text(
                        0.50,
                        0.5,
                        f"A ∩ B\n\n{align_res['n_common']:,}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="#800000",
                    )
                    ax_v.set_xlim(0, 1)
                    ax_v.set_ylim(0.1, 0.9)
                    ax_v.axis("off")
                    st.pyplot(fig_v)
                    plt.close(fig_v)

                    if st.button(T[lang]["alg_btn"], type="primary"):
                        alg_df = perform_spectral_algebra(
                            peaks_a,
                            peaks_b,
                            operation=op_choice[0],
                            ppm_tol=tol_comp,
                        )
                        if not alg_df.empty:
                            alg_df["norm_intensity"] = (
                                alg_df["intensity"] / alg_df["intensity"].max()
                            ) * 100.0
                        new_alg_name = f"{name_a}_{op_choice[0]}_{name_b}.csv"
                        st.session_state["spectra_db"][new_alg_name] = {
                            "raw_path": os.path.join(
                                STORAGE_DIR, new_alg_name
                            ),
                            "file_bytes": alg_df.to_csv(
                                sep="\t", index=False
                            ).encode("utf-8"),
                            "parsed_peaks": alg_df,
                            "assigned_df": None,
                            "raw_df": alg_df,
                        }
                        st.success(
                            T[lang]["alg_success"].format(
                                name=new_alg_name, n=len(alg_df)
                            )
                        )
                        st.rerun()

    # 8. TMDS
    with tabs[7]:
        st.subheader(
            f"{T[lang]['tab_tmds']}: {active_spectrum_name if active_spectrum_name else ''}"
        )
        peaks_df = current_sample.get("parsed_peaks") if current_sample else None
        if peaks_df is None or peaks_df.empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                top_peaks_tmds = st.slider(
                    T[lang]["tmds_top_label"], 200, 3000, 1200, step=100
                )
            with col_t2:
                tol_mda_val = st.number_input(
                    T[lang]["tmds_tol_label"],
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                )

            tmds_summary, tmds_pairs = run_tmds_screening(
                peaks_df, top_n=top_peaks_tmds, tol_mda=tol_mda_val
            )
            if not tmds_summary.empty:
                fig_tmds_bar = px.bar(
                    tmds_summary,
                    x="Transformation",
                    y="Count",
                    color="Transformation",
                    text=tmds_summary["Share_pct"].apply(lambda v: f"{v:.2f}%"),
                    title=T[lang]["tmds_trans_title"],
                )
                fig_tmds_bar.update_layout(
                    showlegend=False, xaxis_tickangle=-25, height=450
                )
                st_plotly(fig_tmds_bar)
                st.write(tmds_summary)

    # 9. Сводные характеристики
    with tabs[8]:
        assigned_data = (
            current_sample.get("assigned_df") if current_sample else None
        )
        if assigned_data is None or assigned_data.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            st.subheader(f"{T[lang]['tab_desc']}: {active_spectrum_name}")
            mw_n = assigned_data["mass"].mean()
            hc_n = assigned_data["H/C"].mean()
            oc_n = assigned_data["O/C"].mean()
            dbe_n = assigned_data["DBE"].mean()
            dbe_o_n = assigned_data["DBE-O"].mean()
            ai_n = assigned_data["AI"].mean()

            weights = assigned_data["intensity"].values
            sum_weights = np.sum(weights)

            if sum_weights > 0:
                mw_w = (
                    np.sum(assigned_data["mass"].values * weights) / sum_weights
                )
                hc_w = np.sum(assigned_data["H/C"].values * weights) / sum_weights
                oc_w = np.sum(assigned_data["O/C"].values * weights) / sum_weights
                dbe_w = (
                    np.sum(assigned_data["DBE"].values * weights) / sum_weights
                )
                dbe_o_w = (
                    np.sum(assigned_data["DBE-O"].values * weights) / sum_weights
                )
                ai_w = np.sum(assigned_data["AI"].values * weights) / sum_weights
            else:
                mw_w, hc_w, oc_w, dbe_w, dbe_o_w, ai_w = (
                    mw_n,
                    hc_n,
                    oc_n,
                    dbe_n,
                    dbe_o_n,
                    ai_n,
                )

            st.markdown(f"#### {T[lang]['desc_num_avg']}")
            c1, c2, c3 = st.columns(3)
            c1.metric(T[lang]["desc_mn"], f"{mw_n:.2f} Da")
            c2.metric("H/C (Mn)", f"{hc_n:.3f}")
            c3.metric("O/C (Mn)", f"{oc_n:.3f}")

            c4, c5, c6 = st.columns(3)
            c4.metric(T[lang]["desc_dbe"], f"{dbe_n:.2f}")
            c5.metric(T[lang]["desc_dbe_o"], f"{dbe_o_n:.2f}")
            c6.metric(T[lang]["desc_ai"], f"{ai_n:.3f}")

            st.markdown("---")
            st.markdown(f"#### {T[lang]['desc_weighted_header']}")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric(
                T[lang]["desc_mw"], f"{mw_w:.2f} Da", delta=f"{mw_w - mw_n:+.2f}"
            )
            w2.metric("H/C (Mw)", f"{hc_w:.3f}", delta=f"{hc_w - hc_n:+.3f}")
            w3.metric("O/C (Mw)", f"{oc_w:.3f}", delta=f"{oc_w - oc_n:+.3f}")
            w4.metric("AI (Mw)", f"{ai_w:.3f}", delta=f"{ai_w - ai_n:+.3f}")

# ==============================================================================
# МОДУЛЬ 2: 3D ОПТИЧЕСКАЯ СПЕКТРОСКОПИЯ (EEM-PARAFAC)
# ==============================================================================
elif active_module == "💡 EEM-PARAFAC (Флуоресценция)":
    st.title("💡 Анализ флуоресценции: EEM-PARAFAC")
    st.caption(
        "Методология лаборатории природных гуминовых систем химфака МГУ | Анализ РОВ и шлам-лигнина"
    )

    if not EEM_CORE_AVAILABLE:
        st.error(
            "⚠️ Файл `eem_core.py` не обнаружен! Поместите файл `eem_core.py` в ту же директорию, где запущен `app.py`."
        )
        st.stop()

    with st.sidebar:
        st.header("📁 Менеджер EEM матриц")
        eem_data_src = st.radio(
            "Источник оптических данных:",
            [
                "Синтетический бенчмарк (Байкал / Лигнин)",
                "Загрузка файлов (CSV/TXT/DAT)",
            ],
            key="eem_data_source_radio",
        )

        eem_files = []
        if eem_data_src == "Загрузка файлов (CSV/TXT/DAT)":
            eem_files = st.file_uploader(
                "Загрузите таблицы EEM:",
                accept_multiple_files=True,
                type=["csv", "txt", "dat"],
                key="eem_files_uploader",
            )

        st.markdown("---")
        st.subheader("Параметры предобработки")
        clean_scatter = st.checkbox(
            "Удалять Рэлеевское рассеяние 1 и 2 порядка",
            value=True,
            key="eem_clean_scatter",
        )
        norm_raman = st.checkbox(
            "Нормировка на Raman пик воды (R.U.)",
            value=False,
            key="eem_norm_raman",
        )
        palette = st.selectbox(
            "Цветовая шкала EEM:",
            ["Viridis", "Plasma", "Inferno", "Turbo"],
            key="eem_palette_select",
        )

    # Загрузка и подготовка EEM
    samples = []
    if eem_data_src == "Синтетический бенчмарк (Байкал / Лигнин)":
        samples = eem_core.generate_synthetic_chemometrics_dataset(n_samples=10)
    elif eem_files:
        for f in eem_files:
            try:
                df = pd.read_csv(f, index_col=0)
                parsed = eem_core.parse_eem_dataframe(
                    df, sample_id=f.name.split(".")[0]
                )
                if clean_scatter:
                    parsed.data = eem_core.remove_scatter_bands(
                        parsed.data, parsed.ex, parsed.em
                    )
                if norm_raman:
                    parsed.data, _ = eem_core.normalize_to_raman_units(
                        parsed.data, parsed.ex, parsed.em
                    )
                samples.append(parsed)
            except Exception as e:
                st.error(f"Ошибка чтения {f.name}: {e}")

    if not samples:
        st.warning(
            "Загрузите файлы матриц флуоресценции или выберите синтетический бенчмарк."
        )
    else:
        tab_eem1, tab_eem2, tab_eem3 = st.tabs(
            [
                "📊 Экран 1: 2D Контурная карта EEM",
                "📈 Экран 2: Спектральные индексы (FI, HIX, SUVA)",
                "🧬 Экран 3: Моделирование PARAFAC",
            ]
        )

        # Экран 1: Контурные карты EEM
        with tab_eem1:
            st.subheader("Визуализация матриц возбуждения-испускания")
            s_names = [s.sample_id for s in samples]
            selected_sname = st.selectbox(
                "Выберите пробу:", s_names, key="eem_active_sample"
            )
            s_obj = next(s for s in samples if s.sample_id == selected_sname)

            fig_eem = go.Figure(
                data=go.Contour(
                    z=s_obj.data,
                    x=s_obj.ex,
                    y=s_obj.em,
                    colorscale=palette.lower(),
                    contours=dict(
                        coloring="heatmap",
                        showlabels=True,
                        labelfont=dict(size=10, color="white"),
                    ),
                    colorbar=dict(title="Интенсивность"),
                )
            )
            fig_eem.update_layout(
                title=f"Контурная карта флуоресценции: {s_obj.sample_id}",
                xaxis_title="Длина волны возбуждения, Ex (нм)",
                yaxis_title="Длина волны эмиссии, Em (нм)",
                height=560,
                template="plotly_dark",
            )
            st_plotly(fig_eem)

        # Экран 2: Индексы
        with tab_eem2:
            st.subheader("Оптические хемометрические индексы")
            indices_list = [
                eem_core.calculate_spectral_indices(s) for s in samples
            ]
            df_indices = pd.DataFrame(indices_list)

            c_idx1, c_idx2 = st.columns([3, 1])
            with c_idx1:
                st.dataframe(
                    df_indices.style.format(
                        {"FI": "{:.2f}", "HIX": "{:.2f}", "SUVA254": "{:.2f}"}
                    ),
                    use_container_width=True,
                )
            with c_idx2:
                st.download_button(
                    label="📥 Скачать индексы (CSV)",
                    data=df_indices.to_csv(index=False).encode("utf-8"),
                    file_name="EEM_Indices.csv",
                    mime="text/csv",
                    key="eem_dl_indices_btn",
                )

        # Экран 3: PARAFAC
        with tab_eem3:
            st.subheader("Неотрицательное трехкомпонентное PARAFAC-разложение")
            with st.spinner("Идет расчет модели PARAFAC..."):
                tensor_x, em_ax, ex_ax, s_ids = eem_core.build_eem_tensor(
                    samples
                )
                results = eem_core.fit_parafac(
                    tensor_x, n_components=3, random_state=42
                )

            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.metric(
                "Объясненная дисперсия", f"{results['explained_variance']:.2f}%"
            )
            m_c2.metric(
                "CORCONDIA (Диагностика)",
                f"{results['corcondia']:.1f}%",
                delta="Модель адекватна"
                if results["corcondia"] > 85.0
                else "Внимание",
            )
            m_c3.metric("Число флуорофоров", "3 компонента")

            st.markdown("---")
            col_p1, col_p2 = st.columns(2)
            c_labels = [
                "C1: Фульвоподобный",
                "C2: Гуминовый (Лигнин)",
                "C3: Белковоподобный",
            ]

            with col_p1:
                fig_em = go.Figure()
                for r in range(3):
                    fig_em.add_trace(
                        go.Scatter(
                            x=em_ax,
                            y=results["em_profiles"][:, r],
                            mode="lines",
                            name=c_labels[r],
                            line=dict(width=2.5),
                        )
                    )
                fig_em.update_layout(
                    title="Спектры эмиссии (Loading B)",
                    xaxis_title="Длина волны эмиссии Em (нм)",
                    yaxis_title="Относительная интенсивность",
                    template="plotly_white",
                )
                st_plotly(fig_em)

            with col_p2:
                fig_ex = go.Figure()
                for r in range(3):
                    fig_ex.add_trace(
                        go.Scatter(
                            x=ex_ax,
                            y=results["ex_profiles"][:, r],
                            mode="lines",
                            name=c_labels[r],
                            line=dict(width=2.5),
                        )
                    )
                fig_ex.update_layout(
                    title="Спектры возбуждения (Loading C)",
                    xaxis_title="Длина волны возбуждения Ex (нм)",
                    yaxis_title="Относительная интенсивность",
                    template="plotly_white",
                )
                st_plotly(fig_ex)

            st.subheader(
                "Вклады компонент по пробам (Концентрационная матрица Scores A)"
            )
            scores_df = pd.DataFrame(
                results["scores"],
                columns=["C1_Fulvic", "C2_Lignin_Humic", "C3_Protein"],
            )
            scores_df.insert(0, "Sample_ID", s_ids)

            fig_scores = px.bar(
                scores_df,
                x="Sample_ID",
                y=["C1_Fulvic", "C2_Lignin_Humic", "C3_Protein"],
                title="Парциальные интенсивности флуорофоров",
                labels={
                    "value": "Интенсивность Fmax",
                    "variable": "Флуорофор",
                },
                barmode="stack",
                template="plotly_white",
            )
            st_plotly(fig_scores)

            ml_export_df = pd.merge(df_indices, scores_df, on="Sample_ID")
            st.session_state["eem_ml_descriptors"] = ml_export_df

            st.download_button(
                label="💾 Экспорт оптических дескрипторов (CSV для PLS-DA)",
                data=ml_export_df.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_EEM_ML_Features.csv",
                mime="text/csv",
                key="eem_dl_ml_btn",
            )

# ==============================================================================
# МОДУЛЬ 3: DATA FUSION И КРОСС-МОДУЛЬНЫЙ ХЕМОМЕТРИЧЕСКИЙ АНАЛИЗ (PLS-DA)
# ==============================================================================
elif active_module == "🧬 ChemoSuite ML (Data Fusion)":
    try:
        import chemo_ml
    except ImportError:
        try:
            from modules import chemo_ml
        except ImportError:
            chemo_ml = None

    st.title("🧬 ChemoSuite ML: Интеграция данных и PLS-DA")
    st.caption(
        "Мультимодальная дискриминантная модель: FT-ICR MS (ячейки Ван-Кревелена) + EEM-PARAFAC (флуорофоры и индексы)"
    )

    if chemo_ml is None:
        st.error("⚠️ Файл `chemo_ml.py` не найден в рабочей директории приложения!")
        st.stop()

    st.subheader("1. Подготовка объединенной матрицы признаков (Data Staging)")

    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        if st.button("📥 Загрузить демо-бенчмарк (Байкал vs Шлам-лигнин)", type="primary", key="btn_ml_load_demo"):
            demo_df, demo_y = chemo_ml.generate_multimodal_benchmark()
            demo_df["Class_Target"] = demo_y
            st.session_state["fused_data"] = demo_df
            st.session_state["fused_source"] = "Синтетический бенчмарк (14 проб)"
            st.rerun()

    fused_df = st.session_state.get("fused_data", None)

    if fused_df is None:
        st.info("💡 Нажмите кнопку выше, чтобы загрузить согласованный датасет дескрипторов.")
    else:
        st.markdown(f"Активный датасет: **{st.session_state.get('fused_source', 'Пользовательский')}** | Число образцов: **{len(fused_df)}**")
        st.caption("Разметка классов: `1` — Шлам-лигнин, `0` — Фоновые воды.")

        edited_df = st.data_editor(
            fused_df,
            column_config={
                "Class_Target": st.column_config.SelectboxColumn(
                    "Класс (Target)",
                    options=[0, 1],
                    required=True,
                )
            },
            disabled=[c for c in fused_df.columns if c != "Class_Target"],
            use_container_width=True,
            key="ml_fused_editor",
        )

        st.markdown("---")
        st.subheader("2. Параметры моделирования PLS-DA")

        max_allowed_lvs = max(2, min(5, len(edited_df) - 1))
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            n_lvs = st.slider("Скрытые переменные (LVs):", min_value=2, max_value=max_allowed_lvs, value=2, key="pls_lvs_slider")
        with col_p2:
            use_block_scale = st.checkbox("Блочное шкалирование (1/√P)", value=True, key="pls_block_scale_chk")
        with col_p3:
            st.write("")
            run_pls = st.button("🚀 Обучить модель PLS-DA", type="primary", key="btn_run_pls_da")

        if run_pls:
            feat_cols = [c for c in edited_df.columns if c not in ["Sample_ID", "Class_Target"]]
            X_input = edited_df[feat_cols]
            y_input = edited_df["Class_Target"].values

            try:
                with st.spinner("Обучение модели PLS-DA и кросс-валидация..."):
                    res = chemo_ml.train_plsda_model(
                        X_input, y_input, n_components=n_lvs, block_scaling=use_block_scale
                    )

                # Безопасное чтение ключей без риска KeyError
                t1_vals = res.get("t1", res.get("scores_t1"))
                t2_vals = res.get("t2", res.get("scores_t2"))
                ell_x = res.get("ell_x", res.get("ellipse_x", np.array([])))
                ell_y = res.get("ell_y", res.get("ellipse_y", np.array([])))
                vip_df = res.get("VIP_df", res.get("vip_df", pd.DataFrame()))

                st.markdown("---")
                st.subheader("3. Результаты и статистическая валидация")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²X (Дисперсия спектров)", f"{res.get('R2X', 0.0):.1f}%")
                m2.metric("R²Y (Объяснение классов)", f"{res.get('R2Y', 0.0):.1f}%")
                q2_val = res.get("Q2", 0.0)
                m3.metric("Q² (Валидация LOO-CV)", f"{q2_val:.1f}%", delta="Надежна" if q2_val > 50 else "Слабая")
                m4.metric("Точность (Accuracy)", f"{res.get('Accuracy', 0.0):.1f}%")

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("##### Scores Plot ($t_1$ vs $t_2$) с эллипсом Хотеллинга 95%")
                    fig_sc = go.Figure()

                    if ell_x is not None and len(ell_x) > 0:
                        fig_sc.add_trace(go.Scatter(
                            x=ell_x, y=ell_y, mode="lines",
                            line=dict(dash="dot", color="#7f7f7f", width=1.5),
                            name="Hotelling T² (95%)", hoverinfo="skip"
                        ))

                    colors = ["#D62728" if int(y) == 1 else "#0020C2" for y in y_input]
                    fig_sc.add_trace(go.Scatter(
                        x=t1_vals, y=t2_vals, mode="markers+text",
                        text=edited_df["Sample_ID"], textposition="top center",
                        marker=dict(size=11, color=colors, line=dict(width=1, color="black"), opacity=0.85),
                        hovertext=[f"<b>{s}</b><br>Класс: {y}<br>t1: {t1:.2f}, t2: {t2:.2f}"
                                   for s, y, t1, t2 in zip(edited_df["Sample_ID"], y_input, t1_vals, t2_vals)],
                        hoverinfo="text", name="Образцы",
                    ))
                    fig_sc.update_layout(
                        xaxis_title="Скрытая переменная 1 (LV1)",
                        yaxis_title="Скрытая переменная 2 (LV2)",
                        plot_bgcolor="white", height=480,
                        margin=dict(l=40, r=30, t=30, b=40),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

                with col_g2:
                    st.markdown("##### Ключевые маркеры (VIP Scores > 1.0)")
                    if not vip_df.empty:
                        top_vip = vip_df.head(12)
                        fig_vip = px.bar(
                            top_vip, x="VIP", y="Descriptor", orientation="h",
                            color="VIP", color_continuous_scale="Reds",
                            text=top_vip["VIP"].apply(lambda v: f"{v:.2f}")
                        )
                        fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="Порог VIP=1.0")
                        fig_vip.update_layout(
                            yaxis=dict(autorange="reversed", title="Дескриптор"),
                            xaxis=dict(title="Индекс VIP"),
                            plot_bgcolor="white", height=480,
                            margin=dict(l=40, r=30, t=30, b=40),
                        )
                        st.plotly_chart(fig_vip, use_container_width=True)

                st.markdown("---")
                col_csv1, col_csv2 = st.columns([3, 1])
                with col_csv1:
                    st.write("**Ранжированный список дескрипторов по VIP:**")
                    st.dataframe(vip_df, use_container_width=True)
                with col_csv2:
                    st.download_button(
                        label="📥 Скачать VIP дескрипторы (CSV)",
                        data=vip_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_PLSDA_VIP.csv",
                        mime="text/csv",
                        key="dl_pls_vip_csv_btn",
                    )
            except ValueError as val_err:
                st.error(f"Ошибка входных данных: {val_err}")