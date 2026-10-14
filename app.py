# app.py — ChemoSuite Unified Analytical Platform
import inspect
import io
import math
import os
from pathlib import Path
import sys
import tempfile
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

# Регистрация папки modules в путях поиска Python
_MODULES_DIR = str(Path(__file__).resolve().parent / "modules")
if _MODULES_DIR not in sys.path:
    sys.path.insert(0, _MODULES_DIR)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        eem_core = None
        EEM_CORE_AVAILABLE = False

try:
    import chemo_ml
    CHEMO_ML_AVAILABLE = True
except ImportError:
    try:
        from modules import chemo_ml
        CHEMO_ML_AVAILABLE = True
    except ImportError:
        chemo_ml = None
        CHEMO_ML_AVAILABLE = False

try:
    import report_generator
    REPORT_GEN_AVAILABLE = True
except ImportError:
    try:
        from modules import report_generator
        REPORT_GEN_AVAILABLE = True
    except ImportError:
        report_generator = None
        REPORT_GEN_AVAILABLE = False

try:
    import project_io
    PROJECT_IO_AVAILABLE = True
except ImportError:
    try:
        from modules import project_io
        PROJECT_IO_AVAILABLE = True
    except ImportError:
        project_io = None
        PROJECT_IO_AVAILABLE = False

try:
    import chemo_pubchem
    CHEMO_PUBCHEM_AVAILABLE = True
except ImportError:
    try:
        from modules import chemo_pubchem
        CHEMO_PUBCHEM_AVAILABLE = True
    except ImportError:
        chemo_pubchem = None
        CHEMO_PUBCHEM_AVAILABLE = False

try:
    import ftir_core
    FTIR_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import ftir_core
        FTIR_CORE_AVAILABLE = True
    except ImportError:
        ftir_core = None
        FTIR_CORE_AVAILABLE = False

try:
    import nmr_core
    NMR_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import nmr_core
        NMR_CORE_AVAILABLE = True
    except ImportError:
        nmr_core = None
        NMR_CORE_AVAILABLE = False

try:
    import thermal_core
    THERMAL_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import thermal_core
        THERMAL_CORE_AVAILABLE = True
    except ImportError:
        thermal_core = None
        THERMAL_CORE_AVAILABLE = False

try:
    import sec_core
    SEC_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import sec_core
        SEC_CORE_AVAILABLE = True
    except ImportError:
        sec_core = None
        SEC_CORE_AVAILABLE = False

try:
    import elemental_core
    ELEMENTAL_CORE_AVAILABLE = True
except ImportError:
    try:
        from modules import elemental_core
        ELEMENTAL_CORE_AVAILABLE = True
    except ImportError:
        elemental_core = None
        ELEMENTAL_CORE_AVAILABLE = False

try:
    import cross_correlation
    CROSS_CORR_AVAILABLE = True
except ImportError:
    try:
        from modules import cross_correlation
        CROSS_CORR_AVAILABLE = True
    except ImportError:
        cross_correlation = None
        CROSS_CORR_AVAILABLE = False

try:
    import batch_screener
    BATCH_SCREENER_AVAILABLE = True
except ImportError:
    try:
        from modules import batch_screener
        BATCH_SCREENER_AVAILABLE = True
    except ImportError:
        batch_screener = None
        BATCH_SCREENER_AVAILABLE = False


# ==============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ И СТИЛИЗАЦИЯ
# ==============================================================================
st.set_page_config(
    page_title="ChemoSuite Platform | FT-ICR MS, EEM & Chemometrics",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Загрузка дизайн-системы Nordic Lab / Apple Pro Scientific
_css_file = Path(__file__).resolve().parent / "assets" / "style.css"
if _css_file.exists():
    with open(_css_file, encoding="utf-8") as _f_css:
        st.markdown(f"<style>{_f_css.read()}</style>", unsafe_allow_html=True)


STORAGE_DIR = "user_spectra"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ==============================================================================
# СЛОВАРЬ ПОЛНОЙ ДВУЯЗЫЧНОЙ ЛОКАЛИЗАЦИИ (I18N)
# ==============================================================================
T = {
    "ru": {
        # Платформа
        "header_title": "⚗️ ChemoSuite: Анализ РОВ и Шлам-лигнина",
        "header_subtitle": "Кафедра аналитической химии & Лаборатория природных гуминовых систем химфака МГУ",
        "nav_title": "### ⚗️ ChemoSuite",
        "nav_module_label": "Аналитический модуль:",
        "nav_cat_ms": "🔬 Масс-спектрометрия",
        "nav_cat_optics": "🌈 Оптическая спектроскопия",
        "nav_cat_resonance": "🧲 Резонанс и термика",
        "nav_cat_chrom": "📏 Хроматография и состав",
        "nav_cat_ml": "🧬 Многоблочный ML & Синтез",
        "proj_modal_btn": "💼 Управление проектом (.chemo)",
        "proj_modal_title": "💼 Управление сессией и проектом ChemoSuite",
        "proj_modal_desc": "Экспорт текущего рабочего пространства или загрузка ранее сохраненного мультимодального анализа.",
        "mod1_name": "🧪 NOM-Spectra FT-ICR MS",
        "mod2_name": "💡 EEM-PARAFAC (Флуоресценция)",
        "mod3_name": "🧬 ChemoSuite ML (Data Fusion)",
        "lang_label": "🌐 Язык / Language",
        # Модуль 1: FT-ICR MS
        "title": "🔬 Спектрометрия NOM сверхвысокого разрешения (FT-ICR MS)",
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
        "no_spectra_info": "Спектры не загружены. Перетащите файлы выше или импортируйте локальную папку.",
        "tab_stage1": "📥 1. Первичный спектр & Обработка",
        "tab_stage2": "🧬 2. Идентификация & Хемотипирование",
        "tab_stage3": "⚖️ 3. Сравнительный анализ & Сети",
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
        "vk_show_bioclasses": "Отображать биогеохимические полигоны (Липиды, Белки, Лигнин, Таннины, CAS)",
        "vk_bioclass_dist_title": "Распределение по биогеохимическим классам (%)",
        "vk_bioclass_col": "Биогеохимический класс",
        "vk_bioclass_pct": "Доля (%)",
        "batch_zip_expander": "📦 Пакетная обработка спектров (ZIP-архив)",
        "batch_zip_uploader": "Загрузите ZIP-архив со спектрами:",
        "batch_ion_mode": "Режим ионизации:",
        "batch_ppm_tol": "Допуск погрешности (ppm):",
        "batch_min_int": "Порог шума (мин. интенсивность):",
        "batch_run_btn": "🚀 Запустить пакетную обработку архива",
        "batch_success": "Успешно обработано спектров: {n}",
        "batch_table_title": "Сводная матрица дескрипторов (20 ячеек, индексы, пулы):",
        "batch_dl_csv": "📥 Скачать матрицу дескрипторов (chemo_fticr_block.csv)",
        "batch_transfer_ml_btn": "🧬 Передать дескрипторы в Модуль 3 (ChemoSuite ML)",
        "batch_transfer_success": "Дескрипторы переданы в сессию Модуля 3! Теперь вы можете собрать датасет в Модуле 3.",
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
        "tmds_mode_freq": "📊 Частотное распределение",
        "tmds_mode_network": "🕸️ Молекулярный граф реакций",
        "tmds_net_title": "Молекулярная сеть биогеохимических реакций (TMDS Network Graph)",
        "tmds_net_peaks": "Число пиков для графа:",
        "tmds_net_layout": "Пространственная раскладка:",
        "tmds_layout_spring": "Пружинная (Силовой граф / Spring)",
        "tmds_layout_chem": "Химические координаты (m/z vs Int)",
        "tmds_hubs_title": "👑 Ключевые узлы-хабы сети реакций (Network Reaction Hubs)",
        "tmds_dl_edges": "📥 Скачать ребра сети реакций (CSV)",
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
        # Модуль 2: EEM-PARAFAC
        "eem_title": "💡 Анализ флуоресценции: EEM-PARAFAC",
        "eem_subtitle": "Методология лаборатории природных гуминовых систем химфака МГУ | Анализ РОВ и шлам-лигнина",
        "eem_sidebar_hdr": "📁 Менеджер EEM матриц",
        "eem_src_label": "Источник оптических данных:",
        "eem_src_synth": "Синтетический бенчмарк (Байкал / Лигнин)",
        "eem_src_upload": "Загрузка файлов (CSV/TXT/DAT)",
        "eem_uploader_label": "Загрузите таблицы EEM:",
        "eem_preproc_hdr": "Параметры предобработки",
        "eem_clean_scatter": "Удалять Рэлеевское рассеяние 1 и 2 порядка",
        "eem_norm_raman": "Нормировка на Raman пик воды (R.U.)",
        "eem_palette_label": "Цветовая шкала EEM:",
        "eem_err_missing": "⚠️ Файл `eem_core.py` не обнаружен! Поместите файл `eem_core.py` в ту же директорию.",
        "eem_warn_no_data": "Загрузите файлы матриц флуоресценции или выберите синтетический бенчмарк.",
        "eem_tab1": "📊 Экран 1: 2D Контурная карта EEM",
        "eem_tab2": "📈 Экран 2: Спектральные индексы (FI, HIX, SUVA)",
        "eem_tab3": "🧬 Экран 3: Моделирование PARAFAC",
        "eem_select_sample": "Выберите пробу:",
        "eem_contour_title": "Контурная карта флуоресценции: {name}",
        "eem_ex_axis": "Длина волны возбуждения, Ex (нм)",
        "eem_em_axis": "Длина волны эмиссии, Em (нм)",
        "eem_int_label": "Интенсивность",
        "eem_indices_title": "Оптические хемометрические индексы",
        "eem_dl_indices": "📥 Скачать индексы (CSV)",
        "eem_parafac_title": "Неотрицательное трехкомпонентное PARAFAC-разложение",
        "eem_parafac_spinner": "Идет расчет модели PARAFAC...",
        "eem_m_exp_var": "Объясненная дисперсия",
        "eem_m_corcondia": "CORCONDIA (Диагностика)",
        "eem_corcondia_ok": "Модель адекватна",
        "eem_corcondia_warn": "Внимание",
        "eem_m_components": "Число флуорофоров",
        "eem_m_comp_val": "3 компонента",
        "eem_comp_c1": "C1: Фульвоподобный",
        "eem_comp_c2": "C2: Гуминовый (Лигнин)",
        "eem_comp_c3": "C3: Белковоподобный",
        "eem_em_prof_title": "Спектры эмиссии (Loading B)",
        "eem_ex_prof_title": "Спектры возбуждения (Loading C)",
        "eem_rel_int": "Относительная интенсивность",
        "eem_scores_title": "Вклады компонент по пробам (Концентрационная матрица Scores A)",
        "eem_scores_chart_title": "Парциальные интенсивности флуорофоров",
        "eem_dl_ml_btn": "💾 Экспорт оптических дескрипторов (CSV для PLS-DA)",
        "eem_parafac_select_tables": "Таблицы EEM для включения в модель PARAFAC:",
        "eem_parafac_min_samples": "⚠️ Для построения 3D-тензора выберите как минимум 2 таблицы EEM (рекомендуется 3+).",
        "eem_parafac_inspect_hdr": "🔍 Детальная диагностика аппроксимации для выбранной таблицы EEM",
        "eem_parafac_inspect_sample": "Выберите таблицу EEM для проверки аппроксимации:",
        "eem_parafac_orig": "Исходная EEM",
        "eem_parafac_model": "Модель PARAFAC",
        "eem_parafac_res": "Карта остатков (Невязки)",
        "eem_splithalf_expander": "🔬 Сплит-хаф валидация (Split-Half Validation)",
        "eem_splithalf_btn": "🚀 Запустить Split-Half анализ",
        "eem_splithalf_need_samples": "Для сплит-хаф анализа необходимо минимум 4 образца.",
        "eem_splithalf_spinner": "Выполняется сплит-хаф разбиение выборки и расчет коэффициентов конгруэнтности Такера (TCC)...",
        "eem_splithalf_success": "✅ Модель PARAFAC стабильна и валидирована (Mean TCC = {tcc:.3f} ≥ 0.90)",
        "eem_splithalf_fail": "⚠️ Модель PARAFAC нестабильна (Mean TCC = {tcc:.3f} < 0.90). Возможно перефакторизована.",
        "eem_splithalf_tcc_title": "Коэффициенты конгруэнтности Такера (TCC) по компонентам:",
        "eem_openfluor_expander": "🏷️ Идентификация по спектральной библиотеке OpenFluor",
        "eem_openfluor_tbl_title": "Результаты сопоставления с эталонной библиотекой OpenFluor:",
        "eem_openfluor_dl_btn": "📥 Скачать результаты идентификации (CSV)",
        # Модуль 4: Колебательная спектроскопия (FTIR)
        "mod4_name": "🔬 Колебательная спектроскопия (FTIR)",
        "ftir_title": "🔬 Колебательная спектроскопия (ATR-FTIR & Raman)",
        "ftir_subtitle": "Хемометрический анализ функционального состава, полос лигнина и деконволюция связей C=O / C=C",
        "ftir_err_missing": "⚠️ Файл ftir_core.py не найден в рабочей директории приложения!",
        "ftir_sidebar_hdr": "📁 Менеджер ИК-спектров",
        "ftir_src_label": "Источник спектров FTIR:",
        "ftir_src_upload": "Загрузить файлы (.csv, .dpt, .txt)",
        "ftir_src_synth": "Эталонный бенчмарк (12 проб)",
        "ftir_uploader_label": "Загрузите файлы спектров FTIR:",
        "ftir_preproc_hdr": "Параметры предобработки",
        "ftir_baseline_method": "Коррекция базовой линии:",
        "ftir_base_als": "AsLS (Asymmetric Least Squares)",
        "ftir_base_rubber": "Rubberband (Резиновая лента)",
        "ftir_base_linear": "Линейная",
        "ftir_base_none": "Без коррекции фона",
        "ftir_norm_label": "Спектральная нормировка:",
        "ftir_norm_snv": "SNV (Standard Normal Variate)",
        "ftir_norm_minmax": "Min-Max [0, 1]",
        "ftir_norm_vec": "Векторная (L2 норма)",
        "ftir_norm_none": "Без нормировки",
        "ftir_tab1_title": "📈 Спектральный профиль и пики",
        "ftir_tab2_title": "✂️ Деконволюция (1500–1800 см⁻¹)",
        "ftir_tab3_title": "🏷️ Идентификация по библиотеке (HQI)",
        "ftir_tab4_title": "📊 Хемометрические индексы и экспорт",
        "ftir_no_samples": "Нет загруженных спектров FTIR. Загрузите файлы в боковой панели или выберите 'Эталонный бенчмарк'.",
        "ftir_active_sample_label": "Активный спектр:",
        "ftir_show_all": "Наложить все спектры сессии",
        "ftir_deriv_label": "Производная спектроскопия:",
        "ftir_deriv_0": "Обычный спектр (d⁰)",
        "ftir_deriv_1": "1-я производная (d¹A)",
        "ftir_deriv_2": "2-я производная (d²A)",
        "ftir_peaks_expander": "🏷️ Автопоиск пиков и аннотация характеристических полос",
        "ftir_peaks_prominence": "Порог выраженности пика (Prominence):",
        "ftir_peaks_dist": "Мин. расстояние между пиками (см⁻¹):",
        "ftir_peaks_show_markers": "Отображать метки полос на спектральном графике",
        "ftir_peaks_tbl_title": "Обнаруженные характеристические полосы образца:",
        "ftir_peaks_dl_btn": "📥 Скачать таблицу полос (CSV)",
        "ftir_lib_hdr": "Сопоставление со спектральной библиотекой эталонов (IHSS, Лигнин, Полимеры)",
        "ftir_lib_sample_select": "Образец для спектральной идентификации:",
        "ftir_lib_top_n": "Число топ-совпадений:",
        "ftir_lib_btn": "🔍 Идентифицировать по библиотеке эталонов",
        "ftir_lib_tbl_title": "Ранжированный список совпадений (HQI / Cosine / Pearson):",
        "ftir_lib_overlay_title": "Спектральное наложение: Образец vs Топ-эталон",
        "ftir_lib_sample_trace": "Исследуемый образец",
        "ftir_lib_ref_trace": "Эталонный стандарт",
        "ftir_deconv_sample": "Образец для деконволюции:",
        "ftir_deconv_range": "Диапазон деконволюции (см⁻¹):",
        "ftir_deconv_n_peaks": "Число Гауссовых полос:",
        "ftir_deconv_btn": "🚀 Выполнить деконволюцию полос",
        "ftir_deconv_r2": "Качество подгонки R²:",
        "ftir_deconv_rmse": "RMSE невязки:",
        "ftir_deconv_tbl_title": "Параметры подогнанных Гауссовых полос:",
        "ftir_sync_ml_btn": "🧬 Синхронизировать дескрипторы с Модулем 3 (ML Data Fusion)",
        "ftir_sync_success": "Дескрипторы FTIR ({n} образцов) успешно синхронизированы с шиной данных Модуля 3!",
        "ftir_dl_csv": "📥 Скачать дескрипторы FTIR (CSV)",
        "ftir_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        "ml_up_ftir": "Таблица дескрипторов FTIR (CSV):",
        # Модуль 5: ЯМР-спектроскопия (13C / 1H NMR)
        "mod5_name": "🧪 ЯМР-спектроскопия (13C / 1H NMR)",
        "nmr_title": "🧪 Спектроскопия ядерного магнитного резонанса (13C / 1H NMR Studio)",
        "nmr_subtitle": "Количественный анализ функционального состава углеродного скелета (IHSS), деконволюция и расчет ароматичности fa",
        "nmr_err_missing": "⚠️ Вычислительное ядро nmr_core.py не найдено в директории modules!",
        "nmr_sidebar_hdr": "📁 Менеджер спектров ЯМР",
        "nmr_src_label": "Источник спектров ЯМР:",
        "nmr_src_upload": "Загрузить файлы (.csv, .txt, .jdx)",
        "nmr_src_synth": "Эталонный бенчмарк (12 проб)",
        "nmr_uploader_label": "Загрузите спектры 13C ЯМР:",
        "nmr_preproc_hdr": "Параметры предобработки",
        "nmr_baseline_method": "Коррекция базовой линии:",
        "nmr_base_asls": "AsLS (Asymmetric Least Squares)",
        "nmr_base_linear": "Линейная",
        "nmr_base_poly": "Полиномиальная",
        "nmr_base_none": "Без коррекции фона",
        "nmr_norm_label": "Нормировка спектра:",
        "nmr_norm_area": "Интегральная (Площадь = 100%)",
        "nmr_norm_max": "Максимальный пик (Max = 1.0)",
        "nmr_norm_minmax": "Min-Max [0, 1]",
        "nmr_norm_none": "Без нормировки",
        "nmr_smooth_label": "Сглаживание Савицкого-Голея",
        "nmr_recal_label": "Калибровка шкалы ppm:",
        "nmr_recal_none": "Без смещения (0.0 ppm)",
        "nmr_recal_dmso": "По растворителю DMSO-d6 (39.5 ppm)",
        "nmr_recal_tms": "По эталону TMS / DSS (0.0 ppm)",
        "nmr_tab1_title": "📈 Спектральный профиль и интегральные зоны",
        "nmr_tab2_title": "📊 Распределение углеродных пулов",
        "nmr_tab3_title": "🔍 Деконволюция и спектральная библиотека",
        "nmr_tab4_title": "🧬 Хемометрические индексы и экспорт в ML",
        "nmr_no_samples": "Нет загруженных спектров ЯМР. Загрузите файлы в боковой панели или выберите 'Эталонный бенчмарк'.",
        "nmr_active_sample_label": "Активный спектр:",
        "nmr_show_all": "Наложить все спектры сессии",
        "nmr_show_regions": "Подсвечивать канонические зоны углерода (IHSS)",
        "nmr_peaks_expander": "🏷️ Автопоиск резонансов и структурная аннотация",
        "nmr_peaks_prominence": "Порог выраженности пика (Prominence):",
        "nmr_peaks_dist": "Мин. расстояние между пиками (ppm):",
        "nmr_peaks_show_markers": "Отображать метки резонансов на спектре",
        "nmr_peaks_tbl_title": "Обнаруженные химические резонансы образца:",
        "nmr_peaks_dl_btn": "📥 Скачать таблицу пиков (CSV)",
        "nmr_deconv_sample": "Образец для деконволюции:",
        "nmr_deconv_n_peaks": "Число Гауссовых полос (108–165 ppm):",
        "nmr_deconv_btn": "🚀 Выполнить деконволюцию ароматической зоны",
        "nmr_deconv_r2": "Качество подгонки R²:",
        "nmr_deconv_rmse": "RMSE невязки:",
        "nmr_deconv_tbl_title": "Параметры подогнанных полос ароматического региона:",
        "nmr_lib_hdr": "Сопоставление со спектральной библиотекой эталонов (IHSS и лигнины)",
        "nmr_lib_sample_select": "Образец для идентификации:",
        "nmr_lib_top_n": "Число топ-совпадений:",
        "nmr_lib_btn": "🔍 Идентифицировать по библиотеке эталонов",
        "nmr_lib_tbl_title": "Ранжированный список совпадений (HQI Cosine %):",
        "nmr_lib_overlay_title": "Спектральное наложение: Образец vs Топ-эталон",
        "nmr_lib_sample_trace": "Исследуемый образец",
        "nmr_lib_ref_trace": "Эталонный стандарт",
        "nmr_sync_ml_btn": "🧬 Синхронизировать дескрипторы с Модулем 3 (Data Fusion)",
        "nmr_sync_success": "Дескрипторы ЯМР ({n} образцов) успешно переданы в Модуль 3!",
        "nmr_dl_csv": "📥 Скачать дескрипторы ЯМР (CSV)",
        "nmr_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        "ml_up_nmr": "Таблица дескрипторов NMR (CSV):",
        "ml_up_thermal": "Таблица дескрипторов Thermal (CSV):",
        # Модуль 6: Термический анализ и пиролиз (Py-GC-MS & TGA/DSC)
        "mod6_name": "🌡️ Термолиз и пиролиз (Py-GC-MS & TGA/DSC)",
        "thermal_title": "🌡️ Термический анализ и аналитический пиролиз (Py-GC-MS & TGA/DSC)",
        "thermal_subtitle": "Молекулярная термодеструкция РОВ и лигнина: пиролитические маркеры H/G/S, термогравиметрия (TGA/DTG) и кинетические зоны",
        "thermal_err_missing": "⚠️ Вычислительное ядро thermal_core.py не найдено в директории modules!",
        "thermal_sidebar_hdr": "📁 Менеджер термолиза и TGA",
        "thermal_src_label": "Источник термических данных:",
        "thermal_src_upload": "Загрузить файлы (Py-GC-MS & TGA)",
        "thermal_src_synth": "Эталонный бенчмарк (12 проб)",
        "thermal_up_py_label": "Загрузите файлы Py-GC-MS (.csv, .txt):",
        "thermal_up_tga_label": "Загрузите файлы TGA (.csv, .txt):",
        "thermal_tab1_title": "📈 Хроматограммы Py-GC-MS и мономеры",
        "thermal_tab2_title": "🔥 Термограммы TGA / DTG и температурные зоны",
        "thermal_tab3_title": "🧬 Хемометрические индексы (S/G, L/C, TI)",
        "thermal_tab4_title": "🌐 Мультимодальная интеграция и экспорт",
        "thermal_no_samples": "Нет загруженных термических данных. Загрузите файлы в боковой панели или выберите 'Эталонный бенчмарк'.",
        "thermal_active_sample": "Активный образец:",
        "thermal_peaks_hdr": "Идентифицированные пиролизные продукты (Библиотека биомаркеров)",
        "thermal_peaks_dl_btn": "📥 Скачать таблицу пиков (CSV)",
        "thermal_dist_pie_title": "Распределение классов продуктов пиролиза",
        "thermal_monomers_bar_title": "Основные мономеры лигнина и биомаркеры (Площадь %)",
        "thermal_tga_title": "Термогравиметрическая кривая (TGA) и производная (DTG)",
        "thermal_tga_zones_chk": "Отображать температурные зоны деструкции (I–IV)",
        "thermal_tga_dtg_chk": "Показывать производную кривую DTG (-dm/dT)",
        "thermal_kpi_sg": "Индекс S/G (Сирингил / Гваяцил)",
        "thermal_kpi_hg": "Индекс H/G (п-Гидроксифенил / Гваяцил)",
        "thermal_kpi_lc": "Индекс L/C (Лигнин / Углеводы)",
        "thermal_kpi_ti": "Индекс термостабильности TI (Зона III / Зона II)",
        "thermal_kpi_t50": "Температура полураспада T₅₀%",
        "thermal_kpi_residue": "Остаточная масса при 800°C",
        "thermal_sync_ml_btn": "🧬 Синхронизировать дескрипторы с Модулем 3 (ChemoSuite ML)",
        "thermal_sync_success": "Дескрипторы термического анализа ({n} образцов) успешно переданы в Модуль 3!",
        "thermal_dl_csv": "📥 Скачать дескрипторы Thermal (CSV)",
        "thermal_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        "ml_up_sec": "Таблица дескрипторов SEC/GPC (CSV):",
        # Модуль 7: Гель-проникающая и эксклюзионная хроматография (SEC / GPC)
        "mod7_name": "📏 Эксклюзионная хроматография (SEC / GPC)",
        "sec_title": "📏 Гель-проникающая и эксклюзионная хроматография (SEC / GPC Studio)",
        "sec_subtitle": "Молекулярно-массовые распределения (ММР), полидисперсность (Đ = Mw/Mn) и экологическое фракционирование РОВ и лигнина",
        "sec_err_missing": "⚠️ Вычислительное ядро sec_core.py не найдено в директории modules!",
        "sec_sidebar_hdr": "📁 Менеджер SEC / GPC",
        "sec_src_label": "Источник хроматографических данных:",
        "sec_src_upload": "Загрузить файлы SEC (.csv, .txt)",
        "sec_src_synth": "Эталонный бенчмарк (12 проб)",
        "sec_up_label": "Загрузите хроматограммы SEC (.csv, .txt):",
        "sec_calib_preset_label": "Стандарт калибровки колонки:",
        "sec_flow_rate_label": "Скорость потока (мл/мин):",
        "sec_tab1_title": "📈 Хроматограмма элюирования и пики",
        "sec_tab2_title": "📊 Молекулярно-массовое распределение (ММР)",
        "sec_tab3_title": "⚖️ Полидисперсность и фракционирование",
        "sec_tab4_title": "🌐 Мультимодальная интеграция и экспорт",
        "sec_no_samples": "Нет загруженных хроматограмм SEC. Загрузите файлы в боковой панели или выберите 'Эталонный бенчмарк'.",
        "sec_active_sample": "Активный образец:",
        "sec_peaks_hdr": "Идентифицированные молекулярные фракции и пики",
        "sec_peaks_dl_btn": "📥 Скачать таблицу пиков (CSV)",
        "sec_xaxis_label": "Шкала оси X:",
        "sec_xaxis_time": "Время удерживания (мин)",
        "sec_xaxis_vol": "Элюционный объем (мл)",
        "sec_kpi_mn": "Mn (Среднечисловая M)",
        "sec_kpi_mw": "Mw (Среднемассовая M)",
        "sec_kpi_mz": "Mz (z-Средняя M)",
        "sec_kpi_mp": "Mp (Пиковая M)",
        "sec_kpi_pdi": "Полидисперсность Đ (Mw/Mn)",
        "sec_kpi_hmw_ratio": "Отношение HMW / LMW",
        "sec_kpi_frac1": "Фракция I (>20 кДа, HMW)",
        "sec_kpi_frac4": "Фракция IV (<1 кДа, LMW)",
        "sec_sync_ml_btn": "🧬 Синхронизировать дескрипторы с Модулем 3 (ChemoSuite ML)",
        "sec_sync_success": "Дескрипторы SEC/GPC ({n} образцов) успешно переданы в Модуль 3!",
        "sec_dl_csv": "📥 Скачать дескрипторы SEC (CSV)",
        "sec_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        "ml_up_sec": "Таблица дескрипторов SEC/GPC (CSV):",
        "ml_up_elemental": "Таблица дескрипторов Elemental CHNS-O (CSV):",
        # Модуль 8: Элементный анализ и валовая стехиометрия (CHNS-O)
        "mod8_name": "⚗️ Элементный анализ и стехиометрия (CHNS-O)",
        "elem_title": "⚗️ Элементный микроанализ и валовая стехиометрия (CHNS-O)",
        "elem_subtitle": "Атомные соотношения (H/C, O/C, N/C, S/C), диаграммы Ван Кревелена, степень окисления углерода (NOSC) и валидация FT-ICR MS",
        "elem_err_missing": "⚠️ Вычислительное ядро elemental_core.py не найдено в директории modules!",
        "elem_sidebar_hdr": "📁 Менеджер CHNS-O",
        "elem_src_label": "Источник данных CHNS-O:",
        "elem_src_upload": "Загрузить файл CHNS-O (.csv, .txt)",
        "elem_src_synth": "Эталонный бенчмарк (12 проб)",
        "elem_up_label": "Загрузите таблицу элементного состава (.csv, .tsv):",
        "elem_tab1_title": "📊 Валовая стехиометрия и индексы",
        "elem_tab2_title": "🗺️ Валовая диаграмма Ван Кревелена",
        "elem_tab3_title": "🔬 Кросс-валидация со спектрами FT-ICR MS",
        "elem_tab4_title": "🧬 Экспорт дескрипторов и синхронизация ML",
        "elem_no_samples": "Нет загруженных данных CHNS-O. Загрузите файл или выберите 'Эталонный бенчмарк'.",
        "elem_active_sample": "Активный образец:",
        "elem_kpi_hc": "H/C (Атомное)",
        "elem_kpi_oc": "O/C (Атомное)",
        "elem_kpi_nc": "N/C (Атомное)",
        "elem_kpi_sc": "S/C (Атомное)",
        "elem_kpi_nosc": "NOSC (Окисление C)",
        "elem_kpi_ai": "AI_bulk (Ароматичность)",
        "elem_sync_ml_btn": "🧬 Синхронизировать дескрипторы с Модулем 3 (ChemoSuite ML)",
        "elem_sync_success": "Дескрипторы CHNS-O ({n} образцов) успешно переданы в Модуль 3!",
        "elem_dl_csv": "📥 Скачать дескрипторы CHNS-O (CSV)",
        "elem_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        # Модуль 3: ChemoSuite ML
        "ml_title": "🧬 ChemoSuite ML: Интеграция данных и PLS-DA",
        "ml_subtitle": "Мультимодальная дискриминантная модель: FT-ICR MS (ячейки Ван-Кревелена) + EEM-PARAFAC (флуорофоры и индексы)",
        "ml_err_missing": "⚠️ Файл `chemo_ml.py` не найден в рабочей директории приложения!",
        "ml_sec1_title": "1. Подготовка объединенной матрицы признаков (Data Staging)",
        "ml_load_demo_btn": "📥 Загрузить демо-бенчмарк (Байкал vs Шлам-лигнин)",
        "ml_demo_source_name": "Синтетический бенчмарк (14 проб)",
        "ml_custom_source_name": "Пользовательский",
        "ml_active_dataset_info": "Активный датасет: **{src}** | Число образцов: **{n}**",
        "ml_classes_caption": "Разметка классов: `1` — Шлам-лигнин, `0` — Фоновые воды.",
        "ml_col_class_target": "Класс (Target)",
        "ml_sec2_title": "2. Параметры моделирования PLS-DA",
        "ml_lvs_slider": "Скрытые переменные (LVs):",
        "ml_block_scale": "Блочное шкалирование (1/√P)",
        "ml_run_btn": "🚀 Обучить модель PLS-DA",
        "ml_spinner": "Обучение модели PLS-DA и кросс-валидация...",
        "ml_sec3_title": "3. Результаты и статистическая валидация",
        "ml_m_r2x": "R²X (Дисперсия спектров)",
        "ml_m_r2y": "R²Y (Объяснение классов)",
        "ml_m_q2": "Q² (Валидация LOO-CV)",
        "ml_q2_ok": "Надежна",
        "ml_q2_warn": "Слабая",
        "ml_m_acc": "Точность (Accuracy)",
        "ml_scores_chart_title": "Scores Plot (t₁ vs t₂) с эллипсом Хотеллинга 95%",
        "ml_hotelling_name": "Hotelling T² (95%)",
        "ml_class1_label": "Шлам-лигнин (Класс 1)",
        "ml_class0_label": "Фон / Байкал (Класс 0)",
        "ml_vip_chart_title": "Ключевые маркеры (VIP Scores > 1.0)",
        "ml_vip_cutoff": "Порог VIP=1.0",
        "ml_vip_tbl_title": "Ранжированный список дескрипторов по VIP:",
        "ml_dl_vip_btn": "📥 Скачать VIP дескрипторы (CSV)",
        "ml_warn_input": "Ошибка входных данных: {err}",
        "ml_info_no_data": "💡 Нажмите кнопку выше, чтобы загрузить согласованный датасет дескрипторов.",
        "ml_src_session": "🔄 Собрать из сессии (Модули 1 и 2)",
        "ml_src_files": "📂 Загрузка внешних CSV файлов",
        "ml_src_batch": "⚡ Пакетный скрининг папок (Batch Screening)",
        "ml_src_benchmark": "🧪 Синтетический бенчмарк",
        "batch_screener_title": "🚀 Автономный пакетный скрининг директорий (8 модальностей)",
        "batch_screener_desc": "Интеллектуальное сканирование файлов спектров и матриц, автоматическая детекция аналитических методов (FT-ICR MS, EEM, UV-Vis, FTIR, NMR, TGA/Py-GC-MS, SEC/GPC, CHNS-O), извлечение дескрипторов и сборка единой мультимодальной матрицы Data Staging в один клик.",
        "batch_source_type_label": "Источник файлов для скрининга:",
        "batch_src_folder": "📁 Сканировать локальную директорию на сервере / компьютере",
        "batch_src_upload": "📤 Загрузить группу файлов через браузер",
        "batch_dir_label": "Путь к директории спектров:",
        "batch_recursive_label": "Рекурсивное сканирование всех поддиректорий",
        "batch_upload_label": "Перетащите файлы спектров (CSV, TSV, TXT, DX):",
        "batch_run_btn": "🚀 Запустить автономный пакетный скрининг",
        "batch_files_count": "Всего файлов",
        "batch_success_count": "Успешно распознано",
        "batch_samples_count": "Собрано образцов",
        "batch_modalities_count": "Аналитических модальностей",
        "batch_manifest_title": "📋 Сводный манифест распознанных файлов и модальностей:",
        "batch_use_ml_btn": "🧬 Использовать собранную матрицу в ChemoSuite ML",
        "batch_dl_zip_btn": "📥 Скачать полный пакет результатов (ZIP)",
        "batch_no_files_found": "В указанной директории не найдено поддерживаемых аналитических файлов.",
        "batch_success_msg": "Пакетный скрининг завершен! Обработано файлов: {n_files}, собрано образцов: {n_samples}.",
        "ml_btn_assemble_session": "📥 Собрать объединенную матрицу из открытых спектров",
        "ml_session_status": "Спектров FT-ICR с формулами: **{n_ms}** | Оптических проб EEM: **{n_eem}** | Спектров УФ-Вид: **{n_uv}**",
        "ml_session_need_more": "⚠️ Для интеграции из сессии необходимо приписать формулы в Модуле 1 либо рассчитать оптические дескрипторы (EEM / UV-Vis) в Модуле 2.",
        "ml_uploader_mode": "Режим загрузки файлов:",
        "ml_upload_two": "Отдельные файлы по блокам (FT-ICR / EEM / UV-Vis)",
        "ml_upload_single": "Один общий файл с дескрипторами",
        "ml_up_ms": "Таблица дескрипторов FT-ICR MS (CSV):",
        "ml_up_eem": "Таблица дескрипторов EEM-PARAFAC (CSV):",
        "ml_up_uv": "Таблица дескрипторов UV-Vis (CSV):",
        "ml_up_single": "Общая таблица дескрипторов (CSV):",
        "ml_strategy_label": "Стратегия слияния блоков (Data Fusion):",
        "ml_strat_low": "Low-Level Fusion (Масштабирование 1/√P)",
        "ml_strat_mid": "Mid-Level Fusion (PCA сжатие FT-ICR + EEM)",
        "ml_perm_chk": "Запустить пермутационный тест валидации (H₀)",
        "ml_tab_scores": "🎯 Проекция Scores (t₁ vs t₂)",
        "ml_tab_vips": "📊 Маркеры (VIP Scores)",
        "ml_tab_val": "📈 Валидация и диагностика",
        "ml_tab_blocks": "🧬 Вклад аналитических блоков",
        "ml_tab_crosscorr": "🌐 Кросс-модальный атлас",
        "ml_crosscorr_title": "🌐 Кросс-модальный корреляционный атлас (Cross-Omics)",
        "ml_crosscorr_subtitle": "2D-гетероспектральная корреляция и выявление ко-вариации между аналитическими блоками",
        "ml_crosscorr_mode_select": "Режим анализа корреляций:",
        "ml_crosscorr_mode_pairs": "Попарная 2D-гетероспектральная корреляция (Блок A vs Блок B)",
        "ml_crosscorr_mode_global": "Глобальная мультиблочная матрица всех дескрипторов",
        "ml_crosscorr_block_a": "Аналитический блок А (Строки / Y):",
        "ml_crosscorr_block_b": "Аналитический блок B (Столбцы / X):",
        "ml_crosscorr_method": "Метод расчета:",
        "ml_crosscorr_threshold": "Минимальный порог значимости (|r|):",
        "ml_crosscorr_top_pairs_title": "🔝 Топ межблочных корреляций и ко-вариаций дескрипторов",
        "ml_crosscorr_dl_edges": "📥 Скачать граф связей (Edges CSV)",
        "ml_crosscorr_dl_top": "📥 Скачать таблицу топ-корреляций (CSV)",
        "ml_crosscorr_no_data": "Для расчета кросс-корреляций требуется не менее 2 аналитических блоков с несколькими образцами.",
        "ml_cm_title": "Матрица ошибок (Confusion Matrix)",
        "ml_perm_title": "Распределение пермутационного теста (H0: модель случайна)",
        "ml_donut_title": "Относительный вес аналитических блоков в модели (VIP²)",
        "ml_warn_two_classes": "⚠️ Для классификации в колонке Class_Target должны присутствовать как минимум два различных класса!",
        "ml_model_type_label": "Архитектура модели:",
        "ml_model_plsda": "PLS-DA (Мультикласс / Проекция LV)",
        "ml_model_oplsda": "OPLS-DA (Ортогональный фильтр шума / S-Plot)",
        "ml_model_splsda": "Sparse PLS-DA (sPLS-DA с L1-отбором биомаркеров)",
        "ml_spls_keep_slider": "Отбор признаков (число дескрипторов на компоненту):",
        "ml_spls_biomarkers_title": "🎯 Отобранная панель биомаркеров (L1 Sparse Features):",
        "ml_spls_weight_axis": "Абсолютный разреженный вес (|W|)",
        "ml_spls_table_title": "Панель отобранных биомаркеров",
        "ml_spls_selected_count": "отобрано",
        "ml_spls_dl_btn": "📥 Скачать разреженные биомаркеры (CSV)",
        "ml_ortho_slider": "Ортогональные компоненты (Orthogonal LVs):",
        "ml_scores_mode_label": "Пространство проекции Scores:",
        "ml_scores_2d": "2D Scores (LV1 vs LV2)",
        "ml_scores_3d": "3D Scores (LV1 vs LV2 vs LV3)",
        "ml_tab_splot": "💠 S-Plot (Маркеры OPLS-DA)",
        "ml_splot_title": "S-Plot: Ковариация p[1] vs Корреляция p(corr)[1]",
        "ml_m_r2x_pred": "R²X(pred)",
        "ml_m_r2x_ortho": "R²X(ortho)",
        "ml_warn_opls_binary": "ℹ️ OPLS-DA оптимизирован для двух контрастных классов. Для 3+ классов выберите PLS-DA.",
        "ml_dl_passport_btn": "📑 Скачать полный аналитический паспорт (.xlsx)",
        "ml_dl_html_passport_btn": "📄 Скачать Экологический паспорт (Интерактивный HTML / PDF)",
        "ml_html_passport_desc": "Автономный интерактивный экологический паспорт: экспертное заключение, матрица 8 методов, VIP-биомаркеры, кросс-корреляции и печать в PDF.",
        "ml_preview_passport_expander": "👁️ Интерактивный предпросмотр экологического паспорта",
        "ml_sens_label": "Чувствительность (Sensitivity):",
        "ml_spec_label": "Специфичность (Specificity):",
        "ml_bal_acc_label": "Сбалансированная точность (Balanced Accuracy):",
        "ml_acc_label": "Общая точность (Accuracy):",
        "ml_mode_multiclass": "Режим: Мультиклассовая классификация ({n} класса)",
        "ml_report_lang_label": "Язык отчета / Report Language:",
        "ml_passport_desc": "Многостраничный Excel-паспорт: Сводка, матрица признаков, 20 ячеек FT-ICR, EEM, UV-Vis, VIP-биомаркеры, S-Plot.",
        "proj_expander_title": "💾 Проект платформы (.chemo)",
        "proj_save_header": "Сохранить текущую сессию",
        "proj_save_btn": "💾 Скачать проект (.chemo)",
        "proj_load_header": "Загрузить проект",
        "proj_load_label": "Файл проекта (.chemo / .zip):",
        "proj_restore_btn": "📥 Восстановить рабочую сессию",
        "proj_restored_success": "Сессия ChemoSuite успешно восстановлена!",
        "proj_summary_spec": "Спектров FT-ICR: {n}",
        "proj_summary_eem": "EEM: {val}",
        "proj_summary_uv": "UV-Vis: {val}",
        "proj_summary_fused": "Data Fusion: {val}",
        "proj_summary_model": "Модель: {val}",
        #UV-Spectre
        "eem_uv_uploader_label": "Загрузить УФ-Вид спектры поглощения (CSV/TXT):",
        "eem_doc_expander": "Параметры DOC для расчета SUVA254",
        "eem_default_doc": "Концентрация DOC по умолчанию (мг C / л):",
        "eem_tab4": "📉 Экран 4: УФ-Вид спектрофотометрия (UV-Vis)",
        "eem_uv_title": "Спектрофотометрия поглощения УФ-Вид",
        "eem_uv_select_sample": "Выберите УФ-Вид спектр:",
        "eem_uv_plot_title": "УФ-Вид спектр поглощения A(λ): {name}",
        "eem_uv_tbl_title": "Оптические дескрипторы УФ-Вид (A254, E2/E3, S_R, SUVA)",
        "eem_dl_uv_indices": "📥 Скачать дескрипторы УФ-Вид (CSV)",
        "eem_uv_no_data": "Загрузите файлы спектров поглощения УФ-Вид в боковой панели.",
        # Горизонт 3: Хемоинформатика PubChem & Векторные пути трансформаций
        "pubchem_inspect_expander": "🧪 Хемоинформатика PubChem (Структурная идентификация брутто-формулы)",
        "pubchem_select_formula": "Выберите формулу из спектра:",
        "pubchem_custom_input": "Или введите произвольную формулу (C_xH_yO_z...):",
        "pubchem_search_btn": "🔍 Искать в базе данных PubChem",
        "pubchem_searching": "Запрос к PubChem PUG-REST API...",
        "pubchem_res_name": "Каноническое название:",
        "pubchem_res_iupac": "IUPAC название:",
        "pubchem_res_class": "Химический класс:",
        "pubchem_res_subclass": "Биогеохимический субкласс:",
        "pubchem_res_smiles": "SMILES:",
        "pubchem_res_cid": "PubChem CID:",
        "pubchem_link_btn": "🌐 Открыть в PubChem",
        "pubchem_chembl_btn": "🧪 Найти в ChEMBL",
        "pubchem_hmdb_btn": "🧬 Найти в HMDB (Метаболом)",
        "pubchem_isomers_label": "Структурные изомеры в PubChem:",
        "pubchem_card_header": "Хемоинформационный паспорт изомера #{i}: {title}",
        "vk_inspect_title": "🔬 Хемоинформатическая идентификация точки (PubChem / ChEMBL / HMDB)",
        "vk_inspect_caption": "Выберите формулу с диаграммы Ван-Кревелена для мгновенного извлечения структурных формул, SMILES и хемотипического класса.",
        "vk_inspect_select": "Формула с диаграммы:",
        "tmds_mode_pathways": "🛣️ Трассировка путей реакций (Reaction Pathways)",
        "tmds_flux_header": "📊 Геохимические векторные потоки (Vector Flux Analysis)",
        "tmds_flux_ox_decarb": "Окисление / Декарбоксилирование (O/CO₂)",
        "tmds_flux_alkyl": "Доля алкилирования / гомологии (% CH₂)",
        "tmds_flux_sulf": "Индекс сульфирования (% SO₃)",
        "tmds_flux_hydr": "Гидратация / Окисление (H₂O/O)",
        "tmds_path_source": "Исходный ион-предшественник (m/z):",
        "tmds_path_target": "Конечный продукт трансформации (m/z):",
        "tmds_path_depth": "Макс. число стадий (глубина пути):",
        "tmds_path_tol": "Погрешность TMDS (mDa):",
        "tmds_path_find_btn": "🛣️ Найти биогеохимические пути",
        "tmds_path_found": "Найдено допустимых путей трансформации: {n}",
        "tmds_path_not_found": "Путей между выбранными ионами в пределах заданной погрешности не найдено.",
        "tmds_dl_paths": "📥 Скачать пути трансформации (CSV)",
        "ml_pubchem_title": "🧪 Хемоинформатическая идентификация биомаркеров (PubChem)",
        "ml_pubchem_caption": "Автоматическое сопоставление формул биомаркеров с химическими структурами через PubChem REST API.",
        "ml_pubchem_btn": "🔍 Аннотировать биомаркеры в PubChem",
        "ml_pubchem_spinner": "Идентификация биомаркеров через PubChem API...",
        "ml_pubchem_dl_btn": "📥 Скачать аннотированные биомаркеры (CSV)",
        # Улучшения UX / Фичи
        "kmd_slicer_title": "🎯 Срез гомологической серии (KMD Slicer)",
        "kmd_slicer_enable": "Включить фильтрацию гомологического ряда по KMD",
        "kmd_target_val": "Целевой KMD (дефект массы):",
        "kmd_tol_val": "Допуск среза (± ΔKMD):",
        "kmd_slicer_count": "Выделено членов серии: **{n}** из {total} ({pct:.1f}%)",
        "kmd_dl_series": "📥 Скачать выделенный гомологический ряд (CSV)",
        "batch_desc_title": "📊 Пакетный экспорт дескрипторов всех спектров",
        "batch_desc_caption": "Сводная таблица биогеохимических индексов (Mn, Mw, H/C, O/C, DBE, AI, NOSC) и 20 ячеек Перминовой для всех идентифицированных спектров сессии.",
        "batch_desc_dl_csv": "📥 Экспорт сводной таблицы (CSV)",
        "batch_desc_dl_excel": "📊 Экспорт в Excel (.xlsx)",
        "batch_desc_no_data": "В текущей сессии еще нет спектров с приписанными брутто-формулами. Выполните приписывание хотя бы для одного образца.",
    },
    "en": {
        # Platform
        "header_title": "⚗️ ChemoSuite: NOM & Slime-Lignin Analysis",
        "header_subtitle": "Department of Analytical Chemistry & Humic Systems Laboratory, Chemistry Faculty, MSU",
        "nav_title": "### ⚗️ ChemoSuite",
        "nav_module_label": "Analytical Module:",
        "nav_cat_ms": "🔬 Mass Spectrometry",
        "nav_cat_optics": "🌈 Optical Spectroscopy",
        "nav_cat_resonance": "🧲 Resonance & Thermal",
        "nav_cat_chrom": "📏 Chromatography & Composition",
        "nav_cat_ml": "🧬 Multi-block ML & Data Fusion",
        "proj_modal_btn": "💼 Project Manager (.chemo)",
        "proj_modal_title": "💼 ChemoSuite Project & Session Manager",
        "proj_modal_desc": "Export current workspace state or restore previously saved multi-modal analytical project.",
        "mod1_name": "🧪 NOM-Spectra FT-ICR MS",
        "mod2_name": "💡 EEM-PARAFAC (Fluorescence)",
        "mod3_name": "🧬 ChemoSuite ML (Data Fusion)",
        "lang_label": "🌐 Language / Язык",
        # Module 1: FT-ICR MS
        "title": "🔬 Ultra-High Resolution Mass Spectrometry (FT-ICR MS)",
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
        "tab_stage1": "📥 1. Raw Spectrum & Processing",
        "tab_stage2": "🧬 2. Identification & Chemotyping",
        "tab_stage3": "⚖️ 3. Comparative Analysis & Networks",
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
        "vk_show_bioclasses": "Display Biogeochemical Polygons (Lipids, Proteins, Lignin, Tannins, CAS)",
        "vk_bioclass_dist_title": "Biogeochemical Class Distribution (%)",
        "vk_bioclass_col": "Biogeochemical Class",
        "vk_bioclass_pct": "Share (%)",
        "batch_zip_expander": "📦 Batch Spectrum Processing (ZIP Archive)",
        "batch_zip_uploader": "Upload ZIP archive with spectra:",
        "batch_ion_mode": "Ionization mode:",
        "batch_ppm_tol": "PPM tolerance:",
        "batch_min_int": "Noise cutoff (min intensity):",
        "batch_run_btn": "🚀 Run Batch Archive Processing",
        "batch_success": "Successfully processed spectra: {n}",
        "batch_table_title": "Summary Descriptor Matrix (20 cells, indices, pools):",
        "batch_dl_csv": "📥 Download Descriptor Matrix (chemo_fticr_block.csv)",
        "batch_transfer_ml_btn": "🧬 Transfer Descriptors to Module 3 (ChemoSuite ML)",
        "batch_transfer_success": "Descriptors transferred to Module 3 session! You can now assemble the dataset in Module 3.",
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
        "tmds_mode_freq": "📊 Frequency Distribution",
        "tmds_mode_network": "🕸️ Reaction Network Graph",
        "tmds_net_title": "Biogeochemical Reaction Network (TMDS Network Graph)",
        "tmds_net_peaks": "Top peaks count for network:",
        "tmds_net_layout": "Network layout:",
        "tmds_layout_spring": "Spring Force-Directed",
        "tmds_layout_chem": "Chemical Coordinates (m/z vs Int)",
        "tmds_hubs_title": "👑 Key Topological Reaction Hubs",
        "tmds_dl_edges": "📥 Download Reaction Edges (CSV)",
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
        # Module 2: EEM-PARAFAC
        "eem_title": "💡 Fluorescence Analysis: EEM-PARAFAC",
        "eem_subtitle": "Methodology of Humic Systems Lab, Chemistry Faculty, MSU | NOM & Slime-Lignin",
        "eem_sidebar_hdr": "📁 EEM Matrix Manager",
        "eem_src_label": "Optical Data Source:",
        "eem_src_synth": "Synthetic Benchmark (Baikal / Lignin)",
        "eem_src_upload": "Upload Files (CSV/TXT/DAT)",
        "eem_uploader_label": "Upload EEM Matrices:",
        "eem_preproc_hdr": "Preprocessing Options",
        "eem_clean_scatter": "Remove 1st & 2nd Order Rayleigh Scattering",
        "eem_norm_raman": "Normalize to Water Raman Peak (R.U.)",
        "eem_palette_label": "EEM Color Palette:",
        "eem_err_missing": "⚠️ `eem_core.py` not found! Place `eem_core.py` in the same directory.",
        "eem_warn_no_data": "Upload fluorescence matrices or select synthetic benchmark.",
        "eem_tab1": "📊 Tab 1: 2D EEM Contour Plot",
        "eem_tab2": "📈 Tab 2: Spectral Indices (FI, HIX, SUVA)",
        "eem_tab3": "🧬 Tab 3: PARAFAC Modeling",
        "eem_select_sample": "Select sample:",
        "eem_contour_title": "Fluorescence Contour Map: {name}",
        "eem_ex_axis": "Excitation Wavelength, Ex (nm)",
        "eem_em_axis": "Emission Wavelength, Em (nm)",
        "eem_int_label": "Intensity",
        "eem_indices_title": "Optical Chemometric Descriptors",
        "eem_dl_indices": "📥 Download Indices (CSV)",
        "eem_parafac_title": "Non-negative 3-Component PARAFAC Decomposition",
        "eem_parafac_spinner": "Fitting PARAFAC model...",
        "eem_m_exp_var": "Explained Variance",
        "eem_m_corcondia": "CORCONDIA Diagnostic",
        "eem_corcondia_ok": "Model Valid",
        "eem_corcondia_warn": "Attention",
        "eem_m_components": "Fluorophores Count",
        "eem_m_comp_val": "3 Components",
        "eem_comp_c1": "C1: Fulvic-like",
        "eem_comp_c2": "C2: Humic-like (Lignin)",
        "eem_comp_c3": "C3: Protein-like",
        "eem_em_prof_title": "Emission Loadings B",
        "eem_ex_prof_title": "Excitation Loadings C",
        "eem_rel_int": "Normalized Intensity",
        "eem_scores_title": "Sample Contributions (Scores Matrix A)",
        "eem_scores_chart_title": "Partial Fluorophore Intensities",
        "eem_dl_ml_btn": "💾 Export Optical Descriptors (CSV for PLS-DA)",
        "eem_parafac_select_tables": "EEM tables to include in PARAFAC model:",
        "eem_parafac_min_samples": "⚠️ Please select at least 2 EEM tables to build the 3D tensor.",
        "eem_parafac_inspect_hdr": "🔍 Detailed Fit Diagnostics for Selected EEM Table",
        "eem_parafac_inspect_sample": "Select EEM table to inspect fit:",
        "eem_parafac_orig": "Original EEM",
        "eem_parafac_model": "PARAFAC Model",
        "eem_parafac_res": "Residual Map",
        "eem_splithalf_expander": "🔬 Split-Half Model Validation",
        "eem_splithalf_btn": "🚀 Run Split-Half Analysis",
        "eem_splithalf_need_samples": "At least 4 samples required for split-half analysis.",
        "eem_splithalf_spinner": "Performing split-half split and computing Tucker Congruence Coefficients (TCC)...",
        "eem_splithalf_success": "✅ PARAFAC model is stable and validated (Mean TCC = {tcc:.3f} ≥ 0.90)",
        "eem_splithalf_fail": "⚠️ PARAFAC model unstable (Mean TCC = {tcc:.3f} < 0.90). May be overfactored.",
        "eem_splithalf_tcc_title": "Tucker Congruence Coefficients (TCC) per component:",
        "eem_openfluor_expander": "🏷️ OpenFluor Spectral Library Matching",
        "eem_openfluor_tbl_title": "OpenFluor Spectral Library Matching Results:",
        "eem_openfluor_dl_btn": "📥 Download Identification Results (CSV)",
        # Module 4: Vibrational Spectroscopy (FTIR)
        "mod4_name": "🔬 Vibrational Spectroscopy (FTIR)",
        "ftir_title": "🔬 Vibrational Spectroscopy (ATR-FTIR & Raman)",
        "ftir_subtitle": "Chemometric analysis of functional groups, lignin bands, and C=O / C=C band deconvolution",
        "ftir_err_missing": "⚠️ Computational core ftir_core.py not found in application directory!",
        "ftir_sidebar_hdr": "📁 FTIR Spectrum Manager",
        "ftir_src_label": "FTIR Spectra Source:",
        "ftir_src_upload": "Upload files (.csv, .dpt, .txt)",
        "ftir_src_synth": "Reference Benchmark (12 samples)",
        "ftir_uploader_label": "Drop FTIR spectral files here:",
        "ftir_preproc_hdr": "Preprocessing Options",
        "ftir_baseline_method": "Baseline Correction:",
        "ftir_base_als": "AsLS (Asymmetric Least Squares)",
        "ftir_base_rubber": "Rubberband (Convex Hull)",
        "ftir_base_linear": "Linear",
        "ftir_base_none": "No Baseline Correction",
        "ftir_norm_label": "Spectral Normalization:",
        "ftir_norm_snv": "SNV (Standard Normal Variate)",
        "ftir_norm_minmax": "Min-Max [0, 1]",
        "ftir_norm_vec": "Vector (L2 norm)",
        "ftir_norm_none": "No Normalization",
        "ftir_tab1_title": "📈 Spectral Profile & Peaks",
        "ftir_tab2_title": "✂️ Deconvolution (1500–1800 cm⁻¹)",
        "ftir_tab3_title": "🏷️ Library Identification (HQI)",
        "ftir_tab4_title": "📊 Descriptors & ML Export",
        "ftir_no_samples": "No FTIR spectra loaded. Upload files in the sidebar or switch to 'Reference Benchmark'.",
        "ftir_active_sample_label": "Active Spectrum:",
        "ftir_show_all": "Overlay all session spectra",
        "ftir_deriv_label": "Derivative Spectroscopy:",
        "ftir_deriv_0": "Absorbance (d⁰)",
        "ftir_deriv_1": "1st Derivative (d¹A)",
        "ftir_deriv_2": "2nd Derivative (d²A)",
        "ftir_peaks_expander": "🏷️ Automated Peak Picking & Functional Annotation",
        "ftir_peaks_prominence": "Peak Prominence Threshold:",
        "ftir_peaks_dist": "Min Peak Distance (cm⁻¹):",
        "ftir_peaks_show_markers": "Display peak annotations on spectral plot",
        "ftir_peaks_tbl_title": "Identified Characteristic Sample Bands:",
        "ftir_peaks_dl_btn": "📥 Download Peak Table (CSV)",
        "ftir_lib_hdr": "Spectral Matching with Reference Library (IHSS, Lignin, Polymers)",
        "ftir_lib_sample_select": "Target Sample for Identification:",
        "ftir_lib_top_n": "Number of Top Matches:",
        "ftir_lib_btn": "🔍 Identify via Reference Library",
        "ftir_lib_tbl_title": "Ranked Hit List (HQI / Cosine / Pearson):",
        "ftir_lib_overlay_title": "Spectral Overlay: Sample vs Top Reference Standard",
        "ftir_lib_sample_trace": "Sample Profile",
        "ftir_lib_ref_trace": "Reference Standard",
        "ftir_deconv_sample": "Sample to deconvolve:",
        "ftir_deconv_range": "Deconvolution Range (cm⁻¹):",
        "ftir_deconv_n_peaks": "Number of Gaussian Bands:",
        "ftir_deconv_btn": "🚀 Run Band Deconvolution",
        "ftir_deconv_r2": "Fit Quality R²:",
        "ftir_deconv_rmse": "Residual RMSE:",
        "ftir_deconv_tbl_title": "Fitted Gaussian Band Parameters:",
        "ftir_sync_ml_btn": "🧬 Sync Descriptors to Module 3 (ML Data Fusion)",
        "ftir_sync_success": "FTIR descriptors ({n} samples) successfully synced to Module 3 data bus!",
        "ftir_dl_csv": "📥 Download FTIR Descriptors (CSV)",
        "ftir_dl_excel": "📊 Export to Excel (.xlsx)",
        "ml_up_ftir": "FTIR Descriptors table (CSV):",
        # Module 5: NMR Spectroscopy (13C / 1H NMR)
        "mod5_name": "🧪 NMR Spectroscopy (13C / 1H NMR)",
        "nmr_title": "🧪 Nuclear Magnetic Resonance Spectroscopy (13C / 1H NMR Studio)",
        "nmr_subtitle": "Quantitative functional analysis of carbon skeleton (IHSS), deconvolution, and aromaticity fa",
        "nmr_err_missing": "⚠️ Computational core nmr_core.py not found in modules directory!",
        "nmr_sidebar_hdr": "📁 NMR Spectrum Manager",
        "nmr_src_label": "NMR Spectra Source:",
        "nmr_src_upload": "Upload files (.csv, .txt, .jdx)",
        "nmr_src_synth": "Reference Benchmark (12 samples)",
        "nmr_uploader_label": "Drop 13C NMR spectral files here:",
        "nmr_preproc_hdr": "Preprocessing Options",
        "nmr_baseline_method": "Baseline Correction:",
        "nmr_base_asls": "AsLS (Asymmetric Least Squares)",
        "nmr_base_linear": "Linear",
        "nmr_base_poly": "Polynomial",
        "nmr_base_none": "No Baseline Correction",
        "nmr_norm_label": "Spectral Normalization:",
        "nmr_norm_area": "Integral Area (Total = 100%)",
        "nmr_norm_max": "Max Peak (Max = 1.0)",
        "nmr_norm_minmax": "Min-Max [0, 1]",
        "nmr_norm_none": "No Normalization",
        "nmr_smooth_label": "Savitzky-Golay Smoothing",
        "nmr_recal_label": "Chemical Shift Recalibration:",
        "nmr_recal_none": "No Recalibration (0.0 ppm)",
        "nmr_recal_dmso": "By DMSO-d6 Solvent (39.5 ppm)",
        "nmr_recal_tms": "By TMS / DSS Internal Standard (0.0 ppm)",
        "nmr_tab1_title": "📈 Spectral Profile & Integral Regions",
        "nmr_tab2_title": "📊 Carbon Pool Fractions & Radar",
        "nmr_tab3_title": "🔍 Deconvolution & Spectral Library",
        "nmr_tab4_title": "🧬 Descriptors & Data Fusion Sync",
        "nmr_no_samples": "No NMR spectra loaded. Upload files in the sidebar or switch to 'Reference Benchmark'.",
        "nmr_active_sample_label": "Active Spectrum:",
        "nmr_show_all": "Overlay all session spectra",
        "nmr_show_regions": "Highlight canonical carbon regions (IHSS)",
        "nmr_peaks_expander": "🏷️ Automated Peak Picking & Chemical Assignment",
        "nmr_peaks_prominence": "Peak Prominence Threshold:",
        "nmr_peaks_dist": "Min Peak Distance (ppm):",
        "nmr_peaks_show_markers": "Display resonance markers on spectrum",
        "nmr_peaks_tbl_title": "Identified Chemical Resonances:",
        "nmr_peaks_dl_btn": "📥 Download Peak Table (CSV)",
        "nmr_deconv_sample": "Sample to deconvolve:",
        "nmr_deconv_n_peaks": "Number of Gaussian Bands (108–165 ppm):",
        "nmr_deconv_btn": "🚀 Run Aromatic Zone Deconvolution",
        "nmr_deconv_r2": "Fit Quality R²:",
        "nmr_deconv_rmse": "Residual RMSE:",
        "nmr_deconv_tbl_title": "Fitted Aromatic Gaussian Band Parameters:",
        "nmr_lib_hdr": "Spectral Matching with Reference Library (IHSS Standards & Lignins)",
        "nmr_lib_sample_select": "Target Sample for Identification:",
        "nmr_lib_top_n": "Number of Top Matches:",
        "nmr_lib_btn": "🔍 Identify via Reference Library",
        "nmr_lib_tbl_title": "Ranked Hit List (HQI Cosine %):",
        "nmr_lib_overlay_title": "Spectral Overlay: Sample vs Top Reference Standard",
        "nmr_lib_sample_trace": "Sample Profile",
        "nmr_lib_ref_trace": "Reference Standard",
        "nmr_sync_ml_btn": "🧬 Sync Descriptors to Module 3 (Data Fusion)",
        "nmr_sync_success": "NMR descriptors ({n} samples) successfully synced to Module 3!",
        "nmr_dl_csv": "📥 Download NMR Descriptors (CSV)",
        "nmr_dl_excel": "📊 Export to Excel (.xlsx)",
        "ml_up_nmr": "NMR Descriptors table (CSV):",
        "ml_up_thermal": "Thermal Descriptors table (CSV):",
        # Module 6: Thermal Analysis & Pyrolysis (Py-GC-MS & TGA)
        "mod6_name": "🌡️ Thermal Analysis & Pyrolysis (Py-GC-MS & TGA)",
        "thermal_title": "🌡️ Thermal Analysis & Analytical Pyrolysis (Py-GC-MS & TGA Studio)",
        "thermal_subtitle": "Molecular thermodegradation of NOM and lignin: Py-GC-MS H/G/S biomarkers, TGA/DTG thermogravimetry & kinetic zones",
        "thermal_err_missing": "⚠️ Computational engine thermal_core.py not found in modules directory!",
        "thermal_sidebar_hdr": "📁 Pyrolysis & TGA Manager",
        "thermal_src_label": "Thermal Data Source:",
        "thermal_src_upload": "Upload Files (Py-GC-MS & TGA)",
        "thermal_src_synth": "Reference Benchmark (12 samples)",
        "thermal_up_py_label": "Upload Py-GC-MS files (.csv, .txt):",
        "thermal_up_tga_label": "Upload TGA files (.csv, .txt):",
        "thermal_tab1_title": "📈 Py-GC-MS Chromatograms & Monomers",
        "thermal_tab2_title": "🔥 TGA / DTG Thermograms & Thermal Zones",
        "thermal_tab3_title": "🧬 Chemometric Indices (S/G, L/C, TI)",
        "thermal_tab4_title": "🌐 Multi-Modal Integration & Export",
        "thermal_no_samples": "No thermal data loaded. Upload files in the sidebar or select 'Reference Benchmark'.",
        "thermal_active_sample": "Active Sample:",
        "thermal_peaks_hdr": "Identified Pyrolysis Products (Biomarker Library)",
        "thermal_peaks_dl_btn": "📥 Download Peak Table (CSV)",
        "thermal_dist_pie_title": "Pyrolysis Product Class Distribution",
        "thermal_monomers_bar_title": "Key Lignin Monomers & Biomarkers (% Area)",
        "thermal_tga_title": "Thermogravimetric Curve (TGA) & Derivative (DTG)",
        "thermal_tga_zones_chk": "Highlight Thermal Decomposition Zones (I–IV)",
        "thermal_tga_dtg_chk": "Show DTG Derivative Curve (-dm/dT)",
        "thermal_kpi_sg": "S/G Ratio (Syringyl / Guaiacyl)",
        "thermal_kpi_hg": "H/G Ratio (p-Hydroxyphenyl / Guaiacyl)",
        "thermal_kpi_lc": "L/C Ratio (Lignin / Carbohydrates)",
        "thermal_kpi_ti": "Thermal Stability Index TI (Zone III / Zone II)",
        "thermal_kpi_t50": "Half-Decomposition Temp T₅₀%",
        "thermal_kpi_residue": "Residual Mass at 800°C",
        "thermal_sync_ml_btn": "🧬 Sync Descriptors to Module 3 (ChemoSuite ML)",
        "thermal_sync_success": "Thermal descriptors ({n} samples) successfully synced to Module 3!",
        "thermal_dl_csv": "📥 Download Thermal Descriptors (CSV)",
        "thermal_dl_excel": "📊 Export to Excel (.xlsx)",
        "ml_up_sec": "SEC/GPC Descriptors table (CSV):",
        # Module 7: Size-Exclusion & Gel Permeation Chromatography (SEC / GPC)
        "mod7_name": "📏 Size-Exclusion Chromatography (SEC / GPC)",
        "sec_title": "📏 Size-Exclusion & Gel Permeation Chromatography (SEC / GPC Studio)",
        "sec_subtitle": "Molecular weight distributions (MWD), polydispersity (Đ = Mw/Mn), and environmental NOM/lignin fractionation",
        "sec_err_missing": "⚠️ Computational engine sec_core.py not found in modules directory!",
        "sec_sidebar_hdr": "📁 SEC / GPC Manager",
        "sec_src_label": "Chromatographic Data Source:",
        "sec_src_upload": "Upload SEC files (.csv, .txt)",
        "sec_src_synth": "Reference Benchmark (12 samples)",
        "sec_up_label": "Upload SEC chromatograms (.csv, .txt):",
        "sec_calib_preset_label": "Column Calibration Standard:",
        "sec_flow_rate_label": "Eluent Flow Rate (mL/min):",
        "sec_tab1_title": "📈 Elution Chromatogram & Peaks",
        "sec_tab2_title": "📊 Molecular Weight Distribution (MWD)",
        "sec_tab3_title": "⚖️ Polydispersity & Fractionation",
        "sec_tab4_title": "🌐 Multi-Modal Integration & Export",
        "sec_no_samples": "No SEC chromatograms loaded. Upload files in the sidebar or select 'Reference Benchmark'.",
        "sec_active_sample": "Active Sample:",
        "sec_peaks_hdr": "Identified Molecular Fractions & Peaks",
        "sec_peaks_dl_btn": "📥 Download Peak Table (CSV)",
        "sec_xaxis_label": "X-Axis Scale:",
        "sec_xaxis_time": "Retention Time (min)",
        "sec_xaxis_vol": "Elution Volume (mL)",
        "sec_kpi_mn": "Mn (Number-Average MW)",
        "sec_kpi_mw": "Mw (Weight-Average MW)",
        "sec_kpi_mz": "Mz (z-Average MW)",
        "sec_kpi_mp": "Mp (Peak MW)",
        "sec_kpi_pdi": "Polydispersity Đ (Mw/Mn)",
        "sec_kpi_hmw_ratio": "HMW / LMW Ratio",
        "sec_kpi_frac1": "Fraction I (>20 kDa, HMW)",
        "sec_kpi_frac4": "Fraction IV (<1 kDa, LMW)",
        "sec_sync_ml_btn": "🧬 Sync Descriptors to Module 3 (ChemoSuite ML)",
        "sec_sync_success": "SEC/GPC descriptors ({n} samples) successfully synced to Module 3!",
        "sec_dl_csv": "📥 Download SEC Descriptors (CSV)",
        "sec_dl_excel": "📊 Export to Excel (.xlsx)",
        "ml_up_sec": "SEC/GPC Descriptors Table (CSV):",
        "ml_up_elemental": "Elemental CHNS-O Descriptors Table (CSV):",
        # Module 8: Elemental Microanalysis & Bulk Stoichiometry (CHNS-O)
        "mod8_name": "⚗️ Elemental Analysis & Stoichiometry (CHNS-O)",
        "elem_title": "⚗️ Elemental Microanalysis & Bulk Stoichiometry (CHNS-O)",
        "elem_subtitle": "Atomic Ratios (H/C, O/C, N/C, S/C), Van Krevelen Diagrams, Carbon Oxidation State (NOSC) & FT-ICR MS Cross-Validation",
        "elem_err_missing": "⚠️ Computational core `elemental_core.py` not found in modules directory!",
        "elem_sidebar_hdr": "📁 CHNS-O Manager",
        "elem_src_label": "CHNS-O Data Source:",
        "elem_src_upload": "Upload CHNS-O File (.csv, .tsv)",
        "elem_src_synth": "Reference Benchmark (12 samples)",
        "elem_up_label": "Upload Elemental Composition Table (.csv, .tsv):",
        "elem_tab1_title": "📊 Bulk Stoichiometry & Indices",
        "elem_tab2_title": "🗺️ Bulk Van Krevelen Diagram",
        "elem_tab3_title": "🔬 Cross-Validation with FT-ICR MS",
        "elem_tab4_title": "🧬 Descriptors Export & ML Sync",
        "elem_no_samples": "No CHNS-O data loaded. Upload a file or select 'Reference Benchmark'.",
        "elem_active_sample": "Active Sample:",
        "elem_kpi_hc": "H/C (Atomic)",
        "elem_kpi_oc": "O/C (Atomic)",
        "elem_kpi_nc": "N/C (Atomic)",
        "elem_kpi_sc": "S/C (Atomic)",
        "elem_kpi_nosc": "NOSC (Carbon Ox. State)",
        "elem_kpi_ai": "AI_bulk (Aromaticity)",
        "elem_sync_ml_btn": "🧬 Sync Descriptors to Module 3 (ChemoSuite ML)",
        "elem_sync_success": "CHNS-O descriptors ({n} samples) successfully synced to Module 3!",
        "elem_dl_csv": "📥 Download CHNS-O Descriptors (CSV)",
        "elem_dl_excel": "📊 Export to Excel (.xlsx)",
        # Module 3: ChemoSuite ML
        "ml_title": "🧬 ChemoSuite ML: Data Fusion & PLS-DA",
        "ml_subtitle": "Multimodal Discriminant Model: FT-ICR MS (Van Krevelen grid) + EEM-PARAFAC (fluorophores & indices)",
        "ml_err_missing": "⚠️ `chemo_ml.py` file not found in the application directory!",
        "ml_sec1_title": "1. Merged Feature Matrix Preparation (Data Staging)",
        "ml_load_demo_btn": "📥 Load Demo Benchmark (Baikal vs Slime-Lignin)",
        "ml_demo_source_name": "Synthetic Benchmark (14 samples)",
        "ml_custom_source_name": "Custom",
        "ml_active_dataset_info": "Active dataset: **{src}** | Samples: **{n}**",
        "ml_classes_caption": "Class labels: `1` — Slime-Lignin, `0` — Background waters.",
        "ml_col_class_target": "Class (Target)",
        "ml_sec2_title": "2. PLS-DA Model Hyperparameters",
        "ml_lvs_slider": "Latent Variables (LVs):",
        "ml_block_scale": "Block Scaling (1/√P)",
        "ml_run_btn": "🚀 Fit PLS-DA Model",
        "ml_spinner": "Fitting PLS-DA model & performing cross-validation...",
        "ml_sec3_title": "3. Statistical Validation & Metrics",
        "ml_m_r2x": "R²X (Spectral Variance)",
        "ml_m_r2y": "R²Y (Class Variance)",
        "ml_m_q2": "Q² (LOO-CV)",
        "ml_q2_ok": "Reliable",
        "ml_q2_warn": "Poor",
        "ml_m_acc": "Accuracy",
        "ml_scores_chart_title": "Scores Plot (t₁ vs t₂) with 95% Hotelling's Ellipse",
        "ml_hotelling_name": "Hotelling T² (95%)",
        "ml_class1_label": "Slime-Lignin (Class 1)",
        "ml_class0_label": "Background / Baikal (Class 0)",
        "ml_vip_chart_title": "Key Markers (VIP Scores > 1.0)",
        "ml_vip_cutoff": "Cutoff VIP=1.0",
        "ml_vip_tbl_title": "Ranked Descriptors by VIP Score:",
        "ml_dl_vip_btn": "📥 Download VIP Descriptors (CSV)",
        "ml_warn_input": "Input Data Error: {err}",
        "ml_info_no_data": "💡 Click the button above to load a synchronized feature benchmark.",
        "ml_src_session": "🔄 Assemble from Session (Modules 1 & 2)",
        "ml_src_files": "📂 Upload External CSV Files",
        "ml_src_batch": "⚡ Folder Batch Screening",
        "ml_src_benchmark": "🧪 Synthetic Benchmark",
        "batch_screener_title": "🚀 Autonomous Directory Batch Screening (8 Modalities)",
        "batch_screener_desc": "Intelligent spectral file scanning, automatic detection of analytical techniques (FT-ICR MS, EEM, UV-Vis, FTIR, NMR, TGA/Py-GC-MS, SEC/GPC, CHNS-O), chemometric descriptor extraction, and 1-click multimodal Data Staging assembly.",
        "batch_source_type_label": "Screening input method:",
        "batch_src_folder": "📁 Scan local directory on server / computer",
        "batch_src_upload": "📤 Multi-file upload via browser",
        "batch_dir_label": "Directory path containing spectra:",
        "batch_recursive_label": "Recursive scanning of all subdirectories",
        "batch_upload_label": "Drop spectra files (CSV, TSV, TXT, DX):",
        "batch_run_btn": "🚀 Run Autonomous Batch Screening",
        "batch_files_count": "Total Files",
        "batch_success_count": "Successfully Parsed",
        "batch_samples_count": "Samples Assembled",
        "batch_modalities_count": "Modalities Detected",
        "batch_manifest_title": "📋 Summary Manifest of Recognized Files & Modalities:",
        "batch_use_ml_btn": "🧬 Use Assembled Matrix in ChemoSuite ML",
        "batch_dl_zip_btn": "📥 Download Full Screening Package (ZIP)",
        "batch_no_files_found": "No supported analytical files found in the specified directory.",
        "batch_success_msg": "Batch screening completed! Files processed: {n_files}, samples assembled: {n_samples}.",
        "ml_btn_assemble_session": "📥 Build merged matrix from active spectra",
        "ml_session_status": "FT-ICR spectra with formulas: **{n_ms}** | EEM optical samples: **{n_eem}** | UV-Vis spectra: **{n_uv}**",
        "ml_session_need_more": "⚠️ Please run formula assignment in Module 1 or calculate optical descriptors (EEM / UV-Vis) in Module 2 first.",
        "ml_uploader_mode": "File upload format:",
        "ml_upload_two": "Separate files by blocks (FT-ICR / EEM / UV-Vis)",
        "ml_upload_single": "Single unified feature CSV file",
        "ml_up_ms": "FT-ICR MS descriptor table (CSV):",
        "ml_up_eem": "EEM-PARAFAC descriptor table (CSV):",
        "ml_up_uv": "UV-Vis descriptor table (CSV):",
        "ml_up_single": "Unified descriptor table (CSV):",
        "ml_strategy_label": "Data Fusion Strategy:",
        "ml_strat_low": "Low-Level Fusion (Block Scaling 1/√P)",
        "ml_strat_mid": "Mid-Level Fusion (PCA compression FT-ICR + Optics)",
        "ml_perm_chk": "Run permutation validation test (H₀)",
        "ml_tab_scores": "🎯 Scores Plot (t₁ vs t₂)",
        "ml_tab_vips": "📊 Biomarkers (VIP Scores)",
        "ml_tab_val": "📈 Validation & Diagnostics",
        "ml_tab_blocks": "🧬 Analytical Block Importance",
        "ml_tab_crosscorr": "🌐 Cross-Omics Atlas",
        "ml_crosscorr_title": "🌐 Cross-Omics Correlation Atlas & 2D-COS",
        "ml_crosscorr_subtitle": "2D Heterospectral correlation and inter-block co-variation discovery",
        "ml_crosscorr_mode_select": "Correlation Analysis Mode:",
        "ml_crosscorr_mode_pairs": "Pairwise 2D Heterospectral Correlation (Block A vs Block B)",
        "ml_crosscorr_mode_global": "Global Multiblock Correlation Matrix",
        "ml_crosscorr_block_a": "Analytical Block A (Rows / Y):",
        "ml_crosscorr_block_b": "Analytical Block B (Columns / X):",
        "ml_crosscorr_method": "Correlation Method:",
        "ml_crosscorr_threshold": "Significance Threshold (|r|):",
        "ml_crosscorr_top_pairs_title": "🔝 Top Inter-Block Descriptor Cross-Correlations",
        "ml_crosscorr_dl_edges": "📥 Download Network Edges (CSV)",
        "ml_crosscorr_dl_top": "📥 Download Top Correlations (CSV)",
        "ml_crosscorr_no_data": "Cross-correlation analysis requires at least 2 analytical blocks across multiple samples.",
        "ml_cm_title": "Confusion Matrix",
        "ml_perm_title": "Permutation Test Distribution (H0: Random Model)",
        "ml_donut_title": "Relative Analytical Block Contribution (VIP²)",
        "ml_warn_two_classes": "⚠️ Classification requires at least two distinct classes in Class_Target!",
        "ml_model_type_label": "Model Architecture:",
        "ml_model_plsda": "PLS-DA (Multiclass / Latent Variables)",
        "ml_model_oplsda": "OPLS-DA (Orthogonal Noise Filter / S-Plot)",
        "ml_model_splsda": "Sparse PLS-DA (sPLS-DA with L1 Biomarker Selection)",
        "ml_spls_keep_slider": "Feature Sparsity (Keep top features per component):",
        "ml_spls_biomarkers_title": "🎯 Selected Biomarker Panel (L1 Sparse Features):",
        "ml_spls_weight_axis": "Absolute Sparse Weight (|W|)",
        "ml_spls_table_title": "Selected Biomarker Panel",
        "ml_spls_selected_count": "selected",
        "ml_spls_dl_btn": "📥 Download Sparse Biomarkers (CSV)",
        "ml_ortho_slider": "Orthogonal Components (Orthogonal LVs):",
        "ml_scores_mode_label": "Scores Projection Space:",
        "ml_scores_2d": "2D Scores (LV1 vs LV2)",
        "ml_scores_3d": "3D Scores (LV1 vs LV2 vs LV3)",
        "ml_tab_splot": "💠 S-Plot (OPLS-DA Markers)",
        "ml_splot_title": "S-Plot: Covariance p[1] vs Correlation p(corr)[1]",
        "ml_m_r2x_pred": "R²X(pred)",
        "ml_m_r2x_ortho": "R²X(ortho)",
        "ml_warn_opls_binary": "ℹ️ OPLS-DA is optimized for binary contrast. For 3+ classes, please select PLS-DA.",
        "ml_dl_passport_btn": "📑 Download Full Analytical Passport (.xlsx)",
        "ml_dl_html_passport_btn": "📄 Download Ecological Passport (Interactive HTML / PDF)",
        "ml_html_passport_desc": "Self-contained interactive ecological passport: expert diagnosis, 8 multi-omics methods, VIP biomarkers, cross-correlations and 1-Click PDF print.",
        "ml_preview_passport_expander": "👁️ Interactive Ecological Passport Live Preview",
        "ml_sens_label": "Sensitivity:",
        "ml_spec_label": "Specificity:",
        "ml_bal_acc_label": "Balanced Accuracy:",
        "ml_acc_label": "Accuracy:",
        "ml_mode_multiclass": "Mode: Multiclass classification ({n} classes)",
        "ml_report_lang_label": "Report Language / Язык отчета:",
        "ml_passport_desc": "Multi-sheet Excel workbook: Summary, Fused Matrix, FT-ICR 20-Grid, EEM, UV-Vis, VIP Biomarkers, S-Plot.",
        "proj_expander_title": "💾 ChemoSuite Project (.chemo)",
        "proj_save_header": "Save Current Session",
        "proj_save_btn": "💾 Download Project (.chemo)",
        "proj_load_header": "Load Saved Project",
        "proj_load_label": "Project File (.chemo / .zip):",
        "proj_restore_btn": "📥 Restore Workspace Session",
        "proj_restored_success": "ChemoSuite session successfully restored!",
        "proj_summary_spec": "FT-ICR Spectra: {n}",
        "proj_summary_eem": "EEM: {val}",
        "proj_summary_uv": "UV-Vis: {val}",
        "proj_summary_fused": "Data Fusion: {val}",
        "proj_summary_model": "Model: {val}",
        #UV-spectre
        "eem_uv_uploader_label": "Upload UV-Vis Absorbance Spectra (CSV/TXT):",
        "eem_doc_expander": "DOC Settings for SUVA254 Calculation",
        "eem_default_doc": "Default DOC concentration (mg C / L):",
        "eem_tab4": "📉 Tab 4: UV-Vis Absorbance",
        "eem_uv_title": "UV-Vis Absorbance Spectrophotometry",
        "eem_uv_select_sample": "Select UV-Vis spectrum:",
        "eem_uv_plot_title": "UV-Vis Absorbance Spectrum A(λ): {name}",
        "eem_uv_tbl_title": "UV-Vis Optical Descriptors (A254, E2/E3, S_R, SUVA)",
        "eem_dl_uv_indices": "📥 Download UV-Vis Descriptors (CSV)",
        "eem_uv_no_data": "Upload UV-Vis absorption spectrum files in the sidebar.",
        # Horizon 3: PubChem Chemoinformatics & Vector Reaction Pathways
        "pubchem_inspect_expander": "🧪 PubChem Chemoinformatics (Formula Structure Identification)",
        "pubchem_select_formula": "Select formula from spectrum:",
        "pubchem_custom_input": "Or enter custom molecular formula (C_xH_yO_z...):",
        "pubchem_search_btn": "🔍 Search PubChem Database",
        "pubchem_searching": "Querying PubChem PUG-REST API...",
        "pubchem_res_name": "Canonical Name:",
        "pubchem_res_iupac": "IUPAC Name:",
        "pubchem_res_class": "Chemical Class:",
        "pubchem_res_subclass": "Biogeochemical Subclass:",
        "pubchem_res_smiles": "SMILES:",
        "pubchem_res_cid": "PubChem CID:",
        "pubchem_link_btn": "🌐 Open in PubChem",
        "pubchem_chembl_btn": "🧪 Search in ChEMBL",
        "pubchem_hmdb_btn": "🧬 Search in HMDB (Metabolome)",
        "pubchem_isomers_label": "Structural Isomers in PubChem:",
        "pubchem_card_header": "Chemoinformatic Passport of Isomer #{i}: {title}",
        "vk_inspect_title": "🔬 Chemoinformatic Point Identification (PubChem / ChEMBL / HMDB)",
        "vk_inspect_caption": "Select a molecular formula from the Van Krevelen diagram for instantaneous structural retrieval, SMILES, and chemotypic classification.",
        "vk_inspect_select": "Formula from diagram:",
        "tmds_mode_pathways": "🛣️ Reaction Pathways Tracing",
        "tmds_flux_header": "📊 Geochemical Vector Flux Analysis",
        "tmds_flux_ox_decarb": "Oxidation / Decarboxylation Ratio (O/CO₂)",
        "tmds_flux_alkyl": "Alkylation / Homology Share (% CH₂)",
        "tmds_flux_sulf": "Sulfonation Stress Index (% SO₃)",
        "tmds_flux_hydr": "Hydration / Oxidation Ratio (H₂O/O)",
        "tmds_path_source": "Precursor Ion (m/z):",
        "tmds_path_target": "Target Product Ion (m/z):",
        "tmds_path_depth": "Max reaction steps (Depth):",
        "tmds_path_tol": "TMDS Tolerance (mDa):",
        "tmds_path_find_btn": "🛣️ Find Transformation Pathways",
        "tmds_path_found": "Valid transformation pathways found: {n}",
        "tmds_path_not_found": "No reaction pathways found between selected ions within tolerance.",
        "tmds_dl_paths": "📥 Download Pathways (CSV)",
        "ml_pubchem_title": "🧪 Chemoinformatic Biomarker Identification (PubChem)",
        "ml_pubchem_caption": "Automatic mapping of biomarker formulas to chemical structures via PubChem REST API.",
        "ml_pubchem_btn": "🔍 Annotate Biomarkers in PubChem",
        "ml_pubchem_spinner": "Annotating biomarkers via PubChem API...",
        "ml_pubchem_dl_btn": "📥 Download Annotated Biomarkers (CSV)",
        # UX Improvements / Features
        "kmd_slicer_title": "🎯 Homologous Series Slicer (KMD Slicer)",
        "kmd_slicer_enable": "Enable KMD homologous series isolation",
        "kmd_target_val": "Target KMD (Mass Defect):",
        "kmd_tol_val": "Slice Tolerance (± ΔKMD):",
        "kmd_slicer_count": "Series members isolated: **{n}** of {total} ({pct:.1f}%)",
        "kmd_dl_series": "📥 Download Isolated Series (CSV)",
        "batch_desc_title": "📊 Batch Export Descriptors Across All Spectra",
        "batch_desc_caption": "Consolidated table of biogeochemical indices (Mn, Mw, H/C, O/C, DBE, AI, NOSC) and Perminova 20-cells for all identified spectra in session.",
        "batch_desc_dl_csv": "📥 Export Summary Table (CSV)",
        "batch_desc_dl_excel": "📊 Export to Excel (.xlsx)",
        "batch_desc_no_data": "No spectra with assigned formulas found in the current session. Run formula assignment for at least one sample.",
    },
}

# ==============================================================================
# КОНСТАНТЫ И ВЫЧИСЛИТЕЛЬНОЕ ЯДРО FT-ICR MS
# ==============================================================================
from fticr_core import (
    EXACT_MASSES,
    C13_DIFF,
    H_ION_MASS,
    KMD_BASES,
    TMDS_LIBRARY,
    VAN_KREVELEN_REGIONS,
    get_biomolecular_distribution,
    batch_process_fticr_spectra,
    calculate_descriptors,
    parse_uploaded_file,
    fast_formula_assigner,
    run_formula_assignment,
    compute_kmd,
    compute_vk20_grid,
    align_two_spectra_fast,
    perform_spectral_algebra,
    run_tmds_screening,
    build_tmds_network_graph,
    compute_geochemical_vector_fluxes,
    find_transformation_pathways,
    get_calibrant_library,
)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ЯДРО FT-ICR MS
# ==============================================================================
def st_df(data, **kwargs):
    if isinstance(data, pd.DataFrame):
        col_cfg = kwargs.get("column_config")
        if col_cfg is None:
            col_cfg = {}
        else:
            col_cfg = dict(col_cfg)
        for col in data.columns:
            if col not in col_cfg:
                col_s = str(col).lower()
                if any(k in col_s for k in ["origin", "генезис", "эколог", "desc", "описан", "status", "статус", "name", "назван", "smiles", "pathway", "trans"]):
                    col_cfg[col] = st.column_config.TextColumn(width="large")
                elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                    max_len = data[col].astype(str).str.len().max() if len(data) > 0 else 0
                    if max_len > 35:
                        col_cfg[col] = st.column_config.TextColumn(width="large")
                    elif max_len > 15:
                        col_cfg[col] = st.column_config.TextColumn(width="medium")
        if col_cfg:
            kwargs["column_config"] = col_cfg
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)


def ensure_sample_peaks(sample: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Гарантирует наличие валидного DataFrame пиков (parsed_peaks) у образца."""
    if sample is None:
        return None
    if sample.get("parsed_peaks") is not None and not sample["parsed_peaks"].empty:
        return sample["parsed_peaks"]

    b_content = sample.get("file_bytes")
    if not b_content and sample.get("raw_path") and os.path.exists(sample["raw_path"]):
        try:
            with open(sample["raw_path"], "rb") as f:
                b_content = f.read()
                sample["file_bytes"] = b_content
        except Exception:
            pass

    if not b_content:
        return None

    try:
        raw_df = parse_uploaded_file(b_content, delimiter="Auto", decimal_sep=".", has_header=True)
        if raw_df is None or raw_df.empty:
            return None
        sample["raw_df"] = raw_df
        col_names = list(raw_df.columns)
        mz_col = col_names[0]
        int_col = col_names[1] if len(col_names) > 1 else col_names[0]
        for idx, cname in enumerate(col_names):
            cn_low = str(cname).lower()
            if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]):
                mz_col = cname
            elif any(k in cn_low for k in ["int", "i", "count", "abund"]):
                int_col = cname
        clean_m = pd.to_numeric(raw_df[mz_col], errors="coerce")
        clean_i = pd.to_numeric(raw_df[int_col], errors="coerce")
        valid_mask = clean_m.notna() & clean_i.notna()
        v_df = pd.DataFrame({"mass": clean_m[valid_mask].astype(float), "intensity": clean_i[valid_mask].astype(float)})
        v_df = v_df[v_df["mass"] > 0].sort_values("mass").reset_index(drop=True)
        if not v_df.empty:
            v_df["norm_intensity"] = (v_df["intensity"] / max(1e-12, float(v_df["intensity"].max()))) * 100.0
            sample["parsed_peaks"] = v_df
            return v_df
    except Exception:
        pass
    return sample.get("parsed_peaks")


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
        return st.plotly_chart(fig, use_container_width=True, config=config, **kwargs)


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


# ==============================================================================
# ИНИЦИАЛИЗАЦИЯ ХРАНИЛИЩА SESSION_STATE
# ==============================================================================
if "spectra_db" not in st.session_state:
    st.session_state["spectra_db"] = {}
    for fname in os.listdir(STORAGE_DIR):
        if fname.lower().endswith((".csv", ".txt", ".tsv", ".xy")):
            fpath = os.path.join(STORAGE_DIR, fname)
            try:
                with open(fpath, "rb") as f: content = f.read()
                s_obj = {
                    "raw_path": fpath, "file_bytes": content,
                    "parsed_peaks": None, "assigned_df": None, "raw_df": None,
                }
                ensure_sample_peaks(s_obj)
                st.session_state["spectra_db"][fname] = s_obj
            except Exception:
                pass

if "eem_ml_descriptors" not in st.session_state:
    st.session_state["eem_ml_descriptors"] = None
if "uv_ml_descriptors" not in st.session_state:
    st.session_state["uv_ml_descriptors"] = None
if "ftir_samples" not in st.session_state:
    st.session_state["ftir_samples"] = {}
if "ftir_ml_descriptors" not in st.session_state:
    st.session_state["ftir_ml_descriptors"] = None
if "nmr_samples" not in st.session_state:
    st.session_state["nmr_samples"] = {}
if "nmr_ml_descriptors" not in st.session_state:
    st.session_state["nmr_ml_descriptors"] = None
if "thermal_samples" not in st.session_state:
    st.session_state["thermal_samples"] = {}
if "thermal_ml_descriptors" not in st.session_state:
    st.session_state["thermal_ml_descriptors"] = None
if "sec_samples" not in st.session_state:
    st.session_state["sec_samples"] = {}
if "sec_ml_descriptors" not in st.session_state:
    st.session_state["sec_ml_descriptors"] = None
if "elemental_samples" not in st.session_state:
    st.session_state["elemental_samples"] = {}
if "elemental_ml_descriptors" not in st.session_state:
    st.session_state["elemental_ml_descriptors"] = None

# ==============================================================================
# ДИАЛОГ УПРАВЛЕНИЯ ПРОЕКТОМ CHEMOSUITE (.chemo)
# ==============================================================================
if hasattr(st, "dialog") and PROJECT_IO_AVAILABLE and project_io is not None:
    @st.dialog(T[st.session_state.get("lang", "en")]["proj_modal_title"], width="large")
    def show_project_manager_dialog():
        cur_lang = st.session_state.get("lang", "en")
        st.caption(T[cur_lang]["proj_modal_desc"])

        col_save, col_load = st.columns([1, 1], gap="medium")

        with col_save:
            st.subheader(T[cur_lang]["proj_save_header"])
            n_spec = len(st.session_state.get("spectra_db", {}))
            has_eem = st.session_state.get("eem_ml_descriptors") is not None
            has_uv = st.session_state.get("uv_ml_descriptors") is not None
            has_ftir = st.session_state.get("ftir_ml_descriptors") is not None
            has_nmr = st.session_state.get("nmr_ml_descriptors") is not None
            has_thermal = st.session_state.get("thermal_ml_descriptors") is not None
            has_sec = st.session_state.get("sec_ml_descriptors") is not None
            has_elem = st.session_state.get("elemental_ml_descriptors") is not None
            has_fused = st.session_state.get("fused_data") is not None
            m_res = st.session_state.get("pls_results")
            m_type = m_res.get("model_type") if isinstance(m_res, dict) else ("Нет" if cur_lang == "ru" else "None")

            yes_lbl = "✓" if cur_lang == "ru" else "✓"
            no_lbl = "—" if cur_lang == "ru" else "—"

            st.info(
                f"📊 **{'Активные данные в сессии' if cur_lang == 'ru' else 'Active Session Data'}:**\n\n"
                f"• FT-ICR MS: **{n_spec}** {'спектров' if cur_lang == 'ru' else 'spectra'}\n"
                f"• EEM: **{yes_lbl if has_eem else no_lbl}** | UV-Vis: **{yes_lbl if has_uv else no_lbl}** | FTIR: **{yes_lbl if has_ftir else no_lbl}**\n"
                f"• NMR: **{yes_lbl if has_nmr else no_lbl}** | Thermal: **{yes_lbl if has_thermal else no_lbl}** | SEC: **{yes_lbl if has_sec else no_lbl}** | CHNS: **{yes_lbl if has_elem else no_lbl}**\n"
                f"• Data Fusion: **{yes_lbl if has_fused else no_lbl}** | ML Model: **{m_type}**"
            )

            try:
                chemo_proj_bytes = project_io.save_chemo_project(
                    dict(st.session_state),
                    project_name=st.session_state.get("fused_source", "ChemoSuite_Project"),
                )
                from datetime import datetime
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label=T[cur_lang]["proj_save_btn"],
                    data=chemo_proj_bytes,
                    file_name=f"ChemoSuite_Session_{time_str}.chemo",
                    mime="application/octet-stream",
                    use_container_width=True,
                    type="primary",
                    key="dl_chemo_project_btn_modal",
                )
            except Exception as e_save:
                st.caption(f"Ошибка сериализации: {e_save}")

        with col_load:
            st.subheader(T[cur_lang]["proj_load_header"])
            proj_file = st.file_uploader(
                T[cur_lang]["proj_load_label"],
                type=["chemo", "zip"],
                key="chemo_uploader_input_modal",
            )
            if proj_file is not None:
                try:
                    p_bytes = proj_file.getvalue()
                    p_sum = project_io.get_chemo_project_summary(p_bytes)
                    st.success(f"✓ {p_sum.get('project_name', 'ChemoSuite Project')} ({p_sum.get('created_at', '')[:10]})")
                    if st.button(T[cur_lang]["proj_restore_btn"], type="primary", use_container_width=True, key="btn_restore_chemo_modal"):
                        restored = project_io.load_chemo_project(p_bytes)
                        for k, v in restored.items():
                            if k != "project_metadata":
                                st.session_state[k] = v
                        for s_name, s_val in restored.get("spectra_db", {}).items():
                            if isinstance(s_val, dict) and s_val.get("file_bytes"):
                                local_path = os.path.join(STORAGE_DIR, s_name)
                                try:
                                    with open(local_path, "wb") as f_loc:
                                        f_loc.write(s_val["file_bytes"])
                                    s_val["raw_path"] = local_path
                                except Exception:
                                    pass
                        st.toast(T[cur_lang]["proj_restored_success"], icon="✅")
                        st.rerun()
                except Exception as e_load:
                    st.error(f"Ошибка чтения проекта: {e_load}")

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ: ВЫБОР МОДУЛЯ И ПАРАМЕТРОВ
# ==============================================================================
with st.sidebar:
    # 1. Верхний ряд: Язык и кнопка Проекта
    c_lang, c_proj = st.columns([1.1, 1.4])
    with c_lang:
        default_lang_idx = 0 if st.session_state.get("lang", "en") == "en" else 1
        lang_choice = st.selectbox("🌐", ["English", "Русский"], index=default_lang_idx, label_visibility="collapsed", key="global_lang_choice")
        lang = "en" if lang_choice == "English" else "ru"
        st.session_state["lang"] = lang
    with c_proj:
        if PROJECT_IO_AVAILABLE and project_io is not None:
            if hasattr(st, "dialog"):
                if st.button("💼 Project", key="btn_open_project_modal", help=T[lang]["proj_modal_btn"], use_container_width=True):
                    show_project_manager_dialog()
            else:
                pass

    st.markdown(T[lang]["nav_title"])

    # Логическая группировка 8 аналитических модулей
    module_catalog = [
        {"cat": T[lang]["nav_cat_ms"], "key": T[lang]["mod1_name"]},
        {"cat": T[lang]["nav_cat_optics"], "key": T[lang]["mod2_name"]},
        {"cat": T[lang]["nav_cat_optics"], "key": T[lang]["mod4_name"]},
        {"cat": T[lang]["nav_cat_resonance"], "key": T[lang]["mod5_name"]},
        {"cat": T[lang]["nav_cat_resonance"], "key": T[lang]["mod6_name"]},
        {"cat": T[lang]["nav_cat_chrom"], "key": T[lang]["mod7_name"]},
        {"cat": T[lang]["nav_cat_chrom"], "key": T[lang]["mod8_name"]},
        {"cat": T[lang]["nav_cat_ml"], "key": T[lang]["mod3_name"]},
    ]

    all_module_keys = [
        T[lang]["mod1_name"],
        T[lang]["mod2_name"],
        T[lang]["mod4_name"],
        T[lang]["mod5_name"],
        T[lang]["mod6_name"],
        T[lang]["mod7_name"],
        T[lang]["mod8_name"],
        T[lang]["mod3_name"],
    ]

    if "selected_module_idx" not in st.session_state:
        st.session_state["selected_module_idx"] = 0

    # Защита от выхода индекса за границы
    safe_idx = min(max(0, st.session_state.get("selected_module_idx", 0)), len(all_module_keys) - 1)

    # Привязываем ключ сессии для мгновенной однократной регистрации клика
    if "nav_module_selection" not in st.session_state or st.session_state["nav_module_selection"] not in all_module_keys:
        st.session_state["nav_module_selection"] = all_module_keys[safe_idx]

    def _on_module_radio_changed():
        val = st.session_state.get("nav_module_selection")
        if val in all_module_keys:
            st.session_state["selected_module_idx"] = all_module_keys.index(val)

    active_module = st.radio(
        T[lang]["nav_module_label"],
        all_module_keys,
        key="nav_module_selection",
        on_change=_on_module_radio_changed,
        format_func=lambda m: m,
        label_visibility="collapsed",
    )
    st.session_state["selected_module_idx"] = all_module_keys.index(active_module)

    st.markdown("---")

    # Fallback для старых версий Streamlit без st.dialog
    if not hasattr(st, "dialog") and PROJECT_IO_AVAILABLE and project_io is not None:
        with st.expander(T[lang]["proj_expander_title"], expanded=False):
            st.markdown(f"**{T[lang]['proj_save_header']}**")
            chemo_proj_bytes = project_io.save_chemo_project(
                dict(st.session_state),
                project_name=st.session_state.get("fused_source", "ChemoSuite_Project"),
            )
            from datetime import datetime
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label=T[lang]["proj_save_btn"],
                data=chemo_proj_bytes,
                file_name=f"ChemoSuite_Session_{time_str}.chemo",
                mime="application/octet-stream",
                use_container_width=True,
                key="dl_chemo_project_btn_fallback",
            )
            st.markdown("---")
            proj_file = st.file_uploader(
                T[lang]["proj_load_label"],
                type=["chemo", "zip"],
                key="chemo_uploader_input_fallback",
            )
            if proj_file is not None:
                p_bytes = proj_file.getvalue()
                if st.button(T[lang]["proj_restore_btn"], type="primary", use_container_width=True, key="btn_restore_chemo_fb"):
                    restored = project_io.load_chemo_project(p_bytes)
                    for k, v in restored.items():
                        if k != "project_metadata":
                            st.session_state[k] = v
                    st.rerun()

    try:
        import importlib
        import modules.chemo_agent
        importlib.reload(modules.chemo_agent)
        import views.agent_view
        importlib.reload(views.agent_view)

        # Sanitize any stale deprecated model in active browser session
        _stale_m = st.session_state.get("agent_model_select")
        if _stale_m and any(bad in str(_stale_m).lower() for bad in ["llama", "mixtral", "gemma", "versatile", "instant"]):
            st.session_state["agent_model_select"] = "qwen/qwen3.8-27b"

        views.agent_view.render_chemo_agent_sidebar(T, lang)
    except Exception as _e_agent:
        st.caption(f"🤖 ChemoAgent: {_e_agent}")

# ==============================================================================
# NORDIC PRO HEADER & TELEMETRY STRIP (LIVE MULTIMODAL TELEMETRY)
# ==============================================================================
_n_ms = len(st.session_state.get("spectra_db", {}))
_has_eem = st.session_state.get("eem_ml_descriptors") is not None
_has_uv = st.session_state.get("uv_ml_descriptors") is not None
_has_ftir = st.session_state.get("ftir_ml_descriptors") is not None
_has_nmr = st.session_state.get("nmr_ml_descriptors") is not None
_has_therm = st.session_state.get("thermal_ml_descriptors") is not None
_has_sec = st.session_state.get("sec_ml_descriptors") is not None
_has_elem = st.session_state.get("elemental_ml_descriptors") is not None
_has_ml = st.session_state.get("fused_data") is not None

# Подготовка телеметрических чипов (8 аналитических блоков)
_chips = []
# 1. MS
if _n_ms > 0:
    _chips.append(f'<span class="telemetry-chip active-ms"><span class="status-dot"></span>🔬 MS ({_n_ms})</span>')
else:
    _chips.append('<span class="telemetry-chip">🔬 MS (0)</span>')

# 2. EEM
if _has_eem:
    _chips.append('<span class="telemetry-chip active-eem"><span class="status-dot"></span>💡 EEM</span>')
else:
    _chips.append('<span class="telemetry-chip">💡 EEM</span>')

# 3. UV-Vis
if _has_uv:
    _chips.append('<span class="telemetry-chip active-eem"><span class="status-dot"></span>🌈 UV-Vis</span>')
else:
    _chips.append('<span class="telemetry-chip">🌈 UV-Vis</span>')

# 4. FTIR
if _has_ftir:
    _chips.append('<span class="telemetry-chip active-ftir"><span class="status-dot"></span>🔍 FTIR</span>')
else:
    _chips.append('<span class="telemetry-chip">🔍 FTIR</span>')

# 5. NMR
if _has_nmr:
    _chips.append('<span class="telemetry-chip active-nmr"><span class="status-dot"></span>🧲 NMR</span>')
else:
    _chips.append('<span class="telemetry-chip">🧲 NMR</span>')

# 6. Thermal
if _has_therm:
    _chips.append('<span class="telemetry-chip active-thermal"><span class="status-dot"></span>🔥 Thermal</span>')
else:
    _chips.append('<span class="telemetry-chip">🔥 Thermal</span>')

# 7. SEC
if _has_sec:
    _chips.append('<span class="telemetry-chip active-sec"><span class="status-dot"></span>📏 SEC/GPC</span>')
else:
    _chips.append('<span class="telemetry-chip">📏 SEC/GPC</span>')

# 8. CHNS-O
if _has_elem:
    _chips.append('<span class="telemetry-chip active-elem"><span class="status-dot"></span>⚗️ CHNS-O</span>')
else:
    _chips.append('<span class="telemetry-chip">⚗️ CHNS-O</span>')

# 9. ML Fusion
if _has_ml:
    _chips.append('<span class="telemetry-chip active-ml"><span class="status-dot"></span>🧬 ML Fused</span>')
else:
    _chips.append('<span class="telemetry-chip">🧬 ML Ready</span>')

# 10. AI Copilot
try:
    import modules.chemo_agent as chemo_agent
    if chemo_agent.get_groq_api_key(st.session_state.get("groq_api_key")):
        _chips.append('<span class="telemetry-chip active-agent"><span class="status-dot"></span>🤖 Groq LPU</span>')
    else:
        _chips.append('<span class="telemetry-chip">🤖 AI Copilot</span>')
except Exception:
    pass

_telemetry_html = "".join(_chips)
_telemetry_caption = "Live Modalities:" if lang == "en" else "Телеметрия сессии:"
_version_tag = '<span style="background: rgba(37,99,235,0.08); color: #1d4ed8; border: 1px solid rgba(37,99,235,0.25); font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 6px; font-family: var(--nl-font-mono);">v3.0.0 Pro</span>'

platform_hdr_html = f"""
<div class="nordic-app-header">
  <div class="nordic-header-top">
    <div class="nordic-brand-lockup">
      <div class="nordic-brand-icon">⚗️</div>
      <div class="nordic-brand-titles">
        <div style="display: flex; align-items: center; gap: 8px;">
          <h2 class="nordic-brand-title">{T[lang]["header_title"]}</h2>
          {_version_tag}
        </div>
        <p class="nordic-brand-subtitle">{T[lang]["header_subtitle"]}</p>
      </div>
    </div>
  </div>
  <div class="nordic-telemetry-strip">
    <span class="telemetry-caption">{_telemetry_caption}</span>
    {_telemetry_html}
  </div>
</div>
"""
st.markdown(platform_hdr_html, unsafe_allow_html=True)


# ==============================================================================
# МОДУЛЬ 1: NOM-SPECTRA FT-ICR MS STUDIO
# ==============================================================================
# ==============================================================================
# MODULAR STUDIO DISPATCHER (ChemoSuite Pro Views)
# ==============================================================================
import views

if active_module == T[lang]["mod1_name"]:
    views.render_fticr_studio(T, lang)
elif active_module == T[lang]["mod2_name"]:
    views.render_eem_studio(T, lang)
elif active_module == T[lang]["mod3_name"]:
    views.render_chemo_ml_studio(T, lang)
elif active_module == T[lang]["mod4_name"]:
    views.render_ftir_studio(T, lang)
elif active_module == T[lang]["mod5_name"]:
    views.render_nmr_studio(T, lang)
elif active_module == T[lang]["mod6_name"]:
    views.render_thermal_studio(T, lang)
elif active_module == T[lang]["mod7_name"]:
    views.render_sec_studio(T, lang)
elif active_module == T[lang]["mod8_name"]:
    views.render_elemental_studio(T, lang)
