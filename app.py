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
if active_module == T[lang]["mod1_name"]:
    with st.sidebar:
        st.title(T[lang]["sidebar_mgr"])
        if not NOMSPECTRA_INSTALLED:
            st.info(T[lang]["nomspectra_missing"])

        uploaded_files = st.file_uploader(
            T[lang]["uploader_label"], type=["csv", "txt", "tsv", "xy"], accept_multiple_files=True,
            key="fticr_uploader_files"
        )
        if uploaded_files:
            latest_uploaded_name = None
            for uf in uploaded_files:
                save_path = os.path.join(STORAGE_DIR, uf.name)
                b_content = uf.getvalue()
                existing = st.session_state["spectra_db"].get(uf.name)
                if existing is not None and existing.get("file_bytes") == b_content:
                    continue

                with open(save_path, "wb") as f:
                    f.write(b_content)
                s_new = {
                    "raw_path": save_path,
                    "file_bytes": b_content,
                    "parsed_peaks": None,
                    "assigned_df": None,
                    "raw_df": None,
                }
                ensure_sample_peaks(s_new)
                st.session_state["spectra_db"][uf.name] = s_new
                latest_uploaded_name = uf.name

            if latest_uploaded_name and st.session_state.get("_last_selected_upload") != latest_uploaded_name:
                st.session_state["active_spectrum_selector"] = latest_uploaded_name
                st.session_state["_last_selected_upload"] = latest_uploaded_name

        # Карточка постоянного реестра спектров в текущей сессии
        total_in_db = len(st.session_state.get("spectra_db", {}))
        if total_in_db > 0:
            with st.expander(
                f"💾 {'Спектры в памяти сессии' if lang == 'ru' else 'Spectra in Session'}: **{total_in_db}**",
                expanded=True
            ):
                st.caption(
                    "Файлы сохраняются в сессии и доступны во всех вкладках платформы, даже если поле загрузчика выше очистилось."
                    if lang == "ru" else
                    "Files are permanently stored in session and accessible across all tabs even if the uploader above resets."
                )
                for s_name, s_obj in list(st.session_state["spectra_db"].items()):
                    c_col1, c_col2 = st.columns([3, 1])
                    with c_col1:
                        p_len = len(s_obj["parsed_peaks"]) if (s_obj.get("parsed_peaks") is not None and not s_obj["parsed_peaks"].empty) else "—"
                        has_form = "🧪" if (s_obj.get("assigned_df") is not None and not s_obj["assigned_df"].empty) else ""
                        is_cur = "👉 " if s_name == st.session_state.get("active_spectrum_selector") else ""
                        st.markdown(f"**{is_cur}`{s_name}`** ({p_len} {has_form})")
                    with c_col2:
                        if st.button("🗑️", key=f"del_quick_{s_name}", help="Удалить" if lang == "ru" else "Delete"):
                            if os.path.exists(s_obj.get("raw_path", "")):
                                try: os.remove(s_obj["raw_path"])
                                except Exception: pass
                            del st.session_state["spectra_db"][s_name]
                            if st.session_state.get("active_spectrum_selector") == s_name:
                                remaining = list(st.session_state["spectra_db"].keys())
                                st.session_state["active_spectrum_selector"] = remaining[0] if remaining else None
                            st.rerun()

        with st.expander(T[lang]["folder_expander"], expanded=False):
            local_folder = st.text_input(
                T[lang]["folder_input"],
                placeholder="C:/data/spectra" if lang == "en" else "например, C:/data/spectra",
                help=T[lang]["folder_help"],
            )
            if st.button(T[lang]["folder_btn"]):
                if local_folder and os.path.isdir(local_folder):
                    added_count = 0
                    for root, _, files in os.walk(local_folder):
                        for file in files:
                            if file.lower().endswith((".csv", ".txt", ".tsv", ".xy")) and file not in st.session_state["spectra_db"]:
                                src_path = os.path.join(root, file)
                                dst_path = os.path.join(STORAGE_DIR, file)
                                try:
                                    with open(src_path, "rb") as sf: content = sf.read()
                                    with open(dst_path, "wb") as df: df.write(content)
                                    s_dir_new = {
                                        "raw_path": dst_path, "file_bytes": content,
                                        "parsed_peaks": None, "assigned_df": None, "raw_df": None,
                                    }
                                    ensure_sample_peaks(s_dir_new)
                                    st.session_state["spectra_db"][file] = s_dir_new
                                    added_count += 1
                                except Exception:
                                    pass
                    st.success(T[lang]["folder_success"].format(n=added_count))
                    st.rerun()
                else:
                    st.error(T[lang]["folder_err"])

        with st.expander(T[lang]["batch_zip_expander"], expanded=False):
            batch_zip_file = st.file_uploader(
                T[lang]["batch_zip_uploader"],
                type=["zip"],
                key="batch_zip_uploader_key",
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                b_ion = st.selectbox(T[lang]["batch_ion_mode"], ["ESI(-)", "ESI(+)"], index=0, key="batch_ion_mode_key")
                b_ppm = st.number_input(T[lang]["batch_ppm_tol"], min_value=0.1, max_value=20.0, value=2.0, step=0.5, key="batch_ppm_key")
            with col_b2:
                b_min_int = st.number_input(T[lang]["batch_min_int"], min_value=0.0, value=0.0, format="%.2e", key="batch_min_int_key")
                b_mz_min = st.number_input("m/z min:", min_value=50.0, max_value=2000.0, value=150.0, step=50.0, key="batch_mz_min_key")
            b_mz_max = st.number_input("m/z max:", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0, key="batch_mz_max_key")

            if batch_zip_file is not None:
                if st.button(T[lang]["batch_run_btn"], type="primary", key="btn_run_batch_zip"):
                    with st.spinner("Выполняется пакетная обработка спектров..." if lang == "ru" else "Batch processing spectra..."):
                        summary_df, processed_dict = batch_process_fticr_spectra(
                            batch_zip_file.getvalue(),
                            ion_mode=b_ion,
                            ppm_tolerance=b_ppm,
                            min_intensity=b_min_int,
                            mz_range=(b_mz_min, b_mz_max),
                            lang=lang,
                        )
                        if not summary_df.empty:
                            for s_name, s_data in processed_dict.items():
                                save_path = os.path.join(STORAGE_DIR, s_name)
                                try:
                                    with open(save_path, "wb") as f_out:
                                        f_out.write(s_data["file_bytes"])
                                except Exception:
                                    pass
                                s_data["raw_path"] = save_path
                                st.session_state["spectra_db"][s_name] = s_data

                            st.session_state["fticr_batch_summary"] = summary_df
                            st.success(T[lang]["batch_success"].format(n=len(summary_df)))
                        else:
                            st.warning("В архиве не найдено подходящих спектральных файлов или не удалось приписать формулы." if lang == "ru" else "No valid spectra found in archive or formula assignment failed.")

            if "fticr_batch_summary" in st.session_state and not st.session_state["fticr_batch_summary"].empty:
                b_sum = st.session_state["fticr_batch_summary"]
                st.markdown(f"**{T[lang]['batch_table_title']}**")
                st_df(b_sum.head(5))
                st.download_button(
                    label=T[lang]["batch_dl_csv"],
                    data=b_sum.to_csv(index=False).encode("utf-8"),
                    file_name="chemo_fticr_block.csv",
                    mime="text/csv",
                    key="dl_batch_fticr_summary_btn",
                )
                if st.button(T[lang]["batch_transfer_ml_btn"], key="btn_transfer_batch_to_ml"):
                    st.session_state["fticr_ml_descriptors"] = b_sum
                    st.success(T[lang]["batch_transfer_success"])

        all_spectra = list(st.session_state["spectra_db"].keys())
        if all_spectra:
            st.markdown("---")
            search_query = st.text_input(T[lang]["search_label"], placeholder=T[lang]["search_ph"]).strip().lower()
            available_spectra = [s for s in all_spectra if search_query in s.lower()] if search_query else all_spectra

            if not available_spectra:
                st.warning(T[lang]["search_empty"])
                active_spectrum_name, current_sample = None, None
            else:
                curr_sel = st.session_state.get("active_spectrum_selector")
                sel_idx = available_spectra.index(curr_sel) if (curr_sel and curr_sel in available_spectra) else 0
                active_spectrum_name = st.selectbox(
                    T[lang]["active_sample"],
                    options=available_spectra,
                    index=sel_idx,
                    key="active_spectrum_selector"
                )
                current_sample = st.session_state["spectra_db"][active_spectrum_name]

                if st.button(T[lang]["del_sample"], type="secondary"):
                    if os.path.exists(current_sample["raw_path"]):
                        try: os.remove(current_sample["raw_path"])
                        except Exception: pass
                    del st.session_state["spectra_db"][active_spectrum_name]
                    remaining = list(st.session_state["spectra_db"].keys())
                    st.session_state["active_spectrum_selector"] = remaining[0] if remaining else None
                    st.rerun()

                with st.expander(T[lang]["parse_expander"], expanded=False):
                    delim_opts = ["Auto" if lang == "en" else "Авто (автоопределение)", "Tab (\\t)" if lang == "en" else "Табуляция (\\t)",
                                  "Comma (,)" if lang == "en" else "Запятая (,)", "Semicolon (;)" if lang == "en" else "Точка с запятой (;)",
                                  "Space" if lang == "en" else "Пробел"]
                    delimiter = st.selectbox(T[lang]["delimiter"], delim_opts, index=0, key=f"delim_{active_spectrum_name}")
                    decimal_sep = st.selectbox(T[lang]["decimal_sep"], [".", ","], index=0, key=f"dec_{active_spectrum_name}")
                    has_header = st.checkbox(T[lang]["has_header"], value=True, key=f"head_{active_spectrum_name}")

                try:
                    raw_df = parse_uploaded_file(current_sample["file_bytes"], delimiter, decimal_sep, has_header)
                    current_sample["raw_df"] = raw_df
                    if raw_df is None or raw_df.empty:
                        st.warning("Внимание: Файл пуст или формат не распознан. Проверьте параметры разделителя." if lang == "ru" else "Warning: File is empty or failed to parse. Check delimiter settings.")
                        current_sample["parsed_peaks"] = None
                    else:
                        st.write(T[lang]["preview_caption"])
                        st_df(raw_df.head(5))

                        col_names = list(raw_df.columns)
                        def_mz_idx = 0
                        def_int_idx = 1 if len(col_names) > 1 else 0
                        for idx, cname in enumerate(col_names):
                            cn_low = str(cname).lower()
                            if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]): def_mz_idx = idx
                            elif any(k in cn_low for k in ["int", "i", "count", "abund"]): def_int_idx = idx

                        col_mz = st.selectbox(T[lang]["col_mz"], col_names, index=def_mz_idx, key=f"mzcol_{active_spectrum_name}")
                        col_int = st.selectbox(T[lang]["col_int"], col_names, index=def_int_idx, key=f"intcol_{active_spectrum_name}")
                        if col_mz == col_int: st.warning(T[lang]["same_col_warn"])

                        st.markdown("---")
                        st.subheader(T[lang]["filtering_header"])

                        norm_choice = st.selectbox(T[lang]["norm_mode_label"], [T[lang]["norm_base"], T[lang]["norm_tic"], T[lang]["norm_raw"]], index=0, key=f"norm_mode_{active_spectrum_name}")
                        clean_mass = pd.to_numeric(raw_df[col_mz], errors="coerce")
                        clean_int = pd.to_numeric(raw_df[col_int], errors="coerce")
                        valid_mask = clean_mass.notna() & clean_int.notna()

                        valid_df = pd.DataFrame({"mass": clean_mass[valid_mask].astype(float), "intensity": clean_int[valid_mask].astype(float)})
                        valid_df = valid_df[valid_df["mass"] > 0]
                        if valid_df.empty:
                            st.warning("Внимание: В выбранных колонках не найдено положительных числовых данных." if lang == "ru" else "Warning: No valid numeric data found in selected columns.")
                            current_sample["parsed_peaks"] = None
                        else:
                            min_m_data = float(valid_df["mass"].min())
                            max_m_data = float(valid_df["mass"].max())
                            min_m_bound = max(50.0, float(np.floor(min_m_data)))
                            max_m_bound = max(min_m_bound + 10.0, min(2500.0, float(np.ceil(max_m_data))))
                            def_low = max(min_m_bound, min(100.0, max_m_bound - 1.0))
                            def_high = min(max_m_bound, max(def_low + 1.0, min(max_m_bound, 1200.0)))
                            mz_range = st.slider(T[lang]["mz_range"], min_value=min_m_bound,
                                                 max_value=max_m_bound,
                                                 value=(def_low, def_high),
                                                 step=10.0, key=f"mzrange_{active_spectrum_name}")

                            max_int_data = float(valid_df["intensity"].max())
                            cutoff_intensity = st.number_input(T[lang]["cutoff_int"], min_value=0.0, max_value=max(1.0, max_int_data), value=0.0,
                                                               step=max(1e-4, max_int_data * 0.001), format="%.2e", key=f"cutoff_{active_spectrum_name}")

                            filtered_df = valid_df[(valid_df["mass"] >= mz_range[0]) & (valid_df["mass"] <= mz_range[1]) & (valid_df["intensity"] >= cutoff_intensity)].sort_values("mass").reset_index(drop=True)
                            if not filtered_df.empty:
                                if norm_choice == T[lang]["norm_base"]:
                                    filtered_df["norm_intensity"] = (filtered_df["intensity"] / filtered_df["intensity"].max()) * 100.0
                                elif norm_choice == T[lang]["norm_tic"]:
                                    filtered_df["norm_intensity"] = (filtered_df["intensity"] / filtered_df["intensity"].sum()) * 100.0
                                else:
                                    filtered_df["norm_intensity"] = filtered_df["intensity"]
                            else:
                                filtered_df["norm_intensity"] = []

                            # Сохраняем полиномиальную калибровку если она активна
                            if current_sample.get("is_recalibrated") and current_sample.get("recal_poly") is not None:
                                filtered_df["mass"] = filtered_df["mass"] - current_sample["recal_poly"](filtered_df["mass"])
                            elif current_sample.get("is_recalibrated") and current_sample.get("parsed_peaks") is not None and len(current_sample["parsed_peaks"]) == len(filtered_df):
                                filtered_df["mass"] = current_sample["parsed_peaks"]["mass"]

                            current_sample["parsed_peaks"] = filtered_df
                            st.success(T[lang]["loaded_peaks_success"].format(n=len(filtered_df)))
                except Exception as err:
                    st.warning(f"Ошибка при разборе файла ({err}). Проверьте настройки разделителя." if lang == "ru" else f"File parse error ({err}). Check delimiter.")
                    current_sample["parsed_peaks"] = None
        else:
            active_spectrum_name, current_sample = None, None
            st.info(T[lang]["no_spectra_info"])

    st.title(T[lang]["title"])

    # Единая панель состояния активного образца (Sample Toolbar)
    if active_spectrum_name and current_sample:
        p_count = len(current_sample["parsed_peaks"]) if (current_sample.get("parsed_peaks") is not None and not current_sample["parsed_peaks"].empty) else 0
        as_df = current_sample.get("assigned_df")
        as_count = len(as_df) if (as_df is not None and not as_df.empty) else 0
        is_recal = current_sample.get("is_recalibrated", False)

        recal_chip = f'<span class="sample-chip green">✓ {"Рекалиброван" if lang == "ru" else "Recalibrated"}</span>' if is_recal else ""
        assign_chip = f'<span class="sample-chip green">🧬 {as_count:,} {"формул" if lang == "ru" else "formulas"} ({(as_count/p_count*100):.1f}%)</span>' if (as_count > 0 and p_count > 0) else f'<span class="sample-chip orange">⚠️ {"Формулы не приписаны" if lang == "ru" else "Formulas not assigned"}</span>'

        sample_tb_html = (
            f'<div class="sample-toolbar">'
            f'<div><span style="font-size: 13.5px; font-weight: 600; color: #1c2833;">📁 {active_spectrum_name}</span></div>'
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap;">'
            f'<span class="sample-chip">📊 {p_count:,} {"пиков" if lang == "ru" else "peaks"}</span>'
            f'{recal_chip}'
            f'{assign_chip}'
            f'</div>'
            f'</div>'
        )
        st.markdown(sample_tb_html, unsafe_allow_html=True)

    stage1, stage2, stage3 = st.tabs([
        T[lang]["tab_stage1"],
        T[lang]["tab_stage2"],
        T[lang]["tab_stage3"],
    ])

    # =========================================================================
    # ЭТАП 1: ПЕРВИЧНЫЙ СПЕКТР И ОБРАБОТКА
    # =========================================================================
    with stage1:
        st_sub1, st_sub2 = st.tabs([T[lang]["tab_stick"], T[lang]["tab_recal"]])

    # 1. Stick Plot
    with st_sub1:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_stick']}: {active_spectrum_name}")
            c_ctrl1, c_ctrl2 = st.columns([3, 1])
            with c_ctrl1: st.caption(T[lang]["stick_signals"].format(n=len(peaks_df)))
            with c_ctrl2: annotate_top = st.checkbox(T[lang]["annotate_top"], value=True, key=f"ann_{active_spectrum_name}")

            m_vals, i_vals = peaks_df["mass"].values, peaks_df["norm_intensity"].values
            x_stick, y_stick = np.empty(len(m_vals) * 3), np.empty(len(m_vals) * 3)
            x_stick[0::3], x_stick[1::3], x_stick[2::3] = m_vals, m_vals, None
            y_stick[0::3], y_stick[1::3], y_stick[2::3] = 0, i_vals, None

            fig_stick = go.Figure()
            fig_stick.add_trace(go.Scattergl(x=x_stick, y=y_stick, mode="lines", line=dict(color="#0b5394", width=1.1), hoverinfo="skip", name=T[lang]["peak_trace"]))
            if annotate_top and len(peaks_df) > 0:
                top5 = peaks_df.nlargest(5, "norm_intensity")
                fig_stick.add_trace(go.Scattergl(
                    x=top5["mass"], y=top5["norm_intensity"], mode="markers+text",
                    text=[f"{val:.4f}" for val in top5["mass"]], textposition="top center",
                    marker=dict(color="#b45f06", size=7), textfont=dict(color="#b45f06", size=10),
                    name=T[lang]["top5_trace"], hovertemplate="<b>m/z</b>: %{x:.4f}<br><b>Int</b>: %{y:.1f}<extra></extra>",
                ))
            y_max_plot = float(peaks_df["norm_intensity"].max() * 1.15) if not peaks_df.empty else 100.0
            fig_stick.update_layout(xaxis=dict(title=T[lang]["mz_axis"], gridcolor="#f1f3f5"), yaxis=dict(title=T[lang]["rel_int_axis"], range=[0, y_max_plot], gridcolor="#f1f3f5"),
                                    plot_bgcolor="white", height=500, margin=dict(l=45, r=30, t=30, b=40))
            st_plotly(fig_stick)

    # 2. Рекалибровка
    with st_sub2:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_recal']}: {active_spectrum_name}")
            cal_opts = ["Fatty Acids (C12–C33 saturated FA)" if lang == "en" else "Жирные кислоты (C12–C33 насыщенные ЖК)",
                        "CHO Homologues (CₙH₂ₙ₋₈O₇)" if lang == "en" else "Гомологи CHO (CₙH₂ₙ₋₈O₇)"]
            c_r1, c_r2, c_r3 = st.columns([2, 1.5, 1.5])
            with c_r1: calib_type = st.selectbox(T[lang]["recal_cal_set"], cal_opts, index=0, key=f"calibtype_{active_spectrum_name}")
            with c_r2: search_tol = st.number_input(T[lang]["recal_tol"], min_value=2.0, max_value=25.0, value=8.0, step=0.5, key=f"stol_{active_spectrum_name}")
            with c_r3: poly_order = st.selectbox(T[lang]["recal_poly"], [1, 2], index=1, key=f"polyord_{active_spectrum_name}")

            calib_ion_mode = st.selectbox(T[lang]["recal_ion"], ["ESI(-) [M - H]⁻", "ESI(+) [M + H]⁺", "Neutral masses [M]" if lang == "en" else "Нейтральные массы [M]"], index=0, key=f"calib_ion_{active_spectrum_name}")
            calib_lib = get_calibrant_library(calib_type, calib_ion_mode)
            exp_masses = peaks_df["mass"].values
            matched_calibs = []

            for _, row in calib_lib.iterrows():
                m_th = row["m_theor"]
                delta_lim = m_th * (search_tol * 1e-6)
                candidates = exp_masses[(exp_masses >= m_th - delta_lim) & (exp_masses <= m_th + delta_lim)]
                if len(candidates) > 0:
                    best_m = candidates[np.argmin(np.abs(candidates - m_th))]
                    matched_calibs.append({"Calibrant": row["name"], "m_theor": m_th, "m_exp": best_m, "delta_m": best_m - m_th, "error_ppm": ((best_m - m_th) / m_th) * 1e6})

            if len(matched_calibs) < 3:
                st.warning(T[lang]["recal_few_peaks"].format(n=len(matched_calibs)))
            else:
                calib_df = pd.DataFrame(matched_calibs)
                st.write(T[lang]["recal_found"].format(n=len(calib_df)))
                actual_poly_order = 1 if (poly_order > 1 and (float(calib_df["m_exp"].max() - calib_df["m_exp"].min()) / max(1.0, float(peaks_df["mass"].max() - peaks_df["mass"].min()))) < 0.60) else poly_order

                poly_fn = np.poly1d(np.polyfit(calib_df["m_exp"].values, calib_df["delta_m"].values, deg=actual_poly_order))
                residual_ppm = (((calib_df["m_exp"] - poly_fn(calib_df["m_exp"])) - calib_df["m_theor"]) / calib_df["m_theor"]) * 1e6

                fig_rec, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)
                ax1.scatter(calib_df["m_exp"], calib_df["error_ppm"], color="#D62728", s=30, label=T[lang]["recal_scatter_before"])
                m_grid = np.linspace(peaks_df["mass"].min(), peaks_df["mass"].max(), 200)
                ax1.plot(m_grid, (poly_fn(m_grid) / m_grid) * 1e6, color="black", linestyle="--", linewidth=1.2)
                ax1.set_xlabel("m/z"); ax1.set_ylabel(T[lang]["ppm_axis"]); ax1.set_title(T[lang]["recal_title_before"]); ax1.grid(True, alpha=0.3)
                ax2.scatter(calib_df["m_exp"], residual_ppm, color="#2CA02C", s=30, label=T[lang]["recal_scatter_after"])
                ax2.axhline(0, color="black", linewidth=0.8); ax2.set_xlabel("m/z"); ax2.set_ylabel(T[lang]["res_ppm_axis"]); ax2.set_title(T[lang]["recal_title_after"]); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_rec)
                plt.close(fig_rec)

                c_btn_r1, c_btn_r2 = st.columns([2, 1])
                with c_btn_r1:
                    if st.button(T[lang]["recal_btn"], type="primary", key=f"btn_recal_{active_spectrum_name}"):
                        current_sample["parsed_peaks"]["mass"] = peaks_df["mass"].values - poly_fn(peaks_df["mass"].values)
                        current_sample["recal_poly"] = poly_fn
                        current_sample["is_recalibrated"] = True
                        current_sample["assigned_df"] = None
                        st.success(T[lang]["recal_success"].format(n=len(peaks_df), before=calib_df["error_ppm"].abs().mean(), after=residual_ppm.abs().mean()))
                with c_btn_r2:
                    if current_sample.get("is_recalibrated"):
                        if st.button("Сбросить калибровку" if lang == "ru" else "Reset Calibration", key=f"btn_reset_recal_{active_spectrum_name}"):
                            current_sample["is_recalibrated"] = False
                            current_sample["recal_poly"] = None
                            current_sample["assigned_df"] = None
                            st.rerun()

                if current_sample.get("is_recalibrated"):
                    st.info("ℹ️ Для данного спектра активна полиномиальная рекалибровка (сохраняется при смене любых вкладок)." if lang == "ru" else "ℹ️ Polynomial recalibration is active for this spectrum (preserved across all tabs).")

    # =========================================================================
    # ЭТАП 2: ИДЕНТИФИКАЦИЯ И ХЕМОТИПИРОВАНИЕ
    # =========================================================================
    with stage2:
        st_sub_assign, st_sub_vk, st_sub_kmd, st_sub_vk20 = st.tabs([
            T[lang]["tab_assign"],
            T[lang]["tab_vk"],
            T[lang]["tab_kmd"],
            T[lang]["tab_vk20"],
        ])

    # 3. Приписывание формул
    with st_sub_assign:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_assign']}: {active_spectrum_name}")
            has_assigned_already = current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty
            with st.expander(T[lang]["assign_expander"], expanded=not has_assigned_already):
                col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
                with col_m1: ion_mode = st.selectbox(T[lang]["ion_mode"], ["ESI(-) [M - H]⁻", "ESI(+) [M + H]⁺", "Neutral masses [M]" if lang == "en" else "Нейтральные массы [M]"], index=0, key=f"ionmode_{active_spectrum_name}")
                with col_m2: max_charge_val = st.selectbox(T[lang]["max_charge"], [1, 2], index=0, key=f"zmax_{active_spectrum_name}")
                with col_m3: ppm_tol = st.number_input(T[lang]["ppm_tol"], min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"ppmtol_{active_spectrum_name}")

                col_el1, col_el2, col_el3 = st.columns(3)
                with col_el1:
                    c_bounds = st.slider(T[lang]["c_lim"], 4, 120, (4, 120), key=f"cb_{active_spectrum_name}")
                    h_bounds = st.slider(T[lang]["h_lim"], 4, 200, (4, 200), key=f"hb_{active_spectrum_name}")
                with col_el2:
                    o_bounds = st.slider(T[lang]["o_lim"], 1, 60, (1, 60), key=f"ob_{active_spectrum_name}")
                    n_bounds = st.slider(T[lang]["n_lim"], 0, 3, (0, 2), key=f"nb_{active_spectrum_name}")
                with col_el3:
                    s_bounds = st.slider(T[lang]["s_lim"], 0, 2, (0, 1), key=f"sb_{active_spectrum_name}")
                    max_oc_val = st.slider(T[lang]["max_oc"], 0.1, 1.0, 1.0, step=0.05, key=f"moc_{active_spectrum_name}")
                    max_hc_val = st.slider(T[lang]["max_hc"], 0.2, 2.2, 2.0, step=0.05, key=f"mhc_{active_spectrum_name}")

                col_iso1, col_iso2 = st.columns(2)
                with col_iso1: iso_chk = st.checkbox(T[lang]["iso_filter_label"], value=False, key=f"isochk_{active_spectrum_name}")
                with col_iso2: iso_strict_chk = st.checkbox(T[lang]["iso_strict_label"], value=False, disabled=not iso_chk, key=f"isostrict_{active_spectrum_name}")

            if st.button(T[lang]["assign_btn"], type="primary", key=f"btn_assign_{active_spectrum_name}"):
                with st.spinner(T[lang]["assign_spinner"]):
                    assigned_res = run_formula_assignment(
                        peaks_df=peaks_df, bounds={"C": c_bounds, "H": h_bounds, "O": o_bounds, "N": n_bounds, "S": s_bounds},
                        max_hc=max_hc_val, max_oc=max_oc_val, ppm_tolerance=ppm_tol,
                        ion_mode=ion_mode, max_charge=max_charge_val, iso_check=iso_chk, iso_strict=iso_strict_chk, lang=lang
                    )
                    current_sample["assigned_df"] = assigned_res

            assigned_data = current_sample["assigned_df"]
            if assigned_data is not None and not assigned_data.empty:
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(T[lang]["metric_total"], f"{len(peaks_df):,}")
                m2.metric(T[lang]["metric_assigned"], f"{len(assigned_data):,}")
                m3.metric(T[lang]["metric_rate"], f"{(len(assigned_data) / len(peaks_df) * 100.0):.1f}%")
                m4.metric(T[lang]["metric_err"], f"{assigned_data['error_ppm'].abs().mean():.3f}")
                st_df(assigned_data)
                st.download_button(label=T[lang]["dl_csv_formulas"], data=assigned_data.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{active_spectrum_name}_formulas.csv", mime="text/csv", key=f"dl_csv_formulas_{active_spectrum_name}")

                if CHEMO_PUBCHEM_AVAILABLE:
                    st.markdown("---")
                    with st.expander(T[lang]["pubchem_inspect_expander"], expanded=False):
                        top_formulas = assigned_data.sort_values("intensity", ascending=False)["Formula"].head(60).tolist()
                        col_pc1, col_pc2 = st.columns([2, 2])
                        with col_pc1:
                            sel_f = st.selectbox(T[lang]["pubchem_select_formula"], top_formulas, key=f"pc_sel_{active_spectrum_name}")
                        with col_pc2:
                            cust_f = st.text_input(T[lang]["pubchem_custom_input"], value="", placeholder="e.g. C20H30O2", key=f"pc_cust_{active_spectrum_name}")

                        target_f = cust_f.strip() if cust_f.strip() else sel_f
                        col_btn_pc1, col_btn_pc2 = st.columns([1, 2])
                        with col_btn_pc1:
                            search_clicked = st.button(T[lang]["pubchem_search_btn"], key=f"btn_pc_search_{active_spectrum_name}")

                        # Автоматическая загрузка при смене формулы или по клику на кнопку
                        cached_target = st.session_state.get(f"pc_last_f_{active_spectrum_name}")
                        if (search_clicked or (target_f and target_f != cached_target)):
                            with st.spinner(T[lang]["pubchem_searching"]):
                                pc_matches = chemo_pubchem.lookup_formula_in_pubchem(target_f, max_records=4, timeout=4.0)
                                st.session_state[f"pc_res_list_{active_spectrum_name}"] = pc_matches if pc_matches else []
                                st.session_state[f"pc_last_f_{active_spectrum_name}"] = target_f

                        pc_matches_list = st.session_state.get(f"pc_res_list_{active_spectrum_name}", [])
                        if pc_matches_list:
                            st.caption(f"{T[lang]['pubchem_isomers_label']} **{len(pc_matches_list)}**")
                            clean_t_f = chemo_pubchem.clean_formula(target_f) if hasattr(chemo_pubchem, "clean_formula") else target_f
                            chembl_search_url = f"https://www.ebi.ac.uk/chembl/search_results/{urllib.parse.quote(clean_t_f)}"
                            hmdb_search_url = f"https://hmdb.ca/unearth/q?query={urllib.parse.quote(clean_t_f)}&searcher=metabolites"

                            ext_c1, ext_c2, _ = st.columns([1.5, 1.5, 3])
                            with ext_c1:
                                st.link_button(T[lang]["pubchem_chembl_btn"], chembl_search_url)
                            with ext_c2:
                                st.link_button(T[lang]["pubchem_hmdb_btn"], hmdb_search_url)

                            for idx, pc_item in enumerate(pc_matches_list, 1):
                                item_title = pc_item.get("Title") or pc_item.get("name", f"Isomer #{idx}")
                                with st.container(border=True):
                                    pc_col_img, pc_col_meta = st.columns([1, 2.2])
                                    with pc_col_img:
                                        img_url = pc_item.get("Image_URL") or pc_item.get("image_url")
                                        if img_url:
                                            st.image(img_url, caption=f"2D: {item_title}", width="stretch")
                                        else:
                                            st.info("2D structure preview not available" if lang == "en" else "2D структура недоступна")
                                    with pc_col_meta:
                                        c_name = item_title
                                        c_iupac = pc_item.get("IUPACName") or pc_item.get("iupac_name", "N/A")
                                        c_class = pc_item.get("Class") or pc_item.get("class", "Unknown")
                                        c_smiles = pc_item.get("SMILES") or pc_item.get("smiles", "N/A")
                                        c_cid = pc_item.get("CID") or pc_item.get("cid")

                                        st.markdown(f"**{T[lang]['pubchem_res_name']}** `{c_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_iupac']}** {c_iupac}")
                                        st.markdown(f"**{T[lang]['pubchem_res_class']}** `{c_class}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_smiles']}** `{c_smiles}`")
                                        if c_cid:
                                            st.markdown(f"**{T[lang]['pubchem_res_cid']}** [{c_cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{c_cid})")
                                            st.link_button(f"{T[lang]['pubchem_link_btn']} #{c_cid}", f"https://pubchem.ncbi.nlm.nih.gov/compound/{c_cid}")
            elif assigned_data is not None:
                st.warning(T[lang]["no_formulas_warn"])

    # 4. Проекции (Ван Кревелен)
    with st_sub_vk:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is None or assigned_data.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            st.subheader(f"{T[lang]['tab_vk']}: {active_spectrum_name}")
            proj_type = st.radio(T[lang]["proj_mode"], [T[lang]["proj_vk"], T[lang]["proj_dbe_c"], T[lang]["proj_custom"]], horizontal=True, key=f"proj_type_{active_spectrum_name}")
            pow_exp = st.slider(T[lang]["pow_exp"], min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=f"pow_{active_spectrum_name}")
            palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}

            if proj_type == T[lang]["proj_vk"]:
                show_bioclasses = st.checkbox(T[lang]["vk_show_bioclasses"], value=True, key=f"vk_bio_{active_spectrum_name}")
                fig_vk = go.Figure()
                if show_bioclasses:
                    for reg_key, reg_info in VAN_KREVELEN_REGIONS.items():
                        oc_min, oc_max = reg_info["oc_range"]
                        hc_min, hc_max = reg_info["hc_range"]
                        fig_vk.add_shape(
                            type="rect",
                            x0=oc_min, x1=min(oc_max, 1.0),
                            y0=hc_min, y1=min(hc_max, 2.2),
                            fillcolor=reg_info["color"],
                            line=dict(color=reg_info["border"], width=1.5, dash="dot"),
                            layer="below",
                        )
                        label_name = reg_info[lang]
                        fig_vk.add_annotation(
                            x=(oc_min + min(oc_max, 1.0)) / 2.0,
                            y=(hc_min + min(hc_max, 2.2)) / 2.0,
                            text=f"<b>{label_name}</b>",
                            showarrow=False,
                            font=dict(size=10, color=reg_info["border"]),
                            opacity=0.75,
                        )

                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    if sub.empty: continue
                    fig_vk.add_trace(go.Scattergl(
                        x=sub["O/C"], y=sub["H/C"], mode="markers", name=f"{cls} ({len(sub):,})",
                        marker=dict(color=palette[cls], size=2.0 + 5.0 * (sub["norm_intensity"] / 100.0) ** pow_exp, opacity=0.65),
                        text=[f"<b>{r.get('Formula', '')}</b><br>m/z: {r['mass']:.4f}<br>AI: {r.get('AI', 0):.2f}" for _, r in sub.iterrows()], hoverinfo="text"
                    ))
                fig_vk.update_layout(xaxis=dict(title=T[lang]["oc_axis"], range=[0.0, 1.0], gridcolor="#f1f3f5"), yaxis=dict(title=T[lang]["hc_axis"], range=[0.2, 2.2], gridcolor="#f1f3f5"), plot_bgcolor="white", height=560)
                st_plotly(fig_vk)

                bio_dist = get_biomolecular_distribution(assigned_data, lang=lang)
                if bio_dist:
                    st.markdown(f"##### {T[lang]['vk_bioclass_dist_title']}")
                    b_col1, b_col2 = st.columns([1, 2])
                    with b_col1:
                        df_bio = pd.DataFrame([
                            {T[lang]["vk_bioclass_col"]: k, T[lang]["vk_bioclass_pct"]: v}
                            for k, v in bio_dist.items()
                        ])
                        st_df(df_bio)
                    with b_col2:
                        fig_bio = px.bar(
                            df_bio, x=T[lang]["vk_bioclass_col"], y=T[lang]["vk_bioclass_pct"],
                            color=T[lang]["vk_bioclass_col"],
                            text=T[lang]["vk_bioclass_pct"],
                            template="plotly_white",
                            height=280,
                        )
                        fig_bio.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_bio.update_layout(showlegend=False, yaxis_title="%", xaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
                        st_plotly(fig_bio)

                if CHEMO_PUBCHEM_AVAILABLE and "Formula" in assigned_data.columns:
                    st.markdown("---")
                    with st.expander(T[lang]["vk_inspect_title"], expanded=False):
                        st.caption(T[lang]["vk_inspect_caption"])
                        vk_formulas = assigned_data.sort_values("intensity", ascending=False)["Formula"].head(80).tolist()
                        v_col_sel, v_col_btn = st.columns([3, 1])
                        with v_col_sel:
                            sel_vk_f = st.selectbox(T[lang]["vk_inspect_select"], vk_formulas, key=f"vk_sel_f_{active_spectrum_name}")
                        with v_col_btn:
                            st.write("")
                            st.write("")
                            find_vk_clicked = st.button("🔍 " + T[lang]["pubchem_search_btn"].split()[-1], key=f"btn_vk_f_{active_spectrum_name}")

                        last_vk_f = st.session_state.get(f"vk_last_f_{active_spectrum_name}")
                        if find_vk_clicked or (sel_vk_f and sel_vk_f != last_vk_f):
                            with st.spinner(T[lang]["pubchem_searching"]):
                                vk_matches = chemo_pubchem.lookup_formula_in_pubchem(sel_vk_f, max_records=2, timeout=4.0)
                                st.session_state[f"vk_res_{active_spectrum_name}"] = vk_matches
                                st.session_state[f"vk_last_f_{active_spectrum_name}"] = sel_vk_f

                        vk_saved_res = st.session_state.get(f"vk_res_{active_spectrum_name}", [])
                        if vk_saved_res:
                            clean_vk_str = chemo_pubchem.clean_formula(sel_vk_f) if hasattr(chemo_pubchem, "clean_formula") else sel_vk_f
                            vk_chembl_url = f"https://www.ebi.ac.uk/chembl/search_results/{urllib.parse.quote(clean_vk_str)}"
                            vk_hmdb_url = f"https://hmdb.ca/unearth/q?query={urllib.parse.quote(clean_vk_str)}&searcher=metabolites"

                            ext_vk1, ext_vk2, _ = st.columns([1.5, 1.5, 3])
                            with ext_vk1:
                                st.link_button(T[lang]["pubchem_chembl_btn"], vk_chembl_url)
                            with ext_vk2:
                                st.link_button(T[lang]["pubchem_hmdb_btn"], vk_hmdb_url)

                            for idx, item in enumerate(vk_saved_res, 1):
                                with st.container(border=True):
                                    c_img, c_info = st.columns([1, 2.2])
                                    with c_img:
                                        img_u = item.get("Image_URL")
                                        if img_u:
                                            st.image(img_u, caption=f"2D: {item.get('Title', 'Compound')}", width="stretch")
                                        else:
                                            st.info("2D structure preview not available" if lang == "en" else "2D структура недоступна")
                                    with c_info:
                                        t_name = item.get("Title", "N/A")
                                        iupac_name = item.get("IUPACName", "N/A")
                                        cls_name = item.get("Class", "Unknown")
                                        smi = item.get("SMILES", "N/A")
                                        cid_val = item.get("CID")
                                        st.markdown(f"**{T[lang]['pubchem_res_name']}** `{t_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_iupac']}** {iupac_name}")
                                        st.markdown(f"**{T[lang]['pubchem_res_class']}** `{cls_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_smiles']}** `{smi}`")
                                        if cid_val:
                                            st.link_button(f"{T[lang]['pubchem_link_btn']} #{cid_val}", f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid_val}")
            elif proj_type == T[lang]["proj_dbe_c"]:
                fig_dbe = go.Figure()
                known_classes = ["CHO", "CHON", "CHOS", "CHONS"]
                has_hc = "Hetero_Class" in assigned_data.columns
                all_classes = [c for c in known_classes if has_hc and c in assigned_data["Hetero_Class"].values]
                if has_hc:
                    other_cls = [c for c in assigned_data["Hetero_Class"].dropna().unique() if c not in known_classes]
                    all_classes.extend(other_cls)
                if not all_classes:
                    all_classes = ["All"]

                for cls in all_classes:
                    if has_hc and cls != "All":
                        sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    else:
                        sub = assigned_data
                    if sub.empty or "C" not in sub.columns or "DBE" not in sub.columns:
                        continue
                    color = palette.get(cls, "#636EFA" if cls == "All" else "#8c564b")
                    if "norm_intensity" in sub.columns:
                        norm_vals = sub["norm_intensity"]
                    elif "intensity" in sub.columns and sub["intensity"].max() > 0:
                        norm_vals = (sub["intensity"] / sub["intensity"].max()) * 100.0
                    else:
                        norm_vals = pd.Series(50.0, index=sub.index)

                    hover_txt = [
                        f"<b>{r.get('Formula', '')}</b><br>C: {r.get('C', '')}<br>DBE: {r.get('DBE', '')}<br>m/z: {r.get('mass', 0):.4f}<br>AI: {r.get('AI', 0):.2f}"
                        for _, r in sub.iterrows()
                    ]
                    fig_dbe.add_trace(go.Scattergl(
                        x=sub["C"], y=sub["DBE"], mode="markers", name=f"{cls} ({len(sub):,})",
                        marker=dict(color=color, size=2.0 + 5.0 * (norm_vals / 100.0) ** pow_exp, opacity=0.65),
                        text=hover_txt, hoverinfo="text"
                    ))

                max_c = float(assigned_data["C"].max()) if "C" in assigned_data.columns and not assigned_data["C"].isna().all() else 60.0
                c_line = np.linspace(4, max(60.0, max_c), 100)
                fig_dbe.add_trace(go.Scatter(x=c_line, y=c_line * 0.9, mode="lines", name="Planar limit", line=dict(color="red", dash="dot")))
                fig_dbe.update_layout(xaxis=dict(title="Carbon (C)", gridcolor="#f1f3f5"), yaxis=dict(title="DBE", gridcolor="#f1f3f5"), plot_bgcolor="white", height=560)
                st_plotly(fig_dbe)
            else:
                candidate_cols = ["mass", "intensity", "norm_intensity", "C", "H", "O", "N", "S", "H/C", "O/C", "DBE", "AI", "NOSC", "error_ppm"]
                avail_cols = [c for c in candidate_cols if c in assigned_data.columns]
                if not avail_cols:
                    avail_cols = [c for c in assigned_data.columns if pd.api.types.is_numeric_dtype(assigned_data[c])]
                if not avail_cols:
                    avail_cols = list(assigned_data.columns)

                x_idx = avail_cols.index("mass") if "mass" in avail_cols else 0
                y_idx = avail_cols.index("DBE") if "DBE" in avail_cols else (1 if len(avail_cols) > 1 else 0)
                c_idx = avail_cols.index("AI") if "AI" in avail_cols else (2 if len(avail_cols) > 2 else 0)

                c1, c2, c3 = st.columns(3)
                with c1: cx = st.selectbox("X:", avail_cols, index=x_idx, key=f"cust_x_{active_spectrum_name}")
                with c2: cy = st.selectbox("Y:", avail_cols, index=y_idx, key=f"cust_y_{active_spectrum_name}")
                with c3: cc = st.selectbox("Color:", avail_cols, index=c_idx, key=f"cust_c_{active_spectrum_name}")

                hover_cols = ["Formula"] if "Formula" in assigned_data.columns else None
                fig_cust = px.scatter(
                    assigned_data, x=cx, y=cy, color=cc,
                    color_continuous_scale="Viridis", opacity=0.7, render_mode="webgl",
                    hover_data=hover_cols
                )
                fig_cust.update_layout(height=560, plot_bgcolor="white")
                st_plotly(fig_cust)

    # 5. KMD
    with st_sub_kmd:
        st.subheader(f"{T[lang]['tab_kmd']}: {active_spectrum_name if active_spectrum_name else ''}")
        assigned_df = current_sample.get("assigned_df") if current_sample else None
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
                base_choice = st.selectbox(T[lang]["kmd_base_group"], list(KMD_BASES.keys()), index=0, key=f"kmd_base_{active_spectrum_name}")
            with col_b2:
                color_options = [T[lang]["kmd_parity_mode"], T[lang]["kmd_int_mode"]]
                if has_formulas:
                    color_options.extend([T[lang]["kmd_o_mode"], T[lang]["kmd_dbe_mode"], T[lang]["kmd_hetero_mode"]])
                color_mode = st.selectbox(T[lang]["kmd_color_mode"], color_options, index=0, key=f"kmd_color_{active_spectrum_name}")
            with col_b3:
                point_sz = st.slider(T[lang]["kmd_pt_size"], 1, 8, 3, key=f"kmd_pt_{active_spectrum_name}")

            # compute_kmd возвращает (km, kmd, nkm)
            km, kmd, nkm = compute_kmd(work_df["mass"].values, base_choice)
            work_df["KM"] = km
            work_df["KMD"] = kmd
            work_df["NKM"] = nkm
            base_label = KMD_BASES[base_choice]["label"]

            hover_text = [
                f"<b>{r.get('Formula', 'N/A')}</b><br>m/z: {r['mass']:.4f}<br>NKM: {int(n)}<br>KMD: {d:.4f}"
                for r, n, d in zip(work_df.to_dict("records"), work_df["NKM"], work_df["KMD"])
            ]

            fig_kmd_inter = go.Figure()
            if color_mode == T[lang]["kmd_parity_mode"]:
                even_mask = (work_df["NKM"] % 2 == 0)
                even_idx = np.where(even_mask)[0]
                odd_idx = np.where(~even_mask)[0]
                fig_kmd_inter.add_trace(go.Scattergl(
                    x=work_df.loc[even_mask, "NKM"],
                    y=work_df.loc[even_mask, "KMD"],
                    mode="markers",
                    name=T[lang]["kmd_even"],
                    text=[hover_text[i] for i in even_idx],
                    hoverinfo="text",
                    marker=dict(color="#0020C2", size=point_sz, opacity=0.6),
                ))
                fig_kmd_inter.add_trace(go.Scattergl(
                    x=work_df.loc[~even_mask, "NKM"],
                    y=work_df.loc[~even_mask, "KMD"],
                    mode="markers",
                    name=T[lang]["kmd_odd"],
                    text=[hover_text[i] for i in odd_idx],
                    hoverinfo="text",
                    marker=dict(color="#FF7F0E", size=point_sz, opacity=0.7),
                ))
            elif color_mode == T[lang]["kmd_hetero_mode"] and has_formulas:
                palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}
                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub_mask = (work_df["Hetero_Class"] == cls)
                    sub_idx = np.where(sub_mask)[0]
                    if len(sub_idx) > 0:
                        fig_kmd_inter.add_trace(go.Scattergl(
                            x=work_df.loc[sub_mask, "NKM"],
                            y=work_df.loc[sub_mask, "KMD"],
                            mode="markers",
                            name=f"{cls} ({len(sub_idx):,})",
                            text=[hover_text[i] for i in sub_idx],
                            hoverinfo="text",
                            marker=dict(color=palette.get(cls, "#7F7F7F"), size=point_sz, opacity=0.65),
                        ))
            else:
                if color_mode == T[lang]["kmd_o_mode"] and has_formulas:
                    c_vals = work_df["O"]
                    cbar_title = "O"
                    c_scale = "Viridis"
                elif color_mode == T[lang]["kmd_dbe_mode"] and has_formulas:
                    c_vals = work_df["DBE"]
                    cbar_title = "DBE"
                    c_scale = "Plasma"
                else:
                    c_vals = work_df["norm_intensity"]
                    cbar_title = T[lang]["kmd_int_mode"]
                    c_scale = "Cividis"

                fig_kmd_inter.add_trace(go.Scattergl(
                    x=work_df["NKM"],
                    y=work_df["KMD"],
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        color=c_vals,
                        colorscale=c_scale,
                        size=point_sz,
                        opacity=0.65,
                        colorbar=dict(title=cbar_title),
                    ),
                ))

            fig_kmd_inter.update_layout(
                xaxis=dict(title=T[lang]["kmd_x_axis"].format(base=base_label), gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                yaxis=dict(title=T[lang]["kmd_y_axis"].format(base=base_label), range=[-0.02, 1.02], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                plot_bgcolor="white",
                height=540,
                margin=dict(l=45, r=30, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.caption(T[lang]["nav_caption"])
            st_plotly(fig_kmd_inter)

            # Matplotlib экспорт
            fig_kmd_mpl, ax_kmd_mpl = plt.subplots(figsize=(12, 6.5), dpi=150)
            if color_mode == T[lang]["kmd_parity_mode"]:
                even_mask = (work_df["NKM"] % 2 == 0)
                ax_kmd_mpl.scatter(work_df.loc[even_mask, "NKM"], work_df.loc[even_mask, "KMD"],
                                   c="#0020C2", label=T[lang]["kmd_even"], s=point_sz, alpha=0.6, edgecolors="none", rasterized=True)
                ax_kmd_mpl.scatter(work_df.loc[~even_mask, "NKM"], work_df.loc[~even_mask, "KMD"],
                                   c="#FF7F0E", label=T[lang]["kmd_odd"], s=point_sz, alpha=0.7, edgecolors="none", rasterized=True)
                ax_kmd_mpl.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
            elif color_mode == T[lang]["kmd_hetero_mode"] and has_formulas:
                palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}
                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = work_df[work_df["Hetero_Class"] == cls]
                    if not sub.empty:
                        ax_kmd_mpl.scatter(sub["NKM"], sub["KMD"], c=palette.get(cls, "#7F7F7F"), label=f"{cls} ({len(sub):,})",
                                           s=point_sz, alpha=0.65, edgecolors="none", rasterized=True)
                ax_kmd_mpl.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
            else:
                sc = ax_kmd_mpl.scatter(work_df["NKM"], work_df["KMD"], c=c_vals, cmap=c_scale.lower(),
                                        s=point_sz, alpha=0.65, edgecolors="none", rasterized=True)
                cbar = plt.colorbar(sc, ax=ax_kmd_mpl, pad=0.015, aspect=25)
                cbar.set_label(cbar_title, fontsize=10)

            ax_kmd_mpl.set_ylim(-0.02, 1.02)
            ax_kmd_mpl.set_xlabel(T[lang]["kmd_x_axis"].format(base=base_label), fontsize=11, fontweight="medium")
            ax_kmd_mpl.set_ylabel(T[lang]["kmd_y_axis"].format(base=base_label), fontsize=11, fontweight="medium")
            ax_kmd_mpl.set_title(f"Kendrick Mass Defect Plot ({base_label}) — {len(work_df):,} peaks", fontsize=12, pad=10)
            ax_kmd_mpl.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
            plt.tight_layout()

            get_plot_download_buttons(fig_kmd_mpl, f"{active_spectrum_name}_kmd_{base_label}", lang)
            plt.close(fig_kmd_mpl)

            # Feature 2: KMD Homologous Series Slicer
            with st.expander(T[lang]["kmd_slicer_title"], expanded=False):
                col_sl1, col_sl2, col_sl3 = st.columns([1, 1, 1])
                with col_sl1:
                    enable_slicer = st.checkbox(T[lang]["kmd_slicer_enable"], value=False, key=f"kmd_sl_chk_{active_spectrum_name}")
                with col_sl2:
                    kmd_min_actual = float(work_df["KMD"].min()) if not work_df.empty else -1.0
                    kmd_max_actual = float(work_df["KMD"].max()) if not work_df.empty else 1.0
                    med_kmd = float(work_df["KMD"].median()) if not work_df.empty else 0.0
                    target_kmd = st.number_input(
                        T[lang]["kmd_target_val"],
                        min_value=min(-1.0, np.floor(kmd_min_actual * 10) / 10),
                        max_value=max(1.0, np.ceil(kmd_max_actual * 10) / 10),
                        value=round(med_kmd, 3),
                        step=0.01,
                        format="%.3f",
                        disabled=not enable_slicer,
                        key=f"kmd_sl_target_{active_spectrum_name}",
                    )
                with col_sl3:
                    tol_kmd = st.number_input(
                        T[lang]["kmd_tol_val"],
                        min_value=0.001,
                        max_value=0.100,
                        value=0.010,
                        step=0.002,
                        format="%.3f",
                        disabled=not enable_slicer,
                        key=f"kmd_sl_tol_{active_spectrum_name}",
                    )

                if enable_slicer:
                    kmd_mask = (work_df["KMD"] >= target_kmd - tol_kmd) & (work_df["KMD"] <= target_kmd + tol_kmd)
                    sliced_df = work_df[kmd_mask].sort_values("mass").reset_index(drop=True)
                    pct_sliced = (len(sliced_df) / len(work_df) * 100.0) if len(work_df) > 0 else 0.0

                    st.markdown(T[lang]["kmd_slicer_count"].format(n=len(sliced_df), total=len(work_df), pct=pct_sliced))
                    if not sliced_df.empty:
                        st_df(sliced_df)
                        st.download_button(
                            label=T[lang]["kmd_dl_series"],
                            data=sliced_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{active_spectrum_name}_kmd_{base_choice}_series_{target_kmd:.3f}.csv",
                            mime="text/csv",
                            key=f"dl_kmd_series_{active_spectrum_name}",
                        )

    # 6. Хемотипирование 20 ячеек
    with st_sub_vk20:
        st.subheader(f"{T[lang]['tab_vk20']}: {active_spectrum_name if active_spectrum_name else ''}")
        assigned_df = current_sample.get("assigned_df") if current_sample else None
        if assigned_df is None or assigned_df.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            pct_count, pct_weight, feat_df = compute_vk20_grid(assigned_df)
            metric_mode = st.radio(T[lang]["vk20_basis"], [T[lang]["vk20_opt_count"], T[lang]["vk20_opt_weight"]], horizontal=True, key=f"vk20_mode_{active_spectrum_name}")
            active_matrix = pct_count if metric_mode == T[lang]["vk20_opt_count"] else pct_weight

            fig_hm, ax_hm = plt.subplots(figsize=(9, 5.5), dpi=120)
            im = ax_hm.imshow(active_matrix, origin="lower", cmap="YlOrRd", aspect="auto")
            ax_hm.set_xticks(range(4)); ax_hm.set_xticklabels(["C1 [0-0.25)", "C2 [0.25-0.5)", "C3 [0.5-0.75)", "C4 [0.75-1.0]"], fontsize=8)
            ax_hm.set_yticks(range(5)); ax_hm.set_yticklabels(["R1 [0.2-0.6)", "R2 [0.6-1.0)", "R3 [1.0-1.4)", "R4 [1.4-1.8)", "R5 [1.8-2.2]"], fontsize=8)
            ax_hm.set_title(T[lang]["vk20_heatmap_title"].format(mode=metric_mode), fontsize=10)
            for r in range(5):
                for c in range(4):
                    val = active_matrix[r, c]
                    ax_hm.text(c, r, f"VK_{1 + r * 4 + c}\n{val:.1f}%", ha="center", va="center", color="white" if val > (active_matrix.max() * 0.65) else "black", fontsize=8, fontweight="bold")
            plt.colorbar(im, ax=ax_hm, pad=0.02)
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)
            st_df(feat_df)
            st.download_button(label=T[lang]["vk20_dl_csv"], data=feat_df.to_csv(index=False).encode("utf-8"), file_name=f"{active_spectrum_name}_vk20.csv", mime="text/csv", key=f"dl_vk20_{active_spectrum_name}")

    # =========================================================================
    # ЭТАП 3: СРАВНИТЕЛЬНЫЙ АНАЛИЗ И СЕТИ
    # =========================================================================
    with stage3:
        st_sub_cmp, st_sub_tmds, st_sub_desc = st.tabs([
            T[lang]["tab_cmp"],
            T[lang]["tab_tmds"],
            T[lang]["tab_desc"],
        ])

    # 7. Сравнение и алгебра спектров
    with st_sub_cmp:
        st.subheader(T[lang]["tab_cmp"])

        with st.popover("ℹ️ " + ("Руководство: Зачем нужна Алгебра спектров и как её применять" if lang == "ru" else "Guide: Spectrum Algebra & Comparative Analysis"), width="stretch"):
            st.markdown(r"""
### 🔬 Зачем нужна Алгебра спектров в экологическом анализе?

В масс-спектрометрии ультравысокого разрешения (FT-ICR MS) природных вод (оз. Байкал, притоки) и техногенных загрязнений (шлам-лигнин БЦБК) спектры содержат тысячи индивидуальных пиков. Алгебра спектров позволяет строго математически разделить природный фон и антропогенные маркеры:

---

#### 1. Вычитание бланка / разность ($A \setminus B$)
* **Физический смысл:** Удаление фонового матричного шума (чистая байкальская вода или бланк растворителя) из загрязненной пробы.
* **Результат:** Остаются **только уникальные компоненты**, характерные для источника воздействия (сульфатный шлам-лигнин, техногенные серосодержащие соединения).

#### 2. Консервативное ядро / пересечение ($A \cap B$)
* **Физический смысл:** Поиск соединений, присутствующих **одновременно** в обоих образцах в пределах заданного допуска $ppm$.
* **Результат:** Выделение устойчивого природного гуминового скелета (NOM), инвариантного к внешним факторам.

#### 3. Симметрическая разность ($A \oplus B$) и объединение ($A \cup B$)
* **$A \oplus B$:** Компоненты, уникальные либо для $A$, либо для $B$ (химическая дивергенция составов).
* **$A \cup B$:** Полная композитная карта молекулярного разнообразия экосистемы.

---

#### 📐 Параметры и метрики:
* **Допуск сопоставления ($\text{ppm}$):** Окно поиска по $m/z$ ($\Delta m = m \cdot \text{ppm} \cdot 10^{-6}$). Для FT-ICR MS стандартно $1.0 - 2.0\text{ ppm}$.
* **Индекс Жаккара ($J$):** Степень совпадения компонентного состава ($J = \frac{|A \cap B|}{|A \cup B|}$, от 0 до 1).
* **Косинусное сходство ($\cos \theta$):** Сходство профилей относительных интенсивностей общих пиков.
""" if lang == "ru" else r"""
### 🔬 Why Spectrum Algebra Matters in Environmental Analysis?

In high-resolution mass spectrometry of natural waters (Lake Baikal) and industrial impacts (kraft pulp mill sludge lignin), spectra contain thousands of peaks. Spectrum algebra provides rigorous set-theoretic operations to isolate background NOM from anthropogenic tracers:

---

#### 1. Blank Subtraction & Difference ($A \setminus B$)
* **Physical Meaning:** Removing baseline solvent/background water signals from an impacted sample.
* **Result:** Yields **only unique tracers** specific to lignin contamination.

#### 2. Conservative Core / Intersection ($A \cap B$)
* **Physical Meaning:** Identifies compounds shared by both samples within the specified $ppm$ tolerance.
* **Result:** Isolates resilient natural organic matter (NOM) background.

#### 3. Symmetric Difference ($A \oplus B$) & Union ($A \cup B$)
* **$A \oplus B$:** Components unique to either sample (chemical divergence).
* **$A \cup B$:** Composite molecular map of the aquatic system.

---

#### 📐 Metrics & Parameters:
* **Matching Tolerance ($\text{ppm}$):** $m/z$ alignment window (typically $1.0 - 2.0\text{ ppm}$).
* **Jaccard Similarity ($J$):** Compositional overlap ($J = \frac{|A \cap B|}{|A \cup B|}$).
* **Cosine Similarity ($\cos \theta$):** Spectral profile intensity vector similarity.
""")

        all_spectra_list = list(st.session_state.get("spectra_db", {}).keys())
        if len(all_spectra_list) < 2:
            st.info(T[lang]["cmp_need_two"])
        else:
            cmp_tab_choice = st.radio(
                "Раздел / Mode:",
                [T[lang]["cmp_subtab_view"], T[lang]["cmp_subtab_sub"], T[lang]["cmp_subtab_algebra"]],
                horizontal=True,
                key="cmp_internal_mode",
            )

            col_s1, col_s2, col_s3 = st.columns([2, 2, 1.5])
            with col_s1:
                name_a = st.selectbox(T[lang]["cmp_spec_a"], all_spectra_list, index=0, key="cmp_spec_a")
            with col_s2:
                default_b_idx = 1 if len(all_spectra_list) > 1 else 0
                name_b = st.selectbox(T[lang]["cmp_spec_b"], all_spectra_list, index=default_b_idx, key="cmp_spec_b")
            with col_s3:
                tol_comp = st.number_input(T[lang]["cmp_tol"], min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="cmp_ppm_tol")

            sample_a = st.session_state["spectra_db"][name_a]
            sample_b = st.session_state["spectra_db"][name_b]

            peaks_a = ensure_sample_peaks(sample_a)
            peaks_b = ensure_sample_peaks(sample_b)

            if peaks_a is None or peaks_a.empty or peaks_b is None or peaks_b.empty:
                st.warning("One of the selected samples is empty or could not be parsed." if lang == "en" else "Один из выбранных образцов пуст или не содержит числовых данных пиков.")
            elif name_a == name_b:
                st.warning("Same sample selected for comparison!" if lang == "en" else "Выбран один и тот же образец для сравнения!")
            else:
                m_a, m_b = align_two_spectra_fast(peaks_a, peaks_b, ppm_tol=tol_comp)
                n_a = len(peaks_a)
                n_b = len(peaks_b)
                n_common = len(m_a)
                jaccard = n_common / (n_a + n_b - n_common) if (n_a + n_b - n_common) > 0 else 0.0

                if n_common > 0:
                    ia_sub = peaks_a.iloc[m_a]["intensity"].values
                    ib_sub = peaks_b.iloc[m_b]["intensity"].values
                    dot_prod = float(np.dot(ia_sub, ib_sub))
                    norm_a = float(np.linalg.norm(peaks_a["intensity"].values))
                    norm_b = float(np.linalg.norm(peaks_b["intensity"].values))
                    cos_sim = dot_prod / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
                else:
                    cos_sim = 0.0

                # 1. Обычное сравнение (A vs B)
                if cmp_tab_choice == T[lang]["cmp_subtab_view"]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Peaks in {name_a}" if lang == "en" else f"Пиков в {name_a}", f"{n_a:,}")
                    m2.metric(f"Peaks in {name_b}" if lang == "en" else f"Пиков в {name_b}", f"{n_b:,}")
                    m3.metric(T[lang]["cmp_common"], f"{n_common:,}", delta=T[lang]["cmp_jaccard"].format(val=jaccard))
                    m4.metric(T[lang]["cmp_cosine"], f"{cos_sim:.4f}")

                    assigned_a = sample_a.get("assigned_df")
                    assigned_b = sample_b.get("assigned_df")

                    if assigned_a is not None and not assigned_a.empty:
                        st.markdown("##### " + (f"Comparative Van Krevelen: {name_a} vs {name_b}" if lang == "en" else f"Сравнительная диаграмма Ван-Кревелена: {name_a} vs {name_b}"))
                        common_masses = set(peaks_a.iloc[m_a]["mass"].round(4))

                        assigned_a_common = assigned_a[assigned_a["mass"].round(4).isin(common_masses)]
                        assigned_a_unique = assigned_a[~assigned_a["mass"].round(4).isin(common_masses)]

                        fig_cmp_inter = go.Figure()
                        if not assigned_a_common.empty:
                            fig_cmp_inter.add_trace(go.Scattergl(
                                x=assigned_a_common["O/C"],
                                y=assigned_a_common["H/C"],
                                mode="markers",
                                name=T[lang]["cmp_common_label"].format(n=len(assigned_a_common)),
                                marker=dict(color="#7F7F7F", size=3.0, opacity=0.45),
                            ))

                        if not assigned_a_unique.empty:
                            fig_cmp_inter.add_trace(go.Scattergl(
                                x=assigned_a_unique["O/C"],
                                y=assigned_a_unique["H/C"],
                                mode="markers",
                                name=T[lang]["cmp_unique_a"].format(name=name_a, n=len(assigned_a_unique)),
                                marker=dict(color="#0020C2", size=3.5, opacity=0.75),
                            ))

                        if assigned_b is not None and not assigned_b.empty:
                            assigned_b_unique = assigned_b[~assigned_b["mass"].round(4).isin(common_masses)]
                            if not assigned_b_unique.empty:
                                fig_cmp_inter.add_trace(go.Scattergl(
                                    x=assigned_b_unique["O/C"],
                                    y=assigned_b_unique["H/C"],
                                    mode="markers",
                                    name=T[lang]["cmp_unique_b"].format(name=name_b, n=len(assigned_b_unique)),
                                    marker=dict(color="#2CA02C", size=3.5, opacity=0.75),
                                ))

                        fig_cmp_inter.update_layout(
                            xaxis=dict(title=T[lang]["oc_axis"], range=[0.0, 1.0], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                            yaxis=dict(title=T[lang]["hc_axis"], range=[0.2, 2.2], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                            plot_bgcolor="white",
                            height=580,
                            margin=dict(l=45, r=30, t=30, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.caption(T[lang]["nav_caption"])
                        st_plotly(fig_cmp_inter)

                        fig_cmp_mpl, ax_cmp_mpl = plt.subplots(figsize=(12, 7.5), dpi=150)
                        if not assigned_a_common.empty:
                            ax_cmp_mpl.scatter(assigned_a_common["O/C"], assigned_a_common["H/C"], c="#7F7F7F",
                                               label=T[lang]["cmp_common_label"].format(n=len(assigned_a_common)), s=2.0, alpha=0.4, edgecolors="none", rasterized=True)
                        if not assigned_a_unique.empty:
                            ax_cmp_mpl.scatter(assigned_a_unique["O/C"], assigned_a_unique["H/C"], c="#0020C2",
                                               label=T[lang]["cmp_unique_a"].format(name=name_a, n=len(assigned_a_unique)), s=2.2, alpha=0.75, edgecolors="none", rasterized=True)
                        if assigned_b is not None and not assigned_b.empty:
                            assigned_b_unique = assigned_b[~assigned_b["mass"].round(4).isin(common_masses)]
                            if not assigned_b_unique.empty:
                                ax_cmp_mpl.scatter(assigned_b_unique["O/C"], assigned_b_unique["H/C"], c="#2CA02C",
                                                   label=T[lang]["cmp_unique_b"].format(name=name_b, n=len(assigned_b_unique)), s=2.2, alpha=0.75, edgecolors="none", rasterized=True)

                        ax_cmp_mpl.set_xlim(0.0, 1.0)
                        ax_cmp_mpl.set_ylim(0.2, 2.2)
                        ax_cmp_mpl.set_xlabel(T[lang]["oc_axis"], fontsize=11, fontweight="medium")
                        ax_cmp_mpl.set_ylabel(T[lang]["hc_axis"], fontsize=11, fontweight="medium")
                        ax_cmp_mpl.set_title(f"Comparative Analysis: {name_a} vs {name_b}", fontsize=11)
                        ax_cmp_mpl.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
                        ax_cmp_mpl.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
                        plt.tight_layout()

                        get_plot_download_buttons(fig_cmp_mpl, f"compare_{name_a}_vs_{name_b}_vk", lang)
                        plt.close(fig_cmp_mpl)

                    st.markdown("##### " + T[lang]["cmp_mirror_title"])
                    ia_norm = (peaks_a["intensity"] / max(1e-12, float(peaks_a["intensity"].max()))) * 100.0
                    ib_norm = (peaks_b["intensity"] / max(1e-12, float(peaks_b["intensity"].max()))) * 100.0

                    ma = peaks_a["mass"].values
                    mb = peaks_b["mass"].values

                    x_a = np.empty(len(ma) * 3)
                    y_a = np.empty(len(ma) * 3)
                    x_a[0::3] = ma
                    x_a[1::3] = ma
                    x_a[2::3] = None
                    y_a[0::3] = 0
                    y_a[1::3] = ia_norm
                    y_a[2::3] = None

                    x_b = np.empty(len(mb) * 3)
                    y_b = np.empty(len(mb) * 3)
                    x_b[0::3] = mb
                    x_b[1::3] = mb
                    x_b[2::3] = None
                    y_b[0::3] = 0
                    y_b[1::3] = -ib_norm
                    y_b[2::3] = None

                    fig_mir_inter = go.Figure()
                    fig_mir_inter.add_trace(go.Scattergl(x=x_a, y=y_a, mode="lines", line=dict(color="#0020C2", width=1.1), name=f"{name_a} (+)"))
                    fig_mir_inter.add_trace(go.Scattergl(x=x_b, y=y_b, mode="lines", line=dict(color="#2CA02C", width=1.1), name=f"{name_b} (-)"))
                    fig_mir_inter.update_layout(
                        xaxis=dict(title=T[lang]["mz_axis"], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                        yaxis=dict(title=T[lang]["rel_int_axis"], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                        plot_bgcolor="white",
                        height=480,
                        margin=dict(l=45, r=30, t=30, b=40),
                    )
                    st.caption(T[lang]["nav_caption"])
                    st_plotly(fig_mir_inter)

                    fig_mir_mpl, ax_mir_mpl = plt.subplots(figsize=(12, 5), dpi=150)
                    ax_mir_mpl.vlines(peaks_a["mass"], 0, ia_norm, color="#0020C2", linewidth=0.6, alpha=0.7, label=f"{name_a} (+)")
                    ax_mir_mpl.vlines(peaks_b["mass"], 0, -ib_norm, color="#2CA02C", linewidth=0.6, alpha=0.7, label=f"{name_b} (-)")
                    ax_mir_mpl.axhline(0, color="black", linewidth=0.8)
                    ax_mir_mpl.set_xlabel(T[lang]["mz_axis"], fontsize=11)
                    ax_mir_mpl.set_ylabel(T[lang]["rel_int_axis"], fontsize=10)
                    ax_mir_mpl.legend(loc="upper right")
                    plt.tight_layout()

                    get_plot_download_buttons(fig_mir_mpl, f"head_to_tail_{name_a}_vs_{name_b}", lang)
                    plt.close(fig_mir_mpl)

                # 2. Вычитание бланка (int_sub)
                elif cmp_tab_choice == T[lang]["cmp_subtab_sub"]:
                    st.markdown(f"#### {T[lang]['cmp_subtab_sub']}")
                    st.write(f"Образец: **{name_a}** | Холостая проба (Бланк): **{name_b}**" if lang == "ru" else f"Sample: **{name_a}** | Blank: **{name_b}**")
                    sub_factor = st.slider(T[lang]["sub_factor_label"], min_value=0.1, max_value=3.0, value=1.0, step=0.05)

                    if st.button(T[lang]["sub_btn"], type="primary"):
                        sub_peaks = peaks_a.copy().reset_index(drop=True)
                        matched_dict = dict(zip(m_a, m_b))

                        new_ints = []
                        for idx_a_row in range(len(sub_peaks)):
                            orig_int = sub_peaks.loc[idx_a_row, "intensity"]
                            if idx_a_row in matched_dict:
                                idx_b_row = matched_dict[idx_a_row]
                                blank_int = peaks_b.iloc[idx_b_row]["intensity"]
                                res_int = max(0.0, float(orig_int - sub_factor * blank_int))
                            else:
                                res_int = float(orig_int)
                            new_ints.append(res_int)

                        sub_peaks["intensity"] = new_ints
                        clean_sub_df = sub_peaks[sub_peaks["intensity"] > 0].copy().reset_index(drop=True)
                        if clean_sub_df.empty:
                            st.warning("В результате вычитания все пики были обнулены." if lang == "ru" else "All peaks were subtracted to zero.")
                        else:
                            clean_sub_df["norm_intensity"] = (clean_sub_df["intensity"] / clean_sub_df["intensity"].max()) * 100.0
                            new_sample_name = f"{name_a}_sub_{name_b}.csv"
                            fpath_sub = os.path.join(STORAGE_DIR, new_sample_name)
                            clean_sub_df.to_csv(fpath_sub, sep="\t", index=False)

                            st.session_state["spectra_db"][new_sample_name] = {
                                "raw_path": fpath_sub,
                                "file_bytes": clean_sub_df.to_csv(sep="\t", index=False).encode("utf-8"),
                                "parsed_peaks": clean_sub_df,
                                "assigned_df": None,
                                "raw_df": clean_sub_df,
                            }
                            st.success(T[lang]["sub_success"].format(name=new_sample_name, n=len(clean_sub_df)))
                            st.rerun()

                # 3. Алгебра спектров и диаграмма Венна
                else:
                    st.markdown(f"#### {T[lang]['cmp_subtab_algebra']}")
                    op_choice = st.selectbox(
                        T[lang]["alg_op_label"],
                        [
                            ("and", T[lang]["alg_and"]),
                            ("or", T[lang]["alg_or"]),
                            ("sub_a_b", T[lang]["alg_sub_a_b"].format(a=name_a, b=name_b) if "{a}" in T[lang]["alg_sub_a_b"] else f"{T[lang]["alg_sub_a_b"]} ({name_a} \\ {name_b})"),
                            ("sub_b_a", T[lang]["alg_sub_b_a"].format(a=name_a, b=name_b) if "{a}" in T[lang]["alg_sub_b_a"] else f"{T[lang]["alg_sub_b_a"]} ({name_b} \\ {name_a})"),
                            ("xor", T[lang]["alg_xor"]),
                        ],
                        format_func=lambda x: x[1],
                    )

                    # Построение диаграммы Венна (Matplotlib Circles)
                    fig_v, ax_v = plt.subplots(figsize=(8, 4.5), dpi=150)
                    c_a = patches.Circle((0.35, 0.5), 0.3, facecolor="#0020C2", alpha=0.4, edgecolor="black", linewidth=1.5)
                    c_b = patches.Circle((0.65, 0.5), 0.3, facecolor="#2CA02C", alpha=0.4, edgecolor="black", linewidth=1.5)
                    ax_v.add_patch(c_a)
                    ax_v.add_patch(c_b)

                    n_uniq_a = n_a - n_common
                    n_uniq_b = n_b - n_common

                    ax_v.text(0.22, 0.5, f"{name_a[:15]}\n\n{n_uniq_a:,}", ha="center", va="center", fontsize=11, fontweight="bold")
                    ax_v.text(0.78, 0.5, f"{name_b[:15]}\n\n{n_uniq_b:,}", ha="center", va="center", fontsize=11, fontweight="bold")
                    ax_v.text(0.50, 0.5, f"A ∩ B\n\n{n_common:,}", ha="center", va="center", fontsize=11, fontweight="bold", color="#800000")

                    ax_v.set_xlim(0, 1)
                    ax_v.set_ylim(0.1, 0.9)
                    ax_v.axis("off")
                    ax_v.set_title(T[lang]["venn_title"].format(tol=tol_comp), fontsize=12, pad=10)
                    plt.tight_layout()
                    st.pyplot(fig_v)
                    get_plot_download_buttons(fig_v, f"venn_{name_a}_vs_{name_b}", lang)
                    plt.close(fig_v)

                    if st.button(T[lang]["alg_btn"], type="primary"):
                        alg_df = perform_spectral_algebra(peaks_a, peaks_b, operation=op_choice[0], ppm_tol=tol_comp)
                        if alg_df.empty:
                            st.warning("В результате операции пики не найдены." if lang == "ru" else "No peaks resulted from this operation.")
                        else:
                            alg_df["norm_intensity"] = (alg_df["intensity"] / alg_df["intensity"].max()) * 100.0
                            new_alg_name = f"{name_a}_{op_choice[0]}_{name_b}.csv"
                            fpath_alg = os.path.join(STORAGE_DIR, new_alg_name)
                            alg_df.to_csv(fpath_alg, sep="\t", index=False)

                            st.session_state["spectra_db"][new_alg_name] = {
                                "raw_path": fpath_alg,
                                "file_bytes": alg_df.to_csv(sep="\t", index=False).encode("utf-8"),
                                "parsed_peaks": alg_df,
                                "assigned_df": None,
                                "raw_df": alg_df,
                            }
                            st.success(T[lang]["alg_success"].format(name=new_alg_name, n=len(alg_df)))
                            st.rerun()

    # 8. TMDS
    with st_sub_tmds:
        st.subheader(f"{T[lang]['tab_tmds']}: {active_spectrum_name if active_spectrum_name else ''}")
        work_peaks = current_sample.get("assigned_df") if (current_sample and current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty) else (current_sample.get("parsed_peaks") if current_sample else None)
        if work_peaks is not None and not work_peaks.empty:
            tmds_view_mode = st.radio(
                "Режим отображения TMDS:" if lang == "ru" else "TMDS Display Mode:",
                [T[lang]["tmds_mode_freq"], T[lang]["tmds_mode_network"], T[lang]["tmds_mode_pathways"]],
                horizontal=True,
                key=f"tmds_view_{active_spectrum_name}"
            )
            if tmds_view_mode == T[lang]["tmds_mode_freq"]:
                flux_res = compute_geochemical_vector_fluxes(work_peaks, tol_mda=2.0, top_n=1000)
                tmds_sum = flux_res.get("summary_df", pd.DataFrame())
                indices = flux_res.get("indices", {})
                if not tmds_sum.empty:
                    st.markdown(f"#### {T[lang]['tmds_flux_header']}")
                    fx1, fx2, fx3, fx4 = st.columns(4)
                    fx1.metric(T[lang]["tmds_flux_ox_decarb"], f"{indices.get('ox_decarb_ratio', 0.0):.2f}")
                    fx2.metric(T[lang]["tmds_flux_alkyl"], f"{indices.get('alkylation_share_pct', 0.0):.1f}%")
                    fx3.metric(T[lang]["tmds_flux_sulf"], f"{indices.get('sulfonation_index_pct', 0.0):.2f}%")
                    fx4.metric(T[lang]["tmds_flux_hydr"], f"{indices.get('hydration_ox_ratio', 0.0):.2f}")

                    fig_tmds = px.bar(tmds_sum, x="Transformation", y="Count", color="Flux_Type", text=tmds_sum["Share_pct"].apply(lambda v: f"{v:.1f}%"))
                    fig_tmds.update_layout(showlegend=True, xaxis_tickangle=-25, height=420)
                    st_plotly(fig_tmds)
                    st_df(tmds_sum)
            elif tmds_view_mode == T[lang]["tmds_mode_pathways"]:
                top_pks = work_peaks.sort_values("intensity", ascending=False).head(40).sort_values("mass")
                cand_options = [
                    (float(r["mass"]), f"m/z {r['mass']:.4f}" + (f" ({r['Formula']})" if 'Formula' in r and pd.notna(r['Formula']) else ""))
                    for _, r in top_pks.iterrows()
                ]
                col_pw1, col_pw2 = st.columns(2)
                with col_pw1:
                    src_sel = st.selectbox(
                        T[lang]["tmds_path_source"],
                        options=cand_options,
                        format_func=lambda x: x[1],
                        index=0,
                        key=f"pw_src_{active_spectrum_name}"
                    )
                    src_m_custom = st.number_input("Или точная m/z:" if lang == "ru" else "Or exact m/z:", value=0.0, step=0.0001, format="%.4f", key=f"cust_src_{active_spectrum_name}")
                    src_m = src_m_custom if src_m_custom > 10.0 else src_sel[0]
                with col_pw2:
                    dst_default_idx = min(len(cand_options)-1, 5)
                    dst_sel = st.selectbox(
                        T[lang]["tmds_path_target"],
                        options=cand_options,
                        format_func=lambda x: x[1],
                        index=dst_default_idx,
                        key=f"pw_dst_{active_spectrum_name}"
                    )
                    dst_m_custom = st.number_input("Или точная m/z продукта:" if lang == "ru" else "Or exact target m/z:", value=0.0, step=0.0001, format="%.4f", key=f"cust_dst_{active_spectrum_name}")
                    dst_m = dst_m_custom if dst_m_custom > 10.0 else dst_sel[0]

                col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
                with col_cfg1:
                    depth_val = st.slider(T[lang]["tmds_path_depth"], min_value=1, max_value=5, value=3, key=f"pw_depth_{active_spectrum_name}")
                with col_cfg2:
                    tol_pw_val = st.number_input(T[lang]["tmds_path_tol"], min_value=0.5, max_value=5.0, value=2.0, step=0.5, key=f"pw_tol_{active_spectrum_name}")
                with col_cfg3:
                    st.write("")
                    find_pw_btn = st.button(T[lang]["tmds_path_find_btn"], type="primary", key=f"btn_find_pw_{active_spectrum_name}")

                if find_pw_btn or f"pw_res_{active_spectrum_name}" in st.session_state:
                    if find_pw_btn:
                        with st.spinner("Поиск цепочек реакций..." if lang == "ru" else "Traversing reaction pathways..."):
                            pw_list = find_transformation_pathways(
                                work_peaks, source_mass=src_m, target_mass=dst_m,
                                max_depth=depth_val, tol_mda=tol_pw_val, max_paths=10
                            )
                            st.session_state[f"pw_res_{active_spectrum_name}"] = pw_list

                    pw_results = st.session_state.get(f"pw_res_{active_spectrum_name}", [])
                    if pw_results:
                        st.success(T[lang]["tmds_path_found"].format(n=len(pw_results)))
                        pw_summary_rows = []
                        for p_idx, p_item in enumerate(pw_results, 1):
                            pw_summary_rows.append({
                                "Path_ID": f"Path #{p_idx}",
                                "Stages": p_item["depth"],
                                "Cumulative_Error_mDa": p_item["cumulative_error_mda"],
                                "Pathway": p_item["path_str"],
                            })
                        st_df(pd.DataFrame(pw_summary_rows))

                        for p_idx, p_item in enumerate(pw_results, 1):
                            with st.expander(f"📍 {p_item['path_str']} (Total err: {p_item['cumulative_error_mda']:.2f} mDa)", expanded=(p_idx == 1)):
                                steps_df = pd.DataFrame(p_item["steps"])
                                st_df(steps_df)
                    else:
                        st.info(T[lang]["tmds_path_not_found"])
            else:
                c_n1, c_n2 = st.columns([1, 1])
                with c_n1:
                    n_top_net = st.slider(T[lang]["tmds_net_peaks"], min_value=30, max_value=300, value=120, step=10, key=f"tmds_top_{active_spectrum_name}")
                with c_n2:
                    layout_choice = st.selectbox(
                        T[lang]["tmds_net_layout"],
                        [T[lang]["tmds_layout_spring"], T[lang]["tmds_layout_chem"]],
                        index=0,
                        key=f"tmds_lay_{active_spectrum_name}"
                    )
                l_type = "chemical" if layout_choice == T[lang]["tmds_layout_chem"] else "spring"
                with st.spinner("Построение сетевого графа реакций..." if lang == "ru" else "Constructing reaction network..."):
                    net_dict = build_tmds_network_graph(work_peaks, top_n=n_top_net, tol_mda=2.0, max_edges=250, layout=l_type)

                nodes_df = net_dict["nodes_df"]
                edges_df = net_dict["edges_df"]
                hubs_df = net_dict["hubs_df"]

                if nodes_df.empty or edges_df.empty:
                    st.warning("В заданных границах связей между пиками не обнаружено." if lang == "ru" else "No connected reactions found in the current selection.")
                else:
                    fig_net = go.Figure()
                    trans_colors = {
                        "CH2": "#0020C2", "O": "#D62728", "H2O": "#2CA02C",
                        "CO2": "#FF7F0E", "CO": "#9467BD", "H2": "#8C564B",
                        "NH3": "#E377C2", "SO3": "#17BECF"
                    }
                    for trans_type, g_edges in edges_df.groupby("Transformation"):
                        edge_x, edge_y = [], []
                        for _, e_row in g_edges.iterrows():
                            edge_x.extend([e_row["x0"], e_row["x1"], None])
                            edge_y.extend([e_row["y0"], e_row["y1"], None])
                        t_key = trans_type.split()[0]
                        e_color = trans_colors.get(t_key, "#7F7F7F")
                        fig_net.add_trace(go.Scatter(
                            x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1.2, color=e_color),
                            name=trans_type,
                            hoverinfo="none",
                            opacity=0.6,
                        ))

                    node_sizes = 5.0 + 3.0 * np.log1p(nodes_df["Degree"].values)
                    hover_texts = [
                        f"<b>{r['Node_ID']}</b><br>Formula: {r.get('Formula', 'N/A')}<br>Degree: {r['Degree']}<br>Class: {r.get('Hetero_Class', 'N/A')} / {r.get('Bio_Class', 'N/A')}"
                        for _, r in nodes_df.iterrows()
                    ]
                    fig_net.add_trace(go.Scatter(
                        x=nodes_df["x"], y=nodes_df["y"], mode="markers",
                        marker=dict(size=node_sizes, color=nodes_df["mass"], colorscale="Viridis", showscale=True, colorbar=dict(title="m/z", len=0.7), line=dict(width=1, color="black")),
                        text=hover_texts, hoverinfo="text", name="Peaks"
                    ))

                    fig_net.update_layout(
                        title=T[lang]["tmds_net_title"],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=560, plot_bgcolor="white"
                    )
                    st_plotly(fig_net)

                    if not hubs_df.empty:
                        st.markdown(f"##### {T[lang]['tmds_hubs_title']}")
                        st_df(hubs_df)

                    st.download_button(
                        label=T[lang]["tmds_dl_edges"],
                        data=edges_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"ChemoSuite_TMDS_Network_Edges_{active_spectrum_name}.csv",
                        mime="text/csv",
                        key=f"dl_tmds_edges_{active_spectrum_name}"
                    )

    # 9. Сводные характеристики
    with st_sub_desc:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is not None and not assigned_data.empty:
            st.subheader(f"{T[lang]['tab_desc']}: {active_spectrum_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T[lang]["desc_mn"], f"{assigned_data['mass'].mean():.2f} Da")
            c2.metric("H/C (Mn)", f"{assigned_data['H/C'].mean():.3f}")
            c3.metric("O/C (Mn)", f"{assigned_data['O/C'].mean():.3f}")
            c4.metric(T[lang]["desc_ai"], f"{assigned_data['AI'].mean():.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric(T[lang]["desc_mw"], f"{np.average(assigned_data['mass'], weights=assigned_data['intensity']):.2f} Da" if "intensity" in assigned_data.columns and assigned_data["intensity"].sum() > 0 else f"{assigned_data['mass'].mean():.2f} Da")
            c6.metric(T[lang]["desc_dbe"], f"{assigned_data['DBE'].mean():.2f}")
            c7.metric("NOSC", f"{assigned_data['NOSC'].mean():.3f}" if "NOSC" in assigned_data.columns else "N/A")
            c8.metric("DBE - O", f"{(assigned_data['DBE'] - assigned_data['O']).mean():.2f}")

        # Feature 4: Batch Descriptors Export across all spectra in session
        st.markdown("---")
        st.write(f"### {T[lang]['batch_desc_title']}")
        st.caption(T[lang]["batch_desc_caption"])

        assigned_spectra_all = {
            s_name: s_obj["assigned_df"]
            for s_name, s_obj in st.session_state.get("spectra_db", {}).items()
            if s_obj.get("assigned_df") is not None and not s_obj["assigned_df"].empty
        }

        if not assigned_spectra_all:
            st.info(T[lang]["batch_desc_no_data"])
        else:
            batch_rows = []
            for s_name, a_df in assigned_spectra_all.items():
                d = chemo_ml.extract_fticr_descriptors(a_df, s_name)
                if d:
                    batch_rows.append(d)

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                st_df(batch_df)

                col_dl_b1, col_dl_b2 = st.columns([1, 1])
                with col_dl_b1:
                    st.download_button(
                        label=T[lang]["batch_desc_dl_csv"],
                        data=batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_Batch_FTICR_Descriptors.csv",
                        mime="text/csv",
                        key="dl_batch_desc_csv",
                    )
                with col_dl_b2:
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        batch_df.to_excel(writer, index=False, sheet_name="FTICR_Descriptors")
                    st.download_button(
                        label=T[lang]["batch_desc_dl_excel"],
                        data=excel_buf.getvalue(),
                        file_name="ChemoSuite_Batch_FTICR_Descriptors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_batch_desc_excel",
                    )

# ==============================================================================
# МОДУЛЬ 2: 3D ОПТИЧЕСКАЯ СПЕКТРОСКОПИЯ (EEM-PARAFAC & UV-VIS)
# ==============================================================================
elif active_module == T[lang]["mod2_name"]:
    st.title(T[lang]["eem_title"])
    st.caption(T[lang]["eem_subtitle"])

    if not EEM_CORE_AVAILABLE:
        st.error(T[lang]["eem_err_missing"])
        st.stop()

    with st.sidebar:
        st.header(T[lang]["eem_sidebar_hdr"])
        eem_data_src = st.radio(
            T[lang]["eem_src_label"],
            [T[lang]["eem_src_upload"], T[lang]["eem_src_synth"]],
            index=0,
            key="eem_data_source_radio",
        )
        if st.session_state.get("_prev_eem_data_src") != eem_data_src:
            st.session_state["_prev_eem_data_src"] = eem_data_src
            if "parafac_active_tables_multiselect" in st.session_state:
                del st.session_state["parafac_active_tables_multiselect"]
            if "_prev_parafac_all_sample_ids" in st.session_state:
                del st.session_state["_prev_parafac_all_sample_ids"]
            if "eem_active_sample" in st.session_state:
                del st.session_state["eem_active_sample"]

        eem_files = []
        uv_files = []
        if eem_data_src == T[lang]["eem_src_upload"]:
            eem_files = st.file_uploader(
                T[lang]["eem_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "txt", "dat"],
                key="eem_files_uploader",
            )
            # Загрузчик 1D спектров поглощения УФ-Вид
            uv_files = st.file_uploader(
                T[lang]["eem_uv_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "txt", "dat"],
                key="uv_files_uploader",
            )

            # Карточка постоянного хранения EEM и UV образцов в текущей сессии
            n_eem = len(st.session_state.get("eem_stored_dict", {}))
            n_uv = len(st.session_state.get("uv_stored_dict", {}))
            if n_eem > 0 or n_uv > 0:
                with st.expander(
                    f"💾 {'Образцы в памяти сессии' if lang == 'ru' else 'Session Samples'}: {n_eem} EEM / {n_uv} UV",
                    expanded=True
                ):
                    st.caption(
                        "Файлы сохраняются в сессии при переключении любых вкладок."
                        if lang == "ru" else
                        "Files are stored in session and retained across tab switches."
                    )
                    if n_eem > 0:
                        st.markdown("**EEM матрицы:**")
                        for ename in list(st.session_state["eem_stored_dict"].keys()):
                            ce1, ce2 = st.columns([3, 1])
                            with ce1: st.markdown(f"• `{ename}`")
                            with ce2:
                                if st.button("🗑️", key=f"del_eem_m2_{ename}", help="Удалить" if lang == "ru" else "Delete"):
                                    del st.session_state["eem_stored_dict"][ename]
                                    st.session_state["eem_stored_samples"] = list(st.session_state["eem_stored_dict"].values())
                                    st.rerun()
                    if n_uv > 0:
                        st.markdown("**UV-Vis спектры:**")
                        for uname in list(st.session_state["uv_stored_dict"].keys()):
                            cu1, cu2 = st.columns([3, 1])
                            with cu1: st.markdown(f"• `{uname}`")
                            with cu2:
                                if st.button("🗑️", key=f"del_uv_m2_{uname}", help="Удалить" if lang == "ru" else "Delete"):
                                    del st.session_state["uv_stored_dict"][uname]
                                    st.session_state["uv_stored_samples"] = list(st.session_state["uv_stored_dict"].values())
                                    st.rerun()

        with st.expander(T[lang]["eem_doc_expander"], expanded=False):
            default_doc_val = st.number_input(
                T[lang]["eem_default_doc"],
                min_value=0.0,
                max_value=200.0,
                value=4.0,
                step=0.5,
                key="default_doc_input"
            )

        st.markdown("---")
        st.subheader(T[lang]["eem_preproc_hdr"])
        clean_scatter = st.checkbox(T[lang]["eem_clean_scatter"], value=True, key="eem_clean_scatter")
        norm_raman = st.checkbox(T[lang]["eem_norm_raman"], value=False, key="eem_norm_raman")
        palette = st.selectbox(T[lang]["eem_palette_label"], ["Viridis", "Plasma", "Inferno", "Turbo"], key="eem_palette_select")

    if "eem_stored_dict" not in st.session_state:
        st.session_state["eem_stored_dict"] = {}
    if "uv_stored_dict" not in st.session_state:
        st.session_state["uv_stored_dict"] = {}

    # Чтение EEM
    if eem_files:
        for f in eem_files:
            try:
                sample_name = f.name.rsplit(".", 1)[0]
                if hasattr(eem_core, "load_eem_file"):
                    parsed = eem_core.load_eem_file(f, sample_id=sample_name)
                else:
                    df = pd.read_csv(f)
                    parsed = eem_core.parse_eem_dataframe(df, sample_id=sample_name)

                if clean_scatter:
                    parsed.data = eem_core.remove_scatter_bands(parsed.data, parsed.ex, parsed.em)
                if norm_raman:
                    parsed.data, _ = eem_core.normalize_to_raman_units(parsed.data, parsed.ex, parsed.em)
                st.session_state["eem_stored_dict"][sample_name] = parsed
            except Exception as e:
                st.error(f"Error reading EEM {f.name}: {e}")

    # Чтение 1D УФ-Вид спектров
    if uv_files:
        for uf in uv_files:
            try:
                u_name = uf.name.rsplit(".", 1)[0]
                parsed_uv = eem_core.parse_uv_vis_spectrum(uf, sample_id=u_name)
                parsed_uv.doc = default_doc_val if default_doc_val > 0 else None
                st.session_state["uv_stored_dict"][u_name] = parsed_uv
            except Exception as e:
                st.error(f"Error reading UV-Vis {uf.name}: {e}")

    # Синхронизация списков образцов
    st.session_state["eem_stored_samples"] = list(st.session_state.get("eem_stored_dict", {}).values())
    st.session_state["uv_stored_samples"] = list(st.session_state.get("uv_stored_dict", {}).values())

    if eem_data_src == T[lang]["eem_src_synth"]:
        samples = eem_core.generate_synthetic_chemometrics_dataset(n_samples=10)
        uv_samples = eem_core.generate_synthetic_uv_vis_dataset(eem_samples=samples)
    else:
        samples = st.session_state.get("eem_stored_samples", [])
        uv_samples = st.session_state.get("uv_stored_samples", [])

    # Автоматическая связка УФ-Вид спектров с EEM образцами
    if samples and uv_samples:
        doc_dict = {
            s.sample_id: (s.doc if s.doc is not None else default_doc_val)
            for s in samples
            if (s.doc is not None or default_doc_val > 0)
        }
        eem_core.link_uv_vis_to_eem(samples, uv_samples, doc_map=doc_dict if doc_dict else None)

    if not samples and not uv_samples:
        st.warning(T[lang]["eem_warn_no_data"])
    else:
        tab_eem1, tab_eem2, tab_eem3, tab_eem4 = st.tabs([
            T[lang]["eem_tab1"], T[lang]["eem_tab2"], T[lang]["eem_tab3"], T[lang]["eem_tab4"]
        ])

        # Экран 1: Контурные карты EEM
        with tab_eem1:
            if not samples:
                st.info("Нет загруженных матриц EEM." if lang == "ru" else "No EEM matrices loaded.")
            else:
                s_names = [s.sample_id for s in samples]
                selected_sname = st.selectbox(T[lang]["eem_select_sample"], s_names, key="eem_active_sample")
                s_obj = next((s for s in samples if s.sample_id == selected_sname), samples[0])

                fig_eem = go.Figure(data=go.Contour(
                    z=s_obj.data, x=s_obj.ex, y=s_obj.em, colorscale=palette.lower(),
                    contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color="white")),
                    colorbar=dict(title=T[lang]["eem_int_label"]),
                ))
                fig_eem.update_layout(
                    title=T[lang]["eem_contour_title"].format(name=s_obj.sample_id),
                    xaxis_title=T[lang]["eem_ex_axis"],
                    yaxis_title=T[lang]["eem_em_axis"],
                    height=560,
                    template="plotly_dark",
                )
                st_plotly(fig_eem)

        # Экран 2: Индексы
        with tab_eem2:
            st.subheader(T[lang]["eem_indices_title"])
            if not samples:
                st.info("Нет загруженных матриц EEM." if lang == "ru" else "No EEM matrices loaded.")
            else:
                indices_list = [eem_core.calculate_spectral_indices(s) for s in samples]
                df_indices = pd.DataFrame(indices_list)

                c_idx1, c_idx2 = st.columns([3, 1])
                with c_idx1:
                    st_df(
                        df_indices.style.format(
                            {"FI": "{:.2f}", "HIX": "{:.2f}", "A254": "{:.4f}", "DOC": "{:.2f}", "SUVA254": "{:.2f}"},
                            na_rep="—"
                        )
                    )
                with c_idx2:
                    st.download_button(
                        label=T[lang]["eem_dl_indices"],
                        data=df_indices.to_csv(index=False).encode("utf-8"),
                        file_name="EEM_Indices.csv",
                        mime="text/csv",
                        key="eem_dl_indices_btn",
                    )

        # Экран 3: PARAFAC
        with tab_eem3:
            st.subheader(T[lang]["eem_parafac_title"])
            if not samples or len(samples) < 2:
                st.warning("Для факторизации PARAFAC необходимо минимум 2 матрицы EEM." if lang == "ru" else "At least 2 EEM matrices required for PARAFAC.")
            else:
                all_sample_ids = [s.sample_id for s in samples]

                # Автоматическая синхронизация при смене источника или состава доступных таблиц
                prev_ids = st.session_state.get("_prev_parafac_all_sample_ids")
                current_selected = st.session_state.get("parafac_active_tables_multiselect")
                valid_selected = [s for s in current_selected if s in all_sample_ids] if current_selected else []

                # Если состав доступных образцов изменился или в session_state нет ни одной валидной таблицы для текущего датасета:
                if prev_ids != all_sample_ids or current_selected is None or (len(current_selected) > 0 and len(valid_selected) == 0):
                    st.session_state["_prev_parafac_all_sample_ids"] = list(all_sample_ids)
                    st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)

                col_sel1, col_sel2 = st.columns([3, 1])
                with col_sel1:
                    selected_parafac_ids = st.multiselect(
                        T[lang]["eem_parafac_select_tables"],
                        options=all_sample_ids,
                        key="parafac_active_tables_multiselect",
                    )
                    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
                    with c_btn1:
                        if st.button("Выбрать все" if lang == "ru" else "Select All", key="btn_sel_all_parafac", type="secondary"):
                            st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)
                            st.rerun()
                    with c_btn2:
                        if st.button("Сбросить" if lang == "ru" else "Deselect", key="btn_desel_parafac", type="secondary"):
                            st.session_state["parafac_active_tables_multiselect"] = []
                            st.rerun()

                with col_sel2:
                    n_components_sel = st.selectbox(
                        "Компонентов (R):" if lang == "ru" else "Components (R):",
                        [2, 3, 4],
                        index=1,
                        key="parafac_n_components_select"
                    )

                if len(selected_parafac_ids) < 2:
                    st.warning(T[lang]["eem_parafac_min_samples"])
                    if st.button("⚡ " + ("Выбрать все таблицы EEM" if lang == "ru" else "Select All EEM Tables"), key="btn_quick_select_all_parafac", type="primary"):
                        st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)
                        st.rerun()
                else:
                    active_parafac_samples = [s for s in samples if s.sample_id in selected_parafac_ids]
                    with st.spinner(T[lang]["eem_parafac_spinner"]):
                        tensor_x, em_ax, ex_ax, s_ids = eem_core.build_eem_tensor(active_parafac_samples)
                        results = eem_core.fit_parafac(tensor_x, n_components=n_components_sel, random_state=42)

                    m_c1, m_c2, m_c3 = st.columns(3)
                    m_c1.metric(T[lang]["eem_m_exp_var"], f"{results['explained_variance']:.2f}%")
                    m_c2.metric(
                        T[lang]["eem_m_corcondia"],
                        f"{results['corcondia']:.1f}%",
                        delta=T[lang]["eem_corcondia_ok"] if results["corcondia"] > 85.0 else T[lang]["eem_corcondia_warn"],
                    )
                    m_c3.metric(T[lang]["eem_m_components"], f"{n_components_sel} флуорофора" if lang == "ru" else f"{n_components_sel} Components")

                    if results["corcondia"] >= 80.0:
                        st.success("✅ **CORCONDIA ≥ 80%**: Трилинейная модель валидирована (высокая степень однозначности разложения)." if lang == "ru" else "✅ **CORCONDIA ≥ 80%**: Trilinear model validated (high degree of uniqueness).")
                    elif results["corcondia"] >= 50.0:
                        st.warning("⚠️ **CORCONDIA 50–80%**: Умеренная адекватность трилинейной структуры. Рекомендуется сплит-хаф анализ." if lang == "ru" else "⚠️ **CORCONDIA 50–80%**: Moderate trilinearity. Split-half validation recommended.")
                    else:
                        st.error("❌ **CORCONDIA < 50%**: Низкая согласованность ядра. Высокий риск перефакторизации (overfitting)." if lang == "ru" else "❌ **CORCONDIA < 50%**: Low core consistency. Risk of overfactoring.")

                    with st.expander(T[lang]["eem_splithalf_expander"], expanded=False):
                        if len(active_parafac_samples) < 4:
                            st.info(T[lang]["eem_splithalf_need_samples"])
                        else:
                            if st.button(T[lang]["eem_splithalf_btn"], key="btn_run_split_half"):
                                with st.spinner(T[lang]["eem_splithalf_spinner"]):
                                    sh_res = eem_core.split_half_analysis(tensor_x, n_components=n_components_sel, random_state=42)
                                if sh_res.get("can_split", False):
                                    mean_tcc = sh_res["mean_tcc"]
                                    if sh_res["is_validated"]:
                                        st.success(T[lang]["eem_splithalf_success"].format(tcc=mean_tcc))
                                    else:
                                        st.warning(T[lang]["eem_splithalf_fail"].format(tcc=mean_tcc))

                                    st.markdown(f"**{T[lang]['eem_splithalf_tcc_title']}**")
                                    tcc_rows = []
                                    comp_names_sh = [T[lang]["eem_comp_c1"], T[lang]["eem_comp_c2"], T[lang]["eem_comp_c3"]] if n_components_sel == 3 else [f"C{r+1}" for r in range(n_components_sel)]
                                    for idx_c in range(n_components_sel):
                                        c_name = comp_names_sh[idx_c] if idx_c < len(comp_names_sh) else f"C{idx_c+1}"
                                        tcc_rows.append({
                                            "Component": c_name,
                                            "TCC Em": sh_res["tcc_em"][idx_c],
                                            "TCC Ex": sh_res["tcc_ex"][idx_c],
                                            "Status": "✅ OK (≥0.90)" if (sh_res["tcc_em"][idx_c] >= 0.90 and sh_res["tcc_ex"][idx_c] >= 0.90) else "⚠️ Low (<0.90)",
                                        })
                                    st_df(pd.DataFrame(tcc_rows))

                    st.markdown("---")
                    col_p1, col_p2 = st.columns(2)
                    comp_names = [f"C{r+1}" for r in range(n_components_sel)]
                    if n_components_sel == 3:
                        comp_names = [T[lang]["eem_comp_c1"], T[lang]["eem_comp_c2"], T[lang]["eem_comp_c3"]]

                    with col_p1:
                        fig_em = go.Figure()
                        for r in range(n_components_sel):
                            fig_em.add_trace(go.Scatter(x=em_ax, y=results["em_profiles"][:, r], mode="lines", name=comp_names[r], line=dict(width=2.5)))
                        fig_em.update_layout(title=T[lang]["eem_em_prof_title"], xaxis_title=T[lang]["eem_em_axis"], yaxis_title=T[lang]["eem_rel_int"], template="plotly_white")
                        st_plotly(fig_em)

                    with col_p2:
                        fig_ex = go.Figure()
                        for r in range(n_components_sel):
                            fig_ex.add_trace(go.Scatter(x=ex_ax, y=results["ex_profiles"][:, r], mode="lines", name=comp_names[r], line=dict(width=2.5)))
                        fig_ex.update_layout(title=T[lang]["eem_ex_prof_title"], xaxis_title=T[lang]["eem_ex_axis"], yaxis_title=T[lang]["eem_rel_int"], template="plotly_white")
                        st_plotly(fig_ex)

                    with st.expander(T[lang]["eem_openfluor_expander"], expanded=False):
                        of_matches = eem_core.match_parafac_to_openfluor(
                            results["em_profiles"], results["ex_profiles"], em_ax, ex_ax, threshold_tcc=0.85, lang=lang
                        )
                        st.markdown(f"**{T[lang]['eem_openfluor_tbl_title']}**")
                        st_df(
                            of_matches,
                            column_config={
                                "Component": st.column_config.TextColumn("Компонент" if lang == "ru" else "Component", width="small"),
                                "Best_Match_ID": st.column_config.TextColumn("ID OpenFluor", width="small"),
                                "Fluorophore_Name": st.column_config.TextColumn("Флуорофор" if lang == "ru" else "Fluorophore", width="medium"),
                                "Category": st.column_config.TextColumn("Категория" if lang == "ru" else "Category", width="small"),
                                "TCC_Mean": st.column_config.NumberColumn("TCC Средний" if lang == "ru" else "TCC Mean", format="%.3f", width="small"),
                                "Confidence_Status": st.column_config.TextColumn("Статус сходства" if lang == "ru" else "Confidence Status", width="medium"),
                                "Ecological_Origin": st.column_config.TextColumn("Экологический генезис" if lang == "ru" else "Ecological Origin", width="large"),
                            }
                        )
                        st.markdown("##### " + ("Экологический генезис идентифицированных флуорофоров:" if lang == "ru" else "Ecological Origin of Identified Fluorophores:"))
                        for _, m_row in of_matches.iterrows():
                            st.markdown(
                                f"- **{m_row.get('Component')}** ({m_row.get('Fluorophore_Name')} | TCC = {m_row.get('TCC_Mean')}): "
                                f"*{m_row.get('Confidence_Status')}* — {m_row.get('Ecological_Origin')}"
                            )
                        st.download_button(
                            label=T[lang]["eem_openfluor_dl_btn"],
                            data=of_matches.to_csv(index=False).encode("utf-8"),
                            file_name="ChemoSuite_PARAFAC_OpenFluor_Matches.csv",
                            mime="text/csv",
                            key="dl_openfluor_matches_btn",
                        )

                    st.subheader(T[lang]["eem_scores_title"])
                    score_col_names = [f"C{r+1}" for r in range(n_components_sel)]
                    if n_components_sel == 3:
                        score_col_names = ["C1_Fulvic", "C2_Lignin_Humic", "C3_Protein"]

                    scores_df = pd.DataFrame(results["scores"], columns=score_col_names)
                    scores_df.insert(0, "Sample_ID", s_ids)

                    fig_scores = px.bar(
                        scores_df, x="Sample_ID", y=score_col_names,
                        title=T[lang]["eem_scores_chart_title"], labels={"value": T[lang]["eem_int_label"], "variable": "Component"},
                        barmode="stack", template="plotly_white",
                    )
                    st_plotly(fig_scores)

                    # Экспорт объединенных оптических признаков
                    ml_export_df = pd.merge(df_indices[df_indices["Sample_ID"].isin(s_ids)], scores_df, on="Sample_ID")
                    st.session_state["eem_ml_descriptors"] = ml_export_df
                    st.download_button(
                        label=T[lang]["eem_dl_ml_btn"],
                        data=ml_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_EEM_ML_Features.csv",
                        mime="text/csv",
                        key="eem_dl_ml_btn",
                    )

        # Экран 4: УФ-Вид спектрофотометрия (UV-Vis)
        with tab_eem4:
            st.subheader(T[lang]["eem_uv_title"])
            if not uv_samples:
                st.info(T[lang]["eem_uv_no_data"])
            else:
                uv_names = [u.sample_id for u in uv_samples]
                col_uv1, col_uv2 = st.columns([3, 1])
                with col_uv1:
                    active_uv_name = st.selectbox(T[lang]["eem_uv_select_sample"], uv_names, key="active_uv_sample_select")
                active_uv = next((u for u in uv_samples if u.sample_id == active_uv_name), uv_samples[0])

                # График A(λ) с реперными точками
                fig_uv = go.Figure()
                fig_uv.add_trace(go.Scatter(
                    x=active_uv.wl, y=active_uv.absorbance,
                    mode="lines",
                    line=dict(color="#0b5394", width=2.0),
                    name="A(λ)",
                    hovertemplate="<b>λ</b>: %{x:.1f} нм<br><b>A</b>: %{y:.4f}<extra></extra>"
                ))

                for mw, col in zip([254.0, 280.0, 365.0], ["#d62728", "#e69138", "#2ca02c"]):
                    idx = np.argmin(np.abs(active_uv.wl - mw))
                    if abs(active_uv.wl[idx] - mw) <= 5.0:
                        fig_uv.add_trace(go.Scatter(
                            x=[active_uv.wl[idx]], y=[active_uv.absorbance[idx]],
                            mode="markers+text",
                            marker=dict(color=col, size=8),
                            text=[f"A_{int(mw)}={active_uv.absorbance[idx]:.3f}"],
                            textposition="top right",
                            name=f"A_{int(mw)}",
                            hoverinfo="skip"
                        ))

                fig_uv.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=0.8)
                fig_uv.update_layout(
                    title=T[lang]["eem_uv_plot_title"].format(name=active_uv.sample_id),
                    xaxis_title="Длина волны λ (нм)" if lang == "ru" else "Wavelength λ (nm)",
                    yaxis_title="Оптическая плотность A" if lang == "ru" else "Absorbance A",
                    plot_bgcolor="white",
                    height=450,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                st_plotly(fig_uv)

                # Таблица оптических дескрипторов УФ-Вид для всех загруженных проб
                uv_indices_list = [
                    eem_core.calculate_uv_vis_indices(u, doc=(u.doc if u.doc is not None else (default_doc_val if default_doc_val > 0 else None)))
                    for u in uv_samples
                ]
                df_uv_indices = pd.DataFrame(uv_indices_list)
                st.session_state["uv_ml_descriptors"] = df_uv_indices

                st.markdown(f"##### {T[lang]['eem_uv_tbl_title']}")
                c_tbl1, c_tbl2 = st.columns([3, 1])
                with c_tbl1:
                    st_df(
                        df_uv_indices.style.format({
                            "A254": "{:.4f}", "A280": "{:.4f}", "A365": "{:.4f}",
                            "E2_E3": "{:.2f}", "E4_E6": "{:.2f}",
                            "S_275_295": "{:.4f}", "S_350_400": "{:.4f}", "S_R": "{:.2f}",
                            "DOC": "{:.2f}", "SUVA254": "{:.2f}"
                        }, na_rep="—")
                    )
                with c_tbl2:
                    st.download_button(
                        label=T[lang]["eem_dl_uv_indices"],
                        data=df_uv_indices.to_csv(index=False).encode("utf-8"),
                        file_name="UV_Vis_Descriptors.csv",
                        mime="text/csv",
                        key="dl_uv_vis_descriptors_btn",
                    )

# ==============================================================================
# МОДУЛЬ 3: DATA FUSION И ХЕМОМЕТРИКА (PLS-DA & VALIDATION)
# ==============================================================================
elif active_module == T[lang]["mod3_name"]:
    st.title(T[lang]["ml_title"])
    st.caption(T[lang]["ml_subtitle"])

    with st.popover("ℹ️ " + ("Руководство: Что такое Data Fusion (Слияние данных) и зачем это нужно?" if lang == "ru" else "Guide: Multiblock Data Fusion & Chemometrics"), width="stretch"):
        st.markdown(r"""
### 🧬 Концепция Data Fusion (Многоблочная хемометрика)

При анализе природных водных объектов (Байкал, притоки) и антропогенных отходов (сульфатный шлам-лигнин БЦБК) ни один спектральный метод в отдельности не дает исчерпывающей картины:

| Аналитический метод | Преимущества | Ограничения |
| :--- | :--- | :--- |
| **3D EEM Флуоресценция** | Экспрессный анализ, селективность к флуорофорам (гуминовые и белковые пулы) | Низкое спектральное разрешение, широкие перекрывающиеся полосы |
| **УФ-Вид (UV-Vis)** | Интегральные оптические индексы ($SUVA_{254}$, $E_2/E_3$, наклон $S_{275-295}$) | Интегральный отклик, не разделяет индивидуальные химические компоненты |
| **FT-ICR MS** | Ультравысокое разрешение ($R > 400\,000$), тысячи индивидуальных брутто-формул | Высокая стоимость, трудоемкость, селективность ионизации |

---

### 🔀 Стратегии слияния блоков (Data Fusion):

1. **Low-Level Fusion (Низкоуровневое слияние):**
   * Все таблицы дескрипторов склеиваются по строкам (образцам) в единую широкую матрицу.
   * **Масштабирование $1/\sqrt{P}$ (Block Scaling):** Чтобы блок FT-ICR (сотни ячеек/формул) математически не заглушил оптический блок (несколько индексов), каждый блок делится на корень из числа его переменных. Все аналитические платформы получают равный начальный вес.

2. **Mid-Level Fusion (Среднеуровневое слияние):**
   * Из каждого блока сначала независимо извлекаются скрытые латентные факторы (главные компоненты PCA или PARAFAC-оценки $C_1, C_2, C_3$).
   * Затем полученные сжатые векторы объединяются для финальной классификации.

---

### 🎯 Результат моделирования:
* **PLS-DA / OPLS-DA:** Разделение образцов на классы (например, *Фоновый Байкал* vs *Шлам-лигнин*).
* **VIP-биомаркеры (Variable Importance in Projection):** Выявление конкретных формул и оптических параметров с $VIP > 1.0$, вносящих определяющий вклад в экологическое состояние водоема.
""" if lang == "ru" else r"""
### 🧬 Multiblock Data Fusion Concept in Environmental Science

In monitoring aquatic ecosystems (Lake Baikal) and industrial pulp waste (kraft sludge lignin), no single analytical technique provides the complete picture:

| Analytical Method | Strengths | Limitations |
| :--- | :--- | :--- |
| **3D EEM Fluorescence** | Rapid screening, selective for aromatic & protein fluorophores | Low spectral resolution, broad overlapping peaks |
| **UV-Vis Spectrophotometry** | Bulk optical indices ($SUVA_{254}$, $E_2/E_3$, slope $S_{275-295}$) | Non-specific, cannot distinguish individual molecules |
| **FT-ICR Mass Spectrometry** | Ultra-high resolution ($R > 400\,000$), thousands of elemental formulas | High instrument cost, ionization selectivity |

---

### 🔀 Data Fusion Strategies:

1. **Low-Level Fusion:** Concatenation of feature blocks with $1/\sqrt{P}$ block scaling to prevent high-dimensional mass spectra from dominating optical metrics.
2. **Mid-Level Fusion:** Independent extraction of latent factors (PCA / PARAFAC components) followed by unified classification.

---

### 🎯 Objective:
Accurate discrimination of pristine vs contaminated waters and non-target identification of high-VIP environmental markers ($VIP > 1.0$).
""")

    if not CHEMO_ML_AVAILABLE or chemo_ml is None:
        st.error(T[lang]["ml_err_missing"])
        st.stop()

    st.subheader(T[lang]["ml_sec1_title"])

    # Выбор источника данных
    data_source_mode = st.radio(
        "Источник мультимодальных данных:" if lang == "ru" else "Multimodal Data Source:",
        [T[lang]["ml_src_session"], T[lang]["ml_src_files"], T[lang]["ml_src_batch"], T[lang]["ml_src_benchmark"]],
        horizontal=True,
        key="ml_data_source_mode_radio",
    )

    # 1. Сборка из сессии
    if data_source_mode == T[lang]["ml_src_session"]:
        assigned_spectra = {
            k: v["assigned_df"] for k, v in st.session_state.get("spectra_db", {}).items()
            if v.get("assigned_df") is not None and not v["assigned_df"].empty
        }
        eem_df = st.session_state.get("eem_ml_descriptors", None)
        uv_df = st.session_state.get("uv_ml_descriptors", None)
        ftir_df = st.session_state.get("ftir_ml_descriptors", None)
        nmr_df = st.session_state.get("nmr_ml_descriptors", None)
        thermal_df = st.session_state.get("thermal_ml_descriptors", None)
        sec_df = st.session_state.get("sec_ml_descriptors", None)
        elem_df = st.session_state.get("elemental_ml_descriptors", None)

        n_fticr_samples = len(st.session_state["fticr_batch_summary"]) if ("fticr_batch_summary" in st.session_state and not st.session_state["fticr_batch_summary"].empty) else len(assigned_spectra)
        st.info(
            f"📊 **{'Доступные аналитические модальности в сессии:' if lang == 'ru' else 'Available Analytical Modalities in Session:'}** "
            f"FT-ICR MS: **{n_fticr_samples}** | EEM: **{len(eem_df) if eem_df is not None else 0}** | "
            f"UV-Vis: **{len(uv_df) if uv_df is not None else 0}** | "
            f"FTIR: **{len(ftir_df) if ftir_df is not None else 0}** | "
            f"NMR: **{len(nmr_df) if nmr_df is not None else 0}** | "
            f"Thermal: **{len(thermal_df) if thermal_df is not None else 0}** | "
            f"SEC: **{len(sec_df) if sec_df is not None else 0}** | "
            f"CHNS-O: **{len(elem_df) if elem_df is not None else 0}**"
        )

        if st.button(T[lang]["ml_btn_assemble_session"], type="primary", key="btn_assemble_session"):
            if not assigned_spectra and ("fticr_batch_summary" not in st.session_state or st.session_state["fticr_batch_summary"].empty) and (eem_df is None or eem_df.empty) and (uv_df is None or uv_df.empty) and (ftir_df is None or ftir_df.empty) and (nmr_df is None or nmr_df.empty) and (thermal_df is None or thermal_df.empty) and (sec_df is None or sec_df.empty) and (elem_df is None or elem_df.empty):
                st.warning(T[lang]["ml_session_need_more"])
            else:
                if "fticr_batch_summary" in st.session_state and not st.session_state["fticr_batch_summary"].empty:
                    df_ms_assembled = st.session_state["fticr_batch_summary"]
                else:
                    fticr_rows = []
                    for s_name, a_df in assigned_spectra.items():
                        desc = chemo_ml.extract_fticr_descriptors(a_df, s_name)
                        if desc:
                            fticr_rows.append(desc)
                    df_ms_assembled = pd.DataFrame(fticr_rows) if fticr_rows else pd.DataFrame()

                merged_df = chemo_ml.merge_feature_blocks(df_ms_assembled, eem_df, df_uv=uv_df, df_ftir=ftir_df, df_nmr=nmr_df, df_thermal=thermal_df, df_sec=sec_df, df_elem=elem_df)
                if not merged_df.empty:
                    if "Class_Target" not in merged_df.columns:
                        # Разметка по умолчанию: 1 для образцов со словом lignin / impact
                        merged_df["Class_Target"] = merged_df["Sample_ID"].apply(
                            lambda s: 1 if any(k in str(s).lower() for k in ["lignin", "impact", "шлам"]) else 0
                        )
                    st.session_state["fused_data"] = merged_df
                    st.session_state["fused_source"] = "Сессионные спектры (8 модальностей)" if lang == "ru" else "Session Spectra (8 Modalities)"
                    st.success(f"Успешно объединено образцов: {len(merged_df)}" if lang == "ru" else f"Successfully assembled samples: {len(merged_df)}")
                    st.rerun()
                else:
                    st.warning("Не удалось сопоставить образцы между модальностями по Sample_ID." if lang == "ru" else "Failed to match samples across modalities by Sample_ID.")

    # 2. Загрузка внешних файлов
    elif data_source_mode == T[lang]["ml_src_files"]:
        upload_type = st.radio(
            T[lang]["ml_uploader_mode"],
            [T[lang]["ml_upload_two"], T[lang]["ml_upload_single"]],
            horizontal=True,
            key="ml_upload_type_radio"
        )
        if upload_type == T[lang]["ml_upload_two"]:
            col_u1, col_u2, col_u3, col_u4, col_u5, col_u6, col_u7, col_u8 = st.columns(8)
            with col_u1:
                f_ms = st.file_uploader(
                    T[lang]["ml_up_ms"],
                    type=["csv", "tsv", "txt"],
                    key="up_ms_features",
                    help="Таблица дескрипторов FT-ICR (Sample_ID, 20 ячеек, AI, DBE). Демо: demo_data/multimodal_ml/chemo_fticr_block.csv" if lang == "ru" else "FT-ICR descriptor table (Sample_ID, 20 cells). Demo: demo_data/multimodal_ml/chemo_fticr_block.csv",
                )
            with col_u2:
                f_eem = st.file_uploader(
                    T[lang]["ml_up_eem"],
                    type=["csv", "tsv", "txt"],
                    key="up_eem_features",
                    help="Таблица дескрипторов EEM (Sample_ID, FI, HIX, C1..C3). Демо: demo_data/multimodal_ml/chemo_eem_block.csv" if lang == "ru" else "EEM descriptor table (Sample_ID, FI, HIX). Demo: demo_data/multimodal_ml/chemo_eem_block.csv",
                )
            with col_u3:
                f_uv = st.file_uploader(
                    T[lang]["ml_up_uv"],
                    type=["csv", "tsv", "txt"],
                    key="up_uv_features",
                    help="Таблица дескрипторов UV-Vis (Sample_ID, A254, E2/E3, SR, Mw). Демо: demo_data/multimodal_ml/chemo_uv_block.csv" if lang == "ru" else "UV-Vis descriptor table (Sample_ID, A254, E2/E3). Demo: demo_data/multimodal_ml/chemo_uv_block.csv",
                )
            with col_u4:
                f_ftir = st.file_uploader(
                    T[lang]["ml_up_ftir"],
                    type=["csv", "tsv", "txt"],
                    key="up_ftir_features",
                    help="Таблица дескрипторов FTIR (Sample_ID, индексы, площади). Демо: demo_data/ftir/ftir_descriptors_matrix.csv" if lang == "ru" else "FTIR descriptor table. Demo: demo_data/ftir/ftir_descriptors_matrix.csv",
                )
            with col_u5:
                f_nmr = st.file_uploader(
                    T[lang]["ml_up_nmr"],
                    type=["csv", "tsv", "txt"],
                    key="up_nmr_features",
                    help="Таблица дескрипторов 13C ЯМР (Sample_ID, fa, HB/HI, %C). Демо: demo_data/nmr/nmr_descriptors_matrix.csv" if lang == "ru" else "NMR descriptor table. Demo: demo_data/nmr/nmr_descriptors_matrix.csv",
                )
            with col_u6:
                f_thermal = st.file_uploader(
                    T[lang]["ml_up_thermal"],
                    type=["csv", "tsv", "txt"],
                    key="up_thermal_features",
                    help="Таблица дескрипторов Thermal (Sample_ID, S/G, TI, T50). Демо: demo_data/thermal/thermal_descriptors_matrix.csv" if lang == "ru" else "Thermal descriptor table. Demo: demo_data/thermal/thermal_descriptors_matrix.csv",
                )
            with col_u7:
                f_sec = st.file_uploader(
                    T[lang]["ml_up_sec"],
                    type=["csv", "tsv", "txt"],
                    key="up_sec_features",
                    help="Таблица дескрипторов SEC/GPC (Sample_ID, Mn, Mw, Đ, Frac). Демо: demo_data/sec/sec_descriptors_matrix.csv" if lang == "ru" else "SEC/GPC descriptor table. Demo: demo_data/sec/sec_descriptors_matrix.csv",
                )
            with col_u8:
                f_elem = st.file_uploader(
                    T[lang]["ml_up_elemental"],
                    type=["csv", "tsv", "txt"],
                    key="up_elem_features",
                    help="Таблица дескрипторов Elemental CHNS-O (Sample_ID, C, H, N, S, O, H/C). Демо: demo_data/elemental/elemental_benchmark_12samples.csv" if lang == "ru" else "Elemental descriptor table. Demo: demo_data/elemental/elemental_benchmark_12samples.csv",
                )

            if f_ms or f_eem or f_uv or f_ftir or f_nmr or f_thermal or f_sec or f_elem:
                try:
                    df_up_ms = pd.read_csv(f_ms) if f_ms else None
                    df_up_eem = pd.read_csv(f_eem) if f_eem else None
                    df_up_uv = pd.read_csv(f_uv) if f_uv else None
                    df_up_ftir = pd.read_csv(f_ftir) if f_ftir else None
                    df_up_nmr = pd.read_csv(f_nmr) if f_nmr else None
                    df_up_thermal = pd.read_csv(f_thermal) if f_thermal else None
                    df_up_sec = pd.read_csv(f_sec) if f_sec else None
                    df_up_elem = pd.read_csv(f_elem) if f_elem else None
                    merged = chemo_ml.merge_feature_blocks(
                        df_up_ms, df_up_eem, df_up_uv,
                        df_ftir=df_up_ftir, df_nmr=df_up_nmr,
                        df_thermal=df_up_thermal, df_sec=df_up_sec,
                        df_elem=df_up_elem
                    )
                    if not merged.empty:
                        if "Class_Target" not in merged.columns:
                            merged["Class_Target"] = 0
                        st.session_state["fused_data"] = merged
                        src_names = [f.name for f in [f_ms, f_eem, f_uv, f_ftir, f_nmr, f_thermal, f_sec, f_elem] if f is not None]
                        st.session_state["fused_source"] = " + ".join(src_names)
                except Exception as e:
                    st.error(f"⚠️ Ошибка загрузки файлов: {e}")
                    st.info(
                        "💡 **Подсказка:** Для Модуля 3 требуются сводные таблицы дескрипторов по образцам из каталога `demo_data/`."
                        if lang == "ru" else
                        "💡 **Tip:** Module 3 requires summary descriptor tables across samples from `demo_data/`."
                    )
        else:
            f_single = st.file_uploader(
                T[lang]["ml_up_single"],
                type=["csv", "tsv", "txt"],
                key="up_single_features",
                help="Единая мультимодальная матрица признаков. Демо: demo_data/multimodal_ml/chemo_unified_multimodal.csv" if lang == "ru" else "Unified multimodal matrix with Sample_ID. Demo: demo_data/multimodal_ml/chemo_unified_multimodal.csv",
            )
            if f_single:
                try:
                    df_s = pd.read_csv(f_single)
                    if "Sample_ID" not in df_s.columns:
                        raise KeyError("В таблице отсутствует колонка 'Sample_ID'. Для единого файла дескрипторов используйте demo_data/multimodal_ml/chemo_unified_multimodal.csv.")
                    if "Class_Target" not in df_s.columns:
                        df_s["Class_Target"] = 0
                    st.session_state["fused_data"] = df_s
                    st.session_state["fused_source"] = f_single.name
                except Exception as e:
                    st.error(f"⚠️ Ошибка чтения файла: {e}")
                    st.info(
                        "💡 **Подсказка:** Для единого файла используйте `demo_data/multimodal_ml/chemo_unified_multimodal.csv`."
                        if lang == "ru" else
                        "💡 **Tip:** For a single file, use `demo_data/multimodal_ml/chemo_unified_multimodal.csv`."
                    )

    # 3. Автономный пакетный скрининг директорий
    elif data_source_mode == T[lang]["ml_src_batch"]:
        if not BATCH_SCREENER_AVAILABLE or batch_screener is None:
            st.error(
                "⚠️ Модуль `batch_screener.py` не доступен. Убедитесь, что файл находится в папке `modules/`."
                if lang == "ru" else
                "⚠️ `batch_screener.py` module is not available. Ensure it is located in the `modules/` folder."
            )
        else:
            st.caption(T[lang]["batch_screener_desc"])
            batch_input_mode = st.radio(
                T[lang]["batch_source_type_label"],
                [T[lang]["batch_src_folder"], T[lang]["batch_src_upload"]],
                horizontal=True,
                key="batch_input_mode_radio"
            )

            if batch_input_mode == T[lang]["batch_src_folder"]:
                col_b1, col_b2 = st.columns([3, 1])
                with col_b1:
                    batch_dir_path = st.text_input(
                        T[lang]["batch_dir_label"],
                        value="demo_data",
                        help="Относительный или абсолютный путь к каталогу со спектрами." if lang == "ru" else "Relative or absolute path to spectra folder.",
                        key="batch_dir_path_input"
                    )
                with col_b2:
                    st.write("")
                    st.write("")
                    batch_recursive = st.checkbox(
                        T[lang]["batch_recursive_label"],
                        value=True,
                        key="batch_recursive_chk"
                    )

                if st.button(T[lang]["batch_run_btn"], type="primary", key="btn_run_batch_screening"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    def _prog_cb(frac: float, msg: str):
                        progress_bar.progress(min(max(frac, 0.0), 1.0))
                        status_text.text(msg)

                    try:
                        res = batch_screener.batch_screen_directory(
                            batch_dir_path,
                            recursive=batch_recursive,
                            progress_callback=_prog_cb,
                        )
                        progress_bar.progress(1.0)
                        status_text.text("Готово!" if lang == "ru" else "Done!")
                        st.session_state["batch_screening_results"] = res
                        st.success(
                            T[lang]["batch_success_msg"].format(
                                n_files=res["statistics"]["total_files"],
                                n_samples=res["statistics"]["unique_samples"]
                            )
                        )
                    except Exception as exc:
                        st.error(f"Ошибка при скрининге: {exc}" if lang == "ru" else f"Screening error: {exc}")

            else:
                uploaded_batch_files = st.file_uploader(
                    T[lang]["batch_upload_label"],
                    type=["csv", "tsv", "txt", "xy", "jdx", "dx"],
                    accept_multiple_files=True,
                    key="batch_uploaded_files_widget"
                )
                if uploaded_batch_files and st.button(T[lang]["batch_run_btn"], type="primary", key="btn_run_batch_screening_upload"):
                    files_tuples = [(f.name, f.getvalue()) for f in uploaded_batch_files]
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    def _prog_cb(frac: float, msg: str):
                        progress_bar.progress(min(max(frac, 0.0), 1.0))
                        status_text.text(msg)

                    try:
                        res = batch_screener.batch_screen_files(files_tuples, progress_callback=_prog_cb)
                        progress_bar.progress(1.0)
                        status_text.text("Готово!" if lang == "ru" else "Done!")
                        st.session_state["batch_screening_results"] = res
                        st.success(
                            T[lang]["batch_success_msg"].format(
                                n_files=res["statistics"]["total_files"],
                                n_samples=res["statistics"]["unique_samples"]
                            )
                        )
                    except Exception as exc:
                        st.error(f"Ошибка при обработке файлов: {exc}" if lang == "ru" else f"File processing error: {exc}")

            # Отображение сводных результатов скрининга, если они есть в session_state
            batch_res = st.session_state.get("batch_screening_results", None)
            if batch_res is not None:
                stats = batch_res.get("statistics", {})
                st.markdown("---")
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                c_m1.metric(T[lang]["batch_files_count"], stats.get("total_files", 0))
                c_m2.metric(T[lang]["batch_success_count"], stats.get("successful", 0))
                c_m3.metric(T[lang]["batch_samples_count"], stats.get("unique_samples", 0))
                c_m4.metric(T[lang]["batch_modalities_count"], len(stats.get("modalities_found", [])))

                if stats.get("modalities_found"):
                    mods_str = " • ".join([f"`{m}`" for m in stats["modalities_found"]])
                    st.markdown(f"**{'Распознанные методы:' if lang == 'ru' else 'Detected Modalities:'}** {mods_str}")

                st.markdown(f"**{T[lang]['batch_manifest_title']}**")
                st.dataframe(batch_res["summary_df"], use_container_width=True, height=220)

                col_act1, col_act2 = st.columns([1, 1])
                with col_act1:
                    if st.button(T[lang]["batch_use_ml_btn"], type="primary", key="btn_apply_batch_to_ml"):
                        st.session_state["fused_data"] = batch_res["fused_df"]
                        st.session_state["fused_source"] = (
                            f"Batch Screening ({stats.get('unique_samples', 0)} проб, {len(stats.get('modalities_found', []))} методов)"
                            if lang == "ru" else
                            f"Batch Screening ({stats.get('unique_samples', 0)} samples, {len(stats.get('modalities_found', []))} modalities)"
                        )
                        st.success("Матрица признаков успешно передана в классификатор ML!" if lang == "ru" else "Feature matrix successfully dispatched to ML Classifier!")
                        st.rerun()

                with col_act2:
                    try:
                        zip_data = batch_screener.export_batch_results_zip(batch_res)
                        st.download_button(
                            label=T[lang]["batch_dl_zip_btn"],
                            data=zip_data,
                            file_name="ChemoSuite_Batch_Screening_Archive.zip",
                            mime="application/zip",
                            key="btn_download_batch_zip"
                        )
                    except Exception as exc:
                        st.error(f"Ошибка подготовки архива: {exc}" if lang == "ru" else f"Archive packaging error: {exc}")

    # 4. Синтетический бенчмарк
    else:
        if st.button(T[lang]["ml_load_demo_btn"], type="primary", key="btn_ml_load_demo_main"):
            demo_df, demo_y = chemo_ml.generate_multimodal_benchmark()
            demo_df["Class_Target"] = demo_y
            st.session_state["fused_data"] = demo_df
            st.session_state["fused_source"] = T[lang]["ml_demo_source_name"]
            st.rerun()

    # Отображение активного датасета и редактора классов
    fused_df = st.session_state.get("fused_data", None)
    if fused_df is None:
        st.info(T[lang]["ml_info_no_data"])
    else:
        src_name = st.session_state.get("fused_source", T[lang]["ml_custom_source_name"])
        st.markdown(T[lang]["ml_active_dataset_info"].format(src=src_name, n=len(fused_df)))
        st.caption(T[lang]["ml_classes_caption"])

        # Редактор классов с поддержкой текстовых меток и мультиклассов
        fused_df_display = fused_df.copy()
        if "Class_Target" in fused_df_display.columns:
            fused_df_display["Class_Target"] = fused_df_display["Class_Target"].astype(str)

        edited_df = st.data_editor(
            fused_df_display,
            column_config={
                "Class_Target": st.column_config.TextColumn(
                    T[lang]["ml_col_class_target"],
                    required=True,
                    help="Метка класса (например: 0, 1, 2 или Baikal, Lignin, Sediment)" if lang == "ru" else "Class label (e.g. 0, 1, 2 or Baikal, Lignin, Sediment)",
                )
            },
            disabled=[c for c in fused_df_display.columns if c != "Class_Target"],
            key="ml_fused_editor",
        )

        st.markdown("---")
        st.subheader(T[lang]["ml_sec2_title"])

        col_m1, col_m2 = st.columns([1.3, 2.7])
        with col_m1:
            model_type_sel = st.selectbox(
                T[lang]["ml_model_type_label"],
                [T[lang]["ml_model_plsda"], T[lang]["ml_model_oplsda"], T[lang]["ml_model_splsda"]],
                index=0,
                key="ml_model_arch_select",
            )
            is_opls = (model_type_sel == T[lang]["ml_model_oplsda"])
            is_spls = (model_type_sel == T[lang]["ml_model_splsda"])

        max_allowed_lvs = max(2, min(5, len(edited_df) - 1))
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            strategy_choice = st.selectbox(
                T[lang]["ml_strategy_label"],
                [T[lang]["ml_strat_low"], T[lang]["ml_strat_mid"]],
                index=0,
                key="pls_strategy_select",
            )
            strat_code = "mid_level" if strategy_choice == T[lang]["ml_strat_mid"] else "low_level"
        with col_p2:
            if is_opls:
                max_ortho = max(1, min(3, len(edited_df) - 2))
                if max_ortho > 1:
                    n_ortho = st.slider(T[lang]["ml_ortho_slider"], min_value=1, max_value=max_ortho, value=1, key="opls_ortho_slider")
                else:
                    n_ortho = 1
                    st.caption(f"{T[lang]['ml_ortho_slider']} **1**")
                use_perm = False
            elif is_spls:
                if max_allowed_lvs > 2:
                    n_lvs = st.slider(T[lang]["ml_lvs_slider"], min_value=2, max_value=max_allowed_lvs, value=2, key="pls_lvs_slider")
                else:
                    n_lvs = 2
                    st.caption(f"{T[lang]['ml_lvs_slider']} **2**")
                feat_count = max(1, len(edited_df.columns) - 2)
                max_keep = max(4, min(50, feat_count))
                if max_keep > 3:
                    keep_x_val = st.slider(T[lang]["ml_spls_keep_slider"], min_value=3, max_value=max_keep, value=min(15, max_keep), step=1, key="spls_keep_slider")
                else:
                    keep_x_val = min(3, feat_count)
                    st.caption(f"{T[lang]['ml_spls_keep_slider']} **{keep_x_val}**")
                use_perm = False
            else:
                if max_allowed_lvs > 2:
                    n_lvs = st.slider(T[lang]["ml_lvs_slider"], min_value=2, max_value=max_allowed_lvs, value=2, key="pls_lvs_slider")
                else:
                    n_lvs = 2
                    st.caption(f"{T[lang]['ml_lvs_slider']} **2**")
                use_perm = st.checkbox(T[lang]["ml_perm_chk"], value=True, key="pls_perm_chk")
                if use_perm:
                    n_perm_iter = st.select_slider(
                        "Число пермутаций / Permutations" if lang == "ru" else "Permutation Iterations",
                        options=[50, 100, 200, 500, 1000],
                        value=100,
                        help="50–100: экспресс-скрининг; 500–1000: публикационный стандарт (min p-value <= 0.001)" if lang == "ru" else "50–100: screening; 500–1000: publication standard (min p-value <= 0.001)",
                        key="pls_perm_iter_slider",
                    )
                else:
                    n_perm_iter = 50
        with col_p3:
            use_block_scale = st.checkbox(T[lang]["ml_block_scale"], value=True, key="pls_block_scale_chk")
            st.write("")
            run_btn_label = "🚀 Обучить sPLS-DA" if (lang == "ru" and is_spls) else ("🚀 Fit sPLS-DA Model" if is_spls else ("🚀 Обучить OPLS-DA" if (lang == "ru" and is_opls) else ("🚀 Fit OPLS-DA Model" if is_opls else T[lang]["ml_run_btn"])))
            run_model = st.button(run_btn_label, type="primary", key="btn_run_chemo_model")

        if run_model:
            feat_cols = [c for c in edited_df.columns if c not in ["Sample_ID", "Class_Target"]]
            X_input = edited_df[feat_cols]
            y_input = edited_df["Class_Target"].values

            # Методологическое предупреждение при малом N (p >> n)
            if len(edited_df) < 20 and len(feat_cols) > len(edited_df):
                st.info(
                    f"ℹ️ **Методологическое замечание ($p \\gg n$):** Выборка содержит {len(edited_df)} образцов при {len(feat_cols)} признаках. "
                    f"В многомерных моделях (PLS-DA / OPLS-DA) при малом $N$ возможно переобучение и завышенное разделение классов. "
                    f"Для надежных геохимических выводов в публикациях рекомендуется использовать выборку $N \\ge 30$."
                    if lang == "ru"
                    else f"ℹ️ **Methodological Note ($p \\gg n$):** Dataset contains {len(edited_df)} samples across {len(feat_cols)} descriptors. "
                    f"High-dimensional models (PLS-DA / OPLS-DA) risk over-optimistic class separation at small $N$. "
                    f"For publication-grade geochemical conclusions, a sample size of $N \\ge 30$ is recommended."
                )

            unique_classes = np.unique(y_input)
            if len(unique_classes) < 2:
                st.warning(T[lang]["ml_warn_two_classes"])
            elif is_opls and len(unique_classes) > 2:
                st.warning(T[lang]["ml_warn_opls_binary"])
            else:
                try:
                    with st.spinner(T[lang]["ml_spinner"]):
                        if is_opls:
                            res = chemo_ml.train_oplsda_model(
                                X_input, y_input,
                                n_ortho=n_ortho,
                                fusion_strategy=strat_code,
                                block_scaling=use_block_scale,
                            )
                            st.session_state["pls_results"] = res
                            st.session_state["pls_sample_ids"] = edited_df["Sample_ID"].values
                            st.session_state["pls_y_input"] = y_input
                            st.session_state["perm_results"] = None
                        elif is_spls:
                            res = chemo_ml.train_splsda_model(
                                X_input, y_input,
                                n_components=n_lvs,
                                keep_x=keep_x_val,
                                fusion_strategy=strat_code,
                                block_scaling=use_block_scale,
                            )
                            st.session_state["pls_results"] = res
                            st.session_state["pls_sample_ids"] = edited_df["Sample_ID"].values
                            st.session_state["pls_y_input"] = y_input
                            st.session_state["perm_results"] = None
                        else:
                            res = chemo_ml.train_plsda_model(
                                X_input, y_input,
                                n_components=n_lvs,
                                fusion_strategy=strat_code,
                                block_scaling=use_block_scale,
                            )
                            st.session_state["pls_results"] = res
                            st.session_state["pls_sample_ids"] = edited_df["Sample_ID"].values
                            st.session_state["pls_y_input"] = y_input

                            if use_perm:
                                perm_res = chemo_ml.run_permutation_test(
                                    X_input, y_input,
                                    n_components=n_lvs,
                                    n_permutations=n_perm_iter,
                                    fusion_strategy=strat_code,
                                    block_scaling=use_block_scale,
                                )
                                st.session_state["perm_results"] = perm_res
                            else:
                                st.session_state["perm_results"] = None

                except ValueError as val_err:
                    st.error(T[lang]["ml_warn_input"].format(err=val_err))

        # Вывод результатов из session_state
        res = st.session_state.get("pls_results", None)
        if res is not None:
            sample_ids_curr = st.session_state.get("pls_sample_ids", edited_df["Sample_ID"].values)
            y_curr = st.session_state.get("pls_y_input", edited_df["Class_Target"].values)
            curr_model_type = res.get("model_type", "PLS-DA")
            is_model_opls = (curr_model_type == "OPLS-DA")

            t1_vals = res.get("t1", res.get("scores_t1"))
            t2_vals = res.get("t2", res.get("scores_t2"))
            t3_vals = res.get("t3", res.get("scores_t3", None))
            ell_x = res.get("ell_x", res.get("ellipse_x", np.array([])))
            ell_y = res.get("ell_y", res.get("ellipse_y", np.array([])))
            vip_df = res.get("VIP_df", res.get("vip_df", pd.DataFrame()))
            perm_res = st.session_state.get("perm_results", None)

            # Палитра классов
            unique_cls = list(pd.unique(y_curr))
            palette = ["#D62728", "#0020C2", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
            cls_color_map = {str(c): palette[i % len(palette)] for i, c in enumerate(unique_cls)}
            point_colors = [cls_color_map.get(str(c), "#7F7F7F") for c in y_curr]

            st.markdown("---")
            st.subheader(f"{T[lang]['ml_sec3_title']} ({curr_model_type})")

            # 4 ключевые сводные метрики
            m1, m2, m3, m4 = st.columns(4)
            if is_model_opls:
                m1.metric(f"{T[lang]['ml_m_r2x_pred']} / {T[lang]['ml_m_r2x_ortho']}", f"{res.get('R2X_pred', 0.0):.1f}% / {res.get('R2X_ortho', 0.0):.1f}%")
            else:
                m1.metric(T[lang]["ml_m_r2x"], f"{res.get('R2X', 0.0):.1f}%")
            m2.metric(T[lang]["ml_m_r2y"], f"{res.get('R2Y', 0.0):.1f}%")
            q2_val = res.get("Q2", 0.0)
            m3.metric(T[lang]["ml_m_q2"], f"{q2_val:.1f}%", delta=T[lang]["ml_q2_ok"] if q2_val > 50 else T[lang]["ml_q2_warn"])
            acc_val = res.get("Balanced_Accuracy", res.get("Accuracy", 0.0))
            m4.metric(T[lang]["ml_m_acc"], f"{acc_val:.1f}%")

            # Вкладки детального дашборда
            tab_names = [
                T[lang]["ml_tab_scores"],
                T[lang]["ml_tab_vips"],
            ]
            if is_model_opls:
                tab_names.append(T[lang]["ml_tab_splot"])
            tab_names.extend([
                T[lang]["ml_tab_val"],
                T[lang]["ml_tab_blocks"],
                T[lang]["ml_tab_crosscorr"],
                "💾 Экспорт отчетов" if lang == "ru" else "💾 Export Reports",
            ])

            tabs = st.tabs(tab_names)
            tab_idx = 0
            tab_ml1 = tabs[tab_idx]; tab_idx += 1
            tab_ml2 = tabs[tab_idx]; tab_idx += 1
            tab_splot = tabs[tab_idx] if is_model_opls else None
            if is_model_opls: tab_idx += 1
            tab_ml3 = tabs[tab_idx]; tab_idx += 1
            tab_ml4 = tabs[tab_idx]; tab_idx += 1
            tab_crosscorr = tabs[tab_idx]; tab_idx += 1
            tab_ml5 = tabs[tab_idx]

            # Таб 1: Scores Plot (2D и 3D для PLS-DA; tp vs to для OPLS-DA)
            with tab_ml1:
                has_3d = (t3_vals is not None and not is_model_opls)
                if has_3d:
                    scores_mode = st.radio(
                        T[lang]["ml_scores_mode_label"],
                        [T[lang]["ml_scores_2d"], T[lang]["ml_scores_3d"]],
                        horizontal=True,
                        key="radio_scores_2d_3d",
                    )
                else:
                    scores_mode = T[lang]["ml_scores_2d"]

                if scores_mode == T[lang]["ml_scores_3d"] and has_3d:
                    fig_3d = go.Figure()

                    # 3D Эллипсоид Хотеллинга
                    ell_3d_x = res.get("ell_3d_x", res.get("ell_x_3d"))
                    ell_3d_y = res.get("ell_3d_y", res.get("ell_y_3d"))
                    ell_3d_z = res.get("ell_3d_z", res.get("ell_z_3d"))
                    if ell_3d_x is not None and len(ell_3d_x) > 0:
                        fig_3d.add_trace(go.Surface(
                            x=ell_3d_x, y=ell_3d_y, z=ell_3d_z,
                            opacity=0.18,
                            colorscale=[[0, "#8da9c4"], [1, "#8da9c4"]],
                            showscale=False,
                            name=f"{T[lang]['ml_hotelling_name']} (3D)",
                            hoverinfo="skip",
                        ))

                    # 3D маркеры проб
                    fig_3d.add_trace(go.Scatter3d(
                        x=t1_vals, y=t2_vals, z=t3_vals,
                        mode="markers+text",
                        text=sample_ids_curr,
                        textposition="top center",
                        marker=dict(size=7, color=point_colors, line=dict(width=1, color="black"), opacity=0.9),
                        hovertext=[
                            f"<b>{s}</b><br>Class: {c}<br>LV1: {v1:.2f}, LV2: {v2:.2f}, LV3: {v3:.2f}"
                            for s, c, v1, v2, v3 in zip(sample_ids_curr, y_curr, t1_vals, t2_vals, t3_vals)
                        ],
                        hoverinfo="text",
                        name="Samples",
                    ))
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title="LV1 (Scores t1)",
                            yaxis_title="LV2 (Scores t2)",
                            zaxis_title="LV3 (Scores t3)",
                        ),
                        height=580,
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st_plotly(fig_3d)

                else:
                    # 2D Scores Plot
                    fig_sc = go.Figure()
                    if ell_x is not None and len(ell_x) > 0:
                        fig_sc.add_trace(go.Scatter(
                            x=ell_x, y=ell_y, mode="lines",
                            line=dict(dash="dot", color="#7f7f7f", width=1.5),
                            name=T[lang]["ml_hotelling_name"], hoverinfo="skip",
                        ))

                    axis_x_lbl = "t_pred (Predictive LV1)" if is_model_opls else "LV1"
                    axis_y_lbl = "t_ortho (Orthogonal LV1)" if is_model_opls else "LV2"

                    for c_name in unique_cls:
                        mask = [str(y) == str(c_name) for y in y_curr]
                        s_sub = [s for s, m in zip(sample_ids_curr, mask) if m]
                        t1_sub = [t for t, m in zip(t1_vals, mask) if m]
                        t2_sub = [t for t, m in zip(t2_vals, mask) if m]
                        col = cls_color_map.get(str(c_name), "#7F7F7F")

                        fig_sc.add_trace(go.Scatter(
                            x=t1_sub, y=t2_sub, mode="markers+text",
                            text=s_sub, textposition="top center",
                            marker=dict(size=11, color=col, line=dict(width=1, color="black"), opacity=0.85),
                            hovertext=[f"<b>{s}</b><br>Class: {c_name}<br>{axis_x_lbl}: {v1:.2f}, {axis_y_lbl}: {v2:.2f}" for s, v1, v2 in zip(s_sub, t1_sub, t2_sub)],
                            hoverinfo="text", name=f"Class {c_name}",
                        ))

                    fig_sc.update_layout(xaxis_title=axis_x_lbl, yaxis_title=axis_y_lbl, plot_bgcolor="white", height=520)
                    st_plotly(fig_sc)

            # Таб 2: VIP Scores / sPLS-DA Sparse Biomarkers
            with tab_ml2:
                sparse_df = res.get("Sparse_Loadings_df", res.get("sparse_df", pd.DataFrame()))
                if curr_model_type == "Sparse PLS-DA (sPLS-DA)" and not sparse_df.empty:
                    st.write(f"### {T[lang]['ml_spls_biomarkers_title']}")
                    sel_only = sparse_df[sparse_df["Is_Selected"]].copy()
                    top_sparse = sel_only.head(20).copy() if not sel_only.empty else sparse_df.head(15).copy()
                    fig_sp = px.bar(
                        top_sparse, x="Absolute_Weight", y="Descriptor", orientation="h",
                        color="Block",
                        color_discrete_map={"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "FTIR": "#8e44ad", "NMR": "#16a085", "Thermal": "#c0392b", "SEC/GPC": "#008080", "Other": "#7f7f7f"},
                        text=top_sparse["Absolute_Weight"].apply(lambda v: f"{v:.3f}"),
                    )
                    fig_sp.update_layout(
                        yaxis=dict(autorange="reversed", title="Descriptor"),
                        xaxis=dict(title=T[lang]["ml_spls_weight_axis"]),
                        plot_bgcolor="white", height=520
                    )
                    st_plotly(fig_sp)

                    st.markdown(f"**{T[lang]['ml_spls_table_title']} ({len(sel_only)} {T[lang]['ml_spls_selected_count']}):**")
                    st_df(sparse_df)
                    st.download_button(
                        label=T[lang]["ml_spls_dl_btn"],
                        data=sparse_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_sPLSDA_Biomarkers.csv",
                        mime="text/csv",
                        key="dl_splsda_biomarkers",
                    )
                elif not vip_df.empty:
                    top_vip = vip_df.head(15).copy()
                    fig_vip = px.bar(
                        top_vip, x="VIP", y="Descriptor", orientation="h",
                        color="Block",
                        color_discrete_map={"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "FTIR": "#8e44ad", "NMR": "#16a085", "Thermal": "#c0392b", "SEC/GPC": "#008080", "Other": "#7f7f7f"},
                        text=top_vip["VIP"].apply(lambda v: f"{v:.2f}"),
                    )
                    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text=T[lang]["ml_vip_cutoff"])
                    fig_vip.update_layout(yaxis=dict(autorange="reversed", title="Descriptor"), xaxis=dict(title="VIP Score"), plot_bgcolor="white", height=520)
                    st_plotly(fig_vip)

                    st_df(vip_df)

                if CHEMO_PUBCHEM_AVAILABLE:
                    st.markdown("---")
                    st.write(f"#### {T[lang]['ml_pubchem_title']}")
                    st.caption(T[lang]["ml_pubchem_caption"])
                    target_bm = sparse_df[sparse_df["Is_Selected"]].copy() if (curr_model_type == "Sparse PLS-DA (sPLS-DA)" and not sparse_df.empty and "Is_Selected" in sparse_df.columns) else (vip_df.head(20).copy() if not vip_df.empty else pd.DataFrame())
                    if not target_bm.empty:
                        if st.button(T[lang]["ml_pubchem_btn"], key="btn_annot_biomarkers_pubchem"):
                            with st.spinner(T[lang]["ml_pubchem_spinner"]):
                                desc_col = "Descriptor" if "Descriptor" in target_bm.columns else target_bm.columns[0]
                                raw_formulas = target_bm[desc_col].dropna().astype(str).tolist()
                                annotated_bm = chemo_pubchem.annotate_formula_table(raw_formulas, max_top=15)
                                st.session_state["annotated_biomarkers"] = annotated_bm

                        annot_saved = st.session_state.get("annotated_biomarkers")
                        if annot_saved is not None and not annot_saved.empty:
                            st_df(
                                annot_saved,
                                column_config={
                                    "PubChem_URL": st.column_config.LinkColumn("PubChem", display_text="Open CID"),
                                    "ChEMBL_URL": st.column_config.LinkColumn("ChEMBL", display_text="Search"),
                                    "HMDB_URL": st.column_config.LinkColumn("HMDB", display_text="Metabolites"),
                                }
                            )
                            st.download_button(
                                label=T[lang]["ml_pubchem_dl_btn"],
                                data=annot_saved.to_csv(index=False).encode("utf-8"),
                                file_name="ChemoSuite_Biomarkers_PubChem.csv",
                                mime="text/csv",
                                key="dl_annot_biomarkers_csv",
                            )

                            # Интерактивная фотогалерея 2D структур ключевых маркеров
                            with st.expander(T[lang]["pubchem_inspect_expander"], expanded=True):
                                bm_cols = st.columns(3)
                                for b_idx, (_, b_row) in enumerate(annot_saved.head(6).iterrows()):
                                    with bm_cols[b_idx % 3]:
                                        with st.container(border=True):
                                            b_img = b_row.get("Image_URL")
                                            if b_img:
                                                st.image(b_img, caption=f"2D: {b_row['Compound_Name']}", width="stretch")
                                            else:
                                                st.info(f"Formula: {b_row['Formula']}")
                                            st.markdown(f"**{b_row['Formula']}**")
                                            st.caption(f"{b_row['Chemical_Class']}")
                                            if pd.notna(b_row.get("CID")) and b_row.get("CID"):
                                                st.link_button(f"{T[lang]['pubchem_link_btn']} #{int(b_row['CID'])}", b_row["PubChem_URL"])

            # Таб S-Plot (для OPLS-DA)
            if is_model_opls and tab_splot is not None:
                with tab_splot:
                    s_df = res.get("S_Plot_df", pd.DataFrame())
                    if not s_df.empty:
                        st.write(f"### {T[lang]['ml_splot_title']}")
                        fig_s = px.scatter(
                            s_df,
                            x="p1_cov",
                            y="p_corr",
                            color="Block",
                            size="VIP",
                            hover_name="Descriptor",
                            color_discrete_map={"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "FTIR": "#8e44ad", "NMR": "#16a085", "Thermal": "#c0392b", "SEC/GPC": "#008080", "Other": "#7f7f7f"},
                            labels={"p1_cov": "Covariance p[1] (Magnitude)", "p_corr": "Correlation p(corr)[1] (Reliability)"},
                        )
                        fig_s.add_hline(y=0.0, line_dash="dash", line_color="#bfbfbf")
                        fig_s.add_vline(x=0.0, line_dash="dash", line_color="#bfbfbf")
                        fig_s.update_layout(plot_bgcolor="white", height=520)
                        st_plotly(fig_s)

                        st_df(s_df)

            # Таб 3: Валидация (Confusion Matrix + Permutation Test)
            with tab_ml3:
                c_v1, c_v2 = st.columns(2)
                with c_v1:
                    st.write(f"**{T[lang]['ml_cm_title']}**")
                    cm_mat = res.get("Confusion_Matrix", np.zeros((2, 2)))
                    cls_labels = [str(c) for c in res.get("classes", unique_cls)]
                    fig_cm = px.imshow(
                        cm_mat,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=cls_labels,
                        y=cls_labels,
                        color_continuous_scale="Blues",
                    )
                    fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
                    st_plotly(fig_cm)

                    if not res.get("is_multiclass", False):
                        st.markdown(
                            f"""
                            * **{T[lang]['ml_sens_label']}** `{res.get('Sensitivity', 0.0):.1f}%`
                            * **{T[lang]['ml_spec_label']}** `{res.get('Specificity', 0.0):.1f}%`
                            * **{T[lang]['ml_bal_acc_label']}** `{res.get('Balanced_Accuracy', 0.0):.1f}%`
                            """
                        )
                    else:
                        st.markdown(
                            f"""
                            * **{T[lang]['ml_mode_multiclass'].format(n=len(cls_labels))}**
                            * **{T[lang]['ml_bal_acc_label']}** `{res.get('Balanced_Accuracy', 0.0):.1f}%`
                            * **{T[lang]['ml_acc_label']}** `{res.get('Accuracy', 0.0):.1f}%`
                            """
                        )

                with c_v2:
                    if perm_res is not None:
                        st.write(f"**{T[lang]['ml_perm_title']}**")
                        q2_o = perm_res["q2_orig"]
                        perm_q2_vals = perm_res["perm_q2"]
                        p_val = perm_res["p_val_q2"]

                        fig_perm = go.Figure()
                        fig_perm.add_trace(go.Histogram(
                            x=perm_q2_vals,
                            name="Permuted Q²",
                            marker_color="#8da9c4",
                            opacity=0.75,
                        ))
                        fig_perm.add_vline(x=q2_o, line_width=2.5, line_dash="dash", line_color="#d62728", annotation_text=f"Q²={q2_o:.1f}%")
                        fig_perm.update_layout(xaxis_title="Q² (%)", yaxis_title="Frequencies", height=360, plot_bgcolor="white", margin=dict(l=20, r=20, t=30, b=20))
                        st_plotly(fig_perm)

                        if p_val < 0.05:
                            st.success(f"✅ Модель статистически значима: эмпирический p-value = **{p_val:.4f}** (< 0.05)." if lang == "ru" else f"✅ Model is statistically significant: empirical p-value = **{p_val:.4f}** (< 0.05).")
                        else:
                            st.warning(f"⚠️ Риск оверфиттинга: эмпирический p-value = **{p_val:.4f}** (≥ 0.05)." if lang == "ru" else f"⚠️ Overfitting risk: empirical p-value = **{p_val:.4f}** (≥ 0.05).")
                    else:
                        st.info("Для OPLS-DA валидация выполнена по LOO / K-Fold CV. Пермутационный тест доступен в режиме PLS-DA." if lang == "ru" else "Validation performed via LOO / K-Fold CV. Permutation testing is available in PLS-DA mode.")

            # Таб 4: Вклад аналитических блоков
            with tab_ml4:
                block_contribs = res.get("Block_Contributions", {})
                if block_contribs:
                    st.write(f"**{T[lang]['ml_donut_title']}**")
                    b_palette = {"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "FTIR": "#8e44ad", "NMR": "#16a085", "Thermal": "#c0392b", "SEC/GPC": "#008080", "Other": "#7f7f7f"}
                    donut_colors = [b_palette.get(k, "#7f7f7f") for k in block_contribs.keys()]
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=list(block_contribs.keys()),
                        values=list(block_contribs.values()),
                        hole=0.45,
                        marker=dict(colors=donut_colors),
                    )])
                    fig_donut.update_layout(height=420)
                    st_plotly(fig_donut)

            # Таб 5: Кросс-модальный атлас (Cross-Omics / 2D Heterospectral Correlation)
            with tab_crosscorr:
                st.write(f"### {T[lang]['ml_crosscorr_title']}")
                st.caption(T[lang]['ml_crosscorr_subtitle'])

                if not CROSS_CORR_AVAILABLE or cross_correlation is None:
                    st.error("⚠️ Модуль `cross_correlation.py` не доступен." if lang == "ru" else "⚠️ `cross_correlation.py` is not available.")
                else:
                    corr_mode = st.radio(
                        T[lang]["ml_crosscorr_mode_select"],
                        [T[lang]["ml_crosscorr_mode_pairs"], T[lang]["ml_crosscorr_mode_global"]],
                        horizontal=True,
                        key="radio_crosscorr_mode",
                    )

                    col_cm1, col_cm2 = st.columns([2, 2])
                    with col_cm1:
                        corr_method = st.selectbox(
                            T[lang]["ml_crosscorr_method"],
                            ["pearson", "spearman"],
                            index=0,
                            key="select_crosscorr_method",
                        )
                    with col_cm2:
                        min_r_thresh = st.slider(
                            T[lang]["ml_crosscorr_threshold"],
                            min_value=0.3,
                            max_value=0.95,
                            value=0.6,
                            step=0.05,
                            key="slider_crosscorr_thresh",
                        )

                    # Расчет межблочных корреляций
                    with st.spinner("Расчет кросс-модальных корреляций..." if lang == "ru" else "Computing cross-omics correlations..."):
                        cc_results = cross_correlation.compute_multiblock_cross_correlation(
                            edited_df,
                            sample_id_col="Sample_ID",
                            target_col="Class_Target",
                            method=corr_method,
                        )

                    full_corr_df = cc_results["correlation_matrix"]
                    block_mapping = cc_results["block_mapping"]
                    top_cross_df = cc_results["top_cross_correlations"]

                    # Список доступных блоков
                    avail_blocks = sorted(list(set(b for b in block_mapping.values() if b != "Other")))
                    if len(avail_blocks) < 2:
                        avail_blocks = sorted(list(set(block_mapping.values())))

                    if corr_mode == T[lang]["ml_crosscorr_mode_pairs"]:
                        if len(avail_blocks) < 2:
                            st.warning(T[lang]["ml_crosscorr_no_data"])
                        else:
                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                block_a_sel = st.selectbox(
                                    T[lang]["ml_crosscorr_block_a"],
                                    avail_blocks,
                                    index=0,
                                    key="sel_block_a",
                                )
                            with col_b2:
                                default_b_idx = 1 if len(avail_blocks) > 1 else 0
                                block_b_sel = st.selectbox(
                                    T[lang]["ml_crosscorr_block_b"],
                                    avail_blocks,
                                    index=default_b_idx,
                                    key="sel_block_b",
                                )

                            feats_a = [f for f, b in block_mapping.items() if b == block_a_sel and f in edited_df.columns]
                            feats_b = [f for f, b in block_mapping.items() if b == block_b_sel and f in edited_df.columns]

                            if not feats_a or not feats_b:
                                st.warning(T[lang]["ml_crosscorr_no_data"])
                            else:
                                mat_a = edited_df[feats_a].values
                                mat_b = edited_df[feats_b].values
                                phi_2d = cross_correlation.compute_heterospectral_2d_correlation(mat_a, mat_b, scale=True)

                                fig_2d = px.imshow(
                                    phi_2d,
                                    x=feats_b,
                                    y=feats_a,
                                    color_continuous_scale="RdBu_r",
                                    zmin=-1.0,
                                    zmax=1.0,
                                    labels=dict(x=f"{block_b_sel} Features", y=f"{block_a_sel} Features", color="r"),
                                    title=f"2D Heterospectral Matrix: {block_a_sel} (Y) ⨂ {block_b_sel} (X)",
                                )
                                fig_2d.update_layout(
                                    height=540,
                                    margin=dict(l=60, r=40, t=50, b=60),
                                    xaxis=dict(tickangle=-45),
                                )
                                st_plotly(fig_2d)

                    else:
                        # Глобальная мультиблочная матрица
                        fig_global = px.imshow(
                            full_corr_df,
                            color_continuous_scale="RdBu_r",
                            zmin=-1.0,
                            zmax=1.0,
                            title="Global Multimodal Cross-Correlation Matrix",
                        )
                        fig_global.update_layout(height=650, margin=dict(l=60, r=40, t=50, b=60))
                        st_plotly(fig_global)

                    # Таблица топ межблочных корреляций
                    st.write(f"#### {T[lang]['ml_crosscorr_top_pairs_title']}")
                    if not top_cross_df.empty:
                        filtered_top = top_cross_df[top_cross_df["Abs_Correlation"] >= min_r_thresh].copy()
                        if filtered_top.empty:
                            st.info(f"Нет межблочных пар с |r| >= {min_r_thresh}. Попробуйте снизить порог." if lang == "ru" else f"No inter-block pairs with |r| >= {min_r_thresh}. Try lowering the threshold.")
                        else:
                            st.dataframe(filtered_top, use_container_width=True, height=260)

                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                csv_top = filtered_top.to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    label=T[lang]["ml_crosscorr_dl_top"],
                                    data=csv_top,
                                    file_name=f"ChemoSuite_Top_Cross_Correlations_{corr_method}.csv",
                                    mime="text/csv",
                                    key="btn_dl_top_cross",
                                )
                            with col_dl2:
                                nodes_df, edges_df = cross_correlation.build_correlation_network_edges(
                                    full_corr_df, block_mapping, threshold=min_r_thresh, cross_block_only=True
                                )
                                csv_edges = edges_df.to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    label=T[lang]["ml_crosscorr_dl_edges"],
                                    data=csv_edges,
                                    file_name=f"ChemoSuite_Network_Edges_{corr_method}_thresh{min_r_thresh}.csv",
                                    mime="text/csv",
                                    key="btn_dl_edges_cross",
                                )
                    else:
                        st.info("Межблочных взаимосвязей с |r| >= 0.5 не обнаружено." if lang == "ru" else "No inter-block correlations with |r| >= 0.5 found.")

            # Таб 6: Экспорт
            with tab_ml5:
                st.write("### 📥 Выгрузка аналитических отчетов" if lang == "ru" else "### 📥 Download Analytical Reports")

                # Комплексные аналитические паспорта
                st.markdown(
                    f'<div style="background-color: #f0f4f8; padding: 14px 18px; border-radius: 8px; border-left: 5px solid #0b2545; margin-bottom: 16px;">'
                    f'<h4 style="margin: 0 0 6px 0; color: #0b2545;">{T[lang]["ml_dl_html_passport_btn"]} & {T[lang]["ml_dl_passport_btn"]}</h4>'
                    f'<p style="margin: 0; color: #495057; font-size: 13px;">{T[lang]["ml_html_passport_desc"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_rep_lang, _ = st.columns([2, 2])
                with col_rep_lang:
                    report_lang_choice = st.radio(
                        T[lang]["ml_report_lang_label"],
                        ["English (EN)", "Русский (RU)"],
                        index=0 if lang == "en" else 1,
                        horizontal=True,
                        key="report_lang_selector"
                    )
                    rep_lang = "en" if "EN" in report_lang_choice else "ru"

                if REPORT_GEN_AVAILABLE and report_generator is not None:
                    try:
                        # 1. Генерация Excel паспорта
                        passport_bytes = report_generator.generate_excel_passport(
                            edited_df,
                            model_results=res,
                            metadata={
                                "project_name": st.session_state.get("fused_source", "ChemoSuite Session"),
                                "operator": "ChemoSuite Unified Analytical Engine",
                            },
                            lang=rep_lang,
                        )
                        rep_suffix = "EN" if rep_lang == "en" else "RU"

                        # 2. Генерация интерактивного HTML экологического паспорта
                        passport_html = report_generator.generate_ecological_passport_html(
                            edited_df,
                            model_results=res,
                            metadata={
                                "project_name": st.session_state.get("fused_source", "ChemoSuite Session"),
                                "operator": "ChemoSuite Unified Analytical Engine",
                            },
                            lang=rep_lang,
                        )

                        col_btn_html, col_btn_xlsx = st.columns(2)
                        with col_btn_html:
                            st.download_button(
                                label=T[lang]["ml_dl_html_passport_btn"],
                                data=passport_html.encode("utf-8"),
                                file_name=f"ChemoSuite_Ecological_Passport_{curr_model_type}_{rep_suffix}.html",
                                mime="text/html",
                                type="primary",
                                key="dl_passport_html_btn",
                            )
                        with col_btn_xlsx:
                            st.download_button(
                                label=T[lang]["ml_dl_passport_btn"],
                                data=passport_bytes,
                                file_name=f"ChemoSuite_Analytical_Passport_{curr_model_type}_{rep_suffix}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_passport_xlsx_btn",
                            )

                        # Интерактивный предпросмотр в Streamlit
                        with st.expander(T[lang]["ml_preview_passport_expander"], expanded=False):
                            import streamlit.components.v1 as components
                            components.html(passport_html, height=750, scrolling=True)

                    except Exception as err:
                        st.error(f"Ошибка формирования паспорта: {err}" if lang == "ru" else f"Passport generation error: {err}")
                else:
                    st.warning("Модуль `report_generator.py` или библиотека `openpyxl` недоступны." if lang == "ru" else "`report_generator.py` or `openpyxl` library is unavailable.")

                st.markdown("---")
                st.write("#### 📄 Табличные выгрузки отдельных компонентов" if lang == "ru" else "#### 📄 Individual Component Exports")
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        label="📥 Скачать объединенную матрицу признаков (CSV)" if lang == "ru" else "📥 Download Fused Feature Matrix (CSV)",
                        data=edited_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_Fused_Matrix.csv",
                        mime="text/csv",
                        key="dl_fused_csv_main_btn",
                    )
                with c_dl2:
                    st.download_button(
                        label=T[lang]["ml_dl_vip_btn"],
                        data=vip_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"ChemoSuite_{curr_model_type}_Biomarkers.csv",
                        mime="text/csv",
                        key="dl_pls_vip_csv_main_btn",
                    )
# ==============================================================================
# МОДУЛЬ 4: КОЛЕБАТЕЛЬНАЯ СПЕКТРОСКОПИЯ (ATR-FTIR & RAMAN)
# ==============================================================================
elif active_module == T[lang]["mod4_name"]:
    st.title(T[lang]["ftir_title"])
    st.caption(T[lang]["ftir_subtitle"])

    with st.popover("ℹ️ " + ("Руководство: Колебательная спектроскопия (FTIR) в анализе РОВ и лигнина" if lang == "ru" else "Guide: FTIR Spectroscopy in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы ИК-спектроскопии органического вещества

В отличие от электронной спектроскопии (УФ-Вид/EEM), отражающей только переходы сопряженных $\pi$-систем, **ИК-Фурье спектроскопия регистрирует фундаментальные колебания ковалентных связей** молекулярного каркаса без разрушения или ионизации пробы.

| Диапазон ($\text{см}^{-1}$) | Колебания связей | Молекулярное отнесение | Экологическое значение |
| :--- | :--- | :--- | :--- |
| **$3400–3200$** | $\nu(\text{O}-\text{H}), \nu(\text{N}-\text{H})$ | Фенолы, спирты, карбоновые кислоты, влага | Общий гидрофильный и водородно-связанный пул |
| **$2960–2850$** | $\nu_{as,s}(\text{C}-\text{H})$ | Алифатические цепи ($\text{CH}_3, \text{CH}_2$) | Степень алифатичности и разветвленности структуры |
| **$1725–1710$** | $\nu(\text{C}=\text{O})$ | Карбоксилы ($\text{COOH}$), неконъюгированные кетоны | Индикатор степени окисленности и гумификации |
| **$1650–1640$** | $\nu(\text{C}=\text{O})_{conj}$ | Хиноны, карбонилы, сопряженные с кольцом, Амид I | Редокс-активные центры природных гуминовых систем |
| **$1605–1600$** | $\nu(\text{C}=\text{C})_{arom}$ | Скелетные колебания ароматических колец | Степень конденсированности и ароматичности |
| **$1510$** | $\nu(\text{C}=\text{C})_{lignin}$ | **Гваяцильное кольцо лигнина (G-ring)** | **Критический специфический маркер шлам-лигнина БЦБК** |
| **$1270, 1220$** | $\nu(\text{C}-\text{O}), \nu(\text{C}-\text{C})$ | Гваяцильные эфирные связи лигнина | Дифференциация лигнина от природных полисахаридов |
| **$1035–1030$** | $\nu(\text{C}-\text{O})$ | Спирты, полисахариды, целлюлоза, метоксилы | Доля биогенного углеводного и терригенного пула |

---

### ⚙️ Алгоритмы предобработки и деконволюции:
* **AsLS (Asymmetric Least Squares):** удаление наклонного фона рассеяния и флуоресценции без смещения максимумов пиков.
* **Деконволюция $1500–1800\text{ см}^{-1}$:** разложение сложной суперпозиции полос на элементарные Гауссовы составляющие методом нелинейной оптимизации Левенберга — Марквардта с расчетом площадей и FWHM.
""")

    if not FTIR_CORE_AVAILABLE or ftir_core is None:
        st.error(T[lang]["ftir_err_missing"])
        st.stop()

    with st.sidebar:
        st.header(T[lang]["ftir_sidebar_hdr"])
        ftir_data_src = st.radio(
            T[lang]["ftir_src_label"],
            [T[lang]["ftir_src_upload"], T[lang]["ftir_src_synth"]],
            index=0,
            key="ftir_data_source_radio"
        )
        if st.session_state.get("_prev_ftir_data_src") != ftir_data_src:
            st.session_state["_prev_ftir_data_src"] = ftir_data_src
            st.session_state["ftir_samples"] = {}

        ftir_files = []
        if ftir_data_src == T[lang]["ftir_src_upload"]:
            ftir_files = st.file_uploader(
                T[lang]["ftir_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "dpt", "txt", "xy"],
                key="ftir_files_uploader"
            )

        st.markdown("---")
        st.subheader(T[lang]["ftir_preproc_hdr"])
        base_method_choice = st.selectbox(
            T[lang]["ftir_baseline_method"],
            [T[lang]["ftir_base_als"], T[lang]["ftir_base_rubber"], T[lang]["ftir_base_linear"], T[lang]["ftir_base_none"]],
            index=0,
            key="ftir_base_method_choice"
        )
        base_method = "als" if base_method_choice == T[lang]["ftir_base_als"] else ("rubberband" if base_method_choice == T[lang]["ftir_base_rubber"] else ("linear" if base_method_choice == T[lang]["ftir_base_linear"] else "none"))

        norm_method_choice = st.selectbox(
            T[lang]["ftir_norm_label"],
            [T[lang]["ftir_norm_snv"], T[lang]["ftir_norm_minmax"], T[lang]["ftir_norm_vec"], T[lang]["ftir_norm_none"]],
            index=0,
            key="ftir_norm_method_choice"
        )
        norm_method = "snv" if norm_method_choice == T[lang]["ftir_norm_snv"] else ("minmax" if norm_method_choice == T[lang]["ftir_norm_minmax"] else ("vector" if norm_method_choice == T[lang]["ftir_norm_vec"] else "none"))

    # Загрузка данных
    current_ftir_samples: Dict[str, Any] = {}

    if ftir_data_src == T[lang]["ftir_src_synth"]:
        synth_samples, synth_desc = ftir_core.generate_synthetic_ftir_dataset(n_samples=12, random_state=42)
        current_ftir_samples = synth_samples
        st.session_state["ftir_samples"] = synth_samples
    else:
        if ftir_files:
            for f in ftir_files:
                try:
                    s_obj = ftir_core.parse_ftir_file(f.getvalue(), sample_id=f.name)
                    current_ftir_samples[f.name] = s_obj
                except Exception as e_parse:
                    st.warning(f"Ошибка чтения {f.name}: {e_parse}")
            if current_ftir_samples:
                st.session_state["ftir_samples"] = current_ftir_samples
        else:
            current_ftir_samples = st.session_state.get("ftir_samples", {})

    if not current_ftir_samples:
        st.info(T[lang]["ftir_no_samples"])
        st.stop()

    tab_ftir1, tab_ftir2, tab_ftir3, tab_ftir4 = st.tabs([
        T[lang]["ftir_tab1_title"],
        T[lang]["ftir_tab2_title"],
        T[lang]["ftir_tab3_title"],
        T[lang]["ftir_tab4_title"],
    ])

    sample_names = list(current_ftir_samples.keys())

    # =========================================================================
    # ВКЛАДКА 1: СПЕКТРАЛЬНЫЙ ПРОФИЛЬ, ПИКИ И ФОН
    # =========================================================================
    with tab_ftir1:
        c_sel1, c_sel2, c_sel3 = st.columns([2.5, 2, 2.5])
        with c_sel1:
            active_name = st.selectbox(T[lang]["ftir_active_sample_label"], sample_names, index=0, key="ftir_active_spec_select")
        with c_sel2:
            overlay_all = st.checkbox(T[lang]["ftir_show_all"], value=False, key="ftir_overlay_all_chk")
        with c_sel3:
            deriv_choice = st.radio(
                T[lang]["ftir_deriv_label"],
                [T[lang]["ftir_deriv_0"], T[lang]["ftir_deriv_1"], T[lang]["ftir_deriv_2"]],
                horizontal=True,
                key="ftir_deriv_mode_radio"
            )
            deriv_order = 0 if deriv_choice == T[lang]["ftir_deriv_0"] else (1 if deriv_choice == T[lang]["ftir_deriv_1"] else 2)

        active_s = current_ftir_samples[active_name]
        # Вычитаем фон если выбран метод
        active_s, base_curve = ftir_core.subtract_baseline(active_s, method=base_method)
        active_abs = active_s.corrected_absorbance if base_method != "none" else active_s.absorbance

        # Настройки поиска пиков
        with st.expander(T[lang]["ftir_peaks_expander"], expanded=False):
            c_pk1, c_pk2, c_pk3 = st.columns([2, 2, 2.5])
            with c_pk1:
                pk_prom = st.slider(T[lang]["ftir_peaks_prominence"], min_value=0.005, max_value=0.20, value=0.02, step=0.005, key="ftir_peak_prom_slider")
            with c_pk2:
                pk_dist = st.slider(T[lang]["ftir_peaks_dist"], min_value=10.0, max_value=60.0, value=20.0, step=5.0, key="ftir_peak_dist_slider")
            with c_pk3:
                show_peak_markers = st.checkbox(T[lang]["ftir_peaks_show_markers"], value=True, key="ftir_show_peak_markers_chk")

            df_peaks = ftir_core.find_and_annotate_peaks(
                active_s.wavenumbers,
                active_abs,
                prominence=pk_prom,
                min_distance_cm1=pk_dist,
                lang=lang
            )

            if not df_peaks.empty:
                st.markdown(f"###### {T[lang]['ftir_peaks_tbl_title']}")
                st_df(
                    df_peaks.style.format({
                        "Wavenumber_cm1": "{:.1f}",
                        "Absorbance": "{:.4f}",
                        "Prominence": "{:.4f}",
                    })
                )
                st.download_button(
                    label=T[lang]["ftir_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_name}_FTIR_Peaks.csv",
                    mime="text/csv",
                    key="dl_ftir_peaks_csv",
                )
            else:
                st.info("Пики с заданным порогом выраженности не обнаружены. Попробуйте уменьшить порог Prominence." if lang == "ru" else "No peaks detected with the current prominence threshold. Try reducing the prominence value.")

        fig_ftir = go.Figure()

        if overlay_all:
            for s_name, s_item in current_ftir_samples.items():
                s_item, _ = ftir_core.subtract_baseline(s_item, method=base_method)
                y_disp = s_item.corrected_absorbance if base_method != "none" else s_item.absorbance
                if norm_method != "none":
                    y_disp = ftir_core.normalize_spectrum(y_disp, method=norm_method)
                if deriv_order > 0:
                    y_disp = ftir_core.smooth_savitzky_golay(y_disp, window_length=15, polyorder=2, deriv=deriv_order)

                is_lignin_sample = "lignin" in s_name.lower() or "шлам" in s_name.lower()
                c_line = "#D62728" if is_lignin_sample else "#0020C2"
                fig_ftir.add_trace(go.Scatter(
                    x=s_item.wavenumbers,
                    y=y_disp,
                    mode="lines",
                    name=s_name,
                    line=dict(width=1.5, color=c_line),
                    opacity=0.75
                ))
            fig_ftir.update_layout(
                title=f"Overlay of {len(current_ftir_samples)} FTIR Spectra" if lang == "en" else f"Наложение всех спектров FTIR ({len(current_ftir_samples)} проб)",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if deriv_order == 0 else f"d^{deriv_order}A"),
                height=540,
                plot_bgcolor="white"
            )
        else:
            y_active_disp = active_abs.copy()
            if norm_method != "none":
                y_active_disp = ftir_core.normalize_spectrum(y_active_disp, method=norm_method)
            if deriv_order > 0:
                y_active_disp = ftir_core.smooth_savitzky_golay(y_active_disp, window_length=15, polyorder=2, deriv=deriv_order)

            # Исходный зарегистрированный профиль
            if deriv_order == 0:
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=active_s.absorbance,
                    mode="lines",
                    name="Raw Absorbance" if lang == "en" else "Исходное поглощение (A)",
                    line=dict(color="#7F7F7F", width=1.2, dash="dot"),
                    opacity=0.65
                ))
                if base_method != "none":
                    fig_ftir.add_trace(go.Scatter(
                        x=active_s.wavenumbers,
                        y=base_curve,
                        mode="lines",
                        name=f"Baseline ({base_method.upper()})" if lang == "en" else f"Базовая линия ({base_method.upper()})",
                        line=dict(color="#FF7F0E", width=1.5, dash="dash")
                    ))
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=y_active_disp,
                    mode="lines",
                    name="Corrected Spectrum" if lang == "en" else "Скорректированный спектр",
                    line=dict(color="#0B2545", width=2.0)
                ))

                # Метки обнаруженных пиков
                if 'show_peak_markers' in locals() and show_peak_markers and not df_peaks.empty:
                    fig_ftir.add_trace(go.Scatter(
                        x=df_peaks["Wavenumber_cm1"],
                        y=df_peaks["Absorbance"],
                        mode="markers+text",
                        marker=dict(size=8, color="#D62728", symbol="diamond"),
                        text=[f"{w:.0f}" for w in df_peaks["Wavenumber_cm1"]],
                        textposition="top center",
                        hovertext=[
                            f"<b>{row['Wavenumber_cm1']:.1f} cm⁻¹</b><br>{row['Functional_Group']}<br>{row['Molecular_Assignment']}<br><i>{row['Compound_Class']}</i>"
                            for _, row in df_peaks.iterrows()
                        ],
                        hoverinfo="text",
                        name="Identified Bands" if lang == "en" else "Найденные полосы"
                    ))
            else:
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=y_active_disp,
                    mode="lines",
                    name=f"Derivative d^{deriv_order}A",
                    line=dict(color="#2CA02C", width=1.8)
                ))

            # Разметка ключевых функциональных диапазонов
            key_bands = [
                (3400, "OH/NH (3400)"),
                (2920, "Aliph CH2 (2920)"),
                (1720, "C=O COOH (1720)"),
                (1600, "C=C Arom (1600)"),
                (1510, "Lignin G-ring (1510)"),
                (1030, "C-O Poly (1030)")
            ]
            for wn_val, band_lbl in key_bands:
                fig_ftir.add_vline(x=wn_val, line_width=1, line_dash="dot", line_color="#CED4DA")
                fig_ftir.add_annotation(x=wn_val, y=float(np.max(y_active_disp)) * 0.95, text=band_lbl, showarrow=False, font=dict(size=9, color="#495057"))

            fig_ftir.update_layout(
                title=f"{active_name} — FTIR Profile" if lang == "en" else f"{active_name} — Спектральный профиль FTIR",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if deriv_order == 0 else f"d^{deriv_order}A"),
                height=520,
                plot_bgcolor="white"
            )

        st_plotly(fig_ftir)

        # Экспресс-дескрипторы активного образца
        ind_single = ftir_core.calculate_ftir_indices(active_s.wavenumbers, active_abs, sample_id=active_name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Lignin Index (1510/1600)", f"{ind_single['FTIR_Lignin_Idx']:.3f}")
        m2.metric("Aromaticity (1600/2920)", f"{ind_single['FTIR_Aromaticity_Idx']:.3f}")
        m3.metric("Oxidation (1720/1600)", f"{ind_single['FTIR_Oxidation_Idx']:.3f}")
        m4.metric("G/S Ratio (1270/1330)", f"{ind_single['FTIR_GS_Ratio']:.3f}")
        m5.metric("Polysaccharide Idx", f"{ind_single['FTIR_Polysaccharide_Idx']:.3f}")

    # =========================================================================
    # ВКЛАДКА 2: ДЕКОНВОЛЮЦИЯ ПОЛОС (1500–1800 см⁻¹)
    # =========================================================================
    with tab_ftir2:
        st.markdown("##### " + ("Математическое разделение карбонильных и ароматических полос (Levenberg-Marquardt)" if lang == "ru" else "Mathematical Deconvolution of Carbonyl and Aromatic Bands"))
        col_d1, col_d2, col_d3 = st.columns([2.5, 2, 2])
        with col_d1:
            deconv_target_name = st.selectbox(T[lang]["ftir_deconv_sample"], sample_names, index=0, key="ftir_deconv_sample_select")
        with col_d2:
            n_gaussians = st.slider(T[lang]["ftir_deconv_n_peaks"], min_value=3, max_value=5, value=5, step=1, key="ftir_n_gauss_slider")
        with col_d3:
            deconv_run_btn = st.button(T[lang]["ftir_deconv_btn"], type="primary", key="btn_run_ftir_deconv")

        deconv_sample = current_ftir_samples[deconv_target_name]
        deconv_sample, _ = ftir_core.subtract_baseline(deconv_sample, method=base_method)
        y_deconv_src = deconv_sample.corrected_absorbance if base_method != "none" else deconv_sample.absorbance

        try:
            deconv_res = ftir_core.deconvolve_carbonyl_region(
                deconv_sample.wavenumbers,
                y_deconv_src,
                wn_min=1500.0,
                wn_max=1800.0,
                n_peaks=n_gaussians
            )

            c_q1, c_q2 = st.columns(2)
            c_q1.metric(T[lang]["ftir_deconv_r2"], f"{deconv_res['r2']:.4f}")
            c_q2.metric(T[lang]["ftir_deconv_rmse"], f"{deconv_res['rmse']:.4f}")

            # График деконволюции
            fig_dec = go.Figure()

            # Экспериментальный спектр
            fig_dec.add_trace(go.Scatter(
                x=deconv_res["wavenumbers"],
                y=deconv_res["observed"],
                mode="markers+lines",
                marker=dict(size=3, color="#0B2545"),
                line=dict(color="#0B2545", width=1.2),
                name="Observed Profile" if lang == "en" else "Экспериментальный спектр"
            ))

            # Суммарная модельная кривая
            fig_dec.add_trace(go.Scatter(
                x=deconv_res["wavenumbers"],
                y=deconv_res["fitted"],
                mode="lines",
                line=dict(color="#D62728", width=2.5, dash="dash"),
                name="Fitted Envelope" if lang == "en" else "Суммарный фиттинг"
            ))

            # Индивидуальные полосы
            band_palette = ["#17BECF", "#9467BD", "#E377C2", "#FF7F0E", "#2CA02C"]
            peaks_df = deconv_res["peaks_df"]
            for idx, c_curve in enumerate(deconv_res["individual_curves"]):
                p_name = peaks_df.iloc[idx]["Assignment"] if idx < len(peaks_df) else f"Peak {idx+1}"
                p_col = band_palette[idx % len(band_palette)]
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["wavenumbers"],
                    y=c_curve,
                    mode="lines",
                    fill="tozeroy",
                    name=p_name,
                    line=dict(color=p_col, width=1.5),
                    opacity=0.65
                ))

            fig_dec.update_layout(
                title=f"Band Deconvolution (1500–1800 cm⁻¹) — {deconv_target_name}",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if lang == "en" else "Оптическая плотность A"),
                height=520,
                plot_bgcolor="white"
            )
            st_plotly(fig_dec)

            # Таблица параметров подогнанных полос
            st.markdown(f"###### {T[lang]['ftir_deconv_tbl_title']}")
            st_df(
                peaks_df.style.format({
                    "Center_cm1": "{:.1f}", "Amplitude": "{:.4f}",
                    "Sigma_cm1": "{:.2f}", "FWHM_cm1": "{:.2f}",
                    "Area": "{:.4f}", "Relative_Area_pct": "{:.2f}%"
                })
            )

        except Exception as e_dec:
            st.warning(f"Ошибка при деконволюции: {e_dec}")

    # =========================================================================
    # ВКЛАДКА 3: ИДЕНТИФИКАЦИЯ ПО БИБЛИОТЕКЕ ЭТАЛОНОВ (HQI)
    # =========================================================================
    with tab_ftir3:
        st.markdown("##### " + T[lang]["ftir_lib_hdr"])
        c_lb1, c_lb2, c_lb3 = st.columns([2.5, 1.5, 2])
        with c_lb1:
            lib_target_name = st.selectbox(T[lang]["ftir_lib_sample_select"], sample_names, index=0, key="ftir_lib_sample_select")
        with c_lb2:
            lib_top_n = st.slider(T[lang]["ftir_lib_top_n"], min_value=3, max_value=8, value=5, step=1, key="ftir_lib_top_n_slider")
        with c_lb3:
            st.markdown("<br>", unsafe_allow_html=True)
            lib_run_btn = st.button(T[lang]["ftir_lib_btn"], type="primary", key="btn_run_ftir_lib_match")

        target_s = current_ftir_samples[lib_target_name]
        target_s_corr, _ = ftir_core.subtract_baseline(target_s, method=base_method)
        y_target_abs = target_s_corr.corrected_absorbance if base_method != "none" else target_s.absorbance

        match_df = ftir_core.match_spectrum_against_library(target_s.wavenumbers, y_target_abs, top_n=lib_top_n, lang=lang)

        if not match_df.empty:
            top_match = match_df.iloc[0]
            top_std_id = top_match["Standard_ID"]
            top_hqi = top_match["HQI_pct"]
            top_name = top_match["Standard_Name"]

            # Статусная плашка
            if top_hqi >= 85.0:
                st.success(
                    f"🏆 **Топ-1 кандидат:** {top_name} (HQI = **{top_hqi:.1f}%**, Cosine = **{top_match['Cosine_Similarity']:.3f}**, Pearson r = **{top_match['Pearson_r']:.3f}**)"
                    if lang == "ru" else
                    f"🏆 **Top-1 Match:** {top_name} (HQI = **{top_hqi:.1f}%**, Cosine = **{top_match['Cosine_Similarity']:.3f}**, Pearson r = **{top_match['Pearson_r']:.3f}**)"
                )
            else:
                st.warning(
                    f"⚠️ **Наиболее близкий эталон:** {top_name} (HQI = **{top_hqi:.1f}%**). Возможно присутствие сложной смешанной матрицы."
                    if lang == "ru" else
                    f"⚠️ **Closest Reference Standard:** {top_name} (HQI = **{top_hqi:.1f}%**). Complex matrix mixture suspected."
                )

            # Оверлей-график: образец против топ-1 стандарта библиотеки
            ref_wn, ref_abs, ref_meta = ftir_core.get_reference_spectrum(top_std_id, wn_grid=target_s.wavenumbers)

            s_disp_norm = ftir_core.normalize_spectrum(y_target_abs, method="minmax")
            r_disp_norm = ftir_core.normalize_spectrum(ref_abs, method="minmax")

            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(
                x=target_s.wavenumbers,
                y=s_disp_norm,
                mode="lines",
                name=f"{lib_target_name} ({T[lang]['ftir_lib_sample_trace']})",
                line=dict(color="#0B2545", width=2.2)
            ))
            fig_overlay.add_trace(go.Scatter(
                x=ref_wn,
                y=r_disp_norm,
                mode="lines",
                name=f"{top_name} ({T[lang]['ftir_lib_ref_trace']})",
                line=dict(color="#FF7F0E", width=2.0, dash="dash")
            ))

            fig_overlay.update_layout(
                title=f"{T[lang]['ftir_lib_overlay_title']} — {lib_target_name} vs {top_name}",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Normalized Absorbance (Min-Max)" if lang == "en" else "Нормированное поглощение (Min-Max)"),
                height=480,
                plot_bgcolor="white"
            )
            st_plotly(fig_overlay)

            # Таблица совпадений
            st.markdown(f"###### {T[lang]['ftir_lib_tbl_title']}")
            disp_match_df = match_df[["Rank", "Standard_Name", "Compound_Class", "HQI_pct", "Cosine_Similarity", "Pearson_r", "Key_Markers", "Description"]]
            st_df(
                disp_match_df.style.format({
                    "HQI_pct": "{:.2f}%",
                    "Cosine_Similarity": "{:.4f}",
                    "Pearson_r": "{:.4f}"
                })
            )

            st.download_button(
                label="📥 " + ("Скачать результаты идентификации (CSV)" if lang == "ru" else "Download Library Hits (CSV)"),
                data=match_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{lib_target_name}_FTIR_Library_Hits.csv",
                mime="text/csv",
                key="dl_ftir_lib_hits_csv",
            )

    # =========================================================================
    # ВКЛАДКА 4: ХЕМОМЕТРИЧЕСКИЕ ИНДЕКСЫ И ЭКСПОРТ
    # =========================================================================
    with tab_ftir4:
        st.markdown("##### " + ("Сводная матрица дескрипторов колебательной спектроскопии" if lang == "ru" else "Summary Matrix of Vibrational Spectroscopy Descriptors"))

        df_ftir_desc = ftir_core.batch_process_ftir_samples(
            list(current_ftir_samples.values()),
            baseline_method=base_method,
            norm_method="none"
        )

        st_df(
            df_ftir_desc.style.format({
                "FTIR_Aromaticity_Idx": "{:.3f}", "FTIR_Oxidation_Idx": "{:.3f}",
                "FTIR_Branching_Idx": "{:.3f}", "FTIR_Lignin_Idx": "{:.3f}",
                "FTIR_GS_Ratio": "{:.3f}", "FTIR_Polysaccharide_Idx": "{:.3f}",
                "FTIR_Phenolic_Idx": "{:.3f}",
                "FTIR_Area_OH": "{:.1f}", "FTIR_Area_Aliphatic": "{:.1f}",
                "FTIR_Area_Carbonyl_Aromatic": "{:.1f}", "FTIR_Area_Polysaccharide": "{:.1f}",
                "FTIR_Area_Ratio_Carb_Aliph": "{:.3f}",
            }, na_rep="—")
        )

        # Сравнительная диаграмма маркеров лигнина и ароматичности
        fig_bar = px.bar(
            df_ftir_desc,
            x="Sample_ID",
            y=["FTIR_Lignin_Idx", "FTIR_Aromaticity_Idx", "FTIR_Oxidation_Idx"],
            barmode="group",
            title="Comparison of Key Structural Indices Across Samples" if lang == "en" else "Сравнение ключевых структурных индексов по пробам"
        )
        fig_bar.update_layout(height=420, xaxis_tickangle=-25)
        st_plotly(fig_bar)

        c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
        with c_d1:
            st.download_button(
                label=T[lang]["ftir_dl_csv"],
                data=df_ftir_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_FTIR_Descriptors.csv",
                mime="text/csv",
                key="dl_ftir_descriptors_csv",
            )
        with c_d2:
            excel_ftir_buf = io.BytesIO()
            with pd.ExcelWriter(excel_ftir_buf, engine="openpyxl") as writer:
                df_ftir_desc.to_excel(writer, index=False, sheet_name="FTIR_Descriptors")
            st.download_button(
                label=T[lang]["ftir_dl_excel"],
                data=excel_ftir_buf.getvalue(),
                file_name="ChemoSuite_FTIR_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_ftir_descriptors_excel",
            )
        with c_d3:
            if st.button(T[lang]["ftir_sync_ml_btn"], type="primary", key="btn_sync_ftir_ml_descriptors"):
                st.session_state["ftir_ml_descriptors"] = df_ftir_desc
                st.toast(T[lang]["ftir_sync_success"].format(n=len(df_ftir_desc)), icon="✅")
                st.rerun()


# ==============================================================================
# МОДУЛЬ 5: ЯМР-СПЕКТРОСКОПИЯ (13C / 1H NMR STUDIO)
# ==============================================================================
elif active_module == T[lang]["mod5_name"]:
    if not NMR_CORE_AVAILABLE or nmr_core is None:
        st.error(T[lang]["nmr_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #1A365D 0%, #2B6CB0 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["nmr_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #E2E8F0; font-size: 13px;">{T[lang]["nmr_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown(f"### {T[lang]['nmr_sidebar_hdr']}")
        nmr_data_src = st.radio(
            T[lang]["nmr_src_label"],
            [T[lang]["nmr_src_upload"], T[lang]["nmr_src_synth"]],
            index=0,
            key="nmr_data_source_radio"
        )

        if st.session_state.get("_prev_nmr_data_src") != nmr_data_src:
            st.session_state["_prev_nmr_data_src"] = nmr_data_src
            st.session_state["nmr_samples"] = {}

        if nmr_data_src == T[lang]["nmr_src_upload"]:
            uploaded_nmr_files = st.file_uploader(
                T[lang]["nmr_uploader_label"],
                type=["csv", "txt", "tsv", "xy", "jdx", "dx"],
                accept_multiple_files=True,
                key="nmr_uploaded_files"
            )
            if uploaded_nmr_files:
                for uf in uploaded_nmr_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0]
                    try:
                        s_obj = nmr_core.parse_nmr_file(b_content, sample_id=s_id, nucleus="13C")
                        st.session_state["nmr_samples"][s_id] = s_obj
                    except Exception as e_up:
                        st.warning(f"Ошибка чтения {uf.name}: {e_up}")
        else:
            if not st.session_state["nmr_samples"]:
                demo_nmr_dir = os.path.join("demo_data", "nmr")
                if os.path.exists(demo_nmr_dir) and any(f.endswith(".csv") for f in os.listdir(demo_nmr_dir)):
                    for fn in os.listdir(demo_nmr_dir):
                        if fn.endswith(".csv") and fn != "nmr_descriptors_matrix.csv":
                            fpath = os.path.join(demo_nmr_dir, fn)
                            with open(fpath, "rb") as f:
                                b_content = f.read()
                            s_obj = nmr_core.parse_nmr_file(b_content, sample_id=fn.replace(".csv", ""), nucleus="13C")
                            st.session_state["nmr_samples"][s_obj.sample_id] = s_obj
                else:
                    s_synth, _ = nmr_core.generate_synthetic_nmr_dataset(n_samples_per_class=6, save_dir=demo_nmr_dir)
                    st.session_state["nmr_samples"] = s_synth

        st.markdown(f"#### {T[lang]['nmr_preproc_hdr']}")
        base_method_choice = st.selectbox(
            T[lang]["nmr_baseline_method"],
            [T[lang]["nmr_base_asls"], T[lang]["nmr_base_linear"], T[lang]["nmr_base_poly"], T[lang]["nmr_base_none"]],
            index=0,
            key="nmr_baseline_select"
        )
        base_method = (
            "asls" if base_method_choice == T[lang]["nmr_base_asls"] else
            ("linear" if base_method_choice == T[lang]["nmr_base_linear"] else
            ("polynomial" if base_method_choice == T[lang]["nmr_base_poly"] else "none"))
        )

        norm_method_choice = st.selectbox(
            T[lang]["nmr_norm_label"],
            [T[lang]["nmr_norm_area"], T[lang]["nmr_norm_max"], T[lang]["nmr_norm_minmax"], T[lang]["nmr_norm_none"]],
            index=0,
            key="nmr_norm_select"
        )
        norm_method = (
            "area" if norm_method_choice == T[lang]["nmr_norm_area"] else
            ("max" if norm_method_choice == T[lang]["nmr_norm_max"] else
            ("minmax" if norm_method_choice == T[lang]["nmr_norm_minmax"] else "none"))
        )

        apply_sg_smoothing = st.checkbox(T[lang]["nmr_smooth_label"], value=True, key="nmr_chk_sg_smooth")

        recal_choice = st.selectbox(
            T[lang]["nmr_recal_label"],
            [T[lang]["nmr_recal_none"], T[lang]["nmr_recal_dmso"], T[lang]["nmr_recal_tms"]],
            index=0,
            key="nmr_recal_select"
        )
        recal_mode = "none"
        if "dmso" in recal_choice.lower():
            recal_mode = "dmso"
        elif "tms" in recal_choice.lower() or "dss" in recal_choice.lower():
            recal_mode = "tms"

    current_nmr_samples: Dict[str, nmr_core.NMRSample] = st.session_state["nmr_samples"]
    if not current_nmr_samples:
        st.info(T[lang]["nmr_no_samples"])
        st.stop()

    tab_nmr1, tab_nmr2, tab_nmr3, tab_nmr4 = st.tabs([
        T[lang]["nmr_tab1_title"],
        T[lang]["nmr_tab2_title"],
        T[lang]["nmr_tab3_title"],
        T[lang]["nmr_tab4_title"],
    ])

    sample_names = list(current_nmr_samples.keys())

    # =========================================================================
    # ВКЛАДКА 1: 📈 СПЕКТРАЛЬНЫЙ ПРОФИЛЬ И ИНТЕГРАЛЬНЫЕ ЗОНЫ
    # =========================================================================
    with tab_nmr1:
        col_ctl1, col_ctl2, col_ctl3 = st.columns([2.5, 1.5, 2.0])
        with col_ctl1:
            active_name = st.selectbox(T[lang]["nmr_active_sample_label"], sample_names, index=0, key="nmr_active_sample_select")
        with col_ctl2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_overlay = st.checkbox(T[lang]["nmr_show_all"], value=False, key="nmr_overlay_checkbox")
        with col_ctl3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_regions = st.checkbox(T[lang]["nmr_show_regions"], value=True, key="nmr_regions_checkbox")

        active_s = current_nmr_samples[active_name]
        active_shifts = active_s.chemical_shifts.copy()
        if recal_mode != "none":
            active_shifts, recal_delta = nmr_core.auto_recalibrate_nmr_shifts(
                active_shifts, active_s.intensity, mode=recal_mode
            )
            if abs(recal_delta) > 1e-4:
                st.info(
                    f"🎯 " + (f"Автокалибровка по {recal_mode.upper()}: сдвиг шкалы Δ = {recal_delta:+.2f} ppm" if lang == "ru" else f"Autorecalibration via {recal_mode.upper()}: scale shift Δ = {recal_delta:+.2f} ppm")
                )

        active_s_c, base_curve = nmr_core.subtract_baseline(active_s, method=base_method)
        active_y = active_s_c.corrected_intensity if base_method != "none" else active_s.intensity

        if apply_sg_smoothing:
            active_y = nmr_core.smooth_nmr_spectrum(active_y, window_length=9, polyorder=2)
        active_y_disp = nmr_core.normalize_nmr_spectrum(active_y, method=norm_method, chemical_shifts=active_shifts)

        # Выдвижная панель: Автопоиск пиков
        with st.expander(T[lang]["nmr_peaks_expander"], expanded=False):
            c_pk1, c_pk2, c_pk3 = st.columns([1.5, 1.5, 2])
            with c_pk1:
                pk_prom = st.slider(T[lang]["nmr_peaks_prominence"], min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="nmr_peak_prom_slider")
            with c_pk2:
                pk_dist = st.slider(T[lang]["nmr_peaks_dist"], min_value=2.0, max_value=15.0, value=4.0, step=0.5, key="nmr_peak_dist_slider")
            with c_pk3:
                st.markdown("<br>", unsafe_allow_html=True)
                show_peak_markers = st.checkbox(T[lang]["nmr_peaks_show_markers"], value=True, key="nmr_show_peak_markers_chk")

            df_peaks = nmr_core.find_and_annotate_nmr_peaks(
                active_shifts, active_y_disp,
                prominence=pk_prom, distance_ppm=pk_dist, lang=lang
            )
            if not df_peaks.empty:
                st.markdown(f"###### {T[lang]['nmr_peaks_tbl_title']}")
                st.dataframe(df_peaks, width="stretch", hide_index=True)
                st.download_button(
                    label=T[lang]["nmr_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_name}_13C_NMR_Peaks.csv",
                    mime="text/csv",
                    key="dl_nmr_peaks_csv",
                )

        # Построение спектрального графика Plotly
        fig_nmr = go.Figure()

        if show_overlay:
            for s_name, s_item in current_nmr_samples.items():
                s_shifts = s_item.chemical_shifts.copy()
                if recal_mode != "none":
                    s_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(s_shifts, s_item.intensity, mode=recal_mode)
                s_item_c, _ = nmr_core.subtract_baseline(s_item, method=base_method)
                y_disp = s_item_c.corrected_intensity if base_method != "none" else s_item.intensity
                if apply_sg_smoothing:
                    y_disp = nmr_core.smooth_nmr_spectrum(y_disp, window_length=9, polyorder=2)
                y_disp = nmr_core.normalize_nmr_spectrum(y_disp, method=norm_method, chemical_shifts=s_shifts)

                fig_nmr.add_trace(go.Scatter(
                    x=s_shifts,
                    y=y_disp,
                    mode="lines",
                    name=s_name,
                    opacity=0.75,
                    line=dict(width=1.5),
                ))
            fig_nmr.update_layout(
                title=f"Overlay of {len(current_nmr_samples)} 13C NMR Spectra" if lang == "en" else f"Наложение всех спектров 13C ЯМР ({len(current_nmr_samples)} проб)",
            )
        else:
            fig_nmr.add_trace(go.Scatter(
                x=active_shifts,
                y=active_y_disp,
                mode="lines",
                name=active_name,
                line=dict(color="#1A365D", width=2.2),
            ))

            # Отображение базовой линии
            if base_method != "none" and len(base_curve) > 0:
                base_disp = nmr_core.normalize_nmr_spectrum(base_curve, method=norm_method, chemical_shifts=active_shifts)
                fig_nmr.add_trace(go.Scatter(
                    x=active_shifts,
                    y=base_disp,
                    mode="lines",
                    name="AsLS Baseline" if lang == "en" else "Базовая линия",
                    line=dict(color="#E53E3E", width=1.2, dash="dash"),
                ))

            # Маркеры пиков
            if not df_peaks.empty and show_peak_markers:
                fig_nmr.add_trace(go.Scatter(
                    x=df_peaks["Chemical_Shift_ppm"],
                    y=df_peaks["Intensity"] if norm_method == "none" else df_peaks["Relative_Height"] * float(np.max(active_y_disp)),
                    mode="markers+text",
                    name="Resonances" if lang == "en" else "Резонансы",
                    text=[f"{p:.1f}" for p in df_peaks["Chemical_Shift_ppm"]],
                    textposition="top center",
                    textfont=dict(size=9, color="#742A2A"),
                    marker=dict(symbol="triangle-down", size=9, color="#E53E3E"),
                ))

            fig_nmr.update_layout(
                title=f"{active_name} — 13C CP/MAS NMR Profile" if lang == "en" else f"{active_name} — Спектральный профиль 13C ЯМР",
            )

        # Канонические интегральные зоны IHSS (Цветная полупрозрачная подсветка)
        if show_regions:
            zones = [
                (0.0, 48.0, "Aliphatic C", "#BEE3F8", "rgba(190, 227, 248, 0.22)"),
                (48.0, 108.0, "O/N-Alkyl C", "#C6F6D5", "rgba(198, 246, 213, 0.22)"),
                (108.0, 165.0, "Aromatic C", "#FEEBC8", "rgba(254, 235, 200, 0.25)"),
                (165.0, 190.0, "Carboxyl C", "#FED7D7", "rgba(254, 215, 215, 0.22)"),
                (190.0, 220.0, "Carbonyl C", "#E9D8FD", "rgba(233, 216, 253, 0.22)"),
            ]
            for z_min, z_max, z_name, z_col, z_fill in zones:
                fig_nmr.add_vrect(
                    x0=z_min, x1=z_max,
                    fillcolor=z_fill,
                    layer="below",
                    line_width=1,
                    line_dash="dot",
                    line_color=z_col,
                    annotation_text=z_name,
                    annotation_position="top left",
                    annotation_font=dict(size=9, color="#4A5568")
                )

        # СТАНДАРТ ЯМР: ОСЬ X ИНВЕРТИРОВАНА (ОТ 220 PPM К -5 PPM)
        fig_nmr.update_xaxes(
            title_text="Chemical Shift δ (ppm) [13C]",
            autorange="reversed",
            range=[225.0, -10.0],
            gridcolor="#EDF2F7",
        )
        fig_nmr.update_yaxes(
            title_text="Intensity (Normalized)" if norm_method != "none" else "Raw Intensity",
            gridcolor="#EDF2F7",
        )
        fig_nmr.update_layout(
            height=480,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st_plotly(fig_nmr)

        # Расчет и вывод фундаментальных метрик активного образца
        ind_single = nmr_core.calculate_13c_nmr_descriptors(active_shifts, active_y, sample_id=active_name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Aromaticity fa (%)", f"{ind_single['NMR_fa_aromaticity']:.1f}%")
        m2.metric("Hydrophobicity (HB/HI)", f"{ind_single['NMR_hb_hi_index']:.2f}")
        m3.metric("Alkyl / O-Alkyl", f"{ind_single['NMR_alkyl_o_alkyl']:.2f}")
        m4.metric("Lignin Signature (%)", f"{ind_single['NMR_lignin_signature']:.1f}%")
        m5.metric("Estimated Mw (Da)", f"{int(ind_single['NMR_estimated_mw'])} Da")

    # =========================================================================
    # ВКЛАДКА 2: 📊 РАСПРЕДЕЛЕНИЕ УГЛЕРОДНЫХ ПУЛОВ
    # =========================================================================
    with tab_nmr2:
        st.markdown(
            "##### " + ("Carbon Skeleton Distribution and Structural Fingerprint" if lang == "en" else "Распределение углеродного скелета и структурный отпечаток")
        )

        ind_current = nmr_core.calculate_13c_nmr_descriptors(active_shifts, active_y, sample_id=active_name)

        col_g1, col_g2 = st.columns([1.2, 1.0])
        with col_g1:
            # Bar chart 5 канонических областей
            df_fractions = pd.DataFrame({
                "Region": ["Aliphatic C (0-48)", "O/N-Alkyl C (48-108)", "Aromatic C (108-165)", "Carboxyl C (165-190)", "Carbonyl C (190-220)"],
                "Fraction_Pct": [
                    ind_current["NMR_C_aliph_pct"],
                    ind_current["NMR_C_O_alkyl_pct"],
                    ind_current["NMR_C_aromatic_pct"],
                    ind_current["NMR_C_carboxyl_pct"],
                    ind_current["NMR_C_carbonyl_pct"],
                ],
                "Color": ["#3182CE", "#38A169", "#DD6B20", "#E53E3E", "#805AD5"]
            })
            fig_frac = px.bar(
                df_fractions,
                x="Region",
                y="Fraction_Pct",
                color="Region",
                color_discrete_sequence=["#3182CE", "#38A169", "#DD6B20", "#E53E3E", "#805AD5"],
                text="Fraction_Pct",
                title=f"{active_name} — Carbon Distribution (% Total C)",
            )
            fig_frac.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_frac.update_layout(height=380, showlegend=False, yaxis=dict(title="% of Total Carbon", range=[0, max(60.0, df_fractions["Fraction_Pct"].max() + 8)]))
            st_plotly(fig_frac)

        with col_g2:
            # Donut chart
            fig_donut = px.pie(
                df_fractions,
                names="Region",
                values="Fraction_Pct",
                hole=0.45,
                color="Region",
                color_discrete_sequence=["#3182CE", "#38A169", "#DD6B20", "#E53E3E", "#805AD5"],
                title=f"{active_name} — Carbon Fractions Donut",
            )
            fig_donut.update_traces(textinfo="label+percent")
            fig_donut.update_layout(height=380, showlegend=False)
            st_plotly(fig_donut)

        # Радарная диаграмма (Radar Fingerprint) активного образца vs Эталоны
        st.markdown("###### " + ("Structural Radar Fingerprint vs Analytical Standards" if lang == "en" else "Радарный структурный отпечаток в сравнении со стандартами"))
        categories = ["Aromaticity fa", "Hydrophobicity (HB/HI*10)", "Alkyl/O-Alkyl (*10)", "Phenolic C (%)", "Carboxyl C (%)", "Aliphatic C (%)"]

        # Значения активного образца с защитой от NaN / inf
        val_active = [
            float(np.nan_to_num(ind_current.get("NMR_fa_aromaticity", 0.0), nan=0.0)),
            float(np.nan_to_num(min(80.0, ind_current.get("NMR_hb_hi_index", 1.0) * 10.0), nan=10.0)),
            float(np.nan_to_num(min(80.0, ind_current.get("NMR_alkyl_o_alkyl", 1.0) * 10.0), nan=10.0)),
            float(np.nan_to_num(ind_current.get("NMR_C_phenolic_pct", 0.0), nan=0.0)),
            float(np.nan_to_num(ind_current.get("NMR_C_carboxyl_pct", 0.0), nan=0.0)),
            float(np.nan_to_num(ind_current.get("NMR_C_aliph_pct", 0.0), nan=0.0)),
        ]
        val_active.append(val_active[0])

        # Эталон Байкал DOM
        val_ref_dom = [22.0, 12.0, 8.0, 3.5, 14.0, 26.0, 22.0]
        # Эталон Шлам-лигнин
        val_ref_lignin = [52.0, 28.0, 14.0, 16.0, 14.0, 16.0, 52.0]

        cat_closed = categories + [categories[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=val_active, theta=cat_closed, fill="toself", name=active_name, line=dict(color="#1A365D", width=2)))
        fig_radar.add_trace(go.Scatterpolar(r=val_ref_dom, theta=cat_closed, name="Ref: Pristine DOM" if lang == "en" else "Эталон: Природное РОВ", line=dict(color="#38A169", width=1.5, dash="dash")))
        fig_radar.add_trace(go.Scatterpolar(r=val_ref_lignin, theta=cat_closed, name="Ref: Sludge Lignin" if lang == "en" else "Эталон: Шлам-лигнин БЦБК", line=dict(color="#DD6B20", width=1.5, dash="dot")))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 65])),
            height=420,
            showlegend=True,
            template="plotly_white",
        )
        st_plotly(fig_radar)

    # =========================================================================
    # ВКЛАДКА 3: 🔍 ДЕКОНВОЛЮЦИЯ И СПЕКТРАЛЬНАЯ БИБЛИОТЕКА
    # =========================================================================
    with tab_nmr3:
        sec_tab1, sec_tab2 = st.tabs([
            "✂️ " + ("Aromatic Zone Deconvolution (108–165 ppm)" if lang == "en" else "Деконволюция ароматической зоны (108–165 ppm)"),
            "🏷️ " + ("IHSS Spectral Library Matching" if lang == "en" else "Идентификация по библиотеке эталонов IHSS")
        ])

        with sec_tab1:
            st.markdown(
                "###### " + (
                    "Resolution Enhancement: Gaussian Component Fitting of Guaiacyl, Syringyl, and Alkyl-Aromatic Resonances"
                    if lang == "en" else
                    "Разрешение перекрывающихся резонансов: подгонка гауссовых компонент гваяцила, сирингила и алкил-ароматики"
                )
            )
            col_dec1, col_dec2, col_dec3 = st.columns([2.5, 1.5, 2.0])
            with col_dec1:
                deconv_target_name = st.selectbox(T[lang]["nmr_deconv_sample"], sample_names, index=0, key="nmr_deconv_sample_select")
            with col_dec2:
                n_gaussians = st.slider(T[lang]["nmr_deconv_n_peaks"], min_value=3, max_value=4, value=4, step=1, key="nmr_n_gauss_slider")
            with col_dec3:
                st.markdown("<br>", unsafe_allow_html=True)
                deconv_run_btn = st.button(T[lang]["nmr_deconv_btn"], type="primary", key="btn_run_nmr_deconv")

            target_dec_s = current_nmr_samples[deconv_target_name]
            target_dec_shifts = target_dec_s.chemical_shifts.copy()
            if recal_mode != "none":
                target_dec_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(target_dec_shifts, target_dec_s.intensity, mode=recal_mode)
            target_dec_s_c, _ = nmr_core.subtract_baseline(target_dec_s, method=base_method)
            y_dec_target = target_dec_s_c.corrected_intensity if base_method != "none" else target_dec_s.intensity

            deconv_res = nmr_core.deconvolve_aromatic_region(
                target_dec_shifts, y_dec_target, n_peaks=n_gaussians
            )

            if deconv_res.get("success"):
                c_q1, c_q2 = st.columns(2)
                c_q1.metric(T[lang]["nmr_deconv_r2"], f"{deconv_res['r2']:.4f}")
                c_q2.metric(T[lang]["nmr_deconv_rmse"], f"{deconv_res['rmse']:.4f}")

                fig_dec = go.Figure()
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["x"], y=deconv_res["y"],
                    mode="lines", name="Observed" if lang == "en" else "Эксперимент",
                    line=dict(color="#1A365D", width=2.5)
                ))
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["x"], y=deconv_res["y_fit"],
                    mode="lines", name="Cumulative Fit" if lang == "en" else "Суммарный фит",
                    line=dict(color="#E53E3E", width=1.8, dash="dash")
                ))

                c_colors = ["#3182CE", "#38A169", "#DD6B20", "#805AD5"]
                for idx_c, comp in enumerate(deconv_res["components"]):
                    fig_dec.add_trace(go.Scatter(
                        x=deconv_res["x"], y=comp["curve"],
                        mode="lines",
                        name=f"{comp['label']} ({comp['area_pct']}%)",
                        line=dict(color=c_colors[idx_c % len(c_colors)], width=1.5),
                        fill="tozeroy",
                        opacity=0.4
                    ))

                fig_dec.update_xaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", gridcolor="#EDF2F7")
                fig_dec.update_yaxes(title_text="Intensity", gridcolor="#EDF2F7")
                fig_dec.update_layout(
                    title=f"{deconv_target_name} — 13C NMR Aromatic Deconvolution",
                    template="plotly_white",
                    height=440
                )
                st_plotly(fig_dec)

                # Таблица компонент
                df_comp = pd.DataFrame([
                    {
                        "Band": c["label"],
                        "Center_ppm": c["center_ppm"],
                        "Width_ppm": c["width_ppm"],
                        "Fraction_Area_Pct": f"{c['area_pct']}%"
                    }
                    for c in deconv_res["components"]
                ])
                st.markdown(f"###### {T[lang]['nmr_deconv_tbl_title']}")
                st.dataframe(df_comp, width="stretch", hide_index=True)
            else:
                st.warning(f"Деконволюция не удалась: {deconv_res.get('message', 'Недостаточно точек')}")

        with sec_tab2:
            st.markdown("##### " + T[lang]["nmr_lib_hdr"])
            c_lb1, c_lb2, c_lb3 = st.columns([2.5, 1.5, 2.0])
            with c_lb1:
                lib_target_name = st.selectbox(T[lang]["nmr_lib_sample_select"], sample_names, index=0, key="nmr_lib_sample_select")
            with c_lb2:
                lib_top_n = st.slider(T[lang]["nmr_lib_top_n"], min_value=3, max_value=8, value=5, step=1, key="nmr_lib_top_n_slider")
            with c_lb3:
                st.markdown("<br>", unsafe_allow_html=True)
                lib_run_btn = st.button(T[lang]["nmr_lib_btn"], type="primary", key="btn_run_nmr_lib_match")

            target_lib_s = current_nmr_samples[lib_target_name]
            target_lib_shifts = target_lib_s.chemical_shifts.copy()
            if recal_mode != "none":
                target_lib_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(target_lib_shifts, target_lib_s.intensity, mode=recal_mode)
            target_lib_s_c, _ = nmr_core.subtract_baseline(target_lib_s, method=base_method)
            y_lib_target = target_lib_s_c.corrected_intensity if base_method != "none" else target_lib_s.intensity

            match_df = nmr_core.match_spectrum_against_library(target_lib_shifts, y_lib_target, top_n=lib_top_n, lang=lang)

            if not match_df.empty:
                top_match = match_df.iloc[0]
                top_std_id = top_match["Reference_ID"]
                top_hqi = top_match["HQI_Similarity_Pct"]
                top_name = top_match["Standard_Name"]

                if top_hqi >= 85.0:
                    st.success(
                        f"🏆 **Топ-1 совпадение:** {top_name} (HQI = **{top_hqi:.1f}%**, Expected fa = **{top_match['Expected_Aromaticity_fa']}%**)"
                        if lang == "ru" else
                        f"🏆 **Top-1 Match:** {top_name} (HQI = **{top_hqi:.1f}%**, Expected fa = **{top_match['Expected_Aromaticity_fa']}%**)"
                    )
                else:
                    st.info(
                        f"🔎 **Лучшее совпадение:** {top_name} (HQI = **{top_hqi:.1f}%**)"
                        if lang == "ru" else
                        f"🔎 **Closest Standard:** {top_name} (HQI = **{top_hqi:.1f}%**)"
                    )

                # Наложение: образец vs эталон
                ref_curve = nmr_core.synthesize_reference_13c_curve(top_std_id, target_lib_shifts)
                s_norm = nmr_core.normalize_nmr_spectrum(y_lib_target, method="minmax")
                r_norm = nmr_core.normalize_nmr_spectrum(ref_curve, method="minmax")

                fig_lib = go.Figure()
                fig_lib.add_trace(go.Scatter(
                    x=target_lib_shifts, y=s_norm,
                    mode="lines", name=f"{lib_target_name} ({T[lang]['nmr_lib_sample_trace']})",
                    line=dict(color="#1A365D", width=2.2)
                ))
                fig_lib.add_trace(go.Scatter(
                    x=target_lib_shifts, y=r_norm,
                    mode="lines", name=f"{top_name} ({T[lang]['nmr_lib_ref_trace']})",
                    line=dict(color="#E53E3E", width=1.8, dash="dash")
                ))
                fig_lib.update_xaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", gridcolor="#EDF2F7")
                fig_lib.update_yaxes(title_text="Normalized Intensity [0, 1]", gridcolor="#EDF2F7")
                fig_lib.update_layout(
                    title=f"{T[lang]['nmr_lib_overlay_title']} — {lib_target_name} vs {top_name}",
                    template="plotly_white",
                    height=420
                )
                st_plotly(fig_lib)

                st.markdown(f"###### {T[lang]['nmr_lib_tbl_title']}")
                st.dataframe(match_df, width="stretch", hide_index=True)

    # =========================================================================
    # ВКЛАДКА 4: 🧬 ХЕМОМЕТРИЧЕСКИЕ ИНДЕКСЫ И ЭКСПОРТ В ML
    # =========================================================================
    with tab_nmr4:
        processed_samples = []
        for s in current_nmr_samples.values():
            s_copy = nmr_core.NMRSample(
                sample_id=s.sample_id,
                nucleus=s.nucleus,
                chemical_shifts=s.chemical_shifts.copy(),
                intensity=s.intensity.copy(),
                metadata=dict(s.metadata)
            )
            if recal_mode != "none":
                s_copy.chemical_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(
                    s_copy.chemical_shifts, s_copy.intensity, mode=recal_mode
                )
            processed_samples.append(s_copy)

        df_nmr_desc = nmr_core.batch_process_nmr_samples(
            processed_samples,
            baseline_method=base_method
        )

        st.markdown(
            f"##### {len(df_nmr_desc)} " + ("Processed Samples Descriptor Matrix" if lang == "en" else "Сводная матрица дескрипторов ЯМР")
        )

        st.dataframe(
            df_nmr_desc.style.format({
                "NMR_fa_aromaticity": "{:.1f}%",
                "NMR_hb_hi_index": "{:.2f}",
                "NMR_alkyl_o_alkyl": "{:.2f}",
                "NMR_lignin_signature": "{:.1f}%",
                "NMR_coo_ar_ratio": "{:.2f}",
                "NMR_C_aliph_pct": "{:.1f}%",
                "NMR_C_O_alkyl_pct": "{:.1f}%",
                "NMR_C_aromatic_pct": "{:.1f}%",
                "NMR_C_carboxyl_pct": "{:.1f}%",
                "NMR_C_carbonyl_pct": "{:.1f}%",
                "NMR_C_methoxy_pct": "{:.1f}%",
                "NMR_C_phenolic_pct": "{:.1f}%",
                "NMR_estimated_mw": "{:.0f}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True
        )

        # Диаграмма рассеяния: Ароматичность fa vs Лигниновый индекс с защитой размера пузырьков
        df_scatter = df_nmr_desc.copy()
        df_scatter["_bubble_size"] = np.clip(
            df_scatter["NMR_hb_hi_index"].fillna(1.0).astype(float),
            0.5, 15.0
        )
        fig_scatter = px.scatter(
            df_scatter,
            x="NMR_fa_aromaticity",
            y="NMR_lignin_signature",
            hover_name="Sample_ID",
            size="_bubble_size",
            color="NMR_fa_aromaticity",
            color_continuous_scale="Viridis",
            hover_data={"_bubble_size": False, "NMR_hb_hi_index": ":.2f"},
            title="Aromaticity (fa %) vs Lignin Signature (%) Across Samples" if lang == "en" else "Степень ароматичности fa (%) vs Лигниновый индекс (%) по всем пробам",
            labels={"NMR_fa_aromaticity": "Aromaticity fa (%)", "NMR_lignin_signature": "Lignin Signature (%)"}
        )
        fig_scatter.update_layout(height=420, template="plotly_white")
        st_plotly(fig_scatter)

        c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
        with c_d1:
            st.download_button(
                label=T[lang]["nmr_dl_csv"],
                data=df_nmr_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_NMR_Descriptors.csv",
                mime="text/csv",
                key="dl_nmr_descriptors_csv",
            )
        with c_d2:
            excel_nmr_buf = io.BytesIO()
            with pd.ExcelWriter(excel_nmr_buf, engine="openpyxl") as writer:
                df_nmr_desc.to_excel(writer, index=False, sheet_name="NMR_Descriptors")
            st.download_button(
                label=T[lang]["nmr_dl_excel"],
                data=excel_nmr_buf.getvalue(),
                file_name="ChemoSuite_NMR_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_nmr_descriptors_excel",
            )
        with c_d3:
            if st.button(T[lang]["nmr_sync_ml_btn"], type="primary", key="btn_sync_nmr_ml_descriptors"):
                st.session_state["nmr_ml_descriptors"] = df_nmr_desc
                st.toast(T[lang]["nmr_sync_success"].format(n=len(df_nmr_desc)), icon="✅")
                st.rerun()

# ==============================================================================
# МОДУЛЬ 6: ТЕРМИЧЕСКИЙ АНАЛИЗ И ПИРОЛИЗ (Py-GC-MS & TGA/DSC)
# ==============================================================================
elif active_module == T[lang]["mod6_name"]:
    if not THERMAL_CORE_AVAILABLE or thermal_core is None:
        st.error(T[lang]["thermal_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #5c0617 0%, #870000 50%, #190a2c 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["thermal_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #F5D0D0; font-size: 13px;">{T[lang]["thermal_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.popover("ℹ️ " + ("Руководство: Аналитический пиролиз (Py-GC-MS) и термогравиметрия (TGA/DTG) в анализе РОВ и лигнина" if lang == "ru" else "Guide: Analytical Pyrolysis (Py-GC-MS) & Thermogravimetry (TGA/DTG) in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы термодеструкции природного и техногенного органического вещества

**Аналитический пиролиз (Flash Py-GC-MS, 500–600°C)** в бескислородной среде гелия вызывает термический разрыв наименее прочных связей ($\beta$-O-4, $\beta$-5, алифатические мостики) макромолекулярного скелета, высвобождая летучие мономерные фрагменты, отражающие структуру исходного полимера:

| Группа мономеров | Ключевые маркеры (Py-GC-MS) | Специфичность | Экологическое / биогеохимическое значение |
| :--- | :--- | :--- | :--- |
| **Гваяцильные (G)** | Гваякол, 4-Метилгваякол, Ванилин, Ванилиновая к-та | Хвойный лигнин (сосна, лиственница Прибайкалья) | Доминирующий маркер сульфатного шлам-лигнина БЦБК |
| **Сирингильные (S)** | Сирингол, 4-Метилсирингол, Сирингальдегид, Сиринговая к-та | Лиственный лигнин (береза, осина) | Источник аллохтонного терригенного стока |
| **п-Гидроксифенильные (H)** | Фенол, о/м/п-Крезолы, 4-Винилфенол | Травянистые растения, мхи, торфяники | Автохтонный и моховой гуминовый пул |
| **Углеводы (Carb)** | Фурфурол, 5-HMF, Левоглюкозан, 2-Фурилметилкетон | Полисахариды, целлюлоза, гемицеллюлозы | Доля биогенного углеводного матрикса |
| **Азотистые (N)** | Пиридин, Пиррол, Индол, Метилиндол | Белки, пептиды, микробная биомасса | Индикатор микробной активности и автохтонного РОВ |

---

### 📊 Хемометрические индексы пиролиза:
* **Индекс $S/G$:** Отношение сирингильных мономеров к гваяцильным. Для чистых хвойных лигнинов $S/G \approx 0.01–0.05$; для лиственных $S/G > 1.0$.
* **Индекс $H/G$:** Доля неметоксилированных фенолов. Возрастает при деметилировании и микробной биодеградации.
* **Индекс $L/C$:** Отношение суммарного пула лигнина $(G+S+H)$ к углеводам. Ключевой биомаркер зрелости и техногенного загрязнения лигнином.
* **Степень окисления $(Ac/Al)_G$ и $(Ac/Al)_S$:** Отношение карбоновых кислот к соответствующим альдегидам (ванилиновая к-та / ванилин). В интактном лигнине $(Ac/Al)_G < 0.20$; при техногенном щелочном окислении или глубоком выветривании возрастает до $0.50–1.20$.

---

### 🔥 Термогравиметрия (TGA) и дифференциальная термогравиметрия (DTG, $-dm/dT$):
Кривая убыли массы и скорость термодеструкции разделяются на 4 температурные зоны:
1. **Зона I ($T < 150^\circ\text{C}$):** Десорбция гигроскопической влаги, окклюдированных газов и сверхлетучих фрагментов.
2. **Зона II ($150–350^\circ\text{C}$):** Деструкция гемицеллюлоз, эфирных связей $\beta$-O-4 и алифатических боковых цепей ($\Delta m_{\text{aliph}}$).
3. **Зона III ($350–600^\circ\text{C}$):** Глубокий пиролиз ароматического углеродного скелета, конденсация и распад стабильных C–C связей лигнина ($\Delta m_{\text{arom}}$).
4. **Зона IV ($T > 600^\circ\text{C}$):** Обугливание высококонденсированного полиароматического кокса и термолиз минерального зольного остатка.
* **Thermal Index ($TI$):** $TI = \frac{\Delta m_{\text{III}}}{\Delta m_{\text{II}}} = \frac{\Delta m_{\text{arom}}}{\Delta m_{\text{aliph}}}$. Отражает термическую стабильность и долю тугоплавкого конденсированного лигнина.
""" if lang == "ru" else r"""
### 🔬 Physicochemical Principles of Thermal Degradation in NOM & Lignin

**Analytical Pyrolysis (Flash Py-GC-MS, 500–600°C)** under inert helium cleaves weaker ether ($\beta$-O-4) and aliphatic bonds, liberating volatile monomers that reflect structural precursors:

| Monomer Class | Representative Biomarkers | Biological Specificity | Environmental / Geochemical Relevance |
| :--- | :--- | :--- | :--- |
| **Guaiacyl (G)** | Guaiacol, 4-Methylguaiacol, Vanillin, Vanillic acid | Softwood lignin (Pine, Larch from Lake Baikal basin) | Primary chemical signature of kraft sludge lignin |
| **Syringyl (S)** | Syringol, 4-Methylsyringol, Syringaldehyde, Syringic acid | Hardwood lignin (Birch, Aspen) | Marker of angiosperm terrigenous runoff |
| **p-Hydroxyphenyl (H)**| Phenol, o/m/p-Cresols, 4-Vinylphenol | Non-woody tissues, mosses, peatlands | Autochthonous & bryophyte organic pool |
| **Carbohydrate (Carb)**| Furfural, 5-HMF, Levoglucosan, 2-Furylmethylketone | Polysaccharides, cellulose, hemicellulose | Biogenic carbohydrate matrix contribution |
| **Nitrogen (N)** | Pyridine, Pyrrole, Indole, Methylindole | Proteins, amino sugars, microbial biomass | Microbial activity and autochthonous NOM |

---

### 📊 Pyrolysis Chemometric Indices:
* **$S/G$ Ratio:** Syringyl-to-guaiacyl ratio ($S/G < 0.05$ for gymnosperms/softwood kraft lignin, $> 1.0$ for angiosperms).
* **$H/G$ Ratio:** p-Hydroxyphenyl to guaiacyl ratio (increases during demethylation & biodegradation).
* **$L/C$ Ratio:** Total lignin $(G+S+H)$ to carbohydrate ratio. Discriminates biogenic NOM from concentrated kraft lignin.
* **Oxidation State $(Ac/Al)_G$ & $(Ac/Al)_S$:** Acid-to-aldehyde ratios. Fresh wood $< 0.20$; degraded or pulping-oxidized lignin reaches $0.50–1.20$.

---

### 🔥 Thermogravimetry (TGA) & Derivative (DTG, $-dm/dT$):
Decomposition is separated into 4 kinetic temperature zones:
1. **Zone I ($T < 150^\circ\text{C}$):** Moisture desorption & volatile degasification.
2. **Zone II ($150–350^\circ\text{C}$):** Hemicellulose, aliphatic scission & $\beta$-O-4 cleavage ($\Delta m_{\text{aliph}}$).
3. **Zone III ($350–600^\circ\text{C}$):** Pyrolysis of cross-linked aromatic lignin core & refractory C-C linkages ($\Delta m_{\text{arom}}$).
4. **Zone IV ($T > 600^\circ\text{C}$):** Polycondensed char graphitization & inorganic mineral ash.
* **Thermal Index ($TI$):** $TI = \frac{\Delta m_{\text{III}}}{\Delta m_{\text{II}}} = \frac{\Delta m_{\text{arom}}}{\Delta m_{\text{aliph}}}$. Quantifies thermal stability and aromatic condensation.
""")

    with st.sidebar:
        st.markdown(f"### {T[lang]['thermal_sidebar_hdr']}")
        thermal_src = st.radio(
            T[lang]["thermal_src_label"],
            [T[lang]["thermal_src_upload"], T[lang]["thermal_src_synth"]],
            index=0,
            key="thermal_data_source_radio"
        )

        if st.session_state.get("_prev_thermal_src") != thermal_src:
            st.session_state["_prev_thermal_src"] = thermal_src
            st.session_state["thermal_samples"] = {}

        if thermal_src == T[lang]["thermal_src_upload"]:
            up_py_files = st.file_uploader(
                T[lang]["thermal_up_py_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="thermal_up_py_files"
            )
            up_tga_files = st.file_uploader(
                T[lang]["thermal_up_tga_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="thermal_up_tga_files"
            )

            if up_py_files:
                for uf in up_py_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_PyGCMS", "").replace("_pygcms", "")
                    try:
                        py_obj = thermal_core.parse_py_gcms_file(b_content, sample_id=s_id)
                        if s_id not in st.session_state["thermal_samples"]:
                            st.session_state["thermal_samples"][s_id] = thermal_core.ThermalSample(sample_id=s_id, py_gcms=py_obj)
                        else:
                            st.session_state["thermal_samples"][s_id].py_gcms = py_obj
                    except Exception as e_py:
                        st.warning(f"Py-GC-MS error ({uf.name}): {e_py}")

            if up_tga_files:
                for uf in up_tga_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_TGA", "").replace("_tga", "")
                    try:
                        tga_obj = thermal_core.parse_tga_file(b_content, sample_id=s_id)
                        if s_id not in st.session_state["thermal_samples"]:
                            st.session_state["thermal_samples"][s_id] = thermal_core.ThermalSample(sample_id=s_id, tga=tga_obj)
                        else:
                            st.session_state["thermal_samples"][s_id].tga = tga_obj
                    except Exception as e_tga:
                        st.warning(f"TGA error ({uf.name}): {e_tga}")
        else:
            if not st.session_state["thermal_samples"]:
                demo_th_dir = os.path.join("demo_data", "thermal")
                if os.path.exists(demo_th_dir) and any(f.endswith(".csv") for f in os.listdir(demo_th_dir)):
                    all_files = os.listdir(demo_th_dir)
                    sample_map = {}
                    for fn in all_files:
                        if not fn.endswith(".csv") or fn == "thermal_descriptors_matrix.csv":
                            continue
                        fpath = os.path.join(demo_th_dir, fn)
                        with open(fpath, "rb") as f:
                            b_content = f.read()
                        if "_PyGCMS" in fn:
                            s_id = fn.replace("_PyGCMS.csv", "")
                            if s_id not in sample_map:
                                sample_map[s_id] = thermal_core.ThermalSample(sample_id=s_id)
                            sample_map[s_id].py_gcms = thermal_core.parse_py_gcms_file(b_content, sample_id=s_id)
                        elif "_TGA" in fn:
                            s_id = fn.replace("_TGA.csv", "")
                            if s_id not in sample_map:
                                sample_map[s_id] = thermal_core.ThermalSample(sample_id=s_id)
                            sample_map[s_id].tga = thermal_core.parse_tga_file(b_content, sample_id=s_id)
                    st.session_state["thermal_samples"] = sample_map
                else:
                    synth_samples, _ = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=6, save_dir=demo_th_dir)
                    st.session_state["thermal_samples"] = synth_samples

    if not st.session_state["thermal_samples"]:
        st.info(T[lang]["thermal_no_samples"])
    else:
        # Сборка матрицы дескрипторов по всем образцам
        th_rows = []
        for s_id, s_obj in st.session_state["thermal_samples"].items():
            row = {"Sample_ID": s_id}
            row.update(thermal_core.extract_thermal_descriptors(s_obj))
            row["Class_Target"] = 1 if any(k in str(s_id).lower() for k in ["lignin", "sludge", "шлам"]) else 0
            th_rows.append(row)
        df_thermal_desc = pd.DataFrame(th_rows)

        sample_list = list(st.session_state["thermal_samples"].keys())
        col_sel1, col_sel2 = st.columns([3, 1.5])
        with col_sel1:
            active_s_id = st.selectbox(
                T[lang]["thermal_active_sample"],
                sample_list,
                index=0,
                key="select_active_thermal_sample"
            )
        active_sample = st.session_state["thermal_samples"][active_s_id]
        has_py = active_sample.py_gcms is not None
        has_tga = active_sample.tga is not None

        with col_sel2:
            st.markdown(
                f"<div style='padding-top: 28px;'>"
                f"<span style='background: {'#27ae60' if has_py else '#7f8c8d'}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Py-GC-MS: {'✓' if has_py else '✗'}</span>"
                f"<span style='background: {'#c0392b' if has_tga else '#7f8c8d'}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;'>TGA: {'✓' if has_tga else '✗'}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        tab_th1, tab_th2, tab_th3, tab_th4 = st.tabs([
            T[lang]["thermal_tab1_title"],
            T[lang]["thermal_tab2_title"],
            T[lang]["thermal_tab3_title"],
            T[lang]["thermal_tab4_title"],
        ])

        # ----------------------------------------------------------------------
        # ТАБ 1: Py-GC-MS Хроматограммы и мономеры
        # ----------------------------------------------------------------------
        with tab_th1:
            if not has_py:
                st.warning("Для активного образца отсутствуют данные Py-GC-MS. Загрузите соответствующий файл или используйте эталонный бенчмарк." if lang == "ru" else "No Py-GC-MS data available for the active sample.")
            else:
                py_data = active_sample.py_gcms
                st.write(f"#### {T[lang]['thermal_tab1_title']} — **{active_s_id}**")

                fig_tic = go.Figure()
                if len(py_data.retention_times) > 0 and len(py_data.tic_intensity) > 0:
                    fig_tic.add_trace(go.Scatter(
                        x=py_data.retention_times,
                        y=py_data.tic_intensity,
                        mode="lines",
                        line=dict(color="#1A365D", width=1.5),
                        name="TIC (Raw Signal)",
                    ))

                class_palette = {
                    "Guaiacyl": "#e67e22",
                    "Syringyl": "#8e44ad",
                    "p-Hydroxyphenyl": "#2980b9",
                    "Carbohydrate": "#27ae60",
                    "Nitrogen": "#c0392b",
                    "Aliphatic": "#f39c12",
                    "Other": "#7f8c8d"
                }

                if py_data.peaks:
                    total_peak_area = sum(getattr(p, "area", getattr(p, "peak_area", 0.0)) for p in py_data.peaks)
                    # Разметка пиков цветными маркерами
                    for p in py_data.peaks:
                        p_cls = getattr(p, "compound_class", getattr(p, "chemical_class", "Other"))
                        p_color = class_palette.get(p_cls, "#7f8c8d")
                        p_name = (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak")
                        p_rt = getattr(p, "rt", getattr(p, "retention_time", 0.0))
                        p_area = getattr(p, "area", getattr(p, "peak_area", 0.0))
                        p_area_pct = getattr(p, "area_pct", 0.0)
                        if p_area_pct <= 0.0 and total_peak_area > 0:
                            p_area_pct = (p_area / total_peak_area) * 100.0
                        p_monomer = getattr(p, "monomer_type", p_cls)
                        p_mw = getattr(p, "mw", 0.0)
                        p_formula = getattr(p, "formula", "") or "—"

                        fig_tic.add_trace(go.Scatter(
                            x=[p_rt],
                            y=[p_area if (len(py_data.tic_intensity) == 0) else np.interp(p_rt, py_data.retention_times, py_data.tic_intensity)],
                            mode="markers",
                            marker=dict(size=9, color=p_color, symbol="diamond", line=dict(width=1, color="black")),
                            name=p_cls,
                            legendgroup=p_cls,
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{p_name}</b><br>"
                                f"Класс: {p_cls} ({p_monomer})<br>"
                                f"RT: {p_rt:.2f} мин | Площадь: {p_area_pct:.2f}%<br>"
                                f"Формула: {p_formula} (MW {p_mw:.1f})"
                                if lang == "ru" else
                                f"<b>{p_name}</b><br>"
                                f"Class: {p_cls} ({p_monomer})<br>"
                                f"RT: {p_rt:.2f} min | Area: {p_area_pct:.2f}%<br>"
                                f"Formula: {p_formula} (MW {p_mw:.1f})"
                            )
                        ))

                fig_tic.update_layout(
                    xaxis_title="Retention Time (min)" if lang == "en" else "Время удерживания RT (мин)",
                    yaxis_title="Total Ion Current (TIC)" if lang == "en" else "Интенсивность ионного тока (TIC)",
                    template="plotly_white",
                    height=440,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st_plotly(fig_tic)

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.write(f"**{T[lang]['thermal_dist_pie_title']}**")
                    class_areas = {
                        "Guaiacyl (G)": py_data.indices.get("Py_G_pct", 0.0),
                        "Syringyl (S)": py_data.indices.get("Py_S_pct", 0.0),
                        "p-Hydroxyphenyl (H)": py_data.indices.get("Py_H_pct", 0.0),
                        "Carbohydrate": py_data.indices.get("Py_Carbohydrate_Area_pct", py_data.indices.get("Py_Carbohydrate_pct", 0.0)),
                        "Nitrogen / Protein": py_data.indices.get("Py_Nitrogen_Area_pct", py_data.indices.get("Py_Nitrogen_pct", 0.0)),
                        "Aliphatic / Lipid": py_data.indices.get("Py_Aliphatic_Area_pct", 0.0),
                        "Unassigned": py_data.indices.get("Py_Unassigned_Area_pct", 0.0),
                    }
                    class_areas_filtered = {k: v for k, v in class_areas.items() if v > 0.01}
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(class_areas_filtered.keys()),
                        values=list(class_areas_filtered.values()),
                        hole=0.45,
                        marker=dict(colors=[
                            class_palette.get(k.split()[0], "#7f8c8d") for k in class_areas_filtered.keys()
                        ])
                    )])
                    fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
                    st_plotly(fig_pie)

                with col_p2:
                    st.write(f"**{T[lang]['thermal_monomers_bar_title']}**")
                    if py_data.peaks:
                        def _get_pk_pct(pk):
                            pct = getattr(pk, "area_pct", 0.0)
                            if pct <= 0.0 and total_peak_area > 0:
                                pct = (getattr(pk, "area", getattr(pk, "peak_area", 0.0)) / total_peak_area) * 100.0
                            return pct

                        top_peaks = sorted(py_data.peaks, key=_get_pk_pct, reverse=True)[:10]
                        df_top = pd.DataFrame([
                            {
                                "Compound": (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak"),
                                "Class": getattr(p, "compound_class", getattr(p, "chemical_class", "Other")),
                                "Area_%": round(_get_pk_pct(p), 2)
                            }
                            for p in top_peaks
                        ])
                        fig_bar = px.bar(
                            df_top,
                            x="Area_%",
                            y="Compound",
                            orientation="h",
                            color="Class",
                            color_discrete_map=class_palette,
                            text=df_top["Area_%"].apply(lambda v: f"{v:.1f}%")
                        )
                        fig_bar.update_layout(
                            yaxis=dict(autorange="reversed"),
                            xaxis_title="Relative Area (%)" if lang == "en" else "Относительная площадь (%)",
                            template="plotly_white",
                            height=360,
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        st_plotly(fig_bar)

                with st.expander(f"📋 {T[lang]['thermal_peaks_hdr']} ({len(py_data.peaks)} пиков)", expanded=False):
                    if py_data.peaks:
                        peaks_table = []
                        for p in py_data.peaks:
                            c_name = (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak")
                            c_class = getattr(p, "compound_class", getattr(p, "chemical_class", "Other"))
                            c_rt = getattr(p, "rt", getattr(p, "retention_time", 0.0))
                            c_pct = getattr(p, "area_pct", 0.0)
                            if c_pct <= 0.0 and total_peak_area > 0:
                                c_pct = (getattr(p, "area", getattr(p, "peak_area", 0.0)) / total_peak_area) * 100.0
                            c_monomer = getattr(p, "monomer_type", c_class)
                            c_mw = getattr(p, "mw", 0.0)
                            c_formula = getattr(p, "formula", "") or "—"
                            c_cas = getattr(p, "cas", "") or "—"
                            peaks_table.append({
                                "RT (min)": round(c_rt, 2),
                                "Compound": c_name,
                                "Class": c_class,
                                "Monomer": c_monomer,
                                "Area (%)": round(c_pct, 2),
                                "Formula": c_formula,
                                "MW": round(float(c_mw), 1) if c_mw > 0 else np.nan,
                                "CAS": c_cas,
                            })
                        df_p_tbl = pd.DataFrame(peaks_table)
                        st_df(df_p_tbl)
                        st.download_button(
                            label=T[lang]["thermal_peaks_dl_btn"],
                            data=df_p_tbl.to_csv(index=False).encode("utf-8"),
                            file_name=f"{active_s_id}_PyGCMS_Peaks.csv",
                            mime="text/csv",
                            key=f"dl_peaks_{active_s_id}"
                        )

        # ----------------------------------------------------------------------
        # ТАБ 2: TGA / DTG Термограммы и температурные зоны
        # ----------------------------------------------------------------------
        with tab_th2:
            if not has_tga:
                st.warning("Для активного образца отсутствуют данные TGA/DTG. Загрузите файл термогравиметрии или выберите эталонный бенчмарк." if lang == "ru" else "No TGA/DTG data available for the active sample.")
            else:
                tga_data = active_sample.tga
                st.write(f"#### {T[lang]['thermal_tga_title']} — **{active_s_id}**")

                col_to1, col_to2 = st.columns(2)
                show_zones = col_to1.checkbox(T[lang]["thermal_tga_zones_chk"], value=True, key="chk_tga_zones")
                show_dtg = col_to2.checkbox(T[lang]["thermal_tga_dtg_chk"], value=True, key="chk_tga_dtg")

                fig_tga = make_subplots(specs=[[{"secondary_y": True}]])
                fig_tga.add_trace(
                    go.Scatter(
                        x=tga_data.temperature,
                        y=tga_data.mass_pct,
                        mode="lines",
                        line=dict(color="#C0392B", width=2.5),
                        name="TGA Mass (%)",
                    ),
                    secondary_y=False
                )

                if show_dtg:
                    fig_tga.add_trace(
                        go.Scatter(
                            x=tga_data.temperature,
                            y=tga_data.dtg,
                            mode="lines",
                            line=dict(color="#2980B9", width=2.0),
                            name="DTG (-dm/dT, %/°C)",
                        ),
                        secondary_y=True
                    )

                if show_zones:
                    # Зона I (<150°C)
                    fig_tga.add_vrect(x0=30, x1=150, fillcolor="#3498DB", opacity=0.10, layer="below", line_width=0,
                                      annotation_text="Zone I: Moisture" if lang == "en" else "Зона I: Влага (<150°C)", annotation_position="top left")
                    # Зона II (150-350°C)
                    fig_tga.add_vrect(x0=150, x1=350, fillcolor="#F1C40F", opacity=0.12, layer="below", line_width=0,
                                      annotation_text="Zone II: Aliphatic" if lang == "en" else "Зона II: Алифатика (150–350°C)", annotation_position="top left")
                    # Зона III (350-600°C)
                    fig_tga.add_vrect(x0=350, x1=600, fillcolor="#E74C3C", opacity=0.10, layer="below", line_width=0,
                                      annotation_text="Zone III: Lignin Core" if lang == "en" else "Зона III: Ядро лигнина (350–600°C)", annotation_position="top left")
                    # Зона IV (>600°C)
                    fig_tga.add_vrect(x0=600, x1=800, fillcolor="#95A5A6", opacity=0.12, layer="below", line_width=0,
                                      annotation_text="Zone IV: Char" if lang == "en" else "Зона IV: Кокс (>600°C)", annotation_position="top left")

                fig_tga.update_xaxes(title_text="Temperature (°C)" if lang == "en" else "Температура (°C)")
                fig_tga.update_yaxes(title_text="TGA Mass Loss (%)" if lang == "en" else "Масса образца TGA (%)", secondary_y=False)
                fig_tga.update_yaxes(title_text="DTG (-dm/dT, %/°C)" if lang == "en" else "Скорость деструкции DTG (-dm/dT, %/°C)", secondary_y=True)
                fig_tga.update_layout(template="plotly_white", height=450, margin=dict(l=40, r=40, t=30, b=40))
                st_plotly(fig_tga)

                # Метрики зон деструкции
                st.markdown("##### " + ("Кинетические зоны термодеструкции" if lang == "ru" else "Thermal Decomposition Kinetic Zones"))
                col_zm1, col_zm2, col_zm3, col_zm4 = st.columns(4)
                col_zm1.metric("Зона I (<150°C)" if lang == "ru" else "Zone I (<150°C)", f"{tga_data.indices.get('TGA_Zone_I_pct', 0.0):.1f}%", help="Влага и десорбция")
                col_zm2.metric("Зона II (150–350°C)" if lang == "ru" else "Zone II (150-350°C)", f"{tga_data.indices.get('TGA_Zone_II_pct', 0.0):.1f}%", help="Гемицеллюлозы и алифатика")
                col_zm3.metric("Зона III (350–600°C)" if lang == "ru" else "Zone III (350-600°C)", f"{tga_data.indices.get('TGA_Zone_III_pct', 0.0):.1f}%", help="Ароматический каркас лигнина")
                col_zm4.metric("Зона IV (>600°C)" if lang == "ru" else "Zone IV (>600°C)", f"{tga_data.indices.get('TGA_Zone_IV_pct', 0.0):.1f}%", help="Кокс и минеральный остаток")

                col_tm1, col_tm2, col_tm3, col_tm4 = st.columns(4)
                col_tm1.metric(T[lang]["thermal_kpi_ti"], f"{tga_data.indices.get('TGA_Thermal_Index_TI', 0.0):.2f}", help="Δm(III) / Δm(II)")
                col_tm2.metric(T[lang]["thermal_kpi_t50"], f"{tga_data.indices.get('TGA_T50_degC', 0.0):.1f} °C", help="Температура 50% убыли массы")
                col_tm3.metric("T_max (DTG)", f"{tga_data.indices.get('DTG_Tmax_degC', 0.0):.1f} °C", help="Пик максимальной скорости деструкции")
                col_tm4.metric(T[lang]["thermal_kpi_residue"], f"{tga_data.indices.get('TGA_Residue_800C_pct', 0.0):.1f}%", help="Неразложившийся остаток при 800°C")

        # ----------------------------------------------------------------------
        # ТАБ 3: Хемометрические индексы (S/G, L/C, TI)
        # ----------------------------------------------------------------------
        with tab_th3:
            st.write(f"#### {T[lang]['thermal_tab3_title']}")

            col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
            sg_val = active_sample.py_gcms.indices.get("Py_S_G_ratio", 0.0) if has_py else 0.0
            hg_val = active_sample.py_gcms.indices.get("Py_H_G_ratio", 0.0) if has_py else 0.0
            lc_val = active_sample.py_gcms.indices.get("Py_L_C_ratio", 0.0) if has_py else 0.0
            ox_val = (active_sample.py_gcms.indices.get("Py_Acid_Aldehyde_G", active_sample.py_gcms.indices.get("Py_Ac_Al_G", 0.0))) if has_py else 0.0
            ti_val = (active_sample.tga.indices.get("TGA_Thermal_Index_TI", active_sample.tga.indices.get("TGA_Thermostability_Index", 0.0))) if has_tga else 0.0
            t50_val = (active_sample.tga.indices.get("TGA_T50_degC", active_sample.tga.indices.get("TGA_T_50pct", 0.0))) if has_tga else 0.0

            col_k1.metric("S/G Ratio", f"{sg_val:.3f}" if has_py else "—")
            col_k2.metric("H/G Ratio", f"{hg_val:.3f}" if has_py else "—")
            col_k3.metric("L/C Ratio", f"{lc_val:.2f}" if has_py else "—")
            col_k4.metric("(Ac/Al)_G", f"{ox_val:.2f}" if has_py else "—")
            col_k5.metric("TI (TGA)", f"{ti_val:.2f}" if has_tga else "—")
            col_k6.metric("T_50% (TGA)", f"{t50_val:.1f} °C" if has_tga else "—")

            st.markdown("---")
            st.write("##### " + ("Сводная таблица термических дескрипторов по всем пробам сессии" if lang == "ru" else "Summary Table of Thermal Descriptors Across All Samples"))
            st_df(df_thermal_desc)

            # Радар-диаграмма сравнения активного образца со средней картиной
            if len(df_thermal_desc) > 1 and has_py and has_tga:
                st.write("##### " + ("Мультипараметрический профиль образца (Fingerprint Radar)" if lang == "ru" else "Sample Thermal Fingerprint Radar"))
                radar_metrics = ["Py_S_G_ratio", "Py_L_C_ratio", "Py_Ac_Al_G", "TGA_Thermal_Index_TI", "Py_Total_Lignin_pct"]
                labels = ["S/G Ratio", "L/C Ratio", "(Ac/Al)_G", "Thermal Index TI", "Lignin Area (%)"]

                # Нормализация для радара [0, 1]
                vals_sample = []
                vals_mean = []
                for m in radar_metrics:
                    m_lookup = m
                    if m not in df_thermal_desc.columns:
                        if m == "Py_Ac_Al_G" and "Py_Acid_Aldehyde_G" in df_thermal_desc.columns:
                            m_lookup = "Py_Acid_Aldehyde_G"
                        elif m == "TGA_Thermal_Index_TI" and "TGA_Thermostability_Index" in df_thermal_desc.columns:
                            m_lookup = "TGA_Thermostability_Index"
                        elif m == "Py_Total_Lignin_pct" and "Py_Total_Lignin_Area_pct" in df_thermal_desc.columns:
                            m_lookup = "Py_Total_Lignin_Area_pct"
                    col_vals = df_thermal_desc[m_lookup] if m_lookup in df_thermal_desc.columns else pd.Series([0.0])
                    v_min, v_max = col_vals.min(), col_vals.max()
                    denom = (v_max - v_min) if (v_max - v_min) > 1e-6 else 1.0
                    val_from_sample = (
                        active_sample.py_gcms.indices.get(m, active_sample.py_gcms.indices.get(m_lookup, 0.0))
                        if active_sample.py_gcms else
                        active_sample.tga.indices.get(m, active_sample.tga.indices.get(m_lookup, 0.0))
                    )
                    v_s = (val_from_sample - v_min) / denom
                    v_m = (col_vals.mean() - v_min) / denom
                    vals_sample.append(float(np.clip(v_s, 0.0, 1.0)))
                    vals_mean.append(float(np.clip(v_m, 0.0, 1.0)))

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_sample + [vals_sample[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    name=active_s_id,
                    line_color="#c0392b"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_mean + [vals_mean[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    name="Mean (All Samples)" if lang == "en" else "Среднее (Все пробы)",
                    line_color="#2980b9",
                    opacity=0.5
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=420,
                    margin=dict(l=40, r=40, t=30, b=30)
                )
                st_plotly(fig_radar)

        # ----------------------------------------------------------------------
        # ТАБ 4: Мультимодальная интеграция и экспорт
        # ----------------------------------------------------------------------
        with tab_th4:
            st.write(f"#### {T[lang]['thermal_tab4_title']}")

            tga_col = "TGA_Thermal_Index_TI" if "TGA_Thermal_Index_TI" in df_thermal_desc.columns else ("TGA_Thermostability_Index" if "TGA_Thermostability_Index" in df_thermal_desc.columns else None)
            lig_col = "Py_Total_Lignin_Area_pct" if "Py_Total_Lignin_Area_pct" in df_thermal_desc.columns else ("Py_Total_Lignin_pct" if "Py_Total_Lignin_pct" in df_thermal_desc.columns else None)

            if len(df_thermal_desc) > 1 and tga_col and "Py_L_C_ratio" in df_thermal_desc.columns:
                df_plot = df_thermal_desc.copy()
                if lig_col:
                    df_plot["_size"] = df_plot[lig_col].apply(lambda v: max(float(v), 5.0) if pd.notnull(v) else 10.0)
                else:
                    df_plot["_size"] = 10.0
                fig_scatter = px.scatter(
                    df_plot,
                    x=tga_col,
                    y="Py_L_C_ratio",
                    text="Sample_ID" if "Sample_ID" in df_plot.columns else None,
                    size="_size",
                    color="Class_Target" if "Class_Target" in df_plot.columns else None,
                    color_continuous_scale="Reds",
                    title="Thermal Stability (TI) vs Lignin/Carbohydrate Ratio (L/C)" if lang == "en" else "Термический индекс (TI) vs Лигнин/Углеводы (L/C)",
                    labels={
                        tga_col: "Thermal Index TI (Zone III / Zone II)",
                        "Py_L_C_ratio": "Lignin / Carbohydrate Ratio (L/C)",
                        "Class_Target": "Class"
                    }
                )
                fig_scatter.update_layout(height=420, template="plotly_white")
                st_plotly(fig_scatter)

            c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
            with c_d1:
                st.download_button(
                    label=T[lang]["thermal_dl_csv"],
                    data=df_thermal_desc.to_csv(index=False).encode("utf-8"),
                    file_name="ChemoSuite_Thermal_Descriptors.csv",
                    mime="text/csv",
                    key="dl_thermal_descriptors_csv",
                )
            with c_d2:
                excel_th_buf = io.BytesIO()
                with pd.ExcelWriter(excel_th_buf, engine="openpyxl") as writer:
                    df_thermal_desc.to_excel(writer, index=False, sheet_name="Thermal_Descriptors")
                st.download_button(
                    label=T[lang]["thermal_dl_excel"],
                    data=excel_th_buf.getvalue(),
                    file_name="ChemoSuite_Thermal_Descriptors.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_thermal_descriptors_excel",
                )
            with c_d3:
                if st.button(T[lang]["thermal_sync_ml_btn"], type="primary", key="btn_sync_thermal_ml_descriptors"):
                    st.session_state["thermal_ml_descriptors"] = df_thermal_desc
                    st.toast(T[lang]["thermal_sync_success"].format(n=len(df_thermal_desc)), icon="✅")
                    st.rerun()

# ==============================================================================
# МОДУЛЬ 7: ГЕЛЬ-ПРОНИКАЮЩАЯ И ЭКСКЛЮЗИОННАЯ ХРОМАТОГРАФИЯ (SEC / GPC)
# ==============================================================================
elif active_module == T[lang]["mod7_name"]:
    if not SEC_CORE_AVAILABLE or sec_core is None:
        st.error(T[lang]["sec_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #063e36 0%, #0d5c4d 50%, #04241d 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["sec_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #BEE3DB; font-size: 13px;">{T[lang]["sec_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.popover("ℹ️ " + ("Руководство: Эксклюзионная хроматография (SEC / GPC) в анализе РОВ и лигнина" if lang == "ru" else "Guide: Size-Exclusion Chromatography (SEC / GPC) in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы гель-проникающей хроматографии (SEC / GPC) природного и техногенного органического вещества

**Эксклюзионная хроматография (HPSEC / GPC)** — фундаментальный физико-химический метод оценки молекулярно-массового распределения (ММР / MWD), полидисперсности и надмолекулярной ассоциации природного органического вещества (РОВ/DOM, гуминовые и фульвокислоты) и техногенных лигносульфонатов и шлам-лигнина.

Разделение компонентов происходит по их гидродинамическому объему ($V_h$) в порах неподвижной фазы при отсутствии сорбционных энтальпийных взаимодействий:

| Молекулярная фракция | Диапазон масс (MW) | Типичные компоненты биогеохимического пула | Экологическое / биосферное значение |
| :--- | :--- | :--- | :--- |
| **Фракция I (HMW)** | $> 20$ кДа | Высококонденсированные полимеры шлам-лигнина, биоколлоиды | Техногенный маркер стоков БЦБК, концентрированные комплексы |
| **Фракция II (MHMW)** | $5\text{--}20$ кДа | Гуминовые кислоты (HA), агрегаты ароматических доменов | Устойчивый гумусовый матрикс, связывание тяжелых металлов |
| **Фракция III (MMW)** | $1\text{--}5$ кДа | Фульвокислоты (FA), гидролизованный окисленный лигнин | Растворимый подвижный миграционный пул в воде озера Байкал |
| **Фракция IV (LMW)** | $< 1$ кДа | Низкомолекулярные мономеры, танины, простые органические кислоты | Биодоступный субстрат автохтонной микробиоты, микробный рециклинг |

---

### 📊 Среднемолекулярные массы и коэффициенты полидисперсности:
* **Среднечисловая масса ($M_n$):** $M_n = \frac{\sum h_i}{\sum (h_i / M_i)}$ — чувствительна к низкомолекулярным олигомерам и мономерам.
* **Среднемассовая масса ($M_w$):** $M_w = \frac{\sum (h_i \cdot M_i)}{\sum h_i}$ — отражает основной гидродинамический объем макромолекул.
* **z-Средняя масса ($M_z$):** $M_z = \frac{\sum (h_i \cdot M_i^2)}{\sum (h_i \cdot M_i)}$ — чувствительна к крупным полимерным сеткам и коллоидным агрегатам.
* **Индекс полидисперсности ($Đ$ или PDI):** $Đ = \frac{M_w}{M_n}$. Для монодисперсных биополимеров $Đ \approx 1.0$; для природных фульвокислот $Đ \approx 1.3–2.2$; для техногенного поликонденсированного лигнина $Đ > 3.0–8.0$.
* **Отношение $HMW/LMW$:** $\frac{\text{Frac I} + \text{Frac II}}{\text{Frac IV}}$ — чувствительный дискриминантный маркер техногенной трансформации водной экосистемы.
""" if lang == "ru" else r"""
### 🔬 Physicochemical Principles of Size-Exclusion Chromatography (SEC / GPC) in NOM & Lignin Analysis

**Size-Exclusion Chromatography (HPSEC / GPC)** separates macromolecules according to their hydrodynamic volume ($V_h$) in porous stationary phase pores without non-size interactions:

| Molecular Fraction | Mass Range (MW) | Representative Natural/Anthropogenic Components | Environmental & Geochemical Relevance |
| :--- | :--- | :--- | :--- |
| **Fraction I (HMW)** | $> 20$ kDa | Supramolecular kraft lignin clusters, biocolloids | Pulp mill sludge lignin marker, high persistence |
| **Fraction II (MHMW)**| $5\text{--}20$ kDa | Humic acids (HA), condensed polyphenolic networks | Stable soil/sediment humic pool, metal chelation |
| **Fraction III (MMW)**| $1\text{--}5$ kDa | Fulvic acids (FA), oxidized polyphenol fragments | Dominant mobile NOM fraction in Lake Baikal pelagic water |
| **Fraction IV (LMW)** | $< 1$ kDa | Low-MW organic acids, tannins, monomeric phenols | Labile autochthonous substrate for microbial uptake |

---

### 📊 Average Molecular Weights & Polydispersity Metrics:
* **Number-Average MW ($M_n$):** $M_n = \frac{\sum h_i}{\sum (h_i / M_i)}$ — sensitive to low-molecular weight oligomers.
* **Weight-Average MW ($M_w$):** $M_w = \frac{\sum (h_i \cdot M_i)}{\sum h_i}$ — represents average hydrodynamic bulk mass.
* **z-Average MW ($M_z$):** $M_z = \frac{\sum (h_i \cdot M_i^2)}{\sum (h_i \cdot M_i)}$ — sensitive to ultra-high mass supramolecular aggregates.
* **Polydispersity Index ($Đ$ or PDI):** $Đ = \frac{M_w}{M_n}$. Monodisperse standards $Đ \approx 1.0$; natural fulvic acids $Đ \approx 1.4–2.2$; industrial kraft lignin $Đ > 3.5–8.0$.
* **$HMW/LMW$ Ratio:** $\frac{\text{Frac I} + \text{Frac II}}{\text{Frac IV}}$ — primary discriminator of anthropogenic lignin pollution.
""")

    with st.sidebar:
        st.markdown(f"### {T[lang]['sec_sidebar_hdr']}")
        sec_src = st.radio(
            T[lang]["sec_src_label"],
            [T[lang]["sec_src_upload"], T[lang]["sec_src_synth"]],
            index=0,
            key="sec_data_source_radio"
        )

        if st.session_state.get("_prev_sec_src") != sec_src:
            st.session_state["_prev_sec_src"] = sec_src
            st.session_state["sec_samples"] = {}

        # Настройки хроматографической колонки
        with st.expander("⚙️ " + ("Параметры колонки и калибровки" if lang == "ru" else "Column & Calibration Settings"), expanded=False):
            calib_preset_choices = list(sec_core.CALIBRATION_PRESETS.keys())
            calib_preset = st.selectbox(
                T[lang]["sec_calib_preset_label"],
                calib_preset_choices,
                index=0,
                format_func=lambda k: sec_core.CALIBRATION_PRESETS[k]["name_ru" if lang == "ru" else "name_en"],
                key="sec_calib_preset_select"
            )
            flow_rate = st.number_input(
                T[lang]["sec_flow_rate_label"],
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="sec_flow_rate_input"
            )

        if sec_src == T[lang]["sec_src_upload"]:
            up_sec_files = st.file_uploader(
                T[lang]["sec_up_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="sec_up_files"
            )
            if up_sec_files:
                for uf in up_sec_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_SEC", "").replace("_sec", "")
                    try:
                        s_obj = sec_core.parse_sec_file(
                            b_content,
                            sample_id=s_id,
                            flow_rate_ml_min=flow_rate,
                            calibration_preset=calib_preset
                        )
                        st.session_state["sec_samples"][s_id] = s_obj
                    except Exception as e_sec:
                        st.warning(f"SEC error ({uf.name}): {e_sec}")
        else:
            if not st.session_state["sec_samples"]:
                demo_sec_dir = os.path.join("demo_data", "sec")
                if os.path.exists(demo_sec_dir) and any(f.endswith(".csv") for f in os.listdir(demo_sec_dir)):
                    all_files = os.listdir(demo_sec_dir)
                    sample_map = {}
                    for fn in all_files:
                        if not fn.endswith(".csv") or fn == "sec_descriptors_matrix.csv":
                            continue
                        fpath = os.path.join(demo_sec_dir, fn)
                        with open(fpath, "rb") as f:
                            b_content = f.read()
                        s_id = fn.replace("_SEC.csv", "").replace(".csv", "")
                        sample_map[s_id] = sec_core.parse_sec_file(
                            b_content,
                            sample_id=s_id,
                            flow_rate_ml_min=flow_rate,
                            calibration_preset=calib_preset
                        )
                    st.session_state["sec_samples"] = sample_map
                else:
                    synth_samples, _ = sec_core.generate_synthetic_sec_dataset(n_samples_per_class=6, save_dir=demo_sec_dir)
                    st.session_state["sec_samples"] = synth_samples

    if not st.session_state["sec_samples"]:
        st.info(T[lang]["sec_no_samples"])
    else:
        # Сборка матрицы дескрипторов по всем образцам
        sec_rows = []
        for s_id, s_obj in st.session_state["sec_samples"].items():
            row = {"Sample_ID": s_id}
            row.update(sec_core.extract_sec_descriptors(s_obj))
            row["Class_Target"] = 1 if any(k in str(s_id).lower() for k in ["lignin", "sludge", "шлам"]) else 0
            sec_rows.append(row)
        df_sec_desc = pd.DataFrame(sec_rows)

        sample_list = list(st.session_state["sec_samples"].keys())
        col_sel1, col_sel2 = st.columns([3, 2])
        with col_sel1:
            active_s_id = st.selectbox(
                T[lang]["sec_active_sample"],
                sample_list,
                index=0,
                key="select_active_sec_sample"
            )
        active_sample = st.session_state["sec_samples"][active_s_id]

        with col_sel2:
            mw_val = active_sample.indices.get("SEC_Mw", 0.0)
            pdi_val = active_sample.indices.get("SEC_Polydispersity", active_sample.indices.get("SEC_PDI", 1.0))
            n_peaks = len(active_sample.peaks)
            st.markdown(
                f"<div style='padding-top: 28px;'>"
                f"<span style='background: #008080; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Mw: {mw_val:,.0f} Da</span>"
                f"<span style='background: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Đ: {pdi_val:.2f}</span>"
                f"<span style='background: #2980b9; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;'>Peaks: {n_peaks}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        tab_sec1, tab_sec2, tab_sec3, tab_sec4 = st.tabs([
            T[lang]["sec_tab1_title"],
            T[lang]["sec_tab2_title"],
            T[lang]["sec_tab3_title"],
            T[lang]["sec_tab4_title"],
        ])

        # ----------------------------------------------------------------------
        # ТАБ 1: Хроматограмма элюирования и пики
        # ----------------------------------------------------------------------
        with tab_sec1:
            st.write(f"#### {T[lang]['sec_tab1_title']} — **{active_s_id}**")

            col_c1, col_c2 = st.columns([2, 2])
            with col_c1:
                x_axis_mode = st.radio(
                    T[lang]["sec_xaxis_label"],
                    [T[lang]["sec_xaxis_vol"], T[lang]["sec_xaxis_time"]],
                    horizontal=True,
                    key="sec_xaxis_toggle"
                )
            use_vol = (x_axis_mode == T[lang]["sec_xaxis_vol"])

            x_data = active_sample.elution_volume if use_vol else active_sample.retention_times
            x_title = "Elution Volume Ve (mL)" if use_vol else "Retention Time tR (min)"
            if lang == "ru":
                x_title = "Элюционный объем Ve (мл)" if use_vol else "Время удерживания tR (мин)"

            fig_chrom = go.Figure()

            # Сырой сигнал
            if len(active_sample.signal_raw) > 0 and len(x_data) == len(active_sample.signal_raw):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.signal_raw,
                    mode="lines",
                    line=dict(color="#95a5a6", width=1.2, dash="dash"),
                    name="Raw Signal (UV/RI)" if lang == "en" else "Сырой сигнал",
                ))

            # Базовая линия ALS
            if active_sample.baseline is not None and len(active_sample.baseline) == len(x_data):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.baseline,
                    mode="lines",
                    line=dict(color="#e74c3c", width=1.0, dash="dot"),
                    name="ALS Baseline" if lang == "en" else "Базовая линия ALS",
                ))

            # Скорректированный профиль
            if len(active_sample.signal_corrected) > 0 and len(x_data) == len(active_sample.signal_corrected):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.signal_corrected,
                    mode="lines",
                    line=dict(color="#008080", width=2.4),
                    name="Corrected Signal" if lang == "en" else "Скорректированный сигнал",
                ))

            # Идентифицированные пики
            fraction_palette = {
                "HMW (>20 kDa)": "#e74c3c",
                "MHMW (5-20 kDa)": "#e67e22",
                "MMW (1-5 kDa)": "#3498db",
                "LMW (<1 kDa)": "#2ecc71",
            }

            if active_sample.peaks:
                for p in active_sample.peaks:
                    p_x = p.elution_volume if use_vol else p.retention_time
                    p_color = fraction_palette.get(p.fraction_name, "#16a085")
                    fig_chrom.add_trace(go.Scatter(
                        x=[p_x],
                        y=[p.height],
                        mode="markers+text",
                        marker=dict(size=10, color=p_color, symbol="diamond", line=dict(width=1, color="black")),
                        text=[f"P{p.peak_id}: {p.peak_mw:,.0f} Da"],
                        textposition="top center",
                        name=p.fraction_name,
                        hoverinfo="text",
                        hovertext=(
                            f"Peak {p.peak_id} ({p.fraction_name})<br>"
                            f"Ve: {p.elution_volume:.3f} mL | tR: {p.retention_time:.2f} min<br>"
                            f"Mp: {p.peak_mw:,.1f} Da<br>"
                            f"Area: {p.area:.2f} ({p.area_pct:.1f}%)"
                        )
                    ))

            fig_chrom.update_layout(
                xaxis_title=x_title,
                yaxis_title="Detector Response (mAU / mV)" if lang == "en" else "Отклик детектора (mAU / mV)",
                template="plotly_white",
                height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st_plotly(fig_chrom)

            # Таблица пиков
            st.markdown(f"##### {T[lang]['sec_peaks_hdr']}")
            if active_sample.peaks:
                peak_records = []
                for p in active_sample.peaks:
                    peak_records.append({
                        "Peak_ID": p.peak_id,
                        "Retention_Time_min": p.retention_time,
                        "Elution_Volume_ml": p.elution_volume,
                        "Peak_MW_Mp_Da": p.peak_mw,
                        "Peak_Height": p.height,
                        "Peak_Area": p.area,
                        "Area_pct": p.area_pct,
                        "Fraction": p.fraction_name,
                    })
                df_peaks = pd.DataFrame(peak_records)
                st_df(df_peaks)

                st.download_button(
                    label=T[lang]["sec_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_s_id}_SEC_Peaks.csv",
                    mime="text/csv",
                    key=f"dl_sec_peaks_{active_s_id}",
                )
            else:
                st.info("Пики не обнаружены выше установленного порога высоты." if lang == "ru" else "No peaks detected above the threshold.")

        # ----------------------------------------------------------------------
        # ТАБ 2: Молекулярно-массовое распределение (ММР / MWD)
        # ----------------------------------------------------------------------
        with tab_sec2:
            st.write(f"#### {T[lang]['sec_tab2_title']} — **{active_s_id}**")

            if len(active_sample.log_mw) > 0 and len(active_sample.mwd_diff) > 0:
                fig_mwd = make_subplots(specs=[[{"secondary_y": True}]])

                # Цветовые зоны фракций по молекулярным массам
                fig_mwd.add_vrect(x0=float(np.log10(20000)), x1=float(np.log10(200000)), fillcolor="#e74c3c", opacity=0.08, layer="below", line_width=0,
                                  annotation_text="Frac I: HMW (>20 kDa)" if lang == "en" else "Фракция I: HMW (>20 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(5000)), x1=float(np.log10(20000)), fillcolor="#e67e22", opacity=0.10, layer="below", line_width=0,
                                  annotation_text="Frac II: MHMW (5-20 kDa)" if lang == "en" else "Фракция II: MHMW (5–20 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(1000)), x1=float(np.log10(5000)), fillcolor="#3498db", opacity=0.08, layer="below", line_width=0,
                                  annotation_text="Frac III: MMW (1-5 kDa)" if lang == "en" else "Фракция III: MMW (1–5 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(200)), x1=float(np.log10(1000)), fillcolor="#2ecc71", opacity=0.10, layer="below", line_width=0,
                                  annotation_text="Frac IV: LMW (<1 kDa)" if lang == "en" else "Фракция IV: LMW (<1 кДа)", annotation_position="top left")

                # Дифференциальная кривая ММР w(log M)
                fig_mwd.add_trace(
                    go.Scatter(
                        x=active_sample.log_mw,
                        y=active_sample.mwd_diff,
                        mode="lines",
                        line=dict(color="#008080", width=2.6),
                        name="Differential MWD w(log M)" if lang == "en" else "Дифференциальное ММР w(log M)",
                    ),
                    secondary_y=False
                )

                # Интегральная кривая ММР W(M)
                if len(active_sample.mwd_cum) == len(active_sample.log_mw):
                    fig_mwd.add_trace(
                        go.Scatter(
                            x=active_sample.log_mw,
                            y=active_sample.mwd_cum,
                            mode="lines",
                            line=dict(color="#8e44ad", width=2.0, dash="dash"),
                            name="Cumulative MWD W(M) %" if lang == "en" else "Интегральное ММР W(M) %",
                        ),
                        secondary_y=True
                    )

                # Линии средних масс: Mn, Mw, Mz, Mp
                mn_val = active_sample.indices.get("SEC_Mn", 0.0)
                mw_val = active_sample.indices.get("SEC_Mw", 0.0)
                mz_val = active_sample.indices.get("SEC_Mz", 0.0)
                mp_val = active_sample.indices.get("SEC_Mp", 0.0)

                if mn_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mn_val, 1.0))), line_width=1.5, line_dash="dash", line_color="#2980b9",
                                      annotation_text=f"Mn={mn_val:,.0f}", annotation_position="bottom left")
                if mw_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mw_val, 1.0))), line_width=2.0, line_dash="solid", line_color="#c0392b",
                                      annotation_text=f"Mw={mw_val:,.0f}", annotation_position="top left")
                if mz_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mz_val, 1.0))), line_width=1.5, line_dash="dot", line_color="#8e44ad",
                                      annotation_text=f"Mz={mz_val:,.0f}", annotation_position="top right")

                fig_mwd.update_xaxes(
                    title_text="log₁₀(Molecular Weight, Da)" if lang == "en" else "log₁₀(Молекулярная масса, Да)",
                )
                fig_mwd.update_yaxes(title_text="Differential Weight w(log M)" if lang == "en" else "Дифференциальная доля w(log M)", secondary_y=False)
                fig_mwd.update_yaxes(title_text="Cumulative Fraction (%)" if lang == "en" else "Кумулятивная доля (%)", secondary_y=True, range=[0, 105])
                fig_mwd.update_layout(template="plotly_white", height=470, margin=dict(l=40, r=40, t=30, b=40))
                st_plotly(fig_mwd)
            else:
                st.info("Недостаточно данных для расчета кривой ММР." if lang == "ru" else "Insufficient data to compute MWD curve.")

        # ----------------------------------------------------------------------
        # ТАБ 3: Полидисперсность и фракционирование
        # ----------------------------------------------------------------------
        with tab_sec3:
            st.write(f"#### {T[lang]['sec_tab3_title']}")

            col_km1, col_km2, col_km3, col_km4 = st.columns(4)
            col_km1.metric(T[lang]["sec_kpi_mn"], f"{active_sample.indices.get('SEC_Mn', 0.0):,.0f} Da", help="Среднечисловая масса")
            col_km2.metric(T[lang]["sec_kpi_mw"], f"{active_sample.indices.get('SEC_Mw', 0.0):,.0f} Da", help="Среднемассовая масса")
            col_km3.metric(T[lang]["sec_kpi_mz"], f"{active_sample.indices.get('SEC_Mz', 0.0):,.0f} Da", help="z-Средняя масса")
            col_km4.metric(T[lang]["sec_kpi_mp"], f"{active_sample.indices.get('SEC_Mp', 0.0):,.0f} Da", help="Пиковая масса главного пика")

            col_kp1, col_kp2, col_kp3, col_kp4 = st.columns(4)
            col_kp1.metric(T[lang]["sec_kpi_pdi"], f"{active_sample.indices.get('SEC_Polydispersity', active_sample.indices.get('SEC_PDI', 1.0)):.2f}", help="Đ = Mw / Mn")
            col_kp2.metric(T[lang]["sec_kpi_hmw_ratio"], f"{active_sample.indices.get('SEC_HMW_LMW_ratio', 0.0):.2f}", help="(Frac I + II) / Frac IV")
            col_kp3.metric(T[lang]["sec_kpi_frac1"], f"{active_sample.indices.get('SEC_Fraction_I_pct', 0.0):.1f}%", help="Высокомолекулярная фракция (>20 кДа)")
            col_kp4.metric(T[lang]["sec_kpi_frac4"], f"{active_sample.indices.get('SEC_Fraction_IV_pct', 0.0):.1f}%", help="Низкомолекулярная фракция (<1 кДа)")

            st.markdown("---")

            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                st.write("##### " + ("Распределение молекулярных фракций (Активный образец)" if lang == "ru" else "Molecular Fraction Distribution (Active Sample)"))
                labels_frac = ["HMW (>20 kDa)", "MHMW (5-20 kDa)", "MMW (1-5 kDa)", "LMW (<1 kDa)"]
                vals_frac = [
                    active_sample.indices.get("SEC_Fraction_I_pct", 0.0),
                    active_sample.indices.get("SEC_Fraction_II_pct", 0.0),
                    active_sample.indices.get("SEC_Fraction_III_pct", 0.0),
                    active_sample.indices.get("SEC_Fraction_IV_pct", 0.0),
                ]
                fig_donut_sec = go.Figure(data=[go.Pie(
                    labels=labels_frac,
                    values=vals_frac,
                    hole=0.45,
                    marker=dict(colors=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]),
                )])
                fig_donut_sec.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
                st_plotly(fig_donut_sec)

            with col_diag2:
                st.write("##### " + ("Сравнение фракций по образцам сессии" if lang == "ru" else "Fraction Comparison Across All Samples"))
                if len(df_sec_desc) > 0:
                    fig_bar_frac = go.Figure()
                    fig_bar_frac.add_trace(go.Bar(name="HMW (>20 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_I_pct"], marker_color="#e74c3c"))
                    fig_bar_frac.add_trace(go.Bar(name="MHMW (5-20 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_II_pct"], marker_color="#e67e22"))
                    fig_bar_frac.add_trace(go.Bar(name="MMW (1-5 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_III_pct"], marker_color="#3498db"))
                    fig_bar_frac.add_trace(go.Bar(name="LMW (<1 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_IV_pct"], marker_color="#2ecc71"))
                    fig_bar_frac.update_layout(barmode="stack", height=360, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20),
                                               yaxis_title="Fraction Share (%)" if lang == "en" else "Доля фракции (%)")
                    st_plotly(fig_bar_frac)

        # ----------------------------------------------------------------------
        # ТАБ 4: Мультимодальная интеграция и экспорт
        # ----------------------------------------------------------------------
        with tab_sec4:
            st.write(f"#### {T[lang]['sec_tab4_title']}")

            if len(df_sec_desc) > 1 and "SEC_Mw" in df_sec_desc.columns and "SEC_Polydispersity" in df_sec_desc.columns:
                df_plot_sec = df_sec_desc.copy()
                df_plot_sec["_size"] = df_plot_sec["SEC_Fraction_I_pct"].apply(lambda v: max(float(v), 4.0) if pd.notnull(v) else 8.0)
                fig_scatter_sec = px.scatter(
                    df_plot_sec,
                    x="SEC_Mw",
                    y="SEC_Polydispersity",
                    text="Sample_ID" if "Sample_ID" in df_plot_sec.columns else None,
                    size="_size",
                    color="Class_Target" if "Class_Target" in df_plot_sec.columns else None,
                    color_continuous_scale="Teal",
                    title="Molecular Weight (Mw) vs Polydispersity Index (Đ)" if lang == "en" else "Молекулярная масса (Mw) vs Полидисперсность (Đ)",
                    labels={
                        "SEC_Mw": "Weight-Average MW Mw (Da)",
                        "SEC_Polydispersity": "Polydispersity Index Đ (Mw/Mn)",
                        "Class_Target": "Class"
                    }
                )
                fig_scatter_sec.update_layout(height=420, template="plotly_white")
                st_plotly(fig_scatter_sec)

            st.write("##### " + ("Сводная матрица дескрипторов SEC/GPC по всем пробам" if lang == "ru" else "Summary Matrix of SEC/GPC Descriptors Across All Samples"))
            st_df(df_sec_desc)

            c_sd1, c_sd2, c_sd3 = st.columns([1.5, 1.5, 2.5])
            with c_sd1:
                st.download_button(
                    label=T[lang]["sec_dl_csv"],
                    data=df_sec_desc.to_csv(index=False).encode("utf-8"),
                    file_name="ChemoSuite_SEC_Descriptors.csv",
                    mime="text/csv",
                    key="dl_sec_descriptors_csv",
                )
            with c_sd2:
                excel_sec_buf = io.BytesIO()
                with pd.ExcelWriter(excel_sec_buf, engine="openpyxl") as writer:
                    df_sec_desc.to_excel(writer, index=False, sheet_name="SEC_Descriptors")
                st.download_button(
                    label=T[lang]["sec_dl_excel"],
                    data=excel_sec_buf.getvalue(),
                    file_name="ChemoSuite_SEC_Descriptors.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_sec_descriptors_excel",
                )
            with c_sd3:
                if st.button(T[lang]["sec_sync_ml_btn"], type="primary", key="btn_sync_sec_ml_descriptors"):
                    st.session_state["sec_ml_descriptors"] = df_sec_desc
                    st.toast(T[lang]["sec_sync_success"].format(n=len(df_sec_desc)), icon="✅")
                    st.rerun()

# ==============================================================================
# МОДУЛЬ 8: ЭЛЕМЕНТНЫЙ АНАЛИЗ И ВАЛОВАЯ СТЕХИОМЕТРИЯ (CHNS-O)
# ==============================================================================
elif active_module == T[lang]["mod8_name"]:
    if not ELEMENTAL_CORE_AVAILABLE or elemental_core is None:
        st.error(T[lang]["elem_err_missing"])
        st.stop()

    st.title(T[lang]["elem_title"])
    st.caption(T[lang]["elem_subtitle"])

    with st.sidebar:
        st.header(T[lang]["elem_sidebar_hdr"])
        elem_data_src = st.radio(
            T[lang]["elem_src_label"],
            [T[lang]["elem_src_upload"], T[lang]["elem_src_synth"]],
            index=0,
            key="elem_data_source_radio",
        )

        if st.session_state.get("_prev_elem_data_src") != elem_data_src:
            st.session_state["_prev_elem_data_src"] = elem_data_src
            st.session_state["elemental_samples"] = {}

        if elem_data_src == T[lang]["elem_src_upload"]:
            up_elem_file = st.file_uploader(
                T[lang]["elem_up_label"],
                type=["csv", "tsv", "txt"],
                key="elem_uploader_file",
            )
            if up_elem_file is not None:
                samples_list = elemental_core.parse_elemental_table(up_elem_file.getvalue())
                st.session_state["elemental_samples"] = {s.sample_id: s for s in samples_list}
        else:
            if not st.session_state.get("elemental_samples"):
                bench_df = elemental_core.generate_synthetic_elemental_dataset()
                samples_list = elemental_core.parse_elemental_table(bench_df)
                st.session_state["elemental_samples"] = {s.sample_id: s for s in samples_list}

    curr_elem_samples = st.session_state.get("elemental_samples", {})
    if not curr_elem_samples:
        st.info(T[lang]["elem_no_samples"])
        st.stop()

    sample_ids = list(curr_elem_samples.keys())
    col_sel1, col_sel2 = st.columns([2.5, 3.5])
    with col_sel1:
        active_s_id = st.selectbox(
            T[lang]["elem_active_sample"],
            sample_ids,
            index=0,
            key="elem_active_sample_select",
        )
    active_elem_sample = curr_elem_samples[active_s_id]

    with col_sel2:
        elem_tb_html = (
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 26px;">'
            f'<span class="sample-chip"><b>H/C</b>: {active_elem_sample.h_c:.3f}</span>'
            f'<span class="sample-chip"><b>O/C</b>: {active_elem_sample.o_c:.3f}</span>'
            f'<span class="sample-chip green"><b>NOSC</b>: {active_elem_sample.nosc:+.2f}</span>'
            f'<span class="sample-chip orange"><b>AI_bulk</b>: {active_elem_sample.aromaticity_index:.3f}</span>'
            f'</div>'
        )
        st.markdown(elem_tb_html, unsafe_allow_html=True)

    tab_el1, tab_el2, tab_el3, tab_el4 = st.tabs([
        T[lang]["elem_tab1_title"],
        T[lang]["elem_tab2_title"],
        T[lang]["elem_tab3_title"],
        T[lang]["elem_tab4_title"],
    ])

    # ----------------------------------------------------------------------
    # ТАБ 1: Валовая стехиометрия и индексы
    # ----------------------------------------------------------------------
    with tab_el1:
        st.subheader(f"{T[lang]['elem_tab1_title']} — {active_s_id}")
        c_kpi1, c_kpi2, c_kpi3, c_kpi4, c_kpi5 = st.columns(5)
        c_kpi1.metric(T[lang]["elem_kpi_hc"], f"{active_elem_sample.h_c:.3f}")
        c_kpi2.metric(T[lang]["elem_kpi_oc"], f"{active_elem_sample.o_c:.3f}")
        c_kpi3.metric(T[lang]["elem_kpi_nc"], f"{active_elem_sample.n_c:.4f}")
        c_kpi4.metric(T[lang]["elem_kpi_sc"], f"{active_elem_sample.s_c:.4f}")
        c_kpi5.metric(T[lang]["elem_kpi_nosc"], f"{active_elem_sample.nosc:+.3f}")

        st.markdown("---")
        st.write("#### " + ("Весовой элементный состав образца (% масс.):" if lang == "ru" else "Mass Elemental Composition (% wt):"))
        elem_weights_df = pd.DataFrame([{
            "Element / Component": "Углерод (C)" if lang == "ru" else "Carbon (C)", "Mass %": active_elem_sample.c_pct,
        }, {
            "Element / Component": "Водород (H)" if lang == "ru" else "Hydrogen (H)", "Mass %": active_elem_sample.h_pct,
        }, {
            "Element / Component": "Азот (N)" if lang == "ru" else "Nitrogen (N)", "Mass %": active_elem_sample.n_pct,
        }, {
            "Element / Component": "Сера (S)" if lang == "ru" else "Sulfur (S)", "Mass %": active_elem_sample.s_pct,
        }, {
            "Element / Component": "Кислород (O, расчет)" if lang == "ru" else "Oxygen (O, by diff)", "Mass %": active_elem_sample.o_pct,
        }, {
            "Element / Component": "Зольность (Ash)" if lang == "ru" else "Ash Content", "Mass %": active_elem_sample.ash_pct,
        }, {
            "Element / Component": "Влажность (Moisture)" if lang == "ru" else "Moisture", "Mass %": active_elem_sample.moisture_pct,
        }])
        st_df(elem_weights_df)

        fig_bar = px.bar(
            elem_weights_df.head(5),
            x="Element / Component",
            y="Mass %",
            color="Element / Component",
            color_discrete_sequence=["#1b4965", "#62b6cb", "#bee9e8", "#ffb703", "#e63946"],
            title=("Баланс элементов CHNS-O" if lang == "ru" else "CHNS-O Elemental Mass Balance"),
        )
        fig_bar.update_layout(height=380, plot_bgcolor="white", showlegend=False)
        st_plotly(fig_bar)

    # ----------------------------------------------------------------------
    # ТАБ 2: Валовая диаграмма Ван Кревелена (Bulk Van Krevelen)
    # ----------------------------------------------------------------------
    with tab_el2:
        st.subheader(T[lang]["elem_tab2_title"])
        all_elem_records = [s.to_dict() for s in curr_elem_samples.values()]
        df_all_elem = pd.DataFrame(all_elem_records)

        fig_vk_bulk = px.scatter(
            df_all_elem,
            x="O_C",
            y="H_C",
            color="Sample_Type",
            size="C_pct",
            hover_name="Sample_ID",
            hover_data=["NOSC", "AI_bulk", "S_C"],
            text="Sample_ID",
            title=("Валовая диаграмма Ван Кревелена образцов (H/C vs O/C)" if lang == "ru" else "Bulk Van Krevelen Diagram (H/C vs O/C)"),
            labels={"O_C": "O/C (Атомное отношение)", "H_C": "H/C (Атомное отношение)"},
        )
        fig_vk_bulk.add_shape(type="rect", x0=0.0, y0=1.5, x1=0.3, y1=2.2, fillcolor="#a8dadc", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.add_shape(type="rect", x0=0.2, y0=0.7, x1=0.6, y1=1.5, fillcolor="#e9c46a", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.add_shape(type="rect", x0=0.6, y0=0.8, x1=1.0, y1=1.8, fillcolor="#f4a261", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.update_traces(textposition="top center")
        fig_vk_bulk.update_layout(height=560, plot_bgcolor="white")
        st_plotly(fig_vk_bulk)

    # ----------------------------------------------------------------------
    # ТАБ 3: Кросс-валидация с FT-ICR MS
    # ----------------------------------------------------------------------
    with tab_el3:
        st.subheader(f"{T[lang]['elem_tab3_title']} — {active_s_id}")
        spectra_db = st.session_state.get("spectra_db", {})

        matched_spec_name = None
        for s_name in spectra_db:
            if active_s_id.lower() in s_name.lower() or s_name.lower() in active_s_id.lower():
                matched_spec_name = s_name
                break
        if matched_spec_name is None and spectra_db:
            matched_spec_name = list(spectra_db.keys())[0]

        spec_options = list(spectra_db.keys()) if spectra_db else []
        if not spec_options:
            st.info("В памяти сессии нет спектров FT-ICR MS с приписанными формулами. Загрузите спектр в Модуле 1 для кросс-валидации." if lang == "ru" else "No FT-ICR MS spectra available in session. Upload a spectrum in Module 1 to perform cross-validation.")
        else:
            def_idx = spec_options.index(matched_spec_name) if matched_spec_name in spec_options else 0
            sel_spec = st.selectbox(
                "Сопоставляемый спектр FT-ICR MS:" if lang == "ru" else "Matching FT-ICR MS Spectrum:",
                spec_options,
                index=def_idx,
                key="elem_crossval_spec_select",
            )
            spec_obj = spectra_db[sel_spec]
            as_df = spec_obj.get("assigned_df")

            if as_df is None or as_df.empty:
                st.warning("Для выбранного спектра еще не выполнено приписывание брутто-формул. Перейдите в Модуль 1 (Этап 2) и нажмите 'Приписать формулы'." if lang == "ru" else "No formulas assigned for selected spectrum. Please run formula assignment in Module 1 first.")
            else:
                cval_res = elemental_core.cross_validate_fticr_elemental(as_df, active_elem_sample)
                st.success(f"✓ **Concordance Score (Сходимость методов): {cval_res['concordance_score']}%** — {cval_res['interpretation']}")

                c_cv1, c_cv2 = st.columns(2)
                with c_cv1:
                    st.write("##### " + ("Элементный микроанализ (CHNS-O bulk):" if lang == "ru" else "Elemental Microanalysis (CHNS-O bulk):"))
                    st_df(pd.DataFrame([cval_res["elemental_bulk"]]))
                with c_cv2:
                    st.write("##### " + ("Средневзвешенное по FT-ICR MS (ESI-):" if lang == "ru" else "Weighted FT-ICR MS Average (ESI-):"))
                    st_df(pd.DataFrame([cval_res["ms_weighted"]]))

                st.write("##### " + ("Невязки и расхождения (MS - CHNS):" if lang == "ru" else "Residuals (MS - CHNS):"))
                st_df(pd.DataFrame([cval_res["differences"]]))

    # ----------------------------------------------------------------------
    # ТАБ 4: Экспорт дескрипторов и синхронизация ML
    # ----------------------------------------------------------------------
    with tab_el4:
        st.subheader(T[lang]["elem_tab4_title"])
        all_elem_records = [s.to_dict() for s in curr_elem_samples.values()]
        df_elem_desc = pd.DataFrame(all_elem_records)
        st_df(df_elem_desc)

        c_ed1, c_ed2, c_ed3 = st.columns([1.5, 1.5, 2.5])
        with c_ed1:
            st.download_button(
                label=T[lang]["elem_dl_csv"],
                data=df_elem_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_Elemental_CHNS_Descriptors.csv",
                mime="text/csv",
                key="dl_elem_desc_csv",
            )
        with c_ed2:
            excel_elem_buf = io.BytesIO()
            with pd.ExcelWriter(excel_elem_buf, engine="openpyxl") as writer:
                df_elem_desc.to_excel(writer, index=False, sheet_name="Elemental_CHNS")
            st.download_button(
                label=T[lang]["elem_dl_excel"],
                data=excel_elem_buf.getvalue(),
                file_name="ChemoSuite_Elemental_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_elem_desc_excel",
            )
        with c_ed3:
            if st.button(T[lang]["elem_sync_ml_btn"], type="primary", key="btn_sync_elem_ml_descriptors"):
                st.session_state["elemental_ml_descriptors"] = df_elem_desc
                st.toast(T[lang]["elem_sync_success"].format(n=len(df_elem_desc)), icon="✅")
                st.rerun()






