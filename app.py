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

# ==============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ И СТИЛИЗАЦИЯ
# ==============================================================================
st.set_page_config(
    page_title="ChemoSuite Platform | FT-ICR MS, EEM & Chemometrics",
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
# СЛОВАРЬ ПОЛНОЙ ДВУЯЗЫЧНОЙ ЛОКАЛИЗАЦИИ (I18N)
# ==============================================================================
T = {
    "ru": {
        # Платформа
        "header_title": "⚗️ ChemoSuite: Анализ РОВ и Шлам-лигнина",
        "header_subtitle": "Кафедра аналитической химии & Лаборатория природных гуминовых систем химфака МГУ",
        "nav_title": "### ⚗️ Платформа ChemoSuite",
        "nav_module_label": "Аналитический модуль:",
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
        "ml_src_benchmark": "🧪 Синтетический бенчмарк",
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
        "ml_perm_chk": "Запустить пермутационный тест валидации (50 итераций)",
        "ml_tab_scores": "🎯 Проекция Scores (t₁ vs t₂)",
        "ml_tab_vips": "📊 Маркеры (VIP Scores)",
        "ml_tab_val": "📈 Валидация и диагностика",
        "ml_tab_blocks": "🧬 Вклад аналитических блоков",
        "ml_cm_title": "Матрица ошибок (Confusion Matrix)",
        "ml_perm_title": "Распределение пермутационного теста (H0: модель случайна)",
        "ml_donut_title": "Относительный вес аналитических блоков в модели (VIP²)",
        "ml_warn_two_classes": "⚠️ Для классификации PLS-DA в колонке Class_Target должны присутствовать оба класса (0 и 1)!",
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
    },
    "en": {
        # Platform
        "header_title": "⚗️ ChemoSuite: NOM & Slime-Lignin Analysis",
        "header_subtitle": "Department of Analytical Chemistry & Humic Systems Laboratory, Chemistry Faculty, MSU",
        "nav_title": "### ⚗️ ChemoSuite Platform",
        "nav_module_label": "Analytical Module:",
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
        "ml_src_benchmark": "🧪 Synthetic Benchmark",
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
        "ml_perm_chk": "Run permutation validation test (50 iterations)",
        "ml_tab_scores": "🎯 Scores Plot (t₁ vs t₂)",
        "ml_tab_vips": "📊 Biomarkers (VIP Scores)",
        "ml_tab_val": "📈 Validation & Diagnostics",
        "ml_tab_blocks": "🧬 Analytical Block Importance",
        "ml_cm_title": "Confusion Matrix",
        "ml_perm_title": "Permutation Test Distribution (H0: Random Model)",
        "ml_donut_title": "Relative Analytical Block Contribution (VIP²)",
        "ml_warn_two_classes": "⚠️ PLS-DA requires at least two distinct classes (0 and 1) in Class_Target!",
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
    calculate_descriptors,
    parse_uploaded_file,
    fast_formula_assigner,
    run_formula_assignment,
    compute_kmd,
    compute_vk20_grid,
    align_two_spectra_fast,
    perform_spectral_algebra,
    run_tmds_screening,
    get_calibrant_library,
)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ЯДРО FT-ICR MS
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
                st.session_state["spectra_db"][fname] = {
                    "raw_path": fpath, "file_bytes": content,
                    "parsed_peaks": None, "assigned_df": None, "raw_df": None,
                }
            except Exception:
                pass

if "eem_ml_descriptors" not in st.session_state:
    st.session_state["eem_ml_descriptors"] = None
if "uv_ml_descriptors" not in st.session_state:
    st.session_state["uv_ml_descriptors"] = None

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ: ВЫБОР МОДУЛЯ И ПАРАМЕТРОВ
# ==============================================================================
with st.sidebar:
    lang_choice = st.selectbox("🌐 Язык / Language", ["Русский", "English"], index=0)
    lang = "ru" if lang_choice == "Русский" else "en"
    st.session_state["lang"] = lang

    st.markdown(T[lang]["nav_title"])
    active_module = st.radio(
        T[lang]["nav_module_label"],
        [T[lang]["mod1_name"], T[lang]["mod2_name"], T[lang]["mod3_name"]],
        index=0,
    )
    st.markdown("---")

# Фирменный баннер платформы
st.markdown(
    f"""
    <div class="platform-header">
        <h2>{T[lang]["header_title"]}</h2>
        <p>{T[lang]["header_subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# МОДУЛЬ 1: NOM-SPECTRA FT-ICR MS STUDIO
# ==============================================================================
if active_module == T[lang]["mod1_name"]:
    with st.sidebar:
        st.title(T[lang]["sidebar_mgr"])
        if not NOMSPECTRA_INSTALLED:
            st.info(T[lang]["nomspectra_missing"])

        uploaded_files = st.file_uploader(
            T[lang]["uploader_label"], type=["csv", "txt", "tsv", "xy"], accept_multiple_files=True
        )
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state["spectra_db"]:
                    save_path = os.path.join(STORAGE_DIR, uf.name)
                    b_content = uf.getvalue()
                    with open(save_path, "wb") as f: f.write(b_content)
                    st.session_state["spectra_db"][uf.name] = {
                        "raw_path": save_path, "file_bytes": b_content,
                        "parsed_peaks": None, "assigned_df": None, "raw_df": None,
                    }

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
                                    st.session_state["spectra_db"][file] = {
                                        "raw_path": dst_path, "file_bytes": content,
                                        "parsed_peaks": None, "assigned_df": None, "raw_df": None,
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
            search_query = st.text_input(T[lang]["search_label"], placeholder=T[lang]["search_ph"]).strip().lower()
            available_spectra = [s for s in all_spectra if search_query in s.lower()] if search_query else all_spectra

            if not available_spectra:
                st.warning(T[lang]["search_empty"])
                active_spectrum_name, current_sample = None, None
            else:
                active_spectrum_name = st.selectbox(T[lang]["active_sample"], options=available_spectra, index=0, key="active_spectrum_selector")
                current_sample = st.session_state["spectra_db"][active_spectrum_name]

                if st.button(T[lang]["del_sample"], type="secondary"):
                    if os.path.exists(current_sample["raw_path"]):
                        try: os.remove(current_sample["raw_path"])
                        except Exception: pass
                    del st.session_state["spectra_db"][active_spectrum_name]
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
                    if raw_df.empty:
                        st.error("Error: File is empty or failed to parse." if lang == "en" else "Ошибка: Файл пуст или не распознан.")
                        st.stop()

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
                        st.error("Error: No valid numeric data found." if lang == "en" else "Ошибка: Нет корректных числовых данных.")
                        st.stop()

                    min_m_data, max_m_data = float(valid_df["mass"].min()), float(valid_df["mass"].max())
                    mz_range = st.slider(T[lang]["mz_range"], min_value=max(50.0, float(np.floor(min_m_data))),
                                         max_value=min(2500.0, float(np.ceil(max_m_data))),
                                         value=(max(100.0, float(np.floor(min_m_data))), min(1200.0, float(np.ceil(max_m_data)))),
                                         step=10.0, key=f"mzrange_{active_spectrum_name}")

                    max_int_data = float(valid_df["intensity"].max())
                    cutoff_intensity = st.number_input(T[lang]["cutoff_int"], min_value=0.0, max_value=max_int_data, value=0.0,
                                                       step=max_int_data * 0.001 if max_int_data > 0 else 1.0, format="%.2e", key=f"cutoff_{active_spectrum_name}")

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

                    current_sample["parsed_peaks"] = filtered_df
                    st.success(T[lang]["loaded_peaks_success"].format(n=len(filtered_df)))
                except Exception as err:
                    st.error(f"Error parsing file: {err}")
                    st.stop()
        else:
            active_spectrum_name, current_sample = None, None
            st.info(T[lang]["no_spectra_info"])

    st.title(T[lang]["title"])
    tabs = st.tabs([
        T[lang]["tab_stick"], T[lang]["tab_recal"], T[lang]["tab_assign"],
        T[lang]["tab_vk"], T[lang]["tab_kmd"], T[lang]["tab_vk20"],
        T[lang]["tab_cmp"], T[lang]["tab_tmds"], T[lang]["tab_desc"],
    ])

    # 1. Stick Plot
    with tabs[0]:
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
    with tabs[1]:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_recal']}: {active_spectrum_name}")
            cal_opts = ["Fatty Acids (C12–C33 saturated FA)" if lang == "en" else "Жирные кислоты (C12–C33 насыщенные ЖК)",
                        "CHO Homologues (C_n H_{2n-8} O7)" if lang == "en" else "Гомологи CHO (C_n H_{2n-8} O7)"]
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

                if st.button(T[lang]["recal_btn"], type="primary", key=f"btn_recal_{active_spectrum_name}"):
                    current_sample["parsed_peaks"]["mass"] = peaks_df["mass"].values - poly_fn(peaks_df["mass"].values)
                    current_sample["assigned_df"] = None
                    st.success(T[lang]["recal_success"].format(n=len(peaks_df), before=calib_df["error_ppm"].abs().mean(), after=residual_ppm.abs().mean()))

    # 3. Приписывание формул
    with tabs[2]:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_assign']}: {active_spectrum_name}")
            with st.expander(T[lang]["assign_expander"], expanded=True):
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
            elif assigned_data is not None:
                st.warning(T[lang]["no_formulas_warn"])

    # 4. Проекции
    with tabs[3]:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is None or assigned_data.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            st.subheader(f"{T[lang]['tab_vk']}: {active_spectrum_name}")
            proj_type = st.radio(T[lang]["proj_mode"], [T[lang]["proj_vk"], T[lang]["proj_dbe_c"], T[lang]["proj_custom"]], horizontal=True, key=f"proj_type_{active_spectrum_name}")
            pow_exp = st.slider(T[lang]["pow_exp"], min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=f"pow_{active_spectrum_name}")
            palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}

            if proj_type == T[lang]["proj_vk"]:
                fig_vk = go.Figure()
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
            elif proj_type == T[lang]["proj_dbe_c"]:
                fig_dbe = go.Figure()
                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    if sub.empty: continue
                    fig_dbe.add_trace(go.Scattergl(x=sub["C"], y=sub["DBE"], mode="markers", name=f"{cls} ({len(sub):,})",
                                                   marker=dict(color=palette[cls], size=2.0 + 5.0 * (sub["norm_intensity"] / 100.0) ** pow_exp, opacity=0.65)))
                c_line = np.linspace(4, 60, 100)
                fig_dbe.add_trace(go.Scatter(x=c_line, y=c_line * 0.9, mode="lines", name="Planar limit", line=dict(color="red", dash="dot")))
                fig_dbe.update_layout(xaxis=dict(title="Carbon (C)"), yaxis=dict(title="DBE"), plot_bgcolor="white", height=560)
                st_plotly(fig_dbe)
            else:
                avail_cols = ["mass", "intensity", "norm_intensity", "C", "H", "O", "N", "S", "H/C", "O/C", "DBE", "AI", "NOSC", "error_ppm"]
                c1, c2, c3 = st.columns(3)
                with c1: cx = st.selectbox("X:", avail_cols, index=avail_cols.index("mass"))
                with c2: cy = st.selectbox("Y:", avail_cols, index=avail_cols.index("DBE"))
                with c3: cc = st.selectbox("Color:", avail_cols, index=avail_cols.index("AI"))
                fig_cust = px.scatter(assigned_data, x=cx, y=cy, color=cc, color_continuous_scale="Viridis", opacity=0.7, render_mode="webgl")
                fig_cust.update_layout(height=560, plot_bgcolor="white")
                st_plotly(fig_cust)

    # 5. KMD
    with tabs[4]:
        st.subheader(f"{T[lang]['tab_kmd']}: {active_spectrum_name if active_spectrum_name else ''}")
        work_df = current_sample.get("assigned_df") if current_sample and current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty else (current_sample.get("parsed_peaks") if current_sample else None)
        if work_df is not None and not work_df.empty:
            c1, c2 = st.columns(2)
            with c1: base_choice = st.selectbox(T[lang]["kmd_base_group"], list(KMD_BASES.keys()), index=0, key=f"kmd_base_{active_spectrum_name}")
            with c2: point_sz = st.slider(T[lang]["kmd_pt_size"], 1, 8, 3, key=f"kmd_pt_{active_spectrum_name}")
            km, nkm, kmd = compute_kmd(work_df["mass"].values, base_choice)
            fig_kmd = go.Figure()
            even_mask = (nkm % 2 == 0)
            fig_kmd.add_trace(go.Scattergl(x=nkm[even_mask], y=kmd[even_mask], mode="markers", name=T[lang]["kmd_even"], marker=dict(color="#0020C2", size=point_sz, opacity=0.6)))
            fig_kmd.add_trace(go.Scattergl(x=nkm[~even_mask], y=kmd[~even_mask], mode="markers", name=T[lang]["kmd_odd"], marker=dict(color="#FF7F0E", size=point_sz, opacity=0.7)))
            fig_kmd.update_layout(xaxis=dict(title=T[lang]["kmd_x_axis"].format(base=base_choice), gridcolor="#f1f3f5"), yaxis=dict(title=T[lang]["kmd_y_axis"].format(base=base_choice), range=[-0.02, 1.02]), height=540, plot_bgcolor="white")
            st_plotly(fig_kmd)

    # 6. Хемотипирование 20 ячеек
    with tabs[5]:
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

    # 7. Сравнение
    with tabs[6]:
        st.subheader(T[lang]["tab_cmp"])
        if len(all_spectra) < 2:
            st.info(T[lang]["cmp_need_two"])
        else:
            c1, c2, c3 = st.columns([2, 2, 1.5])
            with c1: name_a = st.selectbox(T[lang]["cmp_spec_a"], all_spectra, index=0, key="cmp_spec_a")
            with c2: name_b = st.selectbox(T[lang]["cmp_spec_b"], all_spectra, index=1 if len(all_spectra) > 1 else 0, key="cmp_spec_b")
            with c3: tol_comp = st.number_input(T[lang]["cmp_tol"], min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="cmp_ppm_tol")
            peaks_a = st.session_state["spectra_db"][name_a].get("parsed_peaks")
            peaks_b = st.session_state["spectra_db"][name_b].get("parsed_peaks")
            if peaks_a is not None and peaks_b is not None and not peaks_a.empty and not peaks_b.empty:
                align_res = align_two_spectra_fast(peaks_a, peaks_b, ppm_tol=tol_comp)
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Peaks {name_a}", f"{align_res['n_a']:,}")
                m2.metric(f"Peaks {name_b}", f"{align_res['n_b']:,}")
                m3.metric(T[lang]["cmp_common"], f"{align_res['n_common']:,}", delta=f"Jaccard: {align_res['jaccard']:.3f}")

    # 8. TMDS
    with tabs[7]:
        st.subheader(f"{T[lang]['tab_tmds']}: {active_spectrum_name if active_spectrum_name else ''}")
        peaks_df = current_sample.get("parsed_peaks") if current_sample else None
        if peaks_df is not None and not peaks_df.empty:
            tmds_sum, _ = run_tmds_screening(peaks_df, top_n=1000, tol_mda=2.0)
            if not tmds_sum.empty:
                fig_tmds = px.bar(tmds_sum, x="Transformation", y="Count", color="Transformation", text=tmds_sum["Share_pct"].apply(lambda v: f"{v:.1f}%"))
                fig_tmds.update_layout(showlegend=False, xaxis_tickangle=-25, height=420)
                st_plotly(fig_tmds)

    # 9. Сводные характеристики
    with tabs[8]:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is not None and not assigned_data.empty:
            st.subheader(f"{T[lang]['tab_desc']}: {active_spectrum_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T[lang]["desc_mn"], f"{assigned_data['mass'].mean():.2f} Da")
            c2.metric("H/C (Mn)", f"{assigned_data['H/C'].mean():.3f}")
            c3.metric("O/C (Mn)", f"{assigned_data['O/C'].mean():.3f}")
            c4.metric(T[lang]["desc_ai"], f"{assigned_data['AI'].mean():.3f}")

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

    # Чтение EEM
    samples = []
    if eem_data_src == T[lang]["eem_src_synth"]:
        samples = eem_core.generate_synthetic_chemometrics_dataset(n_samples=10)
    elif eem_files:
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
                samples.append(parsed)
            except Exception as e:
                st.error(f"Error reading EEM {f.name}: {e}")

    # Чтение 1D УФ-Вид спектров
    uv_samples = []
    if uv_files:
        for uf in uv_files:
            try:
                u_name = uf.name.rsplit(".", 1)[0]
                parsed_uv = eem_core.parse_uv_vis_spectrum(uf, sample_id=u_name)
                parsed_uv.doc = default_doc_val if default_doc_val > 0 else None
                uv_samples.append(parsed_uv)
            except Exception as e:
                st.error(f"Error reading UV-Vis {uf.name}: {e}")

    # Автоматическая связка УФ-Вид спектров с EEM образцами
    if samples and uv_samples:
        doc_dict = {s.sample_id: default_doc_val for s in samples} if default_doc_val > 0 else None
        eem_core.link_uv_vis_to_eem(samples, uv_samples, doc_map=doc_dict)

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
                s_obj = next(s for s in samples if s.sample_id == selected_sname)

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
                    st.dataframe(
                        df_indices.style.format(
                            {"FI": "{:.2f}", "HIX": "{:.2f}", "A254": "{:.4f}", "DOC": "{:.2f}", "SUVA254": "{:.2f}"},
                            na_rep="—"
                        ),
                        use_container_width=True
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
                col_sel1, col_sel2 = st.columns([3, 1])
                with col_sel1:
                    selected_parafac_ids = st.multiselect(
                        T[lang]["eem_parafac_select_tables"],
                        options=all_sample_ids,
                        default=all_sample_ids,
                        key="parafac_active_tables_multiselect",
                    )
                with col_sel2:
                    n_components_sel = st.selectbox(
                        "Компонентов (R):" if lang == "ru" else "Components (R):",
                        [2, 3, 4],
                        index=1,
                        key="parafac_n_components_select"
                    )

                if len(selected_parafac_ids) < 2:
                    st.warning(T[lang]["eem_parafac_min_samples"])
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
                active_uv = next(u for u in uv_samples if u.sample_id == active_uv_name)

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
                    eem_core.calculate_uv_vis_indices(u, doc=default_doc_val if default_doc_val > 0 else None)
                    for u in uv_samples
                ]
                df_uv_indices = pd.DataFrame(uv_indices_list)
                st.session_state["uv_ml_descriptors"] = df_uv_indices

                st.markdown(f"##### {T[lang]['eem_uv_tbl_title']}")
                c_tbl1, c_tbl2 = st.columns([3, 1])
                with c_tbl1:
                    st.dataframe(
                        df_uv_indices.style.format({
                            "A254": "{:.4f}", "A280": "{:.4f}", "A365": "{:.4f}",
                            "E2_E3": "{:.2f}", "E4_E6": "{:.2f}",
                            "S_275_295": "{:.4f}", "S_350_400": "{:.4f}", "S_R": "{:.2f}",
                            "DOC": "{:.2f}", "SUVA254": "{:.2f}"
                        }, na_rep="—"),
                        use_container_width=True
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

    if not CHEMO_ML_AVAILABLE or chemo_ml is None:
        st.error(T[lang]["ml_err_missing"])
        st.stop()

    st.subheader(T[lang]["ml_sec1_title"])

    # Выбор источника данных
    data_source_mode = st.radio(
        "Источник мультимодальных данных:" if lang == "ru" else "Multimodal Data Source:",
        [T[lang]["ml_src_session"], T[lang]["ml_src_files"], T[lang]["ml_src_benchmark"]],
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

        st.info(T[lang]["ml_session_status"].format(
            n_ms=len(assigned_spectra),
            n_eem=len(eem_df) if eem_df is not None else 0,
            n_uv=len(uv_df) if uv_df is not None else 0,
        ))

        if st.button(T[lang]["ml_btn_assemble_session"], type="primary", key="btn_assemble_session"):
            if not assigned_spectra and (eem_df is None or eem_df.empty) and (uv_df is None or uv_df.empty):
                st.warning(T[lang]["ml_session_need_more"])
            else:
                fticr_rows = []
                for s_name, a_df in assigned_spectra.items():
                    desc = chemo_ml.extract_fticr_descriptors(a_df, s_name)
                    if desc:
                        fticr_rows.append(desc)
                df_ms_assembled = pd.DataFrame(fticr_rows) if fticr_rows else pd.DataFrame()

                merged_df = chemo_ml.merge_feature_blocks(df_ms_assembled, eem_df, df_uv=uv_df)
                if not merged_df.empty:
                    if "Class_Target" not in merged_df.columns:
                        # Разметка по умолчанию: 1 для образцов со словом lignin / impact
                        merged_df["Class_Target"] = merged_df["Sample_ID"].apply(
                            lambda s: 1 if any(k in str(s).lower() for k in ["lignin", "impact", "шлам"]) else 0
                        )
                    st.session_state["fused_data"] = merged_df
                    st.session_state["fused_source"] = "Сессионные спектры" if lang == "ru" else "Session Spectra"
                    st.success(f"Успешно собрано образцов: {len(merged_df)}" if lang == "ru" else f"Successfully assembled samples: {len(merged_df)}")
                    st.rerun()
                else:
                    st.warning("Не удалось сопоставить образцы между Модулем 1 и 2 по Sample_ID." if lang == "ru" else "Failed to match samples by Sample_ID between Module 1 and 2.")

    # 2. Загрузка внешних файлов
    elif data_source_mode == T[lang]["ml_src_files"]:
        upload_type = st.radio(
            T[lang]["ml_uploader_mode"],
            [T[lang]["ml_upload_two"], T[lang]["ml_upload_single"]],
            horizontal=True,
            key="ml_upload_type_radio"
        )
        if upload_type == T[lang]["ml_upload_two"]:
            col_u1, col_u2, col_u3 = st.columns(3)
            with col_u1:
                f_ms = st.file_uploader(T[lang]["ml_up_ms"], type=["csv", "tsv", "txt"], key="up_ms_features")
            with col_u2:
                f_eem = st.file_uploader(T[lang]["ml_up_eem"], type=["csv", "tsv", "txt"], key="up_eem_features")
            with col_u3:
                f_uv = st.file_uploader(T[lang]["ml_up_uv"], type=["csv", "tsv", "txt"], key="up_uv_features")

            if f_ms or f_eem or f_uv:
                try:
                    df_up_ms = pd.read_csv(f_ms) if f_ms else None
                    df_up_eem = pd.read_csv(f_eem) if f_eem else None
                    df_up_uv = pd.read_csv(f_uv) if f_uv else None
                    merged = chemo_ml.merge_feature_blocks(df_up_ms, df_up_eem, df_up_uv)
                    if not merged.empty:
                        if "Class_Target" not in merged.columns:
                            merged["Class_Target"] = 0
                        st.session_state["fused_data"] = merged
                        src_names = [f.name for f in [f_ms, f_eem, f_uv] if f is not None]
                        st.session_state["fused_source"] = " + ".join(src_names)
                except Exception as e:
                    st.error(f"Error reading files: {e}")
        else:
            f_single = st.file_uploader(T[lang]["ml_up_single"], type=["csv", "tsv", "txt"], key="up_single_features")
            if f_single:
                try:
                    df_s = pd.read_csv(f_single)
                    if "Class_Target" not in df_s.columns:
                        df_s["Class_Target"] = 0
                    st.session_state["fused_data"] = df_s
                    st.session_state["fused_source"] = f_single.name
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    # 3. Синтетический бенчмарк
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

        edited_df = st.data_editor(
            fused_df,
            column_config={
                "Class_Target": st.column_config.SelectboxColumn(
                    T[lang]["ml_col_class_target"],
                    options=[0, 1],
                    required=True,
                )
            },
            disabled=[c for c in fused_df.columns if c != "Class_Target"],
            use_container_width=True,
            key="ml_fused_editor",
        )

        st.markdown("---")
        st.subheader(T[lang]["ml_sec2_title"])

        max_allowed_lvs = max(2, min(5, len(edited_df) - 1))
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            strategy_choice = st.selectbox(
                T[lang]["ml_strategy_label"],
                [T[lang]["ml_strat_low"], T[lang]["ml_strat_mid"]],
                index=0,
                key="pls_strategy_select"
            )
            strat_code = "mid_level" if strategy_choice == T[lang]["ml_strat_mid"] else "low_level"
        with col_p2:
            n_lvs = st.slider(T[lang]["ml_lvs_slider"], min_value=2, max_value=max_allowed_lvs, value=2, key="pls_lvs_slider")
            use_perm = st.checkbox(T[lang]["ml_perm_chk"], value=True, key="pls_perm_chk")
        with col_p3:
            use_block_scale = st.checkbox(T[lang]["ml_block_scale"], value=True, key="pls_block_scale_chk")
            st.write("")
            run_pls = st.button(T[lang]["ml_run_btn"], type="primary", key="btn_run_pls_da")

        if run_pls:
            feat_cols = [c for c in edited_df.columns if c not in ["Sample_ID", "Class_Target"]]
            X_input = edited_df[feat_cols]
            y_input = edited_df["Class_Target"].values

            # Проверка наличия двух классов
            unique_classes = np.unique(y_input)
            if len(unique_classes) < 2:
                st.warning(T[lang]["ml_warn_two_classes"])
            else:
                try:
                    with st.spinner(T[lang]["ml_spinner"]):
                        res = chemo_ml.train_plsda_model(
                            X_input, y_input,
                            n_components=n_lvs,
                            fusion_strategy=strat_code,
                            block_scaling=use_block_scale
                        )
                        st.session_state["pls_results"] = res
                        st.session_state["pls_sample_ids"] = edited_df["Sample_ID"].values
                        st.session_state["pls_y_input"] = y_input

                        if use_perm:
                            perm_res = chemo_ml.run_permutation_test(
                                X_input, y_input,
                                n_components=n_lvs,
                                n_permutations=50,
                                fusion_strategy=strat_code,
                                block_scaling=use_block_scale
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
            t1_vals = res.get("t1", res.get("scores_t1"))
            t2_vals = res.get("t2", res.get("scores_t2"))
            ell_x = res.get("ell_x", res.get("ellipse_x", np.array([])))
            ell_y = res.get("ell_y", res.get("ellipse_y", np.array([])))
            vip_df = res.get("VIP_df", res.get("vip_df", pd.DataFrame()))
            perm_res = st.session_state.get("perm_results", None)

            st.markdown("---")
            st.subheader(T[lang]["ml_sec3_title"])

            # 4 ключевые сводные метрики
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(T[lang]["ml_m_r2x"], f"{res.get('R2X', 0.0):.1f}%")
            m2.metric(T[lang]["ml_m_r2y"], f"{res.get('R2Y', 0.0):.1f}%")
            q2_val = res.get("Q2", 0.0)
            m3.metric(T[lang]["ml_m_q2"], f"{q2_val:.1f}%", delta=T[lang]["ml_q2_ok"] if q2_val > 50 else T[lang]["ml_q2_warn"])
            m4.metric(T[lang]["ml_m_acc"], f"{res.get('Accuracy', 0.0):.1f}%")

            # Вкладки детального дашборда
            tab_ml1, tab_ml2, tab_ml3, tab_ml4, tab_ml5 = st.tabs([
                T[lang]["ml_tab_scores"],
                T[lang]["ml_tab_vips"],
                T[lang]["ml_tab_val"],
                T[lang]["ml_tab_blocks"],
                "💾 Экспорт отчетов" if lang == "ru" else "💾 Export Reports"
            ])

            # Таб 1: Scores Plot
            with tab_ml1:
                fig_sc = go.Figure()
                if ell_x is not None and len(ell_x) > 0:
                    fig_sc.add_trace(go.Scatter(
                        x=ell_x, y=ell_y, mode="lines",
                        line=dict(dash="dot", color="#7f7f7f", width=1.5),
                        name=T[lang]["ml_hotelling_name"], hoverinfo="skip"
                    ))

                colors = ["#D62728" if int(y) == 1 else "#0020C2" for y in y_curr]
                labels = [T[lang]["ml_class1_label"] if int(y) == 1 else T[lang]["ml_class0_label"] for y in y_curr]
                fig_sc.add_trace(go.Scatter(
                    x=t1_vals, y=t2_vals, mode="markers+text",
                    text=sample_ids_curr, textposition="top center",
                    marker=dict(size=11, color=colors, line=dict(width=1, color="black"), opacity=0.85),
                    hovertext=[f"<b>{s}</b><br>{lbl}<br>LV1: {t1:.2f}, LV2: {t2:.2f}" for s, lbl, t1, t2 in zip(sample_ids_curr, labels, t1_vals, t2_vals)],
                    hoverinfo="text", name="Samples",
                ))
                fig_sc.update_layout(xaxis_title="LV1", yaxis_title="LV2", plot_bgcolor="white", height=520)
                st_plotly(fig_sc)

            # Таб 2: VIP Scores с цветовой кодировкой аналитических блоков
            with tab_ml2:
                if not vip_df.empty:
                    top_vip = vip_df.head(15).copy()
                    fig_vip = px.bar(
                        top_vip, x="VIP", y="Descriptor", orientation="h",
                        color="Block",
                        color_discrete_map={"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "Other": "#7f7f7f"},
                        text=top_vip["VIP"].apply(lambda v: f"{v:.2f}")
                    )
                    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text=T[lang]["ml_vip_cutoff"])
                    fig_vip.update_layout(yaxis=dict(autorange="reversed", title="Descriptor"), xaxis=dict(title="VIP Score"), plot_bgcolor="white", height=520)
                    st_plotly(fig_vip)

                    st.dataframe(vip_df, use_container_width=True)

            # Таб 3: Валидация (Confusion Matrix + Permutation Test)
            with tab_ml3:
                c_v1, c_v2 = st.columns(2)
                with c_v1:
                    st.write(f"**{T[lang]['ml_cm_title']}**")
                    cm_mat = res.get("Confusion_Matrix", np.zeros((2, 2)))
                    fig_cm = px.imshow(
                        cm_mat,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[T[lang]["ml_class0_label"], T[lang]["ml_class1_label"]],
                        y=[T[lang]["ml_class0_label"], T[lang]["ml_class1_label"]],
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
                    st_plotly(fig_cm)

                    st.markdown(
                        f"""
                        * **Чувствительность (Sensitivity):** `{res.get('Sensitivity', 0.0):.1f}%`
                        * **Специфичность (Specificity):** `{res.get('Specificity', 0.0):.1f}%`
                        * **Сбалансированная точность:** `{res.get('Balanced_Accuracy', 0.0):.1f}%`
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
                            opacity=0.75
                        ))
                        fig_perm.add_vline(x=q2_o, line_width=2.5, line_dash="dash", line_color="#d62728", annotation_text=f"Q²={q2_o:.1f}%")
                        fig_perm.update_layout(xaxis_title="Q² (%)", yaxis_title="Frequencies", height=360, plot_bgcolor="white", margin=dict(l=20, r=20, t=30, b=20))
                        st_plotly(fig_perm)

                        if p_val < 0.05:
                            st.success(f"✅ Модель статистически значима: эмпирический p-value = **{p_val:.4f}** (< 0.05)." if lang == "ru" else f"✅ Model is statistically significant: empirical p-value = **{p_val:.4f}** (< 0.05).")
                        else:
                            st.warning(f"⚠️ Риск оверфиттинга: эмпирический p-value = **{p_val:.4f}** (≥ 0.05)." if lang == "ru" else f"⚠️ Overfitting risk: empirical p-value = **{p_val:.4f}** (≥ 0.05).")

            # Таб 4: Вклад аналитических блоков
            with tab_ml4:
                block_contribs = res.get("Block_Contributions", {})
                if block_contribs:
                    st.write(f"**{T[lang]['ml_donut_title']}**")
                    b_palette = {"FT-ICR MS": "#0b5394", "EEM-PARAFAC": "#e69138", "UV-Vis": "#2ca02c", "Other": "#7f7f7f"}
                    donut_colors = [b_palette.get(k, "#7f7f7f") for k in block_contribs.keys()]
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=list(block_contribs.keys()),
                        values=list(block_contribs.values()),
                        hole=0.45,
                        marker=dict(colors=donut_colors)
                    )])
                    fig_donut.update_layout(height=420)
                    st_plotly(fig_donut)

            # Таб 5: Экспорт
            with tab_ml5:
                st.write("### 📥 Выгрузка аналитических отчетов" if lang == "ru" else "### 📥 Download Analytical Reports")
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
                        file_name="ChemoSuite_PLSDA_VIP_Biomarkers.csv",
                        mime="text/csv",
                        key="dl_pls_vip_csv_main_btn",
                    )