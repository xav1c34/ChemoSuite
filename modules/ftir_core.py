"""
ftir_core.py — Математическое ядро колебательной спектроскопии (ATR-FTIR и Raman).
Методология хемометрического анализа функционального состава природного органического вещества
(РОВ/DOM, гуминовые кислоты) и техногенного шлам-лигнина.
"""
from dataclasses import dataclass, field
import io
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import scipy.signal
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class FTIRSample:
    """Контейнер для отдельного колебательного спектра (ИК-Фурье или КР)."""
    sample_id: str
    wavenumbers: np.ndarray  # Волновые числа, см^-1 (отсортированы по возрастанию)
    absorbance: np.ndarray   # Оптическая плотность A
    raw_signal: np.ndarray   # Исходный зарегистрированный сигнал
    signal_type: str = "absorbance"  # "absorbance" или "transmittance"
    baseline: Optional[np.ndarray] = None
    corrected_absorbance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def decode_bytes(b: bytes) -> str:
    """Универсальное декодирование байтов файла с поддержкой распространенных кодировок."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ==============================================================================
# ПАРСИНГ СПЕКТРОВ (CSV, DPT, TXT, XY)
# ==============================================================================
# Метки столбца/заголовка, указывающие на пропускание (%T) или поглощение (A)
_TRANSMITTANCE_LABEL_RE = re.compile(r"%\s*T\b|\bT\s*%|transmitt|transmission|пропуск", re.IGNORECASE)
_ABSORBANCE_LABEL_RE = re.compile(r"absorb|\babs\b|поглощ|оптическ", re.IGNORECASE)


def transmittance_to_absorbance(values: np.ndarray, percent: bool = True) -> Tuple[np.ndarray, int]:
    """
    A = −log10(T) (T в долях) или A = 2 − log10(%T). Без сдвига минимума к нулю.
    Значения T ≤ 0 (A → ∞) ограничиваются снизу 1e-3 % / 1e-5; возвращает (A, число ограниченных точек).
    """
    t = np.asarray(values, dtype=np.float64)
    floor = 1e-3 if percent else 1e-5
    n_clipped = int(np.sum(t < floor))
    t_c = np.maximum(t, floor)
    absorb = (2.0 - np.log10(t_c)) if percent else -np.log10(t_c)
    return absorb, n_clipped


def parse_ftir_file(
    file_content: Union[str, bytes],
    sample_id: str = "FTIR_Sample",
    signal_type: str = "auto"
) -> FTIRSample:
    """
    Универсальный парсер текстовых спектров FTIR (.csv, .dpt, .txt, .xy).
    Поддерживает спектры приборов Bruker (OPUS), Thermo (OMNIC), PerkinElmer, Shimadzu.

    Автоматически:
    1. Игнорирует текстовые заголовки и метаданные.
    2. Извлекает числовые столбцы (волновое число и интенсивность).
    3. Сортирует волновые числа по строгому возрастанию.

    Пересчет пропускания в поглощение выполняется ЯВНО:
    - signal_type="transmittance": A = 2 − log10(%T) (или −log10(T) для долей 0–1);
    - signal_type="absorbance": значения используются как есть;
    - signal_type="auto" (по умолчанию): пересчет только если заголовок/метаданные файла
      указывают на пропускание (%T, Transmittance, …) И все значения лежат в 0–110.
      Значения 0–100 без такой метки остаются поглощением; в metadata["warnings"] добавляется
      предупреждение (данные могут быть %T, Raman-счетами и т.п.).
    Поглощение не сдвигается так, чтобы минимум был 0.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    lines = text.strip().splitlines()
    mode = (signal_type or "auto").strip().lower()
    if mode not in ("auto", "absorbance", "transmittance"):
        raise ValueError(f"signal_type must be 'auto', 'absorbance' or 'transmittance', got '{signal_type}'")

    data_rows: List[Tuple[float, float]] = []
    header_lines: List[str] = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith(("#", ";", "/", "!", "Spectrum", "Date", "Wavenumber")):
            if len(header_lines) < 200:
                header_lines.append(line_clean)
            continue
        # Определение разделителя колонок
        if ";" in line_clean:
            tokens = [p.strip() for p in line_clean.split(";") if p.strip()]
        elif "\t" in line_clean:
            tokens = [p.strip() for p in line_clean.split("\t") if p.strip()]
        elif "," in line_clean:
            # Если запятая используется как разделитель столбцов
            tokens = [p.strip() for p in line_clean.split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in line_clean.split() if p.strip()]

        if len(tokens) >= 2:
            try:
                # Если в токене нет точки, но есть запятая — это европейская десятичная дробь (напр. "1510,5" -> "1510.5")
                t0 = tokens[0].replace(",", ".") if "." not in tokens[0] else tokens[0]
                t1 = tokens[1].replace(",", ".") if "." not in tokens[1] else tokens[1]
                wn = float(t0)
                val = float(t1)
                data_rows.append((wn, val))
            except ValueError:
                if len(header_lines) < 200:
                    header_lines.append(line_clean)
                continue
        elif len(header_lines) < 200:
            header_lines.append(line_clean)

    if not data_rows:
        raise ValueError(f"Не удалось извлечь спектральные данные из файла '{sample_id}'. Требуются 2 числовые колонки (см^-1 и интенсивность).")

    arr = np.array(data_rows, dtype=np.float64)
    # Сортируем по волновому числу
    arr = arr[np.argsort(arr[:, 0])]

    # Удаляем возможные дубликаты волновых чисел усреднением
    unique_wn, inverse_indices = np.unique(np.round(arr[:, 0], 2), return_inverse=True)
    accum_vals = np.zeros_like(unique_wn)
    counts = np.zeros_like(unique_wn)
    np.add.at(accum_vals, inverse_indices, arr[:, 1])
    np.add.at(counts, inverse_indices, 1)
    averaged_vals = accum_vals / np.maximum(counts, 1)

    wn_vec = unique_wn
    raw_vec = averaged_vals

    median_val = float(np.median(raw_vec))
    max_val = float(np.max(raw_vec))
    min_val = float(np.min(raw_vec))
    header_text = "\n".join(header_lines)
    label_says_t = bool(_TRANSMITTANCE_LABEL_RE.search(header_text))
    label_says_a = bool(_ABSORBANCE_LABEL_RE.search(header_text))
    in_t_range = min_val >= 0.0 and max_val <= 110.0
    warnings: List[str] = []

    if mode == "transmittance":
        is_t = True
        source = "explicit"
        if min_val < 0.0 or max_val > 110.0:
            raise ValueError(
                f"'{sample_id}': values {min_val:g}…{max_val:g} are not a transmittance scale (expected 0–100 %T or 0–1)"
            )
    elif mode == "absorbance":
        is_t = False
        source = "explicit"
    else:
        is_t = label_says_t and not label_says_a and in_t_range
        source = "header" if is_t else "default"
        if not is_t and in_t_range and median_val > 10.0 and not label_says_a:
            warnings.append(
                "Values lie in 0–100 without a transmittance/absorbance label; treated as absorbance. "
                "Select 'transmittance' explicitly if this is a %T spectrum."
            )

    if is_t:
        signal_type_out = "transmittance"
        percent = max_val > 1.5  # 0–1 → доли, иначе проценты
        absorbance, n_clipped = transmittance_to_absorbance(raw_vec, percent=percent)
        if n_clipped:
            warnings.append(f"{n_clipped} point(s) with T ≤ 0 were limited (absorbance capped)")
    else:
        signal_type_out = "absorbance"
        absorbance = raw_vec.copy()

    return FTIRSample(
        sample_id=sample_id,
        wavenumbers=wn_vec,
        absorbance=absorbance,
        raw_signal=raw_vec,
        signal_type=signal_type_out,
        metadata={
            "raw_points": len(data_rows),
            "median_raw": median_val,
            "max_raw": max_val,
            "signal_type_source": source,
            "warnings": warnings,
        }
    )


# ==============================================================================
# БАЗОВАЯ ЛИНИЯ И ПРЕДОБРАБОТКА (ALS, RUBBERBAND, SAVITZKY-GOLAY, SNV)
# ==============================================================================
def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10
) -> np.ndarray:
    """
    Асимметричные наименьшие квадраты (Asymmetric Least Squares, AsLS) по Eilers & Boelens (2005).
    Идеально удаляет наклонный фон рассеяния и флуоресценции в ИК-спектрах без деформации пиков.
    
    Параметры:
        lam: параметр гладкости (обычно 10^4 .. 10^7)
        p: асимметрия (0.001 .. 0.01 — штраф за отклонение выше базовой линии)
        niter: число итераций
    """
    L = len(y)
    if L < 4:
        return np.zeros_like(y)
        
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc", dtype=np.float64)
    D_TD = D.T.dot(D)
    w = np.ones(L)
    
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + lam * D_TD
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
        
    return np.asarray(z, dtype=np.float64)


def baseline_rubberband(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Коррекция базовой линии методом 'резиновой ленты' (нижняя выпуклая оболочка).
    Быстрый альтернативный метод для спектров с выраженным квазилинейным дрейфом.
    """
    n = len(y)
    if n < 3:
        return np.zeros_like(y)
        
    # Ищем локальные минимумы в скользящих сегментах
    n_seg = min(20, n // 5)
    seg_len = n // n_seg
    knot_x = [x[0]]
    knot_y = [y[0]]
    
    for i in range(n_seg):
        idx_start = i * seg_len
        idx_end = (i + 1) * seg_len if i < n_seg - 1 else n
        min_local = np.argmin(y[idx_start:idx_end])
        knot_x.append(x[idx_start + min_local])
        knot_y.append(y[idx_start + min_local])
        
    knot_x.append(x[-1])
    knot_y.append(y[-1])
    
    # Линейная интерполяция между узлами
    knot_x = np.array(knot_x)
    knot_y = np.array(knot_y)
    base = np.interp(x, knot_x, knot_y)
    return base


def subtract_baseline(
    sample: FTIRSample,
    method: str = "als",
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10
) -> Tuple[FTIRSample, np.ndarray]:
    """
    Вычитает базовую линию из спектра поглощения.
    Возвращает обновленный объект FTIRSample и массив базовой линии.
    """
    y = sample.absorbance
    x = sample.wavenumbers
    
    if method == "als":
        base = baseline_als(y, lam=lam, p=p, niter=niter)
    elif method == "rubberband":
        base = baseline_rubberband(x, y)
    elif method == "linear":
        # Простая линия между краями спектра
        base = np.linspace(y[0], y[-1], len(y))
    else:
        base = np.zeros_like(y)
        
    corr = y - base
    corr = np.maximum(corr, 0.0)  # неотрицательный сигнал после вычитания фона
    
    sample.baseline = base
    sample.corrected_absorbance = corr
    return sample, base


def smooth_savitzky_golay(
    y: np.ndarray,
    window_length: int = 15,
    polyorder: int = 2,
    deriv: int = 0
) -> np.ndarray:
    """
    Сглаживание и производная спектроскопия (d^1, d^2) фильтром Савицкого — Голея.
    """
    if len(y) < 5:
        return y.copy()
    w = max(5, window_length)
    if w % 2 == 0:
        w += 1
    if w >= len(y):
        w = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
    w = max(5, w)
    poly = min(polyorder, w - 1)
    
    return scipy.signal.savgol_filter(y, window_length=w, polyorder=poly, deriv=deriv)


def normalize_spectrum(
    y: np.ndarray,
    method: str = "snv"
) -> np.ndarray:
    """
    Нормировка спектра колебаний:
    - 'snv': Standard Normal Variate ((y - mean) / std)
    - 'minmax': линейное масштабирование в диапазон [0, 1]
    - 'vector': деление на L2-норму вектора (векторная нормировка)
    - 'area': нормировка на площадь под спектром (интеграл = 1)
    - 'max': нормировка на максимальный пик (макс = 1)
    """
    if len(y) == 0:
        return y
    std = np.std(y)
    
    if method == "snv":
        return (y - np.mean(y)) / (std if std > 1e-12 else 1.0)
    elif method == "minmax":
        min_v = np.min(y)
        max_v = np.max(y)
        span = max_v - min_v
        return (y - min_v) / (span if span > 1e-12 else 1.0)
    elif method == "vector":
        norm = np.linalg.norm(y)
        return y / (norm if norm > 1e-12 else 1.0)
    elif method == "area":
        area = np.sum(np.abs(y))
        return y / (area if area > 1e-12 else 1.0)
    elif method == "max":
        mx = np.max(np.abs(y))
        return y / (mx if mx > 1e-12 else 1.0)
    return y.copy()


def crop_spectral_range(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    min_wn: float = 650.0,
    max_wn: float = 4000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Обрезает спектр под заданный рабочий диапазон волновых чисел."""
    mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
    if not np.any(mask):
        return wavenumbers, absorbance
    return wavenumbers[mask], absorbance[mask]


# ==============================================================================
# ИЗВЛЕЧЕНИЕ АНАЛИТИЧЕСКИХ ДЕСКРИПТОРОВ И ИНДЕКСОВ (РОВ И ЛИГНИН)
# ==============================================================================
def get_intensity_at(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    target_wn: float,
    window: float = 12.0
) -> float:
    """
    Ищет максимум интенсивности в окрестности целевого волнового числа ± window см^-1.
    Компенсирует аппаратурный сдвиг полос и эффекты матричных перегибов.
    """
    mask = (wavenumbers >= target_wn - window) & (wavenumbers <= target_wn + window)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(wavenumbers - target_wn)))
        return float(absorbance[idx])
    return float(np.max(absorbance[mask]))


def integrate_band_area(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    wn_start: float,
    wn_end: float
) -> float:
    """Интегрирует площадь под спектром в интервале [wn_start, wn_end] методом Симпсона."""
    low, high = min(wn_start, wn_end), max(wn_start, wn_end)
    mask = (wavenumbers >= low) & (wavenumbers <= high)
    if np.sum(mask) < 3:
        return 0.0
    return float(simpson(absorbance[mask], x=wavenumbers[mask]))


def estimate_noise_sigma(absorbance: np.ndarray) -> float:
    """
    Робастная оценка шума σ по второй разности: для белого шума Var(Δ²y) = 6σ²,
    σ ≈ 1.4826·MAD(Δ²y)/√6. Медленно меняющиеся полосы на вторую разность почти не влияют.
    """
    y = np.asarray(absorbance, dtype=np.float64)
    if len(y) < 5:
        return 0.0
    d2 = np.diff(y, n=2)
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return 1.4826 * mad / math.sqrt(6.0)


def band_detection_threshold(
    absorbance: np.ndarray,
    snr_k: float = 3.0,
    min_band_fraction: float = 0.01
) -> float:
    """
    Порог «полоса присутствует»: max(snr_k·σ_шума, min_band_fraction·max|A|).
    Относительный порог нужен для гладких (синтетических / сильно сглаженных) спектров с σ ≈ 0.
    """
    y = np.asarray(absorbance, dtype=np.float64)
    if len(y) == 0:
        return float("inf")
    peak = float(np.max(np.abs(y)))
    return max(snr_k * estimate_noise_sigma(y), min_band_fraction * peak)


def calculate_ftir_indices(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    sample_id: str = "Sample",
    snr_k: float = 3.0,
    min_band_fraction: float = 0.01
) -> Dict[str, Union[str, float]]:
    """
    Вычисляет полный набор хемометрических дескрипторов ИК-спектроскопии
    для природного органического вещества (РОВ/DOM) и шлам-лигнина.

    Ожидается поглощение ПОСЛЕ коррекции базовой линии и БЕЗ SNV/min-max: эти нормировки
    аффинны (y → (y − m)/s), и отношения полос I_a/I_b при них не сохраняются.

    Ключевые полосы:
    - 3400 см^-1: OH / NH (спирты, фенолы, водородные связи)
    - 2960 см^-1: CH3 асимметричные (разветвления алифатики)
    - 2920 см^-1: CH2 асимметричные (алифатические углеводородные цепи)
    - 2850 см^-1: CH2 симметричные
    - 1720 см^-1: C=O карбонилы/карбоксилы (карбоновые кислоты, кетоны)
    - 1650 см^-1: сопряженные C=O, хиноиды, амид I
    - 1600 см^-1: C=C ароматические скелетные колебания
    - 1510 см^-1: скелетные колебания ароматического кольца лигнина (C=C; все H/G/S-звенья,
      у G-богатых лигнинов интенсивнее — Faix 1991, Holzforschung 45, 21; Pandey 1999)
    - 1450 см^-1: деформационные колебания CH2 / CH3
    - 1330 см^-1: сирингильное кольцо (S) + конденсированные G-звенья
    - 1270 см^-1: гваяцильное кольцо (G) + C=O валентные
    - 1220 см^-1: C-C / C-O эфирные связи лигнина
    - 1030 см^-1: C-O спиртов, полисахариды, целлюлоза, метоксилы

    Отношения возвращаются как NaN, если полоса-знаменатель ниже порога обнаружения
    band_detection_threshold (snr_k·σ_шума или min_band_fraction от максимума спектра):
    при отсутствующей полосе отношение не имеет смысла (напр., G/S → 70 при отсутствии 1330 см^-1).
    """
    i_3400 = get_intensity_at(wavenumbers, absorbance, 3400.0, window=30.0)
    i_2960 = get_intensity_at(wavenumbers, absorbance, 2960.0, window=10.0)
    i_2920 = get_intensity_at(wavenumbers, absorbance, 2920.0, window=10.0)
    i_2850 = get_intensity_at(wavenumbers, absorbance, 2850.0, window=10.0)
    i_1720 = get_intensity_at(wavenumbers, absorbance, 1720.0, window=15.0)
    i_1650 = get_intensity_at(wavenumbers, absorbance, 1650.0, window=15.0)
    i_1600 = get_intensity_at(wavenumbers, absorbance, 1605.0, window=15.0)
    i_1510 = get_intensity_at(wavenumbers, absorbance, 1510.0, window=10.0)
    i_1450 = get_intensity_at(wavenumbers, absorbance, 1450.0, window=12.0)
    i_1330 = get_intensity_at(wavenumbers, absorbance, 1330.0, window=10.0)
    i_1270 = get_intensity_at(wavenumbers, absorbance, 1270.0, window=10.0)
    i_1220 = get_intensity_at(wavenumbers, absorbance, 1220.0, window=12.0)
    i_1030 = get_intensity_at(wavenumbers, absorbance, 1030.0, window=15.0)

    threshold = band_detection_threshold(absorbance, snr_k=snr_k, min_band_fraction=min_band_fraction)

    def _ratio(num: float, den: float) -> float:
        if not (np.isfinite(num) and np.isfinite(den)) or den <= threshold:
            return float("nan")
        return float(num / den)

    # 1. Индекс ароматичности (Aromaticity Index)
    # Отношение ароматического скелета к алифатике
    aromaticity_index = _ratio(i_1600, i_2920)

    # 2. Индекс окисленности / карбоксилирования (Carbonylation Index)
    # Отношение карбоксилов к ароматике
    oxidation_index = _ratio(i_1720, i_1600)

    # 3. Индекс алифатического разветвления (CH3 / CH2)
    branching_index = _ratio(i_2960, i_2920)

    # 4. Лигниновый индекс: скелетная ароматика лигнина (1510) к общему ароматическому пулу (1600)
    lignin_index = _ratio(i_1510, i_1600)

    # 5. Ориентировочное отношение G/S (1270 / 1330); NaN, если полоса 1330 не обнаружена
    gs_ratio = _ratio(i_1270, i_1330)

    # 6. Полисахаридный / углеводный индекс
    polysaccharide_index = _ratio(i_1030, i_1600)

    # 7. Фенольно-эфирный индекс
    phenolic_ether_index = _ratio(i_1220, i_1600)

    # Интегральные площади ключевых областей
    area_oh = integrate_band_area(wavenumbers, absorbance, 3000.0, 3600.0)
    area_aliph = integrate_band_area(wavenumbers, absorbance, 2800.0, 3000.0)
    area_carbonyl_arom = integrate_band_area(wavenumbers, absorbance, 1500.0, 1800.0)
    area_fingerprint = integrate_band_area(wavenumbers, absorbance, 900.0, 1500.0)
    area_poly = integrate_band_area(wavenumbers, absorbance, 950.0, 1180.0)

    # Отношение площадей (Carbonyl/Aromatic Area to Aliphatic Area); NaN, если полоса CH2 2920 не обнаружена
    area_ratio_carb_aliph = (
        float(area_carbonyl_arom / area_aliph)
        if i_2920 > threshold and area_aliph > 0.0 else float("nan")
    )

    return {
        "Sample_ID": sample_id,
        "FTIR_Aromaticity_Idx": round(aromaticity_index, 4),
        "FTIR_Oxidation_Idx": round(oxidation_index, 4),
        "FTIR_Branching_Idx": round(branching_index, 4),
        "FTIR_Lignin_Idx": round(lignin_index, 4),
        "FTIR_GS_Ratio": round(gs_ratio, 4),
        "FTIR_Polysaccharide_Idx": round(polysaccharide_index, 4),
        "FTIR_Phenolic_Idx": round(phenolic_ether_index, 4),
        "FTIR_I_3400_OH": round(i_3400, 4),
        "FTIR_I_2920_CH2": round(i_2920, 4),
        "FTIR_I_1720_COOH": round(i_1720, 4),
        "FTIR_I_1600_Arom": round(i_1600, 4),
        "FTIR_I_1510_Lignin": round(i_1510, 4),
        "FTIR_I_1030_Poly": round(i_1030, 4),
        "FTIR_Area_OH": round(area_oh, 2),
        "FTIR_Area_Aliphatic": round(area_aliph, 2),
        "FTIR_Area_Carbonyl_Aromatic": round(area_carbonyl_arom, 2),
        "FTIR_Area_Polysaccharide": round(area_poly, 2),
        "FTIR_Area_Ratio_Carb_Aliph": round(area_ratio_carb_aliph, 4),
    }


# ==============================================================================
# МАТЕМАТИЧЕСКАЯ ДЕКОНВОЛЮЦИЯ СЛОЖНЫХ ПОЛОС (1500-1800 см^-1)
# ==============================================================================
def gaussian_peak(x: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    """Единичная Гауссова полоса."""
    return amp * np.exp(-0.5 * ((x - center) / (sigma + 1e-12)) ** 2)


def multi_gaussian(x: np.ndarray, *params) -> np.ndarray:
    """Сумма K Гауссовых полос: параметры [amp1, center1, sigma1, amp2, center2, sigma2, ...]"""
    y = np.zeros_like(x)
    n_peaks = len(params) // 3
    for k in range(n_peaks):
        amp = params[3 * k]
        center = params[3 * k + 1]
        sigma = params[3 * k + 2]
        y += gaussian_peak(x, amp, center, sigma)
    return y


FTIR_DECONV_MIN_R2 = 0.95  # ниже — подгонка помечается как неудовлетворительная


def deconvolve_carbonyl_region(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    wn_min: float = 1500.0,
    wn_max: float = 1800.0,
    n_peaks: int = 5,
    custom_initial_peaks: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Деконволюция сложного карбонильно-ароматического диапазона 1500–1800 см^-1
    на сумму перекрывающихся Гауссовых полос методом нелинейной оптимизации.

    Идентифицируемые полосы по умолчанию:
    1. ~1510 см^-1: скелетные колебания ароматического кольца лигнина (C=C; все H/G/S-звенья)
    2. ~1605 см^-1: Ароматический каркас C=C (Aromatic C=C)
    3. ~1655 см^-1: Сопряженные карбонилы / хиноиды (Quinones / Conj C=O)
    4. ~1715 см^-1: Свободные карбоксилы (Carboxylic COOH)
    5. ~1740 см^-1: Сложные эфиры и лактоны (Esters / Lactones)

    Returns dict с ключами wavenumbers, observed, fitted, residuals, r2, rmse, peaks_df,
    individual_curves, templates, а также:
    - success (bool): curve_fit сошелся. При неудаче fitted/residuals/r2/rmse = NaN,
      числовые столбцы peaks_df = NaN, individual_curves = [] — начальные приближения
      НЕ выдаются за результат подгонки.
    - fit_ok (bool): success, R² ≥ FTIR_DECONV_MIN_R2 и ни один центр/ширина не на границе.
    - message (str), warnings (list[str]); столбец peaks_df["At_Bound"].
    """
    mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
    if np.sum(mask) < 20:
        raise ValueError(f"Недостаточно спектральных точек в диапазоне {wn_min}–{wn_max} см^-1 для деконволюции.")

    sub_wn = wavenumbers[mask]
    sub_abs = absorbance[mask]

    # Базовая конфигурация полос по умолчанию
    default_templates = [
        {"name": "Aromatic skeletal C=C, lignin (1510)", "center": 1510.0, "c_min": 1495.0, "c_max": 1525.0, "sigma": 15.0},
        {"name": "Aromatic C=C (1605)", "center": 1605.0, "c_min": 1585.0, "c_max": 1625.0, "sigma": 18.0},
        {"name": "Quinone/Conj C=O (1655)", "center": 1655.0, "c_min": 1635.0, "c_max": 1680.0, "sigma": 18.0},
        {"name": "Carboxylic COOH (1715)", "center": 1715.0, "c_min": 1695.0, "c_max": 1735.0, "sigma": 18.0},
        {"name": "Esters/Lactones (1740)", "center": 1740.0, "c_min": 1730.0, "c_max": 1765.0, "sigma": 15.0},
    ]

    if custom_initial_peaks and len(custom_initial_peaks) > 0:
        templates = custom_initial_peaks[:n_peaks]
    else:
        templates = default_templates[:min(n_peaks, len(default_templates))]

    p0: List[float] = []
    bounds_lower: List[float] = []
    bounds_upper: List[float] = []

    max_a = max(float(np.max(sub_abs)), 0.01)

    for t in templates:
        center_init = float(t.get("center", 1600.0))
        # Оценка амплитуды по сигналу в точке
        idx_c = int(np.argmin(np.abs(sub_wn - center_init)))
        amp_init = max(float(sub_abs[idx_c]) * 0.7, max_a * 0.1)
        sigma_init = float(t.get("sigma", 15.0))

        c_min = float(t.get("c_min", center_init - 15.0))
        c_max = float(t.get("c_max", center_init + 15.0))

        p0.extend([amp_init, center_init, sigma_init])
        bounds_lower.extend([0.0, c_min, 5.0])
        bounds_upper.extend([max_a * 2.5, c_max, 50.0])

    warnings: List[str] = []
    try:
        popt, _ = curve_fit(
            multi_gaussian,
            sub_wn,
            sub_abs,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        success = bool(np.all(np.isfinite(popt)))
        message = "" if success else "curve_fit returned non-finite parameters"
    except Exception as e:
        success = False
        message = f"Fit did not converge: {e}"

    names = [t.get("name", f"Band {float(t.get('center', 0.0)):.1f}") for t in templates]
    if not success:
        nan_arr = np.full_like(sub_abs, np.nan, dtype=float)
        peaks_df = pd.DataFrame([{
            "Peak_ID": f"Peak_{k+1}", "Assignment": names[k],
            "Center_cm1": np.nan, "Amplitude": np.nan, "Sigma_cm1": np.nan,
            "FWHM_cm1": np.nan, "Area": np.nan, "Relative_Area_pct": np.nan, "At_Bound": False,
        } for k in range(len(templates))])
        return {
            "success": False,
            "fit_ok": False,
            "message": message,
            "warnings": [message],
            "wavenumbers": sub_wn,
            "observed": sub_abs,
            "fitted": nan_arr,
            "residuals": nan_arr.copy(),
            "r2": float("nan"),
            "rmse": float("nan"),
            "peaks_df": peaks_df,
            "individual_curves": [],
            "templates": templates,
        }

    fitted_curve = multi_gaussian(sub_wn, *popt)
    residuals = sub_abs - fitted_curve

    # Расчет метрик качества
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((sub_abs - np.mean(sub_abs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / len(sub_abs))
    if not np.isfinite(r2) or r2 < FTIR_DECONV_MIN_R2:
        warnings.append(f"Low fit quality: R² = {r2:.3f} < {FTIR_DECONV_MIN_R2}")

    # Извлечение параметров индивидуальных пиков
    peak_rows = []
    individual_curves = []
    total_area = 0.0

    for k in range(len(templates)):
        amp = float(popt[3 * k])
        center = float(popt[3 * k + 1])
        sigma = float(popt[3 * k + 2])
        # Площадь под гауссианой: Area = Amp * sigma * sqrt(2 * pi)
        area = amp * sigma * math.sqrt(2.0 * math.pi)
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        fwhm = 2.35482 * sigma
        lo_c, hi_c = bounds_lower[3 * k + 1], bounds_upper[3 * k + 1]
        tol_c = 1e-3 * max(hi_c - lo_c, 1e-6)
        at_bound = bool(
            abs(center - lo_c) <= tol_c or abs(center - hi_c) <= tol_c
            or abs(sigma - 5.0) <= 1e-3 or abs(sigma - 50.0) <= 1e-3
        )
        if at_bound:
            warnings.append(f"'{names[k]}': centre or width at its bound — band not identifiable")

        total_area += area
        curve_k = gaussian_peak(sub_wn, amp, center, sigma)
        individual_curves.append(curve_k)

        peak_rows.append({
            "Peak_ID": f"Peak_{k+1}",
            "Assignment": names[k],
            "Center_cm1": round(center, 1),
            "Amplitude": round(amp, 4),
            "Sigma_cm1": round(sigma, 2),
            "FWHM_cm1": round(fwhm, 2),
            "Area": round(area, 4),
            "At_Bound": at_bound,
        })

    # Добавляем процент относительной площади
    for row in peak_rows:
        row["Relative_Area_pct"] = round((row["Area"] / total_area) * 100.0, 2) if total_area > 0 else float("nan")

    peaks_df = pd.DataFrame(peak_rows)
    fit_ok = bool(np.isfinite(r2) and r2 >= FTIR_DECONV_MIN_R2 and not any(r["At_Bound"] for r in peak_rows))

    return {
        "success": True,
        "fit_ok": fit_ok,
        "message": "" if fit_ok else "; ".join(warnings),
        "warnings": warnings,
        "wavenumbers": sub_wn,
        "observed": sub_abs,
        "fitted": fitted_curve,
        "residuals": residuals,
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "peaks_df": peaks_df,
        "individual_curves": individual_curves,
        "templates": templates
    }


# ==============================================================================
# ПАКЕТНАЯ ОБРАБОТКА И СИНТЕТИЧЕСКИЙ БЕНЧМАРК
# ==============================================================================
def batch_process_ftir_samples(
    samples: List[FTIRSample],
    baseline_method: str = "als",
    norm_method: str = "snv"
) -> pd.DataFrame:
    """
    Пакетная обработка списка образцов FTIR с извлечением дескрипторов
    в единый DataFrame для передачи в Модуль ML (Data Fusion).

    Индексы и интенсивности всегда рассчитываются по поглощению после коррекции базовой линии
    БЕЗ нормировки. SNV и min-max — аффинные преобразования, не сохраняющие отношения полос
    (напр., лигниновый индекс 0.207 → 0.056 на одном и том же спектре), а масштабирующие
    нормировки (vector/area/max) на отношения не влияют. Параметр norm_method сохранен для
    обратной совместимости и на дескрипторы не влияет; нормировку применяйте только к
    спектральным векторам для визуализации / ML (normalize_spectrum).
    """
    rows = []
    for s in samples:
        # Коррекция фона если еще не выполнена
        if s.corrected_absorbance is None:
            subtract_baseline(s, method=baseline_method)
        abs_to_use = s.corrected_absorbance if s.corrected_absorbance is not None else s.absorbance

        desc = calculate_ftir_indices(s.wavenumbers, abs_to_use, sample_id=s.sample_id)
        rows.append(desc)

    return pd.DataFrame(rows)


def generate_synthetic_ftir_dataset(
    n_samples: int = 12,
    noise_level: float = 0.008,
    random_state: int = 42
) -> Tuple[Dict[str, FTIRSample], pd.DataFrame]:
    """
    Генерирует согласованный синтетический набор эталонных спектров ATR-FTIR
    для двух контрастных экологических типов проб:
    - Baikal_Pristine (чистая вода оз. Байкал): высокая алифатика (2920), полисахариды (1030), минимальный лигнин (1510).
    - Sludge_Lignin (техногенный шлам-лигнин БЦБК): интенсивная скелетная ароматика лигнина (1510), карбоксилы (1720), эфирные связи (1220, 1270).
    """
    rng = np.random.RandomState(random_state)
    wn = np.linspace(650.0, 4000.0, 1676)  # шаг ~2 см^-1
    
    samples_dict: Dict[str, FTIRSample] = {}
    
    # Эталонные гауссианы для Байкала
    def baical_profile(w):
        bg = 0.05 + 0.15 * np.exp(-((w - 650.0) / 1200.0) ** 2)  # наклонный фон
        oh = 0.65 * np.exp(-0.5 * ((w - 3380.0) / 160.0) ** 2)
        ch_aliph = 0.45 * np.exp(-0.5 * ((w - 2924.0) / 35.0) ** 2) + 0.25 * np.exp(-0.5 * ((w - 2854.0) / 25.0) ** 2)
        cooh = 0.22 * np.exp(-0.5 * ((w - 1718.0) / 25.0) ** 2)
        arom = 0.30 * np.exp(-0.5 * ((w - 1605.0) / 30.0) ** 2)
        lignin_1510 = 0.06 * np.exp(-0.5 * ((w - 1510.0) / 15.0) ** 2)  # низкий фоновый уровень
        ch2_def = 0.18 * np.exp(-0.5 * ((w - 1450.0) / 20.0) ** 2)
        ether = 0.12 * np.exp(-0.5 * ((w - 1220.0) / 25.0) ** 2)
        poly = 0.55 * np.exp(-0.5 * ((w - 1035.0) / 60.0) ** 2)
        return bg + oh + ch_aliph + cooh + arom + lignin_1510 + ch2_def + ether + poly

    # Эталонные гауссианы для шлам-лигнина
    def lignin_profile(w):
        bg = 0.08 + 0.25 * np.exp(-((w - 650.0) / 1000.0) ** 2)
        oh = 0.50 * np.exp(-0.5 * ((w - 3400.0) / 150.0) ** 2)
        ch_aliph = 0.25 * np.exp(-0.5 * ((w - 2925.0) / 30.0) ** 2) + 0.15 * np.exp(-0.5 * ((w - 2850.0) / 20.0) ** 2)
        cooh = 0.60 * np.exp(-0.5 * ((w - 1715.0) / 25.0) ** 2)  # высокое окисление
        arom = 0.65 * np.exp(-0.5 * ((w - 1602.0) / 25.0) ** 2)  # высокая ароматичность
        lignin_1510 = 0.58 * np.exp(-0.5 * ((w - 1510.0) / 16.0) ** 2)  # колоссальный маркер шлам-лигнина!
        ch2_def = 0.22 * np.exp(-0.5 * ((w - 1455.0) / 20.0) ** 2)
        g_ring = 0.42 * np.exp(-0.5 * ((w - 1270.0) / 22.0) ** 2)   # гваяцильный пик
        ether = 0.45 * np.exp(-0.5 * ((w - 1220.0) / 25.0) ** 2)   # эфиры лигнина
        poly = 0.30 * np.exp(-0.5 * ((w - 1030.0) / 50.0) ** 2)
        return bg + oh + ch_aliph + cooh + arom + lignin_1510 + ch2_def + g_ring + ether + poly

    sample_list = []
    half = n_samples // 2
    
    for i in range(n_samples):
        is_lignin = i >= half
        base_func = lignin_profile if is_lignin else baical_profile
        prefix = "Sludge_Lignin" if is_lignin else "Baikal_Pristine"
        s_id = f"{prefix}_{i%half + 1:02d}"
        
        # Индивидуальная биологическая вариабельность ± 12%
        var_scale = rng.uniform(0.88, 1.12)
        noise = rng.normal(0.0, noise_level, size=len(wn))
        y_signal = np.maximum(base_func(wn) * var_scale + noise, 0.0)
        
        sample = FTIRSample(
            sample_id=s_id,
            wavenumbers=wn.copy(),
            absorbance=y_signal,
            raw_signal=y_signal.copy(),
            signal_type="absorbance",
            metadata={"sample_type": "Sludge Lignin" if is_lignin else "Pristine Water"}
        )
        samples_dict[s_id] = sample
        sample_list.append(sample)
        
    # Формируем сводный датафрейм дескрипторов
    descriptors_df = batch_process_ftir_samples(sample_list, baseline_method="als", norm_method="none")
    descriptors_df["Sample_Type"] = ["Sludge Lignin" if i >= half else "Pristine Water" for i in range(n_samples)]
    
    return samples_dict, descriptors_df


# ==============================================================================
# АВТОПОИСК ПИКОВ И АННОТАЦИЯ ФУНКЦИОНАЛЬНЫХ ГРУПП (PEAK PICKING)
# ==============================================================================

CHARACTERISTIC_IR_BANDS = [
    {
        "min_wn": 3200.0,
        "max_wn": 3650.0,
        "name": "O-H / N-H stretching",
        "assignment_ru": "Водородно-связанные гидроксилы (O-H) и амины (N-H)",
        "assignment_en": "H-bonded hydroxyls (O-H) and amines (N-H)",
        "compound_class_ru": "Фенолы, спирты, углеводы, пептиды",
        "compound_class_en": "Phenols, alcohols, carbohydrates, peptides",
    },
    {
        "min_wn": 2945.0,
        "max_wn": 2980.0,
        "name": "asym C-H (CH3)",
        "assignment_ru": "Асимметричные валентные колебания метилов (-CH3)",
        "assignment_en": "Asymmetric stretching of methyl (-CH3)",
        "compound_class_ru": "Алифатические разветвления, терпеноиды",
        "compound_class_en": "Aliphatic branching, terpenoids",
    },
    {
        "min_wn": 2905.0,
        "max_wn": 2942.0,
        "name": "asym C-H (CH2)",
        "assignment_ru": "Асимметричные валентные колебания метиленов (-CH2-)",
        "assignment_en": "Asymmetric stretching of methylenes (-CH2-)",
        "compound_class_ru": "Алифатические углеводородные цепи, липиды",
        "compound_class_en": "Aliphatic hydrocarbon chains, lipids",
    },
    {
        "min_wn": 2840.0,
        "max_wn": 2870.0,
        "name": "sym C-H (CH2)",
        "assignment_ru": "Симметричные валентные колебания метиленов (-CH2-)",
        "assignment_en": "Symmetric stretching of methylenes (-CH2-)",
        "compound_class_ru": "Алифатика, воски, биомембраны",
        "compound_class_en": "Aliphatics, waxes, biomembranes",
    },
    {
        "min_wn": 1730.0,
        "max_wn": 1755.0,
        "name": "ester C=O stretching",
        "assignment_ru": "Сложноэфирные карбонильные группы (C=O)",
        "assignment_en": "Ester carbonyl groups (C=O)",
        "compound_class_ru": "Сложные эфиры, ацетилированные гемицеллюлозы",
        "compound_class_en": "Esters, acetylated hemicelluloses",
    },
    {
        "min_wn": 1700.0,
        "max_wn": 1728.0,
        "name": "carboxylic C=O stretching",
        "assignment_ru": "Карбоксильные группы карбоновых кислот (COOH)",
        "assignment_en": "Carboxylic acid groups (COOH, C=O)",
        "compound_class_ru": "Гуминовые и фульвокислоты, карбоновые кислоты",
        "compound_class_en": "Humic & fulvic acids, carboxylic acids",
    },
    {
        "min_wn": 1640.0,
        "max_wn": 1675.0,
        "name": "conjugated C=O / Amide I",
        "assignment_ru": "Сопряженные хиноидные карбонилы / Амид I (C=O)",
        "assignment_en": "Conjugated quinone carbonyls / Amide I (C=O)",
        "compound_class_ru": "Хиноны, пептиды, белки",
        "compound_class_en": "Quinones, peptides, proteins",
    },
    {
        "min_wn": 1590.0,
        "max_wn": 1625.0,
        "name": "aromatic skeletal C=C",
        "assignment_ru": "Ароматический углеродный каркас (C=C)",
        "assignment_en": "Aromatic skeletal ring vibrations (C=C)",
        "compound_class_ru": "Поликонденсированная ароматика, каркас лигнина",
        "compound_class_en": "Polycondensed aromatics, lignin skeleton",
    },
    {
        "min_wn": 1500.0,
        "max_wn": 1525.0,
        "name": "aromatic skeletal C=C (lignin)",
        "assignment_ru": "Скелетные колебания ароматического кольца лигнина (C=C; все H/G/S-звенья, сильнее у G-богатых)",
        "assignment_en": "Aromatic skeletal ring vibration of lignin (C=C; all H/G/S units, stronger in G-rich lignins)",
        "compound_class_ru": "Лигнины всех типов: шлам-лигнин БЦБК, крафт-лигнин, лигнины древесины",
        "compound_class_en": "All lignin types: BCBK sludge lignin, kraft lignin, wood lignins",
    },
    {
        "min_wn": 1445.0,
        "max_wn": 1475.0,
        "name": "C-H bending (CH2 / CH3)",
        "assignment_ru": "Деформационные колебания алифатики (C-H, CH2/CH3)",
        "assignment_en": "Aliphatic C-H bending (CH2 / CH3)",
        "compound_class_ru": "Алифатические группы, метоксилы лигнина (-OCH3)",
        "compound_class_en": "Aliphatics, lignin methoxyls (-OCH3)",
    },
    {
        "min_wn": 1365.0,
        "max_wn": 1390.0,
        "name": "sym C-H (CH3) bend",
        "assignment_ru": "Симметричные деформационные колебания метилов (-CH3)",
        "assignment_en": "Symmetric methyl C-H bending (-CH3)",
        "compound_class_ru": "Разветвленная алифатика, ацетаты",
        "compound_class_en": "Branched aliphatics, acetates",
    },
    {
        "min_wn": 1315.0,
        "max_wn": 1340.0,
        "name": "syringyl ring C-O / C=C",
        "assignment_ru": "Сирингильное кольцо S-типа лигнина (C-O / C=C)",
        "assignment_en": "Syringyl ring S-type lignin breathing",
        "compound_class_ru": "Лигнин лиственных пород растений",
        "compound_class_en": "Hardwood lignin, angiosperms",
    },
    {
        "min_wn": 1255.0,
        "max_wn": 1285.0,
        "name": "guaiacyl C-O / phenolic OH",
        "assignment_ru": "Гваяцильное кольцо G-типа / Фенольные группы C-O(H)",
        "assignment_en": "Guaiacyl G-type ring / Phenolic C-O(H)",
        "compound_class_ru": "Лигнин хвойных пород, гуминовые фенолы",
        "compound_class_en": "Softwood lignin, humic phenols",
    },
    {
        "min_wn": 1205.0,
        "max_wn": 1240.0,
        "name": "ester / ether C-O / C-C",
        "assignment_ru": "Сложноэфирные и эфирные мостики (C-O, C-C)",
        "assignment_en": "Ester and ether linkages (C-O, C-C)",
        "compound_class_ru": "Сшитый лигнин, танины",
        "compound_class_en": "Crosslinked lignin, tannins",
    },
    {
        "min_wn": 1010.0,
        "max_wn": 1075.0,
        "name": "polysaccharide C-O-C / C-O",
        "assignment_ru": "Гликозидные связи полисахаридов (C-O-C, C-O) / Силикаты",
        "assignment_en": "Polysaccharide glycosidic bonds (C-O-C, C-O) / Silicates",
        "compound_class_ru": "Целлюлоза, гемицеллюлоза, пептидогликан, диатомеи",
        "compound_class_en": "Cellulose, hemicellulose, peptidoglycan, diatoms",
    },
    {
        "min_wn": 800.0,
        "max_wn": 890.0,
        "name": "aromatic out-of-plane C-H",
        "assignment_ru": "Внеплоскостные деформационные колебания arom C-H",
        "assignment_en": "Aromatic out-of-plane C-H bending",
        "compound_class_ru": "Замещенные бензольные кольца, конденсированная ароматика",
        "compound_class_en": "Substituted benzenes, condensed aromatics",
    },
]


def find_and_annotate_peaks(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    prominence: float = 0.02,
    min_distance_cm1: float = 20.0,
    lang: str = "ru"
) -> pd.DataFrame:
    """
    Автоматический поиск пиков колебательного спектра и их спектрохимическая
    аннотация по базе характеристических частот функциональных групп.
    """
    if len(wavenumbers) < 10 or len(absorbance) < 10:
        return pd.DataFrame()
        
    wn_step = float(np.mean(np.abs(np.diff(wavenumbers))))
    dist_pts = max(1, int(round(min_distance_cm1 / (wn_step + 1e-9))))
    
    # Поиск локальных максимумов
    peak_idx, props = scipy.signal.find_peaks(
        absorbance,
        prominence=prominence,
        distance=dist_pts
    )
    
    if len(peak_idx) == 0:
        return pd.DataFrame()
        
    rows = []
    proms = props.get("prominences", np.zeros(len(peak_idx)))
    
    is_en = (lang == "en")
    
    for i, p_i in enumerate(peak_idx):
        pos_wn = float(wavenumbers[p_i])
        val_abs = float(absorbance[p_i])
        prom_val = float(proms[i])
        
        # Поиск совпадения в базе характеристических полос
        matched_band = None
        for band in CHARACTERISTIC_IR_BANDS:
            if band["min_wn"] <= pos_wn <= band["max_wn"]:
                matched_band = band
                break
                
        if matched_band is not None:
            group_name = matched_band["name"]
            assignment = matched_band["assignment_en"] if is_en else matched_band["assignment_ru"]
            comp_class = matched_band["compound_class_en"] if is_en else matched_band["compound_class_ru"]
            status = "Verified" if is_en else "Идентифицирован"
        else:
            group_name = "Unassigned"
            assignment = "Unassigned skeletal / fingerprint band" if is_en else "Неидентифицированная полоса"
            comp_class = "Organic matrix" if is_en else "Органическая матрица"
            status = "Candidate" if is_en else "Кандидат"
            
        rows.append({
            "Peak_Index": i + 1,
            "Wavenumber_cm1": round(pos_wn, 1),
            "Absorbance": round(val_abs, 4),
            "Prominence": round(prom_val, 4),
            "Functional_Group": group_name,
            "Molecular_Assignment": assignment,
            "Compound_Class": comp_class,
            "Identification_Status": status,
        })
        
    df_peaks = pd.DataFrame(rows)
    # Сортируем от больших волновых чисел к меньшим (традиционно для ИК)
    df_peaks = df_peaks.sort_values(by="Wavenumber_cm1", ascending=False).reset_index(drop=True)
    df_peaks["Peak_Index"] = range(1, len(df_peaks) + 1)
    return df_peaks


# ==============================================================================
# СПЕКТРАЛЬНАЯ БИБЛИОТЕКА ЭТАЛОНОВ И СОПОСТАВЛЕНИЕ (SPECTRAL LIBRARY MATCHING)
# ==============================================================================

# Пометка для пользовательского интерфейса: библиотека состоит из синтетических моделей
FTIR_LIBRARY_DISCLAIMER_RU = (
    "Встроенная библиотека — приближенные СИНТЕТИЧЕСКИЕ модели (суммы гауссовых полос по литературным "
    "положениям полос), а не измеренные эталонные спектры IHSS/лигнинов. HQI показывает сходство с моделью, "
    "а не идентификацию."
)
FTIR_LIBRARY_DISCLAIMER_EN = (
    "The built-in library holds approximate SYNTHETIC models (sums of Gaussian bands at literature band "
    "positions), not measured IHSS/lignin reference spectra. HQI expresses similarity to a model, not an identification."
)
FTIR_LIBRARY_TYPE = "Synthetic Gaussian model (approximate, not a measured spectrum)"


def get_ftir_reference_library() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает встроенную ПРИБЛИЖЕННУЮ СИНТЕТИЧЕСКУЮ библиотеку ChemoSuite: каждый «эталон» —
    сумма гауссовых полос (profile_func) с типичными для класса положениями, а не измеренный
    спектр стандарта. Записи (названия соответствуют моделируемому классу):
    1. IHSS Suwannee River FA (Фульвокислоты)
    2. IHSS Suwannee River HA (Гуминовые кислоты)
    3. IHSS Elliott Soil HA (Почвенные гуминовые кислоты)
    4. Kraft Lignin (Сульфатный шлам-лигнин)
    5. Hydrolysis Lignin (Гидролизный технический лигнин)
    6. Cellulose Standard (Микрокристаллическая целлюлоза)
    7. Chitosan / Chitin (Аминополисахариды)
    8. Polystyrene Standard (Микропластик / калибровочный стандарт)
    """
    library = {
        "IHSS_Suwannee_River_FA": {
            "name_ru": "IHSS Suwannee River FA (Фульвокислота)",
            "name_en": "IHSS Suwannee River Fulvic Acid (2S101F)",
            "class_ru": "Природное органическое вещество (DOM / FA)",
            "class_en": "Natural Organic Matter (DOM / FA)",
            "key_markers": "1720 (COOH), 1620 (C=C), 3400 (OH), 1220 (C-O)",
            "description_ru": "Международный стандарт гуминовых веществ. Характеризуется высоким содержанием карбоксилов (1720 см^-1) и гидрофильностью.",
            "description_en": "IHSS international reference fulvic acid. High carboxylic content (1720 cm^-1) and high water solubility.",
            "profile_func": lambda w: (
                0.04 + 0.12 * np.exp(-((w - 650.0) / 1300.0) ** 2) +
                0.60 * np.exp(-0.5 * ((w - 3390.0) / 160.0) ** 2) +
                0.35 * np.exp(-0.5 * ((w - 2930.0) / 35.0) ** 2) +
                0.20 * np.exp(-0.5 * ((w - 2855.0) / 25.0) ** 2) +
                0.78 * np.exp(-0.5 * ((w - 1720.0) / 25.0) ** 2) +
                0.48 * np.exp(-0.5 * ((w - 1620.0) / 30.0) ** 2) +
                0.08 * np.exp(-0.5 * ((w - 1510.0) / 15.0) ** 2) +
                0.32 * np.exp(-0.5 * ((w - 1400.0) / 25.0) ** 2) +
                0.45 * np.exp(-0.5 * ((w - 1220.0) / 30.0) ** 2) +
                0.38 * np.exp(-0.5 * ((w - 1050.0) / 45.0) ** 2)
            )
        },
        "IHSS_Suwannee_River_HA": {
            "name_ru": "IHSS Suwannee River HA (Гуминовая кислота)",
            "name_en": "IHSS Suwannee River Humic Acid (2S101H)",
            "class_ru": "Природное органическое вещество (DOM / HA)",
            "class_en": "Natural Organic Matter (DOM / HA)",
            "key_markers": "1615 (C=C), 1715 (COOH), 2925 (CH2), 1225 (C-O)",
            "description_ru": "Водные гуминовые кислоты реки Суванни. Более высокая ароматичность (1615 см^-1) и гидрофобность, чем у FA.",
            "description_en": "Aquatic humic acid from Suwannee River. Higher aromaticity (1615 cm^-1) and hydrophobicity than FA.",
            "profile_func": lambda w: (
                0.05 + 0.15 * np.exp(-((w - 650.0) / 1200.0) ** 2) +
                0.55 * np.exp(-0.5 * ((w - 3400.0) / 150.0) ** 2) +
                0.42 * np.exp(-0.5 * ((w - 2925.0) / 30.0) ** 2) +
                0.25 * np.exp(-0.5 * ((w - 2850.0) / 20.0) ** 2) +
                0.62 * np.exp(-0.5 * ((w - 1715.0) / 25.0) ** 2) +
                0.68 * np.exp(-0.5 * ((w - 1612.0) / 28.0) ** 2) +
                0.12 * np.exp(-0.5 * ((w - 1510.0) / 15.0) ** 2) +
                0.30 * np.exp(-0.5 * ((w - 1410.0) / 25.0) ** 2) +
                0.42 * np.exp(-0.5 * ((w - 1225.0) / 30.0) ** 2) +
                0.40 * np.exp(-0.5 * ((w - 1040.0) / 45.0) ** 2)
            )
        },
        "IHSS_Elliott_Soil_HA": {
            "name_ru": "IHSS Elliott Soil HA (Почвенная ГК)",
            "name_en": "IHSS Elliott Soil Humic Acid (1S102H)",
            "class_ru": "Почвенные гуминовые вещества (Soil HA)",
            "class_en": "Soil Humic Substances (Soil HA)",
            "key_markers": "1620 (C=C arom), 2920 (aliphatic), 1540 (Amide II), 1035 (Si/Poly)",
            "description_ru": "Почвенная гуминовая кислота. Высокая поликонденсированная ароматика, выраженные амидные азотистые полосы и глинисто-силикатный след.",
            "description_en": "Soil humic acid standard. High condensed aromaticity, nitrogenous amide bands and mineral-associated signature.",
            "profile_func": lambda w: (
                0.06 + 0.18 * np.exp(-((w - 650.0) / 1100.0) ** 2) +
                0.52 * np.exp(-0.5 * ((w - 3410.0) / 150.0) ** 2) +
                0.48 * np.exp(-0.5 * ((w - 2920.0) / 30.0) ** 2) +
                0.28 * np.exp(-0.5 * ((w - 2850.0) / 22.0) ** 2) +
                0.50 * np.exp(-0.5 * ((w - 1715.0) / 25.0) ** 2) +
                0.75 * np.exp(-0.5 * ((w - 1618.0) / 30.0) ** 2) +
                0.28 * np.exp(-0.5 * ((w - 1540.0) / 20.0) ** 2) +
                0.35 * np.exp(-0.5 * ((w - 1450.0) / 22.0) ** 2) +
                0.38 * np.exp(-0.5 * ((w - 1230.0) / 28.0) ** 2) +
                0.52 * np.exp(-0.5 * ((w - 1035.0) / 40.0) ** 2)
            )
        },
        "Kraft_Lignin_BCBK": {
            "name_ru": "Крафт-лигнин / Шлам-лигнин БЦБК",
            "name_en": "Kraft Lignin / BCBK Pulp Sludge Lignin",
            "class_ru": "Техногенный лигнин (Pulp & Paper Sludge)",
            "class_en": "Industrial Sludge Lignin",
            "key_markers": "1510 (lignin aromatic skeletal C=C), 1602 (Aromatic), 1270 (G-ring), 1220 (Ether)",
            "description_ru": "Синтетическая модель сульфатного шлам-лигнина: острая полоса скелетной ароматики лигнина 1510 см^-1, ароматика 1602 см^-1 и гваяцильная полоса 1270 см^-1.",
            "description_en": "Synthetic model of pulp mill sludge lignin: sharp lignin aromatic skeletal band at 1510 cm^-1, 1602 cm^-1 aromatics and 1270 cm^-1 guaiacyl band.",
            "profile_func": lambda w: (
                0.08 + 0.22 * np.exp(-((w - 650.0) / 1000.0) ** 2) +
                0.48 * np.exp(-0.5 * ((w - 3410.0) / 140.0) ** 2) +
                0.26 * np.exp(-0.5 * ((w - 2928.0) / 28.0) ** 2) +
                0.16 * np.exp(-0.5 * ((w - 2848.0) / 20.0) ** 2) +
                0.58 * np.exp(-0.5 * ((w - 1714.0) / 24.0) ** 2) +
                0.66 * np.exp(-0.5 * ((w - 1602.0) / 24.0) ** 2) +
                0.62 * np.exp(-0.5 * ((w - 1510.0) / 15.0) ** 2) +
                0.25 * np.exp(-0.5 * ((w - 1455.0) / 20.0) ** 2) +
                0.44 * np.exp(-0.5 * ((w - 1270.0) / 22.0) ** 2) +
                0.46 * np.exp(-0.5 * ((w - 1220.0) / 24.0) ** 2) +
                0.32 * np.exp(-0.5 * ((w - 1030.0) / 45.0) ** 2)
            )
        },
        "Hydrolysis_Lignin": {
            "name_ru": "Гидролизный технический лигнин",
            "name_en": "Hydrolysis Technical Lignin",
            "class_ru": "Технический гидролизный лигнин",
            "class_en": "Hydrolysis Lignin Byproduct",
            "key_markers": "1512 (lignin aromatic skeletal C=C), 1600 (Arom C=C), 1425 (Arom), 1215 (C-C/C-O)",
            "description_ru": "Продукт кислотного гидролиза древесины. Высокая степень конденсации, развитая фенольная сетка, минимум углеводов.",
            "description_en": "Acid hydrolysis wood byproduct. Highly condensed aromatic network, pronounced phenolic OH, low carbohydrates.",
            "profile_func": lambda w: (
                0.07 + 0.20 * np.exp(-((w - 650.0) / 1050.0) ** 2) +
                0.45 * np.exp(-0.5 * ((w - 3420.0) / 140.0) ** 2) +
                0.22 * np.exp(-0.5 * ((w - 2932.0) / 28.0) ** 2) +
                0.14 * np.exp(-0.5 * ((w - 2850.0) / 20.0) ** 2) +
                0.45 * np.exp(-0.5 * ((w - 1710.0) / 25.0) ** 2) +
                0.70 * np.exp(-0.5 * ((w - 1600.0) / 26.0) ** 2) +
                0.55 * np.exp(-0.5 * ((w - 1512.0) / 16.0) ** 2) +
                0.32 * np.exp(-0.5 * ((w - 1425.0) / 22.0) ** 2) +
                0.40 * np.exp(-0.5 * ((w - 1265.0) / 22.0) ** 2) +
                0.48 * np.exp(-0.5 * ((w - 1215.0) / 24.0) ** 2) +
                0.25 * np.exp(-0.5 * ((w - 1035.0) / 45.0) ** 2)
            )
        },
        "Cellulose_Standard": {
            "name_ru": "Микрокристаллическая целлюлоза",
            "name_en": "Microcrystalline Cellulose Standard",
            "class_ru": "Полисахариды и биополимеры",
            "class_en": "Polysaccharides & Biopolymers",
            "key_markers": "1035–1060 (C-O-C), 3350 (O-H broad), 2900 (C-H), 895 (beta-glycosidic)",
            "description_ru": "Чистый полисахаридный эталон. Доминирует колоссальный пик 1030–1060 см^-1. Полностью отсутствуют ароматические полосы 1600 и 1510 см^-1.",
            "description_en": "Pure polysaccharide standard. Dominated by massive 1030–1060 cm^-1 peak. Completely devoid of 1600/1510 cm^-1 aromatic bands.",
            "profile_func": lambda w: (
                0.03 + 0.08 * np.exp(-((w - 650.0) / 1400.0) ** 2) +
                0.85 * np.exp(-0.5 * ((w - 3350.0) / 140.0) ** 2) +
                0.42 * np.exp(-0.5 * ((w - 2898.0) / 32.0) ** 2) +
                0.15 * np.exp(-0.5 * ((w - 1645.0) / 25.0) ** 2) +
                0.28 * np.exp(-0.5 * ((w - 1430.0) / 20.0) ** 2) +
                0.30 * np.exp(-0.5 * ((w - 1370.0) / 18.0) ** 2) +
                0.35 * np.exp(-0.5 * ((w - 1162.0) / 22.0) ** 2) +
                0.95 * np.exp(-0.5 * ((w - 1055.0) / 35.0) ** 2) +
                0.88 * np.exp(-0.5 * ((w - 1032.0) / 30.0) ** 2) +
                0.22 * np.exp(-0.5 * ((w - 895.0) / 15.0) ** 2)
            )
        },
        "Chitosan_Biopolymer": {
            "name_ru": "Хитозан / Хитин (Биополимер)",
            "name_en": "Chitosan / Chitin Biopolymer",
            "class_ru": "Аминополисахариды (Biopolymers)",
            "class_en": "Aminopolysaccharides (Biopolymers)",
            "key_markers": "1655 (Amide I), 1590 (Amide II), 1070 (C-O-C), 3350 (OH/NH)",
            "description_ru": "Биогенный полимер ракообразных и зоопланктона. Характерные полосы Амид I (1655 см^-1), деформации NH2 (1590 см^-1) и гликозидная связь 1070 см^-1.",
            "description_en": "Biogenic crustacean and zooplankton polymer. Distinct Amide I (1655 cm^-1), NH2 bending (1590 cm^-1), and glycosidic C-O-C at 1070 cm^-1.",
            "profile_func": lambda w: (
                0.04 + 0.10 * np.exp(-((w - 650.0) / 1300.0) ** 2) +
                0.78 * np.exp(-0.5 * ((w - 3360.0) / 145.0) ** 2) +
                0.35 * np.exp(-0.5 * ((w - 2878.0) / 30.0) ** 2) +
                0.65 * np.exp(-0.5 * ((w - 1655.0) / 22.0) ** 2) +
                0.58 * np.exp(-0.5 * ((w - 1590.0) / 20.0) ** 2) +
                0.32 * np.exp(-0.5 * ((w - 1380.0) / 20.0) ** 2) +
                0.38 * np.exp(-0.5 * ((w - 1152.0) / 24.0) ** 2) +
                0.86 * np.exp(-0.5 * ((w - 1072.0) / 35.0) ** 2) +
                0.70 * np.exp(-0.5 * ((w - 1028.0) / 32.0) ** 2)
            )
        },
        "Polystyrene_Microplastic": {
            "name_ru": "Полистирол (Калибровочный стандарт / Микропластик)",
            "name_en": "Polystyrene (Microplastic / Calibration Standard)",
            "class_ru": "Синтетический полимер (Microplastic)",
            "class_en": "Synthetic Polymer (Microplastic)",
            "key_markers": "3026 (arom C-H), 2924 (CH2), 1601, 1493, 1452, 756, 698",
            "description_ru": "Калибровочный стандарт ИК-спектрометров и эталон микропластика. Четкие сверхузкие пики монозамещенного бензольного кольца (1601, 1493, 756, 698 см^-1).",
            "description_en": "Spectrometer calibration standard and microplastic pollutant. Extremely sharp monosubstituted benzene peaks (1601, 1493, 756, 698 cm^-1).",
            "profile_func": lambda w: (
                0.02 +
                0.55 * np.exp(-0.5 * ((w - 3026.0) / 8.0) ** 2) +
                0.38 * np.exp(-0.5 * ((w - 2924.0) / 10.0) ** 2) +
                0.22 * np.exp(-0.5 * ((w - 2850.0) / 8.0) ** 2) +
                0.62 * np.exp(-0.5 * ((w - 1601.0) / 7.0) ** 2) +
                0.72 * np.exp(-0.5 * ((w - 1493.0) / 7.0) ** 2) +
                0.68 * np.exp(-0.5 * ((w - 1452.0) / 7.0) ** 2) +
                0.35 * np.exp(-0.5 * ((w - 1028.0) / 7.0) ** 2) +
                0.85 * np.exp(-0.5 * ((w - 756.0) / 8.0) ** 2) +
                0.90 * np.exp(-0.5 * ((w - 698.0) / 8.0) ** 2)
            )
        },
    }
    return library


def get_reference_spectrum(
    standard_id: str,
    wn_grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Генерирует спектр эталона на заданной сетке волновых чисел (по умолчанию 650–4000 см^-1).
    """
    library = get_ftir_reference_library()
    if standard_id not in library:
        raise ValueError(f"Неизвестный эталонный стандарт '{standard_id}'. Доступные: {list(library.keys())}")
        
    std_meta = library[standard_id]
    if wn_grid is None:
        wn_grid = np.linspace(650.0, 4000.0, 1676)
        
    func = std_meta["profile_func"]
    absorbance = np.maximum(func(wn_grid), 0.0)
    return wn_grid, absorbance, std_meta


def match_spectrum_against_library(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    library: Optional[Dict[str, Dict[str, Any]]] = None,
    top_n: int = 5,
    lang: str = "ru"
) -> pd.DataFrame:
    """
    Сопоставление исследуемого спектра с приближенной синтетической библиотекой FTIR
    методом спектрального косинусного сходства (Cosine Similarity),
    корреляции Пирсона (Pearson r) и Hit Quality Index (HQI, %).
    Столбец Library_Type напоминает, что эталоны — синтетические модели, а не измеренные спектры.
    """
    if library is None:
        library = get_ftir_reference_library()
        
    if len(wavenumbers) < 10 or len(absorbance) < 10:
        return pd.DataFrame()
        
    # Сетка сравнения в интервале перекрытия
    w_min = max(float(np.min(wavenumbers)), 650.0)
    w_max = min(float(np.max(wavenumbers)), 4000.0)
    
    if w_max <= w_min + 100.0:
        return pd.DataFrame()
        
    common_grid = np.linspace(w_min, w_max, 1000)
    # Интерполяция исследуемого спектра
    sample_interp = np.interp(common_grid, wavenumbers, absorbance)
    sample_interp = np.maximum(sample_interp, 0.0)
    
    s_norm = np.linalg.norm(sample_interp)
    if s_norm < 1e-9:
        return pd.DataFrame()
        
    results = []
    is_en = (lang == "en")
    
    for std_id, std_data in library.items():
        func = std_data["profile_func"]
        ref_signal = np.maximum(func(common_grid), 0.0)
        r_norm = np.linalg.norm(ref_signal)
        
        if r_norm < 1e-9:
            continue
            
        # 1. Cosine Similarity: dot(s, r) / (||s|| * ||r||)
        dot_prod = float(np.dot(sample_interp, ref_signal))
        cos_sim = dot_prod / (s_norm * r_norm + 1e-12)
        cos_sim = max(0.0, min(1.0, cos_sim))
        
        # 2. HQI (Hit Quality Index) = (dot(s, r))^2 / (dot(s, s) * dot(r, r)) * 100%
        hqi_pct = (cos_sim ** 2) * 100.0
        
        # 3. Корреляция Пирсона (инвариантна к масштабу и фоновому смещению)
        s_centered = sample_interp - np.mean(sample_interp)
        r_centered = ref_signal - np.mean(ref_signal)
        denom = (np.linalg.norm(s_centered) * np.linalg.norm(r_centered)) + 1e-12
        pearson_r = float(np.dot(s_centered, r_centered) / denom)
        pearson_r = max(-1.0, min(1.0, pearson_r))
        
        results.append({
            "Standard_ID": std_id,
            "Standard_Name": std_data["name_en"] if is_en else std_data["name_ru"],
            "Compound_Class": std_data["class_en"] if is_en else std_data["class_ru"],
            "HQI_pct": round(hqi_pct, 2),
            "Cosine_Similarity": round(cos_sim, 4),
            "Pearson_r": round(pearson_r, 4),
            "Key_Markers": std_data["key_markers"],
            "Description": std_data["description_en"] if is_en else std_data["description_ru"],
            "Library_Type": FTIR_LIBRARY_TYPE,
        })
        
    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res
        
    # Сортируем по HQI по убыванию
    df_res = df_res.sort_values(by="HQI_pct", ascending=False).reset_index(drop=True)
    df_res.insert(0, "Rank", range(1, len(df_res) + 1))
    
    return df_res.head(top_n)

