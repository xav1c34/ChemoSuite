"""
nmr_core.py — Математическое ядро спектроскопии ядерного магнитного резонанса (13C / 1H NMR).
Методология хемометрического анализа углеродного скелета и функционального состава
природного органического вещества (РОВ/DOM, гуминовые кислоты) и техногенного шлам-лигнина.
Соответствует стандартам IHSS (International Humic Substances Society) и методикам МГУ.
"""
from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import scipy.signal
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    from open_spectrometry_formats import (
        parse_jcamp_dx as _parse_jcamp_generic,
        jcamp_nmr_x_to_ppm,
        jcamp_metadata_value,
    )
except ImportError:  # импорт как modules.nmr_core
    from .open_spectrometry_formats import (
        parse_jcamp_dx as _parse_jcamp_generic,
        jcamp_nmr_x_to_ppm,
        jcamp_metadata_value,
    )


@dataclass
class NMRSample:
    """Контейнер для отдельного спектра ЯМР (13C CP/MAS или 1H NMR)."""
    sample_id: str
    nucleus: str = "13C"                   # "13C" или "1H"
    chemical_shifts: np.ndarray = field(default_factory=lambda: np.array([]))  # ppm (по возрастанию)
    intensity: np.ndarray = field(default_factory=lambda: np.array([]))        # Сырая интенсивность
    baseline: Optional[np.ndarray] = None
    corrected_intensity: Optional[np.ndarray] = None
    integrals: Dict[str, float] = field(default_factory=dict)
    indices: Dict[str, float] = field(default_factory=dict)
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
# 1. ПАРСИНГ СПЕКТРОВ ЯМР (CSV, TXT, TSV, XY, JCAMP-DX .JDX)
# ==============================================================================

def parse_jcamp_dx(text: Union[str, bytes], sample_id: str = "NMR_Sample") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Парсер IUPAC JCAMP-DX для спектров ЯМР (Bruker TopSpin, Varian/Agilent, Mnova, JEOL).

    Декодирование выполняет общий парсер open_spectrometry_formats.parse_jcamp_dx
    (JCAMP-DX 4.24: AFFN, SQZ/DIF/DUP с Y-check, (XY..XY)/PEAK TABLE, XFACTOR/YFACTOR).
    Абсцисса в Hz переводится в ppm по ##$SF / ##.OBSERVE FREQUENCY с учетом
    ##.SHIFT REFERENCE (Davies & Lancashire, Pure Appl. Chem. 1993, 65, 1085).

    Returns (ppm по возрастанию, интенсивность, метаданные). В метаданных:
    'JCAMP_WARNINGS' — список предупреждений декодера, 'JCAMP_DATA_FORM' — форма таблицы.
    Raises ValueError, если данные извлечь не удалось.
    """
    try:
        parsed = _parse_jcamp_generic(text)
        x_ppm, notes = jcamp_nmr_x_to_ppm(parsed["x"], parsed["metadata"], parsed.get("x_units", ""))
    except ValueError as exc:
        raise ValueError(f"JCAMP-DX '{sample_id}': {exc}") from exc

    meta: Dict[str, Any] = dict(parsed["metadata"])
    meta["JCAMP_WARNINGS"] = list(parsed.get("warnings", [])) + list(notes)
    meta["JCAMP_DATA_FORM"] = parsed.get("data_form", "")

    y_arr = np.asarray(parsed["y"], dtype=np.float64)
    finite = np.isfinite(x_ppm) & np.isfinite(y_arr)
    if not np.any(finite):
        raise ValueError(f"Не удалось извлечь спектральные координаты из JCAMP-DX файла '{sample_id}'.")
    if not np.all(finite):
        meta["JCAMP_WARNINGS"].append(f"{int(np.sum(~finite))} missing ('?') ordinates dropped")
    x_arr = x_ppm[finite]
    y_arr = y_arr[finite]

    # Сортировка по возрастанию химсдвигов ppm
    sort_idx = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    # Удаление дубликатов с усреднением
    u_x, inv = np.unique(np.round(x_sorted, 4), return_inverse=True)
    u_y = np.zeros_like(u_x)
    counts = np.zeros_like(u_x)
    np.add.at(u_y, inv, y_sorted)
    np.add.at(counts, inv, 1.0)
    u_y = u_y / np.maximum(counts, 1.0)

    return u_x, u_y, meta


def parse_nmr_file(
    file_content: Union[str, bytes],
    sample_id: str = "NMR_Sample",
    nucleus: str = "13C"
) -> NMRSample:
    """
    Универсальный парсер файлов спектров ЯМР (.csv, .txt, .tsv, .xy, .jdx, .dx).
    
    Поддерживает:
    1. Формат IUPAC JCAMP-DX (AFFN/ASDF-сжатие, (XY..XY), Hz → ppm). Ошибки JCAMP пробрасываются как ValueError.
    2. Табличные спектры приборов Bruker (TopSpin), Varian, JEOL, Mestrelab Mnova.
    3. Автоматически распознает десятичные запятые, разделители (запятая, точка с запятой, табуляция, пробел).
    4. Автоматически сортирует шкалу химических сдвигов ppm по возрастанию.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    text_clean = text.strip()
    
    # 1. Проверка на формат JCAMP-DX. Ошибки декодирования JCAMP не маскируются табличным
    #    парсером: он прочитал бы сжатые строки (X++(Y..Y)) как бессмысленные пары (X, Y).
    if text_clean.startswith("##") or "##JCAMP-DX" in text_clean:
        x_arr, y_arr, meta = parse_jcamp_dx(text_clean, sample_id=sample_id)
        # Определение ядра из метаданных, если указано (##.OBSERVE NUCLEUS=^13C)
        nuc = nucleus
        raw_nuc = str(jcamp_metadata_value(meta, ".OBSERVE NUCLEUS") or "").upper()
        if "13C" in raw_nuc or "C13" in raw_nuc:
            nuc = "13C"
        elif "1H" in raw_nuc or "H1" in raw_nuc:
            nuc = "1H"
        return NMRSample(
            sample_id=sample_id,
            nucleus=nuc,
            chemical_shifts=x_arr,
            intensity=y_arr,
            corrected_intensity=y_arr.copy(),
            metadata=meta
        )

    # 2. Табличный парсинг (CSV, TSV, TXT, XY)
    lines = text_clean.splitlines()
    data_rows: List[Tuple[float, float]] = []
    
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith(("#", ";", "/", "!", "Spectrum", "Date", "ppm", "Chemical", "Intensity")):
            continue
            
        if ";" in line_s:
            tokens = [p.strip() for p in line_s.split(";") if p.strip()]
        elif "\t" in line_s:
            tokens = [p.strip() for p in line_s.split("\t") if p.strip()]
        elif "," in line_s:
            tokens = [p.strip() for p in line_s.split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in line_s.split() if p.strip()]
            
        if len(tokens) >= 2:
            try:
                t0 = tokens[0].replace(",", ".") if "." not in tokens[0] else tokens[0]
                t1 = tokens[1].replace(",", ".") if "." not in tokens[1] else tokens[1]
                v0 = float(t0)
                v1 = float(t1)
                data_rows.append((v0, v1))
            except ValueError:
                continue
                
    if not data_rows:
        raise ValueError(
            f"Не удалось извлечь спектральные данные ЯМР из файла '{sample_id}'. "
            "Требуются минимум 2 числовые колонки (химический сдвиг ppm и интенсивность) либо формат JCAMP-DX."
        )
        
    arr = np.array(data_rows, dtype=np.float64)
    
    # Определение, какая колонка является химсдвигом ppm, а какая интенсивностью
    # Обычно ppm лежит в диапазоне [-50, 300] и строго монотонна (возрастает или убывает)
    col0_range = (float(arr[:, 0].min()), float(arr[:, 0].max()))
    col1_range = (float(arr[:, 1].min()), float(arr[:, 1].max()))
    
    diff0 = np.diff(arr[:, 0]) if len(arr) > 1 else np.array([1.0])
    diff1 = np.diff(arr[:, 1]) if len(arr) > 1 else np.array([1.0])
    is_mono0 = bool(np.all(diff0 > 0) or np.all(diff0 < 0))
    is_mono1 = bool(np.all(diff1 > 0) or np.all(diff1 < 0))
    
    in_range0 = (-50.0 <= col0_range[0] and col0_range[1] <= 300.0)
    in_range1 = (-50.0 <= col1_range[0] and col1_range[1] <= 300.0)
    
    if is_mono0 and not is_mono1 and in_range0:
        ppm_idx, int_idx = 0, 1
    elif is_mono1 and not is_mono0 and in_range1:
        ppm_idx, int_idx = 1, 0
    elif in_range0 and not in_range1:
        ppm_idx, int_idx = 0, 1
    elif in_range1 and not in_range0:
        ppm_idx, int_idx = 1, 0
    else:
        ppm_idx, int_idx = 0, 1
        
    ppm_raw = arr[:, ppm_idx]
    int_raw = arr[:, int_idx]
    
    # Сортировка по возрастанию ppm
    sort_idx = np.argsort(ppm_raw)
    ppm_sorted = ppm_raw[sort_idx]
    int_sorted = int_raw[sort_idx]
    
    # Усреднение дубликатов с шагом округления
    u_ppm, inv = np.unique(np.round(ppm_sorted, 4), return_inverse=True)
    u_int = np.zeros_like(u_ppm)
    counts = np.zeros_like(u_ppm)
    np.add.at(u_int, inv, int_sorted)
    np.add.at(counts, inv, 1.0)
    u_int = u_int / np.maximum(counts, 1.0)
    
    return NMRSample(
        sample_id=sample_id,
        nucleus=nucleus,
        chemical_shifts=u_ppm,
        intensity=u_int,
        corrected_intensity=u_int.copy(),
        metadata={"raw_points": len(data_rows)}
    )


# ==============================================================================
# 2. МАТЕМАТИЧЕСКАЯ ОБРАБОТКА (AsLS БАЗОВАЯ ЛИНИЯ, СДВИГ ШКАЛЫ, СГЛАЖИВАНИЕ)
# ==============================================================================

def baseline_als_nmr(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 15) -> np.ndarray:
    """
    Асимметричные наименьшие квадраты (AsLS / Eilers baseline) для спектроскопии ЯМР.
    Обеспечивает устойчивую гибкую фоновую кривую без искажения широких гуминовых полос.
    """
    L = len(y)
    if L < 5:
        return np.zeros_like(y)
        
    # Разреженная матрица второй разности
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc", dtype=np.float64)
    DTD = D.T.dot(D) * lam
    w = np.ones(L)
    
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + DTD
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
        
    return np.asarray(z)


def recalibrate_chemical_shifts(
    chemical_shifts: np.ndarray,
    current_ref_ppm: float,
    target_ref_ppm: float = 0.0
) -> np.ndarray:
    """
    Калибровка шкалы химических сдвигов ppm (zero shift / internal reference calibration).
    Например, юстировка по TMS / DSS (0.0 ppm) или остаточному DMSO-d6 (39.5 ppm в 13C, 2.50 ppm в 1H).
    """
    delta = target_ref_ppm - current_ref_ppm
    return chemical_shifts + delta


# Остаточный сигнал растворителя DMSO-d6: δC = 39.52 ppm (септет), δH = 2.50 ppm (квинтет DMSO-d5)
# (Gottlieb, Kotlyar & Nudelman, J. Org. Chem. 1997, 62, 7512–7515). TMS/DSS = 0.00 ppm для обоих ядер.
DMSO_REFERENCE_PPM = {"13C": 39.5, "1H": 2.50}


def _normalize_nucleus(nucleus: str) -> str:
    nuc = str(nucleus or "13C").upper().replace("^", "")
    return "1H" if ("1H" in nuc or "H1" in nuc) else "13C"


def auto_recalibrate_nmr_shifts(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    mode: str = "none",
    nucleus: str = "13C"
) -> Tuple[np.ndarray, float]:
    """
    Автоматическая калибровка шкалы химических сдвигов ppm по внутреннему стандарту:
    - 'dmso': поиск остаточного сигнала DMSO-d6 — 39.5 ppm для 13C (окно 38.0–41.0 ppm)
      или 2.50 ppm для 1H (окно 2.0–3.0 ppm), см. DMSO_REFERENCE_PPM
    - 'tms': поиск пика TMS / DSS (~0.0 ppm в окне -1.5–1.5 ppm)
    - 'none': без калибровки

    Returns
    -------
    Tuple[np.ndarray, float]
        (откалиброванные сдвиги chemical_shifts, величина примененного сдвига delta в ppm)
    """
    if mode == "none" or len(chemical_shifts) == 0:
        return chemical_shifts.copy(), 0.0

    nuc = _normalize_nucleus(nucleus)
    if mode == "dmso":
        target_ppm = DMSO_REFERENCE_PPM[nuc]
        half_win = 0.5 if nuc == "1H" else 1.5
        win_min, win_max = target_ppm - half_win, target_ppm + half_win
        max_delta = half_win
    else:
        target_ppm = 0.0
        win_min, win_max = -1.5, 1.5
        max_delta = 2.5

    mask = (chemical_shifts >= win_min) & (chemical_shifts <= win_max)
    if np.sum(mask) < 2:
        return chemical_shifts.copy(), 0.0

    x_sub = chemical_shifts[mask]
    y_sub = intensity[mask]
    pk_idx = int(np.argmax(y_sub))

    # Пик должен быть локальным максимумом внутри окна, а не монотонным хвостом на границе окна
    if pk_idx == 0 or pk_idx == len(x_sub) - 1:
        return chemical_shifts.copy(), 0.0

    # Интенсивность пика должна быть заметной относительно спектра (не шум базовой линии)
    max_spec = float(np.max(intensity)) if len(intensity) > 0 else 0.0
    if max_spec > 1e-4 and y_sub[pk_idx] < 0.05 * max_spec:
        return chemical_shifts.copy(), 0.0

    found_ppm = float(x_sub[pk_idx])
    delta = target_ppm - found_ppm

    # Защита от ошибочной калибровки: если пик слишком далеко, считаем стандарт отсутствующим
    if abs(delta) > max_delta:
        return chemical_shifts.copy(), 0.0

    return chemical_shifts + delta, float(delta)


def subtract_baseline(
    sample: NMRSample,
    method: str = "asls",
    lam: float = 1e6,
    p: float = 0.01,
    poly_order: int = 1
) -> Tuple[NMRSample, np.ndarray]:
    """
    Вычитание базовой линии спектра ЯМР.
    Методы:
    - 'asls': Asymmetric Least Squares (рекомендуется для 13C CP/MAS)
    - 'linear': линейная коррекция по краям спектра
    - 'polynomial': полиномиальная аппроксимация (по краям)
    - 'none': без вычитания
    """
    y = sample.intensity
    x = sample.chemical_shifts
    
    if len(y) == 0:
        return sample, np.array([])
        
    if method == "asls":
        base = baseline_als_nmr(y, lam=lam, p=p)
    elif method == "linear":
        idx_edge = np.concatenate([np.arange(0, min(15, len(y))), np.arange(max(0, len(y) - 15), len(y))])
        slope, intercept = np.polyfit(x[idx_edge], y[idx_edge], 1)
        base = slope * x + intercept
    elif method == "polynomial":
        idx_edge = np.concatenate([np.arange(0, min(25, len(y))), np.arange(max(0, len(y) - 25), len(y))])
        p_fit = np.polyfit(x[idx_edge], y[idx_edge], poly_order)
        base = np.polyval(p_fit, x)
    elif method == "none":
        base = np.zeros_like(y)
    else:
        base = np.zeros_like(y)
        
    y_corr = y - base
    # Ограничение минимального шума (обнуление отрицательных артефактов)
    y_corr = np.maximum(y_corr, 0.0)
    
    sample.baseline = base
    sample.corrected_intensity = y_corr
    return sample, base


def smooth_nmr_spectrum(
    intensity: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """Сглаживание высокочастотного аппаратного шума фильтром Савицкого-Голея."""
    if len(intensity) < window_length:
        return intensity.copy()
    wl = window_length if window_length % 2 == 1 else window_length + 1
    return scipy.signal.savgol_filter(intensity, window_length=wl, polyorder=polyorder)


def normalize_nmr_spectrum(
    intensity: np.ndarray,
    method: str = "area",
    chemical_shifts: Optional[np.ndarray] = None,
    ppm_range: Tuple[float, float] = (0.0, 220.0)
) -> np.ndarray:
    """
    Нормализация спектра ЯМР:
    - 'area': нормализация полной площади к 100.0 (стандарт ЯМР, где суммарный углерод = 100%)
    - 'max': приведение максимального пика к 1.0
    - 'minmax': линейное масштабирование в [0, 1]
    - 'snv': Standard Normal Variate
    """
    y = intensity.copy()
    if len(y) == 0:
        return y
        
    if method == "area":
        if chemical_shifts is not None and len(chemical_shifts) == len(y):
            mask = (chemical_shifts >= ppm_range[0]) & (chemical_shifts <= ppm_range[1])
            if np.sum(mask) > 3:
                x_sub = chemical_shifts[mask]
                y_sub = y[mask]
                if len(x_sub) >= 2 and x_sub[0] > x_sub[-1]:
                    x_sub = x_sub[::-1]
                    y_sub = y_sub[::-1]
                tot_int = simpson(y_sub, x=x_sub)
                if tot_int > 1e-9:
                    return (y / tot_int) * 100.0
        tot_sum = np.sum(y)
        if tot_sum > 1e-9:
            return (y / tot_sum) * 100.0
        return y
    elif method == "max":
        mx = np.max(y)
        return y / mx if mx > 1e-9 else y
    elif method == "minmax":
        mn, mx = np.min(y), np.max(y)
        return (y - mn) / (mx - mn) if (mx - mn) > 1e-9 else y
    elif method == "snv":
        std = np.std(y)
        return (y - np.mean(y)) / std if std > 1e-9 else y
    return y


# ==============================================================================
# 3. ИНТЕГРИРОВАНИЕ ПО ЗОНАМ И РАСЧЕТ ИНДЕКСОВ 13C ЯМР (IHSS & ПЕРМИНОВА)
# ==============================================================================

# Схема интегрирования 13C ЯМР природного органического вещества.
# Границы (ppm) — общепринятые для почвенного/водного ОВ и гуминовых веществ
# (Kögel-Knabner, Soil Biol. Biochem. 2002, 34, 139–162; Baldock et al., Aust. J. Soil Res. 1997,
# 35, 1061–1083). У разных авторов границы ароматической/О-арильной/карбоксильной зон
# смещены на ±5 ppm (напр., у Kögel-Knabner 110–140 / 140–160 / 160–220 ppm).
#
# Тонкие (непересекающиеся) интервалы; сумма = 100 % углерода 0–220 ppm:
NMR_13C_FINE_BINS = {
    "alkyl": (0.0, 45.0),          # Алкильный C (метил, метилен, метин; липиды, кутин)
    "n_o_alkyl": (45.0, 60.0),     # N-алкил / метоксил (-OCH3 лигнина, Cα аминокислот)
    "o_alkyl": (60.0, 90.0),       # O-алкил (C2–C6 углеводов, спирты)
    "acetal": (90.0, 110.0),       # Ди-O-алкил / ацетальный, аномерный C1 углеводов
    "aromatic": (110.0, 145.0),    # Протонированный и C-замещенный ароматический C
    "phenolic": (145.0, 165.0),    # O-арил (фенольный / эфирный ароматический C)
    "carboxyl": (165.0, 190.0),    # Карбоксильный / амидный / сложноэфирный C
    "carbonyl": (190.0, 220.0),    # Кетонный / альдегидный C
}

# Шесть интегральных зон IHSS-типа (README): C-O алкилы = 45–90 ppm (включая подзону 45–60),
# карбоксилы = 165–220 ppm (включая карбонилы 190–220).
NMR_13C_IHSS_6BIN = {
    "Alkyl C (0–45)": ("alkyl",),
    "C-O alkyl C (45–90)": ("n_o_alkyl", "o_alkyl"),
    "Acetal / anomeric C (90–110)": ("acetal",),
    "Aromatic C (110–145)": ("aromatic",),
    "Phenolic / O-aryl C (145–165)": ("phenolic",),
    "Carboxyl / carbonyl C (165–220)": ("carboxyl", "carbonyl"),
}

# Агрегированные зоны (ключи-дескрипторы сохранены для совместимости)
NMR_13C_REGIONS = {
    "C_aliph_0_45": (0.0, 45.0),
    "C_O_alkyl_45_110": (45.0, 110.0),
    "C_aromatic_110_165": (110.0, 165.0),
    "C_carboxyl_165_190": (165.0, 190.0),
    "C_carbonyl_190_220": (190.0, 220.0),
}

# Подзоны для детального функционального анализа
NMR_13C_SUBREGIONS = {
    "C_methoxy_45_60": (45.0, 60.0),
    "C_O_alkyl_60_90": (60.0, 90.0),
    "C_acetal_90_110": (90.0, 110.0),
    "C_carbohydrate_60_110": (60.0, 110.0),
    "C_ar_HC_110_145": (110.0, 145.0),
    "C_phenolic_145_165": (145.0, 165.0),
}

# Минимальная доля пула (% от общего C), ниже которой отношение считается невычислимым (NaN)
NMR_MIN_DENOMINATOR_PCT = 1e-4


def integrate_nmr_region(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    ppm_min: float,
    ppm_max: float
) -> float:
    """Точное численное интегрирование спектра в заданном диапазоне химических сдвигов (ppm)."""
    mask = (chemical_shifts >= ppm_min) & (chemical_shifts <= ppm_max)
    if np.sum(mask) < 2:
        return 0.0
    x_sub = chemical_shifts[mask]
    y_sub = intensity[mask]
    if len(x_sub) >= 2 and x_sub[0] > x_sub[-1]:
        x_sub = x_sub[::-1]
        y_sub = y_sub[::-1]
    val = simpson(y_sub, x=x_sub)
    return float(max(0.0, val))


def _safe_ratio(num: float, den: float, min_den: float = NMR_MIN_DENOMINATOR_PCT) -> float:
    """Отношение пулов; NaN, если знаменатель не определен или пренебрежимо мал."""
    if not (np.isfinite(num) and np.isfinite(den)) or den < min_den:
        return float("nan")
    return float(num / den)


def calculate_13c_nmr_descriptors(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    sample_id: str = "NMR_Sample"
) -> Dict[str, float]:
    """
    Количественный расчет функционального состава углеродного скелета по 13C CP/MAS ЯМР.

    Схема интегрирования (NMR_13C_FINE_BINS; Kögel-Knabner 2002; Baldock et al. 1997):
    alkyl 0–45, N-alkyl/methoxyl 45–60, O-alkyl 60–90, acetal/anomeric 90–110,
    aromatic 110–145, phenolic/O-aryl 145–165, carboxyl 165–190, carbonyl 190–220 ppm.
    Шесть зон IHSS-типа (alkyl, C-O alkyl 45–90, acetal, aromatic, phenolic, carboxyl 165–220)
    получаются суммированием — см. ihss_6bin_fractions().

    Ключи (% от общего C 0–220 ppm):
    - NMR_C_aliph_pct (0–45), NMR_C_methoxy_pct (45–60), NMR_C_O_alkyl_60_90_pct,
      NMR_C_acetal_pct (90–110), NMR_C_ar_HC_pct (110–145), NMR_C_phenolic_pct (145–165),
      NMR_C_carboxyl_pct (165–190), NMR_C_carbonyl_pct (190–220)
    - агрегаты: NMR_C_O_alkyl_pct (45–110), NMR_C_carbohydrate_pct (60–110),
      NMR_C_aromatic_pct (110–165)
    Индексы:
    - NMR_fa_aromaticity: ароматичность fa = C(110–165) / C(0–220) × 100
    - NMR_hb_hi_index: HB/HI = [C(0–45) + C(110–165)] / [C(45–110) + C(165–220)]
      (гидрофобные / гидрофильные пулы; ср. Spaccini & Piccolo 2009, где карбонилы 190–220 не учитываются)
    - NMR_alkyl_o_alkyl: A/O-A = C(0–45) / C(45–110) (Baldock et al. 1997)
    - NMR_lignin_signature: C(145–165) + C(45–60), %
    - NMR_coo_ar_ratio: C(165–190) / C(110–165)

    Невычислимые величины возвращаются как NaN: все доли — для пустого спектра
    (суммарный интеграл ≤ 0), отношения — при знаменателе < NMR_MIN_DENOMINATOR_PCT %.
    Оценка Mw по ароматичности (ключ NMR_estimated_mw) удалена: регрессия
    «Mw = 1250·exp(0.023·fa)» не имеет проверяемого источника.
    """
    res: Dict[str, float] = {"Sample_ID": sample_id}

    ints = {
        name: integrate_nmr_region(chemical_shifts, intensity, lo, hi)
        for name, (lo, hi) in NMR_13C_FINE_BINS.items()
    }
    total_int = float(sum(ints.values()))
    if not np.isfinite(total_int) or total_int <= 0.0:
        pct = {k: float("nan") for k in ints}  # пустой спектр: доли не определены
    else:
        pct = {k: (v / total_int) * 100.0 for k, v in ints.items()}

    pct_aliph = pct["alkyl"]
    pct_methoxy = pct["n_o_alkyl"]
    pct_o_alkyl_60_90 = pct["o_alkyl"]
    pct_acetal = pct["acetal"]
    pct_ar_hc = pct["aromatic"]
    pct_phenolic = pct["phenolic"]
    pct_carboxyl = pct["carboxyl"]
    pct_carbonyl = pct["carbonyl"]

    pct_o_alkyl = pct_methoxy + pct_o_alkyl_60_90 + pct_acetal     # 45–110 ppm
    pct_carbohydrate = pct_o_alkyl_60_90 + pct_acetal              # 60–110 ppm
    pct_aromatic = pct_ar_hc + pct_phenolic                        # 110–165 ppm

    fa = pct_aromatic
    hb_hi = _safe_ratio(pct_aliph + pct_aromatic, pct_o_alkyl + pct_carboxyl + pct_carbonyl)
    alkyl_o_alkyl = _safe_ratio(pct_aliph, pct_o_alkyl)
    lignin_sig = pct_phenolic + pct_methoxy
    coo_ar = _safe_ratio(pct_carboxyl, pct_aromatic)

    res.update({
        "NMR_C_aliph_pct": round(pct_aliph, 2),
        "NMR_C_O_alkyl_pct": round(pct_o_alkyl, 2),
        "NMR_C_methoxy_pct": round(pct_methoxy, 2),
        "NMR_C_O_alkyl_60_90_pct": round(pct_o_alkyl_60_90, 2),
        "NMR_C_acetal_pct": round(pct_acetal, 2),
        "NMR_C_carbohydrate_pct": round(pct_carbohydrate, 2),
        "NMR_C_aromatic_pct": round(pct_aromatic, 2),
        "NMR_C_ar_HC_pct": round(pct_ar_hc, 2),
        "NMR_C_phenolic_pct": round(pct_phenolic, 2),
        "NMR_C_carboxyl_pct": round(pct_carboxyl, 2),
        "NMR_C_carbonyl_pct": round(pct_carbonyl, 2),
        "NMR_fa_aromaticity": round(fa, 2),
        "NMR_hb_hi_index": round(hb_hi, 3),
        "NMR_alkyl_o_alkyl": round(alkyl_o_alkyl, 3),
        "NMR_lignin_signature": round(lignin_sig, 2),
        "NMR_coo_ar_ratio": round(coo_ar, 3),
    })
    return res


def ihss_6bin_fractions(descriptors: Dict[str, float]) -> Dict[str, float]:
    """
    Шесть интегральных зон IHSS-типа (% общего C) из словаря calculate_13c_nmr_descriptors:
    Alkyl 0–45, C-O alkyl 45–90, Acetal 90–110, Aromatic 110–145, Phenolic 145–165,
    Carboxyl/carbonyl 165–220 ppm. NaN, если доли не определены.
    """
    key_of = {
        "alkyl": "NMR_C_aliph_pct", "n_o_alkyl": "NMR_C_methoxy_pct", "o_alkyl": "NMR_C_O_alkyl_60_90_pct",
        "acetal": "NMR_C_acetal_pct", "aromatic": "NMR_C_ar_HC_pct", "phenolic": "NMR_C_phenolic_pct",
        "carboxyl": "NMR_C_carboxyl_pct", "carbonyl": "NMR_C_carbonyl_pct",
    }
    out: Dict[str, float] = {}
    for label, parts in NMR_13C_IHSS_6BIN.items():
        out[label] = float(sum(float(descriptors.get(key_of[p], float("nan"))) for p in parts))
    return out


# ==============================================================================
# 4. ПОИСК И АННОТАЦИЯ ПИКОВ ЯМР (PEAK PICKING & ASSIGNMENT)
# ==============================================================================

# Справочная база химических отнесений 13C ЯМР для гуминовых систем и лигнинов
NMR_13C_PEAK_ANNOTATIONS = [
    (14.5, 3.0, "Метильные группы терминальных цепей (-CH3)", "Terminal methyl groups"),
    (22.5, 4.0, "Метиленовые группы в β-положении к концевым группам", "Methylene β to terminal methyl"),
    (29.8, 3.5, "Полиметиленовые цепи восков и липидов -(CH2)n-", "Polymethylene lipid/wax chains"),
    (39.5, 2.0, "Остаточный растворитель DMSO-d6 (септет)", "Residual DMSO-d6 solvent"),
    (56.1, 2.5, "Метоксильные группы лигнина (-OCH3)", "Lignin methoxyl groups (-OCH3)"),
    (63.0, 3.0, "Углерод C6 гексоз / алифатические спирты C-OH", "Hexose C6 / Aliphatic C-OH"),
    (72.5, 4.0, "Углероды C2, C3, C5 кольца целлюлозы и полисахаридов", "Cellulose/hemicellulose C2,C3,C5"),
    (83.0, 4.0, "Углерод C4 полисахаридов (кристаллическая/аморфная целлюлоза)", "Carbohydrate C4 (cellulose)"),
    (105.0, 4.5, "Аномерный углерод C1 полисахаридов / танинов (O-CH-O)", "Anomeric carbohydrate C1 / Tannins"),
    (115.5, 3.5, "Протонированный ароматический C-H (напр., C5 гваяцила в лигнине)", "Protonated aromatic C-H (e.g. guaiacyl C5 in lignin)"),
    (129.5, 4.0, "Алкил-ароматический углерод / конденсированные полиароматические кольца", "Alkyl-aromatic / condensed rings"),
    (138.0, 3.5, "Непротонированный ароматический углерод (C-1 лигнина)", "Non-protonated aromatic C-1"),
    (148.0, 3.0, "O-арильный C (фенолы/эфиры; напр., C3/C4 гваяцила, танины)", "O-aryl C (phenols/ethers; e.g. guaiacyl C3/C4, tannins)"),
    (153.0, 3.0, "O-арильный C (напр., C3/C5 сирингила в лигнине, флороглюцинол танинов)", "O-aryl C (e.g. syringyl C3/C5 in lignin, tannin rings)"),
    (174.5, 4.0, "Карбоксильные группы алифатических и ароматических кислот (-COOH)", "Carboxylic acid carbonyls (-COOH)"),
    (181.0, 3.5, "Амидный карбонил / карбоксилат-анионы (-COO-)", "Amide carbonyl / carboxylate salts"),
    (198.0, 5.0, "Хиноидный карбонил / α,β-ненасыщенные кетоны", "Quinoid carbonyl / conjugated ketones"),
    (206.0, 5.0, "Кетоны и альдегиды (несопряженный карбонил)", "Ketones / aldehydes (non-conjugated)"),
]


_NMR_FINE_BIN_LABELS = (
    "Alkyl C", "N-Alkyl / Methoxyl C", "O-Alkyl C", "Acetal / Anomeric C",
    "Aromatic C", "Phenolic / O-Aryl C", "Carboxyl C", "Carbonyl C",
)


def find_and_annotate_nmr_peaks(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    prominence: float = 0.05,
    distance_ppm: float = 3.0,
    lang: str = "ru"
) -> pd.DataFrame:
    """
    Автоматический поиск локальных максимумов (резонансов) в спектре 13C ЯМР
    с отнесением к структурным функциональным группам.
    """
    if len(intensity) < 10:
        return pd.DataFrame()
        
    y_norm = normalize_nmr_spectrum(intensity, method="max")
    
    # Оценка минимального расстояния в индексах
    ppm_step = float(np.median(np.abs(np.diff(chemical_shifts)))) if len(chemical_shifts) > 1 else 0.1
    dist_pts = max(1, int(round(distance_ppm / max(ppm_step, 1e-4))))
    
    peaks, props = scipy.signal.find_peaks(y_norm, prominence=prominence, distance=dist_pts)
    if len(peaks) == 0:
        return pd.DataFrame()
        
    records = []
    for pk in peaks:
        pk_ppm = float(chemical_shifts[pk])
        pk_int = float(intensity[pk])
        pk_rel = float(y_norm[pk])
        
        # Поиск ближайшего справочного отнесения
        best_assign = "Неидентифицированный резонанс" if lang == "ru" else "Unidentified resonance"
        min_d = 999.0
        
        for std_ppm, tol, desc_ru, desc_en in NMR_13C_PEAK_ANNOTATIONS:
            d = abs(pk_ppm - std_ppm)
            if d <= tol and d < min_d:
                min_d = d
                best_assign = desc_ru if lang == "ru" else desc_en
                
        # Определение главного региона (тонкие зоны NMR_13C_FINE_BINS)
        region = "Другое" if lang == "ru" else "Other"
        for (lo, hi), lbl in zip(NMR_13C_FINE_BINS.values(), _NMR_FINE_BIN_LABELS):
            if lo <= pk_ppm < hi or (hi == 220.0 and pk_ppm == hi):
                region = lbl
                break

        records.append({
            "Peak_Index": int(pk),
            "Chemical_Shift_ppm": round(pk_ppm, 2),
            "Intensity": round(pk_int, 4),
            "Relative_Height": round(pk_rel, 3),
            "Region": region,
            "Assignment": best_assign,
        })
        
    df_res = pd.DataFrame(records)
    # Сортировка по убыванию химического сдвига (стандарт ЯМР)
    df_res = df_res.sort_values("Chemical_Shift_ppm", ascending=False).reset_index(drop=True)
    return df_res


# ==============================================================================
# 5. РАЗЛОЖЕНИЕ АРОМАТИЧЕСКОЙ / O-АРИЛЬНОЙ ЗОНЫ (110–165 ppm)
# ==============================================================================

def gaussian(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
    """Гауссова спектральная полоса."""
    return amp * np.exp(-((x - cen) ** 2) / (2.0 * max(wid, 1e-4) ** 2))


def multi_gaussian_nmr(x: np.ndarray, *params) -> np.ndarray:
    """Сумма N гауссовых компонент."""
    y = np.zeros_like(x)
    n = len(params) // 3
    for i in range(n):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        y += gaussian(x, amp, cen, wid)
    return y


# Полосы разложения 110–165 ppm. Окна центров НЕ перекрываются (идентифицируемость модели).
# Отнесения намеренно обобщенные: в 13C спектрах ПОВ/гуминовых веществ зона 145–165 ppm содержит
# фенольный, эфирный (O-арильный) C лигнина, танинов и других полифенолов, поэтому разделение
# на сирингил/гваяцил/п-гидроксифенил (S/G/H) по 1D 13C спектру ПОВ не обосновано.
NMR_AROMATIC_SUBBANDS = {
    4: [
        {"label": "Protonated aromatic C-H (110–122 ppm)", "center": 116.0, "lo": 110.0, "hi": 122.0},
        {"label": "C-substituted / condensed aromatic C (122–140 ppm)", "center": 130.0, "lo": 122.0, "hi": 140.0},
        {"label": "O-aryl C, 140–151 ppm (phenolic / ether)", "center": 147.0, "lo": 140.0, "hi": 151.0},
        {"label": "O-aryl C, 151–165 ppm (di-O-substituted / ether)", "center": 155.0, "lo": 151.0, "hi": 165.0},
    ],
    3: [
        {"label": "Protonated aromatic C-H (110–122 ppm)", "center": 116.0, "lo": 110.0, "hi": 122.0},
        {"label": "C-substituted / condensed aromatic C (122–140 ppm)", "center": 130.0, "lo": 122.0, "hi": 140.0},
        {"label": "O-aryl C (140–165 ppm; phenolic / ether)", "center": 150.0, "lo": 140.0, "hi": 165.0},
    ],
}
NMR_DECONV_MIN_R2 = 0.95       # ниже — подгонка помечается как неудовлетворительная
NMR_DECONV_WIDTH_BOUNDS = (1.5, 12.0)  # σ, ppm


def deconvolve_aromatic_region(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    n_peaks: int = 4
) -> Dict[str, Any]:
    """
    Эмпирическое разложение ароматической / O-арильной зоны 13C ЯМР (110–165 ppm) на 3 или 4
    гауссовы подполосы (NMR_AROMATIC_SUBBANDS) с непересекающимися окнами центров:
    протонированный ароматический C-H, C-замещенный/конденсированный ароматический C и
    O-арильный C (одна полоса 140–165 ppm или две: 140–151 и 151–165 ppm).

    Это НЕ определение мономерного состава лигнина: S/G/H-соотношения по 1D 13C спектру
    природного ОВ не вычисляются, так как 145–165 ppm перекрывается с сигналами танинов,
    фенолов и других O-арильных структур.

    Returns dict:
    - success (bool) — curve_fit сошелся; message — причина неудачи
    - fit_ok (bool) — success и R² ≥ NMR_DECONV_MIN_R2 и ни один центр/ширина не на границе
    - warnings (list[str]) — флаги качества (низкий R², параметры на границах)
    - x, y, y_fit, r2, rmse, components (label, center_ppm, width_ppm, amplitude, area, area_pct,
      at_bound, curve), method_note
    При неудаче: components = [], y_fit = NaN, r2 = rmse = NaN.
    """
    method_note = (
        "Empirical Gaussian sub-band decomposition of 110–165 ppm. Bands are generic aromatic / "
        "O-aryl classes; no S/G/H lignin-unit ratios are derived from 1D 13C spectra of NOM."
    )
    mask = (chemical_shifts >= 110.0) & (chemical_shifts <= 165.0)
    x_sub = chemical_shifts[mask]
    y_sub = intensity[mask]

    def _fail(msg: str) -> Dict[str, Any]:
        return {
            "success": False,
            "fit_ok": False,
            "message": msg,
            "warnings": [msg],
            "x": x_sub,
            "y": y_sub,
            "y_fit": np.full_like(x_sub, np.nan, dtype=float),
            "components": [],
            "r2": float("nan"),
            "rmse": float("nan"),
            "method_note": method_note,
        }

    if len(x_sub) < 15 or np.max(y_sub) <= 1e-6:
        return _fail("Недостаточно спектральных точек или сигнала в диапазоне 110–165 ppm")

    # Гарантируем строгий возрастающий порядок по ppm для корректного интегрирования simpson и curve_fit
    if len(x_sub) >= 2 and x_sub[0] > x_sub[-1]:
        x_sub = x_sub[::-1]
        y_sub = y_sub[::-1]

    warnings: List[str] = []
    n_used = int(n_peaks)
    if n_used not in NMR_AROMATIC_SUBBANDS:
        n_used = 4 if n_used > 4 else 3
        warnings.append(f"n_peaks={n_peaks} not supported; {n_used} sub-bands used")
    templates = NMR_AROMATIC_SUBBANDS[n_used]

    y_scale = float(np.max(y_sub))
    y_norm = y_sub / y_scale
    w_lo, w_hi = NMR_DECONV_WIDTH_BOUNDS

    p0: List[float] = []
    bounds_lower: List[float] = []
    bounds_upper: List[float] = []
    for t in templates:
        idx_c = int(np.argmin(np.abs(x_sub - t["center"])))
        p0.extend([float(np.clip(y_norm[idx_c], 0.05, 1.5)), t["center"], 4.0])
        bounds_lower.extend([0.0, t["lo"], w_lo])
        bounds_upper.extend([2.0, t["hi"], w_hi])

    try:
        popt, _ = curve_fit(
            multi_gaussian_nmr,
            x_sub,
            y_norm,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
    except Exception as e:
        return _fail(f"Подгонка не сошлась: {e}")

    y_fit = multi_gaussian_nmr(x_sub, *popt)
    ss_res = float(np.sum((y_norm - y_fit) ** 2))
    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_norm - y_fit) ** 2)))
    if not np.isfinite(r2) or r2 < NMR_DECONV_MIN_R2:
        warnings.append(f"Low fit quality: R² = {r2:.3f} < {NMR_DECONV_MIN_R2}")

    comp_areas = []
    for i in range(len(templates)):
        c_curve = gaussian(x_sub, popt[3 * i], popt[3 * i + 1], popt[3 * i + 2])
        comp_areas.append(float(simpson(c_curve, x=x_sub)))
    total_comp_area = float(sum(comp_areas))

    components = []
    any_at_bound = False
    for i, t in enumerate(templates):
        amp_val, cen_val, wid_val = (float(v) for v in popt[3 * i:3 * i + 3])
        tol_c = 1e-3 * (t["hi"] - t["lo"])
        at_bound = bool(
            abs(cen_val - t["lo"]) <= tol_c or abs(cen_val - t["hi"]) <= tol_c
            or abs(wid_val - w_lo) <= 1e-3 or abs(wid_val - w_hi) <= 1e-3
        )
        if at_bound:
            any_at_bound = True
            warnings.append(f"'{t['label']}': centre or width at its bound — band not identifiable")
        pct_area = (comp_areas[i] / total_comp_area) * 100.0 if total_comp_area > 0 else float("nan")
        components.append({
            "band_id": i + 1,
            "label": t["label"],
            "center_ppm": round(cen_val, 2),
            "amplitude": round(amp_val, 3),
            "width_ppm": round(wid_val, 2),
            "area": round(comp_areas[i], 3),
            "area_pct": round(pct_area, 1),
            "at_bound": at_bound,
            "curve": gaussian(x_sub, amp_val, cen_val, wid_val) * y_scale,
        })

    fit_ok = bool(np.isfinite(r2) and r2 >= NMR_DECONV_MIN_R2 and not any_at_bound)
    return {
        "success": True,
        "fit_ok": fit_ok,
        "message": "" if fit_ok else "; ".join(warnings),
        "warnings": warnings,
        "x": x_sub,
        "y": y_sub,
        "y_fit": y_fit * y_scale,
        "components": components,
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "method_note": method_note,
    }


# ==============================================================================
# 6. СПЕКТРАЛЬНАЯ БИБЛИОТЕКА ЭТАЛОНОВ IHSS И ШЛАМ-ЛИГНИНА (HQI MATCHING)
# ==============================================================================

def get_nmr_reference_library() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает спектральную библиотеку 8 аналитических эталонов IHSS и техногенных лигнинов.
    Включает паспортные распределения углерода и эталонные кривые 13C ЯМР.
    """
    return {
        "IHSS_SRFA": {
            "id": "IHSS_SRFA",
            "name_ru": "Стандарт фульвокислот р. Суванни (IHSS SRFA 1R101F)",
            "name_en": "Suwannee River Fulvic Acid Standard (IHSS SRFA)",
            "type": "Aquatic Fulvic Acid",
            "fa_expected": 23.5,
            "aliph_expected": 28.0,
            "o_alkyl_expected": 24.5,
            "carboxyl_expected": 18.5,
            "carbonyl_expected": 5.5,
            "phenolic_expected": 4.5,
            "description_ru": "Канонический акватический стандарт. Высокое содержание карбоксильных групп и алифатических цепей, умеренная ароматичность.",
            "description_en": "Canonical aquatic standard. High carboxyl and aliphatic content, moderate aromaticity.",
        },
        "IHSS_SRNOM": {
            "id": "IHSS_SRNOM",
            "name_ru": "Стандарт природного ОВ р. Суванни (IHSS SRNOM 2R101N)",
            "name_en": "Suwannee River Natural Organic Matter (IHSS SRNOM)",
            "type": "Aquatic NOM",
            "fa_expected": 27.8,
            "aliph_expected": 24.2,
            "o_alkyl_expected": 26.0,
            "carboxyl_expected": 16.5,
            "carbonyl_expected": 5.5,
            "phenolic_expected": 5.2,
            "description_ru": "Нативное негидролизованное природное растворенное органическое вещество, сбалансированный спектр.",
            "description_en": "Native unhydrolyzed aquatic natural organic matter with balanced profile.",
        },
        "IHSS_ESHA": {
            "id": "IHSS_ESHA",
            "name_ru": "Почвенные гуминовые кислоты Эллиотт (IHSS ESHA 1S102H)",
            "name_en": "Elliott Soil Humic Acid (IHSS ESHA)",
            "type": "Soil Humic Acid",
            "fa_expected": 42.5,
            "aliph_expected": 22.0,
            "o_alkyl_expected": 16.0,
            "carboxyl_expected": 14.5,
            "carbonyl_expected": 5.0,
            "phenolic_expected": 9.5,
            "description_ru": "Почвенный гуминовый стандарт. Высокая степень ароматичности, конденсированные структуры, развитые фенольные полосы.",
            "description_en": "Soil humic acid standard. High aromaticity, condensed aromatic backbone, pronounced phenolic signal.",
        },
        "IHSS_PPHA": {
            "id": "IHSS_PPHA",
            "name_ru": "Торфяные гуминовые кислоты Пахоки (IHSS PPHA 1S103H)",
            "name_en": "Pahokee Peat Humic Acid (IHSS PPHA)",
            "type": "Peat Humic Acid",
            "fa_expected": 45.2,
            "aliph_expected": 18.5,
            "o_alkyl_expected": 15.0,
            "carboxyl_expected": 16.0,
            "carbonyl_expected": 5.3,
            "phenolic_expected": 10.8,
            "description_ru": "Торфяной эталон высокой степени обугливания и гумификации. Максимальная ароматичность среди гуматов.",
            "description_en": "Peat humic standard with high humification and aromatic core.",
        },
        "NORDIC_NOM": {
            "id": "NORDIC_NOM",
            "name_ru": "Северное акватическое ОВ (Nordic Reservoir NOM 1R108N)",
            "name_en": "Nordic Aquatic Natural Organic Matter",
            "type": "Boreal Aquatic NOM",
            "fa_expected": 21.0,
            "aliph_expected": 25.0,
            "o_alkyl_expected": 36.5,
            "carboxyl_expected": 13.0,
            "carbonyl_expected": 4.5,
            "phenolic_expected": 3.8,
            "description_ru": "Бореальное природное органическое вещество. Доминирование полисахаридных и O-алкильных компонентов (60-105 ppm).",
            "description_en": "Boreal aquatic NOM dominated by polysaccharide and O-alkyl moieties.",
        },
        "SOFTWOOD_LIGNIN": {
            "id": "SOFTWOOD_LIGNIN",
            "name_ru": "Хвойный крафт-лигнин (Softwood Kraft Lignin)",
            "name_en": "Softwood Kraft Lignin Standard",
            "type": "Technical Lignin",
            "fa_expected": 56.5,
            "aliph_expected": 12.0,
            "o_alkyl_expected": 15.5,
            "carboxyl_expected": 9.5,
            "carbonyl_expected": 6.5,
            "phenolic_expected": 18.2,
            "description_ru": "Технический хвойный лигнин. Преобладание гваяциловых субъединиц, узкий интенсивный пик метоксилов (56 ppm) и фенолов (148 ppm).",
            "description_en": "Softwood technical lignin dominated by guaiacyl units, intense 56 ppm methoxyl and 148 ppm phenolic peaks.",
        },
        "HARDWOOD_LIGNIN": {
            "id": "HARDWOOD_LIGNIN",
            "name_ru": "Лиственный органосольвентный лигнин (Hardwood Lignin)",
            "name_en": "Hardwood Organosolv Lignin",
            "type": "Organosolv Lignin",
            "fa_expected": 52.0,
            "aliph_expected": 11.5,
            "o_alkyl_expected": 19.0,
            "carboxyl_expected": 10.0,
            "carbonyl_expected": 7.5,
            "phenolic_expected": 15.5,
            "description_ru": "Лиственный лигнин со смешанной сирингил-гваяциловой структурой. Расщепленная ароматико-фенольная зона (148 и 153 ppm).",
            "description_en": "Hardwood lignin with mixed syringyl-guaiacyl matrix and split 148/153 ppm phenolic resonances.",
        },
        "BAIKAL_SLUDGE": {
            "id": "BAIKAL_SLUDGE",
            "name_ru": "Техногенный шлам-лигнин БЦБК (Baikal Sludge Lignin)",
            "name_en": "Baikal Technogenic Sludge Lignin",
            "type": "Industrial Sludge Waste",
            "fa_expected": 49.5,
            "aliph_expected": 16.5,
            "o_alkyl_expected": 14.0,
            "carboxyl_expected": 14.0,
            "carbonyl_expected": 6.0,
            "phenolic_expected": 14.5,
            "description_ru": "Окисленный промышленный шлам-лигнин Байкальского ЦБК. Сочетание высокой ароматичности, повышенного окисления (-COOH) и хлорированных метоксилов.",
            "description_en": "Oxidized industrial pulp mill sludge lignin from Lake Baikal with high aromaticity and oxidized carboxyl pool.",
        },
    }


def synthesize_reference_13c_curve(
    ref_id: str,
    ppm_grid: np.ndarray
) -> np.ndarray:
    """Генерирует модельный эталонный спектр 13C ЯМР на заданной сетке chemical_shifts ppm."""
    lib = get_nmr_reference_library()
    if ref_id not in lib:
        ref_id = "IHSS_SRNOM"
    meta = lib[ref_id]
    
    # Синтезируем характерные резонансы
    curve = np.zeros_like(ppm_grid)
    
    # 1. Алифатика (0-48 ppm)
    aliph_weight = meta["aliph_expected"] / 100.0
    curve += aliph_weight * 0.4 * gaussian(ppm_grid, amp=1.0, cen=15.0, wid=4.0)
    curve += aliph_weight * 1.2 * gaussian(ppm_grid, amp=1.0, cen=30.0, wid=6.0)
    curve += aliph_weight * 0.5 * gaussian(ppm_grid, amp=1.0, cen=42.0, wid=5.0)
    
    # 2. O-алкилы и метоксилы (48-108 ppm)
    o_alkyl_weight = meta["o_alkyl_expected"] / 100.0
    if "LIGNIN" in ref_id or "SLUDGE" in ref_id:
        # Мощный метоксильный пик 56 ppm
        curve += o_alkyl_weight * 1.5 * gaussian(ppm_grid, amp=1.0, cen=56.2, wid=2.2)
        curve += o_alkyl_weight * 0.6 * gaussian(ppm_grid, amp=1.0, cen=73.0, wid=7.0)
    else:
        # Преобладание углеводного плато 72 ppm и аномерного 105 ppm
        curve += o_alkyl_weight * 0.5 * gaussian(ppm_grid, amp=1.0, cen=56.0, wid=3.5)
        curve += o_alkyl_weight * 1.4 * gaussian(ppm_grid, amp=1.0, cen=72.5, wid=8.0)
        curve += o_alkyl_weight * 0.8 * gaussian(ppm_grid, amp=1.0, cen=104.0, wid=4.5)
        
    # 3. Ароматика и фенолы (108-165 ppm)
    ar_weight = meta["fa_expected"] / 100.0
    phen_weight = meta["phenolic_expected"] / 100.0
    curve += ar_weight * 1.2 * gaussian(ppm_grid, amp=1.0, cen=115.0, wid=6.0)
    curve += ar_weight * 1.8 * gaussian(ppm_grid, amp=1.0, cen=130.0, wid=7.0)
    
    # Фенольный резонанс 148-154 ppm
    if ref_id == "HARDWOOD_LIGNIN":
        curve += phen_weight * 2.0 * gaussian(ppm_grid, amp=1.0, cen=148.0, wid=3.5)
        curve += phen_weight * 2.2 * gaussian(ppm_grid, amp=1.0, cen=153.5, wid=3.5)
    else:
        curve += phen_weight * 3.5 * gaussian(ppm_grid, amp=1.0, cen=148.5, wid=4.5)
        
    # 4. Карбоксилы (165-190 ppm)
    carb_weight = meta["carboxyl_expected"] / 100.0
    curve += carb_weight * 2.0 * gaussian(ppm_grid, amp=1.0, cen=175.0, wid=5.5)
    
    # 5. Карбонилы (190-220 ppm)
    carbonyl_weight = meta["carbonyl_expected"] / 100.0
    curve += carbonyl_weight * 1.5 * gaussian(ppm_grid, amp=1.0, cen=205.0, wid=7.0)
    
    # Нормализация
    return normalize_nmr_spectrum(curve, method="area", chemical_shifts=ppm_grid)


def match_spectrum_against_library(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    top_n: int = 5,
    lang: str = "ru"
) -> pd.DataFrame:
    """
    Спектральная идентификация неизвестного образца по библиотеке стандартов IHSS / Лигнинов.
    Использует спектральный индекс соответствия Hit Quality Index (HQI, косинусное подобие в %):
    HQI = (x · y) / (||x|| * ||y||) * 100%
    """
    lib = get_nmr_reference_library()
    y_target = normalize_nmr_spectrum(intensity, method="area", chemical_shifts=chemical_shifts)
    
    results = []
    norm_target = np.linalg.norm(y_target)
    if norm_target < 1e-9:
        norm_target = 1.0
        
    for ref_id, meta in lib.items():
        y_ref = synthesize_reference_13c_curve(ref_id, chemical_shifts)
        norm_ref = np.linalg.norm(y_ref)
        if norm_ref < 1e-9:
            norm_ref = 1.0
            
        dot_prod = np.dot(y_target, y_ref)
        hqi = float(np.clip((dot_prod / (norm_target * norm_ref)) * 100.0, 0.0, 100.0))
        
        name = meta["name_ru"] if lang == "ru" else meta["name_en"]
        desc = meta["description_ru"] if lang == "ru" else meta["description_en"]
        
        results.append({
            "Reference_ID": ref_id,
            "Standard_Name": name,
            "HQI_Similarity_Pct": round(hqi, 2),
            "Expected_Aromaticity_fa": meta["fa_expected"],
            "Sample_Type": meta["type"],
            "Description": desc,
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("HQI_Similarity_Pct", ascending=False).head(top_n).reset_index(drop=True)
    return df_res


# ==============================================================================
# 7. ПАКЕТНАЯ ОБРАБОТКА И ГЕНЕРАТОР ДЕМО-ДАННЫХ (BATCH & BENCHMARK GENERATION)
# ==============================================================================

def batch_process_nmr_samples(
    samples: List[NMRSample],
    baseline_method: str = "asls",
    lam: float = 1e6
) -> pd.DataFrame:
    """
    Пакетная обработка списка образцов ЯМР:
    вычитание базовой линии, численное интегрирование и формирование
    сводной таблицы дескрипторов для Модуля 3 (Data Fusion).
    """
    records = []
    for s in samples:
        s_corr, _ = subtract_baseline(s, method=baseline_method, lam=lam)
        desc = calculate_13c_nmr_descriptors(
            s_corr.chemical_shifts,
            s_corr.corrected_intensity,
            sample_id=s_corr.sample_id
        )
        records.append(desc)
        
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    # Гарантируем, что Sample_ID идет первым столбцом
    cols = ["Sample_ID"] + [c for c in df.columns if c != "Sample_ID"]
    return df[cols]


def generate_synthetic_nmr_dataset(
    n_samples_per_class: int = 6,
    save_dir: Optional[str] = None
) -> Tuple[Dict[str, NMRSample], pd.DataFrame]:
    """
    Генерирует эталонный хемометрический бенчмарк 13C ЯМР спектров:
    - Класс 1: Baikal_Pristine (чистое природное растворенное органическое вещество)
      (низкая ароматичность fa ~20-25%, высокое содержание сахаров O-alkyl ~35-40%, низкий фенольный индекс).
    - Класс 2: Sludge_Lignin (техногенный шлам-лигнин БЦБК)
      (высокая ароматичность fa ~48-55%, мощные полосы метоксилов 56 ppm и фенольного углерода 148 ppm).
    """
    # Сетка химических сдвигов ppm от 0 до 220 с шагом 0.2 ppm (1101 точка)
    ppm_grid = np.linspace(0.0, 220.0, 1101)
    samples: Dict[str, NMRSample] = {}
    rows = []
    
    np.random.seed(42)
    
    for i in range(1, n_samples_per_class + 1):
        # 1. Природное ОВ Байкала
        s_id_b = f"Baikal_Pristine_{i:02d}"
        ref_b = synthesize_reference_13c_curve("NORDIC_NOM", ppm_grid)
        # Добавляем индивидуальную вариабельность и реалистичный шум
        noise_b = np.random.normal(0.0, 0.008, size=len(ppm_grid))
        var_aliph = np.random.uniform(0.9, 1.1)
        var_o_alk = np.random.uniform(0.95, 1.15)
        curve_b = np.maximum(ref_b * (0.8 * var_aliph + 0.2 * var_o_alk) + noise_b, 0.0)
        
        sample_b = NMRSample(
            sample_id=s_id_b,
            nucleus="13C",
            chemical_shifts=ppm_grid.copy(),
            intensity=curve_b,
            corrected_intensity=curve_b.copy(),
            metadata={"source": "Synthetic Baikal DOM", "class": "Baikal_Pristine"}
        )
        samples[s_id_b] = sample_b
        desc_b = calculate_13c_nmr_descriptors(ppm_grid, curve_b, sample_id=s_id_b)
        desc_b["Class_Target"] = "Baikal_Pristine"
        rows.append(desc_b)
        
        # 2. Шлам-лигнин БЦБК
        s_id_l = f"Sludge_Lignin_{i:02d}"
        ref_l = synthesize_reference_13c_curve("BAIKAL_SLUDGE", ppm_grid)
        noise_l = np.random.normal(0.0, 0.008, size=len(ppm_grid))
        var_ar = np.random.uniform(0.95, 1.15)
        curve_l = np.maximum(ref_l * var_ar + noise_l, 0.0)
        
        sample_l = NMRSample(
            sample_id=s_id_l,
            nucleus="13C",
            chemical_shifts=ppm_grid.copy(),
            intensity=curve_l,
            corrected_intensity=curve_l.copy(),
            metadata={"source": "Synthetic Baikal Sludge Lignin", "class": "Sludge_Lignin"}
        )
        samples[s_id_l] = sample_l
        desc_l = calculate_13c_nmr_descriptors(ppm_grid, curve_l, sample_id=s_id_l)
        desc_l["Class_Target"] = "Sludge_Lignin"
        rows.append(desc_l)
        
    df_desc = pd.DataFrame(rows)
    
    # Сохранение на диск при наличии save_dir
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for s_name, s_obj in samples.items():
            f_path = os.path.join(save_dir, f"{s_name}.csv")
            df_spec = pd.DataFrame({
                "Chemical_Shift_ppm": s_obj.chemical_shifts,
                "Intensity": s_obj.intensity
            })
            df_spec.to_csv(f_path, index=False)
            
        mat_path = os.path.join(save_dir, "nmr_descriptors_matrix.csv")
        df_desc.to_csv(mat_path, index=False)
        
    return samples, df_desc
