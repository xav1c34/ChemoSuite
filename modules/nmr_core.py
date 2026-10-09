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

def parse_jcamp_dx(text: str, sample_id: str = "NMR_Sample") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Парсер международного формата IUPAC JCAMP-DX для спектров ЯМР.
    Поддерживает Bruker TopSpin, Varian/Agilent, Mnova, JEOL экспорты.
    Обрабатывает метаданные (##XUNITS, ##FIRSTX, ##LASTX, ##NPOINTS)
    и блоки данных (##PEAK TABLE=, ##XYDATA=).
    """
    lines = text.splitlines()
    meta: Dict[str, Any] = {}
    x_vals: List[float] = []
    y_vals: List[float] = []
    
    in_data_block = False
    data_format = ""
    first_x = None
    last_x = None
    n_points = None
    x_factor = 1.0
    y_factor = 1.0
    
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
            
        if line_s.startswith("##"):
            # Строка метаданных или директивы
            parts = line_s[2:].split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip() if len(parts) > 1 else ""
            meta[key] = val
            
            if key == "FIRSTX":
                try:
                    first_x = float(val)
                except ValueError:
                    pass
            elif key == "LASTX":
                try:
                    last_x = float(val)
                except ValueError:
                    pass
            elif key == "NPOINTS":
                try:
                    n_points = int(val)
                except ValueError:
                    pass
            elif key == "XFACTOR":
                try:
                    x_factor = float(val)
                except ValueError:
                    pass
            elif key == "YFACTOR":
                try:
                    y_factor = float(val)
                except ValueError:
                    pass
            elif "XYDATA" in key or "PEAK TABLE" in key or "DATA TABLE" in key:
                in_data_block = True
                data_format = val
                continue
            elif key == "END":
                in_data_block = False
                continue
        elif in_data_block:
            # Чтение точек данных
            # Строки могут иметь формат: X Y1 Y2 Y3 ... или X, Y
            clean_l = line_s.replace(";", " ").replace(",", " ")
            tokens = clean_l.split()
            if not tokens:
                continue
                
            try:
                # Первое число — текущий X (химический сдвиг)
                cur_x = float(tokens[0]) * x_factor
                remaining = tokens[1:]
                if remaining:
                    for y_tok in remaining:
                        try:
                            y_v = float(y_tok) * y_factor
                            x_vals.append(cur_x)
                            y_vals.append(y_v)
                        except ValueError:
                            continue
            except ValueError:
                continue
                
    if not x_vals or not y_vals:
        raise ValueError(f"Не удалось извлечь спектральные координаты из JCAMP-DX файла '{sample_id}'.")
        
    x_arr = np.array(x_vals, dtype=np.float64)
    y_arr = np.array(y_vals, dtype=np.float64)
    
    # Сортировка по возрастанию химсдвигов ppm
    sort_idx = np.argsort(x_arr)
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
    1. Формат IUPAC JCAMP-DX.
    2. Табличные спектры приборов Bruker (TopSpin), Varian, JEOL, Mestrelab Mnova.
    3. Автоматически распознает десятичные запятые, разделители (запятая, точка с запятой, табуляция, пробел).
    4. Автоматически сортирует шкалу химических сдвигов ppm по возрастанию.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    text_clean = text.strip()
    
    # 1. Проверка на формат JCAMP-DX
    if text_clean.startswith("##TITLE") or "##JCAMP-DX" in text_clean:
        try:
            x_arr, y_arr, meta = parse_jcamp_dx(text_clean, sample_id=sample_id)
            # Определение ядра из метаданных, если указано
            nuc = nucleus
            if ".OBSERVENUCLEUS" in meta:
                raw_nuc = meta[".OBSERVENUCLEUS"].upper()
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
        except Exception:
            pass  # Fallback к обычному табличному парсеру
            
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
    # Обычно ppm лежит в диапазоне [-50, 300] для 13C или [-5, 20] для 1H
    col0_range = (arr[:, 0].min(), arr[:, 0].max())
    col1_range = (arr[:, 1].min(), arr[:, 1].max())
    
    if -50 <= col0_range[0] and col0_range[1] <= 300:
        ppm_idx, int_idx = 0, 1
    elif -50 <= col1_range[0] and col1_range[1] <= 300:
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
    Например, юстировка по TMS / DSS (0.0 ppm) или остаточному DMSO-d6 (39.5 ppm в 13C).
    """
    delta = target_ref_ppm - current_ref_ppm
    return chemical_shifts + delta


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
                tot_int = simpson(y[mask], x=chemical_shifts[mask])
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

# Канонические границы химических сдвигов 13C ЯМР для природного органического вещества
# (IHSS, Perminova et al., Malcolm, Kögel-Knabner, Keeler & Maciel)
NMR_13C_REGIONS = {
    "C_aliph_0_48": (0.0, 48.0),          # Алифатический углерод без гетероатомов (метилы, метилены, липиды)
    "C_O_alkyl_48_108": (48.0, 108.0),    # Углерод при O/N гетероатомах (метоксилы, углеводы, аномерные)
    "C_aromatic_108_165": (108.0, 165.0), # Суммарный ароматический углерод
    "C_carboxyl_165_190": (165.0, 190.0), # Карбоксильный, сложноэфирный, амидный углерод
    "C_carbonyl_190_220": (190.0, 220.0), # Карбонильный углерод (кетоны, хиноны, альдегиды)
}

# Подзоны для детального функционального анализа
NMR_13C_SUBREGIONS = {
    "C_methoxy_48_60": (48.0, 60.0),         # Метоксильные группы лигнина (-OCH3)
    "C_carbohydrate_60_108": (60.0, 108.0),  # Спиртовые группы сахаров и аномерные C
    "C_ar_HC_108_145": (108.0, 145.0),       # Протонированный и C-замещенный ароматический углерод
    "C_phenolic_145_165": (145.0, 165.0),    # Фенольный и эфирный ароматический углерод (лигнин!)
}


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
    val = simpson(y_sub, x=x_sub)
    return float(max(0.0, val))


def calculate_13c_nmr_descriptors(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    sample_id: str = "NMR_Sample"
) -> Dict[str, float]:
    """
    Количественный расчет функционального состава углеродного скелета по 13C CP/MAS ЯМР.
    
    Вычисляет:
    1. Интегральные процентные доли углерода в 5 главных областях (сумма = 100%):
       - % C_aliph (0–48 ppm)
       - % C_O_alkyl (48–108 ppm)
       - % C_aromatic (108–165 ppm)
       - % C_carboxyl (165–190 ppm)
       - % C_carbonyl (190–220 ppm)
    2. Дополнительные функциональные доли:
       - % C_methoxy (48–60 ppm, лигнин)
       - % C_carbohydrate (60–108 ppm, полисахариды)
       - % C_phenolic (145–165 ppm, фенольный углерод лигнина)
       - % C_ar_HC (108–145 ppm, углеводородная ароматика)
    3. Структурно-хемометрические индексы:
       - fa_aromaticity (%): степень ароматичности = C_aromatic / Total * 100
       - hb_hi_index: гидрофобно-гидрофильный баланс = (C_aliph + C_aromatic) / (C_O_alkyl + C_carboxyl + C_carbonyl)
       - alkyl_o_alkyl: отношение алкил / O-алкил = C_aliph / C_O_alkyl (индекс микробной деградации/диагенеза)
       - phenolic_carbon_pct: доля фенольного углерода
       - lignin_signature_index: комплексный индекс лигнина = (% C_phenolic + % C_methoxy)
       - coo_ar_ratio: отношение карбоксил / ароматика = C_carboxyl / C_aromatic
       - estimated_mw: оценка Mw по калибровке Перминовой на основе ароматичности
    """
    res: Dict[str, float] = {"Sample_ID": sample_id}
    
    # Расчет абсолютных интегралов 5 главных зон
    int_aliph = integrate_nmr_region(chemical_shifts, intensity, 0.0, 48.0)
    int_o_alkyl = integrate_nmr_region(chemical_shifts, intensity, 48.0, 108.0)
    int_aromatic = integrate_nmr_region(chemical_shifts, intensity, 108.0, 165.0)
    int_carboxyl = integrate_nmr_region(chemical_shifts, intensity, 165.0, 190.0)
    int_carbonyl = integrate_nmr_region(chemical_shifts, intensity, 190.0, 220.0)
    
    total_int = int_aliph + int_o_alkyl + int_aromatic + int_carboxyl + int_carbonyl
    if total_int <= 1e-9:
        total_int = 1.0  # Защита от деления на ноль при пустом спектре
        
    # Процентные доли главных углеродных пулов (нормированы к 100%)
    pct_aliph = (int_aliph / total_int) * 100.0
    pct_o_alkyl = (int_o_alkyl / total_int) * 100.0
    pct_aromatic = (int_aromatic / total_int) * 100.0
    pct_carboxyl = (int_carboxyl / total_int) * 100.0
    pct_carbonyl = (int_carbonyl / total_int) * 100.0
    
    # Расчет подзон
    int_methoxy = integrate_nmr_region(chemical_shifts, intensity, 48.0, 60.0)
    int_carbohydrate = integrate_nmr_region(chemical_shifts, intensity, 60.0, 108.0)
    int_ar_hc = integrate_nmr_region(chemical_shifts, intensity, 108.0, 145.0)
    int_phenolic = integrate_nmr_region(chemical_shifts, intensity, 145.0, 165.0)
    
    pct_methoxy = (int_methoxy / total_int) * 100.0
    pct_carbohydrate = (int_carbohydrate / total_int) * 100.0
    pct_ar_hc = (int_ar_hc / total_int) * 100.0
    pct_phenolic = (int_phenolic / total_int) * 100.0
    
    # 1. Степень ароматичности fa (%)
    fa = pct_aromatic
    
    # 2. Индекс гидрофобности HB/HI
    hydrophobic = pct_aliph + pct_aromatic
    hydrophilic = pct_o_alkyl + pct_carboxyl + pct_carbonyl
    hb_hi = hydrophobic / max(hydrophilic, 1e-4)
    
    # 3. Отношение алкил / O-алкил (A/OA ratio)
    alkyl_o_alkyl = pct_aliph / max(pct_o_alkyl, 1e-4)
    
    # 4. Комплексный лигниновый маркер (Lignin Signature)
    lignin_sig = pct_phenolic + pct_methoxy
    
    # 5. Отношение Карбоксил / Ароматика
    coo_ar = pct_carboxyl / max(pct_aromatic, 1e-4)
    
    # 6. Оценка молекулярной массы Mw (Да) по регрессии Перминовой: Mw ~ 1250 * exp(0.023 * fa)
    mw_est = 1250.0 * math.exp(0.023 * fa)
    
    res.update({
        "NMR_C_aliph_pct": round(pct_aliph, 2),
        "NMR_C_O_alkyl_pct": round(pct_o_alkyl, 2),
        "NMR_C_methoxy_pct": round(pct_methoxy, 2),
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
        "NMR_estimated_mw": round(mw_est, 0),
    })
    return res


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
    (115.5, 3.5, "Гваяцил C5 / замещенные фенолы (протонированная ароматика)", "Guaiacyl C5 / Protonated aromatic C"),
    (129.5, 4.0, "Алкил-ароматический углерод / конденсированные полиароматические кольца", "Alkyl-aromatic / condensed rings"),
    (138.0, 3.5, "Непротонированный ароматический углерод (C-1 лигнина)", "Non-protonated aromatic C-1"),
    (148.0, 3.0, "Гваяцил C3, C4 (фенольный углерод с метоксилами)", "Guaiacyl C3/C4 phenolic C"),
    (153.0, 3.0, "Сирингил C3, C5 (эфирный/фенольный углерод лигнина)", "Syringyl C3/C5 etherified C"),
    (174.5, 4.0, "Карбоксильные группы алифатических и ароматических кислот (-COOH)", "Carboxylic acid carbonyls (-COOH)"),
    (181.0, 3.5, "Амидный карбонил / карбоксилат-анионы (-COO-)", "Amide carbonyl / carboxylate salts"),
    (198.0, 5.0, "Хиноидный карбонил / α,β-ненасыщенные кетоны", "Quinoid carbonyl / conjugated ketones"),
    (206.0, 5.0, "Кетоны и альдегиды (несопряженный карбонил)", "Ketones / aldehydes (non-conjugated)"),
]


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
                
        # Определение главного региона
        region = "Другое"
        if 0.0 <= pk_ppm < 48.0:
            region = "Aliphatic C"
        elif 48.0 <= pk_ppm < 108.0:
            region = "O/N-Alkyl C"
        elif 108.0 <= pk_ppm < 165.0:
            region = "Aromatic C"
        elif 165.0 <= pk_ppm < 190.0:
            region = "Carboxyl C"
        elif 190.0 <= pk_ppm <= 220.0:
            region = "Carbonyl C"
            
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
# 5. ДЕКОНВОЛЮЦИЯ АРОМАТИКО-ФЕНОЛЬНОЙ ЗОНЫ (108–165 ppm)
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


def deconvolve_aromatic_region(
    chemical_shifts: np.ndarray,
    intensity: np.ndarray,
    n_peaks: int = 4
) -> Dict[str, Any]:
    """
    Математическая деконволюция сложной перекрывающейся ароматико-фенольной зоны (108–165 ppm)
    в спектре 13C ЯМР на индивидуальные гауссовы компоненты.
    
    Компоненты отражают:
    - ~115 ppm: протонированный C5 (гваяцил)
    - ~130 ppm: алкил-ароматический / конденсированный C
    - ~148 ppm: гваяцил-фенольный C3/C4
    - ~153 ppm: сирингил-эфирный C3/C5
    """
    mask = (chemical_shifts >= 108.0) & (chemical_shifts <= 165.0)
    x_sub = chemical_shifts[mask]
    y_sub = intensity[mask]
    
    if len(x_sub) < 15 or np.max(y_sub) <= 1e-6:
        return {
            "success": False,
            "message": "Недостаточно спектральных точек в диапазоне 108–165 ppm",
            "x": x_sub,
            "y": y_sub,
            "y_fit": np.zeros_like(x_sub),
            "components": [],
            "r2": 0.0,
            "rmse": 0.0,
        }
        
    y_norm = y_sub / np.max(y_sub)
    
    # Начальные приближения центров полос
    initial_centers = [116.0, 130.0, 148.0, 154.0][:n_peaks]
    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    for cen in initial_centers:
        p0.extend([0.6, cen, 4.0])  # amp, cen, wid
        bounds_lower.extend([0.0, cen - 5.0, 1.5])
        bounds_upper.extend([2.0, cen + 5.0, 12.0])
        
    try:
        popt, _ = curve_fit(
            multi_gaussian_nmr,
            x_sub,
            y_norm,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        y_fit = multi_gaussian_nmr(x_sub, *popt)
        
        # Качество подгонки R2 и RMSE
        ss_res = np.sum((y_norm - y_fit) ** 2)
        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
        r2 = max(0.0, float(1.0 - (ss_res / max(ss_tot, 1e-9))))
        rmse = float(np.sqrt(np.mean((y_norm - y_fit) ** 2)))
        
        components = []
        total_comp_area = 0.0
        comp_areas = []
        
        for i in range(len(popt) // 3):
            amp_val = popt[3 * i]
            cen_val = popt[3 * i + 1]
            wid_val = popt[3 * i + 2]
            c_curve = gaussian(x_sub, amp_val, cen_val, wid_val)
            area_val = float(simpson(c_curve, x=x_sub))
            total_comp_area += area_val
            comp_areas.append(area_val)
            
        for i in range(len(popt) // 3):
            cen_val = popt[3 * i + 1]
            c_curve = gaussian(x_sub, popt[3 * i], cen_val, popt[3 * i + 2])
            pct_area = (comp_areas[i] / max(total_comp_area, 1e-9)) * 100.0
            
            # Назначение полосы
            lbl = f"Полоса {i+1} ({cen_val:.1f} ppm)"
            if 112.0 <= cen_val <= 120.0:
                lbl = "Guaiacyl C5 (115 ppm)"
            elif 125.0 <= cen_val <= 135.0:
                lbl = "Alkyl-Aromatic C (130 ppm)"
            elif 145.0 <= cen_val <= 150.0:
                lbl = "Guaiacyl Phenolic C3/C4 (148 ppm)"
            elif 151.0 <= cen_val <= 158.0:
                lbl = "Syringyl Ether C3/C5 (154 ppm)"
                
            components.append({
                "band_id": i + 1,
                "label": lbl,
                "center_ppm": round(cen_val, 2),
                "amplitude": round(popt[3 * i], 3),
                "width_ppm": round(popt[3 * i + 2], 2),
                "area": round(comp_areas[i], 3),
                "area_pct": round(pct_area, 1),
                "curve": c_curve * np.max(y_sub),
            })
            
        return {
            "success": True,
            "x": x_sub,
            "y": y_sub,
            "y_fit": y_fit * np.max(y_sub),
            "components": components,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "x": x_sub,
            "y": y_sub,
            "y_fit": np.zeros_like(x_sub),
            "components": [],
            "r2": 0.0,
            "rmse": 0.0,
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
