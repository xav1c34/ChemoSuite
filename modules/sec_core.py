"""
sec_core.py — Математическое ядро гель-проникающей и эксклюзионной хроматографии (SEC / GPC).
Методология хемометрического анализа молекулярно-массовых распределений (ММР / MWD),
полидисперсности, фракционирования природного органического вещества (РОВ/DOM, гуминовые кислоты)
и техногенного шлам-лигнина целлюлозно-бумажного производства.
"""
from dataclasses import dataclass, field
import io
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.signal
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _trapz_integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Безопасное интегрирование методом трапеций для совместимости со всеми версиями NumPy/SciPy."""
    if len(y) == 0:
        return 0.0
    try:
        return float(scipy.integrate.trapezoid(y, x))
    except (AttributeError, NameError):
        pass
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(getattr(np, "trapz")(y, x))
    return float(np.sum(y) * (x[1] - x[0])) if len(x) > 1 else float(np.sum(y))


def decode_bytes(b: bytes) -> str:
    """Универсальное декодирование байтов файла с поддержкой распространенных кодировок."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ==============================================================================
# ПРЕСЕТЫ ЭТАЛОННЫХ КАЛИБРОВОК SEC / GPC (PSS, PEG/PEO, PULLULAN)
# ==============================================================================
# log10(M) = a - b * V_e (или полином 3-й степени log10(M) = c0 + c1*V + c2*V^2 + c3*V^3)
# Типовые параметры для аналитической колонки (например, 7.8 x 300 мм, V0 ~ 5.5 мл, Vt ~ 11.5 мл):
CALIBRATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "PSS_Sodium_Sulfonate": {
        "name_en": "Sodium Polystyrene Sulfonate (PSS)",
        "name_ru": "Полистиролсульфонат натрия (PSS)",
        "standards": [
            {"name": "PSS-1k", "mw": 1100.0, "vr": 10.25},
            {"name": "PSS-3.5k", "mw": 3600.0, "vr": 9.40},
            {"name": "PSS-8k", "mw": 8200.0, "vr": 8.55},
            {"name": "PSS-18k", "mw": 17800.0, "vr": 7.75},
            {"name": "PSS-32k", "mw": 32000.0, "vr": 7.15},
            {"name": "PSS-65k", "mw": 65400.0, "vr": 6.45},
            {"name": "PSS-120k", "mw": 122000.0, "vr": 5.85},
        ],
        "default_coeffs": [7.38, -0.428],  # log10(M) = 7.38 - 0.428 * V
        "mw_range": (500.0, 150000.0),
    },
    "PEG_PEO_Standards": {
        "name_en": "Polyethylene Glycol / Oxide (PEG/PEO)",
        "name_ru": "Полиэтиленгликоль / оксид (ПЭГ/ПЭО)",
        "standards": [
            {"name": "PEG-600", "mw": 600.0, "vr": 10.60},
            {"name": "PEG-1.5k", "mw": 1470.0, "vr": 9.85},
            {"name": "PEG-4k", "mw": 4120.0, "vr": 8.90},
            {"name": "PEG-10k", "mw": 10100.0, "vr": 8.00},
            {"name": "PEO-24k", "mw": 23800.0, "vr": 7.20},
            {"name": "PEO-50k", "mw": 49000.0, "vr": 6.35},
        ],
        "default_coeffs": [7.45, -0.442],
        "mw_range": (300.0, 100000.0),
    },
    "Pullulan_Polysaccharides": {
        "name_en": "Pullulan Polysaccharides",
        "name_ru": "Пуллулановые стандарты",
        "standards": [
            {"name": "Pul-1k", "mw": 1080.0, "vr": 10.35},
            {"name": "Pul-6k", "mw": 5900.0, "vr": 8.95},
            {"name": "Pul-12k", "mw": 11800.0, "vr": 8.20},
            {"name": "Pul-23k", "mw": 22800.0, "vr": 7.45},
            {"name": "Pul-47k", "mw": 47300.0, "vr": 6.60},
            {"name": "Pul-112k", "mw": 112000.0, "vr": 5.75},
        ],
        "default_coeffs": [7.52, -0.435],
        "mw_range": (500.0, 150000.0),
    }
}


# ==============================================================================
# КОНТЕЙНЕРЫ ДАННЫХ (DATACLASSES)
# ==============================================================================

@dataclass
class SECCalibration:
    """Параметры калибровочной кривой эксклюзионной хроматографии log10(M) = f(V)."""
    calibration_type: str = "PSS_Sodium_Sulfonate"
    coefficients: List[float] = field(default_factory=lambda: [7.38, -0.428])  # [a, b] -> log10(M) = a + b*V (b отрицательный)
    degree: int = 1
    r_squared: float = 0.998
    mw_min: float = 300.0
    mw_max: float = 200000.0
    v0: float = 5.2    # Объем полного исключения (мл)
    vt: float = 11.8   # Объем полного проникновения (мл)

    def v_to_log_mw(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Пересчет элюционного объема (или времени) в log10(MW)."""
        if self.degree == 1:
            a, b = self.coefficients[0], self.coefficients[1]
            return a + b * v
        else:
            # Полиномиальная калибровка: c0 + c1*v + c2*v^2 + ...
            res = np.zeros_like(v, dtype=float)
            for power, coef in enumerate(self.coefficients):
                res += coef * (v ** power)
            return res

    def v_to_mw(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Пересчет элюционного объема в молекулярную массу (Да)."""
        log_m = self.v_to_log_mw(v)
        m = 10.0 ** np.clip(log_m, 1.0, 7.0)
        return m


@dataclass
class SECPeak:
    """Идентифицированный хроматографический пик фракции."""
    peak_id: int
    retention_time: float
    elution_volume: float
    height: float
    area: float
    area_pct: float
    peak_mw: float       # Mp (пиковая молекулярная масса, Да)
    fraction_name: str   # "HMW", "MHMW", "MMW", "LMW"


@dataclass
class SECSample:
    """Контейнер для отдельного хроматографического анализа SEC / GPC."""
    sample_id: str
    retention_times: np.ndarray = field(default_factory=lambda: np.array([]))  # мин
    elution_volume: np.ndarray = field(default_factory=lambda: np.array([]))   # мл
    signal_raw: np.ndarray = field(default_factory=lambda: np.array([]))       # мAU или mV
    baseline: Optional[np.ndarray] = None
    signal_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    calibration: SECCalibration = field(default_factory=SECCalibration)
    log_mw: np.ndarray = field(default_factory=lambda: np.array([]))
    mw: np.ndarray = field(default_factory=lambda: np.array([]))
    mwd_diff: np.ndarray = field(default_factory=lambda: np.array([]))          # Дифференциальное ММР dW/d(logM)
    mwd_cum: np.ndarray = field(default_factory=lambda: np.array([]))           # Интегральное ММР
    peaks: List[SECPeak] = field(default_factory=list)
    indices: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. ПАРСИНГ ХРОМАТОГРАММ SEC / GPC (CSV, TSV, TXT, ASCII)
# ==============================================================================

def baseline_als_sec(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.005,
    niter: int = 15
) -> np.ndarray:
    """
    Асимметричные наименьшие квадраты (Asymmetric Least Squares, ALS) для базовой линии SEC.
    Бережно огибает элюционные пики снизу, убирая дрейф детектора.
    """
    L = len(y)
    if L < 5:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L, L - 2), format="csc", dtype=float)
    D_mat = lam * (D @ D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + D_mat
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return np.asarray(z)


def fit_sec_calibration(
    standards_v: Union[List[float], np.ndarray],
    standards_mw: Union[List[float], np.ndarray],
    degree: int = 1
) -> SECCalibration:
    """
    Строит калибровочную кривую log10(MW) = f(V) по набору эталонных стандартов.
    """
    v_arr = np.asarray(standards_v, dtype=float)
    mw_arr = np.asarray(standards_mw, dtype=float)
    log_mw = np.log10(np.clip(mw_arr, 10.0, None))

    if len(v_arr) < 2:
        return SECCalibration()

    deg = min(degree, len(v_arr) - 1)
    if deg == 1:
        # Линейная регрессия y = a + b*x
        slope, intercept = np.polyfit(v_arr, log_mw, deg=1)
        coeffs = [float(intercept), float(slope)]
        pred = intercept + slope * v_arr
    else:
        # Полином p(x) = c0 + c1*x + c2*x^2 ...
        poly_p = np.polyfit(v_arr, log_mw, deg=deg)
        coeffs = [float(c) for c in poly_p[::-1]]  # c0, c1, c2...
        pred = np.polyval(poly_p, v_arr)

    ss_res = np.sum((log_mw - pred) ** 2)
    ss_tot = np.sum((log_mw - np.mean(log_mw)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return SECCalibration(
        calibration_type="Fitted_Custom",
        coefficients=coeffs,
        degree=deg,
        r_squared=round(float(max(0.0, min(1.0, r2))), 4),
        mw_min=float(np.min(mw_arr)),
        mw_max=float(np.max(mw_arr)),
        v0=float(np.min(v_arr) * 0.85),
        vt=float(np.max(v_arr) * 1.15)
    )


def calculate_sec_indices(
    vol_or_time: np.ndarray,
    signal: np.ndarray,
    calibration: Optional[SECCalibration] = None,
    mw_fraction_limits: Tuple[float, float, float] = (20000.0, 5000.0, 1000.0)
) -> Dict[str, float]:
    """
    Рассчитывает молекулярно-массовые характеристики:
    - Mn (среднечисловая MW);
    - Mw (среднемассовая MW);
    - Mz (z-средняя MW);
    - Mp (пиковая молекулярная масса);
    - PDI / Đ (полидисперсность Mw / Mn);
    - Доли фракций:
      * Фракция I (> 20 кДа): макромолекулярные агрегаты гуминовых кислот / шлам-лигнина;
      * Фракция II (5 - 20 кДа): средне-тяжелый лигнин;
      * Фракция III (1 - 5 кДа): типичное растворенное РОВ (DOM);
      * Фракция IV (< 1 кДа): низкомолекулярные полифенолы и кислоты.
    - Отношение HMW / LMW.
    """
    calib = calibration or SECCalibration()
    if len(vol_or_time) < 5 or np.all(signal <= 0):
        return {
            "SEC_Mn": 0.0, "SEC_Mw": 0.0, "SEC_Mz": 0.0, "SEC_Mp": 0.0, "SEC_PDI": 1.0,
            "SEC_Fraction_I_pct": 0.0, "SEC_Fraction_II_pct": 0.0,
            "SEC_Fraction_III_pct": 0.0, "SEC_Fraction_IV_pct": 0.0,
            "SEC_HMW_LMW_ratio": 0.0,
        }

    # Массив молекулярных масс M(i)
    mw_arr = calib.v_to_mw(vol_or_time)
    mw_arr = np.clip(mw_arr, 50.0, 1e7)

    # Очищаем сигнал от отрицательных шумов
    s_pos = np.clip(signal, 0.0, None)
    tot_s = np.sum(s_pos)
    if tot_s <= 1e-12:
        return {
            "SEC_Mn": 0.0, "SEC_Mw": 0.0, "SEC_Mz": 0.0, "SEC_Mp": 0.0, "SEC_PDI": 1.0,
            "SEC_Fraction_I_pct": 0.0, "SEC_Fraction_II_pct": 0.0,
            "SEC_Fraction_III_pct": 0.0, "SEC_Fraction_IV_pct": 0.0,
            "SEC_HMW_LMW_ratio": 0.0,
        }

    # Интегральные молекулярные средние
    # Mn = sum(S) / sum(S / M)
    sum_s_over_m = np.sum(s_pos / mw_arr)
    mn = tot_s / sum_s_over_m if sum_s_over_m > 1e-12 else 0.0

    # Mw = sum(S * M) / sum(S)
    sum_s_m = np.sum(s_pos * mw_arr)
    mw = sum_s_m / tot_s

    # Mz = sum(S * M^2) / sum(S * M)
    sum_s_m2 = np.sum(s_pos * (mw_arr ** 2))
    mz = sum_s_m2 / sum_s_m if sum_s_m > 1e-12 else mw

    # Mp = MW в точке максимума сигнала
    max_idx = np.argmax(s_pos)
    mp = float(mw_arr[max_idx])

    # Полидисперсность
    pdi = mw / mn if mn > 1e-6 else 1.0
    pdi = max(1.0, pdi)

    # Расчет долей фракций
    lim_high, lim_mid, lim_low = mw_fraction_limits
    mask_f1 = (mw_arr >= lim_high)
    mask_f2 = (mw_arr >= lim_mid) & (mw_arr < lim_high)
    mask_f3 = (mw_arr >= lim_low) & (mw_arr < lim_mid)
    mask_f4 = (mw_arr < lim_low)

    f1_pct = (np.sum(s_pos[mask_f1]) / tot_s) * 100.0
    f2_pct = (np.sum(s_pos[mask_f2]) / tot_s) * 100.0
    f3_pct = (np.sum(s_pos[mask_f3]) / tot_s) * 100.0
    f4_pct = (np.sum(s_pos[mask_f4]) / tot_s) * 100.0

    hmw = f1_pct + f2_pct
    lmw = max(f4_pct, 1e-4)
    hmw_lmw_ratio = hmw / lmw

    return {
        "SEC_Mn": round(float(mn), 1),
        "SEC_Mw": round(float(mw), 1),
        "SEC_Mz": round(float(mz), 1),
        "SEC_Mp": round(float(mp), 1),
        "SEC_PDI": round(float(pdi), 2),
        "SEC_Polydispersity": round(float(pdi), 2),
        "SEC_Fraction_I_pct": round(float(f1_pct), 2),
        "SEC_Fraction_II_pct": round(float(f2_pct), 2),
        "SEC_Fraction_III_pct": round(float(f3_pct), 2),
        "SEC_Fraction_IV_pct": round(float(f4_pct), 2),
        "SEC_HMW_pct": round(float(hmw), 2),
        "SEC_LMW_pct": round(float(f4_pct), 2),
        "SEC_HMW_LMW_ratio": round(float(hmw_lmw_ratio), 3),
    }


def parse_sec_file(
    file_content: Union[str, bytes],
    sample_id: str = "SEC_Sample",
    flow_rate_ml_min: float = 1.0,
    calibration_preset: str = "PSS_Sodium_Sulfonate"
) -> SECSample:
    """
    Универсальный парсер файлов гель-проникающей хроматографии (SEC/GPC):
    - Распознает колонки времени (Time, RT) или объема (Volume, VR);
    - Распознает сигналы детектора (Signal, UV254, UV280, RI, Absorbance);
    - Выполняет коррекцию базовой линии ALS;
    - Применяет калибровочную кривую и рассчитывает дифференциальное/интегральное ММР;
    - Находит характерные пики и вычисляет хемометрические индексы.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    calib_dict = CALIBRATION_PRESETS.get(calibration_preset, CALIBRATION_PRESETS["PSS_Sodium_Sulfonate"])
    calib = SECCalibration(
        calibration_type=calibration_preset,
        coefficients=calib_dict["default_coeffs"],
        mw_min=calib_dict["mw_range"][0],
        mw_max=calib_dict["mw_range"][1]
    )

    if not lines:
        return SECSample(sample_id=sample_id, calibration=calib, indices=calculate_sec_indices(np.array([]), np.array([]), calib))

    # Распознаем разделитель таблицы
    first_line = lines[0]
    sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")

    df_raw = None
    try:
        df_raw = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        try:
            df_raw = pd.read_csv(io.StringIO(text), sep=r"\s+")
        except Exception:
            pass

    if df_raw is None or df_raw.empty:
        return SECSample(sample_id=sample_id, calibration=calib, indices=calculate_sec_indices(np.array([]), np.array([]), calib))

    # Карта колонок
    col_map = {c: str(c).strip().lower() for c in df_raw.columns}

    # Ищем колонку оси X (Time или Volume)
    x_col = None
    is_volume = False
    for orig, low in col_map.items():
        if any(k in low for k in ["vol", "volume", "объем", "v_e", "v_r", "ml", "мл"]):
            x_col = orig
            is_volume = True
            break
    if x_col is None:
        for orig, low in col_map.items():
            if any(k in low for k in ["time", "время", "rt", "retention", "min", "мин", "t_r"]):
                x_col = orig
                is_volume = False
                break

    # Ищем колонку сигнала Y
    y_col = None
    for orig, low in col_map.items():
        if orig != x_col and any(k in low for k in ["uv", "abs", "signal", "intensity", "сигнал", "ri", "mau", "mv", "dabs", "resp"]):
            y_col = orig
            break

    # Если эвристика не нашла, берем числовые колонки
    num_cols = [c for c in df_raw.columns if pd.to_numeric(df_raw[c], errors="coerce").notnull().sum() > len(df_raw) * 0.5]
    if x_col is None and len(num_cols) > 0:
        x_col = num_cols[0]
    if y_col is None and len(num_cols) > 1:
        y_col = num_cols[1]

    if x_col is None or y_col is None:
        return SECSample(sample_id=sample_id, calibration=calib, indices=calculate_sec_indices(np.array([]), np.array([]), calib))

    df_clean = df_raw.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col]).sort_values(by=x_col).reset_index(drop=True)

    x_vals = df_clean[x_col].values.astype(float)
    y_vals = df_clean[y_col].values.astype(float)

    if len(x_vals) < 5:
        return SECSample(sample_id=sample_id, calibration=calib, indices=calculate_sec_indices(np.array([]), np.array([]), calib))

    if is_volume:
        vol_arr = x_vals
        rt_arr = vol_arr / max(flow_rate_ml_min, 1e-4)
    else:
        rt_arr = x_vals
        vol_arr = rt_arr * flow_rate_ml_min

    # Базовая линия ALS
    base = baseline_als_sec(y_vals, lam=1e5, p=0.005)
    sig_corr = np.clip(y_vals - base, 0.0, None)

    # Расчет молекулярных масс
    mw_arr = calib.v_to_mw(vol_arr)
    log_mw = np.log10(np.clip(mw_arr, 1.0, None))

    # Дифференциальная кривая ММР: dw/d(logM)
    # Знак минус, так как чем больше объем V, тем меньше масса M (dV/dlogM отрицателен)
    d_log_m = np.gradient(log_mw)
    d_log_m_safe = np.where(np.abs(d_log_m) < 1e-9, -1e-9, d_log_m)
    mwd_diff = np.abs(sig_corr / d_log_m_safe)
    # Нормализация площади ММР к 100%
    if np.sum(mwd_diff) > 0:
        mwd_diff = (mwd_diff / np.sum(mwd_diff)) * 100.0

    # Интегральное ММР
    mwd_cum = np.cumsum(mwd_diff)
    if len(mwd_cum) > 0 and mwd_cum[-1] > 0:
        mwd_cum = (mwd_cum / mwd_cum[-1]) * 100.0

    # Пик-пикер на хроматограмме
    prom = max(np.max(sig_corr) * 0.05, 1e-3)
    p_indices, _ = scipy.signal.find_peaks(sig_corr, prominence=prom, distance=max(1, int(len(sig_corr) / 60)))
    peaks_list: List[SECPeak] = []
    tot_area = np.sum(sig_corr)

    for idx_num, pi in enumerate(p_indices, start=1):
        p_rt = float(rt_arr[pi])
        p_vr = float(vol_arr[pi])
        p_h = float(sig_corr[pi])
        p_mw = float(mw_arr[pi])
        # Оценка площади трапецией вокруг пика (+- 5 точек)
        li = max(0, pi - 5)
        ri = min(len(sig_corr) - 1, pi + 5)
        p_area = _trapz_integrate(sig_corr[li:ri+1], vol_arr[li:ri+1]) if ri > li else p_h
        p_area_pct = round((p_area / tot_area) * 100.0, 2) if tot_area > 0 else 0.0

        if p_mw >= 20000.0:
            f_name = "HMW (>20 kDa)"
        elif p_mw >= 5000.0:
            f_name = "MHMW (5-20 kDa)"
        elif p_mw >= 1000.0:
            f_name = "MMW (1-5 kDa)"
        else:
            f_name = "LMW (<1 kDa)"

        peaks_list.append(SECPeak(
            peak_id=idx_num,
            retention_time=round(p_rt, 2),
            elution_volume=round(p_vr, 3),
            height=round(p_h, 3),
            area=round(p_area, 3),
            area_pct=p_area_pct,
            peak_mw=round(p_mw, 1),
            fraction_name=f_name
        ))

    indices = calculate_sec_indices(vol_arr, sig_corr, calib)

    return SECSample(
        sample_id=sample_id,
        retention_times=rt_arr,
        elution_volume=vol_arr,
        signal_raw=y_vals,
        baseline=base,
        signal_corrected=sig_corr,
        calibration=calib,
        log_mw=log_mw,
        mw=mw_arr,
        mwd_diff=mwd_diff,
        mwd_cum=mwd_cum,
        peaks=peaks_list,
        indices=indices,
        metadata={"flow_rate": flow_rate_ml_min, "n_peaks": len(peaks_list)}
    )


# ==============================================================================
# 2. ИЗВЛЕЧЕНИЕ СВОДНЫХ ДЕСКРИПТОРОВ ДЛЯ DATA FUSION И ML
# ==============================================================================

def extract_sec_descriptors(sample: SECSample) -> Dict[str, float]:
    """
    Извлекает единый вектор аналитических дескрипторов SEC/GPC
    для добавления в 7-модальный датасет ChemoSuite ML.
    """
    return dict(sample.indices)


# ==============================================================================
# 3. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО БЕНЧМАРКА (12 ПРОБ: БАЙКАЛ vs ШЛАМ-ЛИГНИН)
# ==============================================================================

def generate_synthetic_sec_dataset(
    n_samples_per_class: int = 6,
    random_state: int = 42,
    save_dir: Optional[str] = None
) -> Tuple[Dict[str, SECSample], pd.DataFrame]:
    """
    Генерирует согласованный эксклюзионный бенчмарк (SEC / GPC хроматограммы):
    - Класс 1: Фоновое РОВ оз. Байкал (Pristine Baikal NOM)
      -> умеренные молекулярные массы (Mw: 2500 - 4200 Да, Mn: 1400 - 2000 Да),
         низкая полидисперсность (PDI: 1.8 - 2.2), доминирование фракций 1-5 кДа и <1 кДа (LMW),
         доля HMW (>20 кДа) < 8%.
    - Класс 2: Техногенный шлам-лигнин БЦБК (BPPM Sludge-Lignin)
      -> высокомолекулярные конденсированные агрегаты (Mw: 14000 - 22000 Да, Mn: 3500 - 5500 Да),
         высокая полидисперсность (PDI: 3.5 - 4.8), выраженный ранний пик исключения (HMW > 35%),
         высокий индекс HMW/LMW (> 2.5).
    """
    rng = np.random.RandomState(random_state)
    samples: Dict[str, SECSample] = {}
    matrix_rows: List[Dict[str, Any]] = []

    # Сетка элюционного объема от 4.5 мл до 12.5 мл (шаг 0.01 мл, 801 точка)
    # При скорости потока 0.5 мл/мин это соответствует 9.0 - 25.0 мин
    v_grid = np.linspace(4.8, 12.0, 721)
    flow_rate = 0.5
    rt_grid = v_grid / flow_rate

    # Эталонная калибровка PSS
    calib = SECCalibration(
        calibration_type="PSS_Sodium_Sulfonate",
        coefficients=[7.38, -0.428],
        degree=1,
        mw_min=500.0,
        mw_max=150000.0
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(1, n_samples_per_class + 1):
        # -------------------------------------------------------------
        # ОБРАЗЕЦ КЛАССА 1: Природное фоновое РОВ оз. Байкал (Baikal NOM)
        # -------------------------------------------------------------
        s_id_nom = f"Baikal_NOM_{i:02d}"

        # Моделируем 3 перекрывающихся гауссовых пика на хроматограмме:
        # 1) Малый след HMW при V ~ 6.8 мл (MW ~ 30 кДа)
        # 2) Основной широкий пик РОВ/фульвокислот при V ~ 8.9 мл (MW ~ 3.7 кДа)
        # 3) Низкомолекулярный пик простых фенолов при V ~ 10.4 мл (MW ~ 850 Да)
        a_hmw_nom = rng.uniform(5.0, 12.0)
        a_main_nom = rng.uniform(85.0, 115.0)
        a_lmw_nom = rng.uniform(40.0, 65.0)

        sig_nom = (
            a_hmw_nom * np.exp(-((v_grid - rng.uniform(6.7, 7.0)) ** 2) / (2 * 0.35 ** 2)) +
            a_main_nom * np.exp(-((v_grid - rng.uniform(8.8, 9.1)) ** 2) / (2 * 0.55 ** 2)) +
            a_lmw_nom * np.exp(-((v_grid - rng.uniform(10.3, 10.6)) ** 2) / (2 * 0.40 ** 2))
        )
        # Дрейф базовой линии и шум детектора
        base_nom = 3.0 + 0.5 * (v_grid - 4.8) + rng.normal(0, 0.4, size=len(v_grid))
        raw_sig_nom = sig_nom + base_nom

        sample_nom = parse_sec_file(
            pd.DataFrame({"Elution_Volume_mL": v_grid, "UV254_Signal_mAU": raw_sig_nom}).to_csv(index=False),
            sample_id=s_id_nom,
            flow_rate_ml_min=flow_rate,
            calibration_preset="PSS_Sodium_Sulfonate"
        )
        samples[s_id_nom] = sample_nom

        row_nom = {"Sample_ID": s_id_nom, "Class_Target": "Baikal_Pristine"}
        row_nom.update(extract_sec_descriptors(sample_nom))
        matrix_rows.append(row_nom)

        # -------------------------------------------------------------
        # ОБРАЗЕЦ КЛАССА 2: Техногенный шлам-лигнин БЦБК (Sludge-Lignin)
        # -------------------------------------------------------------
        s_id_lignin = f"Sludge_Lignin_{i:02d}"

        # Для крафт-лигнина:
        # 1) Мощный широкий пик агрегатов HMW при V ~ 6.0 - 6.6 мл (MW ~ 30-70 кДа)
        # 2) Промежуточный пик крафт-лигнина при V ~ 7.8 - 8.3 мл (MW ~ 10-15 кДа)
        # 3) Небольшой хвост мономеров при V ~ 10.2 мл (MW ~ 1 кДа)
        a_hmw_lig = rng.uniform(75.0, 110.0)
        a_mid_lig = rng.uniform(60.0, 85.0)
        a_lmw_lig = rng.uniform(15.0, 28.0)

        sig_lig = (
            a_hmw_lig * np.exp(-((v_grid - rng.uniform(6.1, 6.4)) ** 2) / (2 * 0.45 ** 2)) +
            a_mid_lig * np.exp(-((v_grid - rng.uniform(7.8, 8.1)) ** 2) / (2 * 0.50 ** 2)) +
            a_lmw_lig * np.exp(-((v_grid - rng.uniform(10.1, 10.4)) ** 2) / (2 * 0.35 ** 2))
        )
        base_lig = 2.5 + 0.3 * (v_grid - 4.8) + rng.normal(0, 0.4, size=len(v_grid))
        raw_sig_lig = sig_lig + base_lig

        sample_lig = parse_sec_file(
            pd.DataFrame({"Elution_Volume_mL": v_grid, "UV254_Signal_mAU": raw_sig_lig}).to_csv(index=False),
            sample_id=s_id_lignin,
            flow_rate_ml_min=flow_rate,
            calibration_preset="PSS_Sodium_Sulfonate"
        )
        samples[s_id_lignin] = sample_lig

        row_lig = {"Sample_ID": s_id_lignin, "Class_Target": "Sludge_Lignin"}
        row_lig.update(extract_sec_descriptors(sample_lig))
        matrix_rows.append(row_lig)

        # Сохранение файлов при указании папки
        if save_dir:
            df_out_nom = pd.DataFrame({
                "Retention_Time_min": np.round(rt_grid, 3),
                "Elution_Volume_mL": np.round(v_grid, 3),
                "Signal_Raw_mAU": np.round(raw_sig_nom, 3),
                "Signal_Corrected_mAU": np.round(sample_nom.signal_corrected, 3),
            })
            df_out_nom.to_csv(os.path.join(save_dir, f"{s_id_nom}_SEC.csv"), index=False)

            df_out_lig = pd.DataFrame({
                "Retention_Time_min": np.round(rt_grid, 3),
                "Elution_Volume_mL": np.round(v_grid, 3),
                "Signal_Raw_mAU": np.round(raw_sig_lig, 3),
                "Signal_Corrected_mAU": np.round(sample_lig.signal_corrected, 3),
            })
            df_out_lig.to_csv(os.path.join(save_dir, f"{s_id_lignin}_SEC.csv"), index=False)

    df_matrix = pd.DataFrame(matrix_rows)
    if save_dir:
        df_matrix.to_csv(os.path.join(save_dir, "sec_descriptors_matrix.csv"), index=False)

    return samples, df_matrix
