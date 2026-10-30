"""
sec_core.py — Математическое ядро гель-проникающей и эксклюзионной хроматографии (SEC / GPC).
Методология хемометрического анализа молекулярно-массовых распределений (ММР / MWD),
полидисперсности, фракционирования природного органического вещества (РОВ/DOM, гуминовые кислоты)
и техногенного шлам-лигнина целлюлозно-бумажного производства.
"""
import dataclasses
from dataclasses import dataclass, field
import io
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
# Форма калибровки: полином 3-й степени log10(M) = c0 + c1*V + c2*V^2 + c3*V^3
# (стандартная форма для SEC; Striegel, Yau, Kirkland & Bly, "Modern Size-Exclusion
# Liquid Chromatography", 2nd ed., Wiley, 2009).
#
# "default_coeffs" — МНК-подгонка (numpy.polyfit, deg=3) log10(M) vs V по стандартам,
# перечисленным в самом пресете; тест проверяет совпадение с повторной подгонкой и
# воспроизведение M стандартов с точностью <= 5 %. Прежние коэффициенты
# [7.38, -0.428] / [7.45, -0.442] / [7.52, -0.435] не согласовывались со своими же
# стандартами (ошибка M до 38 % / 22 % / 28 %). Линейная МНК-подгонка ("linear_coeffs",
# например PSS: a = 7.744, b = -0.452) оставляет до 17.5 % ошибки на крайних стандартах,
# поэтому по умолчанию используется кубическая форма (degree = 3).
#
# Калибровка действительна ТОЛЬКО внутри диапазона объёмов стандартов ("v_range"):
# сигнал вне диапазона не экстраполируется, а исключается из интегрирования
# (см. calculate_sec_indices: SEC_Excluded_Area_pct / SEC_Above_Cal_Range_pct /
# SEC_Below_Cal_Range_pct).
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
        # log10(M) = c0 + c1*V + c2*V^2 + c3*V^3 (МНК по 7 стандартам выше)
        "default_coeffs": [12.849232966762662, -2.544666616696549, 0.2793766257806849, -0.012136630188668177],
        "degree": 3,
        "linear_coeffs": [7.74388, -0.45195],
        "v_range": (5.85, 10.25),
        "mw_range": (1100.0, 122000.0),
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
        "default_coeffs": [6.258113283548378, -0.09617941347030647, -0.02622772129767934, 0.0004115307763191064],
        "degree": 3,
        "linear_coeffs": [7.59824, -0.45104],
        "v_range": (6.35, 10.60),
        "mw_range": (600.0, 49000.0),
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
        "default_coeffs": [12.101302360718734, -2.354362842766165, 0.26278640508952156, -0.011590480763013859],
        "degree": 3,
        "linear_coeffs": [7.53274, -0.42787],
        "v_range": (5.75, 10.35),
        "mw_range": (1080.0, 112000.0),
    }
}

# Ключи индексов SEC. То, что невозможно вычислить, возвращается как NaN (а не как правдоподобная константа).
SEC_INDEX_KEYS: Tuple[str, ...] = (
    "SEC_Mn", "SEC_Mw", "SEC_Mz", "SEC_Mp", "SEC_PDI", "SEC_Polydispersity",
    "SEC_Fraction_I_pct", "SEC_Fraction_II_pct", "SEC_Fraction_III_pct", "SEC_Fraction_IV_pct",
    "SEC_HMW_pct", "SEC_LMW_pct", "SEC_HMW_LMW_ratio",
    "SEC_Excluded_Area_pct", "SEC_Above_Cal_Range_pct", "SEC_Below_Cal_Range_pct",
)


def _nan_indices() -> Dict[str, float]:
    return {k: float("nan") for k in SEC_INDEX_KEYS}


def _preset_r_squared(preset_key: str) -> float:
    p = CALIBRATION_PRESETS[preset_key]
    v = np.array([s["vr"] for s in p["standards"]], dtype=float)
    y = np.log10(np.array([s["mw"] for s in p["standards"]], dtype=float))
    pred = np.polyval(np.asarray(p["default_coeffs"], dtype=float)[::-1], v)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return round(1.0 - float(np.sum((y - pred) ** 2)) / ss_tot, 5) if ss_tot > 0 else float("nan")


# ==============================================================================
# КОНТЕЙНЕРЫ ДАННЫХ (DATACLASSES)
# ==============================================================================

@dataclass
class SECCalibration:
    """
    Параметры калибровочной кривой эксклюзионной хроматографии log10(M) = sum(c_k * V^k).

    coefficients — [c0, c1, ...] (для degree = 1: [a, b], log10(M) = a + b*V, b < 0).
    v_min / v_max — диапазон элюционных объёмов калибровочных стандартов; вне его масса
    не экстраполируется (None = диапазон неизвестен, проверка экстраполяции невозможна).
    По умолчанию — пресет PSS (кубическая МНК-подгонка по стандартам пресета).
    """
    calibration_type: str = "PSS_Sodium_Sulfonate"
    coefficients: List[float] = field(default_factory=lambda: list(CALIBRATION_PRESETS["PSS_Sodium_Sulfonate"]["default_coeffs"]))
    degree: int = 3
    r_squared: float = field(default_factory=lambda: _preset_r_squared("PSS_Sodium_Sulfonate"))
    mw_min: float = 1100.0
    mw_max: float = 122000.0
    v0: float = 5.2    # Объем полного исключения (мл)
    vt: float = 11.8   # Объем полного проникновения (мл)
    v_min: Optional[float] = 5.85   # Диапазон объёмов стандартов (мл)
    v_max: Optional[float] = 10.25

    def v_to_log_mw(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Пересчет элюционного объема (или времени) в log10(MW) по калибровочному полиному."""
        coeffs = list(self.coefficients)[: max(1, int(self.degree)) + 1]
        v_arr = np.asarray(v, dtype=float)
        res = np.zeros_like(v_arr, dtype=float)
        for power, coef in enumerate(coeffs):
            res = res + coef * (v_arr ** power)
        return float(res) if np.ndim(res) == 0 else res

    def v_to_mw(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Пересчет элюционного объема в молекулярную массу (Да)."""
        log_m = self.v_to_log_mw(v)
        m = 10.0 ** np.clip(log_m, 1.0, 7.0)
        return m

    def in_calibrated_range(self, v: Union[float, np.ndarray]) -> np.ndarray:
        """Маска точек внутри диапазона объёмов стандартов (True везде, если диапазон неизвестен)."""
        v_arr = np.asarray(v, dtype=float)
        if self.v_min is None or self.v_max is None:
            return np.ones(v_arr.shape, dtype=bool)
        lo, hi = min(self.v_min, self.v_max), max(self.v_min, self.v_max)
        tol = 1e-9 * max(1.0, abs(hi))
        return (v_arr >= lo - tol) & (v_arr <= hi + tol)


def calibration_from_preset(preset_key: str) -> SECCalibration:
    """Создаёт SECCalibration из пресета CALIBRATION_PRESETS (с диапазоном объёмов стандартов)."""
    if preset_key not in CALIBRATION_PRESETS:
        preset_key = "PSS_Sodium_Sulfonate"
    p = CALIBRATION_PRESETS[preset_key]
    coeffs = [float(c) for c in p["default_coeffs"]]
    v_lo, v_hi = p["v_range"]
    return SECCalibration(
        calibration_type=preset_key,
        coefficients=coeffs,
        degree=int(p.get("degree", len(coeffs) - 1)),
        r_squared=_preset_r_squared(preset_key),
        mw_min=float(p["mw_range"][0]),
        mw_max=float(p["mw_range"][1]),
        v_min=float(v_lo),
        v_max=float(v_hi),
    )


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
    lam: Optional[float] = None,
    p: float = 0.001,
    niter: int = 15
) -> np.ndarray:
    """
    Асимметричные наименьшие квадраты (Asymmetric Least Squares, ALS) для базовой линии SEC.
    Бережно огибает элюционные пики снизу, убирая дрейф детектора.

    Eilers P.H.C., Boelens H.F.M. (2005) "Baseline correction with asymmetric least squares
    smoothing": минимизируется sum w_i (y_i - z_i)^2 + lam * sum (Δ²z_i)^2, т.е.
    (W + lam * D'D) z = W y, где D — оператор вторых разностей размера (L-2) x L.
    (Ранее D строился как L x (L-2) и использовалось D·D' — это не D'D: штраф не действовал
    на краях, и плоская базовая линия 5.0 подгонялась как ~0.001 в начале хроматограммы.)

    lam=None -> lam = 1e9 * (L / 721)^4. Эффективная ширина сглаживания в точках ~ lam^(1/4),
    поэтому масштаб L^4 сохраняет её постоянной в единицах объёма. SEC-пики занимают большую
    часть хроматограммы, и при lam ~ 1e5 базовая линия «врезается» в широкий пик; при
    lam = 1e9, p = 0.001 Mw синтетического профиля с известным дрейфом воспроизводится < 1 %.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 5:
        return np.zeros_like(y)
    if lam is None:
        lam = 1e9 * (L / 721.0) ** 4
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc", dtype=float)
    D_mat = lam * (D.T @ D)
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
    ok = np.isfinite(v_arr) & np.isfinite(mw_arr) & (mw_arr > 0)
    v_arr, mw_arr = v_arr[ok], mw_arr[ok]
    log_mw = np.log10(np.clip(mw_arr, 10.0, None))

    if len(v_arr) < 2 or len(np.unique(v_arr)) < 2:
        # Раньше молча возвращался пресет PSS — это подмена пользовательской калибровки.
        raise ValueError("At least two standards with distinct elution volumes are required to fit a SEC calibration.")

    deg = max(1, min(int(degree), len(v_arr) - 1))
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

    calib = SECCalibration(
        calibration_type="Fitted_Custom",
        coefficients=coeffs,
        degree=deg,
        r_squared=round(float(max(0.0, min(1.0, r2))), 4),
        mw_min=float(np.min(mw_arr)),
        mw_max=float(np.max(mw_arr)),
        v0=float(np.min(v_arr) * 0.85),
        vt=float(np.max(v_arr) * 1.15),
        v_min=float(np.min(v_arr)),
        v_max=float(np.max(v_arr)),
    )
    # Калибровочная кривая обязана монотонно убывать внутри диапазона стандартов
    v_chk = np.linspace(calib.v_min, calib.v_max, 200)
    if np.any(np.diff(np.asarray(calib.v_to_log_mw(v_chk))) >= 0):
        calib.calibration_type = "Fitted_Custom_NonMonotonic"
    return calib


def _calibrated_mass_bounds(calib: SECCalibration, v_in: np.ndarray, mw_in: np.ndarray) -> Tuple[float, float]:
    """Диапазон масс, покрытый калибровкой (по границам стандартов, иначе — по точкам в окне)."""
    if calib.v_min is not None and calib.v_max is not None:
        m_a = float(calib.v_to_mw(float(calib.v_min)))
        m_b = float(calib.v_to_mw(float(calib.v_max)))
        return min(m_a, m_b), max(m_a, m_b)
    if len(mw_in) == 0:
        return float("nan"), float("nan")
    return float(np.min(mw_in)), float(np.max(mw_in))


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

    Интегрирование ведётся ТОЛЬКО внутри диапазона объёмов калибровочных стандартов
    (calibration.v_min..v_max) — масса вне диапазона не экстраполируется. Доли фракций
    выражены в % от площади внутри калиброванного окна; фракция, чей интервал масс не
    пересекается с калиброванным диапазоном, не может быть измерена и возвращается как NaN
    (например, < 1 кДа для PSS, чей самый лёгкий стандарт 1.1 кДа). Площадь вне окна
    возвращается отдельно: SEC_Excluded_Area_pct = SEC_Above_Cal_Range_pct (раньше самого
    тяжёлого стандарта, M > M_max) + SEC_Below_Cal_Range_pct (позже самого лёгкого, M < M_min).
    Всё, что нельзя вычислить (пустой/нулевой сигнал, деление на 0), возвращается как NaN.
    """
    calib = calibration or SECCalibration()
    res = _nan_indices()
    v = np.asarray(vol_or_time, dtype=float)
    s = np.asarray(signal, dtype=float)
    if len(v) < 5 or len(s) != len(v):
        return res

    finite = np.isfinite(v) & np.isfinite(s)
    s_pos = np.where(finite, np.clip(np.nan_to_num(s, nan=0.0), 0.0, None), 0.0)
    tot_all = float(np.sum(s_pos))
    if tot_all <= 1e-12:
        return res

    in_rng = calib.in_calibrated_range(v) & finite
    if calib.v_min is not None and calib.v_max is not None:
        lo, hi = min(calib.v_min, calib.v_max), max(calib.v_min, calib.v_max)
        above = float(np.sum(s_pos[finite & (v < lo) & ~in_rng]))   # раньше тяжелейшего стандарта
        below = float(np.sum(s_pos[finite & (v > hi) & ~in_rng]))   # позже легчайшего стандарта
    else:
        above = below = 0.0
    s_in = s_pos[in_rng]
    tot = float(np.sum(s_in))
    res["SEC_Excluded_Area_pct"] = round(100.0 * (tot_all - tot) / tot_all, 2)
    res["SEC_Above_Cal_Range_pct"] = round(100.0 * above / tot_all, 2)
    res["SEC_Below_Cal_Range_pct"] = round(100.0 * below / tot_all, 2)
    if tot <= 1e-12:
        return res

    # Массив молекулярных масс M(i) внутри калиброванного окна
    mw_in = np.clip(np.asarray(calib.v_to_mw(v[in_rng]), dtype=float), 50.0, 1e7)

    # Mn = sum(S) / sum(S / M); Mw = sum(S * M) / sum(S); Mz = sum(S * M^2) / sum(S * M)
    sum_s_over_m = float(np.sum(s_in / mw_in))
    mn = tot / sum_s_over_m if sum_s_over_m > 0 else float("nan")
    sum_s_m = float(np.sum(s_in * mw_in))
    mw = sum_s_m / tot
    mz = float(np.sum(s_in * mw_in ** 2)) / sum_s_m if sum_s_m > 0 else float("nan")

    # Mp = MW в точке глобального максимума сигнала (NaN, если максимум вне калиброванного окна)
    max_idx = int(np.argmax(s_pos))
    mp = float(np.clip(calib.v_to_mw(v[max_idx]), 50.0, 1e7)) if in_rng[max_idx] else float("nan")

    pdi = mw / mn if (np.isfinite(mn) and mn > 0) else float("nan")

    # Доли фракций (в % от площади внутри калиброванного окна)
    lim_high, lim_mid, lim_low = mw_fraction_limits
    m_cal_lo, m_cal_hi = _calibrated_mass_bounds(calib, v[in_rng], mw_in)
    frac_defs = [
        (lim_high, np.inf),
        (lim_mid, lim_high),
        (lim_low, lim_mid),
        (0.0, lim_low),
    ]
    fracs: List[float] = []
    for f_lo, f_hi in frac_defs:
        overlaps = (m_cal_hi > f_lo) and (m_cal_lo < f_hi)
        if not overlaps:
            fracs.append(float("nan"))
            continue
        mask = (mw_in >= f_lo) & (mw_in < f_hi)
        fracs.append(float(np.sum(s_in[mask]) / tot * 100.0))
    f1_pct, f2_pct, f3_pct, f4_pct = fracs

    hmw = f1_pct + f2_pct  # NaN, если одна из фракций не измерима
    hmw_lmw_ratio = hmw / f4_pct if (np.isfinite(f4_pct) and f4_pct > 0 and np.isfinite(hmw)) else float("nan")

    res.update({
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
    })
    return res


def _resolve_calibration(
    calibration: Optional[Union["SECCalibration", Sequence[float]]],
    calibration_preset: str,
    calibration_v_range: Optional[Tuple[float, float]],
) -> Tuple[SECCalibration, List[str]]:
    """Выбирает калибровку: пользовательскую (объект или коэффициенты) либо пресет."""
    notes: List[str] = []
    if isinstance(calibration, SECCalibration):
        calib = dataclasses.replace(calibration, coefficients=list(calibration.coefficients))
    elif calibration is not None:
        coeffs = [float(c) for c in calibration]
        if len(coeffs) < 2 or not all(np.isfinite(coeffs)):
            raise ValueError("Calibration coefficients must be at least [c0, c1] (log10(M) = c0 + c1*V + ...).")
        calib = SECCalibration(
            calibration_type="User_Coefficients",
            coefficients=coeffs,
            degree=len(coeffs) - 1,
            r_squared=float("nan"),
            mw_min=float("nan"),
            mw_max=float("nan"),
            v_min=None,
            v_max=None,
        )
    else:
        calib = calibration_from_preset(calibration_preset)
    if calibration_v_range is not None:
        calib.v_min, calib.v_max = float(min(calibration_v_range)), float(max(calibration_v_range))
    if calib.v_min is None or calib.v_max is None:
        notes.append(
            "Calibrated elution-volume range unknown: masses cannot be checked for extrapolation "
            "beyond the standards (pass calibration_v_range)."
        )
    return calib, notes


def parse_sec_file(
    file_content: Union[str, bytes],
    sample_id: str = "SEC_Sample",
    flow_rate_ml_min: float = 1.0,
    calibration_preset: str = "PSS_Sodium_Sulfonate",
    calibration: Optional[Union[SECCalibration, Sequence[float]]] = None,
    calibration_v_range: Optional[Tuple[float, float]] = None,
) -> SECSample:
    """
    Универсальный парсер файлов гель-проникающей хроматографии (SEC/GPC):
    - Распознает колонки времени (Time, RT) или объема (Volume, VR);
    - Распознает сигналы детектора (Signal, UV254, UV280, RI, Absorbance);
    - Выполняет коррекцию базовой линии AsLS (Eilers & Boelens 2005);
    - Применяет калибровочную кривую и рассчитывает дифференциальное/интегральное ММР
      (только внутри диапазона объёмов стандартов, без экстраполяции);
    - Находит характерные пики и вычисляет хемометрические индексы.

    calibration — пользовательская калибровка: объект SECCalibration (например, из
    fit_sec_calibration) или коэффициенты [c0, c1, ...] для log10(M) = sum(c_k * V^k).
    Если задана, имеет приоритет над calibration_preset. calibration_v_range — диапазон
    объёмов стандартов (мл) для пользовательских коэффициентов.
    """
    calib, calib_notes = _resolve_calibration(calibration, calibration_preset, calibration_v_range)

    def _empty() -> SECSample:
        return SECSample(
            sample_id=sample_id,
            calibration=calib,
            indices=calculate_sec_indices(np.array([]), np.array([]), calib),
            metadata={"calibration_notes": list(calib_notes), "calibration_type": calib.calibration_type},
        )

    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    if not lines:
        return _empty()

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
        return _empty()

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

    # Ищем колонку сигнала Y (предпочитаем сырой сигнал, а не чужую «скорректированную» колонку)
    y_col = None
    y_candidates = [
        orig for orig, low in col_map.items()
        if orig != x_col and any(k in low for k in ["uv", "abs", "signal", "intensity", "сигнал", "ri", "mau", "mv", "dabs", "resp"])
    ]
    if y_candidates:
        raw_first = [c for c in y_candidates if "corr" not in col_map[c] and "baseline" not in col_map[c]]
        y_col = (raw_first or y_candidates)[0]

    # Если эвристика не нашла, берем числовые колонки
    num_cols = [c for c in df_raw.columns if pd.to_numeric(df_raw[c], errors="coerce").notnull().sum() > len(df_raw) * 0.5]
    if x_col is None and len(num_cols) > 0:
        x_col = num_cols[0]
    if y_col is None:
        rest = [c for c in num_cols if c != x_col]
        if rest:
            y_col = rest[0]

    if x_col is None or y_col is None:
        return _empty()

    df_clean = df_raw.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col]).sort_values(by=x_col).reset_index(drop=True)

    x_vals = df_clean[x_col].values.astype(float)
    y_vals = df_clean[y_col].values.astype(float)

    if len(x_vals) < 5:
        return _empty()

    if is_volume:
        vol_arr = x_vals
        rt_arr = vol_arr / max(flow_rate_ml_min, 1e-4)
    else:
        rt_arr = x_vals
        vol_arr = rt_arr * flow_rate_ml_min

    # Базовая линия AsLS (Eilers & Boelens 2005)
    base = baseline_als_sec(y_vals)
    sig_corr = np.clip(y_vals - base, 0.0, None)

    # Расчет молекулярных масс — только внутри калиброванного окна (без экстраполяции)
    in_rng = calib.in_calibrated_range(vol_arr)
    log_mw_full = np.log10(np.clip(np.asarray(calib.v_to_mw(vol_arr), dtype=float), 1.0, None))
    mw_arr = np.where(in_rng, 10.0 ** log_mw_full, np.nan)
    log_mw = np.where(in_rng, log_mw_full, np.nan)

    # Дифференциальная кривая ММР: w(log M) = h(V) / |d logM / dV|
    with np.errstate(divide="ignore", invalid="ignore"):
        d_log_m_dv = np.gradient(log_mw_full, vol_arr) if len(np.unique(vol_arr)) == len(vol_arr) else np.gradient(log_mw_full)
        mwd_raw = np.where(np.abs(d_log_m_dv) > 1e-12, np.abs(sig_corr / d_log_m_dv), np.nan)
    mwd_raw = np.where(in_rng, mwd_raw, np.nan)
    mwd_sum = float(np.nansum(mwd_raw))
    mwd_diff = (mwd_raw / mwd_sum) * 100.0 if mwd_sum > 0 else mwd_raw

    # Интегральное ММР (в порядке элюирования, только внутри калиброванного окна)
    mwd_cum = np.cumsum(np.nan_to_num(mwd_diff, nan=0.0))
    if len(mwd_cum) > 0 and mwd_cum[-1] > 0:
        mwd_cum = (mwd_cum / mwd_cum[-1]) * 100.0
    mwd_cum = np.where(in_rng, mwd_cum, np.nan)

    # Пик-пикер на хроматограмме
    prom = max(np.max(sig_corr) * 0.05, 1e-3)
    p_indices, _ = scipy.signal.find_peaks(sig_corr, prominence=prom, distance=max(1, int(len(sig_corr) / 60)))
    peaks_list: List[SECPeak] = []
    tot_area = np.sum(sig_corr)
    v_lo = min(calib.v_min, calib.v_max) if (calib.v_min is not None and calib.v_max is not None) else None

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

        if not np.isfinite(p_mw):
            if v_lo is not None and p_vr < v_lo:
                f_name = "Above calibration range (M > M_max std)"
            else:
                f_name = "Below calibration range (M < M_min std)"
        elif p_mw >= 20000.0:
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
        metadata={
            "flow_rate": flow_rate_ml_min,
            "n_peaks": len(peaks_list),
            "calibration_type": calib.calibration_type,
            "calibrated_volume_range": (calib.v_min, calib.v_max),
            "excluded_area_pct": indices.get("SEC_Excluded_Area_pct", float("nan")),
            "signal_column": str(y_col),
            "calibration_notes": list(calib_notes),
        }
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
    - Класс 1: Фоновое РОВ оз. Байкал (Pristine Baikal NOM): основной пик ~3.7 кДа,
      малый HMW-след и низкомолекулярный пик, частично элюирующий позже самого лёгкого
      стандарта PSS (учитывается в SEC_Below_Cal_Range_pct, без экстраполяции массы).
    - Класс 2: Техногенный шлам-лигнин БЦБК (BPPM Sludge-Lignin): мощный ранний пик
      агрегатов HMW и промежуточный пик ~10-15 кДа, более высокие Mw и PDI.
    Конкретные значения Mn/Mw/PDI определяются калибровкой пресета PSS и коррекцией
    базовой линии AsLS — они вычисляются, а не задаются.
    """
    rng = np.random.RandomState(random_state)
    samples: Dict[str, SECSample] = {}
    matrix_rows: List[Dict[str, Any]] = []

    # Сетка элюционного объема от 4.5 мл до 12.5 мл (шаг 0.01 мл, 801 точка)
    # При скорости потока 0.5 мл/мин это соответствует 9.0 - 25.0 мин
    v_grid = np.linspace(4.8, 12.0, 721)
    flow_rate = 0.5
    rt_grid = v_grid / flow_rate

    # Эталонная калибровка PSS (МНК-подгонка по стандартам пресета)
    calib = calibration_from_preset("PSS_Sodium_Sulfonate")

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
