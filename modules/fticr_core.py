"""
fticr_core.py — Вычислительное ядро масс-спектрометрии сверхвысокого разрешения (FT-ICR MS).
Методология кафедры аналитической химии и лаборатории природных гуминовых систем химфака МГУ.
"""
import inspect
import io
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import nomspectra as ns
    from nomspectra import Spectrum
    NOMSPECTRA_INSTALLED = True
except ImportError:
    NOMSPECTRA_INSTALLED = False

# ==============================================================================
# ТОЧНЫЕ ИЗОТОПНЫЕ МАССЫ И БАЗОВЫЕ КОНСТАНТЫ
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

# Стандартные биогеохимические полигоны диаграммы Ван-Кревелена (H/C vs O/C)
VAN_KREVELEN_REGIONS = {
    "Lipids": {
        "ru": "Липиды",
        "en": "Lipids",
        "hc_range": (1.5, 2.2),
        "oc_range": (0.0, 0.3),
        "polygon": [(0.0, 1.5), (0.3, 1.5), (0.3, 2.2), (0.0, 2.2)],
        "color": "rgba(46, 204, 113, 0.15)",
        "border": "#27ae60",
    },
    "Proteins": {
        "ru": "Пептиды / Белки",
        "en": "Peptides / Proteins",
        "hc_range": (1.5, 2.2),
        "oc_range": (0.3, 0.67),
        "polygon": [(0.3, 1.5), (0.67, 1.5), (0.67, 2.2), (0.3, 2.2)],
        "color": "rgba(52, 152, 219, 0.15)",
        "border": "#2980b9",
    },
    "Carbohydrates": {
        "ru": "Углеводы",
        "en": "Carbohydrates",
        "hc_range": (1.5, 2.4),
        "oc_range": (0.67, 1.2),
        "polygon": [(0.67, 1.5), (1.2, 1.5), (1.2, 2.4), (0.67, 2.4)],
        "color": "rgba(155, 89, 182, 0.15)",
        "border": "#8e44ad",
    },
    "Lignins": {
        "ru": "Лигнины / Полифенолы / CRAM",
        "en": "Lignins / Polyphenols / CRAM",
        "hc_range": (0.7, 1.5),
        "oc_range": (0.1, 0.67),
        "polygon": [(0.1, 0.7), (0.67, 0.7), (0.67, 1.5), (0.1, 1.5)],
        "color": "rgba(243, 156, 18, 0.15)",
        "border": "#d35400",
    },
    "Tannins": {
        "ru": "Таннины",
        "en": "Tannins",
        "hc_range": (0.5, 1.5),
        "oc_range": (0.67, 1.2),
        "polygon": [(0.67, 0.5), (1.2, 0.5), (1.2, 1.5), (0.67, 1.5)],
        "color": "rgba(231, 76, 60, 0.15)",
        "border": "#c0392b",
    },
    "Condensed_Aromatics": {
        "ru": "Конденсированная ароматика (CAS)",
        "en": "Condensed Aromatics (CAS)",
        "hc_range": (0.2, 0.7),
        "oc_range": (0.0, 0.67),
        "polygon": [(0.0, 0.2), (0.67, 0.2), (0.67, 0.7), (0.0, 0.7)],
        "color": "rgba(52, 73, 94, 0.15)",
        "border": "#2c3e50",
    },
    "Unsaturated_HC": {
        "ru": "Ненасыщенные углеводороды",
        "en": "Unsaturated Hydrocarbons",
        "hc_range": (0.7, 1.5),
        "oc_range": (0.0, 0.1),
        "polygon": [(0.0, 0.7), (0.1, 0.7), (0.1, 1.5), (0.0, 1.5)],
        "color": "rgba(26, 188, 156, 0.15)",
        "border": "#16a085",
    },
}


def calculate_descriptors(df: pd.DataFrame, lang: str = "ru") -> pd.DataFrame:
    """Расчет молекулярных дескрипторов (H/C, O/C, DBE, AI, NOSC, классы соединений)."""
    res = df.copy()
    c = res["C"].astype(float)
    h = res["H"].astype(float)
    o = res["O"].astype(float)
    n = res["N"].astype(float) if "N" in res.columns else 0.0
    s = res["S"].astype(float) if "S" in res.columns else 0.0

    res["H/C"] = np.divide(h, c, out=np.full_like(h, np.nan), where=c > 0)
    res["O/C"] = np.divide(o, c, out=np.full_like(o, np.nan), where=c > 0)
    res["DBE"] = 1.0 + c - 0.5 * h + 0.5 * n
    res["DBE-O"] = res["DBE"] - o

    num_ai = 1.0 + c - o - s - 0.5 * h
    den_ai = c - o - s - n
    ai = np.divide(num_ai, den_ai, out=np.zeros_like(num_ai), where=(den_ai > 0) & (num_ai > 0))
    res["AI"] = np.clip(ai, 0.0, 1.0)
    nosc_num = 4.0 * c + h - 3.0 * n - 2.0 * o - 2.0 * s
    res["NOSC"] = np.where(c > 0, 4.0 - np.divide(nosc_num, c, out=np.zeros_like(nosc_num), where=c > 0), np.nan)

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
            return "CHON pool (0.3 < H/C < 0.8, O/C < 0.4)" if lang == "en" else "Пул CHON (0.3 < H/C < 0.8, O/C < 0.4)"
        if ai_val >= 0.5 or (hc < 0.7 and oc <= 0.67):
            return "Condensed tannins / Aromatics" if lang == "en" else "Конденсированные таннины / Ароматика"
        if 0.7 <= hc < 1.5 and 0.1 <= oc <= 0.67:
            return "Lignin-like / CRAM" if lang == "en" else "Лигнины / CRAM"
        if 0.5 <= hc < 1.5 and 0.67 < oc <= 1.0:
            return "Hydrolyzable tannins" if lang == "en" else "Гидролизуемые таннины"
        if 1.5 <= hc <= 2.0 and oc <= 0.3:
            return "Lipids" if lang == "en" else "Липиды"
        if 1.5 <= hc <= 2.0 and 0.3 < oc <= 0.67:
            return "Peptides / Proteins / Aliphatics" if lang == "en" else "Пептиды / Белки / Алифатика"
        if 1.5 <= hc <= 2.0 and 0.67 < oc <= 1.0:
            return "Carbohydrates" if lang == "en" else "Углеводы"
        return "Other / Unclassified" if lang == "en" else "Прочие компоненты"

    res["Compound_Class"] = res.apply(get_compound_class, axis=1)

    def get_biomolecular_class(row):
        hc = row["H/C"]
        oc = row["O/C"]
        ai_val = row["AI"]
        if pd.isna(hc) or pd.isna(oc):
            return "Не определено" if lang == "ru" else "Undefined"
        if ai_val >= 0.67 or (0.2 <= hc < 0.7 and oc <= 0.67):
            return VAN_KREVELEN_REGIONS["Condensed_Aromatics"][lang]
        if 0.7 <= hc <= 1.5 and 0.0 <= oc < 0.1:
            return VAN_KREVELEN_REGIONS["Unsaturated_HC"][lang]
        if 1.5 <= hc <= 2.2 and 0.0 <= oc <= 0.3:
            return VAN_KREVELEN_REGIONS["Lipids"][lang]
        if 1.5 <= hc <= 2.2 and 0.3 < oc <= 0.67:
            return VAN_KREVELEN_REGIONS["Proteins"][lang]
        if 1.5 <= hc <= 2.4 and 0.67 < oc <= 1.2:
            return VAN_KREVELEN_REGIONS["Carbohydrates"][lang]
        if 0.5 <= hc <= 1.5 and 0.67 < oc <= 1.2:
            return VAN_KREVELEN_REGIONS["Tannins"][lang]
        if 0.7 <= hc <= 1.5 and 0.1 <= oc <= 0.67:
            return VAN_KREVELEN_REGIONS["Lignins"][lang]
        return "Прочие компоненты" if lang == "ru" else "Other / Unclassified"

    res["Bio_Class"] = res.apply(get_biomolecular_class, axis=1)
    return res


def get_biomolecular_distribution(assigned_df: pd.DataFrame, lang: str = "ru") -> Dict[str, float]:
    """Возвращает процентное распределение биомолекулярных классов в масс-спектре."""
    if "Bio_Class" not in assigned_df.columns:
        assigned_df = calculate_descriptors(assigned_df, lang=lang)
    counts = assigned_df["Bio_Class"].value_counts(normalize=True) * 100.0
    return {k: round(float(v), 2) for k, v in counts.items()}


def parse_uploaded_file(file_bytes: bytes, delimiter: str = "Auto", decimal_sep: str = ".", has_header: bool = True) -> pd.DataFrame:
    """Парсер пик-листов масс-спектров из текстовых файлов."""
    sep_map = {
        "Auto": None, "Авто (автоопределение)": None,
        "Comma (,)": ",", "Запятая (,)": ",",
        "Semicolon (;)": ";", "Точка с запятой (;)": ";",
        "Tab (\\t)": "\t", "Табуляция (\\t)": "\t",
        "Space": r"\s+", "Пробел": r"\s+",
    }
    actual_sep = sep_map.get(delimiter, ",")
    engine = "python" if (actual_sep is None or actual_sep == r"\s+") else "c"

    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(
            bio, sep=actual_sep, decimal=decimal_sep,
            header=0 if has_header else None, engine=engine
        )
    except Exception:
        bio.seek(0)
        df = pd.read_csv(
            bio, sep=r"\s+", decimal=decimal_sep,
            header=0 if has_header else None, engine="python"
        )

    cols_lower = [str(c).lower().strip() for c in df.columns]
    mass_col, int_col = None, None

    for idx, c in enumerate(cols_lower):
        if any(k in c for k in ["m/z", "mass", "mz", "exp", "эксперим", "масса"]):
            mass_col = df.columns[idx]
            break
    for idx, c in enumerate(cols_lower):
        if any(k in c for k in ["int", "i", "abund", "height", "area", "интенсив", "высота"]):
            int_col = df.columns[idx]
            break

    if not has_header or mass_col is None:
        mass_col = df.columns[0]
        int_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    res_df = pd.DataFrame()
    res_df["mass"] = pd.to_numeric(df[mass_col], errors="coerce")
    res_df["intensity"] = pd.to_numeric(df[int_col], errors="coerce") if int_col is not None else 100.0
    res_df = res_df.dropna().sort_values("mass").reset_index(drop=True)
    return res_df


def fast_formula_assigner(
    peaks_df: pd.DataFrame, bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    max_hc: float = 2.5, max_oc: float = 1.2, ppm_tolerance: float = 2.0,
    ion_mode: str = "ESI(-)", max_charge: int = 1,
    iso_check: bool = False, iso_strict: bool = False,
) -> pd.DataFrame:
    """Векторный перебор и приписка брутто-формул с азотным правилом и проверкой 13C."""
    if bounds is None:
        bounds = {"C": (4, 120), "H": (4, 200), "O": (1, 60), "N": (0, 2), "S": (0, 1)}
    c_min, c_max = bounds.get("C", (4, 120))
    h_min, h_max = bounds.get("H", (4, 200))
    o_min, o_max = bounds.get("O", (1, 60))
    n_min, n_max = bounds.get("N", (0, 2))
    s_min, s_max = bounds.get("S", (0, 1))

    peaks_m = peaks_df["mass"].values
    peaks_int = peaks_df["intensity"].values
    peaks_norm = peaks_df["norm_intensity"].values if "norm_intensity" in peaks_df.columns else peaks_int

    min_mz, max_mz = float(peaks_m.min()), float(peaks_m.max())
    charges = [1, 2] if max_charge >= 2 else [1]
    all_assigned_rows = []

    for z in charges:
        ion_shift = -z * H_ION_MASS if "ESI(-)" in ion_mode else (z * H_ION_MASS if "ESI(+)" in ion_mode else 0.0)
        c_list, h_list, o_list, n_list, s_list = [], [], [], [], []

        for n in range(n_min, n_max + 1):
            for s in range(s_min, s_max + 1):
                for c in range(c_min, c_max + 1):
                    cur_o_max = min(o_max, int(max_oc * c))
                    for o in range(o_min, cur_o_max + 1):
                        base_neut = c * EXACT_MASSES["C"] + o * EXACT_MASSES["O"] + n * EXACT_MASSES["N"] + s * EXACT_MASSES["S"]
                        base_mz = (base_neut + ion_shift) / z
                        if base_mz > max_mz + 2.0:
                            continue

                        h_low = max(h_min, int(np.ceil(0.2 * c)), int(np.ceil(2 * (c - o - 9) + n)),
                                    int(np.ceil(((min_mz - 2.0) * z - base_neut - ion_shift) / EXACT_MASSES["H"])))
                        h_high = min(h_max, int(max_hc * c), int(2 * c + n + 2), int(2 * (11 + c - o) + n),
                                     int(np.floor(((max_mz + 2.0) * z - base_neut - ion_shift) / EXACT_MASSES["H"])))

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
        n_arr = np.array(n_list, dtype=np.int16)
        s_arr = np.array(s_list, dtype=np.int16)

        cand_masses = (c_arr * EXACT_MASSES["C"] + h_arr * EXACT_MASSES["H"] + o_arr * EXACT_MASSES["O"]
                       + n_arr * EXACT_MASSES["N"] + s_arr * EXACT_MASSES["S"] + ion_shift) / z

        sort_idx = np.argsort(cand_masses)
        cand_masses = cand_masses[sort_idx]
        c_arr = c_arr[sort_idx]
        h_arr = h_arr[sort_idx]
        o_arr = o_arr[sort_idx]
        n_arr = n_arr[sort_idx]
        s_arr = s_arr[sort_idx]

        idx_left = np.searchsorted(cand_masses, peaks_m - (peaks_m * ppm_tolerance * 1e-6), side="left")
        idx_right = np.searchsorted(cand_masses, peaks_m + (peaks_m * ppm_tolerance * 1e-6), side="right")

        for p_idx in range(len(peaks_m)):
            l, r = idx_left[p_idx], idx_right[p_idx]
            if l >= r:
                continue

            exp_m = peaks_m[p_idx]
            intens = peaks_int[p_idx]
            norm_i = peaks_norm[p_idx]

            sub_theor = cand_masses[l:r]
            errors = (exp_m - sub_theor) / exp_m * 1e6
            best_local = np.argmin(np.abs(errors))
            best_idx = l + best_local

            best_err = float(errors[best_local])
            c_val, h_val, o_val = int(c_arr[best_idx]), int(h_arr[best_idx]), int(o_arr[best_idx])
            n_val, s_val = int(n_arr[best_idx]), int(s_arr[best_idx])

            formula_str = f"C{c_val}H{h_val}" + (f"O{o_val}" if o_val > 0 else "") + \
                          (f"N{n_val}" if n_val > 0 else "") + (f"S{s_val}" if s_val > 0 else "")

            has_c13 = False
            if iso_check:
                exp_c13_m = exp_m + (C13_DIFF / z)
                c13_l = np.searchsorted(peaks_m, exp_c13_m - (exp_c13_m * ppm_tolerance * 1e-6), side="left")
                c13_r = np.searchsorted(peaks_m, exp_c13_m + (exp_c13_m * ppm_tolerance * 1e-6), side="right")

                if c13_l < c13_r:
                    c13_intens = peaks_int[c13_l]
                    theor_ratio = c_val * 0.0108
                    obs_ratio = c13_intens / intens if intens > 0 else 0
                    if iso_strict:
                        if 0.4 * theor_ratio <= obs_ratio <= 2.2 * theor_ratio:
                            has_c13 = True
                    else:
                        has_c13 = True

                if iso_strict and not has_c13 and c_val >= 10:
                    continue

            all_assigned_rows.append({
                "mass": exp_m, "intensity": intens, "norm_intensity": norm_i,
                "theor_mass": cand_masses[best_idx], "error_ppm": best_err,
                "C": c_val, "H": h_val, "O": o_val, "N": n_val, "S": s_val,
                "Formula": formula_str, "z": z, "13C_confirmed": has_c13
            })

    if not all_assigned_rows:
        return pd.DataFrame()

    res_df = pd.DataFrame(all_assigned_rows)
    res_df["abs_error"] = res_df["error_ppm"].abs()
    res_df = res_df.sort_values("abs_error").drop_duplicates(subset=["mass"]).sort_values("mass").reset_index(drop=True)
    res_df = res_df.drop(columns=["abs_error"])
    return res_df


def run_formula_assignment(
    peaks_df: pd.DataFrame, bounds: Dict[str, Tuple[int, int]],
    max_hc: float, max_oc: float, ppm_tolerance: float,
    ion_mode: str, max_charge: int = 1,
    iso_check: bool = False, iso_strict: bool = False, lang: str = "ru",
) -> pd.DataFrame:
    """Обертка формульной идентификации с поддержкой NOM-SPECTRa и встроенного алгоритма."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        tmp_path = tmp.name
        peaks_df.to_csv(tmp_path, sep="\t", index=False)

    assigned_df = pd.DataFrame()
    try:
        if NOMSPECTRA_INSTALLED and max_charge == 1 and not iso_check:
            try:
                spec = Spectrum(tmp_path)
                target_method = getattr(spec, "assign_formulas", None) or getattr(spec, "assign", None)
                if target_method is not None:
                    sig = inspect.signature(target_method).parameters
                    kwargs = {}
                    if "error" in sig: kwargs["error"] = ppm_tolerance
                    elif "ppm" in sig: kwargs["ppm"] = ppm_tolerance
                    elif "tolerance" in sig: kwargs["tolerance"] = ppm_tolerance

                    for elem, (low, high) in bounds.items():
                        if elem in sig: kwargs[elem] = (low, high)
                    if "elements" in sig: kwargs["elements"] = bounds
                    if "hc_limits" in sig: kwargs["hc_limits"] = (0.2, max_hc)
                    if "oc_limits" in sig: kwargs["oc_limits"] = (0.0, max_oc)
                    if "mode" in sig: kwargs["mode"] = ("neg" if "ESI(-)" in ion_mode else ("pos" if "ESI(+)" in ion_mode else "neutral"))

                    target_method(**kwargs)
                    for attr in ["data", "df", "assigned", "assigned_data", "peaks"]:
                        if hasattr(spec, attr):
                            val = getattr(spec, attr)
                            if isinstance(val, pd.DataFrame) and not val.empty:
                                assigned_df = val.copy()
                                break
            except Exception:
                assigned_df = pd.DataFrame()

        if assigned_df.empty or "C" not in assigned_df.columns:
            assigned_df = fast_formula_assigner(
                peaks_df=peaks_df, bounds=bounds, max_hc=max_hc, max_oc=max_oc,
                ppm_tolerance=ppm_tolerance, ion_mode=ion_mode, max_charge=max_charge,
                iso_check=iso_check, iso_strict=iso_strict,
            )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not assigned_df.empty and "C" in assigned_df.columns:
        assigned_df = calculate_descriptors(assigned_df, lang=lang)

    return assigned_df


def compute_kmd(masses: np.ndarray, base_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Расчет дефекта массы Кендрика (Kendrick Mass Defect)."""
    nom, exact = KMD_BASES[base_key]["nom"], KMD_BASES[base_key]["exact"]
    km = masses * (nom / exact)
    nkm = np.round(km).astype(int)
    kmd = nkm - km
    return km, kmd, nkm


def compute_vk20_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Хемотипирование по 20 ячейкам Перминовой в координатах Ван-Кревелена."""
    oc_bins = [0.00, 0.25, 0.50, 0.75, 1.000001]
    hc_bins = [0.20, 0.60, 1.00, 1.40, 1.80, 2.200001]

    sub = df[
        (df["O/C"] >= 0.0) & (df["O/C"] <= 1.0) &
        (df["H/C"] >= 0.2) & (df["H/C"] <= 2.2)
    ].copy()

    total_count = len(sub)
    total_int = sub["intensity"].sum() if "intensity" in sub.columns else float(total_count)

    grid_count = np.zeros((5, 4), dtype=float)
    grid_weight = np.zeros((5, 4), dtype=float)

    if total_count > 0:
        c_idx = np.clip(np.digitize(sub["O/C"], oc_bins) - 1, 0, 3)
        r_idx = np.clip(np.digitize(sub["H/C"], hc_bins) - 1, 0, 4)
        np.add.at(grid_count, (r_idx, c_idx), 1.0)
        if "intensity" in sub.columns:
            np.add.at(grid_weight, (r_idx, c_idx), sub["intensity"].values)
        else:
            grid_weight = grid_count.copy()

    pct_count = (grid_count / total_count * 100.0) if total_count > 0 else grid_count
    pct_weight = (grid_weight / total_int * 100.0) if total_int > 0 else grid_weight

    tbl_rows = []
    for r in range(5):
        h_str = f"{hc_bins[r]:.2f}-{hc_bins[r+1]:.2f}"
        for c in range(4):
            o_str = f"{oc_bins[c]:.2f}-{oc_bins[c+1]:.2f}"
            cell_idx = 1 + r * 4 + c
            tbl_rows.append({
                "Cell": f"VK_{cell_idx}",
                "H/C_range": h_str,
                "O/C_range": o_str,
                "Count": int(grid_count[r, c]),
                "Pct_Count": round(float(pct_count[r, c]), 2),
                "Pct_Weight": round(float(pct_weight[r, c]), 2),
            })

    return pct_count, pct_weight, pd.DataFrame(tbl_rows)


def align_two_spectra_fast(df_a: pd.DataFrame, df_b: pd.DataFrame, ppm_tol: float = 1.5):
    """Быстрое выравнивание двух масс-спектров по допуску погрешности (ppm)."""
    m_a = df_a["mass"].values
    m_b = df_b["mass"].values

    idx_b_left = np.searchsorted(m_b, m_a - (m_a * ppm_tol * 1e-6), side="left")
    idx_b_right = np.searchsorted(m_b, m_a + (m_a * ppm_tol * 1e-6), side="right")

    matches_a, matches_b = [], []
    used_b = set()

    for i in range(len(m_a)):
        l, r = idx_b_left[i], idx_b_right[i]
        if l < r:
            valid_cand = [j for j in range(l, r) if j not in used_b]
            if valid_cand:
                best_j = min(valid_cand, key=lambda j: abs(m_a[i] - m_b[j]))
                matches_a.append(i)
                matches_b.append(best_j)
                used_b.add(best_j)

    return matches_a, matches_b


def perform_spectral_algebra(df_a: pd.DataFrame, df_b: pd.DataFrame, operation: str, ppm_tol: float = 1.5) -> pd.DataFrame:
    """Спектральная алгебра (A-B, B-A, A+B, A ∩ B, A ⊕ B)."""
    a = df_a.sort_values("mass").reset_index(drop=True)
    b = df_b.sort_values("mass").reset_index(drop=True)

    m_a, m_b = align_two_spectra_fast(a, b, ppm_tol=ppm_tol)
    m_a_set, m_b_set = set(m_a), set(m_b)

    op_norm = str(operation).strip().lower()

    if op_norm in ["a - b", "sub_a_b", "difference a \\ b", "diff_a_b"]:
        return a[[i not in m_a_set for i in range(len(a))]].reset_index(drop=True)
    elif op_norm in ["b - a", "sub_b_a", "difference b \\ a", "diff_b_a"]:
        return b[[j not in m_b_set for j in range(len(b))]].reset_index(drop=True)
    elif op_norm in ["a ∩ b", "a and b", "intersection", "and"]:
        return a.iloc[m_a].reset_index(drop=True)
    elif op_norm in ["a + b", "a or b", "union", "or"]:
        b_uniq = b[[j not in m_b_set for j in range(len(b))]][["mass", "intensity"]]
        return pd.concat([a, b_uniq], ignore_index=True).sort_values("mass").reset_index(drop=True)
    elif op_norm in ["a ⊕ b", "xor", "symmetric difference", "sym_diff"]:
        part_a = a[[i not in m_a_set for i in range(len(a))]]
        part_b = b[[j not in m_b_set for j in range(len(b))]][["mass", "intensity"]]
        return pd.concat([part_a, part_b], ignore_index=True).sort_values("mass").reset_index(drop=True)
    return pd.DataFrame()


def run_tmds_screening(peaks_df: pd.DataFrame, top_n: int = 1500, tol_mda: float = 2.0):
    """Скрининг массовых разностей (Truncated Mass Difference Screening, TMDS)."""
    sub = peaks_df.sort_values("intensity", ascending=False).head(top_n).sort_values("mass").reset_index(drop=True)
    masses = sub["mass"].values
    n = len(masses)
    if n < 2: return pd.DataFrame(), pd.DataFrame()

    diff_matrix = np.abs(masses[:, None] - masses[None, :])
    i_upper, j_upper = np.triu_indices(n, k=1)
    diffs = diff_matrix[i_upper, j_upper]

    tol_da = tol_mda / 1000.0
    total_pairs = len(diffs)
    summary_rows, pair_rows = [], []

    for item in TMDS_LIBRARY:
        delta_theor = item["delta"]
        mask = np.abs(diffs - delta_theor) <= tol_da
        hit_count = int(np.sum(mask))
        summary_rows.append({
            "Transformation": item["name"], "Delta_m": delta_theor,
            "Count": hit_count, "Share_pct": round((hit_count / total_pairs * 100.0) if total_pairs > 0 else 0.0, 3),
        })
        if hit_count > 0:
            for idx_a, idx_b in zip(i_upper[mask][:300], j_upper[mask][:300]):
                pair_rows.append({
                    "Transformation": item["name"], "Mass_1": masses[idx_a], "Mass_2": masses[idx_b],
                    "Delta_obs": abs(masses[idx_a] - masses[idx_b]),
                    "Error_mDa": (abs(masses[idx_a] - masses[idx_b]) - delta_theor) * 1000.0,
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def build_tmds_network_graph(
    peaks_df: pd.DataFrame,
    top_n: int = 150,
    tol_mda: float = 2.0,
    max_edges: int = 250,
    layout: str = "spring",
) -> Dict[str, Any]:
    """
    Построение молекулярной сети биогеохимических реакций/трансформаций (TMDS Network Graph).
    Определяет узлы (пики масс-спектра), ребра (химические переходы: CH2, O, H2O, CO2, NH3)
    и рассчитывает топологические характеристики сети (Degree Centrality, Hubs).

    Parameters:
    -----------
    peaks_df : pd.DataFrame
        Датафрейм пиков спектра (обязательны 'mass', 'intensity', опционально 'Formula', 'Hetero_Class').
    top_n : int
        Число наиболее интенсивных пиков для построения сети (по умолчанию 150).
    tol_mda : float
        Допуск погрешности разности масс (mDa).
    max_edges : int
        Максимальное количество ребер для визуализации.
    layout : str
        Тип пространственной раскладки графа ('spring' - силовой пружинный граф, 'chemical' - m/z vs int или O/C vs H/C).

    Returns:
    --------
    Dict с ключами:
      - 'nodes_df': pd.DataFrame узлов (id, mass, intensity, degree, x, y, formula, class)
      - 'edges_df': pd.DataFrame ребер (source, target, transformation, delta_m, error_mDa)
      - 'hubs_df': pd.DataFrame топ-хабов сети с наибольшим числом связей
      - 'summary_df': частоты обнаружения трансформаций
    """
    if peaks_df.empty or "mass" not in peaks_df.columns:
        return {
            "nodes_df": pd.DataFrame(),
            "edges_df": pd.DataFrame(),
            "hubs_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
        }

    sub = peaks_df.sort_values("intensity", ascending=False).head(top_n).copy()
    sub = sub.sort_values("mass").reset_index(drop=True)
    masses = sub["mass"].values
    n_nodes = len(masses)

    if n_nodes < 2:
        return {
            "nodes_df": pd.DataFrame(),
            "edges_df": pd.DataFrame(),
            "hubs_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
        }

    # Поиск связанных пар через TMDS
    summary_df, pairs_df = run_tmds_screening(sub, top_n=top_n, tol_mda=tol_mda)

    if pairs_df.empty:
        # Нет обнаруженных связей в заданном окне
        return {
            "nodes_df": pd.DataFrame(),
            "edges_df": pd.DataFrame(),
            "hubs_df": pd.DataFrame(),
            "summary_df": summary_df,
        }

    edges_subset = pairs_df.head(max_edges).copy()

    # Построение карты узлов и степени связанности (Degree)
    degree_map: Dict[float, int] = {}
    trans_map: Dict[float, set] = {}

    for _, row in edges_subset.iterrows():
        m1 = round(float(row["Mass_1"]), 4)
        m2 = round(float(row["Mass_2"]), 4)
        t_name = str(row["Transformation"])
        degree_map[m1] = degree_map.get(m1, 0) + 1
        degree_map[m2] = degree_map.get(m2, 0) + 1
        trans_map.setdefault(m1, set()).add(t_name.split()[0])
        trans_map.setdefault(m2, set()).add(t_name.split()[0])

    # Оставляем только узлы, участвующие хотя бы в одной связи
    connected_masses = set(degree_map.keys())
    sub["mass_round"] = sub["mass"].round(4)
    nodes_df = sub[sub["mass_round"].isin(connected_masses)].copy().reset_index(drop=True)

    if nodes_df.empty:
        return {
            "nodes_df": pd.DataFrame(),
            "edges_df": pd.DataFrame(),
            "hubs_df": pd.DataFrame(),
            "summary_df": summary_df,
        }

    nodes_df["Degree"] = nodes_df["mass_round"].map(degree_map).fillna(0).astype(int)
    nodes_df["Node_ID"] = nodes_df["mass_round"].apply(lambda m: f"m/z {m:.4f}")

    if "Formula" not in nodes_df.columns:
        nodes_df["Formula"] = nodes_df["Node_ID"]
    if "Hetero_Class" not in nodes_df.columns:
        nodes_df["Hetero_Class"] = "Unknown"
    if "Bio_Class" not in nodes_df.columns:
        nodes_df["Bio_Class"] = "NOM"

    # Расчет пространственных координат (X, Y)
    n_pts = len(nodes_df)
    m_to_idx = {r["mass_round"]: idx for idx, r in nodes_df.iterrows()}

    if layout == "chemical" and "O/C" in nodes_df.columns and "H/C" in nodes_df.columns and nodes_df["O/C"].notna().any():
        nodes_df["x"] = nodes_df["O/C"].fillna(0.0)
        nodes_df["y"] = nodes_df["H/C"].fillna(1.0)
    elif layout == "chemical":
        nodes_df["x"] = (nodes_df["mass"] - nodes_df["mass"].min()) / (nodes_df["mass"].max() - nodes_df["mass"].min() + 1e-6) * 2.0 - 1.0
        max_int = nodes_df["intensity"].max()
        nodes_df["y"] = np.log10(np.maximum(1.0, nodes_df["intensity"])) / (np.log10(max_int + 1.0) + 1e-6) * 2.0 - 1.0
    else:
        # Быстрый пружинный алгоритм Фрухтермана — Рейнгольда на NumPy
        rng = np.random.RandomState(42)
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)]) + rng.uniform(-0.1, 0.1, size=(n_pts, 2))

        edge_pairs = []
        for _, r_edge in edges_subset.iterrows():
            m1_r = round(float(r_edge["Mass_1"]), 4)
            m2_r = round(float(r_edge["Mass_2"]), 4)
            if m1_r in m_to_idx and m2_r in m_to_idx:
                edge_pairs.append((m_to_idx[m1_r], m_to_idx[m2_r]))

        k_spring = np.sqrt(1.0 / max(1, n_pts))
        t_temp = 1.0
        n_iters = 40

        for it in range(n_iters):
            # Отталкивание между всеми парами
            disp = np.zeros_like(pos)
            delta = pos[:, None, :] - pos[None, :, :]
            dist = np.sqrt(np.sum(delta ** 2, axis=-1)) + 1e-4
            np.fill_diagonal(dist, np.inf)

            rep_force = (k_spring ** 2) / (dist ** 2)
            disp += np.sum(rep_force[:, :, None] * (delta / dist[:, :, None]), axis=1)

            # Притяжение по ребрам
            for i_u, i_v in edge_pairs:
                d_vec = pos[i_u] - pos[i_v]
                d_val = np.linalg.norm(d_vec) + 1e-4
                att_force = (d_val ** 2) / k_spring
                d_norm = d_vec / d_val
                disp[i_u] -= att_force * d_norm
                disp[i_v] += att_force * d_norm

            # Смещение с температурным затуханием
            disp_norm = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-4
            step = (disp / disp_norm) * np.minimum(disp_norm, t_temp)
            pos += step
            t_temp *= (1.0 - (it / float(n_iters)))

        # Нормировка позиций в [-1, 1]
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        pos_range = np.maximum(pos_max - pos_min, 1e-4)
        pos_norm = (pos - pos_min) / pos_range * 2.0 - 1.0

        nodes_df["x"] = pos_norm[:, 0]
        nodes_df["y"] = pos_norm[:, 1]

    # Добавляем координаты узлов в ребра
    edges_subset["x0"] = edges_subset["Mass_1"].apply(lambda m: nodes_df.loc[m_to_idx[round(m, 4)], "x"] if round(m, 4) in m_to_idx else np.nan)
    edges_subset["y0"] = edges_subset["Mass_1"].apply(lambda m: nodes_df.loc[m_to_idx[round(m, 4)], "y"] if round(m, 4) in m_to_idx else np.nan)
    edges_subset["x1"] = edges_subset["Mass_2"].apply(lambda m: nodes_df.loc[m_to_idx[round(m, 4)], "x"] if round(m, 4) in m_to_idx else np.nan)
    edges_subset["y1"] = edges_subset["Mass_2"].apply(lambda m: nodes_df.loc[m_to_idx[round(m, 4)], "y"] if round(m, 4) in m_to_idx else np.nan)
    edges_valid = edges_subset.dropna(subset=["x0", "y0", "x1", "y1"]).copy().reset_index(drop=True)

    # Топ-хабы сети
    hubs = nodes_df.sort_values("Degree", ascending=False).head(10).copy()
    hubs["Key_Reactions"] = hubs["mass_round"].apply(lambda m: ", ".join(sorted(list(trans_map.get(m, set())))))
    hubs_df = hubs[["Node_ID", "mass", "Formula", "Degree", "Hetero_Class", "Bio_Class", "Key_Reactions"]].reset_index(drop=True)

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_valid,
        "hubs_df": hubs_df,
        "summary_df": summary_df,
    }


def compute_geochemical_vector_fluxes(
    peaks_df: pd.DataFrame,
    tol_mda: float = 2.0,
    top_n: int = 1500,
) -> Dict[str, Any]:
    """
    Количественная оценка геохимических векторных потоков реакций (Geochemical Vector Flux Analysis).
    Рассчитывает баланс и соотношения ключевых реакционных трансформаций:
    окисление (+O), декарбоксилирование (-CO2), метилирование/гомология (CH2),
    гидратация (H2O), сульфирование (SO3) и гидрогенизация (H2).
    """
    summary_df, pairs_df = run_tmds_screening(peaks_df, top_n=top_n, tol_mda=tol_mda)
    if summary_df.empty:
        return {
            "flux_counts": {},
            "flux_shares": {},
            "indices": {},
            "summary_df": pd.DataFrame(),
            "total_reactions": 0,
        }

    total_reactions = int(summary_df["Count"].sum())
    counts_by_code: Dict[str, int] = {}
    for _, row in summary_df.iterrows():
        t_name = str(row["Transformation"])
        cnt = int(row["Count"])
        code = t_name.split()[0]
        counts_by_code[code] = cnt

    c_o = counts_by_code.get("O", 0)
    c_co2 = counts_by_code.get("CO2", 0)
    c_ch2 = counts_by_code.get("CH2", 0)
    c_h2o = counts_by_code.get("H2O", 0)
    c_so3 = counts_by_code.get("SO3", 0)
    c_h2 = counts_by_code.get("H2", 0)
    denom = max(1, total_reactions)

    indices = {
        "ox_decarb_ratio": round(c_o / max(1, c_co2), 2),
        "alkylation_share_pct": round(c_ch2 / denom * 100.0, 2),
        "hydration_ox_ratio": round(c_h2o / max(1, c_o), 2),
        "sulfonation_index_pct": round(c_so3 / denom * 100.0, 2),
        "hydrogenation_index_pct": round(c_h2 / denom * 100.0, 2),
        "total_reaction_pairs": total_reactions,
    }

    summary_df["Flux_Type"] = summary_df["Transformation"].apply(
        lambda t: "Oxidation" if "O (" in t
        else "Mineralization" if "CO2" in t
        else "Homology" if "CH2" in t
        else "Hydration" if "H2O" in t
        else "Sulfonation" if "SO3" in t
        else "Hydrogenation" if "H2" in t
        else "Other"
    )

    return {
        "flux_counts": counts_by_code,
        "flux_shares": {code: round(cnt / denom * 100.0, 2) for code, cnt in counts_by_code.items()},
        "indices": indices,
        "summary_df": summary_df,
        "total_reactions": total_reactions,
    }


def find_transformation_pathways(
    peaks_df: pd.DataFrame,
    source_mass: float,
    target_mass: float,
    max_depth: int = 4,
    tol_mda: float = 2.0,
    max_paths: int = 10,
) -> List[Dict[str, Any]]:
    """
    Многостадийный поиск путей биогеохимической трансформации (Reaction Pathway Discovery).
    Ищет цепочки химических реакций (+O, -CO2, +CH2, -H2O и др.) между молекулярным
    ионом-предшественником (source_mass) и продуктом трансформации (target_mass).
    """
    if peaks_df.empty or "mass" not in peaks_df.columns:
        return []

    df_sorted = peaks_df.sort_values("mass").reset_index(drop=True)
    masses = df_sorted["mass"].to_numpy(dtype=float)
    formulas = df_sorted["Formula"].tolist() if "Formula" in df_sorted.columns else [f"m/z {m:.4f}" for m in masses]

    tol_da = tol_mda / 1000.0

    # Поиск ближайшего стартового и целевого пика
    src_idx = int(np.argmin(np.abs(masses - source_mass)))
    if abs(masses[src_idx] - source_mass) > max(tol_da * 3.0, 0.02):
        return []

    dst_idx = int(np.argmin(np.abs(masses - target_mass)))
    if abs(masses[dst_idx] - target_mass) > max(tol_da * 3.0, 0.02):
        return []

    if src_idx == dst_idx:
        return []

    from collections import deque
    queue = deque([(src_idx, [], {src_idx})])
    found_paths: List[Dict[str, Any]] = []

    while queue:
        curr_idx, path_steps, visited = queue.popleft()

        if curr_idx == dst_idx:
            cum_err = sum(abs(step["error_mda"]) for step in path_steps)
            steps_repr = " -> ".join([f"{step['direction']}{step['code']}" for step in path_steps])
            path_str = f"{formulas[src_idx]} [{steps_repr}] -> {formulas[dst_idx]}"
            found_paths.append({
                "steps": path_steps,
                "path_str": path_str,
                "depth": len(path_steps),
                "cumulative_error_mda": round(cum_err, 3),
                "source_mass": masses[src_idx],
                "target_mass": masses[dst_idx],
                "source_formula": formulas[src_idx],
                "target_formula": formulas[dst_idx],
            })
            if len(found_paths) >= max_paths:
                break
            continue

        if len(path_steps) >= max_depth:
            continue

        curr_mass = masses[curr_idx]

        for tmd in TMDS_LIBRARY:
            delta = tmd["delta"]
            name = tmd["name"]
            code = name.split()[0]

            # 1. Прямой переход: +delta
            t_fwd = curr_mass + delta
            l_f = np.searchsorted(masses, t_fwd - tol_da)
            r_f = np.searchsorted(masses, t_fwd + tol_da)
            for nxt_idx in range(l_f, r_f):
                if nxt_idx not in visited:
                    obs_delta = masses[nxt_idx] - curr_mass
                    err_mda = (obs_delta - delta) * 1000.0
                    step_data = {
                        "from_mass": masses[curr_idx],
                        "to_mass": masses[nxt_idx],
                        "from_formula": formulas[curr_idx],
                        "to_formula": formulas[nxt_idx],
                        "transformation": name,
                        "code": code,
                        "direction": "+",
                        "delta_theor": delta,
                        "delta_obs": obs_delta,
                        "error_mda": round(err_mda, 3),
                    }
                    new_visited = set(visited)
                    new_visited.add(nxt_idx)
                    queue.append((nxt_idx, path_steps + [step_data], new_visited))

            # 2. Обратный переход: -delta
            t_rev = curr_mass - delta
            l_r = np.searchsorted(masses, t_rev - tol_da)
            r_r = np.searchsorted(masses, t_rev + tol_da)
            for nxt_idx in range(l_r, r_r):
                if nxt_idx not in visited:
                    obs_delta = curr_mass - masses[nxt_idx]
                    err_mda = (obs_delta - delta) * 1000.0
                    step_data = {
                        "from_mass": masses[curr_idx],
                        "to_mass": masses[nxt_idx],
                        "from_formula": formulas[curr_idx],
                        "to_formula": formulas[nxt_idx],
                        "transformation": name,
                        "code": code,
                        "direction": "-",
                        "delta_theor": delta,
                        "delta_obs": obs_delta,
                        "error_mda": round(err_mda, 3),
                    }
                    new_visited = set(visited)
                    new_visited.add(nxt_idx)
                    queue.append((nxt_idx, path_steps + [step_data], new_visited))

    found_paths.sort(key=lambda p: (p["depth"], p["cumulative_error_mda"]))
    return found_paths[:max_paths]



def get_calibrant_library(series_name: str, ion_mode: str) -> pd.DataFrame:
    """Генерация теоретической библиотеки калибрантов (FA / CHO)."""
    calibrants = []
    if "FA" in series_name or "Жирные" in series_name or "Fatty" in series_name:
        for n in range(12, 34):
            m_neut = n * EXACT_MASSES["C"] + 2 * n * EXACT_MASSES["H"] + 2 * EXACT_MASSES["O"]
            calibrants.append({"name": f"FA {n}:0 (C{n}H{2*n}O2)", "m_theor": m_neut - H_ION_MASS if "ESI(-)" in ion_mode else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)})
    elif "CHO" in series_name:
        for n in range(14, 32):
            h_count = 2 * n - 8
            m_neut = n * EXACT_MASSES["C"] + h_count * EXACT_MASSES["H"] + 7 * EXACT_MASSES["O"]
            calibrants.append({"name": f"CHO C{n}H{h_count}O7", "m_theor": m_neut - H_ION_MASS if "ESI(-)" in ion_mode else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)})
    return pd.DataFrame(calibrants)


def batch_process_fticr_spectra(
    files_input: Union[Dict[str, bytes], bytes, io.BytesIO, str],
    ion_mode: str = "ESI(-)",
    ppm_tolerance: float = 2.0,
    min_intensity: float = 0.0,
    mz_range: Tuple[float, float] = (150.0, 1000.0),
    lang: str = "ru",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Пакетная обработка коллекции масс-спектров FT-ICR MS (из ZIP-архива, словаря файлов или пути).
    Автоматически парсит, фильтрует пики, приписывает брутто-формулы, рассчитывает
    20 ячеек Ван-Кревелена Перминовой и биомолекулярные дескрипторы для каждого образца.

    Parameters:
    -----------
    files_input : Union[Dict[str, bytes], bytes, io.BytesIO, str]
        Входные данные: словарь {имя_файла: байты}, байты ZIP-архива или путь к ZIP/папке.
    ion_mode : str
        Режим ионизации ('ESI(-)' или 'ESI(+)').
    ppm_tolerance : float
        Допуск погрешности массы в ppm.
    min_intensity : float
        Порог отсечения шума по интенсивности.
    mz_range : Tuple[float, float]
        Диапазон масс (мин, макс).
    lang : str
        Язык локализации ('ru' или 'en').

    Returns:
    --------
    summary_df : pd.DataFrame
        Сводная матрица дескрипторов всех образцов со столбцом 'Sample_ID' (готова для ChemoSuite ML).
    spectra_dict : Dict[str, Dict[str, Any]]
        Словарь обработанных спектров, готовый для обновления st.session_state["spectra_db"].
    """
    files_dict: Dict[str, bytes] = {}

    if isinstance(files_input, dict):
        files_dict = files_input
    elif isinstance(files_input, (bytes, io.BytesIO)) or (isinstance(files_input, str) and files_input.lower().endswith(".zip")):
        bio = io.BytesIO(files_input) if isinstance(files_input, bytes) else (files_input if isinstance(files_input, io.BytesIO) else open(files_input, "rb"))
        with zipfile.ZipFile(bio, "r") as zf:
            for fname in zf.namelist():
                if fname.lower().endswith((".csv", ".tsv", ".txt", ".xy")) and not fname.startswith("__MACOSX"):
                    files_dict[os.path.basename(fname)] = zf.read(fname)
        if isinstance(files_input, str) and not isinstance(bio, io.BytesIO):
            bio.close()
    elif isinstance(files_input, str) and os.path.isdir(files_input):
        for fname in os.listdir(files_input):
            if fname.lower().endswith((".csv", ".tsv", ".txt", ".xy")):
                with open(os.path.join(files_input, fname), "rb") as f_in:
                    files_dict[fname] = f_in.read()
    else:
        raise ValueError("Неподдерживаемый формат входных данных для пакетной обработки спектров.")

    if not files_dict:
        return pd.DataFrame(), {}

    summary_rows = []
    spectra_dict = {}

    for fname, raw_bytes in files_dict.items():
        sample_id = os.path.splitext(fname)[0]
        try:
            # 1. Парсинг
            parsed_df = parse_uploaded_file(raw_bytes, delimiter="Auto", decimal_sep=".", has_header=True)
            if parsed_df.empty or "mass" not in parsed_df.columns:
                continue

            # 2. Фильтрация
            mask = (parsed_df["mass"] >= mz_range[0]) & (parsed_df["mass"] <= mz_range[1])
            if min_intensity > 0:
                mask = mask & (parsed_df["intensity"] >= min_intensity)
            filtered = parsed_df[mask].reset_index(drop=True)
            if filtered.empty:
                continue

            # 3. Приписка формул
            assigned = fast_formula_assigner(filtered, ion_mode=ion_mode, ppm_tolerance=ppm_tolerance)
            if assigned.empty:
                continue

            # 4. Дескрипторы и классы
            desc_df = calculate_descriptors(assigned, lang=lang)

            # 5. Сетка 20 ячеек Перминовой
            _, _, grid_df = compute_vk20_grid(desc_df)

            # 6. Биомолекулярное распределение
            bio_dist = get_biomolecular_distribution(desc_df, lang="en")

            # Формирование строки образца
            row: Dict[str, Any] = {"Sample_ID": sample_id}

            # Добавляем 20 ячеек (VK_1 .. VK_20)
            for _, r_cell in grid_df.iterrows():
                row[str(r_cell["Cell"])] = round(float(r_cell["Pct_Count"]), 2)

            # Базовые хемометрические индексы
            row["AI"] = round(float(desc_df["AI"].mean()), 3)
            row["DBE"] = round(float(desc_df["DBE"].mean()), 2)
            row["H/C"] = round(float(desc_df["H/C"].mean()), 3)
            row["O/C"] = round(float(desc_df["O/C"].mean()), 3)
            row["NOSC"] = round(float(desc_df["NOSC"].mean()), 3)

            # Пулы гетероатомов
            h_counts = desc_df["Hetero_Class"].value_counts(normalize=True) * 100.0
            row["CHO_pct"] = round(float(h_counts.get("CHO", 0.0)), 2)
            row["CHON_pct"] = round(float(h_counts.get("CHON", 0.0)), 2)
            row["CHOS_pct"] = round(float(h_counts.get("CHOS", 0.0)), 2)
            row["CHONS_pct"] = round(float(h_counts.get("CHONS", 0.0)), 2)

            # Биомолекулярные пулы
            for b_name, b_pct in bio_dist.items():
                row[f"Bio_{b_name}"] = b_pct

            row["Assigned_Peaks"] = len(desc_df)
            summary_rows.append(row)

            # Сохранение спектра
            spectra_dict[fname] = {
                "raw_path": fname,
                "file_bytes": raw_bytes,
                "raw_df": parsed_df,
                "parsed_peaks": filtered,
                "assigned_df": desc_df,
            }
        except Exception:
            continue

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, spectra_dict
