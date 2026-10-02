# app.py
import io
import os
import tempfile
import inspect
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

try:
    import nomspectra as ns
    from nomspectra import Spectrum
    NOMSPECTRA_INSTALLED = True
except ImportError:
    NOMSPECTRA_INSTALLED = False

st.set_page_config(
    page_title="NOM-Spectra FT-ICR MS Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2e7bcf;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXACT_MASSES = {
    "C": 12.00000,
    "H": 1.00782,
    "O": 15.99491,
    "N": 14.00307,
    "S": 31.97207,
}
H_ION_MASS = 1.007276

KMD_BASES = {
    "CH2 (Метиленовая группа)": {"nom": 14.00000, "exact": 14.015650, "label": "CH2"},
    "COO (Карбоксилатная группа)": {"nom": 44.00000, "exact": 43.989829, "label": "COO"},
    "O (Оксигенация / окисление)": {"nom": 16.00000, "exact": 15.994915, "label": "O"},
    "H2 (Гидрирование / ненасыщенность)": {"nom": 2.00000, "exact": 2.015650, "label": "H2"},
}

def st_df(data, **kwargs):
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)

def st_plotly(fig, **kwargs):
    try:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kwargs)

def get_plot_download_buttons(fig, base_filename: str):
    col1, col2, _ = st.columns([1.5, 1.5, 5])
    with col1:
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="💾 Скачать PNG (300 DPI)",
            data=png_buf.getvalue(),
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"dl_png_{base_filename}",
        )
    with col2:
        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        st.download_button(
            label="💾 Скачать SVG (Вектор)",
            data=svg_buf.getvalue(),
            file_name=f"{base_filename}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{base_filename}",
        )

def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
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

    res["NOSC"] = np.where(c > 0, 4.0 - (4.0 * c + h - 3.0 * n - 2.0 * o - 2.0 * s) / c, np.nan)

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
            return "Не определено"
        if n_val > 0 and (0.3 < hc < 0.8) and (oc < 0.4):
            return "Пул CHON (0.3 < H/C < 0.8, O/C < 0.4)"
        if ai_val >= 0.5 or (hc < 0.7 and oc <= 0.67):
            return "Конденсированные таннины / Ароматика"
        if 0.7 <= hc < 1.5 and 0.1 <= oc <= 0.67:
            return "Лигнины / CRAM"
        if 0.5 <= hc < 1.5 and 0.67 < oc <= 1.0:
            return "Гидролизуемые таннины"
        if 1.5 <= hc <= 2.0 and oc <= 0.3:
            return "Липиды"
        if 1.5 <= hc <= 2.0 and 0.3 < oc <= 0.67:
            return "Пептиды / Белки / Алифатика"
        if 1.5 <= hc <= 2.0 and 0.67 < oc <= 1.0:
            return "Углеводы"
        return "Прочие компоненты"

    res["Compound_Class"] = res.apply(get_compound_class, axis=1)
    return res

def parse_uploaded_file(
    file_bytes: bytes,
    delimiter: str,
    decimal_sep: str,
    has_header: bool,
) -> pd.DataFrame:
    sep_map = {
        "Авто (автоопределение)": None,
        "Запятая (,)": ",",
        "Точка с запятой (;)": ";",
        "Табуляция (\\t)": "\t",
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

def run_formula_assignment(
    peaks_df: pd.DataFrame,
    bounds: Dict[str, Tuple[int, int]],
    max_hc: float,
    max_oc: float,
    ppm_tolerance: float,
    ion_mode: str,
) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        tmp_path = tmp.name
        peaks_df.to_csv(tmp_path, sep="\t", index=False)

    assigned_df = pd.DataFrame()

    try:
        if NOMSPECTRA_INSTALLED:
            try:
                spec = Spectrum(tmp_path)
                target_method = getattr(spec, "assign_formulas", None) or getattr(spec, "assign", None)

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
                        kwargs["mode"] = "neg" if "ESI(-)" in ion_mode else ("pos" if "ESI(+)" in ion_mode else "neutral")

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
                peaks_df, bounds, max_hc, max_oc, ppm_tolerance, ion_mode
            )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not assigned_df.empty and "C" in assigned_df.columns:
        assigned_df = calculate_descriptors(assigned_df)

    return assigned_df

def fast_formula_assigner(
    peaks_df: pd.DataFrame,
    bounds: Dict[str, Tuple[int, int]],
    max_hc: float,
    max_oc: float,
    ppm_tolerance: float,
    ion_mode: str,
) -> pd.DataFrame:
    c_min, c_max = bounds.get("C", (4, 120))
    h_min, h_max = bounds.get("H", (4, 200))
    o_min, o_max = bounds.get("O", (1, 60))
    n_min, n_max = bounds.get("N", (0, 2))
    s_min, s_max = bounds.get("S", (0, 1))

    if "ESI(-)" in ion_mode:
        ion_offset = -H_ION_MASS
    elif "ESI(+)" in ion_mode:
        ion_offset = H_ION_MASS
    else:
        ion_offset = 0.0

    peaks_m = peaks_df["mass"].values
    peaks_int = peaks_df["intensity"].values
    peaks_norm = peaks_df["norm_intensity"].values if "norm_intensity" in peaks_df.columns else peaks_int

    min_mz = float(peaks_m.min())
    max_mz = float(peaks_m.max())

    c_list, h_list, o_list, n_list, s_list = [], [], [], [], []

    for n in range(n_min, n_max + 1):
        for s in range(s_min, s_max + 1):
            for c in range(c_min, c_max + 1):
                cur_o_max = min(o_max, int(max_oc * c))
                for o in range(o_min, cur_o_max + 1):
                    base_m = (
                        c * EXACT_MASSES["C"]
                        + o * EXACT_MASSES["O"]
                        + n * EXACT_MASSES["N"]
                        + s * EXACT_MASSES["S"]
                        + ion_offset
                    )
                    if base_m > max_mz + 2.0:
                        continue

                    h_low = max(
                        h_min,
                        int(np.ceil(0.2 * c)),
                        int(np.ceil(2 * (c - o - 9) + n)),
                        int(np.ceil((min_mz - 2.0 - base_m) / EXACT_MASSES["H"])),
                    )
                    h_high = min(
                        h_max,
                        int(max_hc * c),
                        int(2 * c + n + 2),
                        int(2 * (11 + c - o) + n),
                        int(np.floor((max_mz + 2.0 - base_m) / EXACT_MASSES["H"])),
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
        return pd.DataFrame()

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
        + ion_offset
    )

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

    rows = []
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
                rows.append({
                    "mass": peaks_m[i],
                    "intensity": peaks_int[i],
                    "norm_intensity": peaks_norm[i],
                    "calc_mass": calc_m,
                    "error_ppm": err_ppm,
                    "C": c_val,
                    "H": h_val,
                    "O": o_val,
                    "N": n_val,
                    "S": s_val,
                    "Formula": f"C{c_val}H{h_val}O{o_val}"
                    + (f"N{n_val}" if n_val > 0 else "")
                    + (f"S{s_val}" if s_val > 0 else ""),
                })

    return pd.DataFrame(rows)

def compute_kmd(masses: np.ndarray, base_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nom = KMD_BASES[base_key]["nom"]
    exact = KMD_BASES[base_key]["exact"]
    km = masses * (nom / exact)
    kmd = km - np.floor(km)
    nkm = np.floor(km).astype(int)
    return km, nkm, kmd

def compute_vk20_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oc_bins = [0.00, 0.25, 0.50, 0.75, 1.000001]
    hc_bins = [0.20, 0.60, 1.00, 1.40, 1.80, 2.200001]

    sub = df[(df["O/C"] >= 0.0) & (df["O/C"] <= 1.0) & (df["H/C"] >= 0.2) & (df["H/C"] <= 2.2)].copy()
    if sub.empty:
        return np.zeros((5, 4)), np.zeros((5, 4)), pd.DataFrame()

    c_idx = np.clip(np.digitize(sub["O/C"], oc_bins) - 1, 0, 3)
    r_idx = np.clip(np.digitize(sub["H/C"], hc_bins) - 1, 0, 4)

    grid_count = np.zeros((5, 4), dtype=float)
    grid_weight = np.zeros((5, 4), dtype=float)

    total_count = len(sub)
    total_int = sub["intensity"].sum() if "intensity" in sub.columns else float(total_count)

    np.add.at(grid_count, (r_idx, c_idx), 1.0)
    if "intensity" in sub.columns:
        np.add.at(grid_weight, (r_idx, c_idx), sub["intensity"].values)
    else:
        grid_weight = grid_count.copy()

    pct_count = (grid_count / total_count) * 100.0 if total_count > 0 else grid_count
    pct_weight = (grid_weight / total_int) * 100.0 if total_int > 0 else grid_weight

    features = []
    for r in range(5):
        for c in range(4):
            vk_num = 1 + r * 4 + c
            features.append({
                "Zone": f"VK_{vk_num}",
                "Count": int(grid_count[r, c]),
                "Relative_Count_pct": round(pct_count[r, c], 3),
                "Weighted_Intensity_pct": round(pct_weight[r, c], 3),
            })
    return pct_count, pct_weight, pd.DataFrame(features)

def align_two_spectra_fast(df_a: pd.DataFrame, df_b: pd.DataFrame, ppm_tol: float = 1.5):
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
        cos_sim = np.dot(ia, ib) / (np.linalg.norm(ia) * np.linalg.norm(ib) + 1e-12)
    else:
        cos_sim = 0.0

    return {
        "df_a": a, "df_b": b,
        "matched_a": matched_a, "matched_b": matched_b,
        "n_a": n_a, "n_b": n_b, "n_common": n_common,
        "jaccard": jaccard, "cos_sim": cos_sim,
    }

def get_calibrant_library(series_name: str, ion_mode: str) -> pd.DataFrame:
    calibrants = []
    if "Жирные кислоты" in series_name:
        for n in range(12, 34):
            m_neut = n * EXACT_MASSES["C"] + 2 * n * EXACT_MASSES["H"] + 2 * EXACT_MASSES["O"]
            m_ion = m_neut - H_ION_MASS if "ESI(-)" in ion_mode else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)
            calibrants.append({"name": f"FA {n}:0 (C{n}H{2*n}O2)", "m_theor": m_ion})
    elif "Гомологи CHO" in series_name:
        for n in range(14, 32):
            h_count = 2 * n - 8
            m_neut = n * EXACT_MASSES["C"] + h_count * EXACT_MASSES["H"] + 7 * EXACT_MASSES["O"]
            m_ion = m_neut - H_ION_MASS if "ESI(-)" in ion_mode else (m_neut + H_ION_MASS if "ESI(+)" in ion_mode else m_neut)
            calibrants.append({"name": f"CHO C{n}H{h_count}O7", "m_theor": m_ion})
    return pd.DataFrame(calibrants)

if "parsed_peaks" not in st.session_state:
    st.session_state["parsed_peaks"] = None
if "assigned_df" not in st.session_state:
    st.session_state["assigned_df"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = "spectrum"

with st.sidebar:
    st.title("⚙️️ Загрузка данных")

    if not NOMSPECTRA_INSTALLED:
        st.info("Библиотека nomspectra не найдена в окружении. Используется встроенное вычислительное ядро.")

    uploaded_file = st.file_uploader(
        "Файл масс-спектра",
        type=["csv", "txt", "tsv", "xy"],
    )

    if uploaded_file is not None:
        st.session_state["filename"] = os.path.splitext(uploaded_file.name)[0]

        with st.expander("Параметры чтения файла", expanded=False):
            delimiter = st.selectbox(
                "Разделитель (Delimiter)",
                [
                    "Авто (автоопределение)",
                    "Табуляция (\\t)",
                    "Запятая (,)",
                    "Точка с запятой (;)",
                    "Пробел",
                ],
                index=0,
            )
            decimal_sep = st.selectbox(
                "Десятичный знак",
                [".", ","],
                index=0,
            )
            has_header = st.checkbox("Файл содержит заголовок", value=True)

        try:
            file_bytes = uploaded_file.getvalue()
            raw_df = parse_uploaded_file(file_bytes, delimiter, decimal_sep, has_header)

            if raw_df.empty:
                st.error("Ошибка: Файл пуст или не удалось распознать строки.")
                st.stop()

            st.write("Предпросмотр данных (первые 5 строк):")
            st_df(raw_df.head(5))

            col_names = list(raw_df.columns)

            def_mz_idx = 0
            def_int_idx = 1 if len(col_names) > 1 else 0
            for idx, cname in enumerate(col_names):
                cn_low = str(cname).lower()
                if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]):
                    def_mz_idx = idx
                elif any(k in cn_low for k in ["int", "i", "count", "abund"]):
                    def_int_idx = idx

            col_mz = st.selectbox("Колонка m/z (масса):", col_names, index=def_mz_idx)
            col_int = st.selectbox("Колонка Intensity (интенсивность):", col_names, index=def_int_idx)

            if col_mz == col_int:
                st.warning("Внимание: выбрана одна и та же колонка для массы и интенсивности!")

            st.markdown("---")
            st.subheader("Фильтрация пиков")

            clean_mass = pd.to_numeric(raw_df[col_mz], errors="coerce")
            clean_int = pd.to_numeric(raw_df[col_int], errors="coerce")
            valid_mask = clean_mass.notna() & clean_int.notna()

            valid_df = pd.DataFrame({
                "mass": clean_mass[valid_mask].astype(float),
                "intensity": clean_int[valid_mask].astype(float),
            })
            valid_df = valid_df[valid_df["mass"] > 0]

            if valid_df.empty:
                st.error("Ошибка: В выбранных колонках нет корректных числовых данных.")
                st.stop()

            min_m_data = float(valid_df["mass"].min())
            max_m_data = float(valid_df["mass"].max())

            mz_range = st.slider(
                "Диапазон m/z (Да):",
                min_value=max(50.0, float(np.floor(min_m_data))),
                max_value=min(2000.0, float(np.ceil(max_m_data))),
                value=(
                    max(100.0, float(np.floor(min_m_data))),
                    min(1000.0, float(np.ceil(max_m_data))),
                ),
                step=10.0,
            )

            max_int_data = float(valid_df["intensity"].max())
            cutoff_intensity = st.number_input(
                "Порог отсечения шума (мин. интенсивность):",
                min_value=0.0,
                max_value=max_int_data,
                value=0.0,
                step=max_int_data * 0.001 if max_int_data > 0 else 1.0,
                format="%.2e",
            )

            filtered_df = valid_df[
                (valid_df["mass"] >= mz_range[0])
                & (valid_df["mass"] <= mz_range[1])
                & (valid_df["intensity"] >= cutoff_intensity)
            ].sort_values("mass").reset_index(drop=True)

            if not filtered_df.empty:
                max_i = filtered_df["intensity"].max()
                filtered_df["norm_intensity"] = (filtered_df["intensity"] / max_i) * 100.0
            else:
                filtered_df["norm_intensity"] = []

            st.session_state["parsed_peaks"] = filtered_df
            st.success(f"Загружено и отфильтровано пиков: {len(filtered_df):,}")

        except KeyError as e:
            st.error(f"Ошибка сопоставления колонок: {e}")
            st.stop()
        except Exception as err:
            st.error(f"Ошибка при обработке файла: {err}")
            st.stop()

    st.markdown("---")
    st.subheader("Параметры формульного присвоения")

    ion_mode = st.selectbox(
        "Режим ионизации:",
        ["ESI(-) [M - H]⁻", "ESI(+) [M + H]⁺", "Нейтральные массы [M]"],
        index=0,
    )
    st.session_state["ion_mode_choice"] = ion_mode

    c_bounds = st.slider("Лимит углерода (C):", 4, 120, (4, 120))
    h_bounds = st.slider("Лимит водорода (H):", 4, 200, (4, 200))
    o_bounds = st.slider("Лимит кислорода (O):", 1, 60, (1, 60))
    n_bounds = st.slider("Лимит азота (N):", 0, 2, (0, 2))
    s_bounds = st.slider("Лимит серы (S):", 0, 1, (0, 1))

    max_oc_val = st.slider("Максимум O/C:", 0.1, 1.0, 1.0, step=0.05)
    max_hc_val = st.slider("Максимум H/C:", 0.2, 2.0, 2.0, step=0.05)
    ppm_tol = st.number_input("Допуск погрешности (ppm):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

st.title("🔬 Спектрометрия NOM сверхвысокого разрешения")

tabs = st.tabs([
    "📈 Масс-спектр (Stick Plot)",
    "🎯 Рекалибровка m/z",
    "🧬 Приписывание формул",
    "🗺️ Диаграмма Ван-Кревелена",
    "🔍 Анализ Кендрика (KMD)",
    "🗂️ Хемотипирование 20 ячеек",
    "⚖️️ Сравнение образцов",
    "📊 Сводные характеристики",
])

with tabs[0]:
    peaks_df = st.session_state["parsed_peaks"]

    if peaks_df is None or peaks_df.empty:
        st.info("Пожалуйста, загрузите файл спектра через боковую панель слева.")
    else:
        st.subheader("Палочковый масс-спектр высокого разрешения")

        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            st.caption(f"Отображено сигналов: {len(peaks_df):,} | Базовый пик: 100.0%")
        with col_ctrl2:
            annotate_top = st.checkbox("Подписать топ-5 пиков", value=True)

        fig, ax = plt.subplots(figsize=(13, 5), dpi=100)

        ax.vlines(
            x=peaks_df["mass"],
            ymin=0,
            ymax=peaks_df["norm_intensity"],
            color="#0b5394",
            linewidth=0.6,
            alpha=0.85,
        )

        if annotate_top and len(peaks_df) > 0:
            top5 = peaks_df.nlargest(5, "norm_intensity")
            for _, row in top5.iterrows():
                ax.annotate(
                    f"{row['mass']:.4f}",
                    xy=(row["mass"], row["norm_intensity"]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=55,
                    fontweight="bold",
                    color="#b45f06",
                )

        ax.set_xlim(peaks_df["mass"].min() - 10, peaks_df["mass"].max() + 10)
        ax.set_ylim(0, 118)
        ax.set_xlabel("m/z (Дальтон)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Относительная интенсивность (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        st.pyplot(fig)
        get_plot_download_buttons(fig, "fticr_mass_spectrum")
        plt.close(fig)

with tabs[1]:
    st.subheader("🎯 Внутренняя рекалибровочная коррекция шкалы m/z")
    peaks_df = st.session_state.get("parsed_peaks")

    if peaks_df is None or peaks_df.empty:
        st.info("Сначала загрузите спектр в боковой панели слева.")
    else:
        c_r1, c_r2, c_r3 = st.columns([2, 1.5, 1.5])
        with c_r1:
            calib_type = st.selectbox(
                "Набор калибрантов (внутренний стандарт):",
                ["Жирные кислоты (C12–C33 насыщенные ЖК)", "Гомологи CHO (C_n H_{2n-8} O7)"],
                index=0
            )
        with c_r2:
            search_tol = st.number_input("Окно поиска реперов (ppm):", min_value=2.0, max_value=25.0, value=8.0, step=0.5)
        with c_r3:
            poly_order = st.selectbox("Порядок полинома коррекции:", [1, 2], index=1)

        cur_mode = st.session_state.get("ion_mode_choice", "ESI(-) [M - H]⁻")
        calib_lib = get_calibrant_library(calib_type, cur_mode)

        exp_masses = peaks_df["mass"].values
        matched_calibs = []

        for _, row in calib_lib.iterrows():
            m_th = row["m_theor"]
            delta_lim = m_th * (search_tol * 1e-6)
            candidates = exp_masses[(exp_masses >= m_th - delta_lim) & (exp_masses <= m_th + delta_lim)]
            if len(candidates) > 0:
                best_m = candidates[np.argmin(np.abs(candidates - m_th))]
                err_ppm = ((best_m - m_th) / m_th) * 1e6
                delta_m = best_m - m_th
                matched_calibs.append({
                    "Calibrant": row["name"],
                    "m_theor": m_th,
                    "m_exp": best_m,
                    "delta_m": delta_m,
                    "error_ppm": err_ppm
                })

        if len(matched_calibs) < 3:
            st.warning(f"Найдено реперных пиков: {len(matched_calibs)} (необходимо минимум 3 для построения полинома). Попробуйте увеличить окно поиска ppm.")
        else:
            calib_df = pd.DataFrame(matched_calibs)
            st.write(f"Найдено калибровочных реперов в спектре: **{len(calib_df)}**")

            # Проверка покрытия диапазона масс спектра
            spec_min = float(peaks_df["mass"].min())
            spec_max = float(peaks_df["mass"].max())
            spec_span = max(1.0, spec_max - spec_min)

            calib_min = float(calib_df["m_exp"].min())
            calib_max = float(calib_df["m_exp"].max())
            calib_span = max(0.0, calib_max - calib_min)
            coverage = calib_span / spec_span

            # Защита от эффекта Рунге: если охват < 60%, снижаем полином до 1
            actual_poly_order = poly_order
            if poly_order > 1 and coverage < 0.60:
                actual_poly_order = 1
                st.warning(
                    f"⚠️ Диапазон обнаруженных калибрантов ({calib_min:.1f}–{calib_max:.1f} Да) покрывает "
                    f"**{coverage * 100:.1f}%** диапазона масс спектра (< 60%). "
                    "Степень полинома принудительно понижена до 1 (линейная аппроксимация) для предотвращения "
                    "эффекта Рунге и краевого улета погрешности на массах > 500 Да."
                )

            coeffs = np.polyfit(calib_df["m_exp"].values, calib_df["delta_m"].values, deg=actual_poly_order)
            poly_fn = np.poly1d(coeffs)

            corrected_exp = calib_df["m_exp"] - poly_fn(calib_df["m_exp"])
            residual_ppm = ((corrected_exp - calib_df["m_theor"]) / calib_df["m_theor"]) * 1e6

            fig_rec, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

            ax1.scatter(calib_df["m_exp"], calib_df["error_ppm"], color="#D62728", s=35, label="Реперы (до коррекции)")
            m_grid = np.linspace(peaks_df["mass"].min(), peaks_df["mass"].max(), 200)
            grid_ppm_drift = (poly_fn(m_grid) / m_grid) * 1e6
            ax1.plot(m_grid, grid_ppm_drift, color="black", linestyle="--", linewidth=1.2, label=f"Полином степени {actual_poly_order}")
            ax1.set_xlabel("m/z", fontsize=10)
            ax1.set_ylabel("Погрешность массы (ppm)", fontsize=10)
            ax1.set_title("Систематический дрейф ppm до рекалибровки", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.4)
            ax1.legend()

            ax2.scatter(calib_df["m_exp"], residual_ppm, color="#2CA02C", s=35, label="Реперы (после коррекции)")
            ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
            ax2.set_xlabel("m/z", fontsize=10)
            ax2.set_ylabel("Остаточная погрешность (ppm)", fontsize=10)
            ax2.set_title("Погрешность после полиномиальной рекалибровки", fontsize=11)
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig_rec)
            get_plot_download_buttons(fig_rec, "recalibration_diagnostics")
            plt.close(fig_rec)

            if st.button("🚀 Применить рекалиброванные массы для приписывания формул", type="primary"):
                m_orig = peaks_df["mass"].values
                m_recalibrated = m_orig - poly_fn(m_orig)

                st.session_state["parsed_peaks"]["mass"] = m_recalibrated
                st.session_state["assigned_df"] = None

                st.success(
                    f"Шкала m/z успешно рекалибрована! Всего скорректировано пиков: {len(m_recalibrated):,}. "
                    f"Среднеквадратичная ошибка снижена с {calib_df['error_ppm'].abs().mean():.2f} ppm до {residual_ppm.abs().mean():.2f} ppm. "
                    "Перейдите во вкладку «Приписывание формул» для получения чистых формул."
                )

with tabs[2]:
    st.subheader("Приписывание формул (Formula Assignment)")
    peaks_df = st.session_state["parsed_peaks"]

    if peaks_df is None or peaks_df.empty:
        st.info("Сначала загрузите спектр в боковой панели.")
    else:
        st.write(
            f"Режим: **{ion_mode}** | C [{c_bounds[0]}-{c_bounds[1]}], H [{h_bounds[0]}-{h_bounds[1]}], "
            f"O [{o_bounds[0]}-{o_bounds[1]}], N [{n_bounds[0]}-{n_bounds[1]}], S [{s_bounds[0]}-{s_bounds[1]}], "
            f"O/C <= {max_oc_val}, H/C <= {max_hc_val}, Допуск <= {ppm_tol:.1f} ppm"
        )

        if st.button("🚀 Запустить приписывание формул", type="primary"):
            elem_bounds = {
                "C": c_bounds,
                "H": h_bounds,
                "O": o_bounds,
                "N": n_bounds,
                "S": s_bounds,
            }

            with st.spinner("Выполняется идентификация брутто-формул и расчёт молекулярных индексов..."):
                assigned_res = run_formula_assignment(
                    peaks_df=peaks_df,
                    bounds=elem_bounds,
                    max_hc=max_hc_val,
                    max_oc=max_oc_val,
                    ppm_tolerance=ppm_tol,
                    ion_mode=ion_mode,
                )
                st.session_state["assigned_df"] = assigned_res

        assigned_data = st.session_state["assigned_df"]
        if assigned_data is not None and not assigned_data.empty:
            total_n = len(peaks_df)
            assigned_n = len(assigned_data)
            rate = (assigned_n / total_n * 100.0) if total_n > 0 else 0.0
            mean_ppm = assigned_data["error_ppm"].abs().mean() if "error_ppm" in assigned_data.columns else 0.0

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Всего пиков", f"{total_n:,}")
            m2.metric("Приписано формул", f"{assigned_n:,}")
            m3.metric("Эффективность (Rate)", f"{rate:.1f}%")
            m4.metric("Средняя |ppm| ошибка", f"{mean_ppm:.3f}")

            st.write("Таблица идентифицированных формул:")
            st_df(assigned_data)

            csv_data = assigned_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать результат идентификации (CSV)",
                data=csv_data,
                file_name="nom_formula_assignments.csv",
                mime="text/csv",
            )
        elif assigned_data is not None and assigned_data.empty:
            st.warning("В заданных границах элементов и ppm-погрешности формул не найдено.")

with tabs[3]:
    assigned_data = st.session_state.get("assigned_df")

    if assigned_data is None or assigned_data.empty:
        st.warning("⚠️ Сначала выполните приписывание формул во вкладке 3.")
    else:
        st.subheader("Диаграмма Ван-Кревелена (Van Krevelen Plot)")

        fig_vk, ax_vk = plt.subplots(figsize=(12, 7.5), dpi=150)

        palette = {
            "CHO": "#0020C2",
            "CHON": "#FF7F0E",
            "CHOS": "#2CA02C",
            "CHONS": "#D62728",
        }

        draw_order = ["CHO", "CHON", "CHOS", "CHONS"]

        for cls in draw_order:
            sub = assigned_data[assigned_data["Hetero_Class"] == cls]
            if sub.empty:
                continue

            pt_alpha = 0.50 if cls == "CHO" else 0.75
            pt_size = 1.8 if cls == "CHO" else 2.5

            ax_vk.scatter(
                sub["O/C"],
                sub["H/C"],
                c=palette[cls],
                label=f"{cls} ({len(sub):,})",
                s=pt_size,
                alpha=pt_alpha,
                edgecolors="none",
                rasterized=True,
            )

        ax_vk.set_xlim(0.0, 1.0)
        ax_vk.set_ylim(0.2, 2.2)
        ax_vk.set_xlabel("O/C", fontsize=11, fontweight="medium")
        ax_vk.set_ylabel("H/C", fontsize=11, fontweight="medium")

        fname = st.session_state.get("filename", "example")
        ax_vk.set_title(f"{fname}, {len(assigned_data):,} formulas", fontsize=12, pad=10)
        ax_vk.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
        ax_vk.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            markerscale=4,
            fontsize=9,
        )
        plt.tight_layout()

        st.pyplot(fig_vk)
        get_plot_download_buttons(fig_vk, f"{fname}_van_krevelen")
        plt.close(fig_vk)

        st.markdown("---")
        st.subheader("Распределение идентифицированных компонентов")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**Распределение по классам гетероатомов**")
            hetero_counts = assigned_data["Hetero_Class"].value_counts().reindex(draw_order).dropna().reset_index()
            hetero_counts.columns = ["Класс", "Количество"]
            hetero_counts["Доля (%)"] = (hetero_counts["Количество"] / len(assigned_data)) * 100.0

            fig_h_bar = px.bar(
                hetero_counts,
                x="Класс",
                y="Количество",
                color="Класс",
                color_discrete_map=palette,
                text=hetero_counts["Доля (%)"].apply(lambda v: f"{v:.1f}%"),
            )
            fig_h_bar.update_layout(showlegend=False, height=350)
            st_plotly(fig_h_bar)

        with col_h2:
            st.markdown("**Распределение по структурным пулам**")
            comp_counts = assigned_data["Compound_Class"].value_counts().reset_index()
            comp_counts.columns = ["Класс/Пул", "Количество"]
            comp_counts["Доля (%)"] = (comp_counts["Количество"] / len(assigned_data)) * 100.0

            fig_c_bar = px.bar(
                comp_counts,
                x="Класс/Пул",
                y="Количество",
                color="Класс/Пул",
                text=comp_counts["Доля (%)"].apply(lambda v: f"{v:.1f}%"),
            )
            fig_c_bar.update_layout(showlegend=False, xaxis_tickangle=-30, height=350)
            st_plotly(fig_c_bar)

with tabs[4]:
    st.subheader("🔍 Анализ дефекта массы Кендрика (Kendrick Mass Defect, KMD)")
    assigned_df = st.session_state.get("assigned_df")
    peaks_df = st.session_state.get("parsed_peaks")

    if assigned_df is not None and not assigned_df.empty:
        work_df = assigned_df.copy()
        has_formulas = True
    elif peaks_df is not None and not peaks_df.empty:
        work_df = peaks_df.copy()
        has_formulas = False
    else:
        st.info("Загрузите спектр в боковой панели слева для построения KMD.")
        work_df = None

    if work_df is not None:
        col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
        with col_b1:
            base_choice = st.selectbox("Базовая функциональная группа KMD:", list(KMD_BASES.keys()), index=0)
        with col_b2:
            color_options = ["Четность NKM (Радикалы / Азот)", "Интенсивность"]
            if has_formulas:
                color_options.extend(["Число атомов кислорода (O)", "Индекс ненасыщенности (DBE)", "Классы гетероатомов"])
            color_mode = st.selectbox("Цветовая дифференциация серий:", color_options, index=0)
        with col_b3:
            point_sz = st.slider("Размер точек:", 1, 8, 2, key="kmd_pt_sz")

        # Расчет Маршалла: KMD = KM - np.floor(KM) в диапазоне [0, 1)
        km, nkm, kmd = compute_kmd(work_df["mass"].values, base_choice)
        work_df["KM"] = km
        work_df["NKM"] = nkm
        work_df["KMD"] = kmd
        base_label = KMD_BASES[base_choice]["label"]

        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

        if color_mode == "Четность NKM (Радикалы / Азот)":
            even_mask = (work_df["NKM"] % 2 == 0)
            ax.scatter(work_df.loc[even_mask, "NKM"], work_df.loc[even_mask, "KMD"],
                       c="#0020C2", label="Четный NKM", s=point_sz, alpha=0.6, edgecolors="none", rasterized=True)
            ax.scatter(work_df.loc[~even_mask, "NKM"], work_df.loc[~even_mask, "KMD"],
                       c="#FF7F0E", label="Нечетный NKM", s=point_sz, alpha=0.7, edgecolors="none", rasterized=True)
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
        elif color_mode == "Классы гетероатомов" and has_formulas:
            palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}
            for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                sub = work_df[work_df["Hetero_Class"] == cls]
                if not sub.empty:
                    ax.scatter(sub["NKM"], sub["KMD"], c=palette[cls], label=f"{cls} ({len(sub):,})",
                               s=point_sz, alpha=0.65, edgecolors="none", rasterized=True)
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
        else:
            if color_mode == "Число атомов кислорода (O)" and has_formulas:
                c_vals = work_df["O"]
                cmap_name = "viridis"
                cbar_label = "Число атомов O"
            elif color_mode == "Индекс ненасыщенности (DBE)" and has_formulas:
                c_vals = work_df["DBE"]
                cmap_name = "plasma"
                cbar_label = "DBE"
            else:
                c_vals = work_df["intensity"]
                cmap_name = "cividis"
                cbar_label = "Интенсивность"

            sc = ax.scatter(work_df["NKM"], work_df["KMD"], c=c_vals, cmap=cmap_name,
                            s=point_sz, alpha=0.65, edgecolors="none", rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, pad=0.015, aspect=25)
            cbar.set_label(cbar_label, fontsize=10)

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"Номинальная масса Кендрика NKM ({base_label})", fontsize=11, fontweight="medium")
        ax.set_ylabel(f"Дефект массы Кендрика KMD [0, 1) ({base_label})", fontsize=11, fontweight="medium")
        ax.set_title(f"Kendrick Mass Defect Plot (База: {base_label}, шкала Маршалла [0, 1)) — {len(work_df):,} сигналов", fontsize=12, pad=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
        plt.tight_layout()

        st.pyplot(fig)
        get_plot_download_buttons(fig, f"kmd_{base_label}")
        plt.close(fig)

with tabs[5]:
    st.subheader("🗂 Хемотипирование по 20 ячейкам Ван-Кревелена (сетка Перминовой И.В.)")
    assigned_df = st.session_state.get("assigned_df")

    if assigned_df is None or assigned_df.empty:
        st.warning("⚠️ Сначала выполните приписывание формул во вкладке 3.")
    else:
        pct_count, pct_weight, feat_df = compute_vk20_grid(assigned_df)

        ctrl_col1, ctrl_col2 = st.columns([2, 2])
        with ctrl_col1:
            metric_mode = st.radio("Базис расчета заселенности:", ["Относительное число формул (%)", "Взвешенная интенсивность (%)"], horizontal=True)
        with ctrl_col2:
            overlay_on_vk = st.checkbox("Наложить границы 20 ячеек на диаграмму Ван-Кревелена", value=True)

        active_matrix = pct_count if "число" in metric_mode else pct_weight

        fig_hm, ax_hm = plt.subplots(figsize=(9, 6.5), dpi=150)
        im = ax_hm.imshow(active_matrix, origin="lower", cmap="YlOrRd", aspect="auto")

        c_labels = ["C1 [0.00-0.25)", "C2 [0.25-0.50)", "C3 [0.50-0.75)", "C4 [0.75-1.00]"]
        r_labels = ["R1 [0.20-0.60)", "R2 [0.60-1.00)", "R3 [1.00-1.40)", "R4 [1.40-1.80)", "R5 [1.80-2.20]"]
        ax_hm.set_xticks(range(4))
        ax_hm.set_xticklabels(c_labels, fontsize=9)
        ax_hm.set_yticks(range(5))
        ax_hm.set_yticklabels(r_labels, fontsize=9)
        ax_hm.set_xlabel("Ось O/C", fontsize=11, fontweight="medium")
        ax_hm.set_ylabel("Ось H/C", fontsize=11, fontweight="medium")
        ax_hm.set_title(f"Тепловая карта заселенности 20 ячеек ({metric_mode})", fontsize=11, pad=10)

        for r in range(5):
            for c in range(4):
                vk_num = 1 + r * 4 + c
                val = active_matrix[r, c]
                txt_color = "white" if val > (active_matrix.max() * 0.65) else "black"
                ax_hm.text(c, r, f"VK_{vk_num}\n{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

        plt.colorbar(im, ax=ax_hm, pad=0.02, label="Заселенность (%)")
        plt.tight_layout()
        st.pyplot(fig_hm)
        get_plot_download_buttons(fig_hm, "vk20_heatmap")
        plt.close(fig_hm)

        if overlay_on_vk:
            st.markdown("---")
            st.markdown("##### Диаграмма Ван-Кревелена с координатной сеткой Перминовой И.В.")
            fig_ov, ax_ov = plt.subplots(figsize=(12, 7.5), dpi=150)

            palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}
            for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                sub = assigned_df[assigned_df["Hetero_Class"] == cls]
                if not sub.empty:
                    ax_ov.scatter(sub["O/C"], sub["H/C"], c=palette[cls], label=f"{cls} ({len(sub)})",
                                  s=1.8 if cls == "CHO" else 2.5, alpha=0.45 if cls == "CHO" else 0.75,
                                  edgecolors="none", rasterized=True)

            for x_line in [0.25, 0.50, 0.75]:
                ax_ov.axvline(x=x_line, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
            for y_line in [0.60, 1.00, 1.40, 1.80]:
                ax_ov.axhline(y=y_line, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)

            c_mids = [0.125, 0.375, 0.625, 0.875]
            r_mids = [0.40, 0.80, 1.20, 1.60, 2.00]
            for r in range(5):
                for c in range(4):
                    vk_num = 1 + r * 4 + c
                    ax_ov.text(c_mids[c], r_mids[r], f"VK_{vk_num}", color="#111111", fontsize=9,
                               fontweight="bold", alpha=0.55, ha="center", va="center")

            ax_ov.set_xlim(0.0, 1.0)
            ax_ov.set_ylim(0.2, 2.2)
            ax_ov.set_xlabel("O/C", fontsize=11, fontweight="medium")
            ax_ov.set_ylabel("H/C", fontsize=11, fontweight="medium")
            ax_ov.set_title("Сетка 20 хемотипических зон (диссертационная классификация лаборатории ГВ)", fontsize=11)
            ax_ov.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
            plt.tight_layout()

            st.pyplot(fig_ov)
            get_plot_download_buttons(fig_ov, "vk20_overlay")
            plt.close(fig_ov)

        st.markdown("---")
        st.write("Таблица численных дескрипторов 20 ячеек (VK_1 ... VK_20):")
        st_df(feat_df)

        csv_buf = feat_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Скачать вектор дескрипторов 20 ячеек (CSV)",
            data=csv_buf,
            file_name="vk20_features.csv",
            mime="text/csv",
        )

with tabs[6]:
    st.subheader("⚖️ Сравнение спектральных ансамблей (Set Operations)")
    peaks_a = st.session_state.get("parsed_peaks")

    if peaks_a is None or peaks_a.empty:
        st.info("Сначала загрузите Образец А через боковую панель слева.")
    else:
        st.markdown("##### Загрузка спектра Образца Б (или холостой пробы/бланка)")
        file_b = st.file_uploader("Файл Образца Б (.csv, .txt, .tsv, .xy)", type=["csv", "txt", "tsv", "xy"], key="file_b")

        if file_b is not None:
            with st.expander("Параметры чтения и выравнивания Образца Б", expanded=False):
                c_sep, c_dec = st.columns(2)
                with c_sep:
                    sep_b = st.selectbox("Разделитель Образца Б:", ["Авто (автоопределение)", "Табуляция (\\t)", "Запятая (,)", "Точка с запятой (;)", "Пробел"], index=0, key="sep_b")
                with c_dec:
                    dec_b = st.selectbox("Десятичный знак Образца Б:", [".", ","], index=0, key="dec_b")
                tol_comp = st.number_input("Допуск совмещения m/z (ppm):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

            raw_df_b = parse_uploaded_file(file_b.getvalue(), sep_b, dec_b, has_header=True)
            cols_b = list(raw_df_b.columns)
            col_mz_b = st.selectbox("Колонка массы Образца Б:", cols_b, index=0, key="mz_b")
            col_int_b = st.selectbox("Колонка интенсивности Образца Б:", cols_b, index=1 if len(cols_b) > 1 else 0, key="int_b")

            clean_mb = pd.to_numeric(raw_df_b[col_mz_b], errors="coerce")
            clean_ib = pd.to_numeric(raw_df_b[col_int_b], errors="coerce")
            val_mask = clean_mb.notna() & clean_ib.notna() & (clean_mb > 0)
            df_b = pd.DataFrame({"mass": clean_mb[val_mask].astype(float), "intensity": clean_ib[val_mask].astype(float)})

            if df_b.empty:
                st.error("В файле Образца Б нет валидных числовых данных.")
            else:
                align_res = align_two_spectra_fast(peaks_a, df_b, ppm_tol=tol_comp)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Пиков в А", f"{align_res['n_a']:,}")
                m2.metric("Пиков в Б", f"{align_res['n_b']:,}")
                m3.metric("Общих (A ∩ B)", f"{align_res['n_common']:,}", delta=f"Индекс Жаккара: {align_res['jaccard']:.3f}")
                m4.metric("Косинусное сходство", f"{align_res['cos_sim']:.4f}")

                assigned_a = st.session_state.get("assigned_df")
                if assigned_a is not None and not assigned_a.empty:
                    st.markdown("##### Сравнительная диаграмма Ван-Кревелена")
                    common_masses = set(align_res["df_a"].loc[align_res["matched_a"], "mass"].round(4))

                    assigned_a_common = assigned_a[assigned_a["mass"].round(4).isin(common_masses)]
                    assigned_a_unique = assigned_a[~assigned_a["mass"].round(4).isin(common_masses)]

                    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 7), dpi=150)
                    if not assigned_a_common.empty:
                        ax_cmp.scatter(assigned_a_common["O/C"], assigned_a_common["H/C"], c="#7F7F7F",
                                       label=f"Общие A ∩ B ({len(assigned_a_common):,})", s=2.0, alpha=0.4, edgecolors="none", rasterized=True)
                    if not assigned_a_unique.empty:
                        ax_cmp.scatter(assigned_a_unique["O/C"], assigned_a_unique["H/C"], c="#0020C2",
                                       label=f"Уникальные для А (A \\ B) ({len(assigned_a_unique):,})", s=2.2, alpha=0.75, edgecolors="none", rasterized=True)

                    ax_cmp.set_xlim(0.0, 1.0)
                    ax_cmp.set_ylim(0.2, 2.2)
                    ax_cmp.set_xlabel("O/C", fontsize=11, fontweight="medium")
                    ax_cmp.set_ylabel("H/C", fontsize=11, fontweight="medium")
                    ax_cmp.set_title("Сравнительный анализ химического пространства (A vs B)", fontsize=11)
                    ax_cmp.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
                    ax_cmp.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
                    plt.tight_layout()

                    st.pyplot(fig_cmp)
                    get_plot_download_buttons(fig_cmp, "samples_comparison_vk")
                    plt.close(fig_cmp)
                else:
                    st.markdown("##### Зеркальный спектр совмещения (Head-to-Tail Stick Plot)")
                    fig_mir, ax_mir = plt.subplots(figsize=(12, 5), dpi=150)

                    ia_norm = (peaks_a["intensity"] / peaks_a["intensity"].max()) * 100.0
                    ib_norm = (df_b["intensity"] / df_b["intensity"].max()) * 100.0

                    ax_mir.vlines(peaks_a["mass"], 0, ia_norm, color="#0020C2", linewidth=0.6, alpha=0.7, label="Образец А (+)")
                    ax_mir.vlines(df_b["mass"], 0, -ib_norm, color="#2CA02C", linewidth=0.6, alpha=0.7, label="Образец Б (-)")
                    ax_mir.axhline(0, color="black", linewidth=0.8)

                    ax_mir.set_xlabel("m/z", fontsize=11)
                    ax_mir.set_ylabel("Отн. интенсивность (%) [A: вверх / Б: вниз]", fontsize=10)
                    ax_mir.legend(loc="upper right")
                    plt.tight_layout()
                    st.pyplot(fig_mir)
                    get_plot_download_buttons(fig_mir, "head_to_tail_comparison")
                    plt.close(fig_mir)

with tabs[7]:
    assigned_data = st.session_state.get("assigned_df")

    if assigned_data is None or assigned_data.empty:
        st.warning("⚠️ Сначала выполните приписывание формул во вкладке 3.")
    else:
        st.subheader("Сводные характеристики ансамбля NOM")

        mw_n = assigned_data["mass"].mean()
        hc_n = assigned_data["H/C"].mean()
        oc_n = assigned_data["O/C"].mean()
        dbe_n = assigned_data["DBE"].mean()
        dbe_o_n = assigned_data["DBE-O"].mean()
        ai_n = assigned_data["AI"].mean()

        weights = assigned_data["intensity"].values
        sum_weights = np.sum(weights)

        if sum_weights > 0:
            mw_w = np.sum(assigned_data["mass"].values * weights) / sum_weights
            hc_w = np.sum(assigned_data["H/C"].values * weights) / sum_weights
            oc_w = np.sum(assigned_data["O/C"].values * weights) / sum_weights
            dbe_w = np.sum(assigned_data["DBE"].values * weights) / sum_weights
            dbe_o_w = np.sum(assigned_data["DBE-O"].values * weights) / sum_weights
            ai_w = np.sum(assigned_data["AI"].values * weights) / sum_weights
        else:
            mw_w, hc_w, oc_w, dbe_w, dbe_o_w, ai_w = mw_n, hc_n, oc_n, dbe_n, dbe_o_n, ai_n

        st.markdown("#### Среднечисленные значения (Number-averaged parameters)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Среднечисленная масса (Mn)", f"{mw_n:.2f} Да")
        c2.metric("Среднечисленное H/C", f"{hc_n:.3f}")
        c3.metric("Среднечисленное O/C", f"{oc_n:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        c4.metric("Среднечисленный DBE", f"{dbe_n:.2f}")
        c5.metric("Среднечисленный DBE - O", f"{dbe_o_n:.2f}")
        c6.metric("Среднечисленный AI (Кох)", f"{ai_n:.3f}")

        st.markdown("---")
        st.markdown("#### Сравнение со средневзвешенными значениями по интенсивности (Mw-weighted)")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Mw (Взвешенная масса)", f"{mw_w:.2f} Да", delta=f"{mw_w - mw_n:+.2f}")
        w2.metric("H/C (Взвешенное)", f"{hc_w:.3f}", delta=f"{hc_w - hc_n:+.3f}")
        w3.metric("O/C (Взвешенное)", f"{oc_w:.3f}", delta=f"{oc_w - oc_n:+.3f}")
        w4.metric("AI (Взвешенное)", f"{ai_w:.3f}", delta=f"{ai_w - ai_n:+.3f}")

        st.markdown("---")
        st.markdown("#### Полная сводная статистика дескрипторов")
        stat_cols = ["mass", "H/C", "O/C", "DBE", "DBE-O", "AI", "error_ppm"]
        summary_stats = assigned_data[stat_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        summary_stats.columns = ["Среднее", "Стд. откл.", "Мин.", "Медиана", "Макс."]
        st_df(summary_stats.style.format("{:.3f}"))