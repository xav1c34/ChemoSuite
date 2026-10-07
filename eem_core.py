"""
eem_core.py — Математическое ядро оптической спектроскопии (EEM-PARAFAC и УФ-Вид).
Методология кафедры аналитической химии и лаборатории природных гуминовых систем химфака МГУ.
"""
from dataclasses import dataclass
import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import griddata
import scipy.signal

try:
    import tensorly as tl
    from tensorly.decomposition import non_negative_parafac
    tl.set_backend("numpy")
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False


@dataclass
class EEMSample:
    sample_id: str
    ex: np.ndarray
    em: np.ndarray
    data: np.ndarray
    doc: Optional[float] = None
    a254: Optional[float] = None


@dataclass
class UVVisSample:
    sample_id: str
    wl: np.ndarray
    absorbance: np.ndarray
    doc: Optional[float] = None


def decode_bytes(b: bytes) -> str:
    """Универсальное декодирование байтов файла."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ==============================================================================
# ПАРСИНГ И ОБРАБОТКА МАТРИЦ ФЛУОРЕСЦЕНЦИИ (EEM)
# ==============================================================================
def parse_eem_dataframe(
    df: pd.DataFrame, sample_id: str = "Sample"
) -> EEMSample:
    """
    Универсальный парсер DataFrame матрицы EEM.
    Автоматически очищает служебные строки (nm, CPS), текстовые шапки,
    находит числовые сетки Ex/Em и ориентирует матрицу (Em — строки, Ex — столбцы).
    """
    cleaned = df.copy()

    if hasattr(cleaned, "map"):
        cleaned = cleaned.map(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        cleaned = cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    col_num = pd.to_numeric(cleaned.columns.astype(str).str.replace(",", "."), errors="coerce")
    col_wv_count = int(np.sum((col_num >= 180) & (col_num <= 1200)))

    if col_wv_count < 3 and len(cleaned) > 1:
        row0_num = pd.to_numeric(cleaned.iloc[0].astype(str).str.replace(",", "."), errors="coerce")
        row0_wv_count = int(np.sum((row0_num >= 180) & (row0_num <= 1200)))
        if row0_wv_count > col_wv_count:
            cleaned.columns = cleaned.iloc[0]
            cleaned = cleaned.iloc[1:].copy()

    idx_num = pd.to_numeric(cleaned.index.astype(str).str.replace(",", "."), errors="coerce")
    idx_wv_count = int(np.sum((idx_num >= 180) & (idx_num <= 1200)))

    if idx_wv_count < 3 and cleaned.shape[1] > 1:
        col0_num = pd.to_numeric(cleaned.iloc[:, 0].astype(str).str.replace(",", "."), errors="coerce")
        col0_wv_count = int(np.sum((col0_num >= 180) & (col0_num <= 1200)))
        if col0_wv_count > idx_wv_count:
            cleaned = cleaned.set_index(cleaned.columns[0])

    idx_num = pd.to_numeric(cleaned.index.astype(str).str.replace(",", "."), errors="coerce")
    row_mask = idx_num.notna() & (idx_num >= 180) & (idx_num <= 1200)
    cleaned = cleaned.loc[row_mask].copy()
    row_wavelengths = idx_num[row_mask].values.astype(float)

    col_num = pd.to_numeric(cleaned.columns.astype(str).str.replace(",", "."), errors="coerce")
    col_mask = col_num.notna() & (col_num >= 180) & (col_num <= 1200)
    cleaned = cleaned.loc[:, col_mask].copy()
    col_wavelengths = col_num[col_mask].values.astype(float)

    if len(row_wavelengths) < 3 or len(col_wavelengths) < 3:
        raise ValueError(
            f"Не удалось распознать сетку длин волн EEM для '{sample_id}'. "
            f"Обнаружено: строк={len(row_wavelengths)}, колонок={len(col_wavelengths)}."
        )

    def to_float_val(v):
        if isinstance(v, (int, float, np.number)):
            return float(v)
        try:
            return float(str(v).replace(",", ".").strip())
        except (ValueError, TypeError):
            return 0.0

    if hasattr(cleaned, "map"):
        mat_df = cleaned.map(to_float_val)
    else:
        mat_df = cleaned.applymap(to_float_val)

    matrix = mat_df.fillna(0.0).values.astype(float)

    if np.mean(col_wavelengths) < np.mean(row_wavelengths):
        ex_vals = col_wavelengths
        em_vals = row_wavelengths
        data = matrix
    else:
        ex_vals = row_wavelengths
        em_vals = col_wavelengths
        data = matrix.T

    if len(ex_vals) > 1 and ex_vals[1] < ex_vals[0]:
        ex_sort = np.argsort(ex_vals)
        ex_vals = ex_vals[ex_sort]
        data = data[:, ex_sort]

    if len(em_vals) > 1 and em_vals[1] < em_vals[0]:
        em_sort = np.argsort(em_vals)
        em_vals = em_vals[em_sort]
        data = data[em_sort, :]

    return EEMSample(
        sample_id=sample_id,
        ex=ex_vals,
        em=em_vals,
        data=data,
    )


def load_eem_file(file_input: Union[str, bytes, Any], sample_id: str = "Sample") -> EEMSample:
    """Чтение матрицы EEM напрямую из файла, пути, байтов или объекта Streamlit UploadedFile."""
    if isinstance(file_input, str):
        if "\n" in file_input or "\r" in file_input:
            text = file_input
        else:
            with open(file_input, "rb") as f:
                text = decode_bytes(f.read())
    elif isinstance(file_input, bytes):
        text = decode_bytes(file_input)
    elif hasattr(file_input, "getvalue"):
        b = file_input.getvalue()
        text = decode_bytes(b) if isinstance(b, bytes) else str(b)
    elif hasattr(file_input, "read"):
        b = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        text = decode_bytes(b) if isinstance(b, bytes) else str(b)
    else:
        raise ValueError(f"Неподдерживаемый тип входного файла: {type(file_input)}")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"Файл {sample_id} пуст.")

    data_lines = [l for l in lines if not l.startswith(("#", "!", "//"))]
    if not data_lines:
        raise ValueError(f"Файл {sample_id} содержит только комментарии.")

    sample_block = data_lines[:min(20, len(data_lines))]
    candidates = [",", ";", "\t", " "]
    counts = {c: sum(l.count(c) for l in sample_block) for c in candidates}
    best_sep = max(counts, key=counts.get)
    sep = r"\s+" if best_sep == " " else best_sep

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=sep,
        header=None,
        engine="python",
        dtype=str,
        on_bad_lines="skip",
    )
    return parse_eem_dataframe(df, sample_id=sample_id)


def remove_scatter_bands(
    eem: np.ndarray,
    ex: np.ndarray,
    em: np.ndarray,
    delta_rayleigh1: float = 12.0,
    delta_rayleigh2: float = 15.0,
) -> np.ndarray:
    ex_g, em_g = np.meshgrid(ex, em)
    proc = eem.copy()

    r1_mask = np.abs(em_g - ex_g) <= delta_rayleigh1
    r2_mask = np.abs(em_g - 2.0 * ex_g) <= delta_rayleigh2
    anti_stokes = em_g < (ex_g - 5.0)
    total_mask = r1_mask | r2_mask | anti_stokes

    valid_coords = np.column_stack((em_g[~total_mask], ex_g[~total_mask]))
    valid_vals = proc[~total_mask]
    missing_coords = np.column_stack((em_g[total_mask], ex_g[total_mask]))

    if len(valid_vals) > 0 and len(missing_coords) > 0:
        interpolated = griddata(
            valid_coords, valid_vals, missing_coords, method="linear", fill_value=0.0
        )
        proc[total_mask] = np.maximum(0.0, interpolated)
    return proc


def normalize_to_raman_units(
    eem: np.ndarray,
    ex: np.ndarray,
    em: np.ndarray,
    water_blank: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    if water_blank is not None:
        idx_350 = np.argmin(np.abs(ex - 350.0))
        mask = (em >= 371.0) & (em <= 428.0)
        area = simpson(water_blank[mask, idx_350], x=em[mask])
        factor = float(area) if area > 0 else 1.0
    else:
        factor = 1.0
    return eem / factor, factor


def calculate_spectral_indices(sample: EEMSample) -> Dict[str, Any]:
    ex, em, d = sample.ex, sample.em, sample.data

    idx_ex_370 = np.argmin(np.abs(ex - 370.0))
    idx_em_470 = np.argmin(np.abs(em - 470.0))
    idx_em_520 = np.argmin(np.abs(em - 520.0))
    denom_fi = d[idx_em_520, idx_ex_370]
    fi = float(d[idx_em_470, idx_ex_370] / denom_fi) if denom_fi > 1e-6 else np.nan

    idx_ex_254 = np.argmin(np.abs(ex - 254.0))
    prof_254 = d[:, idx_ex_254]
    m_hi = (em >= 435.0) & (em <= 480.0)
    m_lo = (em >= 300.0) & (em <= 345.0)
    int_hi = simpson(prof_254[m_hi], x=em[m_hi])
    int_lo = simpson(prof_254[m_lo], x=em[m_lo])
    hix = float(int_hi / int_lo) if int_lo > 1e-6 else np.nan

    suva = np.nan
    if sample.a254 is not None and sample.doc is not None and sample.doc > 0:
        suva = float((sample.a254 / sample.doc) * 100.0)

    return {
        "Sample_ID": sample.sample_id,
        "FI": round(fi, 2) if not np.isnan(fi) else np.nan,
        "HIX": round(hix, 2) if not np.isnan(hix) else np.nan,
        "A254": round(sample.a254, 4) if (sample.a254 is not None and not np.isnan(sample.a254)) else np.nan,
        "DOC": round(sample.doc, 2) if (sample.doc is not None and not np.isnan(sample.doc)) else np.nan,
        "SUVA254": round(suva, 2) if not np.isnan(suva) else np.nan,
    }


def build_eem_tensor(
    samples: List[EEMSample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not samples:
        raise ValueError("Список образцов EEM пуст.")

    ref_ex, ref_em = samples[0].ex, samples[0].em
    tensor = np.zeros((len(samples), len(ref_em), len(ref_ex)), dtype=np.float64)
    names = []

    for i, s in enumerate(samples):
        names.append(s.sample_id)
        if (
            len(s.em) == len(ref_em)
            and len(s.ex) == len(ref_ex)
            and np.allclose(s.em, ref_em)
            and np.allclose(s.ex, ref_ex)
        ):
            tensor[i, :, :] = s.data
        else:
            s_ex_g, s_em_g = np.meshgrid(s.ex, s.em)
            ref_ex_g, ref_em_g = np.meshgrid(ref_ex, ref_em)
            points = np.column_stack((s_em_g.ravel(), s_ex_g.ravel()))
            interp = griddata(
                points, s.data.ravel(), (ref_em_g, ref_ex_g), method="linear", fill_value=0.0
            )
            tensor[i, :, :] = np.maximum(0.0, interp)

    return tensor, ref_em, ref_ex, names


def compute_corcondia(
    tensor: np.ndarray, factors: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    if not TENSORLY_AVAILABLE:
        return 0.0
    a, b, c = factors
    r = a.shape[1]
    g = tl.tenalg.multi_mode_dot(
        tensor,
        [np.linalg.pinv(a), np.linalg.pinv(b), np.linalg.pinv(c)],
        modes=[0, 1, 2],
    )
    t = np.zeros_like(g)
    for i in range(r):
        t[i, i, i] = 1.0
    sse = np.sum((g - t) ** 2)
    return float(max(0.0, 100.0 * (1.0 - (sse / float(r)))))


def fit_parafac(
    tensor: np.ndarray, n_components: int = 3, random_state: int = 42
) -> Dict:
    if not TENSORLY_AVAILABLE:
        raise RuntimeError("Библиотека tensorly не установлена в окружении.")
    weights, factors = non_negative_parafac(
        tensor, rank=n_components, n_iter_max=300, tol=1e-6, init="svd", random_state=random_state
    )
    a, b, c = factors
    for r in range(n_components):
        mb, mc = np.max(b[:, r]), np.max(c[:, r])
        if mb > 0 and mc > 0:
            b[:, r] /= mb
            c[:, r] /= mc
            a[:, r] *= mb * mc * weights[r]

    rec = tl.cp_to_tensor((np.ones(n_components), factors))
    tot_ss = np.sum(tensor**2)
    exp_var = 100.0 * (1.0 - (np.sum((tensor - rec) ** 2) / tot_ss))
    corc = compute_corcondia(tensor, (a, b, c))

    return {
        "scores": a,
        "em_profiles": b,
        "ex_profiles": c,
        "explained_variance": float(exp_var),
        "corcondia": corc,
    }


def split_half_analysis(
    tensor: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Сплит-хаф валидация (Split-Half Validation) PARAFAC модели.
    Разделяет набор образцов (режим 0) на две равные части A и B,
    независимо обучает неотрицательный PARAFAC на каждой подвыборке и
    рассчитывает коэффициенты конгруэнтности Такера (Tucker Congruence Coefficient, TCC)
    для спектров испускания (Em) и возбуждения (Ex).

    Returns:
    --------
    Dict с ключами:
      - 'can_split': bool
      - 'tcc_em': np.ndarray (конгруэнтность по эмиссии для каждого компонента)
      - 'tcc_ex': np.ndarray (конгруэнтность по возбуждению для каждого компонента)
      - 'mean_tcc': float (средняя конгруэнтность)
      - 'is_validated': bool (True, если mean_tcc >= 0.90)
    """
    n_samples = tensor.shape[0]
    if n_samples < 4:
        return {
            "can_split": False,
            "error": "Для сплит-хаф анализа требуется как минимум 4 образца (по 2 на половину).",
            "is_validated": False,
            "mean_tcc": 0.0,
            "tcc_em": np.zeros(n_components),
            "tcc_ex": np.zeros(n_components),
        }

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    mid = n_samples // 2
    idx_a = indices[:mid]
    idx_b = indices[mid:]

    tensor_a = tensor[idx_a]
    tensor_b = tensor[idx_b]

    # Обучение на половине A и B
    res_a = fit_parafac(tensor_a, n_components=n_components, random_state=random_state)
    res_b = fit_parafac(tensor_b, n_components=n_components, random_state=random_state + 1)

    em_a = res_a["em_profiles"]
    ex_a = res_a["ex_profiles"]

    em_b = res_b["em_profiles"]
    ex_b = res_b["ex_profiles"]

    def _tcc(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    from itertools import permutations
    best_perm = None
    best_score = -1.0

    for perm in permutations(range(n_components)):
        score = 0.0
        for i_a, i_b in enumerate(perm):
            score += (_tcc(em_a[:, i_a], em_b[:, i_b]) + _tcc(ex_a[:, i_a], ex_b[:, i_b])) / 2.0
        if score > best_score:
            best_score = score
            best_perm = perm

    tcc_em = []
    tcc_ex = []
    for i_a, i_b in enumerate(best_perm):
        c_em = _tcc(em_a[:, i_a], em_b[:, i_b])
        c_ex = _tcc(ex_a[:, i_a], ex_b[:, i_b])
        tcc_em.append(round(c_em, 4))
        tcc_ex.append(round(c_ex, 4))

    tcc_em_arr = np.array(tcc_em)
    tcc_ex_arr = np.array(tcc_ex)
    mean_tcc = float(np.mean((tcc_em_arr + tcc_ex_arr) / 2.0))

    return {
        "can_split": True,
        "is_validated": bool(mean_tcc >= 0.90),
        "mean_tcc": round(mean_tcc, 3),
        "tcc_em": tcc_em_arr,
        "tcc_ex": tcc_ex_arr,
        "matched_perm": list(best_perm),
    }


def generate_synthetic_chemometrics_dataset(n_samples: int = 10) -> List[EEMSample]:
    np.random.seed(42)
    ex = np.linspace(240, 450, 43)
    em = np.linspace(280, 600, 65)
    ex_g, em_g = np.meshgrid(ex, em)

    p1 = np.exp(-(((ex_g - 315) / 35) ** 2 + ((em_g - 405) / 45) ** 2)) + 0.6 * np.exp(-(((ex_g - 245) / 20) ** 2 + ((em_g - 405) / 45) ** 2))
    p2 = np.exp(-(((ex_g - 365) / 45) ** 2 + ((em_g - 470) / 55) ** 2)) + 0.8 * np.exp(-(((ex_g - 260) / 25) ** 2 + ((em_g - 470) / 55) ** 2))
    p3 = np.exp(-(((ex_g - 275) / 22) ** 2 + ((em_g - 340) / 30) ** 2))

    samples = []
    names = [f"Lignin_Impact_{i+1}" if i < 5 else f"Baikal_Water_{i-4}" for i in range(n_samples)]
    for name in names:
        if "Lignin" in name:
            w = (np.random.uniform(1.2, 2.2), np.random.uniform(4.5, 7.5), np.random.uniform(0.1, 0.4))
            doc = np.random.uniform(14.0, 26.0)
            a254 = doc * np.random.uniform(0.045, 0.065)
        else:
            w = (np.random.uniform(2.5, 4.0), np.random.uniform(0.5, 1.2), np.random.uniform(1.2, 2.6))
            doc = np.random.uniform(1.5, 3.8)
            a254 = doc * np.random.uniform(0.015, 0.026)

        raw = w[0] * p1 + w[1] * p2 + w[2] * p3
        clean = remove_scatter_bands(raw, ex, em)
        samples.append(EEMSample(sample_id=name, ex=ex, em=em, data=clean, doc=round(doc, 2), a254=round(a254, 3)))
    return samples


# ==============================================================================
# ПАРСИНГ, ПРОИЗВОДНЫЕ И РАСЧЕТ УФ-ВИД СПЕКТРОВ ПОГЛОЩЕНИЯ (UV-VIS)
# ==============================================================================
def parse_uv_vis_spectrum(file_input: Any, sample_id: str = "Sample") -> UVVisSample:
    """Парсер 1D-спектров поглощения УФ-Вид (200-800 нм)."""
    if hasattr(file_input, "getvalue"):
        content = file_input.getvalue()
        text = decode_bytes(content) if isinstance(content, bytes) else str(content)
    elif hasattr(file_input, "read"):
        content = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        text = decode_bytes(content) if isinstance(content, bytes) else str(content)
    elif isinstance(file_input, str):
        if "\n" in file_input or "\r" in file_input:
            text = file_input
        else:
            with open(file_input, "rb") as f:
                text = decode_bytes(f.read())
    elif isinstance(file_input, bytes):
        text = decode_bytes(file_input)
    else:
        raise ValueError(f"Неподдерживаемый тип входных данных: {type(file_input)}")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data_lines = [l for l in lines if not l.startswith(("#", "!", "//"))]
    if not data_lines:
        raise ValueError(f"Файл {sample_id} пуст или не содержит данных.")

    sample_block = data_lines[:min(20, len(data_lines))]
    candidates = [",", ";", "\t", " "]
    counts = {c: sum(l.count(c) for l in sample_block) for c in candidates}
    best_sep = max(counts, key=counts.get)
    sep = r"\s+" if best_sep == " " else best_sep

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=sep,
        header=None,
        engine="python",
        dtype=str,
        on_bad_lines="skip"
    )

    row0 = df.iloc[0].astype(str).str.lower()
    has_header = any(any(k in cell for k in ["wave", "wl", "nm", "lambda", "int", "abs", "a", "длина"]) for cell in row0)
    if has_header and len(df) > 1:
        df.columns = df.iloc[0]
        df = df.iloc[1:].copy()

    col_wl, col_abs = None, None
    for c in df.columns:
        clow = str(c).lower()
        if any(k in clow for k in ["wave", "wl", "nm", "lambda", "длина"]):
            col_wl = c
        elif any(k in clow for k in ["int", "abs", "a", "opt", "поглощ", "dens"]):
            col_abs = c

    if col_wl is None:
        col_wl = df.columns[0]
    if col_abs is None:
        col_abs = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    wl_series = pd.to_numeric(df[col_wl].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    abs_series = pd.to_numeric(df[col_abs].astype(str).str.replace(",", ".").str.strip(), errors="coerce")

    mask = wl_series.notna() & abs_series.notna() & (wl_series >= 180) & (wl_series <= 1200)
    wl_clean = wl_series[mask].values.astype(float)
    abs_clean = abs_series[mask].values.astype(float)

    if len(wl_clean) < 5:
        raise ValueError(f"Не удалось извлечь спектр УФ-Вид для '{sample_id}'. Найдено точек: {len(wl_clean)}.")

    sort_idx = np.argsort(wl_clean)
    return UVVisSample(
        sample_id=sample_id,
        wl=wl_clean[sort_idx],
        absorbance=abs_clean[sort_idx]
    )


def compute_uv_derivatives(
    wl: np.ndarray,
    absorbance: np.ndarray,
    window_length: int = 21,
    polyorder: int = 3,
) -> Dict[str, Any]:
    """
    Расчет 1-й и 2-й производных спектра поглощения по алгоритму Савицкого — Голея.
    Выявляет скрытые плечи поглощения монолигнолов и фенолов (минимум d2A в 270–290 нм).
    """
    step = float(np.mean(np.diff(wl)))
    w_len = window_length
    if w_len >= len(wl):
        w_len = len(wl) - 1 if len(wl) % 2 == 0 else len(wl) - 2
    if w_len < 5:
        w_len = 5

    d1_a = scipy.signal.savgol_filter(absorbance, window_length=w_len, polyorder=polyorder, deriv=1, delta=step)
    d2_a = scipy.signal.savgol_filter(absorbance, window_length=w_len, polyorder=polyorder, deriv=2, delta=step)

    mask_280 = (wl >= 270.0) & (wl <= 290.0)
    if np.any(mask_280):
        sub_wl = wl[mask_280]
        sub_d2 = d2_a[mask_280]
        min_idx = np.argmin(sub_d2)
        lignin_min_wl = float(sub_wl[min_idx])
        lignin_d2_val = float(sub_d2[min_idx])
    else:
        lignin_min_wl, lignin_d2_val = np.nan, np.nan

    return {
        "d1_a": d1_a,
        "d2_a": d2_a,
        "lignin_min_wl": lignin_min_wl,
        "lignin_d2_val": lignin_d2_val,
    }


def compute_spectral_slope_curve(
    wl: np.ndarray,
    absorbance: np.ndarray,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """Непрерывная кривая спектрального наклона S(lambda) = -d(ln A) / d(lambda) [нм^-1]."""
    w_len = window_length
    if w_len >= len(wl):
        w_len = len(wl) - 1 if len(wl) % 2 == 0 else len(wl) - 2
    if w_len < 5:
        w_len = 5

    s_curve = np.full_like(absorbance, np.nan, dtype=float)
    pos_mask = absorbance > 1e-4

    if np.sum(pos_mask) >= w_len:
        wl_pos = wl[pos_mask]
        a_pos = absorbance[pos_mask]
        step_pos = float(np.mean(np.diff(wl_pos)))
        ln_a = np.log(a_pos)
        d_ln_a = scipy.signal.savgol_filter(ln_a, window_length=w_len, polyorder=polyorder, deriv=1, delta=step_pos)
        s_curve[pos_mask] = -d_ln_a

    return s_curve


def estimate_molecular_weight_uv(e2_e3: float) -> float:
    """
    Эмпирическая оценка среднемассовой молекулярной массы (Mw)
    по уравнению Перминовой (1998, 2000) для гуминовых веществ:
    Mw = 3450 - 390 * (E2/E3) [Да].
    """
    if e2_e3 is None or np.isnan(e2_e3) or e2_e3 <= 0:
        return np.nan
    mw = 3450.0 - 390.0 * float(e2_e3)
    return float(np.clip(mw, 400.0, 15000.0))


def calculate_uv_vis_indices(
    sample: UVVisSample,
    doc: Optional[float] = None,
    pathlength_cm: float = 1.0,
) -> Dict[str, Any]:
    """
    Расчет оптических индексов УФ-Вид:
    A254, A280, A365, E2/E3, E4/E6, S_275-295, S_350-400, S_R,
    минимум 2-й производной d2A_280, оценка Mw и SUVA254.
    """
    wl, absorbance = sample.wl, sample.absorbance

    def get_abs(target_wl: float) -> float:
        idx = np.argmin(np.abs(wl - target_wl))
        if abs(wl[idx] - target_wl) > 5.0:
            return np.nan
        return float(absorbance[idx])

    a250 = get_abs(250.0)
    a254 = get_abs(254.0)
    a280 = get_abs(280.0)
    a350 = get_abs(350.0)
    a365 = get_abs(365.0)
    a465 = get_abs(465.0)
    a665 = get_abs(665.0)

    e2_e3 = (a250 / a365) if (not np.isnan(a250) and not np.isnan(a365) and a365 > 1e-4) else np.nan
    e4_e6 = (a465 / a665) if (not np.isnan(a465) and not np.isnan(a665) and a665 > 1e-4) else np.nan

    def calc_slope(l_min: float, l_max: float) -> float:
        mask = (wl >= l_min) & (wl <= l_max)
        if np.sum(mask) < 3:
            return np.nan
        sub_wl = wl[mask]
        sub_a = absorbance[mask]
        pos_mask = sub_a > 1e-5
        if np.sum(pos_mask) < 3:
            return np.nan
        slope, _ = np.polyfit(sub_wl[pos_mask], np.log(sub_a[pos_mask]), 1)
        return float(-slope)

    s_275_295 = calc_slope(275.0, 295.0)
    s_350_400 = calc_slope(350.0, 400.0)
    sr = (s_275_295 / s_350_400) if (not np.isnan(s_275_295) and not np.isnan(s_350_400) and s_350_400 > 1e-6) else np.nan

    deriv_res = compute_uv_derivatives(wl, absorbance)
    d2_280 = deriv_res["lignin_d2_val"]
    mw_est = estimate_molecular_weight_uv(e2_e3)

    doc_val = doc if doc is not None else sample.doc
    suva254 = np.nan
    if doc_val is not None and doc_val > 0 and not np.isnan(a254) and a254 > 0:
        suva254 = float((a254 / (pathlength_cm * doc_val)) * 100.0)

    return {
        "Sample_ID": sample.sample_id,
        "A254": round(a254, 4) if not np.isnan(a254) else np.nan,
        "A280": round(a280, 4) if not np.isnan(a280) else np.nan,
        "A365": round(a365, 4) if not np.isnan(a365) else np.nan,
        "E2_E3": round(e2_e3, 2) if not np.isnan(e2_e3) else np.nan,
        "E4_E6": round(e4_e6, 2) if not np.isnan(e4_e6) else np.nan,
        "S_275_295": round(s_275_295, 4) if not np.isnan(s_275_295) else np.nan,
        "S_350_400": round(s_350_400, 4) if not np.isnan(s_350_400) else np.nan,
        "S_R": round(sr, 2) if not np.isnan(sr) else np.nan,
        "d2A_280": round(d2_280 * 1e4, 4) if not np.isnan(d2_280) else np.nan,
        "Mw_est": round(mw_est, 0) if not np.isnan(mw_est) else np.nan,
        "DOC": round(doc_val, 2) if (doc_val is not None and not np.isnan(doc_val)) else np.nan,
        "SUVA254": round(suva254, 2) if not np.isnan(suva254) else np.nan,
    }


def correct_inner_filter_effect(
    eem: Union[np.ndarray, EEMSample],
    ex: Union[np.ndarray, UVVisSample, None] = None,
    em: Optional[np.ndarray] = None,
    uv_wl: Optional[np.ndarray] = None,
    uv_a: Optional[np.ndarray] = None,
    pathlength_cm: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Коррекция эффекта внутреннего фильтра (Inner Filter Effect, IFE):
    F_corr(Ex, Em) = F_obs(Ex, Em) * 10^(0.5 * (A_ex + A_em) * d)
    Поддерживает передачу как массивов NumPy, так и объектов EEMSample и UVVisSample.
    """
    if isinstance(eem, EEMSample) and isinstance(ex, UVVisSample):
        d_val = float(em) if isinstance(em, (int, float)) else pathlength_cm
        return correct_inner_filter_effect(
            eem=eem.data,
            ex=eem.ex,
            em=eem.em,
            uv_wl=ex.wl,
            uv_a=ex.absorbance,
            pathlength_cm=d_val,
        )

    a_clean = np.maximum(0.0, uv_a)
    a_ex = np.interp(ex, uv_wl, a_clean, left=0.0, right=0.0)
    a_em = np.interp(em, uv_wl, a_clean, left=0.0, right=0.0)

    cf_matrix = 0.5 * (a_em[:, None] + a_ex[None, :]) * pathlength_cm
    multiplier = 10.0 ** cf_matrix
    eem_corr = eem * multiplier

    max_abs = max(float(np.max(a_ex)), float(np.max(a_em)))
    warning = None
    if max_abs > 1.5:
        warning = f"Оптическая плотность A_max = {max_abs:.2f} > 1.5. Рекомендуется предварительное разбавление образца (возможна нелинейность IFE)."

    return eem_corr, cf_matrix, warning


def link_uv_vis_to_eem(
    eem_samples: List[EEMSample],
    uv_vis_samples: List[UVVisSample],
    doc_map: Optional[Dict[str, float]] = None,
    apply_ife: bool = False,
    pathlength_cm: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Связывает спектры УФ-Вид с матрицами EEM.
    Заполняет sample.a254 и sample.doc, а при apply_ife=True корректирует флуоресценцию (IFE).
    """
    if not eem_samples or not uv_vis_samples:
        return []

    uv_dict = {u.sample_id: u for u in uv_vis_samples}
    uv_keys = list(uv_dict.keys())
    match_logs = []

    for eem in eem_samples:
        matched_uv = None
        if eem.sample_id in uv_dict:
            matched_uv = uv_dict[eem.sample_id]
        else:
            e_base = eem.sample_id.split("_")[0].strip().lower()
            for k in uv_keys:
                u_base = k.split("_")[0].strip().lower()
                if e_base == u_base or eem.sample_id.lower() in k.lower() or k.lower() in eem.sample_id.lower():
                    matched_uv = uv_dict[k]
                    break

        if matched_uv is not None:
            doc_val = None
            if doc_map and (eem.sample_id in doc_map or matched_uv.sample_id in doc_map):
                doc_val = doc_map.get(eem.sample_id, doc_map.get(matched_uv.sample_id))
            elif matched_uv.doc is not None:
                doc_val = matched_uv.doc
            elif eem.doc is not None:
                doc_val = eem.doc

            indices = calculate_uv_vis_indices(matched_uv, doc=doc_val, pathlength_cm=pathlength_cm)
            eem.a254 = indices["A254"]
            if doc_val is not None:
                eem.doc = doc_val

            ife_applied = False
            ife_warn = None
            if apply_ife:
                eem.data, _, ife_warn = correct_inner_filter_effect(
                    eem.data, eem.ex, eem.em, matched_uv.wl, matched_uv.absorbance, pathlength_cm=pathlength_cm
                )
                ife_applied = True

            match_logs.append({
                "EEM_Sample": eem.sample_id,
                "UV_Sample": matched_uv.sample_id,
                "A254": indices["A254"],
                "DOC": doc_val,
                "IFE_Applied": ife_applied,
                "IFE_Warning": ife_warn,
            })

    return match_logs


# ==============================================================================
# ЭТАЛОННАЯ БИБЛИОТЕКА ФЛУОРОФОРОВ И ИДЕНТИФИКАЦИЯ (OPENFLUOR MATCHING)
# ==============================================================================
OPENFLUOR_REFERENCE_LIBRARY = {
    "C1_Terrestrial_Fulvic": {
        "id": "OF_C1",
        "name_ru": "C1: Терригенный фульвоподобный флуорофор (Fulvic-like)",
        "name_en": "C1: Terrestrial Fulvic-like (Coble A/C, Murphy C1)",
        "category": "Humic / Terrestrial",
        "origin_ru": "Низкомолекулярное окисленное аллохтонное гуминовое вещество почв и речного стока.",
        "origin_en": "Allochthonous degraded humic material from watershed and soil organic matter.",
        "ex_peaks": [(315.0, 35.0, 1.0), (245.0, 20.0, 0.6)],
        "em_peaks": [(410.0, 45.0, 1.0)],
    },
    "C2_Industrial_Lignin_Humic": {
        "id": "OF_C2",
        "name_ru": "C2: Техногенный шлам-лигнин / Высокомолекулярный гуминовый (BPPM Marker)",
        "name_en": "C2: Sludge Lignin / High-MW Humic (BPPM Kraft Pulp Marker)",
        "category": "Anthropogenic / Lignin",
        "origin_ru": "Устойчивый полифенольный ароматический комплекс отходов переработки древесины (маркер шлам-лигнина).",
        "origin_en": "Recalcitrant polyphenolic kraft lignin derivative from wood processing waste.",
        "ex_peaks": [(365.0, 45.0, 1.0), (260.0, 25.0, 0.8)],
        "em_peaks": [(470.0, 55.0, 1.0)],
    },
    "C3_Protein_Tryptophan": {
        "id": "OF_C3",
        "name_ru": "C3: Белковоподобный (Триптофаноподобный, Peak T)",
        "name_en": "C3: Protein-like (Tryptophan-like, Peak T)",
        "category": "Protein / Autochthonous",
        "origin_ru": "Индикатор биогенной продуктивности гидробионтов и активного микробного метаболизма.",
        "origin_en": "Autochthonous biogenic productivity and microbial degradation indicator.",
        "ex_peaks": [(275.0, 22.0, 1.0)],
        "em_peaks": [(340.0, 30.0, 1.0)],
    },
    "C4_Protein_Tyrosine": {
        "id": "OF_C4",
        "name_ru": "C4: Тирозиноподобный белковый компонент (Peak B)",
        "name_en": "C4: Protein-like (Tyrosine-like, Peak B)",
        "category": "Protein / Labile",
        "origin_ru": "Свободные аминокислоты и белковые фракции свежего фотосинтеза микроводорослей.",
        "origin_en": "Free amino acids and labile proteins associated with algal growth.",
        "ex_peaks": [(275.0, 20.0, 1.0)],
        "em_peaks": [(305.0, 25.0, 1.0)],
    },
    "C5_Aquatic_Microbial_Humic": {
        "id": "OF_C5",
        "name_ru": "C5: Микробный гуминовый флуорофор (Marine/Aquatic Humic, Peak M)",
        "name_en": "C5: Aquatic Microbial Humic (Peak M)",
        "category": "Humic / Microbial",
        "origin_ru": "Продукты микробиологической переработки автохтонной органики в водной толще.",
        "origin_en": "Autochthonous humic materials produced in situ by aquatic bacterial reworking.",
        "ex_peaks": [(310.0, 30.0, 1.0), (250.0, 20.0, 0.7)],
        "em_peaks": [(385.0, 38.0, 1.0)],
    },
    "C6_Oxidized_Quinone": {
        "id": "OF_C6",
        "name_ru": "C6: Окисленный полифенольный / Хиноидный компонент (Oxidized Quinone)",
        "name_en": "C6: Oxidized Polyphenol / Quinone-like (Stedmon C6)",
        "category": "Photochemically Altered",
        "origin_ru": "Фотохимически окисленные гуминовые вещества и редокс-активные хиноидные фрагменты.",
        "origin_en": "Photochemically oxidized humic matter and redox-active quinoid fragments.",
        "ex_peaks": [(370.0, 40.0, 1.0), (270.0, 22.0, 0.6)],
        "em_peaks": [(495.0, 50.0, 1.0)],
    },
}


def match_parafac_to_openfluor(
    em_profiles: np.ndarray,
    ex_profiles: np.ndarray,
    em_ax: np.ndarray,
    ex_ax: np.ndarray,
    threshold_tcc: float = 0.85,
    lang: str = "ru",
) -> pd.DataFrame:
    """
    Сравнение извлеченных PARAFAC компонент с библиотекой флуорофоров OpenFluor
    по критерию конгруэнтности Такера (Tucker Congruence Coefficient, TCC).

    Parameters:
    -----------
    em_profiles : np.ndarray
        Матрица профилей эмиссии (len(em_ax) x n_components).
    ex_profiles : np.ndarray
        Матрица профилей возбуждения (len(ex_ax) x n_components).
    em_ax : np.ndarray
        Сетка длин волн эмиссии (нм).
    ex_ax : np.ndarray
        Сетка длин волн возбуждения (нм).
    threshold_tcc : float
        Минимальный порог средней конгруэнтности (по умолчанию 0.85).
    lang : str
        Язык вывода ('ru' или 'en').

    Returns:
    --------
    pd.DataFrame сопоставления с колонками:
      Component, Best_Match_ID, Name, Category, TCC_Mean, TCC_Em, TCC_Ex, Status, Origin
    """
    n_components = em_profiles.shape[1]

    def _eval_peaks(grid: np.ndarray, peaks: List[Tuple[float, float, float]]) -> np.ndarray:
        curve = np.zeros_like(grid, dtype=float)
        for mu, sigma, amp in peaks:
            curve += amp * np.exp(-0.5 * ((grid - mu) / max(1.0, sigma)) ** 2)
        norm = np.linalg.norm(curve)
        return (curve / norm) if norm > 1e-12 else curve

    def _calc_tcc(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    # Предварительно рассчитываем спектры библиотеки на активных сетках
    ref_spectra = {}
    for ref_key, ref_info in OPENFLUOR_REFERENCE_LIBRARY.items():
        ref_ex = _eval_peaks(ex_ax, ref_info["ex_peaks"])
        ref_em = _eval_peaks(em_ax, ref_info["em_peaks"])
        ref_spectra[ref_key] = (ref_ex, ref_em)

    match_rows = []

    for c_idx in range(n_components):
        v_em = em_profiles[:, c_idx]
        v_ex = ex_profiles[:, c_idx]

        best_key = None
        best_tcc_mean = -1.0
        best_tcc_em = 0.0
        best_tcc_ex = 0.0

        for ref_key, (ref_ex, ref_em) in ref_spectra.items():
            tcc_em = _calc_tcc(v_em, ref_em)
            tcc_ex = _calc_tcc(v_ex, ref_ex)
            tcc_mean = (tcc_em + tcc_ex) / 2.0

            if tcc_mean > best_tcc_mean:
                best_tcc_mean = tcc_mean
                best_tcc_em = tcc_em
                best_tcc_ex = tcc_ex
                best_key = ref_key

        if best_key is not None:
            ref_data = OPENFLUOR_REFERENCE_LIBRARY[best_key]
            name = ref_data[f"name_{lang}"] if f"name_{lang}" in ref_data else ref_data["name_en"]
            origin = ref_data[f"origin_{lang}"] if f"origin_{lang}" in ref_data else ref_data["origin_en"]

            if best_tcc_mean >= 0.95:
                status = "Идентичен (TCC ≥ 0.95)" if lang == "ru" else "Identical (TCC ≥ 0.95)"
            elif best_tcc_mean >= 0.90:
                status = "Высокое сходство (TCC ≥ 0.90)" if lang == "ru" else "High Similarity (TCC ≥ 0.90)"
            elif best_tcc_mean >= threshold_tcc:
                status = "Умеренное сходство" if lang == "ru" else "Moderate Match"
            else:
                status = "Низкое сходство (Уникальный флуорофор)" if lang == "ru" else "Low Similarity (Novel Fluorophore)"

            match_rows.append({
                "Component": f"C{c_idx + 1}",
                "Best_Match_ID": ref_data["id"],
                "Fluorophore_Name": name,
                "Category": ref_data["category"],
                "TCC_Mean": round(best_tcc_mean, 3),
                "TCC_Em": round(best_tcc_em, 3),
                "TCC_Ex": round(best_tcc_ex, 3),
                "Confidence_Status": status,
                "Ecological_Origin": origin,
            })

    return pd.DataFrame(match_rows)