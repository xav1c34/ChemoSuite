"""
eem_core.py — Математическое ядро анализа EEM-PARAFAC.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import griddata
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

tl.set_backend("numpy")


@dataclass
class EEMSample:
    sample_id: str
    ex: np.ndarray
    em: np.ndarray
    data: np.ndarray
    doc: Optional[float] = None
    a254: Optional[float] = None


def parse_eem_dataframe(
    df: pd.DataFrame, sample_id: str = "Sample"
) -> EEMSample:
    cleaned = df.copy()
    if not np.issubdtype(cleaned.iloc[:, 0].dtype, np.number):
        cleaned = cleaned.set_index(cleaned.columns[0])
    em_vals = cleaned.index.astype(float).values
    ex_vals = cleaned.columns.astype(float).values
    matrix = cleaned.values.astype(float)

    if ex_vals[0] > em_vals[0] and np.mean(ex_vals) > np.mean(em_vals):
        matrix = matrix.T
        em_vals, ex_vals = ex_vals, em_vals

    return EEMSample(
        sample_id=sample_id, ex=ex_vals, em=em_vals, data=matrix
    )


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


def calculate_spectral_indices(sample: EEMSample) -> Dict[str, float]:
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
        "FI": fi,
        "HIX": hix,
        "SUVA254": suva if not np.isnan(suva) else None,
    }


def build_eem_tensor(
    samples: List[EEMSample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ref_ex, ref_em = samples[0].ex, samples[0].em
    tensor = np.zeros((len(samples), len(ref_em), len(ref_ex)), dtype=np.float64)
    names = []
    for i, s in enumerate(samples):
        tensor[i, :, :] = s.data
        names.append(s.sample_id)
    return tensor, ref_em, ref_ex, names


def compute_corcondia(
    tensor: np.ndarray, factors: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
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