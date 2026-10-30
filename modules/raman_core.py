"""
raman_core.py — Computational Engine for Raman Spectroscopy of Carbonaceous NOM & Lignin.

First-order carbon region (fit window 1000–1800 cm^-1), five-band model of
Sadezky et al. (Carbon 2005, 43, 1731–1742):
- D4 (~1200 cm^-1, Lorentzian): sp2–sp3 / polyene / ionic impurities
- D1 (~1350 cm^-1, Lorentzian): disordered graphitic lattice, A1g breathing mode
- D3 (~1500 cm^-1, Gaussian):   amorphous carbon (organic molecules, fragments)
- G  (~1580 cm^-1, Lorentzian): ideal graphitic lattice, E2g stretching
- D2 (~1620 cm^-1, Lorentzian): disordered graphitic lattice, surface layers (D')

Derived quantities: I_D/I_G (D1/G heights), A_D/A_G (D1/G areas), in-plane crystallite size L_a:
- Tuinstra–Koenig regime (L_a ≳ 2 nm), Cançado et al., Appl. Phys. Lett. 2006, 88, 163106:
  L_a(nm) = 2.4·10^-10 · λ_l^4 · (A_D/A_G)^-1, λ_l = laser wavelength in nm (integrated intensities).
- Amorphous regime (L_a < 2 nm), Ferrari & Robertson, Phys. Rev. B 2000, 61, 14095:
  I_D/I_G = C'(λ)·L_a², C'(514.5 nm) ≈ 0.0055 Å^-2.
I_D/I_G is non-monotonic along the amorphisation trajectory, so the regime cannot be fixed from the
ratio alone; both estimates are reported (la_tk_nm, la_fr_nm) and la_crystallite_nm follows the rule
documented in RamanSpectrum.estimate_crystallite_size.

The fit includes a linear residual background (a + b·x) jointly with the bands: Lorentzian wings
of broad D/G bands reach the window edges, so any pre-subtracted baseline (AsLS, polynomial, linear)
removes part of them; without the joint term the fit compensates by narrowing G and inflating D2/D3,
which biases I_D/I_G (verified on synthetic spectra with known band parameters).

A failed or unconverged fit sets fit_success=False and leaves every derived value NaN.
"""
from dataclasses import dataclass, field
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    from ftir_core import baseline_als
except ImportError:  # импорт как modules.raman_core
    from .ftir_core import baseline_als


FIT_WINDOW_CM1 = (1000.0, 1800.0)
DEFAULT_LASER_NM = 532.0
CANCADO_COEFF_NM3 = 2.4e-10          # nm^-3, Cançado et al. 2006
FR_C_PRIME_514_A2 = 0.0055           # Å^-2 at 514.5 nm, Ferrari & Robertson 2000
TK_MIN_LA_NM = 2.0                   # lower validity limit of the Tuinstra–Koenig relation
RAMAN_FIT_MIN_R2 = 0.95
# Relative standard error of I_D/I_G above which the five-band fit is reported as ill-conditioned
# (broad, merged bands let D3/D4 trade intensity with D1/G at almost no cost in χ²)
RAMAN_RATIO_MAX_REL_SE = 0.20
# Heuristic order scale: G-band FWHM of crystalline graphite (~15 cm^-1) … highly disordered carbon
G_FWHM_ORDERED = 15.0
G_FWHM_DISORDERED = 150.0

# (name, shape, centre init, centre bounds, FWHM init, FWHM bounds) — non-overlapping centre windows
SADEZKY_BANDS: List[Tuple[str, str, float, Tuple[float, float], float, Tuple[float, float]]] = [
    ("D4", "lorentzian", 1200.0, (1150.0, 1260.0), 150.0, (40.0, 300.0)),
    ("D1", "lorentzian", 1350.0, (1300.0, 1390.0), 120.0, (30.0, 300.0)),
    ("D3", "gaussian", 1500.0, (1450.0, 1545.0), 120.0, (30.0, 250.0)),
    ("G", "lorentzian", 1585.0, (1550.0, 1605.0), 60.0, (12.0, 150.0)),
    ("D2", "lorentzian", 1620.0, (1605.0, 1640.0), 40.0, (10.0, 80.0)),
]
_COMPONENT_KEYS = {"D4": "D4_band", "D1": "D1_band", "D3": "D3_amorphous", "G": "G_band", "D2": "D2_band"}


def lorentzian_band(x: np.ndarray, height: float, center: float, fwhm: float) -> np.ndarray:
    """Lorentzian profile parameterised by height and FWHM; area = π·height·FWHM/2."""
    return height / (1.0 + 4.0 * ((x - center) / max(fwhm, 1e-9)) ** 2)


def gaussian_band(x: np.ndarray, height: float, center: float, fwhm: float) -> np.ndarray:
    """Gaussian profile parameterised by height and FWHM; area = height·σ·√(2π), σ = FWHM/2.3548."""
    sigma = max(fwhm, 1e-9) / 2.354820045
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def band_area(shape: str, height: float, fwhm: float) -> float:
    if shape == "lorentzian":
        return float(np.pi * height * fwhm / 2.0)
    return float(height * (fwhm / 2.354820045) * np.sqrt(2.0 * np.pi))


def sadezky_model(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of the five SADEZKY_BANDS; params = [h, x0, fwhm] × 5 in SADEZKY_BANDS order."""
    y = np.zeros_like(x, dtype=float)
    for k, band in enumerate(SADEZKY_BANDS):
        h, c, w = params[3 * k:3 * k + 3]
        y += lorentzian_band(x, h, c, w) if band[1] == "lorentzian" else gaussian_band(x, h, c, w)
    return y


def _active_set_covariance(
    x: np.ndarray, y: np.ndarray, popt: np.ndarray, lower: List[float], upper: List[float]
) -> np.ndarray:
    """
    Parameter covariance s²·(JᵀJ)⁻¹ over the active parameters only. Bands fitted to zero height and
    parameters sitting on a bound are held fixed: otherwise their degenerate Jacobian columns (a
    zero-height band has no centre/width sensitivity) inflate the errors of D1 and G several-fold.
    """
    p = np.asarray(popt, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    active = np.abs(p - lo) > 1e-3 * (hi - lo)
    active &= np.abs(p - hi) > 1e-3 * (hi - lo)
    y_scale = max(float(np.max(np.abs(y))), 1e-12)
    for k in range(len(SADEZKY_BANDS)):
        if p[3 * k] <= 1e-3 * y_scale:
            active[3 * k:3 * k + 3] = False
    idx = np.flatnonzero(active)
    dof = len(x) - len(idx)
    if len(idx) == 0 or dof <= 0:
        return np.full((len(p), len(p)), np.nan)
    cov = np.zeros((len(p), len(p)))  # fixed parameters contribute no variance
    jac = np.empty((len(x), len(idx)))
    for col, i in enumerate(idx):
        step = 1e-6 * max(1.0, abs(p[i]))
        p_hi, p_lo = p.copy(), p.copy()
        p_hi[i] += step
        p_lo[i] -= step
        jac[:, col] = (_sadezky_with_background(x, *p_hi) - _sadezky_with_background(x, *p_lo)) / (2.0 * step)
    s2 = float(np.sum((y - _sadezky_with_background(x, *p)) ** 2)) / dof
    cov[np.ix_(idx, idx)] = s2 * np.linalg.pinv(jac.T @ jac)
    return cov


def _ratio_rel_se(popt: np.ndarray, pcov: np.ndarray) -> Tuple[float, float]:
    """
    Relative standard errors of I_D1/I_G and A_D1/A_G from the parameter covariance (delta method on
    the log-ratio; band area ∝ height·FWHM for both profile shapes). NaN when the covariance is undefined.
    """
    names = [band[0] for band in SADEZKY_BANDS]
    i_hd, i_hg = 3 * names.index("D1"), 3 * names.index("G")
    p = np.asarray(popt, dtype=float)
    cov = np.asarray(pcov, dtype=float)
    if cov.shape != (len(p), len(p)) or not np.all(np.isfinite(cov)) or p[i_hd] <= 0 or p[i_hg] <= 0:
        return float("nan"), float("nan")

    def _rel_se(grad: Dict[int, float]) -> float:
        g = np.zeros(len(p))
        for idx, val in grad.items():
            g[idx] = val
        return float(np.sqrt(max(float(g @ cov @ g), 0.0)))

    height_se = _rel_se({i_hd: 1.0 / p[i_hd], i_hg: -1.0 / p[i_hg]})
    area_se = _rel_se({i_hd: 1.0 / p[i_hd], i_hd + 2: 1.0 / p[i_hd + 2],
                       i_hg: -1.0 / p[i_hg], i_hg + 2: -1.0 / p[i_hg + 2]})
    return height_se, area_se


def _sadezky_with_background(x: np.ndarray, *params: float) -> np.ndarray:
    """sadezky_model + linear residual background; params = 15 band params + [a, b], b per 400 cm^-1 from 1400."""
    return sadezky_model(x, *params[:-2]) + params[-2] + params[-1] * (x - 1400.0) / 400.0


def fr_c_prime(laser_wavelength_nm: float) -> float:
    """
    C'(λ) in Å^-2 for I_D/I_G = C'·L_a² (Ferrari & Robertson 2000). C'(514.5 nm) = 0.0055 Å^-2 is
    the published value; other wavelengths use the λ⁴ scaling implied by continuity with the
    Tuinstra–Koenig relation at L_a = 2 nm (Cançado et al. 2006) — an approximation.
    """
    return FR_C_PRIME_514_A2 * (float(laser_wavelength_nm) / 514.5) ** 4


def raman_asls_baseline(wn: np.ndarray, y: np.ndarray, lam: float = 1e7, p: float = 0.001) -> np.ndarray:
    """
    AsLS baseline (Eilers & Boelens 2005; ftir_core.baseline_als). `lam` is stated for a 2 cm^-1 grid
    and rescaled by (2/Δν)^4 so that the baseline stiffness does not depend on the sampling step.
    """
    if len(y) < 4:
        return np.zeros_like(y)
    step = float(np.median(np.abs(np.diff(wn)))) if len(wn) > 1 else 2.0
    lam_eff = lam * (2.0 / max(step, 1e-6)) ** 4
    return baseline_als(y, lam=lam_eff, p=p, niter=15)


def raman_poly_baseline(wn: np.ndarray, y: np.ndarray, order: int = 3, niter: int = 100) -> np.ndarray:
    """Iterative modified polynomial baseline (Lieber & Mahadevan-Jansen, Appl. Spectrosc. 2003, 57, 1363)."""
    x = (wn - wn.mean()) / max(np.ptp(wn), 1e-9)
    work = y.astype(float).copy()
    base = work
    for _ in range(niter):
        coef = np.polyfit(x, work, order)
        base = np.polyval(coef, x)
        new_work = np.minimum(work, base)
        if np.allclose(new_work, work, rtol=0, atol=1e-9 * max(1.0, float(np.max(np.abs(y))))):
            break
        work = new_work
    return base


@dataclass
class RamanSpectrum:
    """Stores preprocessed and deconvoluted Raman spectrum for carbonaceous organic matter."""
    sample_id: str
    wavenumbers: np.ndarray      # cm^-1
    intensities: np.ndarray      # raw intensity counts
    sample_type: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    laser_wavelength_nm: float = DEFAULT_LASER_NM

    # Preprocessed (on analysis_wavenumbers, the 1000–1800 cm^-1 fit window)
    corrected_intensities: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None
    analysis_wavenumbers: Optional[np.ndarray] = None

    # Fit status — derived values are NaN unless fit_success is True
    fit_success: bool = False
    fit_ok: bool = False             # fit_success and R² ≥ RAMAN_FIT_MIN_R2 and no band at a bound
    fit_r2: float = float("nan")
    fit_message: str = "not processed"
    fit_warnings: List[str] = field(default_factory=list)
    band_params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Deconvolution results (D1 = "D", G)
    d_pos: float = float("nan")
    g_pos: float = float("nan")
    i_d: float = float("nan")             # Peak height of D1 band
    i_g: float = float("nan")             # Peak height of G band
    id_ig_ratio: float = float("nan")     # I_D1 / I_G height ratio
    ad_ag_ratio: float = float("nan")     # Area_D1 / Area_G integrated area ratio
    id_ig_rel_se: float = float("nan")    # relative standard error of I_D/I_G from the fit covariance
    ad_ag_rel_se: float = float("nan")    # relative standard error of A_D/A_G
    la_crystallite_nm: float = float("nan")  # In-plane crystallite size L_a (nm), see la_method
    la_tk_nm: float = float("nan")        # Cançado 2006 (Tuinstra–Koenig regime)
    la_fr_nm: float = float("nan")        # Ferrari & Robertson 2000 (amorphous regime)
    la_method: str = ""
    fwhm_d: float = float("nan")          # FWHM of D1 band (cm^-1)
    fwhm_g: float = float("nan")          # FWHM of G band (cm^-1)
    carbon_order_index: float = float("nan")  # heuristic 0 (disordered) … 1 (graphite), from FWHM(G)

    # Deconvoluted component curves (D1, D2, D3, D4, G, Cumulative_Fit) on analysis_wavenumbers
    components: Dict[str, np.ndarray] = field(default_factory=dict)

    def _reset_results(self, message: str) -> None:
        self.fit_success = False
        self.fit_ok = False
        self.fit_r2 = float("nan")
        self.fit_message = message
        self.fit_warnings = [message] if message else []
        self.band_params = {}
        self.components = {}
        for attr in ("d_pos", "g_pos", "i_d", "i_g", "id_ig_ratio", "ad_ag_ratio", "id_ig_rel_se", "ad_ag_rel_se",
                     "la_crystallite_nm", "la_tk_nm", "la_fr_nm", "fwhm_d", "fwhm_g", "carbon_order_index"):
            setattr(self, attr, float("nan"))
        self.la_method = ""

    def process(
        self,
        baseline_order: int = 3,
        baseline_method: str = "asls",
        asls_lam: float = 1e7,
        asls_p: float = 0.001,
        laser_wavelength_nm: Optional[float] = None,
    ) -> "RamanSpectrum":
        """
        Baseline correction in the 1000–1800 cm^-1 window and five-band deconvolution.

        baseline_method: 'asls' (default; asymmetric least squares, see raman_asls_baseline),
        'poly' (iterative modified polynomial of order `baseline_order`) or 'linear'
        (straight line between the window edges — legacy behaviour).
        """
        if laser_wavelength_nm is not None:
            self.laser_wavelength_nm = float(laser_wavelength_nm)
        wn = np.asarray(self.wavenumbers, dtype=float)
        ints = np.asarray(self.intensities, dtype=float)

        # Sort by wavenumber ascending (intensities follow the same order)
        sort_idx = np.argsort(wn, kind="stable")
        wn = wn[sort_idx]
        ints = ints[sort_idx]
        self.wavenumbers = wn
        self.intensities = ints

        mask = (wn >= FIT_WINDOW_CM1[0]) & (wn <= FIT_WINDOW_CM1[1])
        wn_sub = wn[mask]
        ints_sub = ints[mask]
        self.analysis_wavenumbers = wn_sub
        if len(wn_sub) < 30:
            self.baseline = np.zeros_like(ints_sub)
            self.corrected_intensities = np.zeros_like(ints_sub)
            self._reset_results(
                f"Only {len(wn_sub)} points in {FIT_WINDOW_CM1[0]:.0f}–{FIT_WINDOW_CM1[1]:.0f} cm^-1 (≥ 30 required)"
            )
            return self

        if baseline_method == "linear":
            n_edge = min(5, len(wn_sub) // 4)
            left_val, right_val = float(np.median(ints_sub[:n_edge])), float(np.median(ints_sub[-n_edge:]))
            left_wn, right_wn = float(np.mean(wn_sub[:n_edge])), float(np.mean(wn_sub[-n_edge:]))
            slope = (right_val - left_val) / (right_wn - left_wn) if abs(right_wn - left_wn) > 1e-9 else 0.0
            base = left_val + slope * (wn_sub - left_wn)
        elif baseline_method == "poly":
            base = raman_poly_baseline(wn_sub, ints_sub, order=int(baseline_order))
        else:
            base = raman_asls_baseline(wn_sub, ints_sub, lam=asls_lam, p=asls_p)

        # Not clipped at 0: the joint residual background term absorbs a slightly over-/under-subtracted baseline
        corr_ints = ints_sub - base
        max_c = float(np.max(corr_ints)) if len(corr_ints) else 0.0
        self.baseline = base
        if not np.isfinite(max_c) or max_c <= 0.0:
            self.corrected_intensities = np.zeros_like(corr_ints)
            self._reset_results("No signal above the baseline in the carbon region")
            return self
        # Normalize to max intensity = 100.0 for consistent plotting
        norm_ints = (corr_ints / max_c) * 100.0
        self.corrected_intensities = norm_ints

        self._deconvolute_bands(wn_sub, norm_ints)
        return self

    def _deconvolute_bands(self, wn: np.ndarray, ints: np.ndarray) -> None:
        """
        Five-band Sadezky fit (4 Lorentzians + Gaussian D3) plus a linear residual background.
        On failure all results are NaN.
        """
        p0: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        y_max = float(np.max(ints))
        for name, _shape, c0, (c_lo, c_hi), w0, (w_lo, w_hi) in SADEZKY_BANDS:
            win = (wn >= c_lo) & (wn <= c_hi)
            local = float(np.max(ints[win])) if np.any(win) else 0.1 * y_max
            if name in ("D1", "G"):
                h0 = max(local, 1.0)
            else:
                h0 = max(0.2 * local, 1.0)
            p0.extend([h0, c0, w0])
            lower.extend([0.0, c_lo, w_lo])
            upper.extend([1.5 * y_max, c_hi, w_hi])
        # Linear residual background a + b·(x − 1400)/400, bounded to ±y_max
        p0.extend([0.0, 0.0])
        lower.extend([-y_max, -y_max])
        upper.extend([y_max, y_max])
        p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0, lower, upper)]

        try:
            popt, _ = curve_fit(_sadezky_with_background, wn, ints, p0=p0, bounds=(lower, upper), maxfev=20000)
            if not np.all(np.isfinite(popt)):
                raise RuntimeError("non-finite parameters")
        except Exception as exc:  # no fallback values: the fit failed
            self._reset_results(f"Five-band fit did not converge: {exc}")
            return

        fitted = _sadezky_with_background(wn, *popt)
        ss_res = float(np.sum((ints - fitted) ** 2))
        ss_tot = float(np.sum((ints - np.mean(ints)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        warnings: List[str] = []
        self.band_params = {}
        self.components = {}
        for k, (name, shape, _c0, (c_lo, c_hi), _w0, (w_lo, w_hi)) in enumerate(SADEZKY_BANDS):
            h, c, w = (float(v) for v in popt[3 * k:3 * k + 3])
            at_bound = bool(
                min(abs(c - c_lo), abs(c - c_hi)) <= 1e-3 * (c_hi - c_lo)
                or min(abs(w - w_lo), abs(w - w_hi)) <= 1e-3 * (w_hi - w_lo)
            )
            if at_bound and name in ("D1", "G"):
                warnings.append(f"{name}: centre or FWHM at its bound — band not identifiable")
            self.band_params[name] = {
                "center": c, "fwhm": w, "height": h, "area": band_area(shape, h, w), "at_bound": at_bound,
            }
            curve = lorentzian_band(wn, h, c, w) if shape == "lorentzian" else gaussian_band(wn, h, c, w)
            self.components[_COMPONENT_KEYS[name]] = curve
        self.components["Residual_Background"] = popt[-2] + popt[-1] * (wn - 1400.0) / 400.0
        self.components["Cumulative_Fit"] = fitted  # bands + residual background (compare with corrected_intensities)

        if not np.isfinite(r2) or r2 < RAMAN_FIT_MIN_R2:
            warnings.append(f"Low fit quality: R² = {r2:.3f} < {RAMAN_FIT_MIN_R2}")

        d1, g = self.band_params["D1"], self.band_params["G"]
        self.d_pos = d1["center"]
        self.g_pos = g["center"]
        self.i_d = d1["height"]
        self.i_g = g["height"]
        self.fwhm_d = d1["fwhm"]
        self.fwhm_g = g["fwhm"]
        self.id_ig_ratio = float(d1["height"] / g["height"]) if g["height"] > 1e-6 else float("nan")
        self.ad_ag_ratio = float(d1["area"] / g["area"]) if g["area"] > 1e-6 else float("nan")
        self.id_ig_rel_se, self.ad_ag_rel_se = _ratio_rel_se(popt, _active_set_covariance(wn, ints, popt, lower, upper))
        ill_conditioned = not (np.isfinite(self.id_ig_rel_se) and self.id_ig_rel_se <= RAMAN_RATIO_MAX_REL_SE)
        if ill_conditioned:
            spread = f"±{100.0 * self.id_ig_rel_se:.0f} %" if np.isfinite(self.id_ig_rel_se) else "undefined"
            warnings.append(
                f"I_D/I_G poorly constrained by the five-band fit ({spread}): broad bands let D3/D4 "
                "trade intensity with D1/G"
            )

        self.fit_success = True
        self.fit_r2 = r2
        self.fit_warnings = warnings
        self.fit_ok = bool(np.isfinite(r2) and r2 >= RAMAN_FIT_MIN_R2 and not (d1["at_bound"] or g["at_bound"])
                           and not ill_conditioned)
        self.fit_message = "" if self.fit_ok else "; ".join(warnings)

        # Heuristic order index: linear rescaling of FWHM(G), monotonic in structural order
        # (FWHM(G) narrows continuously from amorphous carbon to graphite; I_D/I_G does not —
        # Ferrari & Robertson 2000). Not a literature index; graphite → ~1, disordered C → 0.
        self.carbon_order_index = float(np.clip(
            (G_FWHM_DISORDERED - self.fwhm_g) / (G_FWHM_DISORDERED - G_FWHM_ORDERED), 0.0, 1.0
        ))
        self.estimate_crystallite_size()

    def estimate_crystallite_size(self, laser_wavelength_nm: Optional[float] = None) -> float:
        """
        L_a from the fitted ratios for laser wavelength λ (nm):
        - la_tk_nm = 2.4·10^-10·λ⁴ / (A_D/A_G)   (Cançado et al. 2006, valid for L_a ≳ 2 nm)
        - la_fr_nm = sqrt((I_D/I_G) / C'(λ)) / 10 (Ferrari & Robertson 2000, L_a < 2 nm; heights)
        la_crystallite_nm = la_tk_nm when la_tk_nm ≥ 2 nm; otherwise the TK relation is outside its
        validity range and la_fr_nm is used (la_method states which). NaN if the fit failed.
        """
        if laser_wavelength_nm is not None:
            self.laser_wavelength_nm = float(laser_wavelength_nm)
        lam = float(self.laser_wavelength_nm)
        self.la_tk_nm = self.la_fr_nm = self.la_crystallite_nm = float("nan")
        self.la_method = ""
        if not self.fit_success:
            return self.la_crystallite_nm
        if np.isfinite(self.ad_ag_ratio) and self.ad_ag_ratio > 0:
            self.la_tk_nm = float(CANCADO_COEFF_NM3 * lam ** 4 / self.ad_ag_ratio)
        if np.isfinite(self.id_ig_ratio) and self.id_ig_ratio > 0:
            self.la_fr_nm = float(np.sqrt(self.id_ig_ratio / fr_c_prime(lam)) / 10.0)
        if np.isfinite(self.la_tk_nm) and self.la_tk_nm >= TK_MIN_LA_NM:
            self.la_crystallite_nm = self.la_tk_nm
            self.la_method = f"Tuinstra-Koenig / Cançado 2006 ({lam:g} nm)"
        elif np.isfinite(self.la_fr_nm):
            self.la_crystallite_nm = self.la_fr_nm
            self.la_method = f"Ferrari-Robertson 2000, amorphous regime (TK < 2 nm; {lam:g} nm)"
        return self.la_crystallite_nm


def parse_raman_spectrum(
    content: Union[str, bytes, pd.DataFrame],
    sample_id: str = "Raman_Sample",
    sample_type: str = "Unknown",
    laser_wavelength_nm: float = DEFAULT_LASER_NM,
) -> RamanSpectrum:
    """
    Parses two-column (Raman shift cm^-1, Intensity) text, CSV, or DataFrame into a RamanSpectrum.
    A single-column file is rejected (ValueError): no Raman-shift axis is fabricated.
    """
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"'{sample_id}': empty Raman file")
        first_line = lines[0]
        if "\t" in first_line:
            sep = "\t"
        elif ";" in first_line:
            sep = ";"
        elif "," in first_line:
            sep = ","
        else:
            sep = r"\s+"
        first_tokens = [t for t in re.split(sep if sep != r"\s+" else r"\s+", first_line.strip()) if t]
        try:
            [float(t) for t in first_tokens]
            header = None
        except ValueError:
            header = 0
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, header=header, engine="python")

    # Clean columns
    df = df.dropna(how="all")
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(
            f"'{sample_id}': a Raman spectrum needs two columns (Raman shift in cm^-1, intensity); "
            "a single-column file has no wavenumber axis"
        )
    wn = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(dtype=float)
    ints = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(dtype=float)

    valid = ~(np.isnan(wn) | np.isnan(ints))
    wn = wn[valid]
    ints = ints[valid]
    if len(wn) < 2:
        raise ValueError(f"'{sample_id}': no numeric (Raman shift, intensity) pairs found")

    spec = RamanSpectrum(
        sample_id=sample_id,
        wavenumbers=wn,
        intensities=ints,
        sample_type=sample_type,
        laser_wavelength_nm=float(laser_wavelength_nm),
    )
    spec.process()
    return spec


def generate_synthetic_raman_dataset() -> Dict[str, RamanSpectrum]:
    """
    Generates synthetic Raman spectra (five-band Sadezky line shapes + sloped background + noise)
    for carbonaceous organic matter:
    1. BPPM Kraft Sludge Lignin Hydrochar / Slime residue (condensed polyaromatic char)
    2. Pristine Lake Baikal Natural Humic Acid (aliphatic-rich, broad amorphous bands)
    3. Pyrogenic Black Carbon / Wildfire Soot (narrower G band, high I_D/I_G)
    Band parameters are illustrative, not measured values.
    """
    rng = np.random.RandomState(42)
    wn = np.linspace(1000.0, 1800.0, 401)  # 2 cm^-1 resolution

    def _sim_carbon_spec(bands: Dict[str, Tuple[float, float, float]], noise_lvl: float = 1.5) -> np.ndarray:
        curve = np.zeros_like(wn)
        for name, shape, *_ in SADEZKY_BANDS:
            h, c, w = bands[name]
            curve += lorentzian_band(wn, h, c, w) if shape == "lorentzian" else gaussian_band(wn, h, c, w)
        noise = rng.normal(0, noise_lvl, len(wn))
        # Add slight fluorescence background baseline slope
        background = 15.0 + 0.02 * (wn - 1000.0)
        return curve + background + noise

    spectra: Dict[str, RamanSpectrum] = {}

    # 1. Kraft Sludge Lignin Hydrochar Samples (BPPM Solzan pond core): I_D/I_G ~ 0.9
    for i in range(1, 4):
        s_id = f"BPPM_Sludge_Hydrochar_0{i}"
        ints = _sim_carbon_spec({
            "D4": (10.0, 1200.0, 160.0),
            "D1": (78.0 + rng.normal(0, 2.0), 1348.0, 110.0),
            "D3": (20.0, 1505.0, 130.0),
            "G": (84.0 + rng.normal(0, 2.5), 1590.0, 70.0),
            "D2": (12.0, 1618.0, 35.0),
        }, noise_lvl=1.8)
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Kraft_Sludge_Char")
        spec.process()
        spectra[s_id] = spec

    # 2. Pristine Natural Humic Acid (Lake Baikal deep sediment NOM): broad bands, strong D3, I_D/I_G ~ 0.7
    for i in range(1, 4):
        s_id = f"Baikal_Natural_Humic_0{i}"
        ints = _sim_carbon_spec({
            "D4": (14.0, 1210.0, 180.0),
            "D1": (52.0 + rng.normal(0, 2.0), 1355.0, 140.0),
            "D3": (38.0, 1500.0, 160.0),
            "G": (75.0 + rng.normal(0, 2.0), 1595.0, 90.0),
            "D2": (8.0, 1620.0, 40.0),
        }, noise_lvl=1.5)
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Pristine_NOM")
        spec.process()
        spectra[s_id] = spec

    # 3. Pyrogenic Black Carbon / Forest Fire Soot (Selenga River Basin): I_D/I_G ~ 1.25, narrower G
    for i in range(1, 3):
        s_id = f"Pyrogenic_BlackCarbon_0{i}"
        ints = _sim_carbon_spec({
            "D4": (9.0, 1195.0, 140.0),
            "D1": (95.0 + rng.normal(0, 2.0), 1345.0, 90.0),
            "D3": (14.0, 1510.0, 110.0),
            "G": (76.0 + rng.normal(0, 2.0), 1585.0, 55.0),
            "D2": (16.0, 1615.0, 30.0),
        }, noise_lvl=1.2)
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Black_Carbon")
        spec.process()
        spectra[s_id] = spec

    return spectra


RAMAN_QC_COLUMNS = ["fit_success", "fit_r2", "id_ig_rel_se", "la_method"]


def calculate_raman_descriptors(spectra: List[RamanSpectrum]) -> pd.DataFrame:
    """
    Extracts tabular chemometric descriptor matrix for ML integration.
    Failed fits give NaN descriptors (never fallback values). QC columns (RAMAN_QC_COLUMNS)
    describe fit status and the L_a method; drop them before numeric data fusion.
    """
    rows = []
    for s in spectra:
        rows.append({
            "sample_id": s.sample_id,
            "id_ig_ratio": s.id_ig_ratio,
            "ad_ag_ratio": s.ad_ag_ratio,
            "la_crystallite_nm": s.la_crystallite_nm,
            "la_tk_nm": s.la_tk_nm,
            "la_fr_nm": s.la_fr_nm,
            "d_band_pos": s.d_pos,
            "g_band_pos": s.g_pos,
            "fwhm_d": s.fwhm_d,
            "fwhm_g": s.fwhm_g,
            "carbon_order_index": s.carbon_order_index,
            "fit_success": bool(s.fit_success),
            "fit_r2": s.fit_r2,
            "id_ig_rel_se": s.id_ig_rel_se,
            "la_method": s.la_method,
        })
    return pd.DataFrame(rows).set_index("sample_id")
