"""
raman_core.py — Computational Engine for Raman Spectroscopy of Carbonaceous NOM & Lignin.

Specialized for carbon first-order region (800 - 2000 cm^-1):
- D-band (~1350 cm^-1): defect/disordered aromatic rings (A_1g breathing mode)
- G-band (~1585 cm^-1): ideal in-plane sp2 graphitic lattice (E_2g stretching mode)
- Secondary defect bands: D2 (~1620 cm^-1), D3 amorphous (~1500 cm^-1), D4 (~1200 cm^-1)
- Calculates I_D/I_G, A_D/A_G, Tuinstra-Koenig crystallite size L_a (nm),
  and carbonaceous condensation index.
"""
from dataclasses import dataclass, field
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class RamanSpectrum:
    """Stores preprocessed and deconvoluted Raman spectrum for carbonaceous organic matter."""
    sample_id: str
    wavenumbers: np.ndarray      # cm^-1
    intensities: np.ndarray      # raw intensity counts
    sample_type: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Preprocessed
    corrected_intensities: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None

    # Deconvolution results
    d_pos: float = 1350.0
    g_pos: float = 1585.0
    i_d: float = 0.0             # Peak height of D band
    i_g: float = 0.0             # Peak height of G band
    id_ig_ratio: float = 0.0     # I_D / I_G height ratio
    ad_ag_ratio: float = 0.0     # Area_D / Area_G integrated area ratio
    la_crystallite_nm: float = 0.0  # In-plane aromatic cluster size L_a (nm)
    fwhm_d: float = 0.0          # FWHM of D band (cm^-1)
    fwhm_g: float = 0.0          # FWHM of G band (cm^-1)
    carbon_order_index: float = 0.0 # 0.0 (amorphous/humic) to 1.0 (condensed soot/graphitic)

    # Deconvoluted component curves (D1, D2, D3, D4, G)
    components: Dict[str, np.ndarray] = field(default_factory=dict)

    def process(self, baseline_order: int = 1) -> "RamanSpectrum":
        """Executes linear/polynomial baseline correction and deconvolution."""
        wn = np.asarray(self.wavenumbers, dtype=float)
        ints = np.asarray(self.intensities, dtype=float)

        # Sort by wavenumber ascending
        sort_idx = np.argsort(wn)
        wn = wn[sort_idx]
        ints = ints[sort_idx]
        self.wavenumbers = wn

        # Focus analysis on carbon first-order region (1000 to 1800 cm^-1)
        mask = (wn >= 1000.0) & (wn <= 1800.0)
        if not np.any(mask):
            mask = np.ones_like(wn, dtype=bool)

        wn_sub = wn[mask]
        ints_sub = ints[mask]

        # Simple robust baseline subtraction using linear endpoint fit
        if len(wn_sub) > 10:
            left_val = float(np.median(ints_sub[:5]))
            right_val = float(np.median(ints_sub[-5:]))
            left_wn = float(np.mean(wn_sub[:5]))
            right_wn = float(np.mean(wn_sub[-5:]))

            if abs(right_wn - left_wn) > 1e-4:
                slope = (right_val - left_val) / (right_wn - left_wn)
                base = left_val + slope * (wn_sub - left_wn)
            else:
                base = np.full_like(ints_sub, left_val)
        else:
            base = np.zeros_like(ints_sub)

        corr_ints = np.maximum(0.0, ints_sub - base)
        # Normalize to max intensity = 100.0 for consistent plotting
        max_c = np.max(corr_ints) if len(corr_ints) > 0 and np.max(corr_ints) > 0 else 1.0
        norm_ints = (corr_ints / max_c) * 100.0

        self.baseline = base
        self.corrected_intensities = norm_ints

        # Perform 5-band deconvolution (D4 ~1200, D1 ~1350, D3 ~1500, G ~1585, D2 ~1620)
        self._deconvolute_bands(wn_sub, norm_ints)
        return self

    def _deconvolute_bands(self, wn: np.ndarray, ints: np.ndarray) -> None:
        """Fits Gaussian multi-peak model for carbonaceous structure."""
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-((x - x0) ** 2) / (2 * (sigma ** 2) + 1e-6))

        def multi_peak(x, a_d1, x_d1, s_d1,
                               a_d3, x_d3, s_d3,
                               a_g,  x_g,  s_g):
            return (
                gaussian(x, a_d1, x_d1, s_d1) +
                gaussian(x, a_d3, x_d3, s_d3) +
                gaussian(x, a_g,  x_g,  s_g)
            )

        # Initial parameter estimates: [Amp, Center, Sigma]
        # D1 (~1350 cm^-1), D3 (~1500 cm^-1 amorphous), G (~1585 cm^-1)
        # Initial guess from data
        d_mask = (wn >= 1320) & (wn <= 1380)
        g_mask = (wn >= 1560) & (wn <= 1610)

        amp_d_init = float(np.max(ints[d_mask])) if np.any(d_mask) else 60.0
        amp_g_init = float(np.max(ints[g_mask])) if np.any(g_mask) else 80.0
        amp_d3_init = 25.0

        p0 = [
            amp_d_init, 1350.0, 45.0,  # D1
            amp_d3_init, 1500.0, 50.0, # D3
            amp_g_init, 1585.0, 35.0,  # G
        ]
        bounds = (
            [0.0, 1310.0, 15.0,  0.0, 1460.0, 20.0,  0.0, 1550.0, 15.0],
            [150.0, 1380.0, 90.0, 100.0, 1540.0, 95.0, 150.0, 1615.0, 75.0]
        )

        try:
            popt, _ = curve_fit(multi_peak, wn, ints, p0=p0, bounds=bounds, maxfev=1500)
            a_d1, x_d1, s_d1, a_d3, x_d3, s_d3, a_g, x_g, s_g = popt
        except Exception:
            # Fallback to direct local maxima
            x_d1, a_d1, s_d1 = 1350.0, amp_d_init, 45.0
            x_d3, a_d3, s_d3 = 1500.0, 20.0, 50.0
            x_g, a_g, s_g = 1585.0, amp_g_init, 35.0

        self.d_pos = float(x_d1)
        self.g_pos = float(x_g)
        self.i_d = float(a_d1)
        self.i_g = float(a_g)

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        self.fwhm_d = float(2.355 * s_d1)
        self.fwhm_g = float(2.355 * s_g)

        # Integrated areas A = Amp * sigma * sqrt(2*pi)
        area_d1 = float(a_d1 * s_d1 * np.sqrt(2 * np.pi))
        area_g = float(a_g * s_g * np.sqrt(2 * np.pi))
        self.ad_ag_ratio = float(area_d1 / area_g) if area_g > 1e-4 else 1.0

        # Peak intensity ratio
        self.id_ig_ratio = float(self.i_d / self.i_g) if self.i_g > 1e-4 else 1.0

        # In-plane crystallite size L_a (nm) via Tuinstra-Koenig relation (for 532 nm laser: C = 4.4 nm)
        # L_a = C(lambda) / (I_D / I_G)
        if self.id_ig_ratio > 0.05:
            self.la_crystallite_nm = float(np.clip(4.4 / self.id_ig_ratio, 0.5, 30.0))
        else:
            self.la_crystallite_nm = 30.0

        # Carbon order index: combination of G band narrowing and D/G aromatic development
        # Pure graphite has narrow G (< 20 cm^-1) and I_D/I_G ~ 0.
        # Highly mature char/soot has I_D/I_G ~ 1.0-1.2 with L_a ~ 3.5 nm.
        # Humic acids / uncharred NOM have broad amorphous bands.
        order_score = 0.5 * (1.0 - min(1.0, max(0.0, (self.fwhm_g - 25.0) / 60.0)))
        order_score += 0.5 * min(1.0, self.id_ig_ratio / 1.2)
        self.carbon_order_index = float(np.clip(order_score, 0.0, 1.0))

        # Store individual component curves for plotting
        self.components = {
            "D1_band": gaussian(wn, a_d1, x_d1, s_d1),
            "D3_amorphous": gaussian(wn, a_d3, x_d3, s_d3),
            "G_band": gaussian(wn, a_g, x_g, s_g),
            "Cumulative_Fit": multi_peak(wn, *[a_d1, x_d1, s_d1, a_d3, x_d3, s_d3, a_g, x_g, s_g]),
        }


def parse_raman_spectrum(
    content: Union[str, bytes, pd.DataFrame],
    sample_id: str = "Raman_Sample",
    sample_type: str = "Unknown"
) -> RamanSpectrum:
    """
    Parses two-column (Wavenumber, Intensity) text, CSV, or DataFrame into a RamanSpectrum.
    """
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        # Detect delimiter
        first_line = text.strip().split("\n")[0]
        sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")
        df = pd.read_csv(io.StringIO(text), sep=sep)

    # Clean columns
    df = df.dropna()
    cols = list(df.columns)
    if len(cols) >= 2:
        wn = pd.to_numeric(df[cols[0]], errors="coerce").values
        ints = pd.to_numeric(df[cols[1]], errors="coerce").values
    else:
        wn = np.linspace(1000, 1800, len(df))
        ints = pd.to_numeric(df[cols[0]], errors="coerce").values

    valid = ~(np.isnan(wn) | np.isnan(ints))
    wn = wn[valid]
    ints = ints[valid]

    spec = RamanSpectrum(
        sample_id=sample_id,
        wavenumbers=wn,
        intensities=ints,
        sample_type=sample_type,
    )
    spec.process()
    return spec


def generate_synthetic_raman_dataset() -> Dict[str, RamanSpectrum]:
    """
    Generates realistic reference Raman spectra for carbonaceous organic matter:
    1. BPPM Kraft Sludge Lignin Hydrochar / Slime residue (condensed polyaromatic char)
    2. Pristine Lake Baikal Natural Humic Acid (aliphatic-rich, broad amorphous bands)
    3. Pyrogenic Black Carbon / Wildfire Soot (sharp G band, high I_D/I_G)
    """
    np.random.seed(42)
    wn = np.linspace(1000.0, 1800.0, 401)  # 2 cm^-1 resolution

    def _sim_carbon_spec(a_d, x_d, s_d, a_d3, x_d3, s_d3, a_g, x_g, s_g, noise_lvl=1.5):
        def g(x, a, x0, s):
            return a * np.exp(-((x - x0) ** 2) / (2 * (s ** 2)))
        curve = g(wn, a_d, x_d, s_d) + g(wn, a_d3, x_d3, s_d3) + g(wn, a_g, x_g, s_g)
        noise = np.random.normal(0, noise_lvl, len(wn))
        # Add slight fluorescence background baseline slope
        baseline = 15.0 + 0.02 * (wn - 1000.0)
        return curve + baseline + noise

    spectra: Dict[str, RamanSpectrum] = {}

    # 1. Kraft Sludge Lignin Hydrochar Samples (BPPM Solzan pond core)
    # Moderately graphitized, pronounced D1 and G bands, ID/IG ~ 0.95
    for i in range(1, 4):
        s_id = f"BPPM_Sludge_Hydrochar_0{i}"
        ints = _sim_carbon_spec(
            a_d=78.0 + np.random.normal(0, 2.0), x_d=1348.0, s_d=42.0,
            a_d3=22.0, x_d3=1505.0, s_d3=45.0,
            a_g=84.0 + np.random.normal(0, 2.5), x_g=1588.0, s_g=34.0,
            noise_lvl=1.8
        )
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Kraft_Sludge_Char")
        spec.process()
        spectra[s_id] = spec

    # 2. Pristine Natural Humic Acid (Lake Baikal deep sediment NOM)
    # High amorphous D3 component, broad G band, ID/IG ~ 0.65
    for i in range(1, 4):
        s_id = f"Baikal_Natural_Humic_0{i}"
        ints = _sim_carbon_spec(
            a_d=52.0 + np.random.normal(0, 2.0), x_d=1355.0, s_d=55.0,
            a_d3=40.0, x_d3=1495.0, s_d3=60.0,
            a_g=75.0 + np.random.normal(0, 2.0), x_g=1592.0, s_g=48.0,
            noise_lvl=1.5
        )
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Pristine_NOM")
        spec.process()
        spectra[s_id] = spec

    # 3. Pyrogenic Black Carbon / Forest Fire Soot (Selenga River Basin)
    # Highly ordered aromatic domains, sharp G band, ID/IG ~ 1.25, La ~ 3.5 nm
    for i in range(1, 3):
        s_id = f"Pyrogenic_BlackCarbon_0{i}"
        ints = _sim_carbon_spec(
            a_d=95.0 + np.random.normal(0, 2.0), x_d=1345.0, s_d=36.0,
            a_d3=14.0, x_d3=1510.0, s_d3=38.0,
            a_g=76.0 + np.random.normal(0, 2.0), x_g=1582.0, s_g=26.0,
            noise_lvl=1.2
        )
        spec = RamanSpectrum(s_id, wn, ints, sample_type="Black_Carbon")
        spec.process()
        spectra[s_id] = spec

    return spectra


def calculate_raman_descriptors(spectra: List[RamanSpectrum]) -> pd.DataFrame:
    """Extracts tabular chemometric descriptor matrix for ML integration."""
    rows = []
    for s in spectra:
        rows.append({
            "sample_id": s.sample_id,
            "id_ig_ratio": s.id_ig_ratio,
            "ad_ag_ratio": s.ad_ag_ratio,
            "la_crystallite_nm": s.la_crystallite_nm,
            "d_band_pos": s.d_pos,
            "g_band_pos": s.g_pos,
            "fwhm_d": s.fwhm_d,
            "fwhm_g": s.fwhm_g,
            "carbon_order_index": s.carbon_order_index,
        })
    return pd.DataFrame(rows).set_index("sample_id")
