"""
irms_core.py — Computational Engine for Isotope Ratio Mass Spectrometry (IRMS).

Analyzes stable isotope ratios (delta13C, delta15N, delta34S, delta2H, delta18O)
for geochemical provenance, tracing terrestrial vs. aquatic inputs, and differentiating
pulp mill kraft sludge-lignin from natural background organic matter.

Missing isotope values stay NaN (they are never replaced by "typical" values);
the provenance score is renormalised over the isotopes actually measured and flagged.

References
----------
* Dansgaard W. (1964) Tellus 16:436 — deuterium excess d = δ2H − 8·δ18O (water only).
* Paul D., Skrzypek G., Fórizs I. (2007) Rapid Commun. Mass Spectrom. 21:3006 —
  two-point normalisation of δ values to international scales.
* Brand W.A. et al. (2010) Pure Appl. Chem. 82:1719 (VPDB 13C/12C = 0.011180, Zhang & Li 1990);
  Junk & Svec (1958) / De Bièvre et al. (1996) (AIR 15N/14N = 0.0036765);
  Hagemann et al. (1970) (VSMOW 2H/1H = 155.76e-6); Baertschi (1976) (VSMOW 18O/16O = 2005.20e-6);
  Ding et al. (2001) GCA 65:2433 (VCDT 34S/32S = 0.0441626).
"""
from dataclasses import dataclass, field
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Canonical reference isotope baselines (‰ VPDB, AIR, VCDT, VSMOW)
STANDARDS = {
    "13C": "VPDB (Vienna Pee Dee Belemnite)",
    "15N": "AIR (Atmospheric Nitrogen)",
    "34S": "VCDT (Vienna Cañon Diablo Troilite)",
    "2H": "VSMOW (Vienna Standard Mean Ocean Water)",
    "18O": "VSMOW (Vienna Standard Mean Ocean Water)",
}

# Absolute isotope ratios of the international reference scales
REFERENCE_RATIOS: Dict[str, Dict[str, Any]] = {
    "13C": {"standard": "VPDB", "ratio_name": "13C/12C", "ratio": 0.011180,
            "source": "Zhang & Li (1990); Brand et al. (2010)"},
    "15N": {"standard": "AIR", "ratio_name": "15N/14N", "ratio": 0.0036765,
            "source": "Junk & Svec (1958); De Bièvre et al. (1996)"},
    "2H": {"standard": "VSMOW", "ratio_name": "2H/1H", "ratio": 155.76e-6,
           "source": "Hagemann et al. (1970)"},
    "18O": {"standard": "VSMOW", "ratio_name": "18O/16O", "ratio": 2005.20e-6,
            "source": "Baertschi (1976)"},
    "34S": {"standard": "VCDT", "ratio_name": "34S/32S", "ratio": 0.0441626,
            "source": "Ding et al. (2001)"},
}

# Geochemical endmember reference ranges (Typical for Siberian/Baikal watershed)
ENDMEMBERS = {
    "Kraft_Lignin_BPPM": {
        "d13C": (-27.8, -26.5),   # Coniferous boreal wood processed in kraft mill
        "d15N": (1.5, 4.2),
        "d34S": (4.5, 8.5),       # Heavy sulfate/sulfide cooking liquor signature
        "d2H": (-145.0, -125.0),
        "d18O": (14.0, 18.0),
        "color": "#e11d48",
    },
    "Baikal_Natural_NOM": {
        "d13C": (-31.5, -28.5),   # Deep cold lake ultra-oligotrophic NOM
        "d15N": (4.5, 8.0),       # Microbial/algal recycled nitrogen
        "d34S": (-2.0, 2.5),      # Terrestrial forest runoff background
        "d2H": (-160.0, -140.0),
        "d18O": (10.0, 13.5),
        "color": "#0ea5e9",
    },
    "Soil_Humics": {
        "d13C": (-28.5, -26.8),
        "d15N": (2.0, 5.5),
        "d34S": (0.0, 3.5),
        "d2H": (-135.0, -115.0),
        "d18O": (13.0, 16.5),
        "color": "#d97706",
    },
    "Aquatic_Phytoplankton": {
        "d13C": (-34.0, -30.0),
        "d15N": (6.0, 11.0),
        "d34S": (2.0, 6.0),
        "d2H": (-170.0, -145.0),
        "d18O": (8.5, 12.0),
        "color": "#10b981",
    },
}

# Sample-type tokens that identify a water sample (d-excess is defined only for water)
_WATER_TOKENS = {
    "water", "waters", "h2o", "precipitation", "precip", "rain", "snow", "snowmelt", "ice",
    "groundwater", "porewater", "meteoric", "lakewater", "riverwater", "seawater",
    "вода", "воды", "дождь", "снег", "лёд", "лед", "подземные", "поровая",
}


def _is_num(v: Any) -> bool:
    """True for a finite real number (None, NaN, inf and strings are not)."""
    try:
        return v is not None and not isinstance(v, bool) and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def is_water_sample_type(sample_type: Any) -> bool:
    """Keyword test on the sample type (e.g. 'Lake_Water', 'Snow', 'Вода')."""
    if sample_type is None:
        return False
    tokens = [t for t in re.split(r"[^a-zа-яё0-9]+", str(sample_type).lower()) if t]
    return any(t in _WATER_TOKENS or t.endswith("water") for t in tokens)


def _reference_ratio(standard: Union[str, float]) -> float:
    if isinstance(standard, str):
        key = standard.strip().upper()
        key = {"D": "2H", "H2": "2H", "DELTAD": "2H"}.get(key, key)
        if key not in REFERENCE_RATIOS:
            raise KeyError(f"Unknown isotope '{standard}'. Use one of {list(REFERENCE_RATIOS)} or a numeric ratio.")
        return float(REFERENCE_RATIOS[key]["ratio"])
    return float(standard)


def ratio_to_delta(ratio: Union[float, np.ndarray], standard: Union[str, float]) -> Union[float, np.ndarray]:
    """
    δ (‰) = (R_sample / R_standard − 1) · 1000.
    ``standard`` is an isotope key of REFERENCE_RATIOS ("13C", "15N", "2H", "18O", "34S")
    or a numeric reference ratio.
    """
    r_std = _reference_ratio(standard)
    r = np.asarray(ratio, dtype=float)
    out = (r / r_std - 1.0) * 1000.0
    return float(out) if out.ndim == 0 else out


def delta_to_ratio(delta: Union[float, np.ndarray], standard: Union[str, float]) -> Union[float, np.ndarray]:
    """R_sample = R_standard · (1 + δ/1000)."""
    r_std = _reference_ratio(standard)
    d = np.asarray(delta, dtype=float)
    out = r_std * (1.0 + d / 1000.0)
    return float(out) if out.ndim == 0 else out


def two_point_normalization(
    measured: Union[float, np.ndarray],
    ref1_measured: float,
    ref1_true: float,
    ref2_measured: float,
    ref2_true: float,
) -> Union[float, np.ndarray]:
    """
    Two-point linear normalisation of δ values to the international scale
    (Paul, Skrzypek & Fórizs 2007, RCM 21:3006):

        δ_true = δ1_true + (δ_meas − δ1_meas) · (δ2_true − δ1_true) / (δ2_meas − δ1_meas)

    Returns NaN when the two reference materials were measured identically
    (the stretch factor is undefined) or any reference value is missing.
    """
    m = np.asarray(measured, dtype=float)
    refs = (ref1_measured, ref1_true, ref2_measured, ref2_true)
    if not all(_is_num(v) for v in refs) or abs(float(ref2_measured) - float(ref1_measured)) < 1e-12:
        out = np.full(m.shape, np.nan)
    else:
        slope = (float(ref2_true) - float(ref1_true)) / (float(ref2_measured) - float(ref1_measured))
        out = float(ref1_true) + (m - float(ref1_measured)) * slope
    return float(out) if np.ndim(out) == 0 else out


@dataclass
class IRMSSample:
    """Stores multi-element stable isotope ratios for an analytical sample (NaN = not measured)."""
    sample_id: str
    d13C: float               # ‰ VPDB
    d15N: float               # ‰ AIR
    d34S: Optional[float] = None  # ‰ VCDT (essential for sulfate lignin)
    d2H: Optional[float] = None   # ‰ VSMOW
    d18O: Optional[float] = None  # ‰ VSMOW
    c_n_atomic: Optional[float] = None
    toc_pct: Optional[float] = None
    sample_type: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_water: Optional[bool] = None   # None -> decided from sample_type keywords

    # Calculated provenance metrics
    kraft_lignin_prob: float = 0.0
    deuterium_excess: float = float("nan")  # d-excess = d2H - 8 * d18O (water samples only)
    missing_isotopes: List[str] = field(default_factory=list)
    kraft_prob_basis: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def compute_metrics(self) -> "IRMSSample":
        """Calculates secondary geochemical and provenance indicators."""
        flags: List[str] = []
        notes: List[str] = []
        self.missing_isotopes = [
            name for name, v in (("d13C", self.d13C), ("d15N", self.d15N), ("d34S", self.d34S),
                                 ("d2H", self.d2H), ("d18O", self.d18O))
            if not _is_num(v)
        ]

        # 1. Deuterium excess (Dansgaard 1964) — defined for waters on the meteoric water line only
        water = self.is_water if self.is_water is not None else is_water_sample_type(self.sample_type)
        if _is_num(self.d2H) and _is_num(self.d18O):
            if water:
                self.deuterium_excess = float(self.d2H - 8.0 * self.d18O)
            else:
                self.deuterium_excess = float("nan")
                notes.append("d-excess not computed: defined only for water samples (Dansgaard 1964); "
                             "set is_water=True or a water sample type.")
        else:
            self.deuterium_excess = float("nan")

        # 2. Heuristic probability of Kraft Mill Pulp Sludge origin, renormalised over measured isotopes
        # Kraft pulp lignin from BPPM typically has heavy d34S (>4.0) and d13C around -27.0
        score = 0.0
        weights = 0.0
        basis: List[str] = []

        if _is_num(self.d34S):
            weights += 0.50
            basis.append("d34S")
            if self.d34S >= 4.0:
                score += 0.50 * min(1.0, (self.d34S - 3.5) / 4.0)

        if _is_num(self.d13C):
            weights += 0.30
            basis.append("d13C")
            # Kraft lignin is less depleted than algal NOM (-27 vs -32)
            if self.d13C > -28.5:
                score += 0.30 * min(1.0, (self.d13C - (-29.0)) / 2.0)

        if _is_num(self.d15N):
            weights += 0.20
            basis.append("d15N")
            # Wood has low nitrogen content and low delta15N (1.0 to 4.0)
            if 0.5 <= self.d15N <= 4.5:
                score += 0.20

        self.kraft_prob_basis = basis
        if weights > 0:
            self.kraft_lignin_prob = float(np.clip(score / weights, 0.0, 1.0))
            skipped = [k for k in ("d34S", "d13C", "d15N") if k not in basis]
            if skipped:
                flags.append("kraft_prob_partial")
                notes.append("Kraft probability renormalised without " + ", ".join(skipped) + " (not measured).")
        else:
            self.kraft_lignin_prob = float("nan")
            flags.append("kraft_prob_no_data")
            notes.append("Kraft probability not computed: no δ34S, δ13C or δ15N measured.")

        if self.missing_isotopes:
            flags.append("missing:" + ",".join(self.missing_isotopes))
        self.flags = flags
        self.notes = notes
        return self


# Column matching (exact match on normalised tokens; no substring search)
_IRMS_SUPERSCRIPTS = str.maketrans({"¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁸": "8", "δ": "d", "Δ": "d"})
_IRMS_STRIP_TOKENS = {"permil", "per", "mil", "vpdb", "v", "pdb", "vsmow", "smow", "vcdt", "cdt", "air"}
_IRMS_ALIASES: Dict[str, List[str]] = {
    "id": ["sampleid", "sample", "id", "name", "samplename", "образец", "проба"],
    "d13c": ["d13c", "delta13c", "13c", "d13corg", "d13ctoc", "dc13"],
    "d15n": ["d15n", "delta15n", "15n", "d15ntn", "dn15"],
    "d34s": ["d34s", "delta34s", "34s", "ds34"],
    "d2h": ["d2h", "delta2h", "dd", "deltad", "2h", "dh2"],
    "d18o": ["d18o", "delta18o", "18o", "do18"],
    "cn": ["cn", "cnatomic", "cnratio", "cnat"],
    "toc": ["toc", "tocpct", "corg", "tocwt"],
    "type": ["type", "sampletype", "origin", "matrix", "тип"],
    "is_water": ["iswater", "water"],
}
_IRMS_OK_SUFFIXES = ("corr", "norm", "normalized", "normalised", "mean", "avg", "value", "val")


def _irms_token(col: Any) -> str:
    h = str(col).strip().lower().translate(_IRMS_SUPERSCRIPTS)
    h = re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", " ", h)
    h = h.replace("‰", " ")
    h = re.sub(r"[\s_\-/,.;:%]+", " ", h)
    return "".join(t for t in h.split() if t not in _IRMS_STRIP_TOKENS)


def parse_irms_table(content: Union[str, bytes, pd.DataFrame]) -> List[IRMSSample]:
    """
    Parses CSV, TSV, or DataFrame into a list of standardized IRMSSample instances.
    Accepts flexible column naming (d13C, δ¹³C (‰ VPDB), delta13C, d13C_permil, d34S, ...).
    Missing columns or blank cells give NaN (listed in ``sample.missing_isotopes``) —
    no default isotope values are substituted.
    """
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
            text = content.decode("utf-8-sig", errors="replace")
        else:
            text = str(content)
        sep = "\t" if "\t" in text.split("\n")[0] else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)

    tokens = {col: _irms_token(col) for col in df.columns}

    def _find_col(key: str) -> Optional[Any]:
        aliases = _IRMS_ALIASES[key]
        for col, tok in tokens.items():
            if tok in aliases:
                return col
        for col, tok in tokens.items():
            for a in aliases:
                if len(a) >= 3 and tok.startswith(a) and tok[len(a):] in _IRMS_OK_SUFFIXES:
                    return col
        return None

    c_id = _find_col("id") or df.columns[0]
    c_13c = _find_col("d13c")
    c_15n = _find_col("d15n")
    c_34s = _find_col("d34s")
    c_2h = _find_col("d2h")
    c_18o = _find_col("d18o")
    c_cn = _find_col("cn")
    c_toc = _find_col("toc")
    c_type = _find_col("type")
    c_water = _find_col("is_water")

    samples: List[IRMSSample] = []
    for idx, row in df.iterrows():
        s_id = str(row[c_id]) if pd.notna(row[c_id]) else f"Sample_{idx+1}"

        def _get_f(col) -> float:
            if col is not None and pd.notna(row[col]):
                try:
                    v = float(row[col])
                    return v if np.isfinite(v) else float("nan")
                except (ValueError, TypeError):
                    return float("nan")
            return float("nan")

        def _opt(col) -> Optional[float]:
            v = _get_f(col)
            return v if np.isfinite(v) else None

        is_water: Optional[bool] = None
        if c_water is not None and pd.notna(row[c_water]):
            is_water = str(row[c_water]).strip().lower() in ("1", "true", "yes", "y", "да", "water")
        s_type = str(row[c_type]) if c_type is not None and pd.notna(row[c_type]) else "Unknown"

        sample = IRMSSample(
            sample_id=s_id,
            d13C=_get_f(c_13c),
            d15N=_get_f(c_15n),
            d34S=_opt(c_34s),
            d2H=_opt(c_2h),
            d18O=_opt(c_18o),
            c_n_atomic=_opt(c_cn),
            toc_pct=_opt(c_toc),
            sample_type=s_type,
            is_water=is_water,
        )
        sample.compute_metrics()
        samples.append(sample)

    return samples


def generate_synthetic_irms_dataset() -> pd.DataFrame:
    """
    Generates a realistic reference IRMS benchmark dataset:
    Lake Baikal pristine waters, tributaries, and BPPM Kraft sludge-lignin dump samples.
    """
    np.random.seed(42)
    rows = []

    # 1. Kraft Pulp Sludge & Effluents (BPPM / Solzan accumulation) — 5 samples
    for i in range(1, 6):
        rows.append({
            "Sample_ID": f"BPPM_SlimeLignin_Core_0{i}",
            "d13C": round(np.random.normal(-27.2, 0.35), 2),
            "d15N": round(np.random.normal(2.4, 0.45), 2),
            "d34S": round(np.random.normal(6.8, 0.75), 2),    # Heavy kraft cooking liquor sulfate
            "d2H": round(np.random.normal(-132.0, 3.5), 1),
            "d18O": round(np.random.normal(15.8, 0.6), 2),
            "C_N_atomic": round(np.random.normal(38.0, 4.5), 1),
            "TOC_pct": round(np.random.normal(48.5, 5.0), 1),
            "Sample_Type": "Sludge_Lignin",
        })

    # 2. Pristine Lake Baikal Deep Waters (South & Middle Basins) — 4 samples
    for i in range(1, 5):
        rows.append({
            "Sample_ID": f"Baikal_Pristine_Pelagic_0{i}",
            "d13C": round(np.random.normal(-30.8, 0.5), 2),
            "d15N": round(np.random.normal(6.2, 0.5), 2),
            "d34S": round(np.random.normal(0.8, 0.4), 2),     # Pristine background
            "d2H": round(np.random.normal(-154.0, 2.5), 1),
            "d18O": round(np.random.normal(11.2, 0.4), 2),
            "C_N_atomic": round(np.random.normal(11.5, 1.2), 1),
            "TOC_pct": round(np.random.normal(1.8, 0.3), 1),
            "Sample_Type": "Pristine_Baikal",
        })

    # 3. Tributary Forested Watersheds (Selenga & Barguzin Rivers) — 3 samples
    for i in range(1, 4):
        rows.append({
            "Sample_ID": f"Tributary_Runoff_NOM_0{i}",
            "d13C": round(np.random.normal(-28.6, 0.4), 2),
            "d15N": round(np.random.normal(3.8, 0.4), 2),
            "d34S": round(np.random.normal(1.6, 0.5), 2),
            "d2H": round(np.random.normal(-141.0, 3.0), 1),
            "d18O": round(np.random.normal(13.4, 0.5), 2),
            "C_N_atomic": round(np.random.normal(22.0, 2.0), 1),
            "TOC_pct": round(np.random.normal(9.5, 1.5), 1),
            "Sample_Type": "Tributary_NOM",
        })

    return pd.DataFrame(rows)


def calculate_irms_descriptors(samples: List[IRMSSample]) -> pd.DataFrame:
    """Extracts tabular chemometric descriptor matrix for ML integration (NaN = not measured)."""
    def _nan(v):
        return float(v) if _is_num(v) else np.nan

    rows = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "d13C": _nan(s.d13C),
            "d15N": _nan(s.d15N),
            "d34S": _nan(s.d34S),
            "d2H": _nan(s.d2H),
            "d18O": _nan(s.d18O),
            "d_excess": _nan(s.deuterium_excess),
            "c_n_atomic": _nan(s.c_n_atomic),
            "kraft_lignin_prob": _nan(s.kraft_lignin_prob),
        })
    return pd.DataFrame(rows).set_index("sample_id")


def calculate_two_source_mixing(
    sample_val: float,
    endmember_a_val: float,
    endmember_b_val: float,
    with_flag: bool = False,
) -> Union[float, Tuple[float, str]]:
    """
    Calculates fraction of endmember A in a 2-source linear mixing model
    f_A = (δ_sample − δ_B) / (δ_A − δ_B), clipped to [0, 1].

    Returns NaN (flag "endmembers_identical") when the end-members coincide and NaN
    (flag "missing_value") when any input is missing. With ``with_flag=True`` returns
    (f_A, flag), where flag is "", "outside_endmember_range" (clipped), "endmembers_identical"
    or "missing_value".
    """
    if not (_is_num(sample_val) and _is_num(endmember_a_val) and _is_num(endmember_b_val)):
        f_a, flag = float("nan"), "missing_value"
    elif abs(float(endmember_b_val) - float(endmember_a_val)) < 1e-6:
        f_a, flag = float("nan"), "endmembers_identical"
    else:
        raw = (float(sample_val) - float(endmember_b_val)) / (float(endmember_a_val) - float(endmember_b_val))
        flag = "outside_endmember_range" if (raw < 0.0 or raw > 1.0) else ""
        f_a = float(np.clip(raw, 0.0, 1.0))
    return (f_a, flag) if with_flag else f_a
