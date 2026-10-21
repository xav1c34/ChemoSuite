"""
irms_core.py — Computational Engine for Isotope Ratio Mass Spectrometry (IRMS).

Analyzes stable isotope ratios (delta13C, delta15N, delta34S, delta2H, delta18O)
for geochemical provenance, tracing terrestrial vs. aquatic inputs, and differentiating
pulp mill kraft sludge-lignin from natural background organic matter.
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


@dataclass
class IRMSSample:
    """Stores multi-element stable isotope ratios for an analytical sample."""
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

    # Calculated provenance metrics
    kraft_lignin_prob: float = 0.0
    deuterium_excess: float = 0.0  # d-excess = d2H - 8 * d18O

    def compute_metrics(self) -> "IRMSSample":
        """Calculates secondary geochemical and provenance indicators."""
        # 1. Deuterium excess (Dansgaard 1964)
        if self.d2H is not None and self.d18O is not None:
            self.deuterium_excess = float(self.d2H - 8.0 * self.d18O)

        # 2. Heuristic probability of Kraft Mill Pulp Sludge origin
        # Kraft pulp lignin from BPPM typically has heavy d34S (>4.0) and d13C around -27.0
        score = 0.0
        weights = 0.0

        if self.d34S is not None:
            weights += 0.50
            if self.d34S >= 4.0:
                score += 0.50 * min(1.0, (self.d34S - 3.5) / 4.0)
            elif self.d34S <= 1.0:
                score += 0.0

        if self.d13C is not None:
            weights += 0.30
            # Kraft lignin is less depleted than algal NOM (-27 vs -32)
            if self.d13C > -28.5:
                score += 0.30 * min(1.0, (self.d13C - (-29.0)) / 2.0)

        if self.d15N is not None:
            weights += 0.20
            # Wood has low nitrogen content and low delta15N (1.0 to 4.0)
            if 0.5 <= self.d15N <= 4.5:
                score += 0.20

        if weights > 0:
            self.kraft_lignin_prob = float(np.clip(score / weights, 0.0, 1.0))
        return self


def parse_irms_table(content: Union[str, bytes, pd.DataFrame]) -> List[IRMSSample]:
    """
    Parses CSV, TSV, or DataFrame into a list of standardized IRMSSample instances.
    Accepts flexible column naming (e.g. d13C, delta13C, d13C_permil, d34S, etc.).
    """
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        sep = "\t" if "\t" in text.split("\n")[0] else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map possible column names
    col_map = {
        "id": ["sample_id", "sample", "id", "name", "sample_name"],
        "d13c": ["d13c", "delta13c", "d13c_permil", "13c", "delta_13c"],
        "d15n": ["d15n", "delta15n", "d15n_permil", "15n", "delta_15n"],
        "d34s": ["d34s", "delta34s", "d34s_permil", "34s", "delta_34s"],
        "d2h": ["d2h", "delta2h", "dd", "deltad", "d2h_permil"],
        "d18o": ["d18o", "delta18o", "d18o_permil", "18o", "delta_18o"],
        "cn": ["c_n", "c/n", "c_n_atomic", "cn_ratio"],
        "toc": ["toc", "toc_pct", "c_org"],
        "type": ["type", "sample_type", "origin", "matrix"],
    }

    def _find_col(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in df.columns:
                return k
        for col in df.columns:
            for k in keys:
                if k in col:
                    return col
        return None

    c_id = _find_col(col_map["id"]) or df.columns[0]
    c_13c = _find_col(col_map["d13c"])
    c_15n = _find_col(col_map["d15n"])
    c_34s = _find_col(col_map["d34s"])
    c_2h = _find_col(col_map["d2h"])
    c_18o = _find_col(col_map["d18o"])
    c_cn = _find_col(col_map["cn"])
    c_toc = _find_col(col_map["toc"])
    c_type = _find_col(col_map["type"])

    samples: List[IRMSSample] = []
    for idx, row in df.iterrows():
        s_id = str(row[c_id]) if pd.notna(row[c_id]) else f"Sample_{idx+1}"

        def _get_f(col):
            if col and col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    return None
            return None

        val_13c = _get_f(c_13c) if c_13c else -28.0
        val_15n = _get_f(c_15n) if c_15n else 3.5
        val_34s = _get_f(c_34s)
        val_2h = _get_f(c_2h)
        val_18o = _get_f(c_18o)
        val_cn = _get_f(c_cn)
        val_toc = _get_f(c_toc)
        s_type = str(row[c_type]) if c_type and pd.notna(row[c_type]) else "Unknown"

        sample = IRMSSample(
            sample_id=s_id,
            d13C=val_13c if val_13c is not None else -28.0,
            d15N=val_15n if val_15n is not None else 3.5,
            d34S=val_34s,
            d2H=val_2h,
            d18O=val_18o,
            c_n_atomic=val_cn,
            toc_pct=val_toc,
            sample_type=s_type,
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
    """Extracts tabular chemometric descriptor matrix for ML integration."""
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "d13C": s.d13C,
            "d15N": s.d15N,
            "d34S": s.d34S if s.d34S is not None else np.nan,
            "d2H": s.d2H if s.d2H is not None else np.nan,
            "d18O": s.d18O if s.d18O is not None else np.nan,
            "d_excess": s.deuterium_excess if s.d2H is not None else np.nan,
            "c_n_atomic": s.c_n_atomic if s.c_n_atomic is not None else np.nan,
            "kraft_lignin_prob": s.kraft_lignin_prob,
        })
    return pd.DataFrame(rows).set_index("sample_id")


def calculate_two_source_mixing(
    sample_val: float,
    endmember_a_val: float,
    endmember_b_val: float,
) -> float:
    """Calculates fraction of endmember A in a 2-source linear mixing model."""
    if abs(endmember_b_val - endmember_a_val) < 1e-6:
        return 0.5
    f_a = (sample_val - endmember_b_val) / (endmember_a_val - endmember_b_val)
    return float(np.clip(f_a, 0.0, 1.0))
