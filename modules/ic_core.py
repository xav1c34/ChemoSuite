"""
ic_core.py — Computational Engine for Ion Chromatography (IC).

Analyzes major inorganic anions (SO4(2-), Cl-, NO3-, F-, PO4(3-)),
low-molecular-weight organic acids (LMWOA: acetate, formate, oxalate, citrate),
and major cations (Na+, K+, Ca(2+), Mg(2+), NH4+).
Calculates charge balance (electroneutrality), ionic strength, SO4/Cl ratio,
and Kraft pulp mill effluent markers.
"""
from dataclasses import dataclass, field
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Molar masses (g/mol) and ionic valence (z)
ION_SPECIES = {
    # Anions
    "f_minus": {"name": "Fluoride (F-)", "mw": 18.998, "z": 1, "type": "mineral_anion"},
    "cl_minus": {"name": "Chloride (Cl-)", "mw": 35.453, "z": 1, "type": "mineral_anion"},
    "no2_minus": {"name": "Nitrite (NO2-)", "mw": 46.005, "z": 1, "type": "mineral_anion"},
    "no3_minus": {"name": "Nitrate (NO3-)", "mw": 62.004, "z": 1, "type": "mineral_anion"},
    "br_minus": {"name": "Bromide (Br-)", "mw": 79.904, "z": 1, "type": "mineral_anion"},
    "po4_3minus": {"name": "Phosphate (PO4 3-)", "mw": 94.971, "z": 3, "type": "mineral_anion"},
    "so4_2minus": {"name": "Sulfate (SO4 2-)", "mw": 96.06, "z": 2, "type": "mineral_anion"},
    # LMWOA (Low-molecular-weight organic acids)
    "acetate": {"name": "Acetate (CH3COO-)", "mw": 59.044, "z": 1, "type": "lmwoa"},
    "formate": {"name": "Formate (HCOO-)", "mw": 45.017, "z": 1, "type": "lmwoa"},
    "oxalate": {"name": "Oxalate (C2O4 2-)", "mw": 88.02, "z": 2, "type": "lmwoa"},
    "lactate": {"name": "Lactate (C3H5O3-)", "mw": 89.07, "z": 1, "type": "lmwoa"},
    "succinate": {"name": "Succinate (C4H4O4 2-)", "mw": 116.07, "z": 2, "type": "lmwoa"},
    "malate": {"name": "Malate (C4H4O5 2-)", "mw": 132.07, "z": 2, "type": "lmwoa"},
    "citrate": {"name": "Citrate (C6H5O7 3-)", "mw": 189.10, "z": 3, "type": "lmwoa"},
    # Cations
    "na_plus": {"name": "Sodium (Na+)", "mw": 22.990, "z": 1, "type": "cation"},
    "nh4_plus": {"name": "Ammonium (NH4+)", "mw": 18.039, "z": 1, "type": "cation"},
    "k_plus": {"name": "Potassium (K+)", "mw": 39.098, "z": 1, "type": "cation"},
    "mg_2plus": {"name": "Magnesium (Mg 2+)", "mw": 24.305, "z": 2, "type": "cation"},
    "ca_2plus": {"name": "Calcium (Ca 2+)", "mw": 40.078, "z": 2, "type": "cation"},
}


@dataclass
class ICSample:
    """Stores quantitative Ion Chromatography (IC) profile for a water or NOM sample."""
    sample_id: str
    # Anions (mg/L)
    so4_2minus: float = 0.0
    cl_minus: float = 0.0
    no3_minus: float = 0.0
    f_minus: float = 0.0
    po4_3minus: float = 0.0
    no2_minus: float = 0.0
    br_minus: float = 0.0
    # LMWOA (mg/L)
    acetate: float = 0.0
    formate: float = 0.0
    oxalate: float = 0.0
    lactate: float = 0.0
    succinate: float = 0.0
    malate: float = 0.0
    citrate: float = 0.0
    # Cations (mg/L)
    na_plus: float = 0.0
    k_plus: float = 0.0
    ca_2plus: float = 0.0
    mg_2plus: float = 0.0
    nh4_plus: float = 0.0

    sample_type: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Calculated indices
    total_mineral_anions_mg_l: float = 0.0
    total_lmwoa_mg_l: float = 0.0
    total_cations_mg_l: float = 0.0
    so4_to_cl_ratio: float = 0.0
    lmwoa_fraction: float = 0.0
    charge_balance_error_pct: float = 0.0
    ionic_strength_mmol_l: float = 0.0
    kraft_effluent_indicator: float = 0.0

    def compute_metrics(self) -> "ICSample":
        """Computes geochemical ratios, charge balance, and Kraft effluent index."""
        mineral_anions = [
            self.so4_2minus, self.cl_minus, self.no3_minus,
            self.f_minus, self.po4_3minus, self.no2_minus, self.br_minus
        ]
        self.total_mineral_anions_mg_l = float(sum(max(0.0, v) for v in mineral_anions))

        lmwoa = [
            self.acetate, self.formate, self.oxalate,
            self.lactate, self.succinate, self.malate, self.citrate
        ]
        self.total_lmwoa_mg_l = float(sum(max(0.0, v) for v in lmwoa))

        cations = [
            self.na_plus, self.k_plus, self.ca_2plus,
            self.mg_2plus, self.nh4_plus
        ]
        self.total_cations_mg_l = float(sum(max(0.0, v) for v in cations))

        # SO4/Cl mass ratio
        if self.cl_minus > 1e-4:
            self.so4_to_cl_ratio = float(self.so4_2minus / self.cl_minus)
        else:
            self.so4_to_cl_ratio = float(self.so4_2minus / 1e-4) if self.so4_2minus > 0 else 0.0

        # LMWOA fraction of total organic/mineral anions
        tot_all_anions = self.total_mineral_anions_mg_l + self.total_lmwoa_mg_l
        if tot_all_anions > 1e-6:
            self.lmwoa_fraction = float(self.total_lmwoa_mg_l / tot_all_anions)
        else:
            self.lmwoa_fraction = 0.0

        # Equivalent charge (meq/L) for charge balance
        # meq/L = (conc_mg_L / mw) * z
        anion_meq = 0.0
        cation_meq = 0.0
        sum_cz2 = 0.0  # for ionic strength (mol/L * z^2)

        for species_key, info in ION_SPECIES.items():
            conc = getattr(self, species_key, 0.0)
            if conc > 0:
                meq = (conc / info["mw"]) * info["z"]
                if info["type"] in ("mineral_anion", "lmwoa"):
                    anion_meq += meq
                elif info["type"] == "cation":
                    cation_meq += meq
                # Ionic strength component (conc in mmol/L)
                sum_cz2 += (conc / info["mw"]) * (info["z"] ** 2)

        # Charge Balance Error (%) = (Cations - Anions) / (Cations + Anions) * 100
        tot_meq = cation_meq + anion_meq
        if tot_meq > 1e-4:
            self.charge_balance_error_pct = float(((cation_meq - anion_meq) / tot_meq) * 100.0)
        else:
            self.charge_balance_error_pct = 0.0

        # Ionic strength I = 0.5 * sum(c_i * z_i^2) in mmol/L
        self.ionic_strength_mmol_l = float(0.5 * sum_cz2)

        # Kraft Pulp Effluent Indicator (0.0 to 1.0)
        # Kraft liquor is dominated by high SO4 (from Na2SO4 cooking liquor oxidation),
        # high Na+, elevated acetate/formate from hemicellulose peeling reactions.
        score = 0.0
        # 1. Sulfate concentration (>25 mg/L in Baikal context indicates plume/dump runoff)
        if self.so4_2minus > 20.0:
            score += 0.40 * min(1.0, (self.so4_2minus - 20.0) / 80.0)
        # 2. SO4/Cl ratio (> 4.0 indicates non-marine sulfate excess)
        if self.so4_to_cl_ratio > 3.0:
            score += 0.30 * min(1.0, (self.so4_to_cl_ratio - 3.0) / 10.0)
        # 3. Acetate + Formate (> 2.0 mg/L)
        lignin_acids = self.acetate + self.formate
        if lignin_acids > 1.5:
            score += 0.20 * min(1.0, (lignin_acids - 1.5) / 15.0)
        # 4. Sodium dominance
        if self.na_plus > 15.0:
            score += 0.10 * min(1.0, (self.na_plus - 15.0) / 50.0)

        self.kraft_effluent_indicator = float(np.clip(score, 0.0, 1.0))
        return self


def parse_ic_table(content: Union[str, bytes, pd.DataFrame]) -> List[ICSample]:
    """
    Parses CSV, TSV, or DataFrame into a list of ICSample instances.
    Accepts flexible column naming for anions, cations, and LMWOA.
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

    orig_cols = list(df.columns)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map possible column naming
    col_aliases = {
        "id": ["sample_id", "sample", "id", "name", "sample_name"],
        "so4_2minus": ["so4", "so4_2-", "so42-", "sulfate", "so4_mg_l", "sulfate_mg_l"],
        "cl_minus": ["cl", "cl-", "chloride", "cl_mg_l", "chloride_mg_l"],
        "no3_minus": ["no3", "no3-", "nitrate", "no3_mg_l", "nitrate_mg_l"],
        "f_minus": ["f", "f-", "fluoride", "f_mg_l"],
        "po4_3minus": ["po4", "po4_3-", "phosphate", "po4_mg_l"],
        "no2_minus": ["no2", "no2-", "nitrite", "no2_mg_l"],
        "br_minus": ["br", "br-", "bromide", "br_mg_l"],
        "acetate": ["acetate", "acetic", "ch3coo-", "ch3coo", "acetate_mg_l"],
        "formate": ["formate", "formic", "hcoo-", "hcoo", "formate_mg_l"],
        "oxalate": ["oxalate", "oxalic", "c2o4", "oxalate_mg_l"],
        "lactate": ["lactate", "lactic", "lactate_mg_l"],
        "succinate": ["succinate", "succinic", "succinate_mg_l"],
        "malate": ["malate", "malic", "malate_mg_l"],
        "citrate": ["citrate", "citric", "citrate_mg_l"],
        "na_plus": ["na", "na+", "sodium", "na_mg_l"],
        "k_plus": ["k", "k+", "potassium", "k_mg_l"],
        "ca_2plus": ["ca", "ca2+", "calcium", "ca_mg_l"],
        "mg_2plus": ["mg", "mg2+", "magnesium", "mg_mg_l"],
        "nh4_plus": ["nh4", "nh4+", "ammonium", "nh4_mg_l"],
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

    c_id = _find_col(col_aliases["id"]) or df.columns[0]
    c_type = _find_col(col_aliases["type"])

    resolved_cols: Dict[str, Optional[str]] = {}
    for species in ION_SPECIES.keys():
        resolved_cols[species] = _find_col(col_aliases.get(species, [species]))

    samples: List[ICSample] = []
    for idx, row in df.iterrows():
        s_id = str(row[c_id]) if pd.notna(row[c_id]) else f"IC_Sample_{idx+1}"

        def _val(col_name: Optional[str]) -> float:
            if col_name and col_name in row and pd.notna(row[col_name]):
                try:
                    return float(row[col_name])
                except (ValueError, TypeError):
                    return 0.0
            return 0.0

        sample_kwargs: Dict[str, Any] = {
            "sample_id": s_id,
            "sample_type": str(row[c_type]) if c_type and pd.notna(row[c_type]) else "Unknown",
        }
        for species in ION_SPECIES.keys():
            sample_kwargs[species] = _val(resolved_cols[species])

        sample = ICSample(**sample_kwargs)
        sample.compute_metrics()
        samples.append(sample)

    return samples


def generate_synthetic_ic_dataset() -> pd.DataFrame:
    """
    Generates a realistic reference Ion Chromatography (IC) benchmark dataset:
    Lake Baikal pristine waters, major tributaries (Selenga), and BPPM Kraft pulp mill sludge leachate.
    """
    np.random.seed(42)
    rows = []

    # 1. BPPM Kraft Slime Lignin Pond Leachate & Effluent — 5 samples
    # Characterized by high SO4 (from oxidized sulfide/sulfate cooking liquor),
    # high Na+, elevated acetate and formate from alkaline carbohydrate peeling
    for i in range(1, 6):
        rows.append({
            "Sample_ID": f"BPPM_Leachate_Core_0{i}",
            "SO4": round(np.random.normal(165.0, 18.0), 2),
            "Cl": round(np.random.normal(12.5, 1.8), 2),
            "NO3": round(np.random.normal(3.2, 0.6), 2),
            "F": round(np.random.normal(0.45, 0.08), 2),
            "PO4": round(np.random.normal(0.85, 0.15), 2),
            "Acetate": round(np.random.normal(24.5, 4.2), 2),
            "Formate": round(np.random.normal(14.0, 2.5), 2),
            "Oxalate": round(np.random.normal(6.5, 1.1), 2),
            "Na": round(np.random.normal(92.0, 8.5), 2),
            "K": round(np.random.normal(14.5, 1.5), 2),
            "Ca": round(np.random.normal(28.0, 3.2), 2),
            "Mg": round(np.random.normal(11.0, 1.4), 2),
            "NH4": round(np.random.normal(2.8, 0.5), 2),
            "Sample_Type": "Pulp_Effluent_Slime",
        })

    # 2. Pristine Lake Baikal Deep Waters (South & Central basins) — 4 samples
    # Ultra-oligotrophic background: low ionic strength, low SO4 (~5.0 mg/L),
    # SO4/Cl ratio ~ 1.5 - 2.0, Ca/Mg typical freshwater, LMWOA below 0.3 mg/L
    for i in range(1, 5):
        rows.append({
            "Sample_ID": f"Baikal_Pristine_Pelagic_0{i}",
            "SO4": round(np.random.normal(5.4, 0.35), 2),
            "Cl": round(np.random.normal(3.1, 0.25), 2),
            "NO3": round(np.random.normal(0.65, 0.08), 2),
            "F": round(np.random.normal(0.12, 0.02), 2),
            "PO4": round(np.random.normal(0.04, 0.01), 2),
            "Acetate": round(np.random.normal(0.18, 0.04), 2),
            "Formate": round(np.random.normal(0.12, 0.03), 2),
            "Oxalate": round(np.random.normal(0.08, 0.02), 2),
            "Na": round(np.random.normal(3.8, 0.3), 2),
            "K": round(np.random.normal(1.1, 0.1), 2),
            "Ca": round(np.random.normal(15.2, 0.8), 2),
            "Mg": round(np.random.normal(4.3, 0.3), 2),
            "NH4": round(np.random.normal(0.05, 0.01), 2),
            "Sample_Type": "Pristine_Baikal",
        })

    # 3. Selenga & Barguzin River Tributary Runoff — 3 samples
    # Intermediate sulfate, soil humic leaching, elevated oxalate from forest soil
    for i in range(1, 4):
        rows.append({
            "Sample_ID": f"Selenga_Tributary_0{i}",
            "SO4": round(np.random.normal(16.5, 2.0), 2),
            "Cl": round(np.random.normal(6.2, 0.8), 2),
            "NO3": round(np.random.normal(1.8, 0.3), 2),
            "F": round(np.random.normal(0.22, 0.04), 2),
            "PO4": round(np.random.normal(0.15, 0.03), 2),
            "Acetate": round(np.random.normal(1.8, 0.35), 2),
            "Formate": round(np.random.normal(1.1, 0.2), 2),
            "Oxalate": round(np.random.normal(2.4, 0.4), 2),
            "Na": round(np.random.normal(8.5, 1.2), 2),
            "K": round(np.random.normal(2.4, 0.3), 2),
            "Ca": round(np.random.normal(22.0, 2.5), 2),
            "Mg": round(np.random.normal(6.8, 0.7), 2),
            "NH4": round(np.random.normal(0.4, 0.08), 2),
            "Sample_Type": "Tributary_Runoff",
        })

    return pd.DataFrame(rows)


def calculate_ic_descriptors(samples: List[ICSample]) -> pd.DataFrame:
    """Extracts tabular chemometric descriptor matrix for ML integration."""
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "so4_mg_l": s.so4_2minus,
            "cl_mg_l": s.cl_minus,
            "no3_mg_l": s.no3_minus,
            "so4_cl_ratio": s.so4_to_cl_ratio,
            "total_mineral_anions": s.total_mineral_anions_mg_l,
            "total_lmwoa": s.total_lmwoa_mg_l,
            "total_cations": s.total_cations_mg_l,
            "acetate_mg_l": s.acetate,
            "formate_mg_l": s.formate,
            "oxalate_mg_l": s.oxalate,
            "lmwoa_fraction": s.lmwoa_fraction,
            "charge_balance_err_pct": s.charge_balance_error_pct,
            "ionic_strength_mmol_l": s.ionic_strength_mmol_l,
            "kraft_effluent_indicator": s.kraft_effluent_indicator,
        })
    return pd.DataFrame(rows).set_index("sample_id")
