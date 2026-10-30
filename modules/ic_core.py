"""
ic_core.py — Computational Engine for Ion Chromatography (IC).

Analyzes major inorganic anions (SO4(2-), Cl-, NO3-, F-, PO4(3-)), carbonate
alkalinity (HCO3-/CO3(2-)), low-molecular-weight organic acids (LMWOA: acetate,
formate, oxalate, citrate), and major cations (Na+, K+, Ca(2+), Mg(2+), NH4+).
Calculates charge balance (electroneutrality), ionic strength, SO4/Cl ratio,
and Kraft pulp mill effluent markers.

Charge balance
--------------
In most fresh waters HCO3- is the dominant anion (Lake Baikal: ~1.1 of ~1.2 meq/L).
A charge balance without carbonate alkalinity is therefore meaningless; when no
HCO3-/CO3(2-) concentration, total alkalinity or pH < 4.5 is available the
charge-balance error and ionic strength are returned as NaN with
``charge_balance_flag = "HCO3_missing"`` instead of a plausible-looking number.

References
----------
* Standard Methods for the Examination of Water and Wastewater, 2320 B
  (alkalinity; 1 meq = 50.04 mg CaCO3) and 1030 E (anion–cation balance).
* Stumm W., Morgan J.J. (1996) Aquatic Chemistry, 3rd ed. — H3PO4 pKa 2.15, 7.20, 12.35.
* Plummer L.N., Busenberg E. (1982) GCA 46:1011 — carbonate pKa2 = 10.33 (25 °C).
* Appelo C.A.J., Postma D. (2005) Geochemistry, Groundwater and Pollution — ±5 % CBE criterion.
* Falkner K.K. et al. (1991) Limnol. Oceanogr. 36:413; Votintsev K.K. (1961)
  Hydrochemistry of Lake Baikal — Baikal pelagic major-ion composition.
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
    # Carbonate alkalinity (counted in the charge balance, not in total_mineral_anions_mg_l)
    "hco3_minus": {"name": "Bicarbonate (HCO3-)", "mw": 61.017, "z": 1, "type": "carbonate"},
    "co3_2minus": {"name": "Carbonate (CO3 2-)", "mw": 60.009, "z": 2, "type": "carbonate"},
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

# Physico-chemical constants (25 °C)
PHOSPHATE_PKA: Tuple[float, float, float] = (2.15, 7.20, 12.35)  # Stumm & Morgan (1996)
CARBONATE_PKA2: float = 10.33   # HCO3- <-> CO3(2-) + H+ (Plummer & Busenberg 1982)
WATER_PKW: float = 14.0
CACO3_MG_PER_MEQ: float = 50.04  # Standard Methods 2320 B
CBE_ACCEPTANCE_PCT: float = 5.0  # Appelo & Postma (2005)

# Ions without which an anion–cation balance of a natural water is not interpretable
MAJOR_IONS_REQUIRED: Tuple[str, ...] = ("ca_2plus", "mg_2plus", "na_plus", "cl_minus", "so4_2minus")

# Pelagic Lake Baikal SO4/Cl mass ratio (SO4 ≈ 5.5, Cl ≈ 0.43 mg/L; Falkner et al. 1991)
BAIKAL_SO4_CL_BACKGROUND: float = 13.0

_CARBONATE_KEYS = ("hco3_minus", "co3_2minus")


def _finite(v: Any) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def phosphate_mean_charge(ph: float) -> Tuple[float, float]:
    """
    Mean charge z̄ and mean z² of dissolved orthophosphate at a given pH
    (speciation H3PO4 / H2PO4- / HPO4(2-) / PO4(3-), pKa 2.15, 7.20, 12.35).
    Returns (nan, nan) if pH is not finite.
    """
    if not _finite(ph):
        return float("nan"), float("nan")
    h = 10.0 ** (-float(ph))
    k1, k2, k3 = (10.0 ** (-pk) for pk in PHOSPHATE_PKA)
    terms = np.array([h ** 3, k1 * h ** 2, k1 * k2 * h, k1 * k2 * k3], dtype=float)
    alpha = terms / terms.sum()
    z = np.array([0.0, 1.0, 2.0, 3.0])
    return float(np.sum(alpha * z)), float(np.sum(alpha * z ** 2))


def partition_alkalinity(alk_meq_l: float, ph: Optional[float] = None) -> Tuple[float, float, List[str]]:
    """
    Splits total alkalinity (meq/L) into HCO3- and CO3(2-) (mmol/L).

    Alk = [HCO3-] + 2[CO3(2-)] + [OH-] - [H+]  (Standard Methods 2320 B; organic/borate
    alkalinity neglected). With pH, [CO3]/[HCO3] = 10^(pH - pKa2). Without pH all
    carbonate alkalinity is attributed to HCO3- (valid below pH ≈ 8.3) and a note is returned.
    """
    notes: List[str] = []
    alk = float(alk_meq_l)
    if not _finite(ph):
        notes.append("pH not given: alkalinity attributed entirely to HCO3- (valid for pH < 8.3).")
        return max(alk, 0.0), 0.0, notes
    ph = float(ph)
    oh = 10.0 ** (ph - WATER_PKW) * 1000.0
    h = 10.0 ** (-ph) * 1000.0
    carb = alk - oh + h
    if carb < 0:
        notes.append("Alkalinity below [OH-] - [H+] at this pH: carbonate set to 0.")
        carb = 0.0
    r = 10.0 ** (ph - CARBONATE_PKA2)
    hco3 = carb / (1.0 + 2.0 * r)
    return float(hco3), float(r * hco3), notes


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
    # Carbonate system (None = not measured)
    hco3_minus: Optional[float] = None       # mg/L HCO3-
    co3_2minus: Optional[float] = None       # mg/L CO3(2-)
    alkalinity_meq_l: Optional[float] = None  # total alkalinity, meq/L
    ph: Optional[float] = None

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
    sum_cations_meq_l: float = float("nan")
    sum_anions_meq_l: float = float("nan")
    hco3_eff_mg_l: float = float("nan")      # HCO3- used in the balance (measured or from alkalinity)
    co3_eff_mg_l: float = float("nan")
    po4_mean_charge: float = float("nan")
    carbonate_basis: str = ""                 # "measured" | "alkalinity" | "pH<4.5" | "missing"
    charge_balance_flag: str = ""             # "ok" | "imbalance" | "HCO3_missing" | "major_ion_missing" | "no_ions"
    notes: List[str] = field(default_factory=list)

    def compute_metrics(self) -> "ICSample":
        """Computes geochemical ratios, charge balance, and Kraft effluent index."""
        notes: List[str] = []
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

        # SO4/Cl mass ratio (undefined without chloride — not a division by 1e-4)
        if self.cl_minus > 1e-4:
            self.so4_to_cl_ratio = float(self.so4_2minus / self.cl_minus)
        else:
            self.so4_to_cl_ratio = float("nan")

        # LMWOA fraction of total organic/mineral anions
        tot_all_anions = self.total_mineral_anions_mg_l + self.total_lmwoa_mg_l
        self.lmwoa_fraction = float(self.total_lmwoa_mg_l / tot_all_anions) if tot_all_anions > 1e-6 else float("nan")

        ph = float(self.ph) if _finite(self.ph) else None

        # Phosphate: mean charge from pH speciation; nominal PO4(3-) otherwise
        if ph is not None:
            po4_z, po4_z2 = phosphate_mean_charge(ph)
        else:
            po4_z, po4_z2 = 3.0, 9.0
            if self.po4_3minus > 0:
                notes.append("PO4 counted as PO4(3-) (z = 3): pH not given, charge overestimated at natural pH.")
        self.po4_mean_charge = float(po4_z)

        # Equivalent charge (meq/L) for charge balance: meq/L = (mg/L / mw) * z
        anion_meq = 0.0
        cation_meq = 0.0
        sum_cz2 = 0.0  # for ionic strength (mmol/L * z^2)

        for species_key, info in ION_SPECIES.items():
            if info["type"] == "carbonate":
                continue
            conc = getattr(self, species_key, 0.0)
            if conc is None or not _finite(conc) or conc <= 0:
                continue
            mmol = conc / info["mw"]
            if species_key == "po4_3minus":
                z, z2 = po4_z, po4_z2
            else:
                z, z2 = info["z"], info["z"] ** 2
            if info["type"] in ("mineral_anion", "lmwoa"):
                anion_meq += mmol * z
            elif info["type"] == "cation":
                cation_meq += mmol * z
            sum_cz2 += mmol * z2

        # Carbonate system
        hco3_mg = self.hco3_minus if _finite(self.hco3_minus) else None
        co3_mg = self.co3_2minus if _finite(self.co3_2minus) else None
        alk = self.alkalinity_meq_l if _finite(self.alkalinity_meq_l) else None
        h_mmol = 10.0 ** (-ph) * 1000.0 if ph is not None else 0.0
        oh_mmol = 10.0 ** (ph - WATER_PKW) * 1000.0 if ph is not None else 0.0

        carbonate_known = True
        if hco3_mg is not None:
            self.carbonate_basis = "measured"
            hco3_mmol = max(hco3_mg, 0.0) / ION_SPECIES["hco3_minus"]["mw"]
            co3_mmol = max(co3_mg or 0.0, 0.0) / ION_SPECIES["co3_2minus"]["mw"]
            anion_meq += hco3_mmol + 2.0 * co3_mmol + oh_mmol
            cation_meq += h_mmol
        elif alk is not None:
            self.carbonate_basis = "alkalinity"
            hco3_mmol, co3_mmol, alk_notes = partition_alkalinity(alk, ph)
            notes.extend(alk_notes)
            # Alk = HCO3 + 2 CO3 + OH - H already contains the H+/OH- terms
            anion_meq += alk
        elif ph is not None and ph < 4.5:
            self.carbonate_basis = "pH<4.5"
            hco3_mmol, co3_mmol = 0.0, 0.0
            cation_meq += h_mmol
            notes.append("pH < 4.5: carbonate alkalinity taken as 0, H+ counted as a cation.")
        else:
            self.carbonate_basis = "missing"
            carbonate_known = False
            hco3_mmol = co3_mmol = float("nan")

        if carbonate_known:
            self.hco3_eff_mg_l = float(hco3_mmol * ION_SPECIES["hco3_minus"]["mw"])
            self.co3_eff_mg_l = float(co3_mmol * ION_SPECIES["co3_2minus"]["mw"])
            sum_cz2 += hco3_mmol + 4.0 * co3_mmol + h_mmol + oh_mmol
        else:
            self.hco3_eff_mg_l = float("nan")
            self.co3_eff_mg_l = float("nan")

        self.sum_cations_meq_l = float(cation_meq)
        self.sum_anions_meq_l = float(anion_meq) if carbonate_known else float("nan")

        missing_major = [k for k in self.metadata.get("missing_species", []) if k in MAJOR_IONS_REQUIRED]
        tot_meq = cation_meq + anion_meq
        if missing_major:
            self.charge_balance_error_pct = float("nan")
            self.ionic_strength_mmol_l = float("nan")
            self.charge_balance_flag = "major_ion_missing"
            notes.append("Charge balance not computed: major ion(s) not measured: " + ", ".join(missing_major) + ".")
        elif not carbonate_known:
            self.charge_balance_error_pct = float("nan")
            self.ionic_strength_mmol_l = float("nan")
            self.charge_balance_flag = "HCO3_missing"
            notes.append("Charge balance not computed: no HCO3-/CO3(2-), alkalinity or pH < 4.5 given "
                         "(HCO3- is usually the dominant anion of fresh water).")
        elif tot_meq <= 1e-4:
            self.charge_balance_error_pct = float("nan")
            self.ionic_strength_mmol_l = float("nan")
            self.charge_balance_flag = "no_ions"
        else:
            # Charge Balance Error (%) = (Cations - Anions) / (Cations + Anions) * 100 (SM 1030 E)
            self.charge_balance_error_pct = float(((cation_meq - anion_meq) / tot_meq) * 100.0)
            # Ionic strength I = 0.5 * sum(c_i * z_i^2) in mmol/L
            self.ionic_strength_mmol_l = float(0.5 * sum_cz2)
            self.charge_balance_flag = "ok" if abs(self.charge_balance_error_pct) <= CBE_ACCEPTANCE_PCT else "imbalance"

        # Kraft Pulp Effluent Indicator (0.0 to 1.0) — heuristic screening score
        # Kraft liquor is dominated by high SO4 (from Na2SO4 cooking liquor oxidation),
        # high Na+, elevated acetate/formate from hemicellulose peeling reactions.
        score = 0.0
        # 1. Sulfate concentration (>20 mg/L vs ~5.5 mg/L Baikal pelagic background)
        if self.so4_2minus > 20.0:
            score += 0.40 * min(1.0, (self.so4_2minus - 20.0) / 80.0)
        # 2. SO4/Cl excess over the Baikal pelagic background (≈13 by mass, Falkner et al. 1991)
        if _finite(self.so4_to_cl_ratio) and self.so4_to_cl_ratio > BAIKAL_SO4_CL_BACKGROUND:
            score += 0.30 * min(1.0, (self.so4_to_cl_ratio - BAIKAL_SO4_CL_BACKGROUND) / BAIKAL_SO4_CL_BACKGROUND)
        # 3. Acetate + Formate (> 1.5 mg/L)
        lignin_acids = self.acetate + self.formate
        if lignin_acids > 1.5:
            score += 0.20 * min(1.0, (lignin_acids - 1.5) / 15.0)
        # 4. Sodium dominance
        if self.na_plus > 15.0:
            score += 0.10 * min(1.0, (self.na_plus - 15.0) / 50.0)

        self.kraft_effluent_indicator = float(np.clip(score, 0.0, 1.0))
        self.notes = notes
        return self


# ==============================================================================
# Column matching (exact match on normalised header tokens — no substring search)
# ==============================================================================

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁻": "-", "⁺": "+", "₋": "-", "₊": "+", "−": "-", "–": "-", "μ": "µ",
})

_UNIT_SUFFIX_RE = re.compile(
    r"[\s_,;:]+(?:as\s*caco3|caco3|"
    r"mg\s*/\s*l|mg_?l|mg\s*/\s*dm3|мг\s*/\s*л|мг\s*/\s*дм3|ppm|ppb|"
    r"µg\s*/\s*l|ug\s*/\s*l|ug_?l|мкг\s*/\s*л|"
    r"meq\s*/\s*l|meq_?l|µeq\s*/\s*l|ueq\s*/\s*l|ueq_?l|"
    r"mmol\s*/\s*l|mmol_?l|ммоль\s*/\s*л|мг[\s-]*экв\s*/\s*л|мэкв\s*/\s*л)\s*$"
)
_BRACKETS_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_COMPACT_RE = re.compile(r"[\s_\-\+\^\(\)\[\]\{\}\.,:;/'\"*]+")

_IC_ALIASES: Dict[str, List[str]] = {
    "so4_2minus": ["so4", "so42", "sulfate", "sulphate", "sulfates", "sulphates", "сульфат", "сульфаты"],
    "cl_minus": ["cl", "chloride", "chlorides", "хлорид", "хлориды"],
    "no3_minus": ["no3", "nitrate", "nitrates", "нитрат", "нитраты"],
    "f_minus": ["f", "fluoride", "fluorides", "фторид", "фториды"],
    "po4_3minus": ["po4", "po43", "phosphate", "orthophosphate", "фосфат", "фосфаты"],
    "no2_minus": ["no2", "nitrite", "nitrites", "нитрит", "нитриты"],
    "br_minus": ["br", "bromide", "bromides", "бромид", "бромиды"],
    "hco3_minus": ["hco3", "bicarbonate", "hydrogencarbonate", "гидрокарбонат", "гидрокарбонаты"],
    "co3_2minus": ["co3", "co32", "carbonate", "карбонат", "карбонаты"],
    "acetate": ["acetate", "acetic", "aceticacid", "ch3coo", "ацетат"],
    "formate": ["formate", "formic", "formicacid", "hcoo", "формиат"],
    "oxalate": ["oxalate", "oxalic", "oxalicacid", "c2o4", "c2o42", "оксалат"],
    "lactate": ["lactate", "lactic", "lacticacid", "лактат"],
    "succinate": ["succinate", "succinic", "succinicacid", "сукцинат"],
    "malate": ["malate", "malic", "malicacid", "малат"],
    "citrate": ["citrate", "citric", "citricacid", "цитрат"],
    "na_plus": ["na", "sodium", "натрий"],
    "k_plus": ["k", "potassium", "калий"],
    "ca_2plus": ["ca", "ca2", "calcium", "кальций"],
    "mg_2plus": ["mg", "mg2", "magnesium", "магний"],
    "nh4_plus": ["nh4", "ammonium", "аммоний"],
}
# Nutrients reported "as N" / "as P": compact token -> (species, mass factor to the ion)
_IC_AS_ELEMENT: Dict[str, Tuple[str, float]] = {
    "no3n": ("no3_minus", 62.004 / 14.007),
    "no2n": ("no2_minus", 46.005 / 14.007),
    "nh4n": ("nh4_plus", 18.039 / 14.007),
    "po4p": ("po4_3minus", 94.971 / 30.974),
}
_IC_META_ALIASES: Dict[str, List[str]] = {
    "id": ["sampleid", "sample", "id", "name", "samplename", "образец", "проба", "шифр"],
    "type": ["type", "sampletype", "origin", "matrix", "тип", "типпробы"],
    "alkalinity": ["alkalinity", "alk", "totalalkalinity", "talk", "щелочность", "общаящелочность"],
    "ph": ["ph", "рн", "phvalue"],
}

_ALK_UNIT_ALIASES = {
    "meq/l": "meq/l", "meq": "meq/l", "mval/l": "meq/l",
    "mmol/l": "mmol/l", "mmol": "mmol/l",
    "ueq/l": "ueq/l", "µeq/l": "ueq/l",
    "mg/l caco3": "mg/l caco3", "mg/l as caco3": "mg/l caco3", "caco3": "mg/l caco3",
}


def _detect_unit(raw: str) -> Optional[str]:
    if "caco3" in raw:
        return "mg/l caco3"
    if re.search(r"(µeq|ueq|мкг[\s-]*экв|мкэкв)", raw):
        return "ueq/l"
    if re.search(r"(meq|мг[\s-]*экв|мэкв|mval)", raw):
        return "meq/l"
    if re.search(r"(mmol|ммоль)", raw):
        return "mmol/l"
    if re.search(r"(µg|ug[\s_/]|ug$|мкг|ppb)", raw):
        return "ug/l"
    if re.search(r"(mg\s*[/_]?\s*(l|dm3)\b|мг\s*/?\s*(л|дм3)|ppm)", raw):
        return "mg/l"
    return None


def _normalise_header(col: Any) -> Tuple[str, Optional[str]]:
    """Returns (compact token, detected unit) for a column header."""
    raw = str(col).strip().lower().translate(_SUPERSCRIPT_MAP)
    unit = _detect_unit(raw)
    h = _BRACKETS_RE.sub(" ", raw).strip()
    prev = None
    while prev != h:
        prev = h
        h = _UNIT_SUFFIX_RE.sub("", h).strip()
    return _COMPACT_RE.sub("", h), unit


def _ion_to_mg_l(value: float, unit: Optional[str], species: str) -> float:
    info = ION_SPECIES[species]
    if unit in (None, "mg/l"):
        return value
    if unit == "ug/l":
        return value / 1000.0
    if unit == "mmol/l":
        return value * info["mw"]
    if unit == "meq/l":
        return value * info["mw"] / info["z"]
    if unit == "ueq/l":
        return value * info["mw"] / info["z"] / 1000.0
    if unit == "mg/l caco3":
        return value / CACO3_MG_PER_MEQ * info["mw"] / info["z"]
    return value


def _alk_to_meq_l(value: float, unit: str) -> float:
    if unit in ("meq/l", "mmol/l"):
        return value
    if unit == "ueq/l":
        return value / 1000.0
    if unit == "mg/l caco3":
        return value / CACO3_MG_PER_MEQ
    raise ValueError(unit)


def resolve_ic_columns(columns: List[Any]) -> Dict[str, Any]:
    """
    Maps table columns onto IC species by exact match of the normalised header
    (lower-case, unicode super/subscripts folded, unit suffixes and charge signs removed)
    against an alias table. Returns {"species": {key: (col, unit, factor)},
    "meta": {key: (col, unit)}, "warnings": [...]}.
    """
    species_map: Dict[str, Tuple[Any, Optional[str], float]] = {}
    meta_map: Dict[str, Tuple[Any, Optional[str]]] = {}
    warnings: List[str] = []
    lookup = {alias: sp for sp, aliases in _IC_ALIASES.items() for alias in aliases}
    meta_lookup = {alias: m for m, aliases in _IC_META_ALIASES.items() for alias in aliases}

    as_element: Dict[str, Tuple[Any, Optional[str], float]] = {}
    for col in columns:
        token, unit = _normalise_header(col)
        if token in lookup:
            sp = lookup[token]
            if sp in species_map:
                warnings.append(f"Column '{col}' ignored: {sp} already taken from '{species_map[sp][0]}'.")
            else:
                species_map[sp] = (col, unit, 1.0)
        elif token in _IC_AS_ELEMENT:
            sp, factor = _IC_AS_ELEMENT[token]
            as_element.setdefault(sp, (col, unit, factor))
        elif token in meta_lookup:
            m = meta_lookup[token]
            if m not in meta_map:
                meta_map[m] = (col, unit)
    for sp, val in as_element.items():
        if sp not in species_map:
            species_map[sp] = val
    return {"species": species_map, "meta": meta_map, "warnings": warnings}


def parse_ic_table(
    content: Union[str, bytes, pd.DataFrame],
    alkalinity_unit: Optional[str] = None,
) -> List[ICSample]:
    """
    Parses CSV, TSV, or DataFrame into a list of ICSample instances.

    Columns are matched by exact normalised tokens (``SO4``, ``SO4²⁻ (mg/L)``, ``sulfate_mg_l``,
    ``Сульфат, мг/л`` → sulfate; ``Mg`` → magnesium; ``so4_mg_l`` is *not* magnesium).
    Concentrations are mg/L unless the header states µg/L, mmol/L, meq/L or mg/L as CaCO3.
    Carbonate: ``HCO3``/``CO3`` concentration columns, or a total-alkalinity column
    (``Alkalinity_meq_L``, ``Alkalinity (mg/L as CaCO3)``); ``alkalinity_unit``
    ("meq/L", "mmol/L", "ueq/L", "mg/L CaCO3") overrides/supplies the alkalinity unit.
    An optional ``pH`` column drives phosphate and carbonate speciation.
    """
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
            text = content.decode("utf-8-sig", errors="replace")
        else:
            text = str(content)
        first = text.split("\n")[0]
        sep = "\t" if "\t" in first else ("," if "," in first or ";" not in first else ";")
        df = pd.read_csv(io.StringIO(text), sep=sep)

    alk_unit_override: Optional[str] = None
    if alkalinity_unit is not None:
        key = str(alkalinity_unit).strip().lower().replace("μ", "µ")
        if key not in _ALK_UNIT_ALIASES:
            raise ValueError(f"Unknown alkalinity_unit '{alkalinity_unit}'. Use one of: meq/L, mmol/L, ueq/L, mg/L CaCO3.")
        alk_unit_override = _ALK_UNIT_ALIASES[key]

    resolved = resolve_ic_columns(list(df.columns))
    species_cols = resolved["species"]
    meta_cols = resolved["meta"]
    table_warnings = list(resolved["warnings"])

    c_id = meta_cols["id"][0] if "id" in meta_cols else df.columns[0]
    c_type = meta_cols["type"][0] if "type" in meta_cols else None
    c_ph = meta_cols["ph"][0] if "ph" in meta_cols else None
    c_alk, alk_unit = (meta_cols["alkalinity"] if "alkalinity" in meta_cols else (None, None))
    if c_alk is not None:
        alk_unit = alk_unit_override or alk_unit
        if alk_unit not in ("meq/l", "mmol/l", "ueq/l", "mg/l caco3"):
            table_warnings.append(
                f"Alkalinity column '{c_alk}' ignored: unit not stated (meq/L and mg/L as CaCO3 differ ×50). "
                "Rename the column or pass alkalinity_unit."
            )
            c_alk = None

    def _num(row: pd.Series, col: Any) -> float:
        if col is None:
            return float("nan")
        try:
            v = float(row[col])
        except (ValueError, TypeError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    samples: List[ICSample] = []
    for idx, row in df.iterrows():
        s_id = str(row[c_id]) if pd.notna(row[c_id]) else f"IC_Sample_{idx+1}"
        sample_kwargs: Dict[str, Any] = {
            "sample_id": s_id,
            "sample_type": str(row[c_type]) if c_type is not None and pd.notna(row[c_type]) else "Unknown",
        }
        missing: List[str] = []
        for species in ION_SPECIES.keys():
            spec = species_cols.get(species)
            v = _num(row, spec[0]) if spec else float("nan")
            if spec and np.isfinite(v):
                sample_kwargs[species] = _ion_to_mg_l(v * spec[2], spec[1], species)
            else:
                missing.append(species)
                if species not in _CARBONATE_KEYS:
                    sample_kwargs[species] = 0.0  # reported as not detected; tracked in metadata
        ph_v = _num(row, c_ph)
        if np.isfinite(ph_v):
            sample_kwargs["ph"] = ph_v
        alk_v = _num(row, c_alk)
        if c_alk is not None and np.isfinite(alk_v):
            sample_kwargs["alkalinity_meq_l"] = _alk_to_meq_l(alk_v, alk_unit)

        sample = ICSample(**sample_kwargs)
        sample.metadata["missing_species"] = missing
        sample.metadata["column_map"] = {k: str(v[0]) for k, v in species_cols.items()}
        if table_warnings:
            sample.metadata["parse_warnings"] = table_warnings
        sample.compute_metrics()
        samples.append(sample)

    return samples


def generate_synthetic_ic_dataset() -> pd.DataFrame:
    """
    Generates a reference Ion Chromatography (IC) benchmark dataset:
    Lake Baikal pristine waters, major tributaries (Selenga), and BPPM Kraft pulp mill sludge leachate.

    The Baikal pelagic composition follows Votintsev (1961) and Falkner et al. (1991):
    HCO3 ≈ 66, Ca ≈ 16, Mg ≈ 3.0, Na ≈ 3.4, K ≈ 1.0, SO4 ≈ 5.5, Cl ≈ 0.4 mg/L, pH ≈ 7.9.
    HCO3- of the tributary and leachate classes closes their charge balance (illustrative).
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
            "HCO3": round(np.random.normal(125.0, 6.0), 1),
            "pH": round(np.random.normal(7.6, 0.1), 2),
            "Sample_Type": "Pulp_Effluent_Slime",
        })

    # 2. Pristine Lake Baikal Deep Waters (South & Central basins) — 4 samples
    # Ultra-oligotrophic Ca–HCO3 water (mineralisation ≈ 96 mg/L): HCO3 ≈ 66, SO4 ≈ 5.5,
    # Cl ≈ 0.4 mg/L (SO4/Cl ≈ 13 by mass), LMWOA below 0.3 mg/L
    for i in range(1, 5):
        rows.append({
            "Sample_ID": f"Baikal_Pristine_Pelagic_0{i}",
            "SO4": round(np.random.normal(5.5, 0.2), 2),
            "Cl": round(np.random.normal(0.42, 0.03), 2),
            "NO3": round(np.random.normal(0.35, 0.05), 2),
            "F": round(np.random.normal(0.12, 0.01), 2),
            "PO4": round(np.random.normal(0.04, 0.008), 3),
            "Acetate": round(np.random.normal(0.18, 0.04), 2),
            "Formate": round(np.random.normal(0.12, 0.03), 2),
            "Oxalate": round(np.random.normal(0.08, 0.02), 2),
            "Na": round(np.random.normal(3.4, 0.1), 2),
            "K": round(np.random.normal(1.0, 0.05), 2),
            "Ca": round(np.random.normal(16.0, 0.3), 2),
            "Mg": round(np.random.normal(3.05, 0.08), 2),
            "NH4": round(np.random.normal(0.03, 0.008), 3),
            "HCO3": round(np.random.normal(66.0, 1.0), 1),
            "pH": round(np.random.normal(7.9, 0.05), 2),
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
            "HCO3": round(np.random.normal(88.0, 4.0), 1),
            "pH": round(np.random.normal(7.6, 0.1), 2),
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
            "hco3_mg_l": s.hco3_eff_mg_l,
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
