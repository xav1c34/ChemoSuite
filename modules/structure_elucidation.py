"""
modules/structure_elucidation.py — Cross-Modal Structure Elucidation Engine for ChemoSuite Pro.

Synthesizes multi-instrument constraints across all modalities:
- FT-ICR MS: Exact neutral mass & assigned elemental formula
- FTIR: Functional groups (carbonyls, carboxylic acids, phenols, aromatics)
- 13C-NMR: Carbon skeletal distribution (Aliphatic, O-Alkyl, Aromatic, Carboxyl)
- CHNS-O: Bulk stoichiometric ratios (H/C, O/C, NOSC)

Retrieves isomeric candidates (PubChem PUG-REST fastformula, via chemo_pubchem) and calculates
Multi-Modal Concordance Scores (MMCS, 0-100%).

Honesty rules:
- No placeholder candidates are ever generated. If PubChem returns nothing or is unreachable,
  the result is an empty ElucidationResult with ``status``/``message`` explaining why.
- A score component whose constraint is not set (or cannot be computed) is NaN and is
  excluded from the weighted MMCS instead of being awarded 100.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    from modules import chemo_pubchem
except ImportError:
    import chemo_pubchem


@dataclass
class MultiModalConstraints:
    """Experimental constraints aggregated from analytical modules."""
    formula: str
    exact_mass: float = 0.0
    mass_tol_ppm: float = 2.0
    # FTIR presence flags: 1 = present, 0 = absent, -1 = ignored
    ftir_carbonyl: int = -1       # ~1700-1750 cm-1
    ftir_carboxyl: int = -1       # ~1700-1725 + 2500-3300 cm-1
    ftir_phenol_oh: int = -1      # ~3200-3550 cm-1
    ftir_aromatic_c_c: int = -1   # ~1500, 1600 cm-1
    ftir_aliphatic_c_h: int = -1  # ~2850-2960 cm-1
    ftir_ether_c_o: int = -1      # ~1050-1150 cm-1
    # 13C-NMR target carbon fraction integrals (sum = 1.0)
    nmr_aliphatic_pct: Optional[float] = None  # 0-50 ppm
    nmr_o_alkyl_pct: Optional[float] = None    # 50-110 ppm
    nmr_aromatic_pct: Optional[float] = None   # 110-165 ppm
    nmr_carboxyl_pct: Optional[float] = None   # 165-220 ppm
    # CHNS-O stoichiometric targets
    target_h_c: Optional[float] = None
    target_o_c: Optional[float] = None
    target_nosc: Optional[float] = None


@dataclass
class CandidateStructure:
    """Ranked chemical candidate with multi-modal matching metrics."""
    cid: Optional[int]
    name: str
    iupac_name: str
    smiles: str
    formula: str
    exact_mass: float
    concordance_score: float  # 0.0 to 100.0% (NaN if no constraint could be scored)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    predicted_features: Dict[str, Any] = field(default_factory=dict)
    svg_depiction: Optional[str] = None
    # Added (optional): provenance of the candidate and per-candidate warnings
    source: str = ""
    warnings: List[str] = field(default_factory=list)


class ElucidationResult(list):
    """
    List of CandidateStructure (fully list-compatible) with diagnostics:
      - status: 'ok' | 'no_candidates' | 'error' | 'invalid_formula'
      - message: human-readable explanation (always set when the list is empty)
      - source: 'custom' | 'pubchem' | 'none'
      - error: raw error text (network/HTTP) or None
    """

    def __init__(self, items=(), status: str = "ok", message: Optional[str] = None,
                 source: str = "none", error: Optional[str] = None):
        super().__init__(items)
        self.status = status
        self.message = message
        self.source = source
        self.error = error


# ==============================================================================
# Minimal SMILES parser (no RDKit dependency)
# ==============================================================================
_ORGANIC_VALENCES = {
    "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6),
    "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,),
}
_BOND_ORDERS = {"-": 1.0, "/": 1.0, "\\": 1.0, "=": 2.0, "#": 3.0, "$": 4.0, ":": 1.5}
_BRACKET_RE = re.compile(
    r"^(?P<iso>\d+)?(?P<sym>se|as|te|[bcnops]|[A-Z][a-z]?|\*)(?P<chiral>@+(?:TH|AL|SP|TB|OH)?\d*)?"
    r"(?P<h>H\d*)?(?P<chg>[+-]+\d*)?(?::\d+)?$"
)


class _MolGraph:
    """Heavy-atom graph: atoms (element, aromatic flag, hydrogen count, charge) and bond orders."""

    def __init__(self):
        self.el: List[str] = []
        self.arom: List[bool] = []
        self.charge: List[int] = []
        self.explicit_h: List[Optional[int]] = []
        self.nbrs: List[Dict[int, float]] = []

    def add_atom(self, el: str, aromatic: bool, h: Optional[int], charge: int) -> int:
        self.el.append(el)
        self.arom.append(aromatic)
        self.explicit_h.append(h)
        self.charge.append(charge)
        self.nbrs.append({})
        return len(self.el) - 1

    def add_bond(self, a: int, b: int, sym: Optional[str]) -> None:
        if a == b or b in self.nbrs[a]:
            raise ValueError("invalid bond")
        if sym is None:
            order = 1.5 if (self.arom[a] and self.arom[b]) else 1.0
        else:
            order = _BOND_ORDERS[sym]
        self.nbrs[a][b] = order
        self.nbrs[b][a] = order

    def n_atoms(self) -> int:
        return len(self.el)

    def h_count(self, i: int) -> int:
        """Implicit/explicit hydrogen count following the OpenSMILES default-valence rules."""
        if self.explicit_h[i] is not None:
            return self.explicit_h[i]
        el = self.el[i]
        vals = _ORGANIC_VALENCES.get(el)
        if not vals:
            return 0
        used = sum(1.0 if o == 1.5 else o for o in self.nbrs[i].values())
        if self.arom[i]:
            used += 1.0
        for v in vals:
            if v >= used:
                return int(max(0, round(v - used)))
        return 0

    def double_partners(self, i: int) -> List[int]:
        return [j for j, o in self.nbrs[i].items() if o == 2.0]


def _parse_smiles(smiles: str) -> Optional[_MolGraph]:
    """Parse a (Daylight/OpenSMILES) string into a heavy-atom graph; None if it cannot be parsed."""
    s = (smiles or "").strip()
    if not s or s.upper() in {"N/A", "NA", "NONE"}:
        return None
    g = _MolGraph()
    prev: Optional[int] = None
    stack: List[Optional[int]] = []
    pending: Optional[str] = None
    rings: Dict[str, Tuple[int, Optional[str]]] = {}
    i, n = 0, len(s)
    try:
        while i < n:
            ch = s[i]
            if ch == "(":
                if prev is None:
                    return None
                stack.append(prev)
                i += 1
                continue
            if ch == ")":
                if not stack:
                    return None
                prev = stack.pop()
                pending = None
                i += 1
                continue
            if ch in _BOND_ORDERS:
                pending = ch
                i += 1
                continue
            if ch == ".":
                prev, pending = None, None
                i += 1
                continue
            if ch.isdigit() or ch == "%":
                if ch == "%":
                    num = s[i + 1:i + 3]
                    if len(num) != 2 or not num.isdigit():
                        return None
                    i += 3
                else:
                    num = ch
                    i += 1
                if prev is None:
                    return None
                if num in rings:
                    j, sym_open = rings.pop(num)
                    g.add_bond(prev, j, pending or sym_open)
                else:
                    rings[num] = (prev, pending)
                pending = None
                continue
            if ch == "[":
                close = s.find("]", i)
                if close < 0:
                    return None
                m = _BRACKET_RE.match(s[i + 1:close])
                if not m:
                    return None
                sym = m.group("sym")
                aromatic = sym.islower()
                el = sym.capitalize() if aromatic else sym
                h_txt = m.group("h")
                h = 0 if not h_txt else (int(h_txt[1:]) if len(h_txt) > 1 else 1)
                chg_txt = m.group("chg") or ""
                if chg_txt:
                    sign = -1 if chg_txt[0] == "-" else 1
                    digits = chg_txt.lstrip("+-")
                    charge = sign * (int(digits) if digits else len(chg_txt))
                else:
                    charge = 0
                idx = g.add_atom(el, aromatic, h, charge)
                i = close + 1
            elif s[i:i + 2] in ("Cl", "Br"):
                idx = g.add_atom(s[i:i + 2], False, None, 0)
                i += 2
            elif ch in "BCNOPSFI":
                idx = g.add_atom(ch, False, None, 0)
                i += 1
            elif ch in "bcnops":
                idx = g.add_atom(ch.upper(), True, None, 0)
                i += 1
            elif ch == "*":
                idx = g.add_atom("*", False, 0, 0)
                i += 1
            else:
                return None
            if prev is not None:
                g.add_bond(prev, idx, pending)
            pending = None
            prev = idx
    except (ValueError, KeyError, IndexError):
        return None
    if rings or stack or g.n_atoms() == 0:
        return None
    return g


def _small_rings(g: _MolGraph, sizes=(5, 6)) -> List[List[int]]:
    """Enumerate simple cycles of the given sizes (ordered atom lists), each reported once."""
    max_len = max(sizes)
    found = {}
    for start in range(g.n_atoms()):
        path = [start]

        def dfs(cur: int):
            for nb in g.nbrs[cur]:
                if nb == start and len(path) in sizes:
                    key = frozenset(path)
                    if key not in found:
                        found[key] = list(path)
                elif nb > start and nb not in path and len(path) < max_len:
                    path.append(nb)
                    dfs(nb)
                    path.pop()

        dfs(start)
    return list(found.values())


def _perceive_aromatic_atoms(g: _MolGraph) -> List[bool]:
    """
    Aromaticity perception that also handles Kekulé SMILES (as returned by PubChem):
    - atoms written in lowercase are aromatic;
    - a 6-membered ring of C/N atoms is aromatic if every ring atom carries exactly one double
      bond whose partner lies in the same ring (benzenoid alternation C=C-C=C-C=C), or in an
      already-aromatic fused ring (naphthalene-type Kekulé structures; iterated to convergence);
    - a 5-membered ring is aromatic (furan/thiophene/pyrrole type) if one atom is O, S or N
      without a double bond and the four remaining C/N atoms form two ring-internal double bonds.
    Exocyclic C=O ring carbons (quinones, pyranones) are not counted as aromatic.
    """
    arom = list(g.arom)
    rings = _small_rings(g)
    changed = True
    while changed:
        changed = False
        for ring in rings:
            if all(arom[a] for a in ring):
                continue
            ring_set = set(ring)
            if len(ring) == 6:
                ok = True
                for a in ring:
                    if g.el[a] not in ("C", "N"):
                        ok = False
                        break
                    dps = g.double_partners(a)
                    if not dps and arom[a]:
                        continue  # already aromatic atom written in lowercase (mixed notation)
                    if len(dps) != 1:
                        ok = False
                        break
                    p = dps[0]
                    if p not in ring_set and not (arom[a] and arom[p]):
                        ok = False
                        break
                if ok:
                    for a in ring:
                        arom[a] = True
                    changed = True
            elif len(ring) == 5:
                hetero = [a for a in ring if g.el[a] in ("O", "S", "N") and not g.double_partners(a)]
                if len(hetero) != 1:
                    continue
                others = [a for a in ring if a != hetero[0]]
                ok = all(
                    g.el[a] in ("C", "N") and len(g.double_partners(a)) == 1
                    and g.double_partners(a)[0] in ring_set
                    for a in others
                )
                if ok:
                    for a in ring:
                        arom[a] = True
                    changed = True
    return arom


def _analyze_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    """Graph-based analysis shared by functional-group detection and 13C-NMR estimation."""
    g = _parse_smiles(smiles)
    if g is None:
        return None
    arom = _perceive_aromatic_atoms(g)

    def is_carbonyl_c(i: int) -> bool:
        return g.el[i] == "C" and any(g.el[j] == "O" and o == 2.0 for j, o in g.nbrs[i].items())

    def is_hydroxyl_o(j: int) -> bool:
        # -OH (implicit/explicit H) or deprotonated -O(-) with a single heavy neighbour
        return (g.el[j] == "O" and len(g.nbrs[j]) == 1 and all(o == 1.0 for o in g.nbrs[j].values())
                and (g.h_count(j) >= 1 or g.charge[j] == -1))

    feats = {
        "has_carbonyl": False,
        "has_carboxyl": False,
        "has_phenol_oh": False,
        "has_aromatic": any(arom),
        "has_aliphatic": False,
        "has_ether": False,
    }
    n_carboxyl = 0
    carbon_regions = {"aliphatic": 0, "o_alkyl": 0, "aromatic": 0, "carboxyl": 0}

    for i in range(g.n_atoms()):
        el = g.el[i]
        if el == "C":
            carbonyl = is_carbonyl_c(i)
            if carbonyl:
                feats["has_carbonyl"] = True
                if any(is_hydroxyl_o(j) for j, o in g.nbrs[i].items() if o == 1.0):
                    feats["has_carboxyl"] = True
                    n_carboxyl += 1
            orders = list(g.nbrs[i].values())
            sp3 = (not arom[i]) and all(o == 1.0 for o in orders)
            if sp3 and g.h_count(i) >= 1:
                feats["has_aliphatic"] = True
            if arom[i] and any(is_hydroxyl_o(j) for j in g.nbrs[i]):
                feats["has_phenol_oh"] = True

            # 13C-NMR chemical-shift region of this carbon
            if carbonyl:
                carbon_regions["carboxyl"] += 1          # C=O: 165-220 ppm
            elif arom[i] or any(o in (1.5, 2.0) for o in orders):
                carbon_regions["aromatic"] += 1          # aromatic + olefinic sp2: 110-165 ppm
            elif any(o == 3.0 for o in orders):
                if any(g.el[j] == "N" for j, o in g.nbrs[i].items() if o == 3.0):
                    carbon_regions["aromatic"] += 1      # nitrile C≡N: ~115-125 ppm
                else:
                    carbon_regions["o_alkyl"] += 1       # alkyne C≡C: ~65-90 ppm (50-110 window)
            elif any(g.el[j] == "O" for j in g.nbrs[i]):
                carbon_regions["o_alkyl"] += 1           # sp3 C-O (methoxyl, carbohydrate): 50-110 ppm
            else:
                carbon_regions["aliphatic"] += 1         # sp3 C/H, C-N: 0-50 ppm
        elif el == "O":
            heavy = list(g.nbrs[i].items())
            if (len(heavy) == 2 and all(g.el[j] == "C" and o in (1.0, 1.5) for j, o in heavy)
                    and not any(is_carbonyl_c(j) for j, _ in heavy)):
                feats["has_ether"] = True                # C-O-C, not ester/anhydride

    return {
        "features": feats,
        "carbon_regions": carbon_regions,
        "n_carbons": sum(carbon_regions.values()),
        "n_carboxyl": n_carboxyl,
    }


def detect_smiles_substructures(smiles: str) -> Dict[str, bool]:
    """
    Detect key diagnostic functional groups from a SMILES string (aromatic lowercase OR Kekulé
    notation, e.g. PubChem's vanillin COC1=C(C=CC(=C1)C=O)O).

    A small built-in SMILES parser builds the heavy-atom graph, so that:
    - aromatic rings are recognised from alternating C=C ring bonds (see _perceive_aromatic_atoms);
    - carbonyl means a C=O double bond on carbon (S(=O), N(=O) in sulfonyl/nitro are excluded);
    - carboxyl means C(=O)-OH / C(=O)-O(-) (esters are not carboxylic acids);
    - phenol means an -OH on an aromatic carbon; ether means C-O-C not adjacent to C=O.
    Extra key ``parse_ok`` is False when the SMILES could not be parsed (all flags False then).
    """
    res = _analyze_smiles(smiles)
    if res is None:
        return {
            "has_carbonyl": False, "has_carboxyl": False, "has_phenol_oh": False,
            "has_aromatic": False, "has_aliphatic": False, "has_ether": False,
            "parse_ok": False,
        }
    out = dict(res["features"])
    out["parse_ok"] = True
    return out


def estimate_13c_nmr_distribution(smiles: str) -> Dict[str, float]:
    """
    Estimate relative 13C-NMR carbon region proportions from carbon environments in SMILES.
    Returns:
    - aliphatic (0-50 ppm): sp3 C bonded only to C/H/N
    - o_alkyl (50-110 ppm): sp3 C bonded to O (methoxyl, carbohydrate, ether/ester alkoxy)
    - aromatic (110-165 ppm): aromatic (incl. Kekulé) and olefinic sp2 carbons, incl. aryl C-O
    - carboxyl (165-220 ppm): carbonyl carbons (acids, esters, ketones, aldehydes, amides)
    - total_carbons
    Fractions are NaN (total_carbons = 0) when the SMILES cannot be parsed or has no carbon.
    """
    res = _analyze_smiles(smiles)
    if res is None or res["n_carbons"] == 0:
        return {"aliphatic": np.nan, "o_alkyl": np.nan, "aromatic": np.nan, "carboxyl": np.nan,
                "total_carbons": 0}
    total = res["n_carbons"]
    cr = res["carbon_regions"]
    return {
        "aliphatic": float(cr["aliphatic"] / total),
        "o_alkyl": float(cr["o_alkyl"] / total),
        "aromatic": float(cr["aromatic"] / total),
        "carboxyl": float(cr["carboxyl"] / total),
        "total_carbons": total,
    }


def _formula_stoichiometry(formula: str) -> Optional[Dict[str, float]]:
    """
    H/C, O/C and NOSC of a molecular formula.
    NOSC = 4 - (4C + H - 3N - 2O - 2S) / C   (Riedel et al., 2012; LaRowe & Van Cappellen, 2011; z = 0)
    """
    counts = chemo_pubchem.parse_formula_counts(chemo_pubchem.clean_formula(formula or ""))
    c = counts.get("C", 0)
    if c <= 0:
        return None
    h, o, n, s = counts.get("H", 0), counts.get("O", 0), counts.get("N", 0), counts.get("S", 0)
    return {
        "H_C": h / c,
        "O_C": o / c,
        "NOSC": 4.0 - (4.0 * c + h - 3.0 * n - 2.0 * o - 2.0 * s) / c,
    }


# Heuristic linear scoring scales for stoichiometry (score = 100 * max(0, 1 - |Δ| / tol))
STOICH_TOLERANCES = {"H_C": 0.5, "O_C": 0.3, "NOSC": 1.0}
# Base weights of the MMCS components (renormalised over the components that can be scored)
MMCS_WEIGHTS = {"ms": 0.20, "ftir": 0.35, "nmr": 0.30, "stoich": 0.15}


def compute_concordance_score(
    candidate: Dict[str, Any],
    constraints: MultiModalConstraints
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """
    Calculate Multi-Modal Concordance Score (MMCS) between a candidate compound and constraints.

    Base weights (renormalised over the components that are actually scored):
    - MS exact mass: 20%  — 100 - 25 * (ppm error / tolerance); needs constraint and candidate mass
    - FTIR functional group concordance: 35% — share of matched present/absent flags
    - 13C-NMR distribution concordance: 30% — 100 * (1 - mean |Δ fraction|) over the provided regions
    - Bulk stoichiometry consistency: 15% — candidate-formula H/C, O/C, NOSC vs CHNS-O targets
      (100 * max(0, 1 - |Δ| / tol), tol: H/C 0.5, O/C 0.3, NOSC 1.0; mean over set targets)

    A component without a set constraint (or that cannot be computed, e.g. unparseable SMILES)
    is reported as NaN in the breakdown and excluded from the weighted total; if nothing can be
    scored, MMCS is NaN.
    """
    smiles = candidate.get("smiles", "") or ""
    substructs = detect_smiles_substructures(smiles)
    nmr_dist = estimate_13c_nmr_distribution(smiles)
    parse_ok = bool(substructs.get("parse_ok", False))

    # 1. MS Score (exact mass matching)
    ms_score = np.nan
    try:
        cand_mass = float(candidate.get("exact_mass"))
    except (TypeError, ValueError):
        cand_mass = np.nan
    if constraints.exact_mass and constraints.exact_mass > 0.0 and np.isfinite(cand_mass) and cand_mass > 0.0:
        ppm_err = abs(cand_mass - constraints.exact_mass) / constraints.exact_mass * 1e6
        ms_score = max(0.0, 100.0 - (ppm_err / max(0.1, constraints.mass_tol_ppm)) * 25.0)

    # 2. FTIR Score
    ftir_matches = 0
    ftir_total = 0
    ftir_checks = [
        (constraints.ftir_carbonyl, substructs["has_carbonyl"]),
        (constraints.ftir_carboxyl, substructs["has_carboxyl"]),
        (constraints.ftir_phenol_oh, substructs["has_phenol_oh"]),
        (constraints.ftir_aromatic_c_c, substructs["has_aromatic"]),
        (constraints.ftir_aliphatic_c_h, substructs["has_aliphatic"]),
        (constraints.ftir_ether_c_o, substructs["has_ether"]),
    ]
    if parse_ok:
        for req, detected in ftir_checks:
            if req == 1:
                ftir_total += 1
                if detected:
                    ftir_matches += 1
            elif req == 0:
                ftir_total += 1
                if not detected:
                    ftir_matches += 1
    ftir_score = (ftir_matches / ftir_total * 100.0) if ftir_total > 0 else np.nan

    # 3. NMR Score: mean absolute deviation over the provided carbon-region fractions only
    nmr_pairs = [
        (constraints.nmr_aliphatic_pct, nmr_dist["aliphatic"]),
        (constraints.nmr_o_alkyl_pct, nmr_dist["o_alkyl"]),
        (constraints.nmr_aromatic_pct, nmr_dist["aromatic"]),
        (constraints.nmr_carboxyl_pct, nmr_dist["carboxyl"]),
    ]
    nmr_devs = [abs(float(t) - float(p)) for t, p in nmr_pairs if t is not None and np.isfinite(p)]
    nmr_score = max(0.0, min(100.0, 100.0 * (1.0 - float(np.mean(nmr_devs))))) if nmr_devs else np.nan

    # 4. Stoichiometry Score (candidate formula vs CHNS-O targets)
    stoich = _formula_stoichiometry(candidate.get("formula") or constraints.formula)
    stoich_parts = []
    if stoich is not None:
        for key, target in (("H_C", constraints.target_h_c), ("O_C", constraints.target_o_c),
                            ("NOSC", constraints.target_nosc)):
            if target is not None and np.isfinite(float(target)):
                delta = abs(stoich[key] - float(target))
                stoich_parts.append(100.0 * max(0.0, 1.0 - delta / STOICH_TOLERANCES[key]))
    stoich_score = float(np.mean(stoich_parts)) if stoich_parts else np.nan

    components = {"ms": ms_score, "ftir": ftir_score, "nmr": nmr_score, "stoich": stoich_score}
    used = {k: v for k, v in components.items() if v is not None and np.isfinite(v)}
    total_weight = sum(MMCS_WEIGHTS[k] for k in used)
    if total_weight > 0:
        mmcs = sum(MMCS_WEIGHTS[k] * v for k, v in used.items()) / total_weight
    else:
        mmcs = np.nan

    def _r(v):
        return round(float(v), 1) if np.isfinite(v) else np.nan

    score_breakdown = {
        "MS Accuracy": _r(ms_score),
        "FTIR Functional Groups": _r(ftir_score),
        "13C-NMR Skeleton": _r(nmr_score),
        "Bulk Stoichiometry": _r(stoich_score),
    }

    predicted_features = {
        "functional_groups": substructs,
        "nmr_fractions": nmr_dist,
        "stoichiometry": stoich,
        "scored_components": sorted(used.keys()),
    }

    return (float(round(mmcs, 1)) if np.isfinite(mmcs) else np.nan), score_breakdown, predicted_features


def elucidate_candidates(
    constraints: MultiModalConstraints,
    custom_candidates: Optional[List[Dict[str, Any]]] = None,
    max_results: int = 15
) -> List[CandidateStructure]:
    """
    Search chemical database and rank candidate isomeric structures by Multi-Modal Concordance.

    Parameters:
    -----------
    constraints: MultiModalConstraints
        Aggregated physical-chemical parameters from active spectra.
    custom_candidates: Optional list of dictionaries with {cid, name, smiles, exact_mass, formula}.
        When omitted, isomers are retrieved from PubChem (chemo_pubchem.fetch_compounds_by_formula,
        exact_mass = PubChem MonoisotopicMass).
    max_results: int
        Max number of top matching candidates to return.

    Returns:
    --------
    ElucidationResult (a list of CandidateStructure ranked descending by concordance_score;
    NaN scores last) with ``status``, ``message``, ``source`` and ``error`` attributes.
    No placeholder candidates are generated: if PubChem has no isomers or is unreachable,
    the list is empty and ``message`` explains why.
    """
    source = "custom"
    if custom_candidates:
        raw_list = list(custom_candidates)
    else:
        source = "pubchem"
        try:
            raw_list = chemo_pubchem.fetch_compounds_by_formula(constraints.formula, max_records=max_results * 2)
        except ValueError as e:
            return ElucidationResult([], status="invalid_formula", source=source, error=str(e),
                                     message=f"Invalid molecular formula '{constraints.formula}': {e}")
        except chemo_pubchem.PubChemError as e:
            return ElucidationResult([], status="error", source=source, error=str(e),
                                     message=f"PubChem search failed, no candidates retrieved: {e}")

    if not raw_list:
        return ElucidationResult(
            [], status="no_candidates", source=source,
            message=(f"PubChem returned no compounds for formula {constraints.formula}."
                     if source == "pubchem" else "No candidate structures were supplied."),
        )

    query_counts = chemo_pubchem.parse_formula_counts(chemo_pubchem.clean_formula(constraints.formula))
    scored_candidates = []
    for cand in raw_list:
        mmcs, breakdown, pred_feats = compute_concordance_score(cand, constraints)
        warns: List[str] = []
        if not pred_feats["functional_groups"].get("parse_ok", False):
            warns.append("SMILES could not be parsed; FTIR/NMR components not scored.")
        cand_formula = cand.get("formula") or constraints.formula
        cand_counts = chemo_pubchem.parse_formula_counts(chemo_pubchem.clean_formula(cand_formula))
        if query_counts and cand_counts and cand_counts != query_counts:
            warns.append(f"Candidate formula {cand_formula} differs from query {constraints.formula}.")
        try:
            mass_val = float(cand.get("exact_mass"))
        except (TypeError, ValueError):
            mass_val = np.nan
        struct = CandidateStructure(
            cid=cand.get("cid"),
            name=cand.get("name") or f"CID {cand.get('cid', 'Unknown')}",
            iupac_name=cand.get("iupac_name") or cand.get("name", "Unknown Compound"),
            smiles=cand.get("smiles", ""),
            formula=cand_formula,
            exact_mass=mass_val,
            concordance_score=mmcs,
            score_breakdown=breakdown,
            predicted_features=pred_feats,
            source=str(cand.get("source") or ("PubChem" if source == "pubchem" else "custom")),
            warnings=warns,
        )
        scored_candidates.append(struct)

    # Sort descending by concordance score (NaN last)
    scored_candidates.sort(
        key=lambda c: c.concordance_score if (c.concordance_score is not None and math.isfinite(c.concordance_score)) else -math.inf,
        reverse=True,
    )
    return ElucidationResult(scored_candidates[:max_results], status="ok", source=source)
