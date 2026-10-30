"""
thermal_core.py — Математическое ядро термического анализа и пиролиза (Py-GC-MS & TGA/DTG/DSC).
Методология хемометрического анализа термической устойчивости, продуктов пиролитической
деструкции природного органического вещества (РОВ/DOM, гуминовые кислоты) и техногенного шлам-лигнина.
"""
from dataclasses import dataclass, field
import io
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.ndimage
import scipy.signal


def decode_bytes(b: bytes) -> str:
    """Универсальное декодирование байтов файла с поддержкой распространенных кодировок."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ==============================================================================
# ЭТАЛОННАЯ БИБЛИОТЕКА ДИАГНОСТИЧЕСКИХ ПРОДУКТОВ ПИРОЛИЗА (Py-GC-MS LIBRARY)
# ==============================================================================
# Систематизирована по типам мономеров лигнина (H, G, S), полисахаридам и белкам
PY_GCMS_REFERENCE_LIBRARY: List[Dict[str, Any]] = [
    # --- 1. p-Hydroxyphenyl (H-лигнин) ---
    {
        "name_en": "Phenol",
        "name_ru": "Фенол",
        "cas": "108-95-2",
        "rt": 7.20,
        "mw": 94.11,
        "class": "H_Lignin",
        "subclass": "H_monomer",
        "formula": "C6H6O",
    },
    {
        "name_en": "4-Methylphenol (p-Cresol)",
        "name_ru": "4-Метилфенол (п-Крезол)",
        "cas": "106-44-5",
        "rt": 9.80,
        "mw": 108.14,
        "class": "H_Lignin",
        "subclass": "H_alkyl",
        "formula": "C7H8O",
    },
    {
        "name_en": "4-Ethylphenol",
        "name_ru": "4-Этилфенол",
        "cas": "123-07-9",
        "rt": 12.40,
        "mw": 122.16,
        "class": "H_Lignin",
        "subclass": "H_alkyl",
        "formula": "C8H10O",
    },
    {
        "name_en": "4-Vinylphenol",
        "name_ru": "4-Винилфенол",
        "cas": "2628-17-3",
        "rt": 14.10,
        "mw": 120.15,
        "class": "H_Lignin",
        "subclass": "H_alkenyl",
        "formula": "C8H8O",
    },

    # --- 2. Guaiacyl (G-лигнин: хвойные породы и технические лигнины) ---
    {
        "name_en": "Guaiacol (2-Methoxyphenol)",
        "name_ru": "Гваякол (2-Метоксифенол)",
        "cas": "90-05-1",
        "rt": 10.50,
        "mw": 124.14,
        "class": "G_Lignin",
        "subclass": "G_monomer",
        "formula": "C7H8O2",
    },
    {
        "name_en": "4-Methylguaiacol (Creosol)",
        "name_ru": "4-Метилгваякол (Креозол)",
        "cas": "93-51-6",
        "rt": 13.20,
        "mw": 138.16,
        "class": "G_Lignin",
        "subclass": "G_alkyl",
        "formula": "C8H10O2",
    },
    {
        "name_en": "4-Ethylguaiacol",
        "name_ru": "4-Этилгваякол",
        "cas": "2785-89-9",
        "rt": 15.80,
        "mw": 152.19,
        "class": "G_Lignin",
        "subclass": "G_alkyl",
        "formula": "C9H12O2",
    },
    {
        "name_en": "4-Vinylguaiacol",
        "name_ru": "4-Винилгваякол",
        "cas": "7786-61-0",
        "rt": 17.40,
        "mw": 150.18,
        "class": "G_Lignin",
        "subclass": "G_alkenyl",
        "formula": "C9H10O2",
    },
    {
        "name_en": "Eugenol (4-Allylguaiacol)",
        "name_ru": "Эвгенол (4-Аллилгваякол)",
        "cas": "97-53-0",
        "rt": 19.50,
        "mw": 164.20,
        "class": "G_Lignin",
        "subclass": "G_alkenyl",
        "formula": "C10H12O2",
    },
    {
        "name_en": "trans-Isoeugenol",
        "name_ru": "транс-Изоэвгенол",
        "cas": "5932-68-3",
        "rt": 21.20,
        "mw": 164.20,
        "class": "G_Lignin",
        "subclass": "G_alkenyl",
        "formula": "C10H12O2",
    },
    {
        "name_en": "Vanillin (4-Hydroxy-3-methoxybenzaldehyde)",
        "name_ru": "Ванилин",
        "cas": "121-33-5",
        "rt": 22.80,
        "mw": 152.15,
        "class": "G_Lignin",
        "subclass": "G_carbonyl",
        "formula": "C8H8O3",
    },
    {
        "name_en": "Acetovanillone (Apocynin)",
        "name_ru": "Ацетованилон (Апоцинин)",
        "cas": "498-02-2",
        "rt": 24.50,
        "mw": 166.17,
        "class": "G_Lignin",
        "subclass": "G_carbonyl",
        "formula": "C9H10O3",
    },
    {
        "name_en": "Methyl vanillate",
        "name_ru": "Метилванилат",
        "cas": "3943-74-6",
        "rt": 26.10,
        "mw": 182.17,
        "class": "G_Lignin",
        "subclass": "G_carbonyl",
        "formula": "C9H10O4",
    },

    # --- 3. Syringyl (S-лигнин: лиственные породы и травянистые растения) ---
    {
        "name_en": "Syringol (2,6-Dimethoxyphenol)",
        "name_ru": "Сирингол (2,6-Диметоксифенол)",
        "cas": "91-10-1",
        "rt": 18.20,
        "mw": 154.16,
        "class": "S_Lignin",
        "subclass": "S_monomer",
        "formula": "C8H10O3",
    },
    {
        "name_en": "4-Methylsyringol",
        "name_ru": "4-Метилсирингол",
        "cas": "6638-05-7",
        "rt": 20.60,
        "mw": 168.19,
        "class": "S_Lignin",
        "subclass": "S_alkyl",
        "formula": "C9H12O3",
    },
    {
        "name_en": "4-Ethylsyringol",
        "name_ru": "4-Этилсирингол",
        "cas": "14059-92-8",
        "rt": 22.90,
        "mw": 182.22,
        "class": "S_Lignin",
        "subclass": "S_alkyl",
        "formula": "C10H14O3",
    },
    {
        "name_en": "4-Vinylsyringol",
        "name_ru": "4-Винилсирингол",
        "cas": "20628-06-2",
        "rt": 24.60,
        "mw": 180.20,
        "class": "S_Lignin",
        "subclass": "S_alkenyl",
        "formula": "C10H12O3",
    },
    {
        "name_en": "4-Allylsyringol",
        "name_ru": "4-Аллилсирингол",
        "cas": "6627-88-9",
        "rt": 26.40,
        "mw": 194.23,
        "class": "S_Lignin",
        "subclass": "S_alkenyl",
        "formula": "C11H14O3",
    },
    {
        "name_en": "Syringaldehyde",
        "name_ru": "Сирингальдегид",
        "cas": "134-96-3",
        "rt": 29.50,
        "mw": 182.17,
        "class": "S_Lignin",
        "subclass": "S_carbonyl",
        "formula": "C9H10O4",
    },
    {
        "name_en": "Acetosyringone",
        "name_ru": "Ацетосирингон",
        "cas": "2478-38-8",
        "rt": 31.20,
        "mw": 196.20,
        "class": "S_Lignin",
        "subclass": "S_carbonyl",
        "formula": "C10H12O4",
    },

    # --- 4. Carbohydrate (Углеводы и полисахариды: целлюлоза/гемицеллюлозы) ---
    {
        "name_en": "Furfural (2-Furaldehyde)",
        "name_ru": "Фурфурол (2-Фуральдегид)",
        "cas": "98-01-1",
        "rt": 5.40,
        "mw": 96.08,
        "class": "Carbohydrate",
        "subclass": "Furan",
        "formula": "C5H4O2",
    },
    {
        "name_en": "2-Furanmethanol (Furfuryl alcohol)",
        "name_ru": "Фурфуриловый спирт",
        "cas": "98-00-0",
        "rt": 6.10,
        "mw": 98.10,
        "class": "Carbohydrate",
        "subclass": "Furan",
        "formula": "C5H6O2",
    },
    {
        "name_en": "5-Methylfurfural",
        "name_ru": "5-Метилфурфурол",
        "cas": "620-02-0",
        "rt": 8.30,
        "mw": 110.11,
        "class": "Carbohydrate",
        "subclass": "Furan",
        "formula": "C6H6O2",
    },
    {
        "name_en": "Levoglucosenone",
        "name_ru": "Левоглюкозенон",
        "cas": "37112-31-5",
        "rt": 11.00,
        "mw": 126.11,
        "class": "Carbohydrate",
        "subclass": "Anhydrosugar",
        "formula": "C6H6O3",
    },
    {
        "name_en": "Levoglucosan (1,6-Anhydro-beta-D-glucopyranose)",
        "name_ru": "Левоглюкозан",
        "cas": "498-07-7",
        "rt": 23.50,
        "mw": 162.14,
        "class": "Carbohydrate",
        "subclass": "Anhydrosugar",
        "formula": "C6H10O5",
    },

    # --- 5. Nitrogen & Protein (Азотсодержащие и белковые фрагменты) ---
    {
        "name_en": "Pyridine",
        "name_ru": "Пиридин",
        "cas": "110-86-1",
        "rt": 4.20,
        "mw": 79.10,
        "class": "Nitrogen_Protein",
        "subclass": "Aromatic_N",
        "formula": "C5H5N",
    },
    {
        "name_en": "Pyrrole",
        "name_ru": "Пиррол",
        "cas": "109-97-7",
        "rt": 4.80,
        "mw": 67.09,
        "class": "Nitrogen_Protein",
        "subclass": "Aromatic_N",
        "formula": "C4H5N",
    },
    {
        "name_en": "Indole",
        "name_ru": "Индол",
        "cas": "120-72-9",
        "rt": 19.00,
        "mw": 117.15,
        "class": "Nitrogen_Protein",
        "subclass": "Indole",
        "formula": "C8H7N",
    },
    {
        "name_en": "3-Methylindole (Skatole)",
        "name_ru": "3-Метилиндол (Скатол)",
        "cas": "83-34-1",
        "rt": 21.50,
        "mw": 131.17,
        "class": "Nitrogen_Protein",
        "subclass": "Indole",
        "formula": "C9H9N",
    },
]


# ==============================================================================
# КОНТЕЙНЕРЫ ДАННЫХ (DATACLASSES)
# ==============================================================================

@dataclass
class PyGCMSPeak:
    """Пик хроматограммы пиролиза Py-GC-MS."""
    retention_time: float
    area: float
    height: float
    compound_name: str
    chemical_class: str  # "H_Lignin", "G_Lignin", "S_Lignin", "Carbohydrate", "Nitrogen_Protein", "Aliphatic_Lipid", "Unclassified"
    cas: Optional[str] = None
    formula: Optional[str] = None
    similarity: float = float("nan")      # псевдоним rt_match_score (НЕ спектральное сходство)
    mw: float = 0.0
    area_pct: float = 0.0
    monomer_type: str = ""
    name_en: str = ""
    name_ru: str = ""
    rt_match_score: float = float("nan")  # 1 − |ΔRT|/window; NaN, если пик идентифицирован по названию
    id_basis: str = ""                    # "name" | "retention_time" | "synthetic_library" | "none"

    def __post_init__(self):
        if not self.name_en:
            self.name_en = self.compound_name
        if not self.name_ru:
            self.name_ru = self.compound_name
        if not self.monomer_type:
            self.monomer_type = self.chemical_class
        # Синхронизация устаревшего поля similarity с rt_match_score
        sim_ok = self.similarity is not None and np.isfinite(self.similarity)
        rt_ok = self.rt_match_score is not None and np.isfinite(self.rt_match_score)
        if rt_ok and not sim_ok:
            self.similarity = float(self.rt_match_score)
        elif sim_ok and not rt_ok:
            self.rt_match_score = float(self.similarity)

    @property
    def rt(self) -> float:
        return self.retention_time

    @rt.setter
    def rt(self, val: float):
        self.retention_time = val

    @property
    def peak_area(self) -> float:
        return self.area

    @peak_area.setter
    def peak_area(self, val: float):
        self.area = val

    @property
    def compound_class(self) -> str:
        mapping = {
            "G_Lignin": "Guaiacyl",
            "S_Lignin": "Syringyl",
            "H_Lignin": "p-Hydroxyphenyl",
            "Carbohydrate": "Carbohydrate",
            "Nitrogen_Protein": "Nitrogen",
            "Aliphatic_Lipid": "Aliphatic",
            "Guaiacyl": "Guaiacyl",
            "Syringyl": "Syringyl",
            "p-Hydroxyphenyl": "p-Hydroxyphenyl",
            "Nitrogen": "Nitrogen",
            "Aliphatic": "Aliphatic",
        }
        return mapping.get(self.chemical_class, "Other")

    @compound_class.setter
    def compound_class(self, val: str):
        self.chemical_class = val


@dataclass
class PyGCMSSample:
    """Контейнер для хроматограммы пиролиза Py-GC-MS."""
    sample_id: str
    retention_times: np.ndarray = field(default_factory=lambda: np.array([]))
    tic_intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    peaks: List[PyGCMSPeak] = field(default_factory=list)
    indices: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TGASample:
    """Контейнер для термогравиметрического профиля (TGA / DTG / DSC)."""
    sample_id: str
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))  # °C (по возрастанию)
    mass_pct: np.ndarray = field(default_factory=lambda: np.array([]))     # % от начальной массы
    raw_mass_mg: Optional[np.ndarray] = None
    dtg: np.ndarray = field(default_factory=lambda: np.array([]))          # -dm/dT (%/°C)
    indices: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalSample:
    """Комплексный контейнер термического анализа (Py-GC-MS + TGA)."""
    sample_id: str
    py_gcms: Optional[PyGCMSSample] = None
    tga: Optional[TGASample] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. ПАРСИНГ ФАЙЛОВ Py-GC-MS
# ==============================================================================

# Упорядоченная таблица правил классификации продуктов пиролиза по названию.
# Порядок важен: S (2,6-диметокси) проверяется раньше G (2-метокси), G раньше H (фенолы),
# затем углеводы и азотсодержащие соединения. Так «2,6-диметоксифенол» не попадает в H,
# «ацетованилон» не считается ванилином, «сиреневая кислота» — S, а не H.
PYROLYSIS_CLASS_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("S_Lignin", (
        "syring", "сиринг", "сирен", "2,6-dimethoxy", "2,6-диметокси", "3,5-dimethoxy", "3,5-диметокси",
        "3,4,5-trimethoxy", "3,4,5-триметокси", "trimethoxybenz", "триметоксибенз", "sinapyl", "синапов",
        "sinapic", "синапин",
    )),
    ("G_Lignin", (
        "guaiac", "гваяк", "гуаяк", "vanill", "ванил", "creosol", "креозол", "eugenol", "эвгенол", "эугенол",
        "coniferyl", "конифер", "ferulic", "ферулов", "2-methoxyphenol", "2-метоксифенол", "apocynin", "апоцинин",
        "3,4-dimethoxy", "3,4-диметокси", "methoxy-4-", "метокси-4-",
    )),
    ("H_Lignin", (
        "phenol", "фенол", "cresol", "крезол", "coumar", "кумар", "4-hydroxybenz", "4-гидроксибенз",
        "hydroxyphenyl", "гидроксифенил", "4-methoxystyrene", "4-метоксистирол",
    )),
    ("Carbohydrate", (
        "furan", "фуран", "furfur", "фурфур", "levoglucos", "левоглюкоз", "anhydro", "ангидро", "pyran", "пиран",
        "maltol", "мальтол", "cyclopenten", "циклопентен", "hydroxyacetaldehyde", "гидроксиацетальдегид",
        "glucopyranose", "глюкопираноз", "sugar", "сахар",
    )),
    ("Nitrogen_Protein", (
        "indol", "индол", "pyridin", "пиридин", "pyrrol", "пиррол", "pyrazin", "пиразин", "nitrile", "нитрил",
        "amide", "амид", "diketopiperazine", "дикетопиперазин", "skatole", "скатол", "imidazol", "имидазол",
        "piperidin", "пиперидин", "quinolin", "хинолин",
    )),
    ("Aliphatic_Lipid", (
        "alkane", "alkene", "алкан", "алкен", "fatty acid", "жирная кислота", "palmit", "пальмит", "stear", "стеарин",
        "hexadecan", "octadecan", "гексадекан", "октадекан", "n-alk", "н-алк",
    )),
]

# Карбонильные мономеры лигнина для (Ac/Al) и Ketone/Al. Кислота (и её метиловый эфир/
# TMAH-производное) проверяется раньше кетона, кетон — раньше альдегида.
_LIGNIN_CARBONYL_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("G_acid", ("vanillic acid", "methyl vanillate", "vanillate", "ванилиновая кислота", "ванилиновой кислоты",
                "метилванилат", "метил ванилат", "3,4-dimethoxybenzoic", "3,4-dimethoxybenzoate",
                "3,4-диметоксибензойн", "3,4-диметоксибензоат")),
    ("S_acid", ("syringic acid", "methyl syringate", "syringate", "сиреневая кислота", "сиреневой кислоты",
                "сиринговая кислота", "метилсирингат", "метил сирингат", "3,4,5-trimethoxybenzoic",
                "3,4,5-trimethoxybenzoate", "3,4,5-триметоксибензойн", "3,4,5-триметоксибензоат")),
    ("G_ketone", ("acetovanillone", "apocynin", "ацетованилон", "апоцинин", "acetoguaiacone",
                  "3,4-dimethoxyacetophenone", "3,4-диметоксиацетофенон")),
    ("S_ketone", ("acetosyringone", "ацетосирингон", "3,4,5-trimethoxyacetophenone", "3,4,5-триметоксиацетофенон")),
    ("G_aldehyde", ("vanillin", "ванилин", "3,4-dimethoxybenzaldehyde", "3,4-диметоксибензальдегид")),
    ("S_aldehyde", ("syringaldehyde", "сирингальдегид", "сиреневый альдегид", "3,4,5-trimethoxybenzaldehyde",
                    "3,4,5-триметоксибензальдегид")),
]
_CARBONYL_EXCLUDE = ("homovanill", "гомованил", "isovanill", "изованил", "vanillyl alcohol", "ванилиловый спирт")


def classify_pyrolysis_compound(name: str) -> str:
    """
    Классифицирует продукт пиролиза по названию (EN/RU) упорядоченной таблицей правил
    PYROLYSIS_CLASS_RULES (S → G → H → углеводы → N-соединения → алифатика).
    Возвращает "S_Lignin", "G_Lignin", "H_Lignin", "Carbohydrate", "Nitrogen_Protein",
    "Aliphatic_Lipid" или "Unclassified".
    """
    low = str(name or "").lower().replace("ё", "е")
    for cls, keys in PYROLYSIS_CLASS_RULES:
        if any(k in low for k in keys):
            return cls
    return "Unclassified"


def _identify_lignin_carbonyl(name: str) -> Optional[str]:
    """'G_acid' | 'G_ketone' | 'G_aldehyde' | 'S_acid' | 'S_ketone' | 'S_aldehyde' | None."""
    low = str(name or "").lower().replace("ё", "е")
    if any(k in low for k in _CARBONYL_EXCLUDE):
        return None
    for tag, keys in _LIGNIN_CARBONYL_RULES:
        if any(k in low for k in keys):
            return tag
    return None


def _library_name_keys(ref: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for nm in (ref.get("name_en", ""), ref.get("name_ru", "")):
        low = str(nm).lower().replace("ё", "е").strip()
        if not low:
            continue
        keys.append(low)
        base = re.sub(r"\s*\(.*?\)\s*", " ", low).strip()
        if base:
            keys.append(base)
        keys.extend(x.strip() for x in re.findall(r"\((.*?)\)", low) if x.strip())
    return keys


def lookup_library_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Точное (без учета регистра и скобочных синонимов) совпадение названия с библиотекой."""
    low = str(name or "").lower().replace("ё", "е").strip()
    if not low:
        return None
    for ref in PY_GCMS_REFERENCE_LIBRARY:
        if low in _library_name_keys(ref):
            return ref
    return None


def match_py_compound(
    rt: float,
    window: float = 0.6,
    lang: str = "en"
) -> Dict[str, Any]:
    """
    Ищет соединение в эталонной библиотеке продуктов пиролиза по времени удерживания (RT).
    Возвращает наилучшее совпадение по минимальному отклонению |rt - ref_rt|.

    ``RT_match_score`` = 1 − |ΔRT| / window (1 — точное совпадение RT, 0 — край окна).
    Это оценка близости времени удерживания, а НЕ спектральное сходство масс-спектров;
    ключ "similarity" сохранён как псевдоним для обратной совместимости.
    """
    best_match = None
    min_diff = float("inf")
    for ref in PY_GCMS_REFERENCE_LIBRARY:
        diff = abs(ref["rt"] - rt)
        if diff <= window and diff < min_diff:
            min_diff = diff
            best_match = ref

    if best_match is not None:
        name = best_match["name_ru"] if lang == "ru" else best_match["name_en"]
        score = float(max(0.0, 1.0 - (min_diff / window))) if window > 0 else float(min_diff == 0.0)
        return {
            "name": name,
            "name_en": best_match["name_en"],
            "name_ru": best_match["name_ru"],
            "class": best_match["class"],
            "subclass": best_match.get("subclass", ""),
            "cas": best_match["cas"],
            "formula": best_match["formula"],
            "mw": float(best_match.get("mw", 0.0)),
            "RT_match_score": score,
            "similarity": score,
            "rt_deviation_min": float(min_diff),
            "match_basis": "retention_time",
        }
    return _unmatched_compound(rt, lang)


def _unmatched_compound(rt: float, lang: str = "en") -> Dict[str, Any]:
    """Запись «неизвестное соединение» в формате match_py_compound."""
    return {
        "name": f"Unknown @ {rt:.2f} min" if lang == "en" else f"Неизвестный @ {rt:.2f} мин",
        "name_en": f"Unknown @ {rt:.2f} min",
        "name_ru": f"Неизвестный @ {rt:.2f} мин",
        "class": "Unclassified",
        "subclass": "Unclassified",
        "cas": None,
        "formula": None,
        "mw": float("nan"),
        "RT_match_score": 0.0,
        "similarity": 0.0,
        "rt_deviation_min": float("nan"),
        "match_basis": "none",
    }


def _detect_tic_peaks(
    rt: np.ndarray,
    tic: np.ndarray,
    snr_k: float = 3.0,
    baseline_window_min: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """
    Пики хроматограммы TIC: (RT, площадь над базовой линией, высота над базовой линией).

    Базовая линия — скользящая медиана в окне baseline_window_min (узкие пики на нее почти
    не влияют); порог выделения — snr_k·σ шума (σ по MAD первых разностей). Прежний порог
    «2 % от максимума» был ~1σ шума и превращал шумовые бугры в пики, а интегрирование без
    вычета базовой линии давало каждому такому бугру площадь «пол × ширина окна».
    Пик должен превышать базовую линию на snr_k·σ и по высоте, и по проминентности
    (у шумовых максимумов проминентность относительно соседних минимумов часто 4–6σ).
    Площадь — интеграл скорректированного сигнала в ±1.5·FWHM вокруг вершины.
    """
    rt = np.asarray(rt, dtype=float)
    y = np.asarray(tic, dtype=float)
    if len(y) < 10 or not np.all(np.isfinite(y)):
        return []
    step = float(np.median(np.diff(rt)))
    if step <= 0:
        return []
    window = max(5, int(round(baseline_window_min / step)) | 1)
    corrected = y - scipy.ndimage.median_filter(y, size=window, mode="nearest")
    d = np.diff(corrected)
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
    threshold = max(snr_k * sigma, 1e-12)
    peak_idx, _ = scipy.signal.find_peaks(corrected, height=threshold, prominence=threshold)
    if len(peak_idx) == 0:
        return []
    fwhm = scipy.signal.peak_widths(corrected, peak_idx, rel_height=0.5)[0]
    peaks = []
    for i, width in zip(peak_idx, fwhm):
        half = max(2, int(math.ceil(1.5 * width)))
        lo, hi = max(0, i - half), min(len(y) - 1, i + half)
        area = float(scipy.integrate.trapezoid(np.clip(corrected[lo:hi + 1], 0.0, None), rt[lo:hi + 1]))
        peaks.append((float(rt[i]), area, float(corrected[i])))
    return peaks


def _enforce_unique_rt_matches(peaks: List["PyGCMSPeak"], lang: str = "en") -> None:
    """
    Одно соединение библиотеки — не более одного пика: остается пик с наименьшим |ΔRT|
    (при равенстве — с большей площадью), остальные становятся «неизвестными». Иначе одно и то же
    соединение (напр., сиреневый альдегид) засчитывалось по 3–4 соседним пикам в окне RT.
    """
    best: Dict[str, Tuple[Tuple[float, float], int]] = {}
    for k, p in enumerate(peaks):
        if p.id_basis != "retention_time":
            continue
        key = p.name_en
        rank = (1.0 - float(p.rt_match_score), -float(p.area))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, k)
    keep = {k for _, k in best.values()}
    for k, p in enumerate(peaks):
        if p.id_basis != "retention_time" or k in keep:
            continue
        info = _unmatched_compound(p.retention_time, lang)
        p.compound_name = info["name"]
        p.name_en = info["name_en"]
        p.name_ru = info["name_ru"]
        p.chemical_class = info["class"]
        p.monomer_type = info["subclass"]
        p.cas = None
        p.formula = None
        p.mw = float("nan")
        p.similarity = 0.0
        p.rt_match_score = 0.0
        p.id_basis = "none"


def parse_py_gcms_file(
    file_content: Union[str, bytes],
    sample_id: str = "PyGCMS_Sample",
    rt_window: float = 0.6,
    lang: str = "en"
) -> PyGCMSSample:
    """
    Универсальный парсер хроматографических данных Py-GC-MS:
    1. Распознает сырые хроматограммы TIC (RT, Intensity) и выполняет автопоиск пиков.
    2. Распознает таблицы интегрированных пиков (RT, Area, Name/Class).
    3. Автоматически сопоставляет пики с библиотекой лигнина/углеводов.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return PyGCMSSample(sample_id=sample_id, indices=calculate_py_gcms_indices([]))

    # Определяем наличие разделителя
    header_line = lines[0]
    sep = "\t" if "\t" in header_line else (";" if ";" in header_line else ",")

    # Пробуем загрузить через pandas
    df_raw = None
    try:
        df_raw = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        try:
            df_raw = pd.read_csv(io.StringIO(text), sep=r"\s+")
        except Exception:
            pass

    if df_raw is None or df_raw.empty:
        return PyGCMSSample(sample_id=sample_id, indices=calculate_py_gcms_indices([]))

    # Приводим названия колонок к нижнему регистру для эвристики
    col_map = {c: str(c).strip().lower() for c in df_raw.columns}

    # Ищем колонку Retention Time
    rt_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["rt", "retention", "time", "мин", "время", "r.t."]):
            rt_col = orig
            break

    # Ищем колонку Area / Интенсивность
    area_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["area", "площадь", "peak_area", "integrated"]):
            area_col = orig
            break

    int_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["intensity", "tic", "abundance", "counts", "интенсивность"]):
            int_col = orig
            break

    name_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["name", "compound", "название", "пик", "вещество"]):
            name_col = orig
            break

    # Если колонка RT не найдена по имени, берем первую числовую
    if rt_col is None:
        numeric_cols = [c for c in df_raw.columns if pd.to_numeric(df_raw[c], errors="coerce").notnull().sum() > len(df_raw) * 0.5]
        if numeric_cols:
            rt_col = numeric_cols[0]
            if len(numeric_cols) > 1 and int_col is None and area_col is None:
                int_col = numeric_cols[1]

    if rt_col is None:
        return PyGCMSSample(sample_id=sample_id, indices=calculate_py_gcms_indices([]))

    # Очищаем данные
    df_clean = df_raw.copy()
    df_clean[rt_col] = pd.to_numeric(df_clean[rt_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[rt_col]).sort_values(by=rt_col).reset_index(drop=True)

    rt_vals = df_clean[rt_col].values.astype(float)
    peaks_list: List[PyGCMSPeak] = []
    tic_vals = np.array([])

    # ВАРИАНТ А: Файл содержит сырую хроматограмму TIC (большое число точек, например > 200)
    if int_col is not None and len(df_clean) > 100:
        df_clean[int_col] = pd.to_numeric(df_clean[int_col], errors="coerce").fillna(0.0)
        tic_vals = df_clean[int_col].values.astype(float)

        for p_rt, p_area, p_height in _detect_tic_peaks(rt_vals, tic_vals):
            match_info = match_py_compound(p_rt, window=rt_window, lang=lang)
            peaks_list.append(PyGCMSPeak(
                retention_time=round(p_rt, 3),
                area=max(0.0, p_area),
                height=max(0.0, p_height),
                compound_name=match_info["name"],
                chemical_class=match_info["class"],
                cas=match_info["cas"],
                formula=match_info["formula"],
                similarity=match_info["RT_match_score"],
                rt_match_score=match_info["RT_match_score"],
                id_basis=match_info["match_basis"],
                mw=match_info.get("mw", float("nan")),
                monomer_type=match_info.get("subclass", match_info["class"]),
                name_en=match_info.get("name_en", match_info["name"]),
                name_ru=match_info.get("name_ru", match_info["name"]),
            ))

    # ВАРИАНТ Б: Файл уже представляет собой таблицу интегрированных пиков
    else:
        val_col = area_col if area_col is not None else int_col
        for _, row in df_clean.iterrows():
            p_rt = float(row[rt_col])
            p_area = float(pd.to_numeric(row[val_col], errors="coerce")) if val_col is not None else 1.0
            p_height = p_area

            # Если название уже есть в файле, используем его
            c_name = str(row[name_col]).strip() if (name_col is not None and pd.notna(row[name_col])) else ""
            if c_name and c_name.lower() not in ["none", "nan", ""]:
                # Классификация по названию (упорядоченная таблица правил S → G → H → углеводы → N)
                chem_class = classify_pyrolysis_compound(c_name)
                # CAS/формула — только при совпадении НАЗВАНИЯ с библиотекой (не по близкому RT)
                lib_ref = lookup_library_by_name(c_name)
                peaks_list.append(PyGCMSPeak(
                    retention_time=round(p_rt, 3),
                    area=max(0.0, p_area),
                    height=max(0.0, p_height),
                    compound_name=c_name,
                    chemical_class=chem_class,
                    cas=lib_ref.get("cas") if lib_ref else None,
                    formula=lib_ref.get("formula") if lib_ref else None,
                    similarity=float("nan"),
                    rt_match_score=float("nan"),
                    id_basis="name",
                    mw=float(lib_ref.get("mw", float("nan"))) if lib_ref else float("nan"),
                    monomer_type=(lib_ref.get("subclass", chem_class) if lib_ref else chem_class),
                    name_en=c_name,
                    name_ru=c_name,
                ))
            else:
                match_info = match_py_compound(p_rt, window=rt_window, lang=lang)
                peaks_list.append(PyGCMSPeak(
                    retention_time=round(p_rt, 3),
                    area=max(0.0, p_area),
                    height=max(0.0, p_height),
                    compound_name=match_info["name"],
                    chemical_class=match_info["class"],
                    cas=match_info["cas"],
                    formula=match_info["formula"],
                    similarity=match_info["RT_match_score"],
                    rt_match_score=match_info["RT_match_score"],
                    id_basis=match_info["match_basis"],
                    mw=match_info.get("mw", float("nan")),
                    monomer_type=match_info.get("subclass", match_info["class"]),
                    name_en=match_info.get("name_en", match_info["name"]),
                    name_ru=match_info.get("name_ru", match_info["name"]),
                ))

    _enforce_unique_rt_matches(peaks_list, lang)

    # Рассчитываем относительные площади (area_pct)
    tot_area = sum(p.area for p in peaks_list)
    for p in peaks_list:
        p.area_pct = round((p.area / tot_area) * 100.0, 2) if tot_area > 0 else 0.0

    # Рассчитываем хемометрические индексы
    indices = calculate_py_gcms_indices(peaks_list)

    return PyGCMSSample(
        sample_id=sample_id,
        retention_times=rt_vals,
        tic_intensity=tic_vals,
        peaks=peaks_list,
        indices=indices,
        metadata={"n_peaks": len(peaks_list)},
    )


# ==============================================================================
# 2. РАСЧЕТ ХЕМОМЕТРИЧЕСКИХ ИНДЕКСОВ Py-GC-MS
# ==============================================================================

PY_GCMS_INDEX_KEYS: Tuple[str, ...] = (
    "Py_H_pct", "Py_G_pct", "Py_S_pct",
    "Py_S_G_ratio", "Py_H_G_ratio", "Py_L_C_ratio",
    "Py_Ac_Al_G", "Py_Ac_Al_S",
    "Py_Ketone_Aldehyde_G", "Py_Ketone_Aldehyde_S",
    "Py_Total_Lignin_pct", "Py_Carbohydrate_pct", "Py_Nitrogen_pct", "Py_Aromaticity_pct",
    "Py_Acid_Aldehyde_G", "Py_Acid_Aldehyde_S",
    "Py_Total_Lignin_Area_pct", "Py_Carbohydrate_Area_pct", "Py_Nitrogen_Area_pct", "Py_Aromaticity_Area_pct",
)


def _ratio(num: float, den: float, nd: int = 3) -> float:
    """num/den, NaN if the denominator (or numerator) is absent — never a 1e-6 floor."""
    if den is None or num is None or not np.isfinite(den) or not np.isfinite(num) or den <= 0:
        return float("nan")
    return round(float(num / den), nd)


def calculate_py_gcms_indices(peaks: List[PyGCMSPeak]) -> Dict[str, float]:
    """
    Рассчитывает ключевые хемометрические параметры состава пиролизатов:
    - S/G, H/G отношения мономеров лигнина;
    - Доли мономеров H%, G%, S%;
    - Соотношение лигнин/углеводы (L/C);
    - Окислительно-деградационные индексы (Ac/Al)_G = ванилиновая кислота / ванилин и
      (Ac/Al)_S = сиреневая кислота / сиреневый альдегид (Hedges & Ertel 1982, CuO;
      для TMAH-термохемолиза — метиловые эфиры/метилированные производные, G6/G4 и S6/S4:
      Challinor 1995; Clifford et al. 1995). Если кислота или альдегид не обнаружены — NaN.
      Py_Acid_Aldehyde_G/S — псевдонимы (Ac/Al); отношение кетон/альдегид
      (ацетованилон/ванилин, ацетосирингон/сирингальдегид) — Py_Ketone_Aldehyde_G/S;
    - Суммарную степень ароматичности продуктов пиролиза.
    Отношения с нулевым знаменателем и пустой набор пиков дают NaN.
    """
    if not peaks:
        return {k: float("nan") for k in PY_GCMS_INDEX_KEYS}

    total_area = float(sum(p.area for p in peaks))

    area_h = sum(p.area for p in peaks if p.chemical_class == "H_Lignin")
    area_g = sum(p.area for p in peaks if p.chemical_class == "G_Lignin")
    area_s = sum(p.area for p in peaks if p.chemical_class == "S_Lignin")
    area_carb = sum(p.area for p in peaks if p.chemical_class == "Carbohydrate")
    area_n = sum(p.area for p in peaks if p.chemical_class == "Nitrogen_Protein")
    area_lignin = area_h + area_g + area_s

    # Пропорции внутри лигнина
    h_pct = _ratio(area_h * 100.0, area_lignin, 2)
    g_pct = _ratio(area_g * 100.0, area_lignin, 2)
    s_pct = _ratio(area_s * 100.0, area_lignin, 2)

    # Отношения S/G, H/G и Лигнин / Углеводы (L/C)
    s_g_ratio = _ratio(area_s, area_g)
    h_g_ratio = _ratio(area_h, area_g)
    l_c_ratio = _ratio(area_lignin, area_carb)

    # Карбонильные мономеры (кислота / кетон / альдегид) по названию соединения
    carbonyl: Dict[str, float] = {}
    for p in peaks:
        tag = None
        for nm in (p.compound_name, getattr(p, "name_en", ""), getattr(p, "name_ru", "")):
            tag = _identify_lignin_carbonyl(nm)
            if tag is not None:
                break
        if tag is not None:
            carbonyl[tag] = carbonyl.get(tag, 0.0) + float(p.area)

    def _carb_ratio(num_tag: str, den_tag: str) -> float:
        num, den = carbonyl.get(num_tag, 0.0), carbonyl.get(den_tag, 0.0)
        if num <= 0 or den <= 0:
            return float("nan")
        return round(float(num / den), 3)

    ac_al_g = _carb_ratio("G_acid", "G_aldehyde")
    ac_al_s = _carb_ratio("S_acid", "S_aldehyde")
    ket_al_g = _carb_ratio("G_ketone", "G_aldehyde")
    ket_al_s = _carb_ratio("S_ketone", "S_aldehyde")

    # Общие доли от всей смеси
    total_lignin_pct = _ratio(area_lignin * 100.0, total_area, 2)
    carb_pct = _ratio(area_carb * 100.0, total_area, 2)
    n_pct = _ratio(area_n * 100.0, total_area, 2)
    aromatic_pct = _ratio((area_lignin + area_n) * 100.0, total_area, 2)

    return {
        "Py_H_pct": h_pct,
        "Py_G_pct": g_pct,
        "Py_S_pct": s_pct,
        "Py_S_G_ratio": s_g_ratio,
        "Py_H_G_ratio": h_g_ratio,
        "Py_L_C_ratio": l_c_ratio,
        "Py_Ac_Al_G": ac_al_g,
        "Py_Ac_Al_S": ac_al_s,
        "Py_Ketone_Aldehyde_G": ket_al_g,
        "Py_Ketone_Aldehyde_S": ket_al_s,
        "Py_Total_Lignin_pct": total_lignin_pct,
        "Py_Carbohydrate_pct": carb_pct,
        "Py_Nitrogen_pct": n_pct,
        "Py_Aromaticity_pct": aromatic_pct,
        "Py_Acid_Aldehyde_G": ac_al_g,
        "Py_Acid_Aldehyde_S": ac_al_s,
        "Py_Total_Lignin_Area_pct": total_lignin_pct,
        "Py_Carbohydrate_Area_pct": carb_pct,
        "Py_Nitrogen_Area_pct": n_pct,
        "Py_Aromaticity_Area_pct": aromatic_pct,
    }


# ==============================================================================
# 3. ПАРСИНГ И АНАЛИЗ ТЕРМОГРАВИМЕТРИИ (TGA / DTG / DSC)
# ==============================================================================

def parse_tga_file(
    file_content: Union[str, bytes],
    sample_id: str = "TGA_Sample"
) -> TGASample:
    """
    Парсер данных термогравиметрии (Netzsch, TA Instruments, Mettler, PerkinElmer).
    Поддерживает текстовые экспорты с метаданными и автоматическим поиском
    столбцов температуры (°C) и массы (% или мг).
    Рассчитывает кривую дифференциальной термогравиметрии DTG = -dm/dT.
    """
    text = decode_bytes(file_content) if isinstance(file_content, bytes) else file_content
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return TGASample(sample_id=sample_id, indices=calculate_tga_indices(np.array([]), np.array([])))

    # Ищем начало числовых данных (строка, где начинается число или разделители)
    start_line_idx = 0
    sep = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")

    # Пробуем определить строку заголовка с температурой/массой
    for i, line in enumerate(lines[:50]):
        low = line.lower()
        if any(k in low for k in ["temp", "темп", "°c", "celsius"]) and any(k in low for k in ["mass", "масс", "weight", "вес", "%", "mg"]):
            start_line_idx = i
            if "\t" in line: sep = "\t"
            elif ";" in line: sep = ";"
            elif "," in line: sep = ","
            break

    data_text = "\n".join(lines[start_line_idx:])
    df_raw = None
    try:
        df_raw = pd.read_csv(io.StringIO(data_text), sep=sep)
    except Exception:
        try:
            df_raw = pd.read_csv(io.StringIO(data_text), sep=r"\s+")
        except Exception:
            pass

    if df_raw is None or df_raw.empty:
        return TGASample(sample_id=sample_id, indices=calculate_tga_indices(np.array([]), np.array([])))

    col_map = {c: str(c).strip().lower() for c in df_raw.columns}

    # Ищем колонку температуры
    t_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["temp", "темп", "t (", "t[", "°c"]):
            t_col = orig
            break

    # Ищем колонку массы
    m_col = None
    for orig, low in col_map.items():
        if any(k in low for k in ["mass", "масс", "weight", "вес", "tg", "tga"]):
            m_col = orig
            break

    if t_col is None or m_col is None:
        # Резервный поиск по числовым столбцам
        num_cols = [c for c in df_raw.columns if pd.to_numeric(df_raw[c], errors="coerce").notnull().sum() > len(df_raw) * 0.5]
        if len(num_cols) >= 2:
            t_col, m_col = num_cols[0], num_cols[1]
        else:
            return TGASample(sample_id=sample_id, indices=calculate_tga_indices(np.array([]), np.array([])))

    df_clean = df_raw.copy()
    df_clean[t_col] = pd.to_numeric(df_clean[t_col], errors="coerce")
    df_clean[m_col] = pd.to_numeric(df_clean[m_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[t_col, m_col]).sort_values(by=t_col).reset_index(drop=True)

    t_vals = df_clean[t_col].values.astype(float)
    m_vals = df_clean[m_col].values.astype(float)

    # Если масса представлена в мг или абсолютных величинах, нормируем к 100%
    # Начальная масса при комнатной температуре (T <= 40 °C)
    initial_idx = np.where(t_vals <= 45.0)[0]
    base_m = np.median(m_vals[initial_idx]) if len(initial_idx) > 0 else m_vals[0]
    if base_m > 0 and (np.max(m_vals) > 105.0 or np.max(m_vals) < 50.0):
        m_pct = (m_vals / base_m) * 100.0
    else:
        m_pct = m_vals

    # Расчет DTG = -dm/dT (%/°C) со сглаживанием Савицкого — Голея
    t_safe = t_vals.copy()
    if len(t_safe) > 1:
        diffs = np.diff(t_safe)
        if np.any(diffs <= 0):
            t_safe = np.maximum.accumulate(t_safe) + np.arange(len(t_safe)) * 1e-4

    # Знак DTG сохраняется: отрицательные значения означают прирост массы (окисление,
    # адсорбция, дрейф плавучести) и не обрезаются до нуля.
    dtg = np.zeros_like(m_pct)
    if len(m_pct) > 15:
        window = min(21, len(m_pct) if len(m_pct) % 2 != 0 else len(m_pct) - 1)
        if window >= 5:
            smooth_m = scipy.signal.savgol_filter(m_pct, window_length=window, polyorder=2)
            dtg = -np.gradient(smooth_m, t_safe)
    elif len(m_pct) > 1:
        dtg = -np.gradient(m_pct, t_safe)

    # Расчет индексов термогравиметрии
    indices = calculate_tga_indices(t_vals, m_pct, dtg)

    return TGASample(
        sample_id=sample_id,
        temperature=t_vals,
        mass_pct=m_pct,
        raw_mass_mg=m_vals if np.max(m_vals) != np.max(m_pct) else None,
        dtg=dtg,
        indices=indices,
        metadata={"t_min": float(np.min(t_vals)), "t_max": float(np.max(t_vals)), "t_final": float(np.max(t_vals))},
    )


TGA_INDEX_KEYS: Tuple[str, ...] = (
    "TGA_Moisture_Loss_pct", "TGA_Aliphatic_Loss_pct", "TGA_Aromatic_Loss_pct",
    "TGA_Char_Yield_600C", "TGA_Char_Yield_800C", "TGA_Thermostability_Index",
    "TGA_T_5pct", "TGA_T_10pct", "TGA_T_50pct",
    "DTG_T_max1", "DTG_T_max2", "DTG_Max_Rate",
    "TGA_Zone_I_pct", "TGA_Zone_II_pct", "TGA_Zone_III_pct", "TGA_Zone_IV_pct",
    "TGA_Thermal_Index_TI", "TGA_T5_degC", "TGA_T10_degC", "TGA_T50_degC",
    "TGA_Residue_800C_pct", "DTG_Tmax_degC",
    "TGA_Residue_Final_pct", "TGA_T_final_degC",
)


def _interp_crossing_temperature(
    temperature: np.ndarray, mass_pct: np.ndarray, target: float, t_from: float = 150.0
) -> float:
    """
    Первая температура (T ≥ t_from), при которой масса опускается до ``target``,
    с линейной интерполяцией между соседними точками. NaN, если уровень не достигнут.
    """
    if not np.isfinite(target):
        return float("nan")
    idx = np.where((temperature >= t_from) & (mass_pct <= target))[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0 or temperature[i - 1] < t_from or mass_pct[i - 1] == mass_pct[i]:
        return float(temperature[i])
    t0, t1 = float(temperature[i - 1]), float(temperature[i])
    m0, m1 = float(mass_pct[i - 1]), float(mass_pct[i])
    return float(t0 + (target - m0) * (t1 - t0) / (m1 - m0))


def calculate_tga_indices(
    temperature: np.ndarray,
    mass_pct: np.ndarray,
    dtg: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Рассчитывает стандартные термогравиметрические параметры:
    1. Потери массы по температурным интервалам:
       - Зона I (25–150 °C): влага и легколетучие соединения (Moisture loss);
       - Зона II (150–350 °C): термодеструкция алифатики/гемицеллюлозы/целлюлозы;
       - Зона III (350–550 °C): термодеструкция ароматического каркаса и лигнина;
       - Зона IV (TGA_Zone_IV_pct = TGA_Residue_Final_pct): остаток при конечной
         температуре опыта T_final (кокс + зола), а не «100 − сумма потерь».
       Коксовый остаток при 600 / 800 °C — только если эти температуры достигнуты.
    2. Индекс термостабильности TI = Δm_arom / Δm_aliph.
    3. Характеристические температуры T_5%, T_10%, T_50% (на сухую массу m(150 °C)),
       линейная интерполяция между точками; NaN, если уровень потери не достигнут.
    4. Максимумы скорости деструкции DTG (T_max1, T_max2) в диапазоне 180–600 °C.
    Величины, для которых данных нет (интервал не покрыт), возвращаются как NaN.
    """
    temperature = np.asarray(temperature, dtype=float)
    mass_pct = np.asarray(mass_pct, dtype=float)
    nan = float("nan")
    if len(temperature) == 0 or len(mass_pct) == 0 or len(temperature) != len(mass_pct):
        return {k: nan for k in TGA_INDEX_KEYS}

    t_lo, t_hi = float(np.min(temperature)), float(np.max(temperature))
    tol = 0.5  # °C

    def get_mass_at_t(target_t: float) -> float:
        """Линейная интерполяция; NaN вне измеренного диапазона температур."""
        if target_t < t_lo - tol or target_t > t_hi + tol:
            return nan
        if t_hi - t_lo < 1e-9:
            return float(mass_pct[0])
        return float(np.interp(target_t, temperature, mass_pct))

    # Начальная масса: первая точка, если опыт начат не выше 150 °C
    m_start = float(np.interp(max(25.0, t_lo), temperature, mass_pct)) if t_lo <= 150.0 and t_hi > t_lo else nan
    m_150 = get_mass_at_t(150.0)
    m_350 = get_mass_at_t(350.0)
    m_550 = get_mass_at_t(550.0)
    m_600 = get_mass_at_t(600.0)
    m_800 = get_mass_at_t(800.0)
    m_final = float(mass_pct[int(np.argmax(temperature))]) if t_hi > t_lo else nan

    # Потери массы в интервалах (знак сохраняется: отрицательное значение = прирост массы)
    dm_moisture = m_start - m_150
    dm_aliph = m_150 - m_350
    dm_arom = m_350 - m_550

    # Индекс термостабильности TI = Δm_arom / Δm_aliph
    ti = float(dm_arom / dm_aliph) if (np.isfinite(dm_arom) and np.isfinite(dm_aliph) and dm_aliph > 1e-4) else nan

    # Характеристические температуры потери 5%, 10%, 50% от сухого образца (m_150)
    dry_base = m_150
    t_5 = _interp_crossing_temperature(temperature, mass_pct, dry_base * 0.95)
    t_10 = _interp_crossing_temperature(temperature, mass_pct, dry_base * 0.90)
    t_50 = _interp_crossing_temperature(temperature, mass_pct, dry_base * 0.50)

    # Максимумы DTG (без обрезки отрицательных значений)
    t_max1, t_max2, max_rate = nan, nan, nan
    if dtg is not None and len(dtg) == len(temperature):
        dtg = np.asarray(dtg, dtype=float)
        valid_mask = (temperature >= 180.0) & (temperature <= 600.0) & np.isfinite(dtg)
        if np.count_nonzero(valid_mask) >= 3:
            sub_t = temperature[valid_mask]
            sub_dtg = dtg[valid_mask]
            top = float(np.max(sub_dtg))
            if top > 0:
                peaks_i, _ = scipy.signal.find_peaks(sub_dtg, prominence=top * 0.15)
                if len(peaks_i) > 0:
                    sorted_p = sorted(peaks_i, key=lambda pi: sub_dtg[pi], reverse=True)
                    t_max1 = float(sub_t[sorted_p[0]])
                    max_rate = float(sub_dtg[sorted_p[0]])
                    if len(sorted_p) > 1:
                        t_max2 = float(sub_t[sorted_p[1]])
                else:
                    max_idx = int(np.argmax(sub_dtg))
                    t_max1 = float(sub_t[max_idx])
                    max_rate = float(sub_dtg[max_idx])

    def _r(v: float, nd: int) -> float:
        return round(float(v), nd) if np.isfinite(v) else nan

    return {
        "TGA_Moisture_Loss_pct": _r(dm_moisture, 2),
        "TGA_Aliphatic_Loss_pct": _r(dm_aliph, 2),
        "TGA_Aromatic_Loss_pct": _r(dm_arom, 2),
        "TGA_Char_Yield_600C": _r(m_600, 2),
        "TGA_Char_Yield_800C": _r(m_800, 2),
        "TGA_Thermostability_Index": _r(ti, 3),
        "TGA_T_5pct": _r(t_5, 1),
        "TGA_T_10pct": _r(t_10, 1),
        "TGA_T_50pct": _r(t_50, 1),
        "DTG_T_max1": _r(t_max1, 1),
        "DTG_T_max2": _r(t_max2, 1),
        "DTG_Max_Rate": _r(max_rate, 4),
        "TGA_Zone_I_pct": _r(dm_moisture, 2),
        "TGA_Zone_II_pct": _r(dm_aliph, 2),
        "TGA_Zone_III_pct": _r(dm_arom, 2),
        "TGA_Zone_IV_pct": _r(m_final, 2),
        "TGA_Thermal_Index_TI": _r(ti, 3),
        "TGA_T5_degC": _r(t_5, 1),
        "TGA_T10_degC": _r(t_10, 1),
        "TGA_T50_degC": _r(t_50, 1),
        "TGA_Residue_800C_pct": _r(m_800, 2),
        "DTG_Tmax_degC": _r(t_max1, 1),
        "TGA_Residue_Final_pct": _r(m_final, 2),
        "TGA_T_final_degC": _r(t_hi, 1) if t_hi > t_lo else nan,
    }


# ==============================================================================
# 4. ИЗВЛЕЧЕНИЕ СВОДНЫХ ДЕСКРИПТОРОВ ДЛЯ DATA FUSION И ML
# ==============================================================================

def extract_thermal_descriptors(sample: Union[ThermalSample, PyGCMSSample, TGASample]) -> Dict[str, float]:
    """
    Извлекает единый вектор аналитических дескрипторов (Py-GC-MS + TGA)
    для интеграции в общий пул признаков ChemoSuite ML.
    """
    res: Dict[str, float] = {}

    if isinstance(sample, ThermalSample):
        if sample.py_gcms is not None:
            res.update(sample.py_gcms.indices)
        if sample.tga is not None:
            res.update(sample.tga.indices)
    elif isinstance(sample, PyGCMSSample):
        res.update(sample.indices)
    elif isinstance(sample, TGASample):
        res.update(sample.indices)

    return res


# ==============================================================================
# 5. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО БЕНЧМАРКА (12 ПРОБ: БАЙКАЛ vs ШЛАМ-ЛИГНИН)
# ==============================================================================

def generate_synthetic_thermal_dataset(
    n_samples_per_class: int = 6,
    random_state: int = 42,
    save_dir: Optional[str] = None
) -> Tuple[Dict[str, ThermalSample], pd.DataFrame]:
    """
    Генерирует согласованный термический бенчмарк (Py-GC-MS хроматограммы + TGA/DTG профили):
    - Класс 1: Фоновое органическое вещество оз. Байкал (Pristine NOM / Humic Acids)
      -> высокое содержание углеводов, умеренный S/G, пониженная термостабильность TI, влагоемкость;
    - Класс 2: Техногенный шлам-лигнин БЦБК (BPPM Sludge-Lignin)
      -> доминирование G-лигнина (хвойная крафт-варка), высокий L/C (> 3.5), высокий индекс TI (> 1.2),
         сдвиг пика термодеструкции DTG в область 390–430 °C, высокий коксовый остаток.
    """
    rng = np.random.RandomState(random_state)
    samples: Dict[str, ThermalSample] = {}
    matrix_rows: List[Dict[str, Any]] = []

    # Температурная сетка TGA (25 .. 850 °C, шаг 1 °C)
    t_grid = np.linspace(25.0, 850.0, 826)

    # Сетка времен удерживания Py-GC-MS (3.0 .. 35.0 мин, шаг 0.05 мин)
    rt_grid = np.linspace(3.0, 35.0, 641)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(1, n_samples_per_class + 1):
        # -------------------------------------------------------------
        # ОБРАЗЕЦ КЛАССА 1: Природное РОВ оз. Байкал (Pristine Baikal NOM)
        # -------------------------------------------------------------
        s_id_nom = f"Baikal_NOM_{i:02d}"

        # 1.1 TGA профиль
        # Потеря влаги: 6-10%, алифатика (150-350°C): 35-42%, ароматика (350-550°C): 25-30%
        w_moist = rng.uniform(7.0, 10.0)
        w_aliph = rng.uniform(36.0, 42.0)
        w_arom = rng.uniform(24.0, 29.0)
        char_rem = 100.0 - (w_moist + w_aliph + w_arom)

        # Моделируем сигмоидальные переходы массы m(T)
        m_tga_nom = 100.0 - w_moist / (1.0 + np.exp(-(t_grid - 90.0) / 18.0)) \
                          - w_aliph / (1.0 + np.exp(-(t_grid - 285.0) / 32.0)) \
                          - w_arom / (1.0 + np.exp(-(t_grid - 440.0) / 45.0))
        m_tga_nom += rng.normal(0, 0.05, size=len(t_grid))
        m_tga_nom = np.clip(m_tga_nom, char_rem - 2.0, 100.0)

        dtg_nom = -np.gradient(m_tga_nom, t_grid)
        tga_indices_nom = calculate_tga_indices(t_grid, m_tga_nom, dtg_nom)

        tga_obj_nom = TGASample(
            sample_id=s_id_nom,
            temperature=t_grid,
            mass_pct=m_tga_nom,
            dtg=dtg_nom,
            indices=tga_indices_nom
        )

        # 1.2 Py-GC-MS профиль
        # Природный гумус: сбалансированные H/G/S, много фуранов и углеводов
        peaks_nom: List[PyGCMSPeak] = []
        tic_nom = np.zeros_like(rt_grid) + rng.normal(50.0, 5.0, size=len(rt_grid))

        for ref in PY_GCMS_REFERENCE_LIBRARY:
            ref_rt = ref["rt"]
            ref_cls = ref["class"]

            # Задаем характерную площадь
            if ref_cls == "Carbohydrate":
                base_area = rng.uniform(400.0, 900.0)
            elif ref_cls == "H_Lignin":
                base_area = rng.uniform(180.0, 350.0)
            elif ref_cls == "G_Lignin":
                base_area = rng.uniform(250.0, 500.0)
            elif ref_cls == "S_Lignin":
                base_area = rng.uniform(200.0, 450.0)
            else:  # Nitrogen
                base_area = rng.uniform(150.0, 300.0)

            # Гауссов пик на TIC
            peak_shape = (base_area / 5.0) * np.exp(-((rt_grid - ref_rt) ** 2) / (2 * 0.06 ** 2))
            tic_nom += peak_shape

            peaks_nom.append(PyGCMSPeak(
                retention_time=ref_rt,
                area=round(float(base_area), 2),
                height=round(float(base_area / 5.0), 2),
                compound_name=ref["name_en"],
                chemical_class=ref["class"],
                cas=ref["cas"],
                formula=ref["formula"],
                rt_match_score=1.0,          # пик помещён точно на RT библиотеки
                id_basis="synthetic_library",
                mw=float(ref.get("mw", 0.0)),
                monomer_type=ref.get("subclass", ref["class"]),
                name_en=ref["name_en"],
                name_ru=ref["name_ru"],
            ))

        tot_area_nom = sum(p.area for p in peaks_nom)
        for p in peaks_nom:
            p.area_pct = round((p.area / tot_area_nom) * 100.0, 2) if tot_area_nom > 0 else 0.0

        py_indices_nom = calculate_py_gcms_indices(peaks_nom)
        py_obj_nom = PyGCMSSample(
            sample_id=s_id_nom,
            retention_times=rt_grid,
            tic_intensity=tic_nom,
            peaks=peaks_nom,
            indices=py_indices_nom
        )

        therm_sample_nom = ThermalSample(sample_id=s_id_nom, py_gcms=py_obj_nom, tga=tga_obj_nom)
        samples[s_id_nom] = therm_sample_nom

        row_nom = {"Sample_ID": s_id_nom, "Class_Target": "Baikal_Pristine"}
        row_nom.update(extract_thermal_descriptors(therm_sample_nom))
        matrix_rows.append(row_nom)

        # -------------------------------------------------------------
        # ОБРАЗЕЦ КЛАССА 2: Шлам-лигнин БЦБК (Sludge-Lignin)
        # -------------------------------------------------------------
        s_id_lignin = f"Sludge_Lignin_{i:02d}"

        # 2.1 TGA профиль
        # Низкая влажность (3-5%), малая алифатика (18-24%), высокая ароматика (38-46%), высокий остаток (Char: 30-38%)
        w_moist_l = rng.uniform(3.5, 5.5)
        w_aliph_l = rng.uniform(19.0, 23.0)
        w_arom_l = rng.uniform(38.0, 44.0)
        char_rem_l = 100.0 - (w_moist_l + w_aliph_l + w_arom_l)

        m_tga_lignin = 100.0 - w_moist_l / (1.0 + np.exp(-(t_grid - 85.0) / 16.0)) \
                             - w_aliph_l / (1.0 + np.exp(-(t_grid - 275.0) / 28.0)) \
                             - w_arom_l / (1.0 + np.exp(-(t_grid - 415.0) / 38.0))
        m_tga_lignin += rng.normal(0, 0.05, size=len(t_grid))
        m_tga_lignin = np.clip(m_tga_lignin, char_rem_l - 2.0, 100.0)

        dtg_lignin = -np.gradient(m_tga_lignin, t_grid)
        tga_indices_lignin = calculate_tga_indices(t_grid, m_tga_lignin, dtg_lignin)

        tga_obj_lignin = TGASample(
            sample_id=s_id_lignin,
            temperature=t_grid,
            mass_pct=m_tga_lignin,
            dtg=dtg_lignin,
            indices=tga_indices_lignin
        )

        # 2.2 Py-GC-MS профиль
        # Крафт-лигнин: подавляющее доминирование G-лигнина (гваякол, эвгенол, ванилин) над углеводами и S-лигнином
        peaks_lignin: List[PyGCMSPeak] = []
        tic_lignin = np.zeros_like(rt_grid) + rng.normal(50.0, 5.0, size=len(rt_grid))

        for ref in PY_GCMS_REFERENCE_LIBRARY:
            ref_rt = ref["rt"]
            ref_cls = ref["class"]

            if ref_cls == "G_Lignin":
                base_area = rng.uniform(800.0, 1600.0)  # Доминанта G-лигнина
            elif ref_cls == "H_Lignin":
                base_area = rng.uniform(120.0, 250.0)
            elif ref_cls == "S_Lignin":
                base_area = rng.uniform(80.0, 180.0)   # Низкий сирингол для хвойного сульфатного лигнина
            elif ref_cls == "Carbohydrate":
                base_area = rng.uniform(100.0, 250.0)  # Деградированные полисахариды
            else:
                base_area = rng.uniform(40.0, 120.0)

            peak_shape = (base_area / 5.0) * np.exp(-((rt_grid - ref_rt) ** 2) / (2 * 0.06 ** 2))
            tic_lignin += peak_shape

            peaks_lignin.append(PyGCMSPeak(
                retention_time=ref_rt,
                area=round(float(base_area), 2),
                height=round(float(base_area / 5.0), 2),
                compound_name=ref["name_en"],
                chemical_class=ref["class"],
                cas=ref["cas"],
                formula=ref["formula"],
                rt_match_score=1.0,
                id_basis="synthetic_library",
                mw=float(ref.get("mw", 0.0)),
                monomer_type=ref.get("subclass", ref["class"]),
                name_en=ref["name_en"],
                name_ru=ref["name_ru"],
            ))

        tot_area_lig = sum(p.area for p in peaks_lignin)
        for p in peaks_lignin:
            p.area_pct = round((p.area / tot_area_lig) * 100.0, 2) if tot_area_lig > 0 else 0.0

        py_indices_lignin = calculate_py_gcms_indices(peaks_lignin)
        py_obj_lignin = PyGCMSSample(
            sample_id=s_id_lignin,
            retention_times=rt_grid,
            tic_intensity=tic_lignin,
            peaks=peaks_lignin,
            indices=py_indices_lignin
        )

        therm_sample_lignin = ThermalSample(sample_id=s_id_lignin, py_gcms=py_obj_lignin, tga=tga_obj_lignin)
        samples[s_id_lignin] = therm_sample_lignin

        row_lignin = {"Sample_ID": s_id_lignin, "Class_Target": "Sludge_Lignin"}
        row_lignin.update(extract_thermal_descriptors(therm_sample_lignin))
        matrix_rows.append(row_lignin)

        # Сохраняем индивидуальные CSV-файлы демо-данных при необходимости
        if save_dir:
            # TGA CSV
            df_tga_nom = pd.DataFrame({"Temperature_C": t_grid, "Mass_pct": m_tga_nom, "DTG_pct_C": dtg_nom})
            df_tga_nom.to_csv(os.path.join(save_dir, f"{s_id_nom}_TGA.csv"), index=False)

            df_tga_lig = pd.DataFrame({"Temperature_C": t_grid, "Mass_pct": m_tga_lignin, "DTG_pct_C": dtg_lignin})
            df_tga_lig.to_csv(os.path.join(save_dir, f"{s_id_lignin}_TGA.csv"), index=False)

            # Py-GC-MS TIC CSV
            df_py_nom = pd.DataFrame({"Retention_Time_min": rt_grid, "Intensity_TIC": tic_nom})
            df_py_nom.to_csv(os.path.join(save_dir, f"{s_id_nom}_PyGCMS.csv"), index=False)

            df_py_lig = pd.DataFrame({"Retention_Time_min": rt_grid, "Intensity_TIC": tic_lignin})
            df_py_lig.to_csv(os.path.join(save_dir, f"{s_id_lignin}_PyGCMS.csv"), index=False)

    df_matrix = pd.DataFrame(matrix_rows)
    if save_dir:
        df_matrix.to_csv(os.path.join(save_dir, "thermal_descriptors_matrix.csv"), index=False)

    return samples, df_matrix
