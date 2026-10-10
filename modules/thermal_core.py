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
    similarity: float = 1.0
    mw: float = 0.0
    area_pct: float = 0.0
    monomer_type: str = ""
    name_en: str = ""
    name_ru: str = ""

    def __post_init__(self):
        if not self.name_en:
            self.name_en = self.compound_name
        if not self.name_ru:
            self.name_ru = self.compound_name
        if not self.monomer_type:
            self.monomer_type = self.chemical_class

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

def match_py_compound(
    rt: float,
    window: float = 0.6,
    lang: str = "en"
) -> Dict[str, Any]:
    """
    Ищет соединение в эталонной библиотеке продуктов пиролиза по времени удерживания (RT).
    Возвращает наилучшее совпадение по минимальному отклонению |rt - ref_rt|.
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
        sim = max(0.5, 1.0 - (min_diff / window) * 0.5)
        return {
            "name": name,
            "name_en": best_match["name_en"],
            "name_ru": best_match["name_ru"],
            "class": best_match["class"],
            "subclass": best_match.get("subclass", ""),
            "cas": best_match["cas"],
            "formula": best_match["formula"],
            "mw": float(best_match.get("mw", 0.0)),
            "similarity": float(sim),
        }
    return {
        "name": f"Unknown @ {rt:.2f} min" if lang == "en" else f"Неизвестный @ {rt:.2f} мин",
        "name_en": f"Unknown @ {rt:.2f} min",
        "name_ru": f"Неизвестный @ {rt:.2f} мин",
        "class": "Unclassified",
        "subclass": "Unclassified",
        "cas": None,
        "formula": None,
        "mw": 0.0,
        "similarity": 0.0,
    }


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

        # Пик-пикер на хроматограмме
        prominence = max(np.max(tic_vals) * 0.02, 1e-4) if len(tic_vals) > 0 else 1.0
        peak_indices, props = scipy.signal.find_peaks(tic_vals, prominence=prominence, distance=max(1, int(len(tic_vals) / 300)))

        for p_idx in peak_indices:
            p_rt = float(rt_vals[p_idx])
            p_height = float(tic_vals[p_idx])
            # Оценка площади трапецией вокруг пика (± 5 точек)
            left_i = max(0, p_idx - 5)
            right_i = min(len(rt_vals) - 1, p_idx + 5)
            if right_i > left_i:
                try:
                    p_area = float(scipy.integrate.trapezoid(tic_vals[left_i:right_i+1], rt_vals[left_i:right_i+1]))
                except AttributeError:
                    p_area = float(np.trapz(tic_vals[left_i:right_i+1], rt_vals[left_i:right_i+1]))
            else:
                p_area = p_height

            match_info = match_py_compound(p_rt, window=rt_window, lang=lang)
            peaks_list.append(PyGCMSPeak(
                retention_time=round(p_rt, 3),
                area=max(0.0, p_area),
                height=max(0.0, p_height),
                compound_name=match_info["name"],
                chemical_class=match_info["class"],
                cas=match_info["cas"],
                formula=match_info["formula"],
                similarity=match_info["similarity"],
                mw=match_info.get("mw", 0.0),
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
                # Пытаемся классифицировать известное имя
                c_low = c_name.lower()
                chem_class = "Unclassified"
                if any(k in c_low for k in ["syring", "сиринг"]):
                    chem_class = "S_Lignin"
                elif any(k in c_low for k in ["guaiac", "гваяк", "vanil", "ванил", "creosol", "креозол", "eugenol", "эвгенол"]):
                    chem_class = "G_Lignin"
                elif any(k in c_low for k in ["phenol", "фенол", "cresol", "крезол"]):
                    chem_class = "H_Lignin"
                elif any(k in c_low for k in ["furan", "фуран", "levoglucos", "левоглюкоз"]):
                    chem_class = "Carbohydrate"
                elif any(k in c_low for k in ["indol", "индол", "pyridin", "пиридин", "pyrrol", "пиррол"]):
                    chem_class = "Nitrogen_Protein"
                
                match_info = match_py_compound(p_rt, window=rt_window, lang=lang)
                peaks_list.append(PyGCMSPeak(
                    retention_time=round(p_rt, 3),
                    area=max(0.0, p_area),
                    height=max(0.0, p_height),
                    compound_name=c_name,
                    chemical_class=chem_class,
                    cas=match_info.get("cas"),
                    formula=match_info.get("formula"),
                    similarity=1.0,
                    mw=match_info.get("mw", 0.0),
                    monomer_type=match_info.get("subclass", chem_class),
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
                    similarity=match_info["similarity"],
                    mw=match_info.get("mw", 0.0),
                    monomer_type=match_info.get("subclass", match_info["class"]),
                    name_en=match_info.get("name_en", match_info["name"]),
                    name_ru=match_info.get("name_ru", match_info["name"]),
                ))

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

def calculate_py_gcms_indices(peaks: List[PyGCMSPeak]) -> Dict[str, float]:
    """
    Рассчитывает ключевые хемометрические параметры состава пиролизатов:
    - S/G, H/G отношения мономеров лигнина;
    - Доли мономеров H%, G%, S%;
    - Соотношение лигнин/углеводы (L/C);
    - Окислительно-деградационные индексы (Ac/Al)_G, (Ac/Al)_S;
    - Суммарную степень ароматичности продуктов пиролиза.
    """
    if not peaks:
        return {
            "Py_H_pct": 0.0, "Py_G_pct": 0.0, "Py_S_pct": 0.0,
            "Py_S_G_ratio": 0.0, "Py_H_G_ratio": 0.0, "Py_L_C_ratio": 0.0,
            "Py_Ac_Al_G": 0.0, "Py_Ac_Al_S": 0.0,
            "Py_Total_Lignin_pct": 0.0, "Py_Carbohydrate_pct": 0.0,
            "Py_Nitrogen_pct": 0.0, "Py_Aromaticity_pct": 0.0,
        }

    total_area = sum(p.area for p in peaks)
    if total_area <= 0:
        total_area = 1.0

    area_h = sum(p.area for p in peaks if p.chemical_class == "H_Lignin")
    area_g = sum(p.area for p in peaks if p.chemical_class == "G_Lignin")
    area_s = sum(p.area for p in peaks if p.chemical_class == "S_Lignin")
    area_carb = sum(p.area for p in peaks if p.chemical_class == "Carbohydrate")
    area_n = sum(p.area for p in peaks if p.chemical_class == "Nitrogen_Protein")
    area_lignin = area_h + area_g + area_s

    # Пропорции внутри лигнина
    denom_lignin = max(area_lignin, 1e-9)
    h_pct = (area_h / denom_lignin) * 100.0
    g_pct = (area_g / denom_lignin) * 100.0
    s_pct = (area_s / denom_lignin) * 100.0

    # Отношения S/G и H/G
    s_g_ratio = area_s / max(area_g, 1e-6)
    h_g_ratio = area_h / max(area_g, 1e-6)

    # Отношение Лигнин / Углеводы (L/C)
    l_c_ratio = area_lignin / max(area_carb, 1e-6)

    # Карбонильные индексы кислота/кетон к альдегиду (Ac/Al)
    # G-блок: Acetovanillone / Vanillin
    area_vanillin = sum(p.area for p in peaks if "vanillin" in p.compound_name.lower() or "ванил" in p.compound_name.lower() and "acet" not in p.compound_name.lower())
    area_acetovanillone = sum(p.area for p in peaks if "acetovanillone" in p.compound_name.lower() or "ацетованил" in p.compound_name.lower() or "apocynin" in p.compound_name.lower())
    ac_al_g = area_acetovanillone / max(area_vanillin, 1e-6) if area_vanillin > 0 else (0.5 if area_acetovanillone > 0 else 0.0)

    # S-блок: Acetosyringone / Syringaldehyde
    area_syringaldehyde = sum(p.area for p in peaks if "syringaldehyde" in p.compound_name.lower() or "сирингальдегид" in p.compound_name.lower())
    area_acetosyringone = sum(p.area for p in peaks if "acetosyringone" in p.compound_name.lower() or "ацетосиринг" in p.compound_name.lower())
    ac_al_s = area_acetosyringone / max(area_syringaldehyde, 1e-6) if area_syringaldehyde > 0 else (0.5 if area_acetosyringone > 0 else 0.0)

    # Общие доли от всей смеси
    total_lignin_pct = (area_lignin / total_area) * 100.0
    carb_pct = (area_carb / total_area) * 100.0
    n_pct = (area_n / total_area) * 100.0
    aromatic_pct = ((area_lignin + area_n) / total_area) * 100.0

    return {
        "Py_H_pct": round(float(h_pct), 2),
        "Py_G_pct": round(float(g_pct), 2),
        "Py_S_pct": round(float(s_pct), 2),
        "Py_S_G_ratio": round(float(s_g_ratio), 3),
        "Py_H_G_ratio": round(float(h_g_ratio), 3),
        "Py_L_C_ratio": round(float(l_c_ratio), 3),
        "Py_Ac_Al_G": round(float(ac_al_g), 3),
        "Py_Ac_Al_S": round(float(ac_al_s), 3),
        "Py_Total_Lignin_pct": round(float(total_lignin_pct), 2),
        "Py_Carbohydrate_pct": round(float(carb_pct), 2),
        "Py_Nitrogen_pct": round(float(n_pct), 2),
        "Py_Aromaticity_pct": round(float(aromatic_pct), 2),
        "Py_Acid_Aldehyde_G": round(float(ac_al_g), 3),
        "Py_Acid_Aldehyde_S": round(float(ac_al_s), 3),
        "Py_Total_Lignin_Area_pct": round(float(total_lignin_pct), 2),
        "Py_Carbohydrate_Area_pct": round(float(carb_pct), 2),
        "Py_Nitrogen_Area_pct": round(float(n_pct), 2),
        "Py_Aromaticity_Area_pct": round(float(aromatic_pct), 2),
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

    dtg = np.zeros_like(m_pct)
    if len(m_pct) > 15:
        window = min(21, len(m_pct) if len(m_pct) % 2 != 0 else len(m_pct) - 1)
        if window >= 5:
            smooth_m = scipy.signal.savgol_filter(m_pct, window_length=window, polyorder=2)
            dtg = -np.gradient(smooth_m, t_safe)
            dtg = np.clip(dtg, 0.0, None)  # DTG >= 0
    elif len(m_pct) > 1:
        dtg = -np.gradient(m_pct, t_safe)
        dtg = np.clip(dtg, 0.0, None)

    # Расчет индексов термогравиметрии
    indices = calculate_tga_indices(t_vals, m_pct, dtg)

    return TGASample(
        sample_id=sample_id,
        temperature=t_vals,
        mass_pct=m_pct,
        raw_mass_mg=m_vals if np.max(m_vals) != np.max(m_pct) else None,
        dtg=dtg,
        indices=indices,
        metadata={"t_min": float(np.min(t_vals)), "t_max": float(np.max(t_vals))},
    )


def calculate_tga_indices(
    temperature: np.ndarray,
    mass_pct: np.ndarray,
    dtg: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Рассчитывает стандартные термогравиметрические параметры:
    1. Потери массы по 4 температурным интервалам:
       - Зона I (25–150 °C): влага и легколетучие соединения (Moisture loss);
       - Зона II (150–350 °C): термодеструкция алифатики/гемицеллюлозы/целлюлозы;
       - Зона III (350–550 °C): термодеструкция ароматического каркаса и лигнина;
       - Остаточный коксовый остаток (Char yield at 600°C / 800°C).
    2. Индекс термостабильности TI = Δm_arom / Δm_aliph.
    3. Характеристические температуры T_5%, T_10%, T_50%.
    4. Максимумы скорости деструкции DTG (T_max1, T_max2).
    """
    if len(temperature) == 0 or len(mass_pct) == 0:
        return {
            "TGA_Moisture_Loss_pct": 0.0,
            "TGA_Aliphatic_Loss_pct": 0.0,
            "TGA_Aromatic_Loss_pct": 0.0,
            "TGA_Char_Yield_600C": 0.0,
            "TGA_Char_Yield_800C": 0.0,
            "TGA_Thermostability_Index": 0.0,
            "TGA_T_5pct": 0.0, "TGA_T_10pct": 0.0, "TGA_T_50pct": 0.0,
            "DTG_T_max1": 0.0, "DTG_T_max2": 0.0, "DTG_Max_Rate": 0.0,
            "TGA_Zone_I_pct": 0.0, "TGA_Zone_II_pct": 0.0, "TGA_Zone_III_pct": 0.0, "TGA_Zone_IV_pct": 0.0,
            "TGA_Thermal_Index_TI": 0.0, "TGA_T5_degC": 0.0, "TGA_T10_degC": 0.0, "TGA_T50_degC": 0.0,
            "TGA_Residue_800C_pct": 0.0, "DTG_Tmax_degC": 0.0,
        }

    def get_mass_at_t(target_t: float) -> float:
        idx = np.searchsorted(temperature, target_t)
        if idx == 0: return float(mass_pct[0])
        if idx >= len(temperature): return float(mass_pct[-1])
        # линейная интерполяция
        t0, t1 = temperature[idx-1], temperature[idx]
        m0, m1 = mass_pct[idx-1], mass_pct[idx]
        if t1 == t0: return float(m0)
        return float(m0 + (m1 - m0) * (target_t - t0) / (t1 - t0))

    m_start = get_mass_at_t(25.0)
    m_150 = get_mass_at_t(150.0)
    m_350 = get_mass_at_t(350.0)
    m_550 = get_mass_at_t(550.0)
    m_600 = get_mass_at_t(600.0)
    m_800 = get_mass_at_t(800.0) if np.max(temperature) >= 750.0 else float(mass_pct[-1])

    # Потери массы в интервалах
    dm_moisture = max(0.0, m_start - m_150)
    dm_aliph = max(0.0, m_150 - m_350)
    dm_arom = max(0.0, m_350 - m_550)

    # Индекс термостабильности TI = Δm_arom / Δm_aliph
    ti = dm_arom / max(dm_aliph, 1e-4)

    # Характеристические температуры потери 5%, 10%, 50% от сухого образца (m_150)
    dry_base = m_150
    t_5, t_10, t_50 = 0.0, 0.0, 0.0

    target_95 = dry_base * 0.95
    target_90 = dry_base * 0.90
    target_50 = dry_base * 0.50

    idx_5 = np.where(mass_pct <= target_95)[0]
    t_5 = float(temperature[idx_5[0]]) if len(idx_5) > 0 else float(np.max(temperature))

    idx_10 = np.where(mass_pct <= target_90)[0]
    t_10 = float(temperature[idx_10[0]]) if len(idx_10) > 0 else float(np.max(temperature))

    idx_50 = np.where(mass_pct <= target_50)[0]
    t_50 = float(temperature[idx_50[0]]) if len(idx_50) > 0 else float(np.max(temperature))

    # Максимумы DTG
    t_max1, t_max2, max_rate = 0.0, 0.0, 0.0
    if dtg is not None and len(dtg) == len(temperature):
        # Ищем главный максимум в диапазоне 200–600 °C
        valid_mask = (temperature >= 180.0) & (temperature <= 600.0)
        if np.any(valid_mask):
            sub_t = temperature[valid_mask]
            sub_dtg = dtg[valid_mask]
            peaks_i, _ = scipy.signal.find_peaks(sub_dtg, prominence=np.max(sub_dtg)*0.15)
            if len(peaks_i) > 0:
                # Сортируем пики по высоте DTG
                sorted_p = sorted(peaks_i, key=lambda pi: sub_dtg[pi], reverse=True)
                t_max1 = float(sub_t[sorted_p[0]])
                max_rate = float(sub_dtg[sorted_p[0]])
                if len(sorted_p) > 1:
                    t_max2 = float(sub_t[sorted_p[1]])
            else:
                max_idx = np.argmax(sub_dtg)
                t_max1 = float(sub_t[max_idx])
                max_rate = float(sub_dtg[max_idx])

    dm_char = max(0.0, 100.0 - (float(dm_moisture) + float(dm_aliph) + float(dm_arom)))
    return {
        "TGA_Moisture_Loss_pct": round(float(dm_moisture), 2),
        "TGA_Aliphatic_Loss_pct": round(float(dm_aliph), 2),
        "TGA_Aromatic_Loss_pct": round(float(dm_arom), 2),
        "TGA_Char_Yield_600C": round(float(m_600), 2),
        "TGA_Char_Yield_800C": round(float(m_800), 2),
        "TGA_Thermostability_Index": round(float(ti), 3),
        "TGA_T_5pct": round(float(t_5), 1),
        "TGA_T_10pct": round(float(t_10), 1),
        "TGA_T_50pct": round(float(t_50), 1),
        "DTG_T_max1": round(float(t_max1), 1),
        "DTG_T_max2": round(float(t_max2), 1),
        "DTG_Max_Rate": round(float(max_rate), 4),
        "TGA_Zone_I_pct": round(float(dm_moisture), 2),
        "TGA_Zone_II_pct": round(float(dm_aliph), 2),
        "TGA_Zone_III_pct": round(float(dm_arom), 2),
        "TGA_Zone_IV_pct": round(float(dm_char), 2),
        "TGA_Thermal_Index_TI": round(float(ti), 3),
        "TGA_T5_degC": round(float(t_5), 1),
        "TGA_T10_degC": round(float(t_10), 1),
        "TGA_T50_degC": round(float(t_50), 1),
        "TGA_Residue_800C_pct": round(float(m_800), 2),
        "DTG_Tmax_degC": round(float(t_max1), 1),
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
        dtg_nom = np.clip(dtg_nom, 0.0, None)
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
                similarity=round(float(rng.uniform(0.92, 0.99)), 3),
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
        dtg_lignin = np.clip(dtg_lignin, 0.0, None)
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
                similarity=round(float(rng.uniform(0.94, 0.99)), 3),
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
