"""
elemental_core.py — Математическое ядро элементного анализа и валовой стехиометрии (CHNS-O).
Расчет атомных соотношений (H/C, O/C, N/C, S/C), номинальной степени окисления углерода (NOSC),
валовых диаграмм Ван Кревелена и кросс-валидация с молекулярными формулами FT-ICR MS.
"""
from dataclasses import dataclass, field
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Атомные массы элементов (IUPAC стандарт)
ATOMIC_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "S": 32.065,
}


@dataclass
class ElementalSample:
    """Контейнер данных элементного состава индивидуального образца."""
    sample_id: str
    c_pct: float          # Массовая доля углерода, %
    h_pct: float          # Массовая доля водорода, %
    n_pct: float          # Массовая доля азота, %
    s_pct: float          # Массовая доля серы, %
    o_pct: float          # Массовая доля кислорода, % (измеренная или по разности)
    ash_pct: float = 0.0  # Зольность, %
    moisture_pct: float = 0.0  # Влажность, %
    sample_type: str = "Unknown"  # Тип пробы (NOM, Lignin, Humic, Soil)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Рассчитанные атомные соотношения и индексы
    h_c: float = 0.0
    o_c: float = 0.0
    n_c: float = 0.0
    s_c: float = 0.0
    nosc: float = 0.0
    aromaticity_index: float = 0.0
    combustion_heat_mj_kg: float = 0.0

    def compute_stoichiometry(self) -> "ElementalSample":
        """Вычисляет атомные отношения и хемометрические дескрипторы."""
        # Пересчет на абсолютно сухое беззольное вещество (daf - dry ash-free) если зольность/влажность заданы
        organic_fraction = 100.0 - (self.ash_pct + self.moisture_pct)
        if organic_fraction > 0.0 and (self.ash_pct > 0.0 or self.moisture_pct > 0.0):
            scale = 100.0 / organic_fraction
            c_daf = self.c_pct * scale
            h_daf = self.h_pct * scale
            n_daf = self.n_pct * scale
            s_daf = self.s_pct * scale
            o_daf = self.o_pct * scale
        else:
            c_daf, h_daf, n_daf, s_daf, o_daf = self.c_pct, self.h_pct, self.n_pct, self.s_pct, self.o_pct

        # Атомные количества (моль на 100 г)
        mol_c = c_daf / ATOMIC_WEIGHTS["C"] if c_daf > 0 else 1e-9
        mol_h = h_daf / ATOMIC_WEIGHTS["H"] if h_daf > 0 else 0.0
        mol_o = o_daf / ATOMIC_WEIGHTS["O"] if o_daf > 0 else 0.0
        mol_n = n_daf / ATOMIC_WEIGHTS["N"] if n_daf > 0 else 0.0
        mol_s = s_daf / ATOMIC_WEIGHTS["S"] if s_daf > 0 else 0.0

        # Атомные соотношения
        self.h_c = float(mol_h / mol_c)
        self.o_c = float(mol_o / mol_c)
        self.n_c = float(mol_n / mol_c)
        self.s_c = float(mol_s / mol_c)

        # Номинальная степень окисления углерода (NOSC по Riedel et al., 2012 / Kroll et al., 2011)
        # NOSC = 4 - (4*C + H - 3*N - 2*O - 2*S) / C = - (H - 3*N - 2*O - 2*S) / C
        self.nosc = float(- (mol_h - 3.0 * mol_n - 2.0 * mol_o - 2.0 * mol_s) / mol_c)

        # Индекс ароматичности валовый (AI_bulk по Koch & Dittmar / Van Krevelen)
        # AI = (1 + C - O - 0.5*H) / (C - O)
        c_minus_o = mol_c - mol_o
        if c_minus_o > 0:
            ai_val = (1.0 + mol_c - mol_o - 0.5 * mol_h) / c_minus_o
            self.aromaticity_index = float(np.clip(ai_val, 0.0, 1.0))
        else:
            self.aromaticity_index = 0.0

        # Высшая теплота сгорания (формула Менделеева / Дюлонга в МДж/кг)
        # Q_s = 0.339*C + 1.256*H - 0.1088*(O - S)
        q_val = 0.339 * c_daf + 1.256 * h_daf - 0.1088 * max(0.0, o_daf - s_daf)
        self.combustion_heat_mj_kg = float(max(0.0, q_val))

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает словарь признаков образца для хемометрических таблиц."""
        return {
            "Sample_ID": self.sample_id,
            "Sample_Type": self.sample_type,
            "C_pct": round(self.c_pct, 2),
            "H_pct": round(self.h_pct, 2),
            "N_pct": round(self.n_pct, 2),
            "S_pct": round(self.s_pct, 2),
            "O_pct": round(self.o_pct, 2),
            "Ash_pct": round(self.ash_pct, 2),
            "Moisture_pct": round(self.moisture_pct, 2),
            "H_C": round(self.h_c, 3),
            "O_C": round(self.o_c, 3),
            "N_C": round(self.n_c, 4),
            "S_C": round(self.s_c, 4),
            "NOSC": round(self.nosc, 3),
            "AI_bulk": round(self.aromaticity_index, 3),
            "Q_combustion_MJ_kg": round(self.combustion_heat_mj_kg, 2),
        }


def parse_elemental_table(df_or_bytes: Union[pd.DataFrame, bytes, str]) -> List[ElementalSample]:
    """
    Парсит таблицу элементного анализа (CSV, TSV, Excel или DataFrame).
    Определяет колонки C, H, N, S, O, Ash, Moisture и рассчитывает стехиометрию.
    """
    if df_or_bytes is None:
        return []

    df = None
    if isinstance(df_or_bytes, pd.DataFrame):
        df = df_or_bytes.copy()
    elif isinstance(df_or_bytes, bytes):
        if not df_or_bytes.strip():
            return []
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                text = df_or_bytes.decode(enc)
                lines = [l for l in text.splitlines() if l.strip()]
                if not lines:
                    return []
                sep = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")
                df = pd.read_csv(io.StringIO(text), sep=sep)
                break
            except Exception:
                continue
    elif isinstance(df_or_bytes, str):
        s_clean = df_or_bytes.strip()
        if not s_clean:
            return []
        if os.path.exists(s_clean) and os.path.isfile(s_clean):
            try:
                df = pd.read_csv(s_clean)
            except Exception:
                return []
        else:
            try:
                lines = [l for l in s_clean.splitlines() if l.strip()]
                if not lines:
                    return []
                sep = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")
                df = pd.read_csv(io.StringIO(s_clean), sep=sep)
            except Exception:
                return []
    elif hasattr(df_or_bytes, "read"):
        try:
            df = pd.read_csv(df_or_bytes)
        except Exception:
            return []

    if df is None or df.empty:
        return []

    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(k == cl for k in ["sample", "id", "sample_id", "name", "образ", "название"]):
            col_map["id"] = c
        elif cl in ["c", "c%", "c_pct", "carbon", "углерод"]:
            col_map["c"] = c
        elif cl in ["h", "h%", "h_pct", "hydrogen", "водород"]:
            col_map["h"] = c
        elif cl in ["n", "n%", "n_pct", "nitrogen", "азот"]:
            col_map["n"] = c
        elif cl in ["s", "s%", "s_pct", "sulfur", "сера"]:
            col_map["s"] = c
        elif cl in ["o", "o%", "o_pct", "oxygen", "кислород"]:
            col_map["o"] = c
        elif any(k in cl for k in ["ash", "зольн"]):
            col_map["ash"] = c
        elif any(k in cl for k in ["moist", "влаж"]):
            col_map["moist"] = c
        elif any(k in cl for k in ["type", "тип", "group", "class"]):
            col_map["type"] = c

    samples: List[ElementalSample] = []
    for idx, row in df.iterrows():
        s_id = str(row[col_map["id"]]) if "id" in col_map else f"Sample_{idx + 1}"
        s_type = str(row[col_map["type"]]) if "type" in col_map else "Unknown"

        c_val = float(pd.to_numeric(row.get(col_map.get("c", ""), 0.0), errors="coerce") or 0.0)
        h_val = float(pd.to_numeric(row.get(col_map.get("h", ""), 0.0), errors="coerce") or 0.0)
        n_val = float(pd.to_numeric(row.get(col_map.get("n", ""), 0.0), errors="coerce") or 0.0)
        s_val = float(pd.to_numeric(row.get(col_map.get("s", ""), 0.0), errors="coerce") or 0.0)
        ash_val = float(pd.to_numeric(row.get(col_map.get("ash", ""), 0.0), errors="coerce") or 0.0)
        moist_val = float(pd.to_numeric(row.get(col_map.get("moist", ""), 0.0), errors="coerce") or 0.0)

        # Если O не задан явно, рассчитываем по балансу 100 - (C+H+N+S+Ash+Moisture)
        if "o" in col_map and pd.notna(row[col_map["o"]]):
            o_val = float(pd.to_numeric(row[col_map["o"]], errors="coerce") or 0.0)
        else:
            diff = 100.0 - (c_val + h_val + n_val + s_val + ash_val + moist_val)
            o_val = max(0.0, diff)

        s_obj = ElementalSample(
            sample_id=s_id,
            c_pct=c_val,
            h_pct=h_val,
            n_pct=n_val,
            s_pct=s_val,
            o_pct=o_val,
            ash_pct=ash_val,
            moisture_pct=moist_val,
            sample_type=s_type,
        )
        s_obj.compute_stoichiometry()
        samples.append(s_obj)

    return samples


def generate_synthetic_elemental_dataset() -> pd.DataFrame:
    """
    Генерирует согласованный эталонный датасет из 12 образцов:
    - 4 пробы природного РОВ (Байкал, Селенга, Баргузин, Снежная)
    - 4 пробы техногенного шлам-лигнина БЦБК (накопители, фильтрат, окисленный шлам)
    - 4 пробы эталонных гуминовых кислот (IHSS Suwannee River, Leonardite, Peat HA, Soil HA)
    """
    records = [
        # Байкал и природное РОВ
        {"Sample_ID": "Baikal_Deep_Water", "Sample_Type": "Natural_NOM", "C": 48.5, "H": 5.4, "N": 2.1, "S": 0.6, "O": 41.2, "Ash": 2.2, "Moist": 3.1},
        {"Sample_ID": "Selenga_River_NOM", "Sample_Type": "Natural_NOM", "C": 51.2, "H": 4.9, "N": 1.8, "S": 0.8, "O": 39.5, "Ash": 1.8, "Moist": 2.9},
        {"Sample_ID": "Barguzin_River_NOM", "Sample_Type": "Natural_NOM", "C": 52.0, "H": 5.1, "N": 1.6, "S": 0.7, "O": 38.8, "Ash": 1.9, "Moist": 3.0},
        {"Sample_ID": "Snezhnaya_River_NOM", "Sample_Type": "Natural_NOM", "C": 49.8, "H": 5.3, "N": 2.0, "S": 0.5, "O": 40.5, "Ash": 1.9, "Moist": 2.8},
        # Техногенный шлам-лигнин БЦБК
        {"Sample_ID": "BPPM_Sludge_Lignin_1", "Sample_Type": "Technogenic_Lignin", "C": 58.4, "H": 5.8, "N": 0.9, "S": 3.8, "O": 28.5, "Ash": 2.6, "Moist": 3.5},
        {"Sample_ID": "BPPM_Sludge_Lignin_2", "Sample_Type": "Technogenic_Lignin", "C": 57.2, "H": 5.6, "N": 1.1, "S": 4.2, "O": 29.1, "Ash": 2.8, "Moist": 3.8},
        {"Sample_ID": "BPPM_Cardboard_Effluent", "Sample_Type": "Technogenic_Lignin", "C": 55.8, "H": 5.4, "N": 1.3, "S": 3.4, "O": 31.2, "Ash": 2.9, "Moist": 3.4},
        {"Sample_ID": "BPPM_Landfill_Filtrate", "Sample_Type": "Technogenic_Lignin", "C": 54.1, "H": 5.2, "N": 1.5, "S": 3.1, "O": 33.2, "Ash": 2.9, "Moist": 3.7},
        # Гуминовые эталоны IHSS
        {"Sample_ID": "Suwannee_River_HA", "Sample_Type": "Humic_Standard", "C": 52.6, "H": 4.3, "N": 1.2, "S": 0.6, "O": 41.3, "Ash": 1.0, "Moist": 2.5},
        {"Sample_ID": "Leonardite_HA", "Sample_Type": "Humic_Standard", "C": 63.8, "H": 3.7, "N": 1.2, "S": 0.8, "O": 30.5, "Ash": 1.8, "Moist": 3.2},
        {"Sample_ID": "Pahokee_Peat_HA", "Sample_Type": "Humic_Standard", "C": 56.4, "H": 4.1, "N": 3.7, "S": 0.7, "O": 35.1, "Ash": 1.1, "Moist": 2.8},
        {"Sample_ID": "Elliott_Soil_HA", "Sample_Type": "Humic_Standard", "C": 58.1, "H": 3.7, "N": 4.1, "S": 0.4, "O": 33.7, "Ash": 1.2, "Moist": 2.6},
    ]
    return pd.DataFrame(records)


def cross_validate_fticr_elemental(
    assigned_df: pd.DataFrame,
    elemental_sample: ElementalSample,
) -> Dict[str, Any]:
    """
    Кросс-валидация между приписанными формулами FT-ICR MS и данными элементного микроанализа CHNS-O.
    Вычисляет средневзвешенные по интенсивности атомные отношения (H/C_ms, O/C_ms, N/C_ms, S/C_ms, NOSC_ms)
    и сопоставляет их с валовым составом, находя расхождения (bias, residual %).
    """
    if assigned_df is None or assigned_df.empty:
        return {"status": "error", "message": "Assigned formulas DataFrame is empty"}

    # Проверяем наличие колонок C, H, O, N, S, norm_intensity / intensity
    int_col = "norm_intensity" if "norm_intensity" in assigned_df.columns else "intensity"
    w = assigned_df[int_col].values
    total_w = np.sum(w) if np.sum(w) > 0 else 1.0

    c_vals = assigned_df["C"].values if "C" in assigned_df.columns else np.ones(len(assigned_df))
    h_vals = assigned_df["H"].values if "H" in assigned_df.columns else np.zeros(len(assigned_df))
    o_vals = assigned_df["O"].values if "O" in assigned_df.columns else np.zeros(len(assigned_df))
    n_vals = assigned_df["N"].values if "N" in assigned_df.columns else np.zeros(len(assigned_df))
    s_vals = assigned_df["S"].values if "S" in assigned_df.columns else np.zeros(len(assigned_df))

    # Средневзвешенные атомные количества
    w_c = np.sum(c_vals * w) / total_w
    w_h = np.sum(h_vals * w) / total_w
    w_o = np.sum(o_vals * w) / total_w
    w_n = np.sum(n_vals * w) / total_w
    w_s = np.sum(s_vals * w) / total_w

    hc_ms = float(w_h / w_c) if w_c > 0 else 0.0
    oc_ms = float(w_o / w_c) if w_c > 0 else 0.0
    nc_ms = float(w_n / w_c) if w_c > 0 else 0.0
    sc_ms = float(w_s / w_c) if w_c > 0 else 0.0
    nosc_ms = float(- (w_h - 3.0 * w_n - 2.0 * w_o - 2.0 * w_s) / w_c) if w_c > 0 else 0.0

    # Расхождения со стехиометрией CHNS-O
    diff_hc = hc_ms - elemental_sample.h_c
    diff_oc = oc_ms - elemental_sample.o_c
    diff_nosc = nosc_ms - elemental_sample.nosc

    # Оценка соответствия (Concordance Score 0..100%)
    rel_err_hc = abs(diff_hc) / max(0.1, elemental_sample.h_c)
    rel_err_oc = abs(diff_oc) / max(0.1, elemental_sample.o_c)
    concordance = float(max(0.0, 100.0 * (1.0 - 0.5 * (rel_err_hc + rel_err_oc))))

    return {
        "status": "success",
        "sample_id": elemental_sample.sample_id,
        "concordance_score": round(concordance, 1),
        "ms_weighted": {
            "H_C": round(hc_ms, 3),
            "O_C": round(oc_ms, 3),
            "N_C": round(nc_ms, 4),
            "S_C": round(sc_ms, 4),
            "NOSC": round(nosc_ms, 3),
        },
        "elemental_bulk": {
            "H_C": round(elemental_sample.h_c, 3),
            "O_C": round(elemental_sample.o_c, 3),
            "N_C": round(elemental_sample.n_c, 4),
            "S_C": round(elemental_sample.s_c, 4),
            "NOSC": round(elemental_sample.nosc, 3),
        },
        "differences": {
            "delta_H_C": round(diff_hc, 3),
            "delta_O_C": round(diff_oc, 3),
            "delta_NOSC": round(diff_nosc, 3),
        },
        "interpretation": (
            "Высокая сходимость FT-ICR MS и CHNS-O (ионизация репрезентативна)"
            if concordance >= 80.0 else
            "Умеренное расхождение (возможно смещение ESI(-) в сторону более полярных/окисленных кислот)"
            if concordance >= 60.0 else
            "Значительное расхождение (в FT-ICR MS не зарегистрированы гидрофобные или неполярные фракции)"
        )
    }
