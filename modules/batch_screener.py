"""
batch_screener.py — Автономный Batch Processing & фоновый пайплайн скрининга директорий ChemoSuite 3.0.
Позволяет осуществлять пакетное сканирование файлов спектров и матриц, автоматическое
распознавание аналитических модальностей (все 8 методов), фильтрацию, извлечение дескрипторов
и сборку объединенной хемометрической матрицы (Data Staging) без рутинных ручных действий.

Порядок распознавания (detect_file_modality):
1. Матрица дескрипторов (столбец Sample_ID + столбцы-дескрипторы известного блока) —
   раньше токенов имени файла, чтобы «ftir_descriptors_matrix.csv» не разбирался как спектр.
2. Токены имени файла.
3. Содержимое: JCAMP-DX → EEM-сетка → заголовки осей спектров/хроматограмм
   (TGA, FTIR, ЯМР, UV-Vis, FT-ICR, SEC, Py-GC-MS) → эвристика по диапазону оси X →
   элементный анализ по токенам заголовка (≥3 из C/H/N/S/O) → общий CSV.
Отсутствующие значения сохраняются как NaN (без fillna(0.0)).
"""
from datetime import datetime
import io
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd

# Импорты аналитических ядер платформы
try:
    import fticr_core
except ImportError:
    from . import fticr_core

try:
    import eem_core
except ImportError:
    from . import eem_core

try:
    import ftir_core
except ImportError:
    from . import ftir_core

try:
    import nmr_core
except ImportError:
    from . import nmr_core

try:
    import thermal_core
except ImportError:
    from . import thermal_core

try:
    import sec_core
except ImportError:
    from . import sec_core

try:
    import elemental_core
except ImportError:
    from . import elemental_core

try:
    import chemo_ml
except ImportError:
    from . import chemo_ml

try:
    import report_generator
except ImportError:
    from . import report_generator


# Столбцы-идентификаторы образца (нормализованные токены) и служебные столбцы
_ID_TOKENS = {"sampleid", "sample", "id", "name", "samplename", "образец", "проба", "шифр"}
_NON_FEATURE_TOKENS = {"classtarget", "class", "sampletype", "type", "group", "label", "basis", "matrix"}

# Элементный анализ: токены столбцов после удаления единиц (%, wt, pct, mass)
_ELEMENT_TOKENS = {
    "c": "C", "h": "H", "n": "N", "s": "S", "o": "O",
    "carbon": "C", "hydrogen": "H", "nitrogen": "N", "sulfur": "S", "sulphur": "S", "oxygen": "O",
    "углерод": "C", "водород": "H", "азот": "N", "сера": "S", "кислород": "O",
}

# Резервная карта префиксов дескрипторов (если chemo_ml недоступен)
_PREFIX_BLOCKS = [
    ("SEC_", "SEC/GPC"), ("Py_", "Thermal"), ("TGA_", "Thermal"), ("DTG_", "Thermal"),
    ("FTIR_", "FTIR"), ("NMR_", "NMR"), ("EEM_", "EEM-PARAFAC"), ("UV_", "UV-Vis"),
    ("Elem_", "Elemental"), ("VK_", "FT-ICR MS"), ("FTICR_", "FT-ICR MS"), ("Raman_", "Raman"),
]


def extract_sample_id_from_filename(filename: str) -> str:
    """
    Извлекает чистое имя образца (Sample_ID) из названия файла,
    отсекая расширение и типовые суффиксы модальностей.
    """
    stem = Path(filename).stem
    # Очистка суффиксов
    patterns = [
        r"[-_](?:fticr|ms|ftms|mass|spectra)$",
        r"[-_](?:eem|parafac|fluo|fluor)$",
        r"[-_](?:uv|uvvis|abs|absorption)$",
        r"[-_](?:ftir|atr|raman|ir)$",
        r"[-_](?:nmr|13c|1h|cpmas)$",
        r"[-_](?:tga|dtg|dsc|pygcms|thermal|py)$",
        r"[-_](?:sec|gpc|chrom|mwd)$",
        r"[-_](?:elem|elemental|chns|chnso)$",
    ]
    cleaned = stem
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or stem


def _decode(file_bytes_or_str: Union[str, bytes]) -> str:
    if isinstance(file_bytes_or_str, bytes):
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                return file_bytes_or_str.decode(enc)
            except Exception:
                continue
        return file_bytes_or_str.decode("utf-8", errors="replace")
    return str(file_bytes_or_str)


def _split_header(line: str) -> List[str]:
    sep = "\t" if "\t" in line else (";" if ";" in line else ",")
    return [c.strip().strip('"').strip("'") for c in line.split(sep)]


def _col_token(col: str) -> str:
    """Нормализованный токен столбца: без единиц/скобок/разделителей, нижний регистр."""
    h = str(col).strip().lower()
    h = re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", " ", h)
    h = re.sub(r"(wt\.?\s*%|mass\s*%|% ?масс\.?|масс\.?\s*%|%|pct|percent)", " ", h)
    h = re.sub(r"(?<=[a-zа-я])[\s_,;:\-]+(?:wt|mass|масс)$", " ", h)
    return re.sub(r"[\s_\-\.,:;/'\"]+", "", h)


def _elemental_hits(columns: List[str]) -> List[str]:
    hits = []
    for c in columns:
        tok = _col_token(c)
        el = _ELEMENT_TOKENS.get(tok)
        if el and el not in hits:
            hits.append(el)
    return hits


def _find_id_column(columns: List[str]) -> Optional[str]:
    for c in columns:
        if _col_token(c) in _ID_TOKENS:
            return c
    return None


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def _classify_descriptor_columns(cols: List[str]) -> Dict[str, str]:
    try:
        return dict(chemo_ml.classify_feature_blocks(cols))
    except Exception:
        out = {}
        for c in cols:
            out[c] = next((b for p, b in _PREFIX_BLOCKS if str(c).startswith(p)), "Other")
        return out


def _descriptor_matrix_modality(columns: List[str], data_rows: List[List[str]]) -> Optional[Dict[str, Any]]:
    """Распознаёт таблицу «строки = образцы, столбцы = дескрипторы» и её блок(и)."""
    id_col = _find_id_column(columns)
    if id_col is None:
        return None
    feat = [c for c in columns if c != id_col and _col_token(c) not in _NON_FEATURE_TOKENS and c]
    if len(feat) < 2:
        return None
    # Значения в столбце ID — не числа (имена образцов)
    id_idx = columns.index(id_col)
    id_vals = [r[id_idx] for r in data_rows if len(r) > id_idx and r[id_idx]]
    if id_vals and sum(_is_number(v) for v in id_vals) > 0.5 * len(id_vals):
        return None
    blocks = _classify_descriptor_columns(feat)
    counts: Dict[str, int] = {}
    for b in blocks.values():
        if b and b != "Other":
            counts[b] = counts.get(b, 0) + 1
    n_known = sum(counts.values())
    if n_known < max(2, 0.5 * len(feat)):
        return None
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_block, top_n = ranked[0]
    significant = [b for b, n in ranked if n >= 3 or n >= 0.2 * n_known]
    if len(significant) >= 2:
        return {"modality": "Multimodal", "confidence": "high", "kind": "descriptor_matrix",
                "reason": "Descriptor matrix (blocks: " + ", ".join(f"{b}={n}" for b, n in ranked) + ")",
                "blocks": dict(ranked)}
    return {"modality": top_block, "confidence": "high", "kind": "descriptor_matrix",
            "reason": f"Descriptor matrix ({top_n}/{len(feat)} columns in block {top_block})",
            "blocks": dict(ranked)}


def _axis_range_modality(data_rows: List[List[str]]) -> Optional[Dict[str, Any]]:
    """Эвристика для числовых 2+-колоночных таблиц без информативного заголовка."""
    xs, ys = [], []
    for r in data_rows:
        if len(r) >= 2 and _is_number(r[0]) and _is_number(r[1]):
            xs.append(float(r[0]))
            ys.append(float(r[1]))
    if len(xs) < 20:
        return None
    x = np.asarray(xs)
    y = np.asarray(ys)
    lo, hi = float(np.min(x)), float(np.max(x))
    d = np.diff(np.sort(x))
    d = d[d > 0]
    cv = float(np.std(d) / np.mean(d)) if len(d) > 2 and np.mean(d) > 0 else 0.0
    res = lambda m: {"modality": m, "confidence": "medium", "kind": "spectrum", "reason": f"Axis range {lo:g}–{hi:g}"}
    if lo >= 350 and 3000 <= hi <= 8000:
        return res("FTIR")
    if 15 <= lo <= 60 and 300 <= hi <= 1500 and np.nanmax(y) <= 110 and y[0] > y[-1]:
        return res("Thermal")
    if -50 <= lo < 10 and 100 < hi <= 300:
        return res("NMR")
    if 180 <= lo and hi <= 1100 and hi - lo >= 100 and cv < 0.05:
        return res("UV-Vis")
    if lo >= 50 and hi <= 3000 and cv > 0.5:
        return res("FT-ICR MS")
    return None


def detect_file_modality(
    file_bytes_or_str: Union[str, bytes],
    filename: str = "",
) -> Dict[str, Any]:
    """
    Интеллектуальный анализатор типа аналитического файла.
    Определяет принадлежность к одному из 8 методов по структуре таблицы и ключевым словам.
    Возвращает {"modality", "confidence", "reason", "kind"}, где kind — "spectrum",
    "table", "descriptor_matrix" или "unknown".
    """
    stem_low = Path(filename).stem.lower()

    text = _decode(file_bytes_or_str)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()][:40]
    header_cols = _split_header(lines[0]) if lines else []
    data_rows = [_split_header(ln) for ln in lines[1:]]

    # 0. Матрица дескрипторов (раньше токенов имени файла)
    if lines and not lines[0].startswith("##") and len(_elemental_hits(header_cols)) < 3:
        dm = _descriptor_matrix_modality(header_cols, data_rows)
        if dm is not None:
            return dm

    # 1. Проверка по суффиксам и токенам имени файла
    tokens = re.split(r"[-_.\s]+", stem_low)
    by_name = None
    if any(k in tokens for k in ["fticr", "ftms", "mass", "ms"]):
        by_name = "FT-ICR MS"
    elif any(k in tokens for k in ["eem", "parafac", "fluo", "fluor"]):
        by_name = "EEM-PARAFAC"
    elif any(k in tokens for k in ["uv", "uvvis", "absorbance"]):
        by_name = "UV-Vis"
    elif any(k in tokens for k in ["ftir", "raman", "ir", "wavenumber"]):
        by_name = "FTIR"
    elif any(k in tokens for k in ["nmr", "13c", "1h", "cpmas"]) or filename.lower().endswith((".jdx", ".dx")):
        by_name = "NMR"
    elif any(k in tokens for k in ["tga", "dtg", "dsc", "pygcms", "thermal", "py"]):
        by_name = "Thermal"
    elif any(k in tokens for k in ["sec", "gpc", "mwd", "chrom"]):
        by_name = "SEC/GPC"
    elif any(k in tokens for k in ["chns", "chnso", "elemental", "elem"]):
        by_name = "Elemental"
    if by_name is not None:
        kind = "table" if by_name == "Elemental" else "spectrum"
        return {"modality": by_name, "confidence": "high", "reason": "Filename token", "kind": kind}

    # 2. Содержимое
    if not lines:
        return {"modality": "Unknown", "confidence": "low", "reason": "Empty file", "kind": "unknown"}

    header_text = " ".join(lines[:5]).lower()
    col_low = [c.lower() for c in header_cols]

    col_words = [set(t for t in re.split(r"[^0-9a-zа-яё°]+", c) if t) for c in col_low]

    def _any_col(keys) -> bool:
        """Короткие буквенно-цифровые ключи (rt, ri, uv, nm, …) — только целым словом."""
        for k in keys:
            short = len(k) <= 3 and re.fullmatch(r"[0-9a-zа-яё]+", k) is not None
            for c, words in zip(col_low, col_words):
                if (k in words) if short else (k in c):
                    return True
        return False

    # JCAMP-DX
    if "##jcamp-dx" in header_text or "##data type= nmr" in header_text:
        return {"modality": "NMR", "confidence": "high", "reason": "JCAMP-DX Header", "kind": "spectrum"}

    # EEM Matrix (числа в первой строке как длины волн возбуждения/испускания)
    numeric_tokens = [float(t) for t in header_cols if _is_number(t)]
    eem_grid = len(numeric_tokens) >= 5 and all(200 <= v <= 900 for v in numeric_tokens)
    eem_label = bool(col_low) and any(k in col_low[0] for k in ["em/ex", "ex/em"])
    if eem_grid or eem_label:
        return {"modality": "EEM-PARAFAC", "confidence": "medium", "reason": "EEM Spectral Grid", "kind": "spectrum"}

    # Thermal TGA/DTG (температура + масса)
    if _any_col(["temp", "темп", "°c"]) and _any_col(["weight", "mass", "масс", "tg", "dtg", "вес"]):
        return {"modality": "Thermal", "confidence": "high", "reason": "TGA Temperature/Weight Header", "kind": "spectrum"}

    # FTIR (wavenumber, cm-1)
    if _any_col(["wavenumber", "cm-1", "cm^-1", "cm1", "волнов"]):
        return {"modality": "FTIR", "confidence": "high", "reason": "FTIR Wavenumber Header", "kind": "spectrum"}

    # NMR (chemical_shift, ppm)
    if _any_col(["ppm", "chemical_shift", "chemical shift", "химсдвиг", "хим. сдвиг"]):
        return {"modality": "NMR", "confidence": "high", "reason": "NMR Chemical Shift Header", "kind": "spectrum"}

    # UV-Vis (wavelength / nm + absorbance)
    if _any_col(["wavelength", "длина волны", "lambda", "nm", "нм"]) and _any_col(["abs", "absorb", "поглощ", "оптическ", "od"]):
        return {"modality": "UV-Vis", "confidence": "high", "reason": "UV-Vis Wavelength Header", "kind": "spectrum"}

    # FT-ICR MS (m/z, mass + intensity)
    if _any_col(["m/z", "mass", "m_z", "mz", "theor_mass", "calc_mass"]) and _any_col(
        ["intensity", "abundance", "intens", "resol"]
    ):
        return {"modality": "FT-ICR MS", "confidence": "high", "reason": "Mass Spectrometry Headers", "kind": "spectrum"}

    # SEC / GPC (elution volume + detector signal)
    if _any_col(["elution", "v_e", "объем элюц", "объём элюц"]) and _any_col(
        ["ri", "uv", "signal", "mau", "mv", "detector", "dabs", "сигнал"]
    ):
        return {"modality": "SEC/GPC", "confidence": "high", "reason": "SEC Elution Volume Header", "kind": "spectrum"}

    # Thermal Py-GC-MS (retention time + TIC/area)
    if _any_col(["retention", "rt", "время"]) and _any_col(["area", "tic", "площад", "abundance", "intensity"]):
        return {"modality": "Thermal", "confidence": "high", "reason": "Py-GC-MS Header", "kind": "spectrum"}

    # Числовые данные без информативного заголовка: эвристика по диапазону оси X
    all_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()][:20000]
    axis_rows = [_split_header(ln) for ln in all_lines[1:]]
    rows_for_axis = axis_rows if not all(_is_number(t) for t in header_cols) else [header_cols] + axis_rows
    axis = _axis_range_modality(rows_for_axis)
    if axis is not None:
        return axis

    # Элементный анализ по токенам заголовка (≥3 из C/H/N/S/O, в т.ч. %C, C_%, Carbon, Углерод)
    el_hits = _elemental_hits(header_cols)
    if len(el_hits) >= 3:
        return {"modality": "Elemental", "confidence": "high", "kind": "table",
                "reason": "CHNS-O Table Headers (" + ",".join(el_hits) + ")"}

    if _find_id_column(header_cols) is not None:
        return {"modality": "General / CSV", "confidence": "low", "reason": "Generic table with sample IDs", "kind": "table"}
    return {"modality": "General / CSV", "confidence": "low", "reason": "Generic tabular data", "kind": "unknown"}


def _rows_as_samples(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """Таблица «строки = образцы»: {Sample_ID: {дескриптор: значение}} (NaN пропускаются)."""
    id_col = _find_id_column([str(c) for c in df.columns])
    multi: Dict[str, Dict[str, float]] = {}
    labels: Dict[str, str] = {}
    if id_col is None:
        return multi, labels
    class_col = next((c for c in df.columns if _col_token(c) in ("classtarget", "class", "label", "group")), None)
    feat_cols = [
        c for c in df.columns
        if c != id_col and _col_token(c) not in _NON_FEATURE_TOKENS and pd.api.types.is_numeric_dtype(df[c])
    ]
    for _, r in df.iterrows():
        if pd.isna(r[id_col]):
            continue
        sid = str(r[id_col]).strip()
        multi[sid] = {
            col: round(float(r[col]), 4) for col in feat_cols if pd.notna(r[col]) and np.isfinite(float(r[col]))
        }
        if class_col is not None and pd.notna(r[class_col]):
            labels[sid] = str(r[class_col])
    return multi, labels


def process_single_file(
    content_bytes_or_str: Union[str, bytes],
    filename: str,
    modality: Optional[str] = None,
    sample_id: Optional[str] = None,
    fticr_max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Универсальный обработчик отдельного аналитического файла.
    Извлекает вектор дескрипторов и формирует отчет по статусу.

    ``fticr_max_peaks``: None (по умолчанию) — приписка формул по ВСЕМ пикам масс-спектра;
    целое N — только N наиболее интенсивных пиков (это указывается в сообщении).
    Матрицы дескрипторов и таблицы со столбцом Sample_ID разбираются построчно
    (строки = образцы); таблицы без идентификатора образца помечаются как нераспознанные.
    """
    s_id = sample_id or extract_sample_id_from_filename(filename)
    b_data = content_bytes_or_str if isinstance(content_bytes_or_str, bytes) else content_bytes_or_str.encode("utf-8")

    kind = None
    if not modality or modality == "Unknown":
        detection = detect_file_modality(b_data, filename)
        detected_mod = detection["modality"]
        kind = detection.get("kind")
    else:
        detected_mod = modality

    descriptors: Dict[str, float] = {}
    multi_samples: Optional[Dict[str, Dict[str, float]]] = None
    class_labels: Dict[str, str] = {}
    n_points = 0
    status = "success"
    msg = "OK"

    def _put(d: Dict[str, Any], prefix: str = "") -> None:
        for k, v in d.items():
            if k != "Sample_ID" and isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                descriptors[f"{prefix}{k}"] = round(float(v), 4)

    try:
        if kind == "descriptor_matrix" or detected_mod == "Multimodal":
            df_mat = pd.read_csv(io.BytesIO(b_data), sep=None, engine="python")
            n_points = len(df_mat)
            multi_samples, class_labels = _rows_as_samples(df_mat)
            if multi_samples:
                msg = f"Descriptor matrix: {len(multi_samples)} samples (rows) × {len(next(iter(multi_samples.values())))} descriptors"
            else:
                status = "warning"
                msg = "Descriptor matrix without usable rows"

        elif detected_mod == "FT-ICR MS":
            df_peaks = fticr_core.parse_uploaded_file(b_data)
            n_points = len(df_peaks)
            if not df_peaks.empty:
                if fticr_max_peaks is not None and fticr_max_peaks > 0 and len(df_peaks) > fticr_max_peaks:
                    use_p = df_peaks.sort_values("intensity", ascending=False).head(int(fticr_max_peaks))
                    peak_note = f"top {len(use_p)} of {len(df_peaks)} peaks by intensity"
                else:
                    use_p = df_peaks
                    peak_note = f"all {len(df_peaks)} peaks"
                assigned = fticr_core.fast_formula_assigner(use_p, ppm_tolerance=2.5)
                if not assigned.empty:
                    descriptors = chemo_ml.extract_fticr_descriptors(assigned, s_id)
                    descriptors = {k: v for k, v in descriptors.items() if k != "Sample_ID"}
                    msg = f"OK ({peak_note} used; {len(assigned)} formulas assigned)"
                else:
                    msg = f"No formulas assigned (low signal/noise; {peak_note})"
                    status = "warning"
            else:
                msg = "No mass spectral peaks found"
                status = "warning"

        elif detected_mod == "EEM-PARAFAC":
            # Парсинг матрицы EEM
            df_eem = pd.read_csv(io.BytesIO(b_data))
            eem_samp = eem_core.parse_eem_dataframe(df_eem, sample_id=s_id)
            n_points = int(eem_samp.data.size) if hasattr(eem_samp, "data") else len(df_eem)
            _put(eem_core.calculate_spectral_indices(eem_samp), "EEM_")

        elif detected_mod == "UV-Vis":
            # Парсинг спектра поглощения
            uv_samp = eem_core.parse_uv_vis_spectrum(io.BytesIO(b_data), sample_id=s_id)
            n_points = len(uv_samp.wl)
            # DOC берётся только из файла (UVVisSample.doc); без измеренного DOC SUVA254 = NaN
            # (раньше подставлялось условное DOC = 5 мг/л — выдуманная величина SUVA)
            _put(eem_core.calculate_uv_vis_indices(uv_samp), "UV_")

        elif detected_mod == "FTIR":
            ftir_samp = ftir_core.parse_ftir_file(b_data, sample_id=s_id)
            n_points = len(ftir_samp.wavenumbers)
            _put(ftir_core.calculate_ftir_indices(ftir_samp.wavenumbers, ftir_samp.absorbance, sample_id=s_id))

        elif detected_mod == "NMR":
            nmr_samp = nmr_core.parse_nmr_file(b_data, sample_id=s_id)
            n_points = len(nmr_samp.chemical_shifts)
            _put(nmr_core.calculate_13c_nmr_descriptors(nmr_samp.chemical_shifts, nmr_samp.intensity, sample_id=s_id))

        elif detected_mod == "Thermal":
            # TGA или Py-GC-MS: по заголовку (температура vs время удерживания), затем по имени файла
            head_low = _decode(b_data[:2000]).splitlines()[0].lower() if b_data else ""
            name_tokens = re.split(r"[-_.\s]+", Path(filename).stem.lower())
            if any(k in head_low for k in ("temp", "темп", "°c")):
                th_kind = "tga"
            elif any(k in head_low for k in ("retention", "время удерж")) or re.search(r"(^|[^a-z])rt([^a-z]|$)", head_low):
                th_kind = "py"
            elif any(t in name_tokens for t in ("pygcms", "py", "pyrolysis")):
                th_kind = "py"
            else:
                th_kind = "tga"
            tga_samp = thermal_core.parse_tga_file(b_data, sample_id=s_id) if th_kind == "tga" else None
            if tga_samp is not None and len(tga_samp.temperature) > 10:
                n_points = len(tga_samp.temperature)
                _put(tga_samp.indices)
            else:
                py_samp = thermal_core.parse_py_gcms_file(b_data, sample_id=s_id)
                n_points = len(py_samp.retention_times)
                _put(py_samp.indices)

        elif detected_mod == "SEC/GPC":
            sec_samp = sec_core.parse_sec_file(b_data, sample_id=s_id)
            n_points = len(sec_samp.elution_volume)
            _put(sec_core.extract_sec_descriptors(sec_samp))
            excl = sec_samp.indices.get("SEC_Excluded_Area_pct", np.nan)
            if excl is not None and np.isfinite(excl) and excl > 0.05:
                msg = f"OK ({excl:.1f}% of signal area outside the calibrated range excluded)"

        elif detected_mod == "Elemental":
            samps = elemental_core.parse_elemental_table(b_data)
            n_points = len(samps)

            def _elem_desc(s) -> Dict[str, float]:
                vals = {
                    "Elem_C_pct": (s.c_pct, 2), "Elem_H_pct": (s.h_pct, 2), "Elem_N_pct": (s.n_pct, 2),
                    "Elem_S_pct": (s.s_pct, 2), "Elem_O_pct": (s.o_pct, 2), "Elem_H_C": (s.h_c, 4),
                    "Elem_O_C": (s.o_c, 4), "Elem_N_C": (s.n_c, 4), "Elem_S_C": (s.s_c, 4),
                    "Elem_NOSC": (s.nosc, 4), "Elem_AI_bulk": (s.aromaticity_index, 4),
                    "Elem_Q_combustion": (s.combustion_heat_mj_kg, 2),
                }
                out = {}
                for k, (v, nd) in vals.items():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fv):
                        out[k] = round(fv, nd)
                return out

            if samps:
                descriptors.update(_elem_desc(samps[0]))
                if len(samps) > 1:
                    multi_samples = {s.sample_id: _elem_desc(s) for s in samps}

        else:
            # Общий CSV: строки = образцы, если есть столбец идентификатора
            df_gen = pd.read_csv(io.BytesIO(b_data), sep=None, engine="python")
            n_points = len(df_gen)
            multi_samples, class_labels = _rows_as_samples(df_gen)
            if multi_samples:
                msg = f"Generic table: {len(multi_samples)} samples (rows-as-samples)"
            else:
                multi_samples = None
                status = "warning"
                msg = ("Unrecognised format: no Sample_ID column and no known spectral/chromatographic header — "
                       "file not merged (column means are not used as descriptors)")

    except Exception as exc:
        status = "error"
        msg = f"Processing error: {str(exc)}"

    res_dict: Dict[str, Any] = {
        "status": status,
        "filename": filename,
        "sample_id": s_id,
        "modality": detected_mod,
        "n_points": n_points,
        "descriptors": descriptors,
        "message": msg,
    }
    if kind is not None:
        res_dict["kind"] = kind
    if multi_samples:
        res_dict["multi_samples"] = multi_samples
    if class_labels:
        res_dict["class_labels"] = class_labels
    return res_dict


def batch_screen_files(
    files: List[Tuple[str, bytes]],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    fticr_max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Пакетная обработка списка файлов [(filename, bytes_data), ...].
    Формирует объединенную матрицу дескрипторов и сводную таблицу статуса.
    Отсутствующие дескрипторы остаются NaN (не заменяются нулями).
    """
    summary_records = []
    samples_descriptors: Dict[str, Dict[str, float]] = {}
    file_labels: Dict[str, str] = {}

    total = len(files)
    for idx, (fname, fbytes) in enumerate(files, 1):
        if progress_callback:
            progress_callback(idx / max(total, 1), f"Обработка {fname} ({idx}/{total})...")

        res = process_single_file(fbytes, fname, fticr_max_peaks=fticr_max_peaks)
        s_id = res["sample_id"]
        mod = res["modality"]
        stat = res["status"]
        n_pts = res["n_points"]
        descs = res["descriptors"]
        msg = res["message"]

        n_desc = len(descs)
        if res.get("multi_samples"):
            n_desc = max((len(v) for v in res["multi_samples"].values()), default=0)
        summary_records.append({
            "Filename": fname,
            "Sample_ID": s_id if not res.get("multi_samples") else f"{len(res['multi_samples'])} samples",
            "Detected_Modality": mod,
            "Status": stat,
            "Points_Detected": n_pts,
            "Descriptors_Count": n_desc,
            "Notes": msg,
        })

        if "multi_samples" in res and res["multi_samples"]:
            for ms_id, ms_descs in res["multi_samples"].items():
                if ms_id not in samples_descriptors:
                    samples_descriptors[ms_id] = {}
                samples_descriptors[ms_id].update(ms_descs)
            for ms_id, lab in res.get("class_labels", {}).items():
                file_labels.setdefault(ms_id, lab)
        elif descs:
            if s_id not in samples_descriptors:
                samples_descriptors[s_id] = {}
            samples_descriptors[s_id].update(descs)

    summary_df = pd.DataFrame(summary_records)

    # Сборка единого датасета fused_df
    fused_rows = []
    for s_id, d_dict in samples_descriptors.items():
        row_data = {"Sample_ID": s_id}
        if s_id in file_labels:
            row_data["Class_Target"] = file_labels[s_id]
        else:
            # Автоматическая метка класса по ключевым словам
            s_low = s_id.lower()
            if any(k in s_low for k in ["lignin", "sludge", "impact", "waste", "шлам"]):
                row_data["Class_Target"] = "Sludge_Lignin"
            elif any(k in s_low for k in ["baikal", "control", "background", "pristine", "фон"]):
                row_data["Class_Target"] = "Baikal_Background"
            else:
                row_data["Class_Target"] = "Unassigned"

        row_data.update(d_dict)
        fused_rows.append(row_data)

    fused_df = pd.DataFrame(fused_rows)
    if not fused_df.empty:
        lead_cols = ["Sample_ID", "Class_Target"]
        feat_cols = [c for c in fused_df.columns if c not in lead_cols]
        # Без fillna(0.0): отсутствие измерения ≠ нулевое значение дескриптора
        fused_df = fused_df[lead_cols + sorted(feat_cols)]

    unique_modalities = sorted(list(set(r["Detected_Modality"] for r in summary_records)))

    return {
        "summary_df": summary_df,
        "fused_df": fused_df,
        "statistics": {
            "total_files": total,
            "successful": len([r for r in summary_records if r["Status"] == "success"]),
            "warnings": len([r for r in summary_records if r["Status"] == "warning"]),
            "failed": len([r for r in summary_records if r["Status"] == "error"]),
            "unique_samples": len(fused_df),
            "modalities_found": unique_modalities,
            "missing_values": int(fused_df.drop(columns=["Sample_ID", "Class_Target"], errors="ignore").isna().sum().sum()) if not fused_df.empty else 0,
        },
    }


def batch_screen_directory(
    directory_path: str,
    recursive: bool = True,
    extensions: Tuple[str, ...] = (".csv", ".tsv", ".txt", ".xy", ".jdx", ".dx"),
    progress_callback: Optional[Callable[[float, str], None]] = None,
    fticr_max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Сканирует локальную директорию на диске и выполняет полный пакетный скрининг.
    """
    p = Path(directory_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Директория '{directory_path}' не существует.")

    matched_files = []
    pattern = "**/*" if recursive else "*"
    for item in p.glob(pattern):
        if item.is_file() and item.suffix.lower() in extensions:
            matched_files.append(item)
            if len(matched_files) >= 500:
                break

    files_data = []
    tot_bytes = 0
    for f in matched_files:
        try:
            sz = f.stat().st_size
            if sz > 50_000_000:  # пропуск одиночных файлов > 50 МБ
                continue
            tot_bytes += sz
            if tot_bytes > 200_000_000:  # лимит 200 МБ для безопасности памяти
                break
            with open(f, "rb") as fh:
                b = fh.read()
            files_data.append((f.name, b))
        except Exception:
            continue

    return batch_screen_files(files_data, progress_callback=progress_callback, fticr_max_peaks=fticr_max_peaks)


def export_batch_results_zip(
    screening_result: Dict[str, Any],
    metadata: Optional[Dict] = None,
    lang: str = "ru",
) -> bytes:
    """
    Формирует готовый ZIP-архив с результатами пакетного скрининга:
    - Fused_Matrix.csv (объединенная матрица признаков для ChemoSuite ML)
    - Screening_Summary.csv (паспорт-реестр обработки файлов)
    - Ecological_Passport.html (интерактивный аналитический паспорт)
    - Analytical_Passport.xlsx (многостраничный паспорт Excel)
    """
    fused_df = screening_result.get("fused_df", pd.DataFrame())
    summary_df = screening_result.get("summary_df", pd.DataFrame())

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Сводная матрица дескрипторов
        if not fused_df.empty:
            zf.writestr("ChemoSuite_Batch_Fused_Matrix.csv", fused_df.to_csv(index=False).encode("utf-8-sig"))

        # 2. Реестр обработки файлов
        if not summary_df.empty:
            zf.writestr("ChemoSuite_Batch_Screening_Summary.csv", summary_df.to_csv(index=False).encode("utf-8-sig"))

        # 3. Интерактивный экологический паспорт HTML
        if not fused_df.empty:
            html_pass = report_generator.generate_ecological_passport_html(
                fused_df,
                metadata=metadata,
                lang=lang,
            )
            zf.writestr("ChemoSuite_Batch_Ecological_Passport.html", html_pass.encode("utf-8"))

            # 4. Аналитический паспорт Excel
            excel_pass = report_generator.generate_excel_passport(
                fused_df,
                metadata=metadata,
                lang=lang,
            )
            zf.writestr("ChemoSuite_Batch_Analytical_Passport.xlsx", excel_pass)

    zip_buf.seek(0)
    return zip_buf.getvalue()
