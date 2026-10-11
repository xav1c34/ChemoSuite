"""
batch_screener.py — Автономный Batch Processing & фоновый пайплайн скрининга директорий ChemoSuite 3.0.
Позволяет осуществлять пакетное сканирование файлов спектров и матриц, автоматическое
распознавание аналитических модальностей (все 8 методов), фильтрацию, извлечение дескрипторов
и сборку объединенной хемометрической матрицы (Data Staging) без рутинных ручных действий.
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


def detect_file_modality(
    file_bytes_or_str: Union[str, bytes],
    filename: str = "",
) -> Dict[str, Any]:
    """
    Интеллектуальный анализатор типа аналитического файла.
    Определяет принадлежность к одному из 8 методов по структуре таблицы и ключевым словам.
    """
    fn_low = filename.lower()
    stem_low = Path(filename).stem.lower()

    # 1. Проверка по суффиксам и токенам имени файла
    tokens = re.split(r"[-_.\s]+", stem_low)
    if any(k in tokens for k in ["fticr", "ftms", "mass", "ms"]):
        return {"modality": "FT-ICR MS", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["eem", "parafac", "fluo", "fluor"]):
        return {"modality": "EEM-PARAFAC", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["uv", "uvvis", "absorbance"]):
        return {"modality": "UV-Vis", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["ftir", "raman", "ir", "wavenumber"]):
        return {"modality": "FTIR", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["nmr", "13c", "1h", "cpmas"]) or filename.lower().endswith((".jdx", ".dx")):
        return {"modality": "NMR", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["tga", "dtg", "dsc", "pygcms", "thermal", "py"]):
        return {"modality": "Thermal", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["sec", "gpc", "mwd", "chrom"]):
        return {"modality": "SEC/GPC", "confidence": "high", "reason": "Filename token"}
    if any(k in tokens for k in ["chns", "chnso", "elemental", "elem"]):
        return {"modality": "Elemental", "confidence": "high", "reason": "Filename token"}

    # 2. Декодирование содержимого
    if isinstance(file_bytes_or_str, bytes):
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                text = file_bytes_or_str.decode(enc)
                break
            except Exception:
                continue
        else:
            text = file_bytes_or_str.decode("utf-8", errors="replace")
    else:
        text = str(file_bytes_or_str)

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()][:25]
    if not lines:
        return {"modality": "Unknown", "confidence": "low", "reason": "Empty file"}

    header_text = " ".join(lines[:5]).lower()

    # JCAMP-DX
    if "##jcamp-dx" in header_text or "##data type= nmr" in header_text:
        return {"modality": "NMR", "confidence": "high", "reason": "JCAMP-DX Header"}

    # Элементный анализ
    if any(k in header_text for k in ["c_pct", "c,h,n", "c_wt", "n_wt", "ash_pct", "h/c,o/c"]) or (
        "c" in header_text and "h" in header_text and "n" in header_text and "s" in header_text and "o" in header_text
    ):
        return {"modality": "Elemental", "confidence": "high", "reason": "CHNS-O Table Headers"}

    # FT-ICR MS (m/z, mass, intensity)
    if any(k in header_text for k in ["m/z", "mass", "m_z", "theor_mass", "calc_mass"]) and any(
        k in header_text for k in ["intensity", "abundance", "intens", "resol"]
    ):
        return {"modality": "FT-ICR MS", "confidence": "high", "reason": "Mass Spectrometry Headers"}

    # FTIR (wavenumber, cm-1, cm1, transmittance, absorbance)
    if any(k in header_text for k in ["wavenumber", "cm-1", "cm^-1", "cm1", "волновое"]):
        return {"modality": "FTIR", "confidence": "high", "reason": "FTIR Wavenumber Header"}

    # NMR (chemical_shift, ppm)
    if any(k in header_text for k in ["ppm", "chemical_shift", "chemical shift", "shift", "химсдвиг"]):
        return {"modality": "NMR", "confidence": "high", "reason": "NMR Chemical Shift Header"}

    # Thermal TGA/DTG
    if any(k in header_text for k in ["temp", "temperature", "t, °c", "t(°c)"]) and any(
        k in header_text for k in ["weight", "mass", "tg", "dtg", "масса", "%"]
    ):
        return {"modality": "Thermal", "confidence": "high", "reason": "TGA Temperature/Weight Header"}

    # Thermal Py-GC-MS (retention_time / rt, area, peak, class)
    if any(k in header_text for k in ["retention_time", "retention time", "rt", "время"]) and any(
        k in header_text for k in ["area", "tic", "py_", "monomer", "lignin"]
    ):
        return {"modality": "Thermal", "confidence": "high", "reason": "Py-GC-MS Header"}

    # SEC / GPC (retention time, elution volume, RI, UV)
    if any(k in header_text for k in ["elution", "retention", "v_e", "v_r", "vr", "ve", "объем"]) and any(
        k in header_text for k in ["ri", "uv", "signal", "mw", "dabs"]
    ):
        return {"modality": "SEC/GPC", "confidence": "high", "reason": "SEC Elution Volume Header"}

    # UV-Vis (wavelength, nm, absorbance)
    if any(k in header_text for k in ["wavelength", "длина волны", "lambda", "nm", "нм"]) and any(
        k in header_text for k in ["abs", "absorbance", "поглощение", "оптическая"]
    ):
        return {"modality": "UV-Vis", "confidence": "high", "reason": "UV-Vis Wavelength Header"}

    # EEM Matrix (числа в первой строке как длины волн испускания)
    try:
        first_tokens = lines[0].replace(";", ",").replace("\t", ",").split(",")
        numeric_tokens = [float(t) for t in first_tokens if t.strip().replace(".", "").isdigit()]
        if len(numeric_tokens) >= 5 and all(200 <= v <= 900 for v in numeric_tokens):
            return {"modality": "EEM-PARAFAC", "confidence": "medium", "reason": "EEM Spectral Grid"}
    except Exception:
        pass

    return {"modality": "General / CSV", "confidence": "low", "reason": "Generic tabular data"}


def process_single_file(
    content_bytes_or_str: Union[str, bytes],
    filename: str,
    modality: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Универсальный обработчик отдельного аналитического файла.
    Извлекает вектор дескрипторов и формирует отчет по статусу.
    """
    s_id = sample_id or extract_sample_id_from_filename(filename)
    b_data = content_bytes_or_str if isinstance(content_bytes_or_str, bytes) else content_bytes_or_str.encode("utf-8")

    if not modality or modality == "Unknown":
        detection = detect_file_modality(b_data, filename)
        detected_mod = detection["modality"]
    else:
        detected_mod = modality

    descriptors: Dict[str, float] = {}
    multi_samples: Optional[Dict[str, Dict[str, float]]] = None
    n_points = 0
    status = "success"
    msg = "OK"

    try:
        if detected_mod == "FT-ICR MS":
            df_peaks = fticr_core.parse_uploaded_file(b_data)
            n_points = len(df_peaks)
            if not df_peaks.empty:
                top_p = df_peaks.sort_values("intensity", ascending=False).head(350)
                assigned = fticr_core.fast_formula_assigner(top_p, ppm_tolerance=2.5)
                if not assigned.empty:
                    descriptors = chemo_ml.extract_fticr_descriptors(assigned, s_id)
                else:
                    msg = "No formulas assigned (low signal/noise)"
                    status = "warning"
            else:
                msg = "No mass spectral peaks found"
                status = "warning"

        elif detected_mod == "EEM-PARAFAC":
            # Парсинг матрицы EEM
            df_eem = pd.read_csv(io.BytesIO(b_data))
            eem_samp = eem_core.parse_eem_dataframe(df_eem, sample_id=s_id)
            n_points = int(eem_samp.data.size) if hasattr(eem_samp, "data") else len(df_eem)
            indices = eem_core.calculate_spectral_indices(eem_samp)
            for k, v in indices.items():
                if k != "Sample_ID" and isinstance(v, (int, float)) and not np.isnan(v):
                    descriptors[f"EEM_{k}"] = round(float(v), 4)

        elif detected_mod == "UV-Vis":
            # Парсинг спектра поглощения
            uv_samp = eem_core.parse_uv_vis_spectrum(io.BytesIO(b_data), sample_id=s_id)
            n_points = len(uv_samp.wavelengths)
            uv_res = eem_core.calculate_uv_vis_indices(uv_samp, doc=5.0)
            for k, v in uv_res.items():
                if k != "Sample_ID" and isinstance(v, (int, float)) and not np.isnan(v):
                    descriptors[f"UV_{k}"] = round(float(v), 4)

        elif detected_mod == "FTIR":
            ftir_samp = ftir_core.parse_ftir_file(b_data, sample_id=s_id)
            n_points = len(ftir_samp.wavenumbers)
            indices = ftir_core.calculate_ftir_indices(ftir_samp.wavenumbers, ftir_samp.absorbance, sample_id=s_id)
            for k, v in indices.items():
                if k != "Sample_ID" and isinstance(v, (int, float)) and not np.isnan(v):
                    descriptors[k] = round(float(v), 4)

        elif detected_mod == "NMR":
            nmr_samp = nmr_core.parse_nmr_file(b_data, sample_id=s_id)
            n_points = len(nmr_samp.chemical_shifts)
            desc_nmr = nmr_core.calculate_13c_nmr_descriptors(nmr_samp.chemical_shifts, nmr_samp.intensity, sample_id=s_id)
            for k, v in desc_nmr.items():
                if k != "Sample_ID" and isinstance(v, (int, float)) and not np.isnan(v):
                    descriptors[k] = round(float(v), 4)

        elif detected_mod == "Thermal":
            # Пробуем как TGA, если не вышло — как Py-GC-MS
            tga_samp = thermal_core.parse_tga_file(b_data, sample_id=s_id)
            if len(tga_samp.temperature) > 10:
                n_points = len(tga_samp.temperature)
                for k, v in tga_samp.indices.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        descriptors[k] = round(float(v), 4)
            else:
                py_samp = thermal_core.parse_py_gcms_file(b_data, sample_id=s_id)
                n_points = len(py_samp.retention_times)
                for k, v in py_samp.indices.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        descriptors[k] = round(float(v), 4)

        elif detected_mod == "SEC/GPC":
            sec_samp = sec_core.parse_sec_file(b_data, sample_id=s_id)
            n_points = len(sec_samp.elution_volume)
            for k, v in sec_samp.indices.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    descriptors[k] = round(float(v), 4)

        elif detected_mod == "Elemental":
            samps = elemental_core.parse_elemental_table(b_data)
            n_points = len(samps)
            if samps:
                s0 = samps[0]
                descriptors.update({
                    "Elem_C_pct": round(float(s0.c_pct), 2),
                    "Elem_H_pct": round(float(s0.h_pct), 2),
                    "Elem_N_pct": round(float(s0.n_pct), 2),
                    "Elem_S_pct": round(float(s0.s_pct), 2),
                    "Elem_O_pct": round(float(s0.o_pct), 2),
                    "Elem_H_C": round(float(s0.h_c), 4),
                    "Elem_O_C": round(float(s0.o_c), 4),
                    "Elem_N_C": round(float(s0.n_c), 4),
                    "Elem_S_C": round(float(s0.s_c), 4),
                    "Elem_NOSC": round(float(s0.nosc), 4),
                    "Elem_AI_bulk": round(float(s0.aromaticity_index), 4),
                    "Elem_Q_combustion": round(float(s0.combustion_heat_mj_kg), 2),
                })

                if len(samps) > 1:
                    multi_samples = {}
                    for s in samps:
                        multi_samples[s.sample_id] = {
                            "Elem_C_pct": round(float(s.c_pct), 2),
                            "Elem_H_pct": round(float(s.h_pct), 2),
                            "Elem_N_pct": round(float(s.n_pct), 2),
                            "Elem_S_pct": round(float(s.s_pct), 2),
                            "Elem_O_pct": round(float(s.o_pct), 2),
                            "Elem_H_C": round(float(s.h_c), 4),
                            "Elem_O_C": round(float(s.o_c), 4),
                            "Elem_N_C": round(float(s.n_c), 4),
                            "Elem_S_C": round(float(s.s_c), 4),
                            "Elem_NOSC": round(float(s.nosc), 4),
                            "Elem_AI_bulk": round(float(s.aromaticity_index), 4),
                            "Elem_Q_combustion": round(float(s.combustion_heat_mj_kg), 2),
                        }

        else:
            # Общий CSV файл или матрица признаков
            df_gen = pd.read_csv(io.BytesIO(b_data))
            n_points = len(df_gen)
            if "Sample_ID" in df_gen.columns and len(df_gen) > 1:
                multi_samples = {}
                num_cols = [c for c in df_gen.columns if pd.api.types.is_numeric_dtype(df_gen[c])]
                for _, r in df_gen.iterrows():
                    sid = str(r["Sample_ID"]).strip()
                    multi_samples[sid] = {
                        col: round(float(r[col]), 4)
                        for col in num_cols
                        if not np.isnan(float(r[col]))
                    }
                if len(df_gen) > 0:
                    first_sid = str(df_gen.iloc[0]["Sample_ID"]).strip()
                    descriptors = multi_samples.get(first_sid, {})
            else:
                num_cols = [c for c in df_gen.columns if pd.api.types.is_numeric_dtype(df_gen[c])]
                for col in num_cols[:20]:
                    val = float(df_gen[col].mean())
                    if not np.isnan(val):
                        descriptors[col] = round(val, 4)

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
    if multi_samples:
        res_dict["multi_samples"] = multi_samples
    return res_dict


def batch_screen_files(
    files: List[Tuple[str, bytes]],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Пакетная обработка списка файлов [(filename, bytes_data), ...].
    Формирует объединенную матрицу дескрипторов и сводную таблицу статуса.
    """
    summary_records = []
    samples_descriptors: Dict[str, Dict[str, float]] = {}

    total = len(files)
    for idx, (fname, fbytes) in enumerate(files, 1):
        if progress_callback:
            progress_callback(idx / max(total, 1), f"Обработка {fname} ({idx}/{total})...")

        res = process_single_file(fbytes, fname)
        s_id = res["sample_id"]
        mod = res["modality"]
        stat = res["status"]
        n_pts = res["n_points"]
        descs = res["descriptors"]
        msg = res["message"]

        summary_records.append({
            "Filename": fname,
            "Sample_ID": s_id,
            "Detected_Modality": mod,
            "Status": stat,
            "Points_Detected": n_pts,
            "Descriptors_Count": len(descs),
            "Notes": msg,
        })

        if "multi_samples" in res and res["multi_samples"]:
            for ms_id, ms_descs in res["multi_samples"].items():
                if ms_id not in samples_descriptors:
                    samples_descriptors[ms_id] = {}
                samples_descriptors[ms_id].update(ms_descs)
        elif descs:
            if s_id not in samples_descriptors:
                samples_descriptors[s_id] = {}
            samples_descriptors[s_id].update(descs)

    summary_df = pd.DataFrame(summary_records)

    # Сборка единого датасета fused_df
    fused_rows = []
    for s_id, d_dict in samples_descriptors.items():
        row_data = {"Sample_ID": s_id}
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
        fused_df = fused_df[lead_cols + sorted(feat_cols)].fillna(0.0)

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
        },
    }


def batch_screen_directory(
    directory_path: str,
    recursive: bool = True,
    extensions: Tuple[str, ...] = (".csv", ".tsv", ".txt", ".xy", ".jdx", ".dx"),
    progress_callback: Optional[Callable[[float, str], None]] = None,
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

    return batch_screen_files(files_data, progress_callback=progress_callback)


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
