"""
ChemoSuite Project I/O Module (project_io.py)
=============================================
Сериализация и десериализация полного состояния аналитической сессии
в компактный переносимый файл проекта (.chemo / ZIP).

Сохраняет:
1. Базу масс-спектров FT-ICR MS (исходные байты файлов, сырые таблицы, приписанные формулы).
2. Оптические дескрипторы EEM-PARAFAC (флуоресценция) и UV-Vis (поглощение).
3. Сводную мультимодальную матрицу признаков (Data Fusion).
4. Обученные хемометрические модели (PLS-DA / OPLS-DA: scores, loadings, Hotelling, VIP, S-Plot).
5. Результаты пермутационного тестирования (Q2 эмпирические распределения и p-value).
6. Метаданные проекта (версия, дата, параметры, имена образцов).
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

PROJECT_FORMAT_VERSION = "1.0.0"
PLATFORM_SIGNATURE = "ChemoSuite Unified Analytical Engine"


def _to_json_serializable(val: Any) -> Any:
    """Рекурсивная конвертация numpy массивов, скаляров и вложенных структур в JSON-совместимые типы."""
    if hasattr(val, "to_numpy"):
        return [_to_json_serializable(x) for x in val.to_numpy().tolist()]
    elif isinstance(val, np.ndarray):
        return [_to_json_serializable(x) for x in val.tolist()]
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val) if not np.isnan(val) else None
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, pd.DataFrame):
        return val.to_dict(orient="split")
    elif isinstance(val, pd.Series):
        return val.to_dict()
    elif isinstance(val, dict):
        return {str(k): _to_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [_to_json_serializable(v) for v in val]
    return val


def _restore_json_obj(val: Any) -> Any:
    """Восстановление списков обратно в numpy arrays для ключей, содержащих численные координаты/оценки."""
    if isinstance(val, dict):
        return {k: _restore_json_obj(v) for k, v in val.items()}
    elif isinstance(val, list):
        # Если список чисел или вложенных списков чисел, преобразуем в np.ndarray
        if len(val) > 0 and all(isinstance(x, (int, float, list)) for x in val if x is not None):
            try:
                arr = np.array(val)
                if arr.dtype.kind in ("i", "f"):
                    return arr
            except Exception:
                pass
        return [_restore_json_obj(v) for v in val]
    return val


def save_chemo_project(
    session_data: Dict[str, Any],
    project_name: str = "ChemoSuite_Project",
    notes: str = "",
) -> bytes:
    """
    Упаковывает текущее состояние сессии ChemoSuite в бинарный поток .chemo (ZIP).

    Parameters
    ----------
    session_data : Dict[str, Any]
        Словарь состояния (обычно копия или подмножество st.session_state).
    project_name : str
        Название проекта.
    notes : str
        Пользовательские заметки или описание проекта.

    Returns
    -------
    bytes
        Байтовый массив упакованного .chemo архива.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Формирование метаданных
        spectra_db = session_data.get("spectra_db", {}) or {}
        eem_df = session_data.get("eem_ml_descriptors", None)
        uv_df = session_data.get("uv_ml_descriptors", None)
        fused_df = session_data.get("fused_data", None)
        pls_res = session_data.get("pls_results", None)
        perm_res = session_data.get("perm_results", None)

        metadata = {
            "format_version": PROJECT_FORMAT_VERSION,
            "platform": PLATFORM_SIGNATURE,
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "lang": session_data.get("lang", "ru"),
            "summary": {
                "n_spectra": len(spectra_db),
                "spectra_names": list(spectra_db.keys()),
                "has_eem": eem_df is not None and not eem_df.empty,
                "has_uv": uv_df is not None and not uv_df.empty,
                "has_fused": fused_df is not None and not fused_df.empty,
                "fused_source": session_data.get("fused_source", "Unknown"),
                "has_trained_model": pls_res is not None,
                "model_type": pls_res.get("model_type") if isinstance(pls_res, dict) else None,
                "has_permutation": perm_res is not None,
            },
        }
        zf.writestr("chemo_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

        # 2. Сериализация базы масс-спектров (spectra_db)
        spectra_meta = []
        for idx, (spec_name, spec_dict) in enumerate(spectra_db.items()):
            if not isinstance(spec_dict, dict):
                continue

            entry_meta = {
                "id": idx,
                "name": spec_name,
                "raw_path": spec_dict.get("raw_path", ""),
                "has_bytes": spec_dict.get("file_bytes") is not None,
                "has_raw_df": spec_dict.get("raw_df") is not None,
                "has_parsed_peaks": spec_dict.get("parsed_peaks") is not None,
                "has_assigned_df": spec_dict.get("assigned_df") is not None,
            }
            spectra_meta.append(entry_meta)

            # Сохранение исходных байт
            file_bytes = spec_dict.get("file_bytes")
            if file_bytes is not None:
                zf.writestr(f"spectra_db/{idx}_raw.bin", file_bytes)

            # Сохранение таблиц
            raw_df = spec_dict.get("raw_df")
            if isinstance(raw_df, pd.DataFrame):
                zf.writestr(f"spectra_db/{idx}_raw_df.csv", raw_df.to_csv(index=False))

            parsed_peaks = spec_dict.get("parsed_peaks")
            if isinstance(parsed_peaks, pd.DataFrame):
                zf.writestr(f"spectra_db/{idx}_parsed_peaks.csv", parsed_peaks.to_csv(index=False))

            assigned_df = spec_dict.get("assigned_df")
            if isinstance(assigned_df, pd.DataFrame):
                zf.writestr(f"spectra_db/{idx}_assigned_df.csv", assigned_df.to_csv(index=False))

        zf.writestr("spectra_db/index.json", json.dumps(spectra_meta, ensure_ascii=False, indent=2))

        # 3. Дескрипторы EEM и UV-Vis
        if isinstance(eem_df, pd.DataFrame) and not eem_df.empty:
            zf.writestr("descriptors/eem_ml_descriptors.csv", eem_df.to_csv(index=False))

        if isinstance(uv_df, pd.DataFrame) and not uv_df.empty:
            zf.writestr("descriptors/uv_ml_descriptors.csv", uv_df.to_csv(index=False))

        # 4. Мультимодальная матрица слияния данных (Data Fusion)
        if isinstance(fused_df, pd.DataFrame) and not fused_df.empty:
            zf.writestr("fusion/fused_data.csv", fused_df.to_csv(index=False))
            fused_src = str(session_data.get("fused_source", "Session"))
            zf.writestr("fusion/fused_source.txt", fused_src.encode("utf-8"))

        # 5. Хемометрическая модель и результаты
        if isinstance(pls_res, dict):
            # Отдельно сохраняем DataFrames модели
            vip_df = pls_res.get("VIP_df")
            if vip_df is None:
                vip_df = pls_res.get("vip_df")
            if isinstance(vip_df, pd.DataFrame):
                zf.writestr("models/vip_df.csv", vip_df.to_csv(index=False))

            s_plot_df = pls_res.get("S_Plot_df")
            if s_plot_df is None:
                s_plot_df = pls_res.get("s_plot_df")
            if isinstance(s_plot_df, pd.DataFrame):
                zf.writestr("models/s_plot_df.csv", s_plot_df.to_csv(index=False))

            # Остальные числовые параметры, матрицы и координаты
            pls_dict_clean = {}
            for k, v in pls_res.items():
                if k in ("VIP_df", "vip_df", "s_plot_df", "S_Plot_df"):
                    continue
                pls_dict_clean[k] = _to_json_serializable(v)

            zf.writestr("models/pls_results.json", json.dumps(pls_dict_clean, ensure_ascii=False, indent=2))

        # Вектора меток Y и Sample IDs
        pls_sample_ids = session_data.get("pls_sample_ids")
        if pls_sample_ids is not None:
            zf.writestr("models/pls_sample_ids.json", json.dumps(_to_json_serializable(pls_sample_ids)))

        pls_y_input = session_data.get("pls_y_input")
        if pls_y_input is not None:
            zf.writestr("models/pls_y_input.json", json.dumps(_to_json_serializable(pls_y_input)))

        # Результаты пермутации
        if isinstance(perm_res, dict):
            perm_clean = _to_json_serializable(perm_res)
            zf.writestr("models/perm_results.json", json.dumps(perm_clean, ensure_ascii=False, indent=2))

    return buf.getvalue()


def get_chemo_project_summary(source: Union[bytes, io.BytesIO, str]) -> Dict[str, Any]:
    """
    Быстрое извлечение метаданных проекта без полной распаковки всех таблиц.

    Parameters
    ----------
    source : Union[bytes, io.BytesIO, str]
        Байты, поток или путь к файлу .chemo.

    Returns
    -------
    Dict[str, Any]
        Словарь метаданных и сводки проекта.
    """
    if isinstance(source, bytes):
        fp = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        fp = source
    else:
        fp = open(source, "rb")

    try:
        with zipfile.ZipFile(fp, mode="r") as zf:
            if "chemo_metadata.json" not in zf.namelist():
                raise ValueError("Некорректный файл .chemo: отсутствует chemo_metadata.json")
            meta_raw = zf.read("chemo_metadata.json").decode("utf-8")
            return json.loads(meta_raw)
    finally:
        if isinstance(source, str):
            fp.close()


def load_chemo_project(source: Union[bytes, io.BytesIO, str]) -> Dict[str, Any]:
    """
    Распаковывает файл проекта .chemo и возвращает словарь с восстановленным состоянием.

    Parameters
    ----------
    source : Union[bytes, io.BytesIO, str]
        Байты, поток или путь к файлу .chemo.

    Returns
    -------
    Dict[str, Any]
        Словарь, содержащий восстановленные ключи для st.session_state:
        - 'spectra_db'
        - 'eem_ml_descriptors'
        - 'uv_ml_descriptors'
        - 'fused_data'
        - 'fused_source'
        - 'pls_results'
        - 'pls_sample_ids'
        - 'pls_y_input'
        - 'perm_results'
        - 'project_metadata'
    """
    if isinstance(source, bytes):
        fp = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        fp = source
    else:
        fp = open(source, "rb")

    try:
        with zipfile.ZipFile(fp, mode="r") as zf:
            names = set(zf.namelist())

            if "chemo_metadata.json" not in names:
                raise ValueError("Недействительный формат проекта ChemoSuite (.chemo)")

            meta = json.loads(zf.read("chemo_metadata.json").decode("utf-8"))

            restored: Dict[str, Any] = {
                "project_metadata": meta,
                "spectra_db": {},
                "eem_ml_descriptors": None,
                "uv_ml_descriptors": None,
                "fused_data": None,
                "fused_source": meta.get("summary", {}).get("fused_source", "Restored Session"),
                "pls_results": None,
                "pls_sample_ids": None,
                "pls_y_input": None,
                "perm_results": None,
            }

            # 1. Восстановление базы спектров
            if "spectra_db/index.json" in names:
                spectra_meta = json.loads(zf.read("spectra_db/index.json").decode("utf-8"))
                for item in spectra_meta:
                    idx = item["id"]
                    spec_name = item["name"]

                    spec_entry = {
                        "raw_path": item.get("raw_path", ""),
                        "file_bytes": None,
                        "raw_df": None,
                        "parsed_peaks": None,
                        "assigned_df": None,
                    }

                    raw_bin_name = f"spectra_db/{idx}_raw.bin"
                    if raw_bin_name in names:
                        spec_entry["file_bytes"] = zf.read(raw_bin_name)

                    raw_df_name = f"spectra_db/{idx}_raw_df.csv"
                    if raw_df_name in names:
                        spec_entry["raw_df"] = pd.read_csv(io.BytesIO(zf.read(raw_df_name)))

                    parsed_name = f"spectra_db/{idx}_parsed_peaks.csv"
                    if parsed_name in names:
                        spec_entry["parsed_peaks"] = pd.read_csv(io.BytesIO(zf.read(parsed_name)))

                    assigned_name = f"spectra_db/{idx}_assigned_df.csv"
                    if assigned_name in names:
                        spec_entry["assigned_df"] = pd.read_csv(io.BytesIO(zf.read(assigned_name)))

                    restored["spectra_db"][spec_name] = spec_entry

            # 2. Дескрипторы EEM и UV-Vis
            if "descriptors/eem_ml_descriptors.csv" in names:
                restored["eem_ml_descriptors"] = pd.read_csv(io.BytesIO(zf.read("descriptors/eem_ml_descriptors.csv")))

            if "descriptors/uv_ml_descriptors.csv" in names:
                restored["uv_ml_descriptors"] = pd.read_csv(io.BytesIO(zf.read("descriptors/uv_ml_descriptors.csv")))

            # 3. Fused Data
            if "fusion/fused_data.csv" in names:
                restored["fused_data"] = pd.read_csv(io.BytesIO(zf.read("fusion/fused_data.csv")))

            if "fusion/fused_source.txt" in names:
                restored["fused_source"] = zf.read("fusion/fused_source.txt").decode("utf-8")

            # 4. Модель PLS-DA / OPLS-DA
            if "models/pls_results.json" in names:
                pls_json = json.loads(zf.read("models/pls_results.json").decode("utf-8"))
                pls_res = _restore_json_obj(pls_json)

                # Восстановление DataFrame VIP
                if "models/vip_df.csv" in names:
                    vdf = pd.read_csv(io.BytesIO(zf.read("models/vip_df.csv")))
                    pls_res["VIP_df"] = vdf
                    pls_res["vip_df"] = vdf

                # Восстановление DataFrame S-plot
                if "models/s_plot_df.csv" in names:
                    sdf = pd.read_csv(io.BytesIO(zf.read("models/s_plot_df.csv")))
                    pls_res["s_plot_df"] = sdf
                    pls_res["S_Plot_df"] = sdf

                # Преобразование матриц и координат в numpy arrays
                for arr_key in [
                    "scores_t1", "t1", "scores_t2", "t2", "scores_t3", "t3",
                    "ellipse_x", "ell_x", "ellipse_y",
                    "ell_3d_x", "ell_3d_y", "ell_3d_z",
                    "ell_x_3d", "ell_y_3d", "ell_z_3d",
                    "Confusion_Matrix", "y_pred", "y_cv_pred",
                ]:
                    if arr_key in pls_res and isinstance(pls_res[arr_key], (list, tuple)):
                        pls_res[arr_key] = np.array(pls_res[arr_key])

                restored["pls_results"] = pls_res

            if "models/pls_sample_ids.json" in names:
                ids_json = json.loads(zf.read("models/pls_sample_ids.json").decode("utf-8"))
                restored["pls_sample_ids"] = np.array(ids_json)

            if "models/pls_y_input.json" in names:
                y_json = json.loads(zf.read("models/pls_y_input.json").decode("utf-8"))
                restored["pls_y_input"] = np.array(y_json)

            # 5. Результаты пермутации
            if "models/perm_results.json" in names:
                perm_json = json.loads(zf.read("models/perm_results.json").decode("utf-8"))
                perm_res = _restore_json_obj(perm_json)
                if "perm_q2" in perm_res and isinstance(perm_res["perm_q2"], (list, tuple)):
                    perm_res["perm_q2"] = np.array(perm_res["perm_q2"])
                restored["perm_results"] = perm_res

            return restored
    finally:
        if isinstance(source, str):
            fp.close()
