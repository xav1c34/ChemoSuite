"""
views/fticr_view.py — FT-ICR Ultra-High Resolution Mass Spectrometry Studio.
"""

import os
import io
import urllib.parse
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from views.common import st_df, st_plotly, get_plot_download_buttons, ensure_sample_peaks

try:
    from modules.fticr_core import (
        EXACT_MASSES,
        C13_DIFF,
        H_ION_MASS,
        KMD_BASES,
        TMDS_LIBRARY,
        VAN_KREVELEN_REGIONS,
        get_biomolecular_distribution,
        batch_process_fticr_spectra,
        calculate_descriptors,
        parse_uploaded_file,
        fast_formula_assigner,
        run_formula_assignment,
        compute_kmd,
        compute_vk20_grid,
        align_two_spectra_fast,
        perform_spectral_algebra,
        run_tmds_screening,
        build_tmds_network_graph,
        compute_geochemical_vector_fluxes,
        find_transformation_pathways,
        get_calibrant_library,
    )
except ImportError:
    from fticr_core import (
        EXACT_MASSES,
        C13_DIFF,
        H_ION_MASS,
        KMD_BASES,
        TMDS_LIBRARY,
        VAN_KREVELEN_REGIONS,
        get_biomolecular_distribution,
        batch_process_fticr_spectra,
        calculate_descriptors,
        parse_uploaded_file,
        fast_formula_assigner,
        run_formula_assignment,
        compute_kmd,
        compute_vk20_grid,
        align_two_spectra_fast,
        perform_spectral_algebra,
        run_tmds_screening,
        build_tmds_network_graph,
        compute_geochemical_vector_fluxes,
        find_transformation_pathways,
        get_calibrant_library,
    )

try:
    from modules import chemo_pubchem
    CHEMO_PUBCHEM_AVAILABLE = True
except ImportError:
    chemo_pubchem = None
    CHEMO_PUBCHEM_AVAILABLE = False

try:
    from modules import chemo_ml
    CHEMO_ML_AVAILABLE = True
except ImportError:
    chemo_ml = None
    CHEMO_ML_AVAILABLE = False

try:
    from modules.open_spectrometry_formats import parse_mzml_content, parse_jcamp_dx, export_mztab_m
    OPEN_FORMATS_AVAILABLE = True
except ImportError:
    parse_mzml_content = None
    parse_jcamp_dx = None
    export_mztab_m = None
    OPEN_FORMATS_AVAILABLE = False

STORAGE_DIR = "uploaded_spectra"
os.makedirs(STORAGE_DIR, exist_ok=True)


def render_fticr_studio(T: dict, lang: str):
    """Render Module 1: FT-ICR MS Studio."""
    with st.sidebar:
        st.title(T[lang]["sidebar_mgr"])

        uploaded_files = st.file_uploader(
            T[lang]["uploader_label"],
            type=["csv", "txt", "tsv", "xy", "mzml", "mzxml", "jdx", "dx"],
            accept_multiple_files=True,
            key="fticr_uploader_files"
        )
        if uploaded_files:
            latest_uploaded_name = None
            for uf in uploaded_files:
                save_path = os.path.join(STORAGE_DIR, uf.name)
                b_content = uf.getvalue()
                existing = st.session_state["spectra_db"].get(uf.name)
                if existing is not None and existing.get("file_bytes") == b_content:
                    continue

                parsed_custom_df = None
                fn_lower = uf.name.lower()
                if OPEN_FORMATS_AVAILABLE:
                    if fn_lower.endswith((".mzml", ".mzxml")) and parse_mzml_content is not None:
                        try:
                            parsed_custom_df = parse_mzml_content(b_content)
                        except Exception as e_mz:
                            st.warning(f"Ошибка mzML {uf.name}: {e_mz}")
                    elif fn_lower.endswith((".jdx", ".dx")) and parse_jcamp_dx is not None:
                        try:
                            parsed_custom_df = parse_jcamp_dx(b_content)
                        except Exception as e_jx:
                            st.warning(f"Ошибка JCAMP {uf.name}: {e_jx}")

                with open(save_path, "wb") as f:
                    f.write(b_content)
                s_new = {
                    "raw_path": save_path,
                    "file_bytes": b_content,
                    "parsed_peaks": parsed_custom_df,
                    "assigned_df": None,
                    "raw_df": parsed_custom_df,
                }
                if parsed_custom_df is None:
                    ensure_sample_peaks(s_new)
                st.session_state["spectra_db"][uf.name] = s_new
                latest_uploaded_name = uf.name

            if latest_uploaded_name and st.session_state.get("_last_selected_upload") != latest_uploaded_name:
                st.session_state["active_spectrum_selector"] = latest_uploaded_name
                st.session_state["_last_selected_upload"] = latest_uploaded_name

        total_in_db = len(st.session_state.get("spectra_db", {}))
        if total_in_db > 0:
            with st.expander(
                f"💾 {'Спектры в памяти сессии' if lang == 'ru' else 'Spectra in Session'}: **{total_in_db}**",
                expanded=True
            ):
                st.caption(
                    "Файлы сохраняются в сессии и доступны во всех вкладках платформы."
                    if lang == "ru" else
                    "Files are permanently stored in session and accessible across all tabs."
                )
                for s_name, s_obj in list(st.session_state["spectra_db"].items()):
                    c_col1, c_col2 = st.columns([3, 1])
                    with c_col1:
                        p_len = len(s_obj["parsed_peaks"]) if (s_obj.get("parsed_peaks") is not None and not s_obj["parsed_peaks"].empty) else "—"
                        has_form = "🧪" if (s_obj.get("assigned_df") is not None and not s_obj["assigned_df"].empty) else ""
                        is_cur = "👉 " if s_name == st.session_state.get("active_spectrum_selector") else ""
                        st.markdown(f"**{is_cur}`{s_name}`** ({p_len} {has_form})")
                    with c_col2:
                        if st.button("🗑️", key=f"del_quick_{s_name}", help="Удалить" if lang == "ru" else "Delete"):
                            if os.path.exists(s_obj.get("raw_path", "")):
                                try: os.remove(s_obj["raw_path"])
                                except Exception: pass
                            del st.session_state["spectra_db"][s_name]
                            if st.session_state.get("active_spectrum_selector") == s_name:
                                remaining = list(st.session_state["spectra_db"].keys())
                                st.session_state["active_spectrum_selector"] = remaining[0] if remaining else None
                            st.rerun()

        with st.expander(T[lang]["folder_expander"], expanded=False):
            local_folder = st.text_input(
                T[lang]["folder_input"],
                placeholder="C:/data/spectra" if lang == "en" else "например, C:/data/spectra",
                help=T[lang]["folder_help"],
            )
            if st.button(T[lang]["folder_btn"]):
                if local_folder and os.path.isdir(local_folder):
                    added_count = 0
                    for root, _, files in os.walk(local_folder):
                        for file in files:
                            if file.lower().endswith((".csv", ".txt", ".tsv", ".xy", ".mzml", ".jdx")) and file not in st.session_state["spectra_db"]:
                                src_path = os.path.join(root, file)
                                dst_path = os.path.join(STORAGE_DIR, file)
                                try:
                                    with open(src_path, "rb") as sf: content = sf.read()
                                    with open(dst_path, "wb") as df: df.write(content)
                                    s_dir_new = {
                                        "raw_path": dst_path, "file_bytes": content,
                                        "parsed_peaks": None, "assigned_df": None, "raw_df": None,
                                    }
                                    ensure_sample_peaks(s_dir_new)
                                    st.session_state["spectra_db"][file] = s_dir_new
                                    added_count += 1
                                except Exception:
                                    pass
                    st.success(T[lang]["folder_success"].format(n=added_count))
                    st.rerun()
                else:
                    st.error(T[lang]["folder_err"])

        with st.expander(T[lang]["batch_zip_expander"], expanded=False):
            batch_zip_file = st.file_uploader(
                T[lang]["batch_zip_uploader"],
                type=["zip"],
                key="batch_zip_uploader_key",
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                b_ion = st.selectbox(T[lang]["batch_ion_mode"], ["ESI(-)", "ESI(+)"], index=0, key="batch_ion_mode_key")
                b_ppm = st.number_input(T[lang]["batch_ppm_tol"], min_value=0.1, max_value=20.0, value=2.0, step=0.5, key="batch_ppm_key")
            with col_b2:
                b_min_int = st.number_input(T[lang]["batch_min_int"], min_value=0.0, value=0.0, format="%.2e", key="batch_min_int_key")
                b_mz_min = st.number_input("m/z min:", min_value=50.0, max_value=2000.0, value=150.0, step=50.0, key="batch_mz_min_key")
            b_mz_max = st.number_input("m/z max:", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0, key="batch_mz_max_key")

            if batch_zip_file is not None:
                if st.button(T[lang]["batch_run_btn"], type="primary", key="btn_run_batch_zip"):
                    with st.spinner("Выполняется пакетная обработка спектров..." if lang == "ru" else "Batch processing spectra..."):
                        summary_df, processed_dict = batch_process_fticr_spectra(
                            batch_zip_file.getvalue(),
                            ion_mode=b_ion,
                            ppm_tolerance=b_ppm,
                            min_intensity=b_min_int,
                            mz_range=(b_mz_min, b_mz_max),
                            lang=lang,
                        )
                        if not summary_df.empty:
                            for s_name, s_data in processed_dict.items():
                                save_path = os.path.join(STORAGE_DIR, s_name)
                                try:
                                    with open(save_path, "wb") as f_out:
                                        f_out.write(s_data["file_bytes"])
                                except Exception:
                                    pass
                                s_data["raw_path"] = save_path
                                st.session_state["spectra_db"][s_name] = s_data

                            st.session_state["fticr_batch_summary"] = summary_df
                            st.success(T[lang]["batch_success"].format(n=len(summary_df)))
                        else:
                            st.warning("В архиве не найдено подходящих спектральных файлов или не удалось приписать формулы." if lang == "ru" else "No valid spectra found in archive or formula assignment failed.")

            if "fticr_batch_summary" in st.session_state and not st.session_state["fticr_batch_summary"].empty:
                b_sum = st.session_state["fticr_batch_summary"]
                st.markdown(f"**{T[lang]['batch_table_title']}**")
                st_df(b_sum.head(5))
                st.download_button(
                    label=T[lang]["batch_desc_dl_csv"],
                    data=b_sum.to_csv(index=False).encode("utf-8"),
                    file_name="chemo_fticr_block.csv",
                    mime="text/csv",
                    key="dl_batch_fticr_summary_btn",
                )
                if st.button(T[lang]["batch_transfer_ml_btn"], key="btn_transfer_batch_to_ml"):
                    st.session_state["fticr_ml_descriptors"] = b_sum
                    st.success(T[lang]["batch_transfer_success"])

        all_spectra = list(st.session_state["spectra_db"].keys())
        if all_spectra:
            st.markdown("---")
            search_query = st.text_input(T[lang]["search_label"], placeholder=T[lang]["search_ph"]).strip().lower()
            available_spectra = [s for s in all_spectra if search_query in s.lower()] if search_query else all_spectra

            if not available_spectra:
                st.warning(T[lang]["search_empty"])
                active_spectrum_name, current_sample = None, None
            else:
                curr_sel = st.session_state.get("active_spectrum_selector")
                sel_idx = available_spectra.index(curr_sel) if (curr_sel and curr_sel in available_spectra) else 0
                active_spectrum_name = st.selectbox(
                    T[lang]["active_sample"],
                    options=available_spectra,
                    index=sel_idx,
                    key="active_spectrum_selector"
                )
                current_sample = st.session_state["spectra_db"][active_spectrum_name]

                if st.button(T[lang]["del_sample"], type="secondary"):
                    if os.path.exists(current_sample["raw_path"]):
                        try: os.remove(current_sample["raw_path"])
                        except Exception: pass
                    del st.session_state["spectra_db"][active_spectrum_name]
                    remaining = list(st.session_state["spectra_db"].keys())
                    st.session_state["active_spectrum_selector"] = remaining[0] if remaining else None
                    st.rerun()

                with st.expander(T[lang]["parse_expander"], expanded=False):
                    delim_opts = ["Auto" if lang == "en" else "Авто (автоопределение)", "Tab (\\t)" if lang == "en" else "Табуляция (\\t)",
                                  "Comma (,)" if lang == "en" else "Запятая (,)", "Semicolon (;)" if lang == "en" else "Точка с запятой (;)",
                                  "Space" if lang == "en" else "Пробел"]
                    delimiter = st.selectbox(T[lang]["delimiter"], delim_opts, index=0, key=f"delim_{active_spectrum_name}")
                    decimal_sep = st.selectbox(T[lang]["decimal_sep"], [".", ","], index=0, key=f"dec_{active_spectrum_name}")
                    has_header = st.checkbox(T[lang]["has_header"], value=True, key=f"head_{active_spectrum_name}")

                try:
                    if current_sample.get("parsed_peaks") is None:
                        raw_df = parse_uploaded_file(current_sample["file_bytes"], delimiter, decimal_sep, has_header)
                        current_sample["raw_df"] = raw_df
                    else:
                        raw_df = current_sample.get("raw_df", current_sample["parsed_peaks"])

                    if raw_df is None or raw_df.empty:
                        st.warning("Внимание: Файл пуст или формат не распознан." if lang == "ru" else "Warning: File is empty or failed to parse.")
                        current_sample["parsed_peaks"] = None
                    else:
                        st.write(T[lang]["preview_caption"])
                        st_df(raw_df.head(5))

                        col_names = list(raw_df.columns)
                        def_mz_idx = 0
                        def_int_idx = 1 if len(col_names) > 1 else 0
                        for idx, cname in enumerate(col_names):
                            cn_low = str(cname).lower()
                            if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]): def_mz_idx = idx
                            elif any(k in cn_low for k in ["int", "i", "count", "abund"]): def_int_idx = idx

                        col_mz = st.selectbox(T[lang]["col_mz"], col_names, index=def_mz_idx, key=f"mzcol_{active_spectrum_name}")
                        col_int = st.selectbox(T[lang]["col_int"], col_names, index=def_int_idx, key=f"intcol_{active_spectrum_name}")
                        if col_mz == col_int: st.warning(T[lang]["same_col_warn"])

                        st.markdown("---")
                        st.subheader(T[lang]["filtering_header"])

                        norm_choice = st.selectbox(T[lang]["norm_mode_label"], [T[lang]["norm_base"], T[lang]["norm_tic"], T[lang]["norm_raw"]], index=0, key=f"norm_mode_{active_spectrum_name}")
                        clean_mass = pd.to_numeric(raw_df[col_mz], errors="coerce")
                        clean_int = pd.to_numeric(raw_df[col_int], errors="coerce")
                        valid_mask = clean_mass.notna() & clean_int.notna()

                        valid_df = pd.DataFrame({"mass": clean_mass[valid_mask].astype(float), "intensity": clean_int[valid_mask].astype(float)})
                        valid_df = valid_df[valid_df["mass"] > 0]
                        if valid_df.empty:
                            st.warning("Внимание: В выбранных колонках не найдено положительных числовых данных." if lang == "ru" else "Warning: No valid numeric data found in selected columns.")
                            current_sample["parsed_peaks"] = None
                        else:
                            min_m_data = float(valid_df["mass"].min())
                            max_m_data = float(valid_df["mass"].max())
                            min_m_bound = max(50.0, float(np.floor(min_m_data)))
                            max_m_bound = max(min_m_bound + 10.0, min(2500.0, float(np.ceil(max_m_data))))
                            def_low = max(min_m_bound, min(100.0, max_m_bound - 1.0))
                            def_high = min(max_m_bound, max(def_low + 1.0, min(max_m_bound, 1200.0)))
                            mz_range = st.slider(T[lang]["mz_range"], min_value=min_m_bound, max_value=max_m_bound, value=(def_low, def_high), step=10.0, key=f"mzrange_{active_spectrum_name}")

                            max_int_data = float(valid_df["intensity"].max())
                            cutoff_intensity = st.number_input(T[lang]["cutoff_int"], min_value=0.0, max_value=max(1.0, max_int_data), value=0.0, step=max(1e-4, max_int_data * 0.001), format="%.2e", key=f"cutoff_{active_spectrum_name}")

                            filtered_df = valid_df[(valid_df["mass"] >= mz_range[0]) & (valid_df["mass"] <= mz_range[1]) & (valid_df["intensity"] >= cutoff_intensity)].sort_values("mass").reset_index(drop=True)
                            if not filtered_df.empty:
                                if norm_choice == T[lang]["norm_base"]:
                                    filtered_df["norm_intensity"] = (filtered_df["intensity"] / filtered_df["intensity"].max()) * 100.0
                                elif norm_choice == T[lang]["norm_tic"]:
                                    filtered_df["norm_intensity"] = (filtered_df["intensity"] / filtered_df["intensity"].sum()) * 100.0
                                else:
                                    filtered_df["norm_intensity"] = filtered_df["intensity"]
                            else:
                                filtered_df["norm_intensity"] = []

                            if current_sample.get("is_recalibrated") and current_sample.get("recal_poly") is not None:
                                filtered_df["mass"] = filtered_df["mass"] - current_sample["recal_poly"](filtered_df["mass"])
                            elif current_sample.get("is_recalibrated") and current_sample.get("parsed_peaks") is not None and len(current_sample["parsed_peaks"]) == len(filtered_df):
                                filtered_df["mass"] = current_sample["parsed_peaks"]["mass"]

                            current_sample["parsed_peaks"] = filtered_df
                            st.success(T[lang]["loaded_peaks_success"].format(n=len(filtered_df)))
                except Exception as err:
                    st.warning(f"Ошибка при разборе файла ({err})." if lang == "ru" else f"File parse error ({err}).")
                    current_sample["parsed_peaks"] = None
        else:
            active_spectrum_name, current_sample = None, None
            st.info(T[lang]["no_spectra_info"])

    st.title(T[lang]["title"])

    if active_spectrum_name and current_sample:
        p_count = len(current_sample["parsed_peaks"]) if (current_sample.get("parsed_peaks") is not None and not current_sample["parsed_peaks"].empty) else 0
        as_df = current_sample.get("assigned_df")
        as_count = len(as_df) if (as_df is not None and not as_df.empty) else 0
        is_recal = current_sample.get("is_recalibrated", False)

        recal_chip = f'<span class="sample-chip green">✓ {"Рекалиброван" if lang == "ru" else "Recalibrated"}</span>' if is_recal else ""
        assign_chip = f'<span class="sample-chip green">🧬 {as_count:,} {"формул" if lang == "ru" else "formulas"} ({(as_count/p_count*100):.1f}%)</span>' if (as_count > 0 and p_count > 0) else f'<span class="sample-chip orange">⚠️ {"Формулы не приписаны" if lang == "ru" else "Formulas not assigned"}</span>'

        sample_tb_html = (
            f'<div class="sample-toolbar">'
            f'<div><span style="font-size: 13.5px; font-weight: 600; color: #1c2833;">📁 {active_spectrum_name}</span></div>'
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap;">'
            f'<span class="sample-chip">📊 {p_count:,} {"пиков" if lang == "ru" else "peaks"}</span>'
            f'{recal_chip}'
            f'{assign_chip}'
            f'</div>'
            f'</div>'
        )
        st.markdown(sample_tb_html, unsafe_allow_html=True)

    stage1, stage2, stage3 = st.tabs([
        T[lang]["tab_stage1"],
        T[lang]["tab_stage2"],
        T[lang]["tab_stage3"],
    ])

    with stage1:
        st_sub1, st_sub2 = st.tabs([T[lang]["tab_stick"], T[lang]["tab_recal"]])

    with st_sub1:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_stick']}: {active_spectrum_name}")
            c_ctrl1, c_ctrl2 = st.columns([3, 1])
            with c_ctrl1: st.caption(T[lang]["stick_signals"].format(n=len(peaks_df)))
            with c_ctrl2: annotate_top = st.checkbox(T[lang]["annotate_top"], value=True, key=f"ann_{active_spectrum_name}")

            m_vals, i_vals = peaks_df["mass"].values, peaks_df["norm_intensity"].values
            x_stick, y_stick = np.empty(len(m_vals) * 3), np.empty(len(m_vals) * 3)
            x_stick[0::3], x_stick[1::3], x_stick[2::3] = m_vals, m_vals, None
            y_stick[0::3], y_stick[1::3], y_stick[2::3] = 0, i_vals, None

            fig_stick = go.Figure()
            fig_stick.add_trace(go.Scattergl(x=x_stick, y=y_stick, mode="lines", line=dict(color="#0b5394", width=1.1), hoverinfo="skip", name=T[lang]["peak_trace"]))
            if annotate_top and len(peaks_df) > 0:
                top5 = peaks_df.nlargest(5, "norm_intensity")
                fig_stick.add_trace(go.Scattergl(
                    x=top5["mass"], y=top5["norm_intensity"], mode="markers+text",
                    text=[f"{val:.4f}" for val in top5["mass"]], textposition="top center",
                    marker=dict(color="#b45f06", size=7), textfont=dict(color="#b45f06", size=10),
                    name=T[lang]["top5_trace"], hovertemplate="<b>m/z</b>: %{x:.4f}<br><b>Int</b>: %{y:.1f}<extra></extra>",
                ))
            y_max_plot = float(peaks_df["norm_intensity"].max() * 1.15) if not peaks_df.empty else 100.0
            fig_stick.update_layout(xaxis=dict(title=T[lang]["mz_axis"], gridcolor="#f1f3f5"), yaxis=dict(title=T[lang]["rel_int_axis"], range=[0, y_max_plot], gridcolor="#f1f3f5"),
                                    plot_bgcolor="white", height=500, margin=dict(l=45, r=30, t=30, b=40))
            st_plotly(fig_stick)

    with st_sub2:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_recal']}: {active_spectrum_name}")
            cal_opts = ["Fatty Acids (C12–C33 saturated FA)" if lang == "en" else "Жирные кислоты (C12–C33 насыщенные ЖК)",
                        "CHO Homologues (CₙH₂ₙ₋₈O₇)" if lang == "en" else "Гомологи CHO (CₙH₂ₙ₋₈O₇)"]
            c_r1, c_r2, c_r3 = st.columns([2, 1.5, 1.5])
            with c_r1: calib_type = st.selectbox(T[lang]["recal_cal_set"], cal_opts, index=0, key=f"calibtype_{active_spectrum_name}")
            with c_r2: search_tol = st.number_input(T[lang]["recal_tol"], min_value=2.0, max_value=25.0, value=8.0, step=0.5, key=f"stol_{active_spectrum_name}")
            with c_r3: poly_order = st.selectbox(T[lang]["recal_poly"], [1, 2], index=1, key=f"polyord_{active_spectrum_name}")

            calib_ion_mode = st.selectbox(T[lang]["recal_ion"], ["ESI(-) [M - H]⁻", "ESI(+) [M + H]⁺", "Neutral masses [M]" if lang == "en" else "Нейтральные массы [M]"], index=0, key=f"calib_ion_{active_spectrum_name}")
            calib_lib = get_calibrant_library(calib_type, calib_ion_mode)
            exp_masses = peaks_df["mass"].values
            matched_calibs = []

            for _, row in calib_lib.iterrows():
                m_th = row["m_theor"]
                delta_lim = m_th * (search_tol * 1e-6)
                candidates = exp_masses[(exp_masses >= m_th - delta_lim) & (exp_masses <= m_th + delta_lim)]
                if len(candidates) > 0:
                    best_m = candidates[np.argmin(np.abs(candidates - m_th))]
                    matched_calibs.append({"Calibrant": row["name"], "m_theor": m_th, "m_exp": best_m, "delta_m": best_m - m_th, "error_ppm": ((best_m - m_th) / m_th) * 1e6})

            if len(matched_calibs) < 3:
                st.warning(T[lang]["recal_few_peaks"].format(n=len(matched_calibs)))
            else:
                calib_df = pd.DataFrame(matched_calibs)
                st.write(T[lang]["recal_found"].format(n=len(calib_df)))
                actual_poly_order = 1 if (poly_order > 1 and (float(calib_df["m_exp"].max() - calib_df["m_exp"].min()) / max(1.0, float(peaks_df["mass"].max() - peaks_df["mass"].min()))) < 0.60) else poly_order

                poly_fn = np.poly1d(np.polyfit(calib_df["m_exp"].values, calib_df["delta_m"].values, deg=actual_poly_order))
                residual_ppm = (((calib_df["m_exp"] - poly_fn(calib_df["m_exp"])) - calib_df["m_theor"]) / calib_df["m_theor"]) * 1e6

                fig_rec, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)
                ax1.scatter(calib_df["m_exp"], calib_df["error_ppm"], color="#D62728", s=30, label=T[lang]["recal_scatter_before"])
                m_grid = np.linspace(peaks_df["mass"].min(), peaks_df["mass"].max(), 200)
                ax1.plot(m_grid, (poly_fn(m_grid) / m_grid) * 1e6, color="black", linestyle="--", linewidth=1.2)
                ax1.set_xlabel("m/z"); ax1.set_ylabel(T[lang]["ppm_axis"]); ax1.set_title(T[lang]["recal_title_before"]); ax1.grid(True, alpha=0.3)
                ax2.scatter(calib_df["m_exp"], residual_ppm, color="#2CA02C", s=30, label=T[lang]["recal_scatter_after"])
                ax2.axhline(0, color="black", linewidth=0.8); ax2.set_xlabel("m/z"); ax2.set_ylabel(T[lang]["res_ppm_axis"]); ax2.set_title(T[lang]["recal_title_after"]); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_rec)
                plt.close(fig_rec)

                c_btn_r1, c_btn_r2 = st.columns([2, 1])
                with c_btn_r1:
                    if st.button(T[lang]["recal_btn"], type="primary", key=f"btn_recal_{active_spectrum_name}"):
                        current_sample["parsed_peaks"]["mass"] = peaks_df["mass"].values - poly_fn(peaks_df["mass"].values)
                        current_sample["recal_poly"] = poly_fn
                        current_sample["is_recalibrated"] = True
                        current_sample["assigned_df"] = None
                        st.success(T[lang]["recal_success"].format(n=len(peaks_df), before=calib_df["error_ppm"].abs().mean(), after=residual_ppm.abs().mean()))
                with c_btn_r2:
                    if current_sample.get("is_recalibrated"):
                        if st.button("Сбросить калибровку" if lang == "ru" else "Reset Calibration", key=f"btn_reset_recal_{active_spectrum_name}"):
                            current_sample["is_recalibrated"] = False
                            current_sample["recal_poly"] = None
                            current_sample["assigned_df"] = None
                            st.rerun()

                if current_sample.get("is_recalibrated"):
                    st.info("ℹ️ Для данного спектра активна полиномиальная рекалибровка (сохраняется при смене любых вкладок)." if lang == "ru" else "ℹ️ Polynomial recalibration is active for this spectrum (preserved across all tabs).")

    with stage2:
        st_sub_assign, st_sub_vk, st_sub_kmd, st_sub_vk20 = st.tabs([
            T[lang]["tab_assign"],
            T[lang]["tab_vk"],
            T[lang]["tab_kmd"],
            T[lang]["tab_vk20"],
        ])

    with st_sub_assign:
        if current_sample is None or current_sample["parsed_peaks"] is None or current_sample["parsed_peaks"].empty:
            st.info(T[lang]["no_spectra_info"])
        else:
            peaks_df = current_sample["parsed_peaks"]
            st.subheader(f"{T[lang]['tab_assign']}: {active_spectrum_name}")
            has_assigned_already = current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty
            with st.expander(T[lang]["assign_expander"], expanded=not has_assigned_already):
                col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
                with col_m1: ion_mode = st.selectbox(T[lang]["ion_mode"], ["ESI(-) [M - H]⁻", "ESI(+) [M + H]⁺", "Neutral masses [M]" if lang == "en" else "Нейтральные массы [M]"], index=0, key=f"ionmode_{active_spectrum_name}")
                with col_m2: max_charge_val = st.selectbox(T[lang]["max_charge"], [1, 2], index=0, key=f"zmax_{active_spectrum_name}")
                with col_m3: ppm_tol = st.number_input(T[lang]["ppm_tol"], min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"ppmtol_{active_spectrum_name}")

                col_el1, col_el2, col_el3 = st.columns(3)
                with col_el1:
                    c_bounds = st.slider(T[lang]["c_lim"], 4, 120, (4, 120), key=f"cb_{active_spectrum_name}")
                    h_bounds = st.slider(T[lang]["h_lim"], 4, 200, (4, 200), key=f"hb_{active_spectrum_name}")
                with col_el2:
                    o_bounds = st.slider(T[lang]["o_lim"], 1, 60, (1, 60), key=f"ob_{active_spectrum_name}")
                    n_bounds = st.slider(T[lang]["n_lim"], 0, 3, (0, 2), key=f"nb_{active_spectrum_name}")
                with col_el3:
                    s_bounds = st.slider(T[lang]["s_lim"], 0, 2, (0, 1), key=f"sb_{active_spectrum_name}")
                    max_oc_val = st.slider(T[lang]["max_oc"], 0.1, 1.0, 1.0, step=0.05, key=f"moc_{active_spectrum_name}")
                    max_hc_val = st.slider(T[lang]["max_hc"], 0.2, 2.2, 2.0, step=0.05, key=f"mhc_{active_spectrum_name}")

                col_iso1, col_iso2 = st.columns(2)
                with col_iso1: iso_chk = st.checkbox(T[lang]["iso_filter_label"], value=False, key=f"isochk_{active_spectrum_name}")
                with col_iso2: iso_strict_chk = st.checkbox(T[lang]["iso_strict_label"], value=False, disabled=not iso_chk, key=f"isostrict_{active_spectrum_name}")

            if st.button(T[lang]["assign_btn"], type="primary", key=f"btn_assign_{active_spectrum_name}"):
                with st.spinner(T[lang]["assign_spinner"]):
                    assigned_res = run_formula_assignment(
                        peaks_df=peaks_df, bounds={"C": c_bounds, "H": h_bounds, "O": o_bounds, "N": n_bounds, "S": s_bounds},
                        max_hc=max_hc_val, max_oc=max_oc_val, ppm_tolerance=ppm_tol,
                        ion_mode=ion_mode, max_charge=max_charge_val, iso_check=iso_chk, iso_strict=iso_strict_chk, lang=lang
                    )
                    current_sample["assigned_df"] = assigned_res

            assigned_data = current_sample["assigned_df"]
            if assigned_data is not None and not assigned_data.empty:
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(T[lang]["metric_total"], f"{len(peaks_df):,}")
                m2.metric(T[lang]["metric_assigned"], f"{len(assigned_data):,}")
                m3.metric(T[lang]["metric_rate"], f"{(len(assigned_data) / len(peaks_df) * 100.0):.1f}%")
                m4.metric(T[lang]["metric_err"], f"{assigned_data['error_ppm'].abs().mean():.3f}")
                st_df(assigned_data)

                col_dl_f1, col_dl_f2 = st.columns(2)
                with col_dl_f1:
                    st.download_button(label=T[lang]["dl_csv_formulas"], data=assigned_data.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{active_spectrum_name}_formulas.csv", mime="text/csv", key=f"dl_csv_formulas_{active_spectrum_name}")
                with col_dl_f2:
                    if OPEN_FORMATS_AVAILABLE and export_mztab_m is not None:
                        mztab_str = export_mztab_m(assigned_data, sample_id=active_spectrum_name)
                        st.download_button(label="📥 Экспорт mzTab-M (HUPO-PSI)", data=mztab_str.encode("utf-8"),
                                           file_name=f"{active_spectrum_name}.mzTab", mime="text/plain", key=f"dl_mztab_{active_spectrum_name}")

                if CHEMO_PUBCHEM_AVAILABLE:
                    st.markdown("---")
                    with st.expander(T[lang]["pubchem_inspect_expander"], expanded=False):
                        top_formulas = assigned_data.sort_values("intensity", ascending=False)["Formula"].head(60).tolist()
                        col_pc1, col_pc2 = st.columns([2, 2])
                        with col_pc1:
                            sel_f = st.selectbox(T[lang]["pubchem_select_formula"], top_formulas, key=f"pc_sel_{active_spectrum_name}")
                        with col_pc2:
                            cust_f = st.text_input(T[lang]["pubchem_custom_input"], value="", placeholder="e.g. C20H30O2", key=f"pc_cust_{active_spectrum_name}")

                        target_f = cust_f.strip() if cust_f.strip() else sel_f
                        search_clicked = st.button(T[lang]["pubchem_search_btn"], key=f"btn_pc_search_{active_spectrum_name}")

                        cached_target = st.session_state.get(f"pc_last_f_{active_spectrum_name}")
                        if search_clicked or (target_f and target_f != cached_target):
                            with st.spinner(T[lang]["pubchem_searching"]):
                                pc_matches = chemo_pubchem.lookup_formula_in_pubchem(target_f, max_records=4, timeout=4.0)
                                st.session_state[f"pc_res_list_{active_spectrum_name}"] = pc_matches if pc_matches else []
                                st.session_state[f"pc_last_f_{active_spectrum_name}"] = target_f

                        pc_matches_list = st.session_state.get(f"pc_res_list_{active_spectrum_name}", [])
                        if pc_matches_list:
                            st.caption(f"{T[lang]['pubchem_isomers_label']} **{len(pc_matches_list)}**")
                            clean_t_f = chemo_pubchem.clean_formula(target_f) if hasattr(chemo_pubchem, "clean_formula") else target_f
                            chembl_search_url = f"https://www.ebi.ac.uk/chembl/search_results/{urllib.parse.quote(clean_t_f)}"
                            hmdb_search_url = f"https://hmdb.ca/unearth/q?query={urllib.parse.quote(clean_t_f)}&searcher=metabolites"

                            ext_c1, ext_c2, _ = st.columns([1.5, 1.5, 3])
                            with ext_c1:
                                st.link_button(T[lang]["pubchem_chembl_btn"], chembl_search_url)
                            with ext_c2:
                                st.link_button(T[lang]["pubchem_hmdb_btn"], hmdb_search_url)

                            for idx, pc_item in enumerate(pc_matches_list, 1):
                                item_title = pc_item.get("Title") or pc_item.get("name", f"Isomer #{idx}")
                                with st.container(border=True):
                                    pc_col_img, pc_col_meta = st.columns([1, 2.2])
                                    with pc_col_img:
                                        img_url = pc_item.get("Image_URL") or pc_item.get("image_url")
                                        if img_url:
                                            st.image(img_url, caption=f"2D: {item_title}", width="stretch")
                                        else:
                                            st.info("2D preview not available")
                                    with pc_col_meta:
                                        c_name = item_title
                                        c_iupac = pc_item.get("IUPACName") or pc_item.get("iupac_name", "N/A")
                                        c_class = pc_item.get("Class") or pc_item.get("class", "Unknown")
                                        c_smiles = pc_item.get("SMILES") or pc_item.get("smiles", "N/A")
                                        c_cid = pc_item.get("CID") or pc_item.get("cid")
                                        st.markdown(f"**{T[lang]['pubchem_res_name']}** `{c_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_iupac']}** `{c_iupac}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_class']}** `{c_class}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_smiles']}** `{c_smiles}`")
                                        if c_cid:
                                            st.link_button(f"{T[lang]['pubchem_link_btn']} #{c_cid}", f"https://pubchem.ncbi.nlm.nih.gov/compound/{c_cid}")

                st.markdown("---")
                from views.agent_view import render_sample_ai_card
                render_sample_ai_card(active_spectrum_name, current_sample, lang=lang)

    with st_sub_vk:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is None or assigned_data.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            st.subheader(f"{T[lang]['tab_vk']}: {active_spectrum_name}")
            proj_type = st.radio(T[lang]["proj_mode"], [T[lang]["proj_vk"], T[lang]["proj_dbe_c"], T[lang]["proj_custom"]], horizontal=True, key=f"proj_type_{active_spectrum_name}")
            pow_exp = st.slider(T[lang]["pow_exp"], min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=f"pow_{active_spectrum_name}")
            palette = {"CHO": "#0020C2", "CHON": "#FF7F0E", "CHOS": "#2CA02C", "CHONS": "#D62728"}

            if proj_type == T[lang]["proj_vk"]:
                show_bioclasses = st.checkbox(T[lang]["vk_show_bioclasses"], value=True, key=f"vk_bio_{active_spectrum_name}")
                fig_vk = go.Figure()
                if show_bioclasses:
                    for reg_key, reg_info in VAN_KREVELEN_REGIONS.items():
                        oc_min, oc_max = reg_info["oc_range"]
                        hc_min, hc_max = reg_info["hc_range"]
                        fig_vk.add_shape(
                            type="rect",
                            x0=oc_min, x1=min(oc_max, 1.0),
                            y0=hc_min, y1=min(hc_max, 2.2),
                            fillcolor=reg_info["color"],
                            line=dict(color=reg_info["border"], width=1.5, dash="dot"),
                            layer="below",
                        )
                        fig_vk.add_annotation(
                            x=(oc_min + min(oc_max, 1.0)) / 2.0,
                            y=(hc_min + min(hc_max, 2.2)) / 2.0,
                            text=f"<b>{reg_info[lang]}</b>",
                            showarrow=False,
                            font=dict(size=10, color=reg_info["border"]),
                            opacity=0.75,
                        )

                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls]
                    if sub.empty: continue
                    fig_vk.add_trace(go.Scattergl(
                        x=sub["O/C"], y=sub["H/C"], mode="markers", name=f"{cls} ({len(sub):,})",
                        marker=dict(color=palette[cls], size=2.0 + 5.0 * (sub["norm_intensity"] / 100.0) ** pow_exp, opacity=0.65),
                        text=[f"<b>{r.get('Formula', '')}</b><br>m/z: {r['mass']:.4f}<br>AI: {r.get('AI', 0):.2f}" for _, r in sub.iterrows()], hoverinfo="text"
                    ))
                fig_vk.update_layout(xaxis=dict(title=T[lang]["oc_axis"], range=[0.0, 1.0], gridcolor="#f1f3f5"), yaxis=dict(title=T[lang]["hc_axis"], range=[0.2, 2.2], gridcolor="#f1f3f5"), plot_bgcolor="white", height=560)
                st_plotly(fig_vk)

                bio_dist = get_biomolecular_distribution(assigned_data, lang=lang)
                if bio_dist:
                    st.markdown(f"##### {T[lang]['vk_bioclass_dist_title']}")
                    b_col1, b_col2 = st.columns([1, 2])
                    with b_col1:
                        df_bio = pd.DataFrame([
                            {T[lang]["vk_bioclass_col"]: k, T[lang]["vk_bioclass_pct"]: v}
                            for k, v in bio_dist.items()
                        ])
                        st_df(df_bio)
                    with b_col2:
                        fig_bio = px.bar(
                            df_bio, x=T[lang]["vk_bioclass_col"], y=T[lang]["vk_bioclass_pct"],
                            color=T[lang]["vk_bioclass_col"], text=T[lang]["vk_bioclass_pct"],
                            template="plotly_white", height=280,
                        )
                        fig_bio.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_bio.update_layout(showlegend=False, yaxis_title="%", xaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
                        st_plotly(fig_bio)

                if CHEMO_PUBCHEM_AVAILABLE and "Formula" in assigned_data.columns:
                    st.markdown("---")
                    with st.expander(T[lang]["vk_inspect_title"], expanded=False):
                        st.caption(T[lang]["vk_inspect_caption"])
                        vk_formulas = assigned_data.sort_values("intensity", ascending=False)["Formula"].head(80).tolist()
                        v_col_sel, v_col_btn = st.columns([3, 1])
                        with v_col_sel:
                            sel_vk_f = st.selectbox(T[lang]["vk_inspect_select"], vk_formulas, key=f"vk_sel_f_{active_spectrum_name}")
                        with v_col_btn:
                            st.write("")
                            st.write("")
                            find_vk_clicked = st.button("🔍 " + T[lang]["pubchem_search_btn"].split()[-1], key=f"btn_vk_f_{active_spectrum_name}")

                        last_vk_f = st.session_state.get(f"vk_last_f_{active_spectrum_name}")
                        if find_vk_clicked or (sel_vk_f and sel_vk_f != last_vk_f):
                            with st.spinner(T[lang]["pubchem_searching"]):
                                vk_matches = chemo_pubchem.lookup_formula_in_pubchem(sel_vk_f, max_records=2, timeout=4.0)
                                st.session_state[f"vk_res_{active_spectrum_name}"] = vk_matches
                                st.session_state[f"vk_last_f_{active_spectrum_name}"] = sel_vk_f

                        vk_saved_res = st.session_state.get(f"vk_res_{active_spectrum_name}", [])
                        if vk_saved_res:
                            clean_vk_str = chemo_pubchem.clean_formula(sel_vk_f) if hasattr(chemo_pubchem, "clean_formula") else sel_vk_f
                            vk_chembl_url = f"https://www.ebi.ac.uk/chembl/search_results/{urllib.parse.quote(clean_vk_str)}"
                            vk_hmdb_url = f"https://hmdb.ca/unearth/q?query={urllib.parse.quote(clean_vk_str)}&searcher=metabolites"

                            ext_vk1, ext_vk2, _ = st.columns([1.5, 1.5, 3])
                            with ext_vk1:
                                st.link_button(T[lang]["pubchem_chembl_btn"], vk_chembl_url)
                            with ext_vk2:
                                st.link_button(T[lang]["pubchem_hmdb_btn"], vk_hmdb_url)

                            for idx, item in enumerate(vk_saved_res, 1):
                                with st.container(border=True):
                                    c_img, c_info = st.columns([1, 2.2])
                                    with c_img:
                                        img_u = item.get("Image_URL")
                                        if img_u:
                                            st.image(img_u, caption=f"2D: {item.get('Title', 'Compound')}", width="stretch")
                                        else:
                                            st.info("2D structure preview not available" if lang == "en" else "2D структура недоступна")
                                    with c_info:
                                        t_name = item.get("Title", "N/A")
                                        iupac_name = item.get("IUPACName", "N/A")
                                        cls_name = item.get("Class", "Unknown")
                                        smi = item.get("SMILES", "N/A")
                                        cid_val = item.get("CID")
                                        st.markdown(f"**{T[lang]['pubchem_res_name']}** `{t_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_iupac']}** {iupac_name}")
                                        st.markdown(f"**{T[lang]['pubchem_res_class']}** `{cls_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_smiles']}** `{smi}`")
                                        if cid_val:
                                            st.link_button(f"{T[lang]['pubchem_link_btn']} #{cid_val}", f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid_val}")
            elif proj_type == T[lang]["proj_dbe_c"]:
                fig_dbe = go.Figure()
                for cls in ["CHO", "CHON", "CHOS", "CHONS"]:
                    sub = assigned_data[assigned_data["Hetero_Class"] == cls] if "Hetero_Class" in assigned_data.columns else assigned_data
                    if sub.empty or "C" not in sub.columns or "DBE" not in sub.columns: continue
                    color = palette.get(cls, "#0020C2")
                    norm_vals = sub["norm_intensity"] if "norm_intensity" in sub.columns else pd.Series(50.0, index=sub.index)
                    fig_dbe.add_trace(go.Scattergl(
                        x=sub["C"], y=sub["DBE"], mode="markers", name=f"{cls} ({len(sub):,})",
                        marker=dict(color=color, size=2.0 + 5.0 * (norm_vals / 100.0) ** pow_exp, opacity=0.65),
                    ))
                max_c = float(assigned_data["C"].max()) if "C" in assigned_data.columns and not assigned_data["C"].isna().all() else 60.0
                c_line = np.linspace(4, max(60.0, max_c), 100)
                fig_dbe.add_trace(go.Scatter(x=c_line, y=c_line * 0.9, mode="lines", name="Planar limit", line=dict(color="red", dash="dot")))

                fig_dbe.update_layout(xaxis=dict(title="Carbon (C)", gridcolor="#f1f3f5"), yaxis=dict(title="DBE", gridcolor="#f1f3f5"), plot_bgcolor="white", height=560)
                st_plotly(fig_dbe)
            else:
                avail_cols = [c for c in ["mass", "intensity", "norm_intensity", "C", "H", "O", "N", "S", "H/C", "O/C", "DBE", "AI"] if c in assigned_data.columns]
                c1, c2, c3 = st.columns(3)
                with c1: cx = st.selectbox("X:", avail_cols, index=0, key=f"cust_x_{active_spectrum_name}")
                with c2: cy = st.selectbox("Y:", avail_cols, index=min(1, len(avail_cols)-1), key=f"cust_y_{active_spectrum_name}")
                with c3: cc = st.selectbox("Color:", avail_cols, index=min(2, len(avail_cols)-1), key=f"cust_c_{active_spectrum_name}")
                fig_cust = px.scatter(assigned_data, x=cx, y=cy, color=cc, color_continuous_scale="Viridis", opacity=0.7)
                fig_cust.update_layout(height=560, plot_bgcolor="white")
                st_plotly(fig_cust)

    with st_sub_kmd:
        st.subheader(f"{T[lang]['tab_kmd']}: {active_spectrum_name if active_spectrum_name else ''}")
        work_df = current_sample.get("assigned_df") if (current_sample and current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty) else (current_sample.get("parsed_peaks") if current_sample else None)
        if work_df is not None and not work_df.empty:
            work_df = work_df.copy()
            col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
            with col_b1:
                base_choice = st.selectbox(T[lang]["kmd_base_group"], list(KMD_BASES.keys()), index=0, key=f"kmd_base_{active_spectrum_name}")
            with col_b2:
                color_mode = st.selectbox(T[lang]["kmd_color_mode"], [T[lang]["kmd_parity_mode"], T[lang]["kmd_int_mode"]], index=0, key=f"kmd_color_{active_spectrum_name}")
            with col_b3:
                point_sz = st.slider(T[lang]["kmd_pt_size"], 1, 8, 3, key=f"kmd_pt_{active_spectrum_name}")

            km, kmd, nkm = compute_kmd(work_df["mass"].values, base_choice)
            work_df["KM"], work_df["KMD"], work_df["NKM"] = km, kmd, nkm
            base_label = KMD_BASES[base_choice]["label"]

            fig_kmd = px.scatter(
                work_df, x="NKM", y="KMD", color=(work_df["NKM"] % 2 == 0).astype(str) if color_mode == T[lang]["kmd_parity_mode"] else "norm_intensity",
                opacity=0.7, height=540, labels={"NKM": T[lang]["kmd_x_axis"].format(base=base_label), "KMD": T[lang]["kmd_y_axis"].format(base=base_label)}
            )
            fig_kmd.update_layout(plot_bgcolor="white")
            st_plotly(fig_kmd)

    with st_sub_vk20:
        st.subheader(f"{T[lang]['tab_vk20']}: {active_spectrum_name if active_spectrum_name else ''}")
        assigned_df = current_sample.get("assigned_df") if current_sample else None
        if assigned_df is None or assigned_df.empty:
            st.warning(T[lang]["need_assign_first"])
        else:
            pct_count, pct_weight, feat_df = compute_vk20_grid(assigned_df)
            metric_mode = st.radio(T[lang]["vk20_basis"], [T[lang]["vk20_opt_count"], T[lang]["vk20_opt_weight"]], horizontal=True, key=f"vk20_mode_{active_spectrum_name}")
            active_matrix = pct_count if metric_mode == T[lang]["vk20_opt_count"] else pct_weight

            fig_hm, ax_hm = plt.subplots(figsize=(9, 5.5), dpi=120)
            im = ax_hm.imshow(active_matrix, origin="lower", cmap="YlOrRd", aspect="auto")
            ax_hm.set_xticks(range(4)); ax_hm.set_xticklabels(["C1 [0-0.25)", "C2 [0.25-0.5)", "C3 [0.5-0.75)", "C4 [0.75-1.0]"], fontsize=8)
            ax_hm.set_yticks(range(5)); ax_hm.set_yticklabels(["R1 [0.2-0.6)", "R2 [0.6-1.0)", "R3 [1.0-1.4)", "R4 [1.4-1.8)", "R5 [1.8-2.2]"], fontsize=8)
            ax_hm.set_title(T[lang]["vk20_heatmap_title"].format(mode=metric_mode), fontsize=10)
            for r in range(5):
                for c in range(4):
                    val = active_matrix[r, c]
                    ax_hm.text(c, r, f"VK_{1 + r * 4 + c}\n{val:.1f}%", ha="center", va="center", color="white" if val > (active_matrix.max() * 0.65) else "black", fontsize=8, fontweight="bold")
            plt.colorbar(im, ax=ax_hm, pad=0.02)
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)
            st_df(feat_df)
            st.download_button(label=T[lang]["vk20_dl_csv"], data=feat_df.to_csv(index=False).encode("utf-8"), file_name=f"{active_spectrum_name}_vk20.csv", mime="text/csv", key=f"dl_vk20_{active_spectrum_name}")

    with stage3:
        st_sub_cmp, st_sub_tmds, st_sub_desc = st.tabs([
            T[lang]["tab_cmp"],
            T[lang]["tab_tmds"],
            T[lang]["tab_desc"],
        ])

    with st_sub_cmp:
        st.subheader(T[lang]["tab_cmp"])

        with st.popover("ℹ️ " + ("Руководство: Зачем нужна Алгебра спектров и как её применять" if lang == "ru" else "Guide: Spectrum Algebra & Comparative Analysis"), width="stretch"):
            st.markdown(r"""
### 🔬 Зачем нужна Алгебра спектров в экологическом анализе?

В масс-спектрометрии ультравысокого разрешения (FT-ICR MS) природных вод (оз. Байкал, притоки) и техногенных загрязнений (шлам-лигнин БЦБК) спектры содержат тысячи индивидуальных пиков. Алгебра спектров позволяет строго математически разделить природный фон и антропогенные маркеры:

---

#### 1. Вычитание бланка / разность ($A \setminus B$)
* **Физический смысл:** Удаление фонового матричного шума (чистая байкальская вода или бланк растворителя) из загрязненной пробы.
* **Результат:** Остаются **только уникальные компоненты**, характерные для источника воздействия (сульфатный шлам-лигнин, техногенные серосодержащие соединения).

#### 2. Консервативное ядро / пересечение ($A \cap B$)
* **Физический смысл:** Поиск соединений, присутствующих **одновременно** в обоих образцах в пределах заданного допуска $ppm$.
* **Результат:** Выделение устойчивого природного гуминового скелета (NOM), инвариантного к внешним факторам.

#### 3. Симметрическая разность ($A \oplus B$) и объединение ($A \cup B$)
* **$A \oplus B$:** Компоненты, уникальные либо для $A$, либо для $B$ (химическая дивергенция составов).
* **$A \cup B$:** Полная композитная карта молекулярного разнообразия экосистемы.

---

#### 📐 Параметры и метрики:
* **Допуск сопоставления ($\text{ppm}$):** Окно поиска по $m/z$ ($\Delta m = m \cdot \text{ppm} \cdot 10^{-6}$). Для FT-ICR MS стандартно $1.0 - 2.0\text{ ppm}$.
* **Индекс Жаккара ($J$):** Степень совпадения компонентного состава ($J = \frac{|A \cap B|}{|A \cup B|}$, от 0 до 1).
* **Косинусное сходство ($\cos \theta$):** Сходство профилей относительных интенсивностей общих пиков.
""" if lang == "ru" else r"""
### 🔬 Why Spectrum Algebra Matters in Environmental Analysis?

In high-resolution mass spectrometry of natural waters (Lake Baikal) and industrial impacts (kraft pulp mill sludge lignin), spectra contain thousands of peaks. Spectrum algebra provides rigorous set-theoretic operations to isolate background NOM from anthropogenic tracers:

---

#### 1. Blank Subtraction & Difference ($A \setminus B$)
* **Physical Meaning:** Removing baseline solvent/background water signals from an impacted sample.
* **Result:** Yields **only unique tracers** specific to lignin contamination.

#### 2. Conservative Core / Intersection ($A \cap B$)
* **Physical Meaning:** Identifies compounds shared by both samples within the specified $ppm$ tolerance.
* **Result:** Isolates resilient natural organic matter (NOM) background.

#### 3. Symmetric Difference ($A \oplus B$) & Union ($A \cup B$)
* **$A \oplus B$:** Components unique to either sample (chemical divergence).
* **$A \cup B$:** Composite molecular map of the aquatic system.

---

#### 📐 Metrics & Parameters:
* **Matching Tolerance ($\text{ppm}$):** $m/z$ alignment window (typically $1.0 - 2.0\text{ ppm}$).
* **Jaccard Similarity ($J$):** Compositional overlap ($J = \frac{|A \cap B|}{|A \cup B|}$).
* **Cosine Similarity ($\cos \theta$):** Spectral profile intensity vector similarity.
""")

        all_spectra_list = list(st.session_state.get("spectra_db", {}).keys())
        if len(all_spectra_list) < 2:
            st.info(T[lang]["cmp_need_two"])
        else:
            cmp_tab_choice = st.radio(
                "Раздел / Mode:",
                [T[lang]["cmp_subtab_view"], T[lang]["cmp_subtab_sub"], T[lang]["cmp_subtab_algebra"]],
                horizontal=True,
                key="cmp_internal_mode",
            )

            col_s1, col_s2, col_s3 = st.columns([2, 2, 1.5])
            with col_s1:
                name_a = st.selectbox(T[lang]["cmp_spec_a"], all_spectra_list, index=0, key="cmp_spec_a")
            with col_s2:
                default_b_idx = 1 if len(all_spectra_list) > 1 else 0
                name_b = st.selectbox(T[lang]["cmp_spec_b"], all_spectra_list, index=default_b_idx, key="cmp_spec_b")
            with col_s3:
                tol_comp = st.number_input(T[lang]["cmp_tol"], min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="cmp_ppm_tol")

            sample_a = st.session_state["spectra_db"][name_a]
            sample_b = st.session_state["spectra_db"][name_b]

            peaks_a = ensure_sample_peaks(sample_a)
            peaks_b = ensure_sample_peaks(sample_b)

            if peaks_a is None or peaks_a.empty or peaks_b is None or peaks_b.empty:
                st.warning("One of the selected samples is empty or could not be parsed." if lang == "en" else "Один из выбранных образцов пуст или не содержит числовых данных пиков.")
            elif name_a == name_b:
                st.warning("Same sample selected for comparison!" if lang == "en" else "Выбран один и тот же образец для сравнения!")
            else:
                m_a, m_b = align_two_spectra_fast(peaks_a, peaks_b, ppm_tol=tol_comp)
                n_a = len(peaks_a)
                n_b = len(peaks_b)
                n_common = len(m_a)
                jaccard = n_common / (n_a + n_b - n_common) if (n_a + n_b - n_common) > 0 else 0.0

                if n_common > 0:
                    ia_sub = peaks_a.iloc[m_a]["intensity"].values
                    ib_sub = peaks_b.iloc[m_b]["intensity"].values
                    dot_prod = float(np.dot(ia_sub, ib_sub))
                    norm_a = float(np.linalg.norm(peaks_a["intensity"].values))
                    norm_b = float(np.linalg.norm(peaks_b["intensity"].values))
                    cos_sim = dot_prod / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
                else:
                    cos_sim = 0.0

                # 1. Обычное сравнение (A vs B)
                if cmp_tab_choice == T[lang]["cmp_subtab_view"]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Peaks in {name_a}" if lang == "en" else f"Пиков в {name_a}", f"{n_a:,}")
                    m2.metric(f"Peaks in {name_b}" if lang == "en" else f"Пиков в {name_b}", f"{n_b:,}")
                    m3.metric(T[lang]["cmp_common"], f"{n_common:,}", delta=T[lang]["cmp_jaccard"].format(val=jaccard))
                    m4.metric(T[lang]["cmp_cosine"], f"{cos_sim:.4f}")

                    assigned_a = sample_a.get("assigned_df")
                    assigned_b = sample_b.get("assigned_df")

                    if assigned_a is not None and not assigned_a.empty:
                        st.markdown("##### " + (f"Comparative Van Krevelen: {name_a} vs {name_b}" if lang == "en" else f"Сравнительная диаграмма Ван-Кревелена: {name_a} vs {name_b}"))
                        common_masses = set(peaks_a.iloc[m_a]["mass"].round(4))

                        assigned_a_common = assigned_a[assigned_a["mass"].round(4).isin(common_masses)]
                        assigned_a_unique = assigned_a[~assigned_a["mass"].round(4).isin(common_masses)]

                        fig_cmp_inter = go.Figure()
                        if not assigned_a_common.empty:
                            fig_cmp_inter.add_trace(go.Scattergl(
                                x=assigned_a_common["O/C"],
                                y=assigned_a_common["H/C"],
                                mode="markers",
                                name=T[lang]["cmp_common_label"].format(n=len(assigned_a_common)),
                                marker=dict(color="#7F7F7F", size=3.0, opacity=0.45),
                            ))

                        if not assigned_a_unique.empty:
                            fig_cmp_inter.add_trace(go.Scattergl(
                                x=assigned_a_unique["O/C"],
                                y=assigned_a_unique["H/C"],
                                mode="markers",
                                name=T[lang]["cmp_unique_a"].format(name=name_a, n=len(assigned_a_unique)),
                                marker=dict(color="#0020C2", size=3.5, opacity=0.75),
                            ))

                        if assigned_b is not None and not assigned_b.empty:
                            assigned_b_unique = assigned_b[~assigned_b["mass"].round(4).isin(common_masses)]
                            if not assigned_b_unique.empty:
                                fig_cmp_inter.add_trace(go.Scattergl(
                                    x=assigned_b_unique["O/C"],
                                    y=assigned_b_unique["H/C"],
                                    mode="markers",
                                    name=T[lang]["cmp_unique_b"].format(name=name_b, n=len(assigned_b_unique)),
                                    marker=dict(color="#2CA02C", size=3.5, opacity=0.75),
                                ))

                        fig_cmp_inter.update_layout(
                            xaxis=dict(title=T[lang]["oc_axis"], range=[0.0, 1.0], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                            yaxis=dict(title=T[lang]["hc_axis"], range=[0.2, 2.2], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                            plot_bgcolor="white",
                            height=580,
                            margin=dict(l=45, r=30, t=30, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.caption(T[lang]["nav_caption"])
                        st_plotly(fig_cmp_inter)

                        fig_cmp_mpl, ax_cmp_mpl = plt.subplots(figsize=(12, 7.5), dpi=150)
                        if not assigned_a_common.empty:
                            ax_cmp_mpl.scatter(assigned_a_common["O/C"], assigned_a_common["H/C"], c="#7F7F7F",
                                               label=T[lang]["cmp_common_label"].format(n=len(assigned_a_common)), s=2.0, alpha=0.4, edgecolors="none", rasterized=True)
                        if not assigned_a_unique.empty:
                            ax_cmp_mpl.scatter(assigned_a_unique["O/C"], assigned_a_unique["H/C"], c="#0020C2",
                                               label=T[lang]["cmp_unique_a"].format(name=name_a, n=len(assigned_a_unique)), s=2.2, alpha=0.75, edgecolors="none", rasterized=True)
                        if assigned_b is not None and not assigned_b.empty:
                            assigned_b_unique = assigned_b[~assigned_b["mass"].round(4).isin(common_masses)]
                            if not assigned_b_unique.empty:
                                ax_cmp_mpl.scatter(assigned_b_unique["O/C"], assigned_b_unique["H/C"], c="#2CA02C",
                                                   label=T[lang]["cmp_unique_b"].format(name=name_b, n=len(assigned_b_unique)), s=2.2, alpha=0.75, edgecolors="none", rasterized=True)

                        ax_cmp_mpl.set_xlim(0.0, 1.0)
                        ax_cmp_mpl.set_ylim(0.2, 2.2)
                        ax_cmp_mpl.set_xlabel(T[lang]["oc_axis"], fontsize=11, fontweight="medium")
                        ax_cmp_mpl.set_ylabel(T[lang]["hc_axis"], fontsize=11, fontweight="medium")
                        ax_cmp_mpl.set_title(f"Comparative Analysis: {name_a} vs {name_b}", fontsize=11)
                        ax_cmp_mpl.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=3)
                        ax_cmp_mpl.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
                        plt.tight_layout()

                        get_plot_download_buttons(fig_cmp_mpl, f"compare_{name_a}_vs_{name_b}_vk", lang)
                        plt.close(fig_cmp_mpl)

                    st.markdown("##### " + T[lang]["cmp_mirror_title"])
                    ia_norm = (peaks_a["intensity"] / max(1e-12, float(peaks_a["intensity"].max()))) * 100.0
                    ib_norm = (peaks_b["intensity"] / max(1e-12, float(peaks_b["intensity"].max()))) * 100.0

                    ma = peaks_a["mass"].values
                    mb = peaks_b["mass"].values

                    x_a = np.empty(len(ma) * 3)
                    y_a = np.empty(len(ma) * 3)
                    x_a[0::3] = ma
                    x_a[1::3] = ma
                    x_a[2::3] = None
                    y_a[0::3] = 0
                    y_a[1::3] = ia_norm
                    y_a[2::3] = None

                    x_b = np.empty(len(mb) * 3)
                    y_b = np.empty(len(mb) * 3)
                    x_b[0::3] = mb
                    x_b[1::3] = mb
                    x_b[2::3] = None
                    y_b[0::3] = 0
                    y_b[1::3] = -ib_norm
                    y_b[2::3] = None

                    fig_mir_inter = go.Figure()
                    fig_mir_inter.add_trace(go.Scattergl(x=x_a, y=y_a, mode="lines", line=dict(color="#0020C2", width=1.1), name=f"{name_a} (+)"))
                    fig_mir_inter.add_trace(go.Scattergl(x=x_b, y=y_b, mode="lines", line=dict(color="#2CA02C", width=1.1), name=f"{name_b} (-)"))
                    fig_mir_inter.update_layout(
                        xaxis=dict(title=T[lang]["mz_axis"], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                        yaxis=dict(title=T[lang]["rel_int_axis"], gridcolor="#f1f3f5", zerolinecolor="#ced4da"),
                        plot_bgcolor="white",
                        height=480,
                        margin=dict(l=45, r=30, t=30, b=40),
                    )
                    st.caption(T[lang]["nav_caption"])
                    st_plotly(fig_mir_inter)

                    fig_mir_mpl, ax_mir_mpl = plt.subplots(figsize=(12, 5), dpi=150)
                    ax_mir_mpl.vlines(peaks_a["mass"], 0, ia_norm, color="#0020C2", linewidth=0.6, alpha=0.7, label=f"{name_a} (+)")
                    ax_mir_mpl.vlines(peaks_b["mass"], 0, -ib_norm, color="#2CA02C", linewidth=0.6, alpha=0.7, label=f"{name_b} (-)")
                    ax_mir_mpl.axhline(0, color="black", linewidth=0.8)
                    ax_mir_mpl.set_xlabel(T[lang]["mz_axis"], fontsize=11)
                    ax_mir_mpl.set_ylabel(T[lang]["rel_int_axis"], fontsize=10)
                    ax_mir_mpl.legend(loc="upper right")
                    plt.tight_layout()

                    get_plot_download_buttons(fig_mir_mpl, f"head_to_tail_{name_a}_vs_{name_b}", lang)
                    plt.close(fig_mir_mpl)

                # 2. Вычитание бланка (int_sub)
                elif cmp_tab_choice == T[lang]["cmp_subtab_sub"]:
                    st.markdown(f"#### {T[lang]['cmp_subtab_sub']}")
                    st.write(f"Образец: **{name_a}** | Холостая проба (Бланк): **{name_b}**" if lang == "ru" else f"Sample: **{name_a}** | Blank: **{name_b}**")
                    sub_factor = st.slider(T[lang]["sub_factor_label"], min_value=0.1, max_value=3.0, value=1.0, step=0.05)

                    if st.button(T[lang]["sub_btn"], type="primary"):
                        sub_peaks = peaks_a.copy().reset_index(drop=True)
                        matched_dict = dict(zip(m_a, m_b))

                        new_ints = []
                        for idx_a_row in range(len(sub_peaks)):
                            orig_int = sub_peaks.loc[idx_a_row, "intensity"]
                            if idx_a_row in matched_dict:
                                idx_b_row = matched_dict[idx_a_row]
                                blank_int = peaks_b.iloc[idx_b_row]["intensity"]
                                res_int = max(0.0, float(orig_int - sub_factor * blank_int))
                            else:
                                res_int = float(orig_int)
                            new_ints.append(res_int)

                        sub_peaks["intensity"] = new_ints
                        clean_sub_df = sub_peaks[sub_peaks["intensity"] > 0].copy().reset_index(drop=True)
                        if clean_sub_df.empty:
                            st.warning("В результате вычитания все пики были обнулены." if lang == "ru" else "All peaks were subtracted to zero.")
                        else:
                            clean_sub_df["norm_intensity"] = (clean_sub_df["intensity"] / clean_sub_df["intensity"].max()) * 100.0
                            new_sample_name = f"{name_a}_sub_{name_b}.csv"
                            fpath_sub = os.path.join(STORAGE_DIR, new_sample_name)
                            clean_sub_df.to_csv(fpath_sub, sep="\t", index=False)

                            st.session_state["spectra_db"][new_sample_name] = {
                                "raw_path": fpath_sub,
                                "file_bytes": clean_sub_df.to_csv(sep="\t", index=False).encode("utf-8"),
                                "parsed_peaks": clean_sub_df,
                                "assigned_df": None,
                                "raw_df": clean_sub_df,
                            }
                            st.success(T[lang]["sub_success"].format(name=new_sample_name, n=len(clean_sub_df)))
                            st.rerun()

                # 3. Алгебра спектров и диаграмма Венна
                else:
                    st.markdown(f"#### {T[lang]['cmp_subtab_algebra']}")
                    label_sub_a_b = (
                        T[lang]["alg_sub_a_b"].format(a=name_a, b=name_b)
                        if "{a}" in T[lang]["alg_sub_a_b"]
                        else f"{T[lang]['alg_sub_a_b']} ({name_a} \\ {name_b})"
                    )
                    label_sub_b_a = (
                        T[lang]["alg_sub_b_a"].format(a=name_a, b=name_b)
                        if "{a}" in T[lang]["alg_sub_b_a"]
                        else f"{T[lang]['alg_sub_b_a']} ({name_b} \\ {name_a})"
                    )
                    op_choice = st.selectbox(
                        T[lang]["alg_op_label"],
                        [
                            ("and", T[lang]["alg_and"]),
                            ("or", T[lang]["alg_or"]),
                            ("sub_a_b", label_sub_a_b),
                            ("sub_b_a", label_sub_b_a),
                            ("xor", T[lang]["alg_xor"]),
                        ],
                        format_func=lambda x: x[1],
                    )

                    # Построение диаграммы Венна (Matplotlib Circles)
                    fig_v, ax_v = plt.subplots(figsize=(8, 4.5), dpi=150)
                    c_a = patches.Circle((0.35, 0.5), 0.3, facecolor="#0020C2", alpha=0.4, edgecolor="black", linewidth=1.5)
                    c_b = patches.Circle((0.65, 0.5), 0.3, facecolor="#2CA02C", alpha=0.4, edgecolor="black", linewidth=1.5)
                    ax_v.add_patch(c_a)
                    ax_v.add_patch(c_b)

                    n_uniq_a = n_a - n_common
                    n_uniq_b = n_b - n_common

                    ax_v.text(0.22, 0.5, f"{name_a[:15]}\n\n{n_uniq_a:,}", ha="center", va="center", fontsize=11, fontweight="bold")
                    ax_v.text(0.78, 0.5, f"{name_b[:15]}\n\n{n_uniq_b:,}", ha="center", va="center", fontsize=11, fontweight="bold")
                    ax_v.text(0.50, 0.5, f"A ∩ B\n\n{n_common:,}", ha="center", va="center", fontsize=11, fontweight="bold", color="#800000")

                    ax_v.set_xlim(0, 1)
                    ax_v.set_ylim(0.1, 0.9)
                    ax_v.axis("off")
                    ax_v.set_title(T[lang]["venn_title"].format(tol=tol_comp), fontsize=12, pad=10)
                    plt.tight_layout()
                    st.pyplot(fig_v)
                    get_plot_download_buttons(fig_v, f"venn_{name_a}_vs_{name_b}", lang)
                    plt.close(fig_v)

                    if st.button(T[lang]["alg_btn"], type="primary"):
                        alg_df = perform_spectral_algebra(peaks_a, peaks_b, operation=op_choice[0], ppm_tol=tol_comp)
                        if alg_df.empty:
                            st.warning("В результате операции пики не найдены." if lang == "ru" else "No peaks resulted from this operation.")
                        else:
                            alg_df["norm_intensity"] = (alg_df["intensity"] / alg_df["intensity"].max()) * 100.0
                            new_alg_name = f"{name_a}_{op_choice[0]}_{name_b}.csv"
                            fpath_alg = os.path.join(STORAGE_DIR, new_alg_name)
                            alg_df.to_csv(fpath_alg, sep="\t", index=False)

                            st.session_state["spectra_db"][new_alg_name] = {
                                "raw_path": fpath_alg,
                                "file_bytes": alg_df.to_csv(sep="\t", index=False).encode("utf-8"),
                                "parsed_peaks": alg_df,
                                "assigned_df": None,
                                "raw_df": alg_df,
                            }
                            st.success(T[lang]["alg_success"].format(name=new_alg_name, n=len(alg_df)))
                            st.rerun()

    # 8. TMDS

    with st_sub_tmds:
        st.subheader(f"{T[lang]['tab_tmds']}: {active_spectrum_name if active_spectrum_name else ''}")
        work_peaks = current_sample.get("assigned_df") if (current_sample and current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty) else (current_sample.get("parsed_peaks") if current_sample else None)
        if work_peaks is not None and not work_peaks.empty:
            tmds_view_mode = st.radio(
                "Режим отображения TMDS:" if lang == "ru" else "TMDS Display Mode:",
                [T[lang]["tmds_mode_freq"], T[lang]["tmds_mode_network"], T[lang]["tmds_mode_pathways"]],
                horizontal=True,
                key=f"tmds_view_{active_spectrum_name}"
            )
            if tmds_view_mode == T[lang]["tmds_mode_freq"]:
                flux_res = compute_geochemical_vector_fluxes(work_peaks, tol_mda=2.0, top_n=1000)
                tmds_sum = flux_res.get("summary_df", pd.DataFrame())
                indices = flux_res.get("indices", {})
                if not tmds_sum.empty:
                    st.markdown(f"#### {T[lang]['tmds_flux_header']}")
                    fx1, fx2, fx3, fx4 = st.columns(4)
                    fx1.metric(T[lang]["tmds_flux_ox_decarb"], f"{indices.get('ox_decarb_ratio', 0.0):.2f}")
                    fx2.metric(T[lang]["tmds_flux_alkyl"], f"{indices.get('alkylation_share_pct', 0.0):.1f}%")
                    fx3.metric(T[lang]["tmds_flux_sulf"], f"{indices.get('sulfonation_index_pct', 0.0):.2f}%")
                    fx4.metric(T[lang]["tmds_flux_hydr"], f"{indices.get('hydration_ox_ratio', 0.0):.2f}")

                    fig_tmds = px.bar(tmds_sum, x="Transformation", y="Count", color="Flux_Type", text=tmds_sum["Share_pct"].apply(lambda v: f"{v:.1f}%"))
                    fig_tmds.update_layout(showlegend=True, xaxis_tickangle=-25, height=420)
                    st_plotly(fig_tmds)
                    st_df(tmds_sum)
            elif tmds_view_mode == T[lang]["tmds_mode_pathways"]:
                top_pks = work_peaks.sort_values("intensity", ascending=False).head(40).sort_values("mass")
                cand_options = [
                    (float(r["mass"]), f"m/z {r['mass']:.4f}" + (f" ({r['Formula']})" if 'Formula' in r and pd.notna(r['Formula']) else ""))
                    for _, r in top_pks.iterrows()
                ]
                col_pw1, col_pw2 = st.columns(2)
                with col_pw1:
                    src_sel = st.selectbox(
                        T[lang]["tmds_path_source"],
                        options=cand_options,
                        format_func=lambda x: x[1],
                        index=0,
                        key=f"pw_src_{active_spectrum_name}"
                    )
                    src_m_custom = st.number_input("Или точная m/z:" if lang == "ru" else "Or exact m/z:", value=0.0, step=0.0001, format="%.4f", key=f"cust_src_{active_spectrum_name}")
                    src_m = src_m_custom if src_m_custom > 10.0 else src_sel[0]
                with col_pw2:
                    dst_default_idx = min(len(cand_options)-1, 5)
                    dst_sel = st.selectbox(
                        T[lang]["tmds_path_target"],
                        options=cand_options,
                        format_func=lambda x: x[1],
                        index=dst_default_idx,
                        key=f"pw_dst_{active_spectrum_name}"
                    )
                    dst_m_custom = st.number_input("Или точная m/z продукта:" if lang == "ru" else "Or exact target m/z:", value=0.0, step=0.0001, format="%.4f", key=f"cust_dst_{active_spectrum_name}")
                    dst_m = dst_m_custom if dst_m_custom > 10.0 else dst_sel[0]

                col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
                with col_cfg1:
                    depth_val = st.slider(T[lang]["tmds_path_depth"], min_value=1, max_value=5, value=3, key=f"pw_depth_{active_spectrum_name}")
                with col_cfg2:
                    tol_pw_val = st.number_input(T[lang]["tmds_path_tol"], min_value=0.5, max_value=5.0, value=2.0, step=0.5, key=f"pw_tol_{active_spectrum_name}")
                with col_cfg3:
                    st.write("")
                    find_pw_btn = st.button(T[lang]["tmds_path_find_btn"], type="primary", key=f"btn_find_pw_{active_spectrum_name}")

                if find_pw_btn or f"pw_res_{active_spectrum_name}" in st.session_state:
                    if find_pw_btn:
                        with st.spinner("Поиск цепочек реакций..." if lang == "ru" else "Traversing reaction pathways..."):
                            pw_list = find_transformation_pathways(
                                work_peaks, source_mass=src_m, target_mass=dst_m,
                                max_depth=depth_val, tol_mda=tol_pw_val, max_paths=10
                            )
                            st.session_state[f"pw_res_{active_spectrum_name}"] = pw_list

                    pw_results = st.session_state.get(f"pw_res_{active_spectrum_name}", [])
                    if pw_results:
                        st.success(T[lang]["tmds_path_found"].format(n=len(pw_results)))
                        pw_summary_rows = []
                        for p_idx, p_item in enumerate(pw_results, 1):
                            pw_summary_rows.append({
                                "Path_ID": f"Path #{p_idx}",
                                "Stages": p_item["depth"],
                                "Cumulative_Error_mDa": p_item["cumulative_error_mda"],
                                "Pathway": p_item["path_str"],
                            })
                        st_df(pd.DataFrame(pw_summary_rows))

                        for p_idx, p_item in enumerate(pw_results, 1):
                            with st.expander(f"📍 {p_item['path_str']} (Total err: {p_item['cumulative_error_mda']:.2f} mDa)", expanded=(p_idx == 1)):
                                steps_df = pd.DataFrame(p_item["steps"])
                                st_df(steps_df)
                    else:
                        st.info(T[lang]["tmds_path_not_found"])
            else:
                c_n1, c_n2 = st.columns([1, 1])
                with c_n1:
                    n_top_net = st.slider(T[lang]["tmds_net_peaks"], min_value=30, max_value=300, value=120, step=10, key=f"tmds_top_{active_spectrum_name}")
                with c_n2:
                    layout_choice = st.selectbox(
                        T[lang]["tmds_net_layout"],
                        [T[lang]["tmds_layout_spring"], T[lang]["tmds_layout_chem"]],
                        index=0,
                        key=f"tmds_lay_{active_spectrum_name}"
                    )
                l_type = "chemical" if layout_choice == T[lang]["tmds_layout_chem"] else "spring"
                with st.spinner("Построение сетевого графа реакций..." if lang == "ru" else "Constructing reaction network..."):
                    net_dict = build_tmds_network_graph(work_peaks, top_n=n_top_net, tol_mda=2.0, max_edges=250, layout=l_type)

                nodes_df = net_dict["nodes_df"]
                edges_df = net_dict["edges_df"]
                hubs_df = net_dict["hubs_df"]

                if nodes_df.empty or edges_df.empty:
                    st.warning("В заданных границах связей между пиками не обнаружено." if lang == "ru" else "No connected reactions found in the current selection.")
                else:
                    fig_net = go.Figure()
                    trans_colors = {
                        "CH2": "#0020C2", "O": "#D62728", "H2O": "#2CA02C",
                        "CO2": "#FF7F0E", "CO": "#9467BD", "H2": "#8C564B",
                        "NH3": "#E377C2", "SO3": "#17BECF"
                    }
                    for trans_type, g_edges in edges_df.groupby("Transformation"):
                        edge_x, edge_y = [], []
                        for _, e_row in g_edges.iterrows():
                            edge_x.extend([e_row["x0"], e_row["x1"], None])
                            edge_y.extend([e_row["y0"], e_row["y1"], None])
                        t_key = trans_type.split()[0]
                        e_color = trans_colors.get(t_key, "#7F7F7F")
                        fig_net.add_trace(go.Scatter(
                            x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1.2, color=e_color),
                            name=trans_type,
                            hoverinfo="none",
                            opacity=0.6,
                        ))

                    node_sizes = 5.0 + 3.0 * np.log1p(nodes_df["Degree"].values)
                    hover_texts = [
                        f"<b>{r['Node_ID']}</b><br>Formula: {r.get('Formula', 'N/A')}<br>Degree: {r['Degree']}<br>Class: {r.get('Hetero_Class', 'N/A')} / {r.get('Bio_Class', 'N/A')}"
                        for _, r in nodes_df.iterrows()
                    ]
                    fig_net.add_trace(go.Scatter(
                        x=nodes_df["x"], y=nodes_df["y"], mode="markers",
                        marker=dict(size=node_sizes, color=nodes_df["mass"], colorscale="Viridis", showscale=True, colorbar=dict(title="m/z", len=0.7), line=dict(width=1, color="black")),
                        text=hover_texts, hoverinfo="text", name="Peaks"
                    ))

                    fig_net.update_layout(
                        title=T[lang]["tmds_net_title"],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=560, plot_bgcolor="white"
                    )
                    st_plotly(fig_net)

                    if not hubs_df.empty:
                        st.markdown(f"##### {T[lang]['tmds_hubs_title']}")
                        st_df(hubs_df)

                    st.download_button(
                        label=T[lang]["tmds_dl_edges"],
                        data=edges_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"ChemoSuite_TMDS_Network_Edges_{active_spectrum_name}.csv",
                        mime="text/csv",
                        key=f"dl_tmds_edges_{active_spectrum_name}"
                    )

    with st_sub_desc:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is not None and not assigned_data.empty:
            st.subheader(f"{T[lang]['tab_desc']}: {active_spectrum_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T[lang]["desc_mn"], f"{assigned_data['mass'].mean():.2f} Da")
            c2.metric("H/C (Mn)", f"{assigned_data['H/C'].mean():.3f}")
            c3.metric("O/C (Mn)", f"{assigned_data['O/C'].mean():.3f}")
            c4.metric(T[lang]["desc_ai"], f"{assigned_data['AI'].mean():.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric(T[lang]["desc_mw"], f"{np.average(assigned_data['mass'], weights=assigned_data['intensity']):.2f} Da" if "intensity" in assigned_data.columns and assigned_data["intensity"].sum() > 0 else f"{assigned_data['mass'].mean():.2f} Da")
            c6.metric(T[lang]["desc_dbe"], f"{assigned_data['DBE'].mean():.2f}")
            c7.metric("NOSC", f"{assigned_data['NOSC'].mean():.3f}" if "NOSC" in assigned_data.columns else "N/A")
            c8.metric("DBE - O", f"{(assigned_data['DBE'] - assigned_data['O']).mean():.2f}")

        st.markdown("---")
        st.write(f"### {T[lang]['batch_desc_title']}")
        assigned_spectra_all = {
            s_name: s_obj["assigned_df"]
            for s_name, s_obj in st.session_state.get("spectra_db", {}).items()
            if s_obj.get("assigned_df") is not None and not s_obj["assigned_df"].empty
        }
        if assigned_spectra_all and CHEMO_ML_AVAILABLE and chemo_ml is not None:
            batch_rows = []
            for s_name, a_df in assigned_spectra_all.items():
                d = chemo_ml.extract_fticr_descriptors(a_df, s_name)
                if d: batch_rows.append(d)
            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                st_df(batch_df)
                col_dl_b1, col_dl_b2 = st.columns([1, 1])
                with col_dl_b1:
                    st.download_button(
                        label=T[lang]["batch_desc_dl_csv"],
                        data=batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_Batch_FTICR_Descriptors.csv",
                        mime="text/csv",
                        key="dl_batch_desc_csv",
                    )
                with col_dl_b2:
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        batch_df.to_excel(writer, index=False, sheet_name="FTICR_Descriptors")
                    st.download_button(
                        label=T[lang]["batch_desc_dl_excel"] if "batch_desc_dl_excel" in T[lang] else "📊 Скачать Excel (.xlsx)",
                        data=excel_buf.getvalue(),
                        file_name="ChemoSuite_Batch_FTICR_Descriptors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_batch_desc_excel",
                    )
