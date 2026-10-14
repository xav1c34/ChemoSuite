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
                                        c_cid = pc_item.get("CID") or pc_item.get("cid")
                                        st.markdown(f"**{T[lang]['pubchem_res_name']}** `{c_name}`")
                                        st.markdown(f"**{T[lang]['pubchem_res_smiles']}** `{pc_item.get('SMILES', 'N/A')}`")
                                        if c_cid:
                                            st.link_button(f"{T[lang]['pubchem_link_btn']} #{c_cid}", f"https://pubchem.ncbi.nlm.nih.gov/compound/{c_cid}")

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
        all_spectra_list = list(st.session_state.get("spectra_db", {}).keys())
        if len(all_spectra_list) < 2:
            st.info(T[lang]["cmp_need_two"])
        else:
            col_s1, col_s2, col_s3 = st.columns([2, 2, 1.5])
            with col_s1: name_a = st.selectbox(T[lang]["cmp_spec_a"], all_spectra_list, index=0, key="cmp_spec_a")
            with col_s2: name_b = st.selectbox(T[lang]["cmp_spec_b"], all_spectra_list, index=1 if len(all_spectra_list) > 1 else 0, key="cmp_spec_b")
            with col_s3: tol_comp = st.number_input(T[lang]["cmp_tol"], min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="cmp_ppm_tol")

            sample_a, sample_b = st.session_state["spectra_db"][name_a], st.session_state["spectra_db"][name_b]
            peaks_a, peaks_b = ensure_sample_peaks(sample_a), ensure_sample_peaks(sample_b)

            if peaks_a is not None and not peaks_a.empty and peaks_b is not None and not peaks_b.empty and name_a != name_b:
                m_a, m_b = align_two_spectra_fast(peaks_a, peaks_b, ppm_tol=tol_comp)
                n_a, n_b, n_common = len(peaks_a), len(peaks_b), len(m_a)
                jaccard = n_common / (n_a + n_b - n_common) if (n_a + n_b - n_common) > 0 else 0.0

                m1, m2, m3 = st.columns(3)
                m1.metric(f"Peaks in {name_a}", f"{n_a:,}")
                m2.metric(f"Peaks in {name_b}", f"{n_b:,}")
                m3.metric(T[lang]["cmp_common"], f"{n_common:,}", delta=T[lang]["cmp_jaccard"].format(val=jaccard))

                op_choice = st.selectbox("Спектральная алгебра / Operation:", ["and", "or", "sub_a_b", "sub_b_a", "xor"])
                if st.button("🚀 Выполнить операцию" if lang == "ru" else "🚀 Execute Algebra", type="primary"):
                    alg_df = perform_spectral_algebra(peaks_a, peaks_b, operation=op_choice, ppm_tol=tol_comp)
                    if not alg_df.empty:
                        alg_df["norm_intensity"] = (alg_df["intensity"] / alg_df["intensity"].max()) * 100.0
                        new_alg_name = f"{name_a}_{op_choice}_{name_b}.csv"
                        fpath_alg = os.path.join(STORAGE_DIR, new_alg_name)
                        alg_df.to_csv(fpath_alg, sep="\t", index=False)
                        st.session_state["spectra_db"][new_alg_name] = {
                            "raw_path": fpath_alg, "file_bytes": alg_df.to_csv(sep="\t", index=False).encode("utf-8"),
                            "parsed_peaks": alg_df, "assigned_df": None, "raw_df": alg_df,
                        }
                        st.success(f"Создан спектр: {new_alg_name} ({len(alg_df)} пиков)")
                        st.rerun()

    with st_sub_tmds:
        work_peaks = current_sample.get("assigned_df") if (current_sample and current_sample.get("assigned_df") is not None and not current_sample["assigned_df"].empty) else (current_sample.get("parsed_peaks") if current_sample else None)
        if work_peaks is not None and not work_peaks.empty:
            flux_res = compute_geochemical_vector_fluxes(work_peaks, tol_mda=2.0, top_n=1000)
            tmds_sum = flux_res.get("summary_df", pd.DataFrame())
            if not tmds_sum.empty:
                st.markdown(f"#### {T[lang]['tmds_flux_header']}")
                fig_tmds = px.bar(tmds_sum, x="Transformation", y="Count", color="Flux_Type", text=tmds_sum["Share_pct"].apply(lambda v: f"{v:.1f}%"))
                fig_tmds.update_layout(showlegend=True, xaxis_tickangle=-25, height=420)
                st_plotly(fig_tmds)
                st_df(tmds_sum)

    with st_sub_desc:
        assigned_data = current_sample.get("assigned_df") if current_sample else None
        if assigned_data is not None and not assigned_data.empty:
            st.subheader(f"{T[lang]['tab_desc']}: {active_spectrum_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T[lang]["desc_mn"], f"{assigned_data['mass'].mean():.2f} Da")
            c2.metric("H/C (Mn)", f"{assigned_data['H/C'].mean():.3f}")
            c3.metric("O/C (Mn)", f"{assigned_data['O/C'].mean():.3f}")
            c4.metric(T[lang]["desc_ai"], f"{assigned_data['AI'].mean():.3f}")

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
                st.download_button(
                    label=T[lang]["batch_desc_dl_csv"], data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="ChemoSuite_Batch_FTICR_Descriptors.csv", mime="text/csv", key="dl_batch_desc_csv"
                )
