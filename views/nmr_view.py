"""
views/nmr_view.py — NMR Spectroscopy Studio (13C / 1H NMR).
"""

import io
import os
from typing import Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly
try:
    from modules import nmr_core
except ImportError:
    nmr_core = None


def _fmt_num(v, fmt: str = "{:.2f}", suffix: str = "") -> str:
    """Форматирует число; NaN/None показывается как «—» (величина не вычислима)."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(fv) else fmt.format(fv) + suffix


def render_nmr_studio(T: dict, lang: str):
    """Render Module 5: 13C / 1H NMR Spectroscopy Studio."""
    if nmr_core is None:
        st.error(T[lang]["nmr_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #1A365D 0%, #2B6CB0 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["nmr_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #E2E8F0; font-size: 13px;">{T[lang]["nmr_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown(f"### {T[lang]['nmr_sidebar_hdr']}")
        nmr_data_src = st.radio(
            T[lang]["nmr_src_label"],
            [T[lang]["nmr_src_upload"], T[lang]["nmr_src_synth"]],
            index=0,
            key="nmr_data_source_radio"
        )

        if st.session_state.get("_prev_nmr_data_src") != nmr_data_src:
            st.session_state["_prev_nmr_data_src"] = nmr_data_src
            st.session_state["nmr_samples"] = {}

        if nmr_data_src == T[lang]["nmr_src_upload"]:
            uploaded_nmr_files = st.file_uploader(
                T[lang]["nmr_uploader_label"],
                type=["csv", "txt", "tsv", "xy", "jdx", "dx"],
                accept_multiple_files=True,
                key="nmr_uploaded_files"
            )
            if uploaded_nmr_files:
                for uf in uploaded_nmr_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0]
                    try:
                        s_obj = nmr_core.parse_nmr_file(b_content, sample_id=s_id, nucleus="13C")
                        st.session_state["nmr_samples"][s_id] = s_obj
                        for w_msg in s_obj.metadata.get("JCAMP_WARNINGS", []) or []:
                            st.caption(f"⚠️ {uf.name}: {w_msg}")
                    except Exception as e_up:
                        st.warning(f"Ошибка чтения {uf.name}: {e_up}")
        else:
            if not st.session_state["nmr_samples"]:
                demo_nmr_dir = os.path.join("demo_data", "nmr")
                if os.path.exists(demo_nmr_dir) and any(f.endswith(".csv") for f in os.listdir(demo_nmr_dir)):
                    for fn in os.listdir(demo_nmr_dir):
                        if fn.endswith(".csv") and fn != "nmr_descriptors_matrix.csv":
                            fpath = os.path.join(demo_nmr_dir, fn)
                            with open(fpath, "rb") as f:
                                b_content = f.read()
                            s_obj = nmr_core.parse_nmr_file(b_content, sample_id=fn.replace(".csv", ""), nucleus="13C")
                            st.session_state["nmr_samples"][s_obj.sample_id] = s_obj
                else:
                    s_synth, _ = nmr_core.generate_synthetic_nmr_dataset(n_samples_per_class=6, save_dir=demo_nmr_dir)
                    st.session_state["nmr_samples"] = s_synth

        st.markdown(f"#### {T[lang]['nmr_preproc_hdr']}")
        base_method_choice = st.selectbox(
            T[lang]["nmr_baseline_method"],
            [T[lang]["nmr_base_asls"], T[lang]["nmr_base_linear"], T[lang]["nmr_base_poly"], T[lang]["nmr_base_none"]],
            index=0,
            key="nmr_baseline_select"
        )
        base_method = (
            "asls" if base_method_choice == T[lang]["nmr_base_asls"] else
            ("linear" if base_method_choice == T[lang]["nmr_base_linear"] else
            ("polynomial" if base_method_choice == T[lang]["nmr_base_poly"] else "none"))
        )

        norm_method_choice = st.selectbox(
            T[lang]["nmr_norm_label"],
            [T[lang]["nmr_norm_area"], T[lang]["nmr_norm_max"], T[lang]["nmr_norm_minmax"], T[lang]["nmr_norm_none"]],
            index=0,
            key="nmr_norm_select"
        )
        norm_method = (
            "area" if norm_method_choice == T[lang]["nmr_norm_area"] else
            ("max" if norm_method_choice == T[lang]["nmr_norm_max"] else
            ("minmax" if norm_method_choice == T[lang]["nmr_norm_minmax"] else "none"))
        )

        apply_sg_smoothing = st.checkbox(T[lang]["nmr_smooth_label"], value=True, key="nmr_chk_sg_smooth")

        recal_choice = st.selectbox(
            T[lang]["nmr_recal_label"],
            [T[lang]["nmr_recal_none"], T[lang]["nmr_recal_dmso"], T[lang]["nmr_recal_tms"]],
            index=0,
            key="nmr_recal_select"
        )
        recal_mode = "none"
        if "dmso" in recal_choice.lower():
            recal_mode = "dmso"
        elif "tms" in recal_choice.lower() or "dss" in recal_choice.lower():
            recal_mode = "tms"

    current_nmr_samples: Dict[str, nmr_core.NMRSample] = st.session_state["nmr_samples"]
    if not current_nmr_samples:
        st.info(T[lang]["nmr_no_samples"])
        st.stop()

    tab_nmr1, tab_nmr2, tab_nmr3, tab_nmr4 = st.tabs([
        T[lang]["nmr_tab1_title"],
        T[lang]["nmr_tab2_title"],
        T[lang]["nmr_tab3_title"],
        T[lang]["nmr_tab4_title"],
    ])

    sample_names = list(current_nmr_samples.keys())

    # =========================================================================
    # ВКЛАДКА 1: 📈 СПЕКТРАЛЬНЫЙ ПРОФИЛЬ И ИНТЕГРАЛЬНЫЕ ЗОНЫ
    # =========================================================================
    with tab_nmr1:
        col_ctl1, col_ctl2, col_ctl3 = st.columns([2.5, 1.5, 2.0])
        with col_ctl1:
            active_name = st.selectbox(T[lang]["nmr_active_sample_label"], sample_names, index=0, key="nmr_active_sample_select")
        with col_ctl2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_overlay = st.checkbox(T[lang]["nmr_show_all"], value=False, key="nmr_overlay_checkbox")
        with col_ctl3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_regions = st.checkbox(T[lang]["nmr_show_regions"], value=True, key="nmr_regions_checkbox")

        active_s = current_nmr_samples[active_name]
        active_shifts = active_s.chemical_shifts.copy()
        if recal_mode != "none":
            active_shifts, recal_delta = nmr_core.auto_recalibrate_nmr_shifts(
                active_shifts, active_s.intensity, mode=recal_mode, nucleus=active_s.nucleus
            )
            if abs(recal_delta) > 1e-4:
                st.info(
                    f"🎯 " + (f"Автокалибровка по {recal_mode.upper()}: сдвиг шкалы Δ = {recal_delta:+.2f} ppm" if lang == "ru" else f"Autorecalibration via {recal_mode.upper()}: scale shift Δ = {recal_delta:+.2f} ppm")
                )

        active_s_c, base_curve = nmr_core.subtract_baseline(active_s, method=base_method)
        active_y = active_s_c.corrected_intensity if base_method != "none" else active_s.intensity

        if apply_sg_smoothing:
            active_y = nmr_core.smooth_nmr_spectrum(active_y, window_length=9, polyorder=2)
        active_y_disp = nmr_core.normalize_nmr_spectrum(active_y, method=norm_method, chemical_shifts=active_shifts)

        # Выдвижная панель: Автопоиск пиков
        with st.expander(T[lang]["nmr_peaks_expander"], expanded=False):
            c_pk1, c_pk2, c_pk3 = st.columns([1.5, 1.5, 2])
            with c_pk1:
                pk_prom = st.slider(T[lang]["nmr_peaks_prominence"], min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="nmr_peak_prom_slider")
            with c_pk2:
                pk_dist = st.slider(T[lang]["nmr_peaks_dist"], min_value=2.0, max_value=15.0, value=4.0, step=0.5, key="nmr_peak_dist_slider")
            with c_pk3:
                st.markdown("<br>", unsafe_allow_html=True)
                show_peak_markers = st.checkbox(T[lang]["nmr_peaks_show_markers"], value=True, key="nmr_show_peak_markers_chk")

            df_peaks = nmr_core.find_and_annotate_nmr_peaks(
                active_shifts, active_y_disp,
                prominence=pk_prom, distance_ppm=pk_dist, lang=lang
            )
            if not df_peaks.empty:
                st.markdown(f"###### {T[lang]['nmr_peaks_tbl_title']}")
                st.dataframe(df_peaks, width="stretch", hide_index=True)
                st.download_button(
                    label=T[lang]["nmr_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_name}_13C_NMR_Peaks.csv",
                    mime="text/csv",
                    key="dl_nmr_peaks_csv",
                )

        # Построение спектрального графика Plotly
        fig_nmr = go.Figure()

        if show_overlay:
            for s_name, s_item in current_nmr_samples.items():
                s_shifts = s_item.chemical_shifts.copy()
                if recal_mode != "none":
                    s_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(s_shifts, s_item.intensity, mode=recal_mode, nucleus=s_item.nucleus)
                s_item_c, _ = nmr_core.subtract_baseline(s_item, method=base_method)
                y_disp = s_item_c.corrected_intensity if base_method != "none" else s_item.intensity
                if apply_sg_smoothing:
                    y_disp = nmr_core.smooth_nmr_spectrum(y_disp, window_length=9, polyorder=2)
                y_disp = nmr_core.normalize_nmr_spectrum(y_disp, method=norm_method, chemical_shifts=s_shifts)

                fig_nmr.add_trace(go.Scatter(
                    x=s_shifts,
                    y=y_disp,
                    mode="lines",
                    name=s_name,
                    opacity=0.75,
                    line=dict(width=1.5),
                ))
            fig_nmr.update_layout(
                title=f"Overlay of {len(current_nmr_samples)} 13C NMR Spectra" if lang == "en" else f"Наложение всех спектров 13C ЯМР ({len(current_nmr_samples)} проб)",
            )
        else:
            fig_nmr.add_trace(go.Scatter(
                x=active_shifts,
                y=active_y_disp,
                mode="lines",
                name=active_name,
                line=dict(color="#1A365D", width=2.2),
            ))

            # Отображение базовой линии
            if base_method != "none" and len(base_curve) > 0:
                base_disp = nmr_core.normalize_nmr_spectrum(base_curve, method=norm_method, chemical_shifts=active_shifts)
                fig_nmr.add_trace(go.Scatter(
                    x=active_shifts,
                    y=base_disp,
                    mode="lines",
                    name="AsLS Baseline" if lang == "en" else "Базовая линия",
                    line=dict(color="#E53E3E", width=1.2, dash="dash"),
                ))

            # Маркеры пиков
            if not df_peaks.empty and show_peak_markers:
                fig_nmr.add_trace(go.Scatter(
                    x=df_peaks["Chemical_Shift_ppm"],
                    y=df_peaks["Intensity"] if norm_method == "none" else df_peaks["Relative_Height"] * float(np.max(active_y_disp)),
                    mode="markers+text",
                    name="Resonances" if lang == "en" else "Резонансы",
                    text=[f"{p:.1f}" for p in df_peaks["Chemical_Shift_ppm"]],
                    textposition="top center",
                    textfont=dict(size=9, color="#742A2A"),
                    marker=dict(symbol="triangle-down", size=9, color="#E53E3E"),
                ))

            fig_nmr.update_layout(
                title=f"{active_name} — 13C CP/MAS NMR Profile" if lang == "en" else f"{active_name} — Спектральный профиль 13C ЯМР",
            )

        # Канонические интегральные зоны IHSS (Цветная полупрозрачная подсветка)
        if show_regions:
            zones = [
                (0.0, 45.0, "Alkyl C", "#BEE3F8", "rgba(190, 227, 248, 0.22)"),
                (45.0, 90.0, "C-O Alkyl C", "#C6F6D5", "rgba(198, 246, 213, 0.22)"),
                (90.0, 110.0, "Acetal C", "#B2F5EA", "rgba(178, 245, 234, 0.22)"),
                (110.0, 145.0, "Aromatic C", "#FEEBC8", "rgba(254, 235, 200, 0.25)"),
                (145.0, 165.0, "Phenolic C", "#FBD38D", "rgba(251, 211, 141, 0.22)"),
                (165.0, 220.0, "Carboxyl/Carbonyl C", "#FED7D7", "rgba(254, 215, 215, 0.22)"),
            ]
            for z_min, z_max, z_name, z_col, z_fill in zones:
                fig_nmr.add_vrect(
                    x0=z_min, x1=z_max,
                    fillcolor=z_fill,
                    layer="below",
                    line_width=1,
                    line_dash="dot",
                    line_color=z_col,
                    annotation_text=z_name,
                    annotation_position="top left",
                    annotation_font=dict(size=9, color="#4A5568")
                )

        # СТАНДАРТ ЯМР: ОСЬ X ИНВЕРТИРОВАНА (ОТ 220 PPM К -5 PPM)
        fig_nmr.update_xaxes(
            title_text="Chemical Shift δ (ppm) [13C]",
            autorange="reversed",
            range=[225.0, -10.0],
            gridcolor="#EDF2F7",
        )
        fig_nmr.update_yaxes(
            title_text="Intensity (Normalized)" if norm_method != "none" else "Raw Intensity",
            gridcolor="#EDF2F7",
        )
        fig_nmr.update_layout(
            height=480,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st_plotly(fig_nmr)

        # Расчет и вывод фундаментальных метрик активного образца
        ind_single = nmr_core.calculate_13c_nmr_descriptors(active_shifts, active_y, sample_id=active_name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Aromaticity fa (%)", _fmt_num(ind_single["NMR_fa_aromaticity"], "{:.1f}", "%"))
        m2.metric("Hydrophobicity (HB/HI)", _fmt_num(ind_single["NMR_hb_hi_index"]))
        m3.metric("Alkyl / O-Alkyl", _fmt_num(ind_single["NMR_alkyl_o_alkyl"]))
        m4.metric("Lignin Signature (%)", _fmt_num(ind_single["NMR_lignin_signature"], "{:.1f}", "%"))
        m5.metric("Acetal C 90–110 (%)", _fmt_num(ind_single["NMR_C_acetal_pct"], "{:.1f}", "%"))
        if not np.isfinite(float(ind_single["NMR_fa_aromaticity"])):
            st.warning(
                "Спектр не содержит положительного интегрального сигнала в 0–220 ppm: доли углерода и индексы не вычислимы."
                if lang == "ru" else
                "The spectrum has no positive integrated signal in 0–220 ppm: carbon fractions and indices cannot be computed."
            )
        elif any(not np.isfinite(float(ind_single[k])) for k in ("NMR_hb_hi_index", "NMR_alkyl_o_alkyl", "NMR_coo_ar_ratio")):
            st.caption(
                "«—»: отношение не вычислимо (пул-знаменатель практически отсутствует)."
                if lang == "ru" else
                "“—”: ratio not computable (denominator pool essentially absent)."
            )

    # =========================================================================
    # ВКЛАДКА 2: 📊 РАСПРЕДЕЛЕНИЕ УГЛЕРОДНЫХ ПУЛОВ
    # =========================================================================
    with tab_nmr2:
        st.markdown(
            "##### " + ("Carbon Skeleton Distribution and Structural Fingerprint" if lang == "en" else "Распределение углеродного скелета и структурный отпечаток")
        )

        ind_current = nmr_core.calculate_13c_nmr_descriptors(active_shifts, active_y, sample_id=active_name)

        col_g1, col_g2 = st.columns([1.2, 1.0])
        with col_g1:
            # Bar chart: 6 интегральных зон IHSS-типа (Kögel-Knabner 2002)
            six_bins = nmr_core.ihss_6bin_fractions(ind_current)
            df_fractions = pd.DataFrame({
                "Region": list(six_bins.keys()),
                "Fraction_Pct": [float(np.nan_to_num(v, nan=0.0)) for v in six_bins.values()],
            })
            if not np.isfinite(float(ind_current["NMR_fa_aromaticity"])):
                st.warning("Доли углерода не вычислимы для пустого спектра." if lang == "ru" else "Carbon fractions are not computable for an empty spectrum.")
            fig_frac = px.bar(
                df_fractions,
                x="Region",
                y="Fraction_Pct",
                color="Region",
                color_discrete_sequence=["#3182CE", "#38A169", "#319795", "#DD6B20", "#D69E2E", "#E53E3E"],
                text="Fraction_Pct",
                title=f"{active_name} — Carbon Distribution (% Total C)",
            )
            fig_frac.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_frac.update_layout(height=380, showlegend=False, yaxis=dict(title="% of Total Carbon", range=[0, max(60.0, df_fractions["Fraction_Pct"].max() + 8)]))
            st_plotly(fig_frac)

        with col_g2:
            # Donut chart
            fig_donut = px.pie(
                df_fractions,
                names="Region",
                values="Fraction_Pct",
                hole=0.45,
                color="Region",
                color_discrete_sequence=["#3182CE", "#38A169", "#319795", "#DD6B20", "#D69E2E", "#E53E3E"],
                title=f"{active_name} — Carbon Fractions Donut",
            )
            fig_donut.update_traces(textinfo="label+percent")
            fig_donut.update_layout(height=380, showlegend=False)
            st_plotly(fig_donut)

        # Радарная диаграмма (Radar Fingerprint) активного образца vs Эталоны
        st.markdown("###### " + ("Structural Radar Fingerprint vs Analytical Standards" if lang == "en" else "Радарный структурный отпечаток в сравнении со стандартами"))
        categories = ["Aromaticity fa", "Hydrophobicity (HB/HI*10)", "Alkyl/O-Alkyl (*10)", "Phenolic C (%)", "Carboxyl C (%)", "Aliphatic C (%)"]

        # Значения активного образца с защитой от NaN / inf
        def _radar_val(key: str, scale: float = 1.0) -> float:
            v = float(ind_current.get(key, np.nan))
            return float(min(80.0, v * scale)) if np.isfinite(v) else 0.0

        radar_keys = [("NMR_fa_aromaticity", 1.0), ("NMR_hb_hi_index", 10.0), ("NMR_alkyl_o_alkyl", 10.0),
                      ("NMR_C_phenolic_pct", 1.0), ("NMR_C_carboxyl_pct", 1.0), ("NMR_C_aliph_pct", 1.0)]
        val_active = [_radar_val(k, sc) for k, sc in radar_keys]
        missing_axes = [cat for cat, (k, _) in zip(categories, radar_keys) if not np.isfinite(float(ind_current.get(k, np.nan)))]
        if missing_axes:
            st.caption(("Не вычислимо (показано как 0): " if lang == "ru" else "Not computable (drawn as 0): ") + ", ".join(missing_axes))
        val_active.append(val_active[0])

        # Эталон Байкал DOM
        val_ref_dom = [22.0, 12.0, 8.0, 3.5, 14.0, 26.0, 22.0]
        # Эталон Шлам-лигнин
        val_ref_lignin = [52.0, 28.0, 14.0, 16.0, 14.0, 16.0, 52.0]

        cat_closed = categories + [categories[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=val_active, theta=cat_closed, fill="toself", name=active_name, line=dict(color="#1A365D", width=2)))
        fig_radar.add_trace(go.Scatterpolar(r=val_ref_dom, theta=cat_closed, name="Ref: Pristine DOM" if lang == "en" else "Эталон: Природное РОВ", line=dict(color="#38A169", width=1.5, dash="dash")))
        fig_radar.add_trace(go.Scatterpolar(r=val_ref_lignin, theta=cat_closed, name="Ref: Sludge Lignin" if lang == "en" else "Эталон: Шлам-лигнин БЦБК", line=dict(color="#DD6B20", width=1.5, dash="dot")))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 65])),
            height=420,
            showlegend=True,
            template="plotly_white",
        )
        st_plotly(fig_radar)

    # =========================================================================
    # ВКЛАДКА 3: 🔍 ДЕКОНВОЛЮЦИЯ И СПЕКТРАЛЬНАЯ БИБЛИОТЕКА
    # =========================================================================
    with tab_nmr3:
        sec_tab1, sec_tab2 = st.tabs([
            "✂️ " + ("Aromatic / O-aryl Sub-band Decomposition (110–165 ppm)" if lang == "en" else "Разложение ароматической / O-арильной зоны (110–165 ppm)"),
            "🏷️ " + ("IHSS Spectral Library Matching" if lang == "en" else "Идентификация по библиотеке эталонов IHSS")
        ])

        with sec_tab1:
            st.markdown(
                "###### " + (
                    "Empirical Gaussian sub-bands: protonated aromatic, C-substituted/condensed aromatic and O-aryl carbon"
                    if lang == "en" else
                    "Эмпирические гауссовы подполосы: протонированный, C-замещенный/конденсированный ароматический и O-арильный углерод"
                )
            )
            st.caption(
                "Подполосы не являются мономерами лигнина: в 13C спектрах ПОВ зона 145–165 ppm содержит также C танинов, фенолов и "
                "других O-арильных структур, поэтому отношения S/G и H/G по 1D 13C спектру не вычисляются."
                if lang == "ru" else
                "Sub-bands are not lignin monomers: in 13C spectra of NOM the 145–165 ppm region also holds tannin, phenolic and other "
                "O-aryl carbon, so S/G and H/G ratios are not derived from 1D 13C spectra."
            )
            col_dec1, col_dec2, col_dec3 = st.columns([2.5, 1.5, 2.0])
            with col_dec1:
                deconv_target_name = st.selectbox(T[lang]["nmr_deconv_sample"], sample_names, index=0, key="nmr_deconv_sample_select")
            with col_dec2:
                n_gaussians = st.slider(T[lang]["nmr_deconv_n_peaks"], min_value=3, max_value=4, value=4, step=1, key="nmr_n_gauss_slider")
            with col_dec3:
                st.markdown("<br>", unsafe_allow_html=True)
                deconv_run_btn = st.button(T[lang]["nmr_deconv_btn"], type="primary", key="btn_run_nmr_deconv")

            target_dec_s = current_nmr_samples[deconv_target_name]
            target_dec_shifts = target_dec_s.chemical_shifts.copy()
            if recal_mode != "none":
                target_dec_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(target_dec_shifts, target_dec_s.intensity, mode=recal_mode, nucleus=target_dec_s.nucleus)
            target_dec_s_c, _ = nmr_core.subtract_baseline(target_dec_s, method=base_method)
            y_dec_target = target_dec_s_c.corrected_intensity if base_method != "none" else target_dec_s.intensity

            deconv_res = nmr_core.deconvolve_aromatic_region(
                target_dec_shifts, y_dec_target, n_peaks=n_gaussians
            )

            if deconv_res.get("success"):
                c_q1, c_q2 = st.columns(2)
                c_q1.metric(T[lang]["nmr_deconv_r2"], _fmt_num(deconv_res["r2"], "{:.4f}"))
                c_q2.metric(T[lang]["nmr_deconv_rmse"], _fmt_num(deconv_res["rmse"], "{:.4f}"))
                if not deconv_res.get("fit_ok", False):
                    st.warning(
                        ("⚠️ Качество разложения неудовлетворительно — площади подполос не интерпретировать: " if lang == "ru"
                         else "⚠️ Poor decomposition — do not interpret sub-band areas: ")
                        + "; ".join(deconv_res.get("warnings", []))
                    )

                fig_dec = go.Figure()
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["x"], y=deconv_res["y"],
                    mode="lines", name="Observed" if lang == "en" else "Эксперимент",
                    line=dict(color="#1A365D", width=2.5)
                ))
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["x"], y=deconv_res["y_fit"],
                    mode="lines", name="Cumulative Fit" if lang == "en" else "Суммарный фит",
                    line=dict(color="#E53E3E", width=1.8, dash="dash")
                ))

                c_colors = ["#3182CE", "#38A169", "#DD6B20", "#805AD5"]
                for idx_c, comp in enumerate(deconv_res["components"]):
                    fig_dec.add_trace(go.Scatter(
                        x=deconv_res["x"], y=comp["curve"],
                        mode="lines",
                        name=f"{comp['label']} ({comp['area_pct']}%)",
                        line=dict(color=c_colors[idx_c % len(c_colors)], width=1.5),
                        fill="tozeroy",
                        opacity=0.4
                    ))

                fig_dec.update_xaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", gridcolor="#EDF2F7")
                fig_dec.update_yaxes(title_text="Intensity", gridcolor="#EDF2F7")
                fig_dec.update_layout(
                    title=f"{deconv_target_name} — 13C NMR Aromatic / O-aryl Sub-bands",
                    template="plotly_white",
                    height=440
                )
                st_plotly(fig_dec)

                # Таблица компонент
                df_comp = pd.DataFrame([
                    {
                        "Band": c["label"],
                        "Center_ppm": c["center_ppm"],
                        "Width_ppm": c["width_ppm"],
                        "Fraction_Area_Pct": f"{c['area_pct']}%",
                        "At_Bound": c.get("at_bound", False),
                    }
                    for c in deconv_res["components"]
                ])
                st.markdown(f"###### {T[lang]['nmr_deconv_tbl_title']}")
                st.dataframe(df_comp, width="stretch", hide_index=True)
            else:
                st.warning(
                    (f"Разложение не выполнено: {deconv_res.get('message', 'Недостаточно точек')}" if lang == "ru"
                     else f"Decomposition failed: {deconv_res.get('message', 'not enough points')}")
                )

        with sec_tab2:
            st.markdown("##### " + T[lang]["nmr_lib_hdr"])
            c_lb1, c_lb2, c_lb3 = st.columns([2.5, 1.5, 2.0])
            with c_lb1:
                lib_target_name = st.selectbox(T[lang]["nmr_lib_sample_select"], sample_names, index=0, key="nmr_lib_sample_select")
            with c_lb2:
                lib_top_n = st.slider(T[lang]["nmr_lib_top_n"], min_value=3, max_value=8, value=5, step=1, key="nmr_lib_top_n_slider")
            with c_lb3:
                st.markdown("<br>", unsafe_allow_html=True)
                lib_run_btn = st.button(T[lang]["nmr_lib_btn"], type="primary", key="btn_run_nmr_lib_match")

            target_lib_s = current_nmr_samples[lib_target_name]
            target_lib_shifts = target_lib_s.chemical_shifts.copy()
            if recal_mode != "none":
                target_lib_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(target_lib_shifts, target_lib_s.intensity, mode=recal_mode, nucleus=target_lib_s.nucleus)
            target_lib_s_c, _ = nmr_core.subtract_baseline(target_lib_s, method=base_method)
            y_lib_target = target_lib_s_c.corrected_intensity if base_method != "none" else target_lib_s.intensity

            match_df = nmr_core.match_spectrum_against_library(target_lib_shifts, y_lib_target, top_n=lib_top_n, lang=lang)

            if not match_df.empty:
                top_match = match_df.iloc[0]
                top_std_id = top_match["Reference_ID"]
                top_hqi = top_match["HQI_Similarity_Pct"]
                top_name = top_match["Standard_Name"]

                if top_hqi >= 85.0:
                    st.success(
                        f"🏆 **Топ-1 совпадение:** {top_name} (HQI = **{top_hqi:.1f}%**, Expected fa = **{top_match['Expected_Aromaticity_fa']}%**)"
                        if lang == "ru" else
                        f"🏆 **Top-1 Match:** {top_name} (HQI = **{top_hqi:.1f}%**, Expected fa = **{top_match['Expected_Aromaticity_fa']}%**)"
                    )
                else:
                    st.info(
                        f"🔎 **Лучшее совпадение:** {top_name} (HQI = **{top_hqi:.1f}%**)"
                        if lang == "ru" else
                        f"🔎 **Closest Standard:** {top_name} (HQI = **{top_hqi:.1f}%**)"
                    )

                # Наложение: образец vs эталон
                ref_curve = nmr_core.synthesize_reference_13c_curve(top_std_id, target_lib_shifts)
                s_norm = nmr_core.normalize_nmr_spectrum(y_lib_target, method="minmax")
                r_norm = nmr_core.normalize_nmr_spectrum(ref_curve, method="minmax")

                fig_lib = go.Figure()
                fig_lib.add_trace(go.Scatter(
                    x=target_lib_shifts, y=s_norm,
                    mode="lines", name=f"{lib_target_name} ({T[lang]['nmr_lib_sample_trace']})",
                    line=dict(color="#1A365D", width=2.2)
                ))
                fig_lib.add_trace(go.Scatter(
                    x=target_lib_shifts, y=r_norm,
                    mode="lines", name=f"{top_name} ({T[lang]['nmr_lib_ref_trace']})",
                    line=dict(color="#E53E3E", width=1.8, dash="dash")
                ))
                fig_lib.update_xaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", gridcolor="#EDF2F7")
                fig_lib.update_yaxes(title_text="Normalized Intensity [0, 1]", gridcolor="#EDF2F7")
                fig_lib.update_layout(
                    title=f"{T[lang]['nmr_lib_overlay_title']} — {lib_target_name} vs {top_name}",
                    template="plotly_white",
                    height=420
                )
                st_plotly(fig_lib)

                st.markdown(f"###### {T[lang]['nmr_lib_tbl_title']}")
                st.dataframe(match_df, width="stretch", hide_index=True)

    # =========================================================================
    # ВКЛАДКА 4: 🧬 ХЕМОМЕТРИЧЕСКИЕ ИНДЕКСЫ И ЭКСПОРТ В ML
    # =========================================================================
    with tab_nmr4:
        processed_samples = []
        for s in current_nmr_samples.values():
            s_copy = nmr_core.NMRSample(
                sample_id=s.sample_id,
                nucleus=s.nucleus,
                chemical_shifts=s.chemical_shifts.copy(),
                intensity=s.intensity.copy(),
                metadata=dict(s.metadata)
            )
            if recal_mode != "none":
                s_copy.chemical_shifts, _ = nmr_core.auto_recalibrate_nmr_shifts(
                    s_copy.chemical_shifts, s_copy.intensity, mode=recal_mode, nucleus=s_copy.nucleus
                )
            processed_samples.append(s_copy)

        df_nmr_desc = nmr_core.batch_process_nmr_samples(
            processed_samples,
            baseline_method=base_method
        )

        st.markdown(
            f"##### {len(df_nmr_desc)} " + ("Processed Samples Descriptor Matrix" if lang == "en" else "Сводная матрица дескрипторов ЯМР")
        )

        st.dataframe(
            df_nmr_desc.style.format({
                "NMR_fa_aromaticity": "{:.1f}%",
                "NMR_hb_hi_index": "{:.2f}",
                "NMR_alkyl_o_alkyl": "{:.2f}",
                "NMR_lignin_signature": "{:.1f}%",
                "NMR_coo_ar_ratio": "{:.2f}",
                "NMR_C_aliph_pct": "{:.1f}%",
                "NMR_C_O_alkyl_pct": "{:.1f}%",
                "NMR_C_aromatic_pct": "{:.1f}%",
                "NMR_C_carboxyl_pct": "{:.1f}%",
                "NMR_C_carbonyl_pct": "{:.1f}%",
                "NMR_C_methoxy_pct": "{:.1f}%",
                "NMR_C_phenolic_pct": "{:.1f}%",
                "NMR_C_O_alkyl_60_90_pct": "{:.1f}%",
                "NMR_C_acetal_pct": "{:.1f}%",
                "NMR_C_carbohydrate_pct": "{:.1f}%",
                "NMR_C_ar_HC_pct": "{:.1f}%",
            }, na_rep="—"),
            width="stretch",
            hide_index=True
        )

        # Диаграмма рассеяния: Ароматичность fa vs Лигниновый индекс с защитой размера пузырьков
        df_scatter = df_nmr_desc.copy()
        df_scatter["_bubble_size"] = np.clip(
            df_scatter["NMR_hb_hi_index"].fillna(1.0).astype(float),
            0.5, 15.0
        )
        fig_scatter = px.scatter(
            df_scatter,
            x="NMR_fa_aromaticity",
            y="NMR_lignin_signature",
            hover_name="Sample_ID",
            size="_bubble_size",
            color="NMR_fa_aromaticity",
            color_continuous_scale="Viridis",
            hover_data={"_bubble_size": False, "NMR_hb_hi_index": ":.2f"},
            title="Aromaticity (fa %) vs Lignin Signature (%) Across Samples" if lang == "en" else "Степень ароматичности fa (%) vs Лигниновый индекс (%) по всем пробам",
            labels={"NMR_fa_aromaticity": "Aromaticity fa (%)", "NMR_lignin_signature": "Lignin Signature (%)"}
        )
        fig_scatter.update_layout(height=420, template="plotly_white")
        st_plotly(fig_scatter)

        c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
        with c_d1:
            st.download_button(
                label=T[lang]["nmr_dl_csv"],
                data=df_nmr_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_NMR_Descriptors.csv",
                mime="text/csv",
                key="dl_nmr_descriptors_csv",
            )
        with c_d2:
            excel_nmr_buf = io.BytesIO()
            with pd.ExcelWriter(excel_nmr_buf, engine="openpyxl") as writer:
                df_nmr_desc.to_excel(writer, index=False, sheet_name="NMR_Descriptors")
            st.download_button(
                label=T[lang]["nmr_dl_excel"],
                data=excel_nmr_buf.getvalue(),
                file_name="ChemoSuite_NMR_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_nmr_descriptors_excel",
            )
        with c_d3:
            if st.button(T[lang]["nmr_sync_ml_btn"], type="primary", key="btn_sync_nmr_ml_descriptors"):
                st.session_state["nmr_ml_descriptors"] = df_nmr_desc
                st.toast(T[lang]["nmr_sync_success"].format(n=len(df_nmr_desc)), icon="✅")
                st.rerun()
