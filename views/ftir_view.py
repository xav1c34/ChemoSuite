"""
views/ftir_view.py — FTIR Spectroscopy Studio View.
"""

import io
from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly
try:
    from modules import ftir_core
except ImportError:
    ftir_core = None


def render_ftir_studio(T: dict, lang: str):
    """Render Module 4: Vibrational Spectroscopy Studio (FTIR)."""
    st.title(T[lang]["ftir_title"])
    st.caption(T[lang]["ftir_subtitle"])

    with st.popover("ℹ️ " + ("Руководство: Колебательная спектроскопия (FTIR) в анализе РОВ и лигнина" if lang == "ru" else "Guide: FTIR Spectroscopy in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы ИК-спектроскопии органического вещества

В отличие от электронной спектроскопии (УФ-Вид/EEM), отражающей только переходы сопряженных $\pi$-систем, **ИК-Фурье спектроскопия регистрирует фундаментальные колебания ковалентных связей** молекулярного каркаса без разрушения или ионизации пробы.

| Диапазон ($\text{см}^{-1}$) | Колебания связей | Молекулярное отнесение | Экологическое значение |
| :--- | :--- | :--- | :--- |
| **$3400–3200$** | $\nu(\text{O}-\text{H}), \nu(\text{N}-\text{H})$ | Фенолы, спирты, карбоновые кислоты, влага | Общий гидрофильный и водородно-связанный пул |
| **$2960–2850$** | $\nu_{as,s}(\text{C}-\text{H})$ | Алифатические цепи ($\text{CH}_3, \text{CH}_2$) | Степень алифатичности и разветвленности структуры |
| **$1725–1710$** | $\nu(\text{C}=\text{O})$ | Карбоксилы ($\text{COOH}$), неконъюгированные кетоны | Индикатор степени окисленности и гумификации |
| **$1650–1640$** | $\nu(\text{C}=\text{O})_{conj}$ | Хиноны, карбонилы, сопряженные с кольцом, Амид I | Редокс-активные центры природных гуминовых систем |
| **$1605–1600$** | $\nu(\text{C}=\text{C})_{arom}$ | Скелетные колебания ароматических колец | Степень конденсированности и ароматичности |
| **$1510$** | $\nu(\text{C}=\text{C})_{lignin}$ | **Гваяцильное кольцо лигнина (G-ring)** | **Критический специфический маркер шлам-лигнина БЦБК** |
| **$1270, 1220$** | $\nu(\text{C}-\text{O}), \nu(\text{C}-\text{C})$ | Гваяцильные эфирные связи лигнина | Дифференциация лигнина от природных полисахаридов |
| **$1035–1030$** | $\nu(\text{C}-\text{O})$ | Спирты, полисахариды, целлюлоза, метоксилы | Доля биогенного углеводного и терригенного пула |

---

### ⚙️ Алгоритмы предобработки и деконволюции:
* **AsLS (Asymmetric Least Squares):** удаление наклонного фона рассеяния и флуоресценции без смещения максимумов пиков.
* **Деконволюция $1500–1800\text{ см}^{-1}$:** разложение сложной суперпозиции полос на элементарные Гауссовы составляющие методом нелинейной оптимизации Левенберга — Марквардта с расчетом площадей и FWHM.
""")

    if ftir_core is None:
        st.error(T[lang]["ftir_err_missing"])
        st.stop()

    with st.sidebar:
        st.header(T[lang]["ftir_sidebar_hdr"])
        ftir_data_src = st.radio(
            T[lang]["ftir_src_label"],
            [T[lang]["ftir_src_upload"], T[lang]["ftir_src_synth"]],
            index=0,
            key="ftir_data_source_radio"
        )
        if st.session_state.get("_prev_ftir_data_src") != ftir_data_src:
            st.session_state["_prev_ftir_data_src"] = ftir_data_src
            st.session_state["ftir_samples"] = {}

        ftir_files = []
        if ftir_data_src == T[lang]["ftir_src_upload"]:
            ftir_files = st.file_uploader(
                T[lang]["ftir_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "dpt", "txt", "xy"],
                key="ftir_files_uploader"
            )

        st.markdown("---")
        st.subheader(T[lang]["ftir_preproc_hdr"])
        base_method_choice = st.selectbox(
            T[lang]["ftir_baseline_method"],
            [T[lang]["ftir_base_als"], T[lang]["ftir_base_rubber"], T[lang]["ftir_base_linear"], T[lang]["ftir_base_none"]],
            index=0,
            key="ftir_base_method_choice"
        )
        base_method = "als" if base_method_choice == T[lang]["ftir_base_als"] else ("rubberband" if base_method_choice == T[lang]["ftir_base_rubber"] else ("linear" if base_method_choice == T[lang]["ftir_base_linear"] else "none"))

        norm_method_choice = st.selectbox(
            T[lang]["ftir_norm_label"],
            [T[lang]["ftir_norm_snv"], T[lang]["ftir_norm_minmax"], T[lang]["ftir_norm_vec"], T[lang]["ftir_norm_none"]],
            index=0,
            key="ftir_norm_method_choice"
        )
        norm_method = "snv" if norm_method_choice == T[lang]["ftir_norm_snv"] else ("minmax" if norm_method_choice == T[lang]["ftir_norm_minmax"] else ("vector" if norm_method_choice == T[lang]["ftir_norm_vec"] else "none"))

    # Загрузка данных
    current_ftir_samples: Dict[str, Any] = {}

    if ftir_data_src == T[lang]["ftir_src_synth"]:
        synth_samples, synth_desc = ftir_core.generate_synthetic_ftir_dataset(n_samples=12, random_state=42)
        current_ftir_samples = synth_samples
        st.session_state["ftir_samples"] = synth_samples
    else:
        if ftir_files:
            for f in ftir_files:
                try:
                    s_obj = ftir_core.parse_ftir_file(f.getvalue(), sample_id=f.name)
                    current_ftir_samples[f.name] = s_obj
                except Exception as e_parse:
                    st.warning(f"Ошибка чтения {f.name}: {e_parse}")
            if current_ftir_samples:
                st.session_state["ftir_samples"] = current_ftir_samples
        else:
            current_ftir_samples = st.session_state.get("ftir_samples", {})

    if not current_ftir_samples:
        st.info(T[lang]["ftir_no_samples"])
        st.stop()

    tab_ftir1, tab_ftir2, tab_ftir3, tab_ftir4 = st.tabs([
        T[lang]["ftir_tab1_title"],
        T[lang]["ftir_tab2_title"],
        T[lang]["ftir_tab3_title"],
        T[lang]["ftir_tab4_title"],
    ])

    sample_names = list(current_ftir_samples.keys())

    # =========================================================================
    # ВКЛАДКА 1: СПЕКТРАЛЬНЫЙ ПРОФИЛЬ, ПИКИ И ФОН
    # =========================================================================
    with tab_ftir1:
        c_sel1, c_sel2, c_sel3 = st.columns([2.5, 2, 2.5])
        with c_sel1:
            active_name = st.selectbox(T[lang]["ftir_active_sample_label"], sample_names, index=0, key="ftir_active_spec_select")
        with c_sel2:
            overlay_all = st.checkbox(T[lang]["ftir_show_all"], value=False, key="ftir_overlay_all_chk")
        with c_sel3:
            deriv_choice = st.radio(
                T[lang]["ftir_deriv_label"],
                [T[lang]["ftir_deriv_0"], T[lang]["ftir_deriv_1"], T[lang]["ftir_deriv_2"]],
                horizontal=True,
                key="ftir_deriv_mode_radio"
            )
            deriv_order = 0 if deriv_choice == T[lang]["ftir_deriv_0"] else (1 if deriv_choice == T[lang]["ftir_deriv_1"] else 2)

        active_s = current_ftir_samples[active_name]
        # Вычитаем фон если выбран метод
        active_s, base_curve = ftir_core.subtract_baseline(active_s, method=base_method)
        active_abs = active_s.corrected_absorbance if base_method != "none" else active_s.absorbance

        # Настройки поиска пиков
        with st.expander(T[lang]["ftir_peaks_expander"], expanded=False):
            c_pk1, c_pk2, c_pk3 = st.columns([2, 2, 2.5])
            with c_pk1:
                pk_prom = st.slider(T[lang]["ftir_peaks_prominence"], min_value=0.005, max_value=0.20, value=0.02, step=0.005, key="ftir_peak_prom_slider")
            with c_pk2:
                pk_dist = st.slider(T[lang]["ftir_peaks_dist"], min_value=10.0, max_value=60.0, value=20.0, step=5.0, key="ftir_peak_dist_slider")
            with c_pk3:
                show_peak_markers = st.checkbox(T[lang]["ftir_peaks_show_markers"], value=True, key="ftir_show_peak_markers_chk")

            df_peaks = ftir_core.find_and_annotate_peaks(
                active_s.wavenumbers,
                active_abs,
                prominence=pk_prom,
                min_distance_cm1=pk_dist,
                lang=lang
            )

            if not df_peaks.empty:
                st.markdown(f"###### {T[lang]['ftir_peaks_tbl_title']}")
                st_df(
                    df_peaks.style.format({
                        "Wavenumber_cm1": "{:.1f}",
                        "Absorbance": "{:.4f}",
                        "Prominence": "{:.4f}",
                    })
                )
                st.download_button(
                    label=T[lang]["ftir_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_name}_FTIR_Peaks.csv",
                    mime="text/csv",
                    key="dl_ftir_peaks_csv",
                )
            else:
                st.info("Пики с заданным порогом выраженности не обнаружены. Попробуйте уменьшить порог Prominence." if lang == "ru" else "No peaks detected with the current prominence threshold. Try reducing the prominence value.")

        fig_ftir = go.Figure()

        if overlay_all:
            for s_name, s_item in current_ftir_samples.items():
                s_item, _ = ftir_core.subtract_baseline(s_item, method=base_method)
                y_disp = s_item.corrected_absorbance if base_method != "none" else s_item.absorbance
                if norm_method != "none":
                    y_disp = ftir_core.normalize_spectrum(y_disp, method=norm_method)
                if deriv_order > 0:
                    y_disp = ftir_core.smooth_savitzky_golay(y_disp, window_length=15, polyorder=2, deriv=deriv_order)

                is_lignin_sample = "lignin" in s_name.lower() or "шлам" in s_name.lower()
                c_line = "#D62728" if is_lignin_sample else "#0020C2"
                fig_ftir.add_trace(go.Scatter(
                    x=s_item.wavenumbers,
                    y=y_disp,
                    mode="lines",
                    name=s_name,
                    line=dict(width=1.5, color=c_line),
                    opacity=0.75
                ))
            fig_ftir.update_layout(
                title=f"Overlay of {len(current_ftir_samples)} FTIR Spectra" if lang == "en" else f"Наложение всех спектров FTIR ({len(current_ftir_samples)} проб)",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if deriv_order == 0 else f"d^{deriv_order}A"),
                height=540,
                plot_bgcolor="white"
            )
        else:
            y_active_disp = active_abs.copy()
            if norm_method != "none":
                y_active_disp = ftir_core.normalize_spectrum(y_active_disp, method=norm_method)
            if deriv_order > 0:
                y_active_disp = ftir_core.smooth_savitzky_golay(y_active_disp, window_length=15, polyorder=2, deriv=deriv_order)

            # Исходный зарегистрированный профиль
            if deriv_order == 0:
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=active_s.absorbance,
                    mode="lines",
                    name="Raw Absorbance" if lang == "en" else "Исходное поглощение (A)",
                    line=dict(color="#7F7F7F", width=1.2, dash="dot"),
                    opacity=0.65
                ))
                if base_method != "none":
                    fig_ftir.add_trace(go.Scatter(
                        x=active_s.wavenumbers,
                        y=base_curve,
                        mode="lines",
                        name=f"Baseline ({base_method.upper()})" if lang == "en" else f"Базовая линия ({base_method.upper()})",
                        line=dict(color="#FF7F0E", width=1.5, dash="dash")
                    ))
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=y_active_disp,
                    mode="lines",
                    name="Corrected Spectrum" if lang == "en" else "Скорректированный спектр",
                    line=dict(color="#0B2545", width=2.0)
                ))

                # Метки обнаруженных пиков
                if 'show_peak_markers' in locals() and show_peak_markers and not df_peaks.empty:
                    fig_ftir.add_trace(go.Scatter(
                        x=df_peaks["Wavenumber_cm1"],
                        y=df_peaks["Absorbance"],
                        mode="markers+text",
                        marker=dict(size=8, color="#D62728", symbol="diamond"),
                        text=[f"{w:.0f}" for w in df_peaks["Wavenumber_cm1"]],
                        textposition="top center",
                        hovertext=[
                            f"<b>{row['Wavenumber_cm1']:.1f} cm⁻¹</b><br>{row['Functional_Group']}<br>{row['Molecular_Assignment']}<br><i>{row['Compound_Class']}</i>"
                            for _, row in df_peaks.iterrows()
                        ],
                        hoverinfo="text",
                        name="Identified Bands" if lang == "en" else "Найденные полосы"
                    ))
            else:
                fig_ftir.add_trace(go.Scatter(
                    x=active_s.wavenumbers,
                    y=y_active_disp,
                    mode="lines",
                    name=f"Derivative d^{deriv_order}A",
                    line=dict(color="#2CA02C", width=1.8)
                ))

            # Разметка ключевых функциональных диапазонов
            key_bands = [
                (3400, "OH/NH (3400)"),
                (2920, "Aliph CH2 (2920)"),
                (1720, "C=O COOH (1720)"),
                (1600, "C=C Arom (1600)"),
                (1510, "Lignin G-ring (1510)"),
                (1030, "C-O Poly (1030)")
            ]
            for wn_val, band_lbl in key_bands:
                fig_ftir.add_vline(x=wn_val, line_width=1, line_dash="dot", line_color="#CED4DA")
                fig_ftir.add_annotation(x=wn_val, y=float(np.max(y_active_disp)) * 0.95, text=band_lbl, showarrow=False, font=dict(size=9, color="#495057"))

            fig_ftir.update_layout(
                title=f"{active_name} — FTIR Profile" if lang == "en" else f"{active_name} — Спектральный профиль FTIR",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if deriv_order == 0 else f"d^{deriv_order}A"),
                height=520,
                plot_bgcolor="white"
            )

        st_plotly(fig_ftir)

        # Экспресс-дескрипторы активного образца
        ind_single = ftir_core.calculate_ftir_indices(active_s.wavenumbers, active_abs, sample_id=active_name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Lignin Index (1510/1600)", f"{ind_single['FTIR_Lignin_Idx']:.3f}")
        m2.metric("Aromaticity (1600/2920)", f"{ind_single['FTIR_Aromaticity_Idx']:.3f}")
        m3.metric("Oxidation (1720/1600)", f"{ind_single['FTIR_Oxidation_Idx']:.3f}")
        m4.metric("G/S Ratio (1270/1330)", f"{ind_single['FTIR_GS_Ratio']:.3f}")
        m5.metric("Polysaccharide Idx", f"{ind_single['FTIR_Polysaccharide_Idx']:.3f}")

    # =========================================================================
    # ВКЛАДКА 2: ДЕКОНВОЛЮЦИЯ ПОЛОС (1500–1800 см⁻¹)
    # =========================================================================
    with tab_ftir2:
        st.markdown("##### " + ("Математическое разделение карбонильных и ароматических полос (Levenberg-Marquardt)" if lang == "ru" else "Mathematical Deconvolution of Carbonyl and Aromatic Bands"))
        col_d1, col_d2, col_d3 = st.columns([2.5, 2, 2])
        with col_d1:
            deconv_target_name = st.selectbox(T[lang]["ftir_deconv_sample"], sample_names, index=0, key="ftir_deconv_sample_select")
        with col_d2:
            n_gaussians = st.slider(T[lang]["ftir_deconv_n_peaks"], min_value=3, max_value=5, value=5, step=1, key="ftir_n_gauss_slider")
        with col_d3:
            deconv_run_btn = st.button(T[lang]["ftir_deconv_btn"], type="primary", key="btn_run_ftir_deconv")

        deconv_sample = current_ftir_samples[deconv_target_name]
        deconv_sample, _ = ftir_core.subtract_baseline(deconv_sample, method=base_method)
        y_deconv_src = deconv_sample.corrected_absorbance if base_method != "none" else deconv_sample.absorbance

        try:
            deconv_res = ftir_core.deconvolve_carbonyl_region(
                deconv_sample.wavenumbers,
                y_deconv_src,
                wn_min=1500.0,
                wn_max=1800.0,
                n_peaks=n_gaussians
            )

            c_q1, c_q2 = st.columns(2)
            c_q1.metric(T[lang]["ftir_deconv_r2"], f"{deconv_res['r2']:.4f}")
            c_q2.metric(T[lang]["ftir_deconv_rmse"], f"{deconv_res['rmse']:.4f}")

            # График деконволюции
            fig_dec = go.Figure()

            # Экспериментальный спектр
            fig_dec.add_trace(go.Scatter(
                x=deconv_res["wavenumbers"],
                y=deconv_res["observed"],
                mode="markers+lines",
                marker=dict(size=3, color="#0B2545"),
                line=dict(color="#0B2545", width=1.2),
                name="Observed Profile" if lang == "en" else "Экспериментальный спектр"
            ))

            # Суммарная модельная кривая
            fig_dec.add_trace(go.Scatter(
                x=deconv_res["wavenumbers"],
                y=deconv_res["fitted"],
                mode="lines",
                line=dict(color="#D62728", width=2.5, dash="dash"),
                name="Fitted Envelope" if lang == "en" else "Суммарный фиттинг"
            ))

            # Индивидуальные полосы
            band_palette = ["#17BECF", "#9467BD", "#E377C2", "#FF7F0E", "#2CA02C"]
            peaks_df = deconv_res["peaks_df"]
            for idx, c_curve in enumerate(deconv_res["individual_curves"]):
                p_name = peaks_df.iloc[idx]["Assignment"] if idx < len(peaks_df) else f"Peak {idx+1}"
                p_col = band_palette[idx % len(band_palette)]
                fig_dec.add_trace(go.Scatter(
                    x=deconv_res["wavenumbers"],
                    y=c_curve,
                    mode="lines",
                    fill="tozeroy",
                    name=p_name,
                    line=dict(color=p_col, width=1.5),
                    opacity=0.65
                ))

            fig_dec.update_layout(
                title=f"Band Deconvolution (1500–1800 cm⁻¹) — {deconv_target_name}",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Absorbance" if lang == "en" else "Оптическая плотность A"),
                height=520,
                plot_bgcolor="white"
            )
            st_plotly(fig_dec)

            # Таблица параметров подогнанных полос
            st.markdown(f"###### {T[lang]['ftir_deconv_tbl_title']}")
            st_df(
                peaks_df.style.format({
                    "Center_cm1": "{:.1f}", "Amplitude": "{:.4f}",
                    "Sigma_cm1": "{:.2f}", "FWHM_cm1": "{:.2f}",
                    "Area": "{:.4f}", "Relative_Area_pct": "{:.2f}%"
                })
            )

        except Exception as e_dec:
            st.warning(f"Ошибка при деконволюции: {e_dec}")

    # =========================================================================
    # ВКЛАДКА 3: ИДЕНТИФИКАЦИЯ ПО БИБЛИОТЕКЕ ЭТАЛОНОВ (HQI)
    # =========================================================================
    with tab_ftir3:
        st.markdown("##### " + T[lang]["ftir_lib_hdr"])
        c_lb1, c_lb2, c_lb3 = st.columns([2.5, 1.5, 2])
        with c_lb1:
            lib_target_name = st.selectbox(T[lang]["ftir_lib_sample_select"], sample_names, index=0, key="ftir_lib_sample_select")
        with c_lb2:
            lib_top_n = st.slider(T[lang]["ftir_lib_top_n"], min_value=3, max_value=8, value=5, step=1, key="ftir_lib_top_n_slider")
        with c_lb3:
            st.markdown("<br>", unsafe_allow_html=True)
            lib_run_btn = st.button(T[lang]["ftir_lib_btn"], type="primary", key="btn_run_ftir_lib_match")

        target_s = current_ftir_samples[lib_target_name]
        target_s_corr, _ = ftir_core.subtract_baseline(target_s, method=base_method)
        y_target_abs = target_s_corr.corrected_absorbance if base_method != "none" else target_s.absorbance

        match_df = ftir_core.match_spectrum_against_library(target_s.wavenumbers, y_target_abs, top_n=lib_top_n, lang=lang)

        if not match_df.empty:
            top_match = match_df.iloc[0]
            top_std_id = top_match["Standard_ID"]
            top_hqi = top_match["HQI_pct"]
            top_name = top_match["Standard_Name"]

            # Статусная плашка
            if top_hqi >= 85.0:
                st.success(
                    f"🏆 **Топ-1 кандидат:** {top_name} (HQI = **{top_hqi:.1f}%**, Cosine = **{top_match['Cosine_Similarity']:.3f}**, Pearson r = **{top_match['Pearson_r']:.3f}**)"
                    if lang == "ru" else
                    f"🏆 **Top-1 Match:** {top_name} (HQI = **{top_hqi:.1f}%**, Cosine = **{top_match['Cosine_Similarity']:.3f}**, Pearson r = **{top_match['Pearson_r']:.3f}**)"
                )
            else:
                st.warning(
                    f"⚠️ **Наиболее близкий эталон:** {top_name} (HQI = **{top_hqi:.1f}%**). Возможно присутствие сложной смешанной матрицы."
                    if lang == "ru" else
                    f"⚠️ **Closest Reference Standard:** {top_name} (HQI = **{top_hqi:.1f}%**). Complex matrix mixture suspected."
                )

            # Оверлей-график: образец против топ-1 стандарта библиотеки
            ref_wn, ref_abs, ref_meta = ftir_core.get_reference_spectrum(top_std_id, wn_grid=target_s.wavenumbers)

            s_disp_norm = ftir_core.normalize_spectrum(y_target_abs, method="minmax")
            r_disp_norm = ftir_core.normalize_spectrum(ref_abs, method="minmax")

            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(
                x=target_s.wavenumbers,
                y=s_disp_norm,
                mode="lines",
                name=f"{lib_target_name} ({T[lang]['ftir_lib_sample_trace']})",
                line=dict(color="#0B2545", width=2.2)
            ))
            fig_overlay.add_trace(go.Scatter(
                x=ref_wn,
                y=r_disp_norm,
                mode="lines",
                name=f"{top_name} ({T[lang]['ftir_lib_ref_trace']})",
                line=dict(color="#FF7F0E", width=2.0, dash="dash")
            ))

            fig_overlay.update_layout(
                title=f"{T[lang]['ftir_lib_overlay_title']} — {lib_target_name} vs {top_name}",
                xaxis=dict(title="Wavenumber (cm⁻¹)" if lang == "en" else "Волновое число (см⁻¹)", autorange="reversed"),
                yaxis=dict(title="Normalized Absorbance (Min-Max)" if lang == "en" else "Нормированное поглощение (Min-Max)"),
                height=480,
                plot_bgcolor="white"
            )
            st_plotly(fig_overlay)

            # Таблица совпадений
            st.markdown(f"###### {T[lang]['ftir_lib_tbl_title']}")
            disp_match_df = match_df[["Rank", "Standard_Name", "Compound_Class", "HQI_pct", "Cosine_Similarity", "Pearson_r", "Key_Markers", "Description"]]
            st_df(
                disp_match_df.style.format({
                    "HQI_pct": "{:.2f}%",
                    "Cosine_Similarity": "{:.4f}",
                    "Pearson_r": "{:.4f}"
                })
            )

            st.download_button(
                label="📥 " + ("Скачать результаты идентификации (CSV)" if lang == "ru" else "Download Library Hits (CSV)"),
                data=match_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{lib_target_name}_FTIR_Library_Hits.csv",
                mime="text/csv",
                key="dl_ftir_lib_hits_csv",
            )

    # =========================================================================
    # ВКЛАДКА 4: ХЕМОМЕТРИЧЕСКИЕ ИНДЕКСЫ И ЭКСПОРТ
    # =========================================================================
    with tab_ftir4:
        st.markdown("##### " + ("Сводная матрица дескрипторов колебательной спектроскопии" if lang == "ru" else "Summary Matrix of Vibrational Spectroscopy Descriptors"))

        df_ftir_desc = ftir_core.batch_process_ftir_samples(
            list(current_ftir_samples.values()),
            baseline_method=base_method,
            norm_method="none"
        )

        st_df(
            df_ftir_desc.style.format({
                "FTIR_Aromaticity_Idx": "{:.3f}", "FTIR_Oxidation_Idx": "{:.3f}",
                "FTIR_Branching_Idx": "{:.3f}", "FTIR_Lignin_Idx": "{:.3f}",
                "FTIR_GS_Ratio": "{:.3f}", "FTIR_Polysaccharide_Idx": "{:.3f}",
                "FTIR_Phenolic_Idx": "{:.3f}",
                "FTIR_Area_OH": "{:.1f}", "FTIR_Area_Aliphatic": "{:.1f}",
                "FTIR_Area_Carbonyl_Aromatic": "{:.1f}", "FTIR_Area_Polysaccharide": "{:.1f}",
                "FTIR_Area_Ratio_Carb_Aliph": "{:.3f}",
            }, na_rep="—")
        )

        # Сравнительная диаграмма маркеров лигнина и ароматичности
        fig_bar = px.bar(
            df_ftir_desc,
            x="Sample_ID",
            y=["FTIR_Lignin_Idx", "FTIR_Aromaticity_Idx", "FTIR_Oxidation_Idx"],
            barmode="group",
            title="Comparison of Key Structural Indices Across Samples" if lang == "en" else "Сравнение ключевых структурных индексов по пробам"
        )
        fig_bar.update_layout(height=420, xaxis_tickangle=-25)
        st_plotly(fig_bar)

        c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
        with c_d1:
            st.download_button(
                label=T[lang]["ftir_dl_csv"],
                data=df_ftir_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_FTIR_Descriptors.csv",
                mime="text/csv",
                key="dl_ftir_descriptors_csv",
            )
        with c_d2:
            excel_ftir_buf = io.BytesIO()
            with pd.ExcelWriter(excel_ftir_buf, engine="openpyxl") as writer:
                df_ftir_desc.to_excel(writer, index=False, sheet_name="FTIR_Descriptors")
            st.download_button(
                label=T[lang]["ftir_dl_excel"],
                data=excel_ftir_buf.getvalue(),
                file_name="ChemoSuite_FTIR_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_ftir_descriptors_excel",
            )
        with c_d3:
            if st.button(T[lang]["ftir_sync_ml_btn"], type="primary", key="btn_sync_ftir_ml_descriptors"):
                st.session_state["ftir_ml_descriptors"] = df_ftir_desc
                st.toast(T[lang]["ftir_sync_success"].format(n=len(df_ftir_desc)), icon="✅")
                st.rerun()
