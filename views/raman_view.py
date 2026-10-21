"""
views/raman_view.py — Raman Spectroscopy Studio for Carbonaceous NOM & Lignin.
Analyzes D and G bands (800-2000 cm^-1), I_D/I_G, A_D/A_G deconvolution,
Tuinstra-Koenig crystallite size L_a (nm), and carbonaceous graphitization index.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly

try:
    from modules import raman_core
except ImportError:
    raman_core = None


def render_raman_studio(T: dict, lang: str):
    """Render Module 11: Raman Spectroscopy Studio."""
    if raman_core is None:
        st.error("Raman core module not found.")
        st.stop()

    st.title(T[lang].get("raman_title", "💎 Рамановская спектроскопия (Raman: D/G пики)"))
    st.caption(T[lang].get("raman_subtitle", "Деконволюция углеродных D- и G-полос, отношение I_D/I_G, размер ароматических кластеров L_a и степень упорядоченности"))

    with st.sidebar:
        st.header(T[lang].get("raman_sidebar_hdr", "📁 Данные Raman"))
        data_src = st.radio(
            T[lang].get("raman_src_label", "Источник данных:"),
            [
                T[lang].get("raman_src_upload", "Загрузить спектр (CSV/TXT)"),
                T[lang].get("raman_src_synth", "Бенчмарк (Шлам-гидроуголь / Сажа / Гуматы)")
            ],
            index=1,
            key="raman_data_src_radio",
        )

        if st.session_state.get("_prev_raman_data_src") != data_src:
            st.session_state["_prev_raman_data_src"] = data_src
            st.session_state["raman_samples"] = {}

        if data_src == T[lang].get("raman_src_upload", "Загрузить спектр (CSV/TXT)"):
            up_files = st.file_uploader(
                T[lang].get("raman_up_label", "Загрузить спектры Raman (.csv, .txt):"),
                type=["csv", "txt", "dat"],
                accept_multiple_files=True,
                key="raman_uploader_file",
            )
            if up_files:
                for f in up_files:
                    s_id = str(f.name).rsplit(".", 1)[0]
                    spec = raman_core.parse_raman_spectrum(f.getvalue(), sample_id=s_id)
                    st.session_state.setdefault("raman_samples", {})[s_id] = spec
        else:
            if not st.session_state.get("raman_samples"):
                bench_dict = raman_core.generate_synthetic_raman_dataset()
                st.session_state["raman_samples"] = bench_dict

    curr_samples = st.session_state.get("raman_samples", {})
    if not curr_samples:
        st.info("Нет загруженных спектров комбинационного рассеяния (Raman). Загрузите файл или выберите бенчмарк в боковой панели.")
        st.stop()

    sample_ids = list(curr_samples.keys())
    col_sel1, col_sel2 = st.columns([2.5, 3.5])
    with col_sel1:
        active_id = st.selectbox(
            T[lang].get("raman_active_sample", "Активный спектр:"),
            sample_ids,
            index=0,
            key="raman_active_sample_select",
        )
    active_spec = curr_samples[active_id]

    with col_sel2:
        chips = [
            f'<span class="sample-chip"><b>I_D/I_G</b>: {active_spec.id_ig_ratio:.2f}</span>',
            f'<span class="sample-chip"><b>A_D/A_G</b>: {active_spec.ad_ag_ratio:.2f}</span>',
            f'<span class="sample-chip green"><b>L_a</b>: {active_spec.la_crystallite_nm:.1f} nm</span>',
            f'<span class="sample-chip orange"><b>G-FWHM</b>: {active_spec.fwhm_g:.1f} cm⁻¹</span>',
        ]
        ord_color = "green" if active_spec.carbon_order_index > 0.6 else ("orange" if active_spec.carbon_order_index > 0.35 else "blue")
        chips.append(f'<span class="sample-chip {ord_color}"><b>Carbon Order</b>: {active_spec.carbon_order_index:.2f}</span>')

        st.markdown(
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 26px;">{"".join(chips)}</div>',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Спектр & Деконволюция" if lang == "ru" else "📈 Spectrum & Deconvolution",
        "🗺️ Сравнение & Диаграмма зрелости" if lang == "ru" else "🗺️ Multi-Overlay & Maturity",
        "🔥 Мультимодальная связь" if lang == "ru" else "🔥 Cross-Modal Links",
        "📊 ML Дескрипторы & Экспорт" if lang == "ru" else "📊 ML Descriptors & Export",
    ])

    # TAB 1: Спектр & Деконволюция
    with tab1:
        st.subheader(f"{'Деконволюция полос D и G' if lang == 'ru' else 'D & G Band Deconvolution'} — {active_id}")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("I_D / I_G (Высота)", f"{active_spec.id_ig_ratio:.2f}")
        kpi2.metric("A_D / A_G (Площадь)", f"{active_spec.ad_ag_ratio:.2f}")
        kpi3.metric("Размер кластеров L_a", f"{active_spec.la_crystallite_nm:.1f} нм")
        kpi4.metric("Позиция G-полосы", f"{active_spec.g_pos:.1f} см⁻¹")
        kpi5.metric("Позиция D-полосы", f"{active_spec.d_pos:.1f} см⁻¹")

        st.markdown("---")
        fig_fit = go.Figure()

        # Raw / normalized spectrum
        wn = active_spec.wavenumbers
        # Filter for carbon region 1000-1800
        mask = (wn >= 1000.0) & (wn <= 1800.0)
        wn_plot = wn[mask] if np.any(mask) else wn
        y_plot = active_spec.corrected_intensities if active_spec.corrected_intensities is not None else active_spec.intensities[mask]

        fig_fit.add_trace(go.Scatter(
            x=wn_plot,
            y=y_plot,
            mode="lines",
            name="Эксперимент (нормир.)",
            line=dict(color="#1e293b", width=2.5),
        ))

        # Add fitted components if available
        if active_spec.components:
            comp_colors = {
                "D1_band": ("#e11d48", "D1 полоса (~1350 см⁻¹: дефекты колец)"),
                "D3_amorphous": ("#d97706", "D3 полоса (~1500 см⁻¹: аморфный C)"),
                "G_band": ("#0ea5e9", "G полоса (~1585 см⁻¹: sp² решетка)"),
                "Cumulative_Fit": ("#8b5cf6", "Суммарная аппроксимация"),
            }
            for comp_key, (color, label) in comp_colors.items():
                if comp_key in active_spec.components:
                    is_cum = comp_key == "Cumulative_Fit"
                    fig_fit.add_trace(go.Scatter(
                        x=wn_plot,
                        y=active_spec.components[comp_key],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=2 if is_cum else 1.5, dash="dash" if not is_cum else "solid"),
                        opacity=0.9 if is_cum else 0.7,
                    ))

        fig_fit.update_layout(
            template="plotly_white",
            title=f"Рамановский профиль первого порядка углерода (1000–1800 см⁻¹) — {active_id}",
            xaxis_title="Волновое число (см⁻¹)",
            yaxis_title="Относительная интенсивность (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            height=460,
        )
        st_plotly(fig_fit)

    # TAB 2: Сравнение & Диаграмма зрелости углерода
    with tab2:
        st.subheader("🗺️ " + ("Сравнительный анализ спектров и диаграмма Туинстра-Кенига" if lang == "ru" else "Comparative Raman Profiles & Carbon Maturity"))

        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.write("#### " + ("Наложение всех спектров сессии:" if lang == "ru" else "Session Raman Spectra Overlay:"))
            fig_multi = go.Figure()
            for s_id, s in curr_samples.items():
                w = s.wavenumbers
                m = (w >= 1000.0) & (w <= 1800.0)
                y = s.corrected_intensities if s.corrected_intensities is not None else s.intensities[m]
                fig_multi.add_trace(go.Scatter(
                    x=w[m] if np.any(m) else w,
                    y=y,
                    mode="lines",
                    name=s_id,
                ))
            fig_multi.update_layout(
                template="plotly_white",
                xaxis_title="Волновое число (см⁻¹)",
                yaxis_title="Интенсивность (%)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=400,
            )
            st_plotly(fig_multi)

        with col_comp2:
            st.write("#### " + ("Диаграмма упорядоченности углерода (I_D/I_G vs FWHM_G):" if lang == "ru" else "Carbon Ordering Diagram:"))
            overview_rows = []
            for s in curr_samples.values():
                overview_rows.append({
                    "Sample_ID": s.sample_id,
                    "ID_IG": s.id_ig_ratio,
                    "AD_AG": s.ad_ag_ratio,
                    "La_nm": s.la_crystallite_nm,
                    "FWHM_G": s.fwhm_g,
                    "Carbon_Order": s.carbon_order_index,
                    "Sample_Type": s.sample_type,
                })
            df_raman_meta = pd.DataFrame(overview_rows)

            fig_order = px.scatter(
                df_raman_meta,
                x="FWHM_G",
                y="ID_IG",
                color="Sample_Type",
                size="Carbon_Order",
                size_max=16,
                hover_name="Sample_ID",
                template="plotly_white",
                labels={"FWHM_G": "Ширина G-полосы FWHM (см⁻¹)", "ID_IG": "Отношение I_D / I_G"},
            )
            # Reference region annotations
            fig_order.add_hrect(
                y0=0.85, y1=1.4,
                line_width=0, fillcolor="#e11d48", opacity=0.08,
                annotation_text="Высокоароматический чар / Сажа / Шлам БЦБК",
                annotation_position="top left",
            )
            fig_order.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)
            st_plotly(fig_order)

    # TAB 3: Мультимодальная связь (Thermal & FT-ICR)
    with tab3:
        st.subheader("🔥 " + ("Кросс-модальная валидация (Raman ↔ Py-GC-MS / TGA ↔ FT-ICR MS)" if lang == "ru" else "Cross-Modal Synergy"))
        st.write(
            "Параметр $I_D/I_G$ и размер кластеров $L_a$ прямо коррелируют с выходом коксового остатка (Char Yield при 600°C по данным ТГА) и долей конденсированных полиароматических структур ($AI_{mod} \\ge 0.67$) по данным FT-ICR масс-спектрометрии."
            if lang == "ru" else
            "Raman I_D/I_G ratio and La crystallite size directly correlate with TGA char yield and FT-ICR condensed aromatic formulas (AI_mod >= 0.67)."
        )
        st.info(
            "Для гидролизованного шлам-лигнина БЦБК наблюдается характерный кластер: $I_D/I_G \\approx 0.95 - 1.15$ и $L_a \\approx 3.8 - 4.8$ нм, что отличает его от рыхлых природных гуминовых кислот почв с низким выходом пиролитического кокса."
        )

    # TAB 4: ML Дескрипторы & Экспорт
    with tab4:
        st.subheader("📊 " + ("Хемометрическая матрица дескрипторов Raman" if lang == "ru" else "Raman Chemometric Descriptors"))
        desc_df = raman_core.calculate_raman_descriptors(list(curr_samples.values()))
        st_df(desc_df)

        c_sync, c_exp_csv, c_exp_xlsx = st.columns([2.5, 1.5, 1.5])
        with c_sync:
            if st.button("🧬 " + ("Отправить в ChemoSuite ML (Data Fusion)" if lang == "ru" else "Push to ChemoSuite ML"), type="primary", use_container_width=True):
                st.session_state["raman_ml_descriptors"] = desc_df.reset_index()
                st.success("Дескрипторы Raman успешно зарегистрированы в блоке слияния данных!")
        with c_exp_csv:
            csv_data = desc_df.to_csv().encode("utf-8")
            st.download_button("📥 Скачать CSV", csv_data, "raman_descriptors.csv", "text/csv", use_container_width=True)
        with c_exp_xlsx:
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                desc_df.to_excel(writer, sheet_name="Raman_Data")
            st.download_button("📊 Скачать Excel", out_buf.getvalue(), "raman_descriptors.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
