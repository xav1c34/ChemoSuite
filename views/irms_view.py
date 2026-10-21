"""
views/irms_view.py — Isotope Ratio Mass Spectrometry Studio (IRMS).
Analyzes delta13C, delta15N, delta34S, delta2H, delta18O stable isotopes
for geochemical provenance, Kraft pulp sludge tracing, and mixing models.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly

try:
    from modules import irms_core
except ImportError:
    irms_core = None


def render_irms_studio(T: dict, lang: str):
    """Render Module 9: Isotope Ratio Mass Spectrometry Studio (IRMS)."""
    if irms_core is None:
        st.error("IRMS core module not found.")
        st.stop()

    st.title(T[lang].get("irms_title", "🎯 Изотопная масс-спектрометрия (IRMS)"))
    st.caption(T[lang].get("irms_subtitle", "Прецизионный анализ стабильных изотопов (δ13C, δ15N, δ34S, δ2H, δ18O), модели смешения источников и трассировка шлам-лигнина"))

    with st.sidebar:
        st.header(T[lang].get("irms_sidebar_hdr", "📁 Данные IRMS"))
        data_src = st.radio(
            T[lang].get("irms_src_label", "Источник данных:"),
            [
                T[lang].get("irms_src_upload", "Загрузить таблицу (CSV/TSV)"),
                T[lang].get("irms_src_synth", "Байкальский бенчмарк (БЦБК / Пелагиаль)")
            ],
            index=1,
            key="irms_data_src_radio",
        )

        if st.session_state.get("_prev_irms_data_src") != data_src:
            st.session_state["_prev_irms_data_src"] = data_src
            st.session_state["irms_samples"] = {}

        if data_src == T[lang].get("irms_src_upload", "Загрузить таблицу (CSV/TSV)"):
            up_file = st.file_uploader(
                T[lang].get("irms_up_label", "Загрузить таблицу изотопов (.csv, .tsv, .txt):"),
                type=["csv", "tsv", "txt"],
                key="irms_uploader_file",
            )
            if up_file is not None:
                samples = irms_core.parse_irms_table(up_file.getvalue())
                st.session_state["irms_samples"] = {s.sample_id: s for s in samples}
        else:
            if not st.session_state.get("irms_samples"):
                bench_df = irms_core.generate_synthetic_irms_dataset()
                samples = irms_core.parse_irms_table(bench_df)
                st.session_state["irms_samples"] = {s.sample_id: s for s in samples}

    curr_samples = st.session_state.get("irms_samples", {})
    if not curr_samples:
        st.info("Нет загруженных изотопных данных. Загрузите CSV/TSV или выберите синтетический бенчмарк в боковой панели.")
        st.stop()

    sample_ids = list(curr_samples.keys())
    col_sel1, col_sel2 = st.columns([2.5, 3.5])
    with col_sel1:
        active_id = st.selectbox(
            T[lang].get("irms_active_sample", "Активный образец:"),
            sample_ids,
            index=0,
            key="irms_active_sample_select",
        )
    active_sample = curr_samples[active_id]

    with col_sel2:
        chips = [
            f'<span class="sample-chip"><b>δ¹³C</b>: {active_sample.d13C:.2f} ‰</span>',
            f'<span class="sample-chip"><b>δ¹⁵N</b>: {active_sample.d15N:.2f} ‰</span>',
        ]
        if active_sample.d34S is not None:
            color = "orange" if active_sample.d34S >= 4.0 else "green"
            chips.append(f'<span class="sample-chip {color}"><b>δ³⁴S</b>: {active_sample.d34S:.2f} ‰</span>')
        if active_sample.d2H is not None and active_sample.d18O is not None:
            chips.append(f'<span class="sample-chip"><b>d-excess</b>: {active_sample.deuterium_excess:+.1f} ‰</span>')

        p_color = "red" if active_sample.kraft_lignin_prob > 0.6 else ("orange" if active_sample.kraft_lignin_prob > 0.3 else "green")
        chips.append(f'<span class="sample-chip {p_color}"><b>Kraft Prob</b>: {active_sample.kraft_lignin_prob*100:.0f}%</span>')

        st.markdown(
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 26px;">{"".join(chips)}</div>',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Изотопные диаграммы" if lang == "ru" else "🗺️ Isotope Cross-Plots",
        "⚖️ Модели смешения" if lang == "ru" else "⚖️ Mixing Models",
        "🧬 Мультимодальная связь" if lang == "ru" else "🧬 Cross-Modal Links",
        "📊 ML Таблица & Экспорт" if lang == "ru" else "📊 ML Descriptors & Export",
    ])

    # TAB 1: Изотопные диаграммы
    with tab1:
        st.subheader(f"{'Изотопный профиль образца' if lang == 'ru' else 'Isotope Profile'} — {active_id}")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("δ¹³C (VPDB)", f"{active_sample.d13C:.2f} ‰")
        kpi2.metric("δ¹⁵N (AIR)", f"{active_sample.d15N:.2f} ‰")
        kpi3.metric("δ³⁴S (VCDT)", f"{active_sample.d34S:.2f} ‰" if active_sample.d34S is not None else "N/A")
        kpi4.metric("δ²H (VSMOW)", f"{active_sample.d2H:.1f} ‰" if active_sample.d2H is not None else "N/A")
        kpi5.metric("δ¹⁸O (VSMOW)", f"{active_sample.d18O:.2f} ‰" if active_sample.d18O is not None else "N/A")

        st.markdown("---")
        c_plot1, c_plot2 = st.columns(2)

        # Build overview DataFrame
        df_all = []
        for s in curr_samples.values():
            df_all.append({
                "Sample_ID": s.sample_id,
                "d13C": s.d13C,
                "d15N": s.d15N,
                "d34S": s.d34S,
                "d2H": s.d2H,
                "d18O": s.d18O,
                "d_excess": s.deuterium_excess,
                "Kraft_Prob": s.kraft_lignin_prob,
                "Sample_Type": s.sample_type,
            })
        plot_df = pd.DataFrame(df_all)

        with c_plot1:
            st.write("#### " + ("Диаграмма δ¹³C vs δ¹⁵N (Генезис углерода и азота)" if lang == "ru" else "δ¹³C vs δ¹⁵N Biplot"))
            fig_cn = px.scatter(
                plot_df,
                x="d13C",
                y="d15N",
                color="Sample_Type",
                hover_name="Sample_ID",
                size="Kraft_Prob",
                size_max=14,
                template="plotly_white",
                labels={"d13C": "δ¹³C (‰ VPDB)", "d15N": "δ¹⁵N (‰ AIR)"},
            )
            # Add endmember reference boxes
            for em_name, em_info in irms_core.ENDMEMBERS.items():
                fig_cn.add_shape(
                    type="rect",
                    x0=em_info["d13C"][0], x1=em_info["d13C"][1],
                    y0=em_info["d15N"][0], y1=em_info["d15N"][1],
                    line=dict(color=em_info["color"], width=1, dash="dot"),
                    fillcolor=em_info["color"],
                    opacity=0.12,
                )
                fig_cn.add_annotation(
                    x=(em_info["d13C"][0] + em_info["d13C"][1]) / 2,
                    y=em_info["d15N"][1],
                    text=em_name.replace("_", " "),
                    showarrow=False,
                    font=dict(size=9, color=em_info["color"]),
                )
            fig_cn.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=420)
            st_plotly(fig_cn)

        with c_plot2:
            st.write("#### " + ("Диаграмма δ¹³C vs δ³⁴S (Трассировка сульфатного шлам-лигнина)" if lang == "ru" else "δ¹³C vs δ³⁴S (Kraft Lignin Tracing)"))
            if plot_df["d34S"].notna().sum() > 0:
                fig_cs = px.scatter(
                    plot_df.dropna(subset=["d34S"]),
                    x="d13C",
                    y="d34S",
                    color="Sample_Type",
                    hover_name="Sample_ID",
                    size="Kraft_Prob",
                    size_max=14,
                    template="plotly_white",
                    labels={"d13C": "δ¹³C (‰ VPDB)", "d34S": "δ³⁴S (‰ VCDT)"},
                )
                # Benchmark line for heavy Kraft sulfur (> 4.5 ‰)
                fig_cs.add_hline(y=4.5, line_dash="dash", line_color="#e11d48", annotation_text="Порог шлам-лигнина БЦБК (δ³⁴S > +4.5‰)")
                fig_cs.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=420)
                st_plotly(fig_cs)
            else:
                st.info("В выборке отсутствуют измерения δ³⁴S.")

    # TAB 2: Модели смешения (2-source mixing model)
    with tab2:
        st.subheader("⚖️ " + ("Линейная модель изотопного смешения (2-Source Mixing)" if lang == "ru" else "Two-Source Linear Mixing Model"))
        st.write(
            "Расчет доли техногенного шлам-лигнина БЦБК ($f_{\\text{Kraft}}$) по изотопной сигнатуре серы $\\delta^{34}S$ относительно ультраолиготрофного байкальского фона."
            if lang == "ru" else
            "Calculates fraction of industrial Kraft pulp sludge lignin based on heavy sulfur isotope signature."
        )

        c_em1, c_em2 = st.columns(2)
        with c_em1:
            em_kraft_s = st.slider("Эндмембер шлам-лигнина БЦБК δ³⁴S (‰):", 4.0, 10.0, 6.8, 0.1)
        with c_em2:
            em_baikal_s = st.slider("Эндмембер природного фона Байкала δ³⁴S (‰):", -1.0, 2.5, 0.8, 0.1)

        mixing_results = []
        for s in curr_samples.values():
            if s.d34S is not None:
                f_kraft = irms_core.calculate_two_source_mixing(s.d34S, em_kraft_s, em_baikal_s)
                mixing_results.append({
                    "Sample_ID": s.sample_id,
                    "Sample_Type": s.sample_type,
                    "d34S": s.d34S,
                    "Kraft_Lignin_Pct": round(f_kraft * 100.0, 1),
                    "Natural_Baikal_Pct": round((1.0 - f_kraft) * 100.0, 1),
                    "Status": "⚠️ Техногенный шлам" if f_kraft > 0.5 else ("🟡 Смешанный сток" if f_kraft > 0.15 else "🟢 Природный фон"),
                })

        if mixing_results:
            mix_df = pd.DataFrame(mixing_results)
            fig_bar = px.bar(
                mix_df,
                x="Sample_ID",
                y="Kraft_Lignin_Pct",
                color="Status",
                color_discrete_map={
                    "⚠️ Техногенный шлам": "#e11d48",
                    "🟡 Смешанный сток": "#d97706",
                    "🟢 Природный фон": "#0ea5e9",
                },
                template="plotly_white",
                title="Доля шлам-лигнина БЦБК в исследованных пробах (%):" if lang == "ru" else "Kraft Sludge Lignin Contribution (%):",
                labels={"Kraft_Lignin_Pct": "Шлам-лигнин (%)", "Sample_ID": "Образец"},
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
            st_plotly(fig_bar)
            st_df(mix_df)
        else:
            st.info("Нет данных δ³⁴S для расчета модели смешения.")

    # TAB 3: Мультимодальная связь (CHNS-O & FT-ICR)
    with tab3:
        st.subheader("🧬 " + ("Кросс-модальная валидация (IRMS ↔ CHNS-O ↔ FT-ICR MS)" if lang == "ru" else "Cross-Modal Correlation"))
        st.write(
            "Сопоставление стабильных изотопов с валовым элементным составом (CHNS-O) и серосодержащими формулами FT-ICR ($CHOS$)."
            if lang == "ru" else
            "Cross-validation between stable isotope ratios, CHNS-O elemental analysis, and FT-ICR MS CHOS pools."
        )

        has_elem = "elemental_samples" in st.session_state and st.session_state["elemental_samples"]
        has_ms = "spectra_db" in st.session_state and st.session_state["spectra_db"]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Валовой CHNS-O:** {'🟢 Доступен' if has_elem else '⚪ Не загружен'}")
            if has_elem:
                elem_dict = st.session_state["elemental_samples"]
                st.caption(f"Обнаружено образцов CHNS-O: {len(elem_dict)}")
        with col_m2:
            st.markdown(f"**FT-ICR MS Спектры:** {'🟢 Доступны' if has_ms else '⚪ Не загружены'}")
            if has_ms:
                ms_dict = st.session_state["spectra_db"]
                st.caption(f"Обнаружено спектров: {len(ms_dict)}")

        # Synthetic cross-correlation preview
        st.markdown("---")
        st.write("#### " + ("Теоретическая корреляция δ³⁴S с содержанием серы S (% масс.)" if lang == "ru" else "δ³⁴S vs Total Mass % S"))
        st.info(
            "Для гидролизного и сульфатного лигнина БЦБК характерно одновременное обогащение валовой серой (> 1.2% S по CHNS-O) и утяжеление изотопного состава δ³⁴S (+5...+8 ‰ VCDT), что служит надежным мультимодальным маркером техногенеза."
        )

    # TAB 4: ML Дескрипторы & Экспорт
    with tab4:
        st.subheader("📊 " + ("Хемометрическая матрица дескрипторов IRMS" if lang == "ru" else "IRMS Chemometric Descriptors"))
        desc_df = irms_core.calculate_irms_descriptors(list(curr_samples.values()))
        st_df(desc_df)

        c_sync, c_exp_csv, c_exp_xlsx = st.columns([2.5, 1.5, 1.5])
        with c_sync:
            if st.button("🧬 " + ("Отправить в ChemoSuite ML (Data Fusion)" if lang == "ru" else "Push to ChemoSuite ML"), type="primary", use_container_width=True):
                st.session_state["irms_ml_descriptors"] = desc_df.reset_index()
                st.success("Дескрипторы IRMS успешно зарегистрированы в блоке слияния данных!")
        with c_exp_csv:
            csv_data = desc_df.to_csv().encode("utf-8")
            st.download_button("📥 Скачать CSV", csv_data, "irms_descriptors.csv", "text/csv", use_container_width=True)
        with c_exp_xlsx:
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                desc_df.to_excel(writer, sheet_name="IRMS_Data")
            st.download_button("📊 Скачать Excel", out_buf.getvalue(), "irms_descriptors.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
