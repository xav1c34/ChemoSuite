"""
views/elemental_view.py — Elemental Microanalysis & Bulk Stoichiometry Studio (CHNS-O).
"""

import io
import pandas as pd
import plotly.express as px
import streamlit as st

from views.common import st_df, st_plotly
try:
    from modules import elemental_core
except ImportError:
    elemental_core = None


def render_elemental_studio(T: dict, lang: str):
    """Render Module 8: CHNS-O Elemental Microanalysis Studio."""
    if elemental_core is None:
        st.error(T[lang]["elem_err_missing"])
        st.stop()

    st.title(T[lang]["elem_title"])
    st.caption(T[lang]["elem_subtitle"])

    with st.sidebar:
        st.header(T[lang]["elem_sidebar_hdr"])
        elem_data_src = st.radio(
            T[lang]["elem_src_label"],
            [T[lang]["elem_src_upload"], T[lang]["elem_src_synth"]],
            index=0,
            key="elem_data_source_radio",
        )

        if st.session_state.get("_prev_elem_data_src") != elem_data_src:
            st.session_state["_prev_elem_data_src"] = elem_data_src
            st.session_state["elemental_samples"] = {}

        if elem_data_src == T[lang]["elem_src_upload"]:
            up_elem_file = st.file_uploader(
                T[lang]["elem_up_label"],
                type=["csv", "tsv", "txt"],
                key="elem_uploader_file",
            )
            basis_labels = {
                "auto": "Авто (колонка 'Basis' или исходная навеска)" if lang == "ru" else "Auto ('Basis' column or as received)",
                "as_received": "На исходную навеску (as received)" if lang == "ru" else "As received (wet, with ash)",
                "dry": "На сухое вещество (dry)" if lang == "ru" else "Dry basis",
                "daf": "Уже на сухое беззольное (daf)" if lang == "ru" else "Already dry ash-free (daf)",
            }
            basis_choice = st.selectbox(
                "Базис массовых долей C, H, N, S, O:" if lang == "ru" else "Basis of C, H, N, S, O mass fractions:",
                list(basis_labels.keys()),
                format_func=lambda k: basis_labels[k],
                key="elem_basis_select",
                help=(
                    "Пересчет на daf выполняется только согласно базису: as received — 100/(100−W−A), dry — 100/(100−A), daf — без пересчета."
                    if lang == "ru" else
                    "Conversion to daf follows the declared basis: as received — 100/(100−W−A), dry — 100/(100−A), daf — none."
                ),
            )
            if up_elem_file is not None:
                samples_list = elemental_core.parse_elemental_table(
                    up_elem_file.getvalue(), basis=None if basis_choice == "auto" else basis_choice
                )
                st.session_state["elemental_samples"] = {s.sample_id: s for s in samples_list}
        else:
            if not st.session_state.get("elemental_samples"):
                bench_df = elemental_core.generate_synthetic_elemental_dataset()
                samples_list = elemental_core.parse_elemental_table(bench_df)
                st.session_state["elemental_samples"] = {s.sample_id: s for s in samples_list}

    curr_elem_samples = st.session_state.get("elemental_samples", {})
    if not curr_elem_samples:
        st.info(T[lang]["elem_no_samples"])
        st.stop()

    sample_ids = list(curr_elem_samples.keys())
    col_sel1, col_sel2 = st.columns([2.5, 3.5])
    with col_sel1:
        active_s_id = st.selectbox(
            T[lang]["elem_active_sample"],
            sample_ids,
            index=0,
            key="elem_active_sample_select",
        )
    active_elem_sample = curr_elem_samples[active_s_id]

    with col_sel2:
        elem_tb_html = (
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 26px;">'
            f'<span class="sample-chip"><b>H/C</b>: {active_elem_sample.h_c:.3f}</span>'
            f'<span class="sample-chip"><b>O/C</b>: {active_elem_sample.o_c:.3f}</span>'
            f'<span class="sample-chip green"><b>NOSC</b>: {active_elem_sample.nosc:+.2f}</span>'
            f'<span class="sample-chip orange"><b>AI_bulk</b>: {active_elem_sample.aromaticity_index:.3f}</span>'
            f'</div>'
        )
        st.markdown(elem_tb_html, unsafe_allow_html=True)

    tab_el1, tab_el2, tab_el3, tab_el4 = st.tabs([
        T[lang]["elem_tab1_title"],
        T[lang]["elem_tab2_title"],
        T[lang]["elem_tab3_title"],
        T[lang]["elem_tab4_title"],
    ])

    # ТАБ 1: Валовая стехиометрия и индексы
    with tab_el1:
        st.subheader(f"{T[lang]['elem_tab1_title']} — {active_s_id}")
        c_kpi1, c_kpi2, c_kpi3, c_kpi4, c_kpi5 = st.columns(5)
        c_kpi1.metric(T[lang]["elem_kpi_hc"], f"{active_elem_sample.h_c:.3f}")
        c_kpi2.metric(T[lang]["elem_kpi_oc"], f"{active_elem_sample.o_c:.3f}")
        c_kpi3.metric(T[lang]["elem_kpi_nc"], f"{active_elem_sample.n_c:.4f}")
        c_kpi4.metric(T[lang]["elem_kpi_sc"], f"{active_elem_sample.s_c:.4f}")
        c_kpi5.metric(T[lang]["elem_kpi_nosc"], f"{active_elem_sample.nosc:+.3f}")

        c_kpi6, c_kpi7, c_kpi8, c_kpi9 = st.columns(4)
        c_kpi6.metric(
            "AI_bulk (Koch & Dittmar, C-норм.)" if lang == "ru" else "AI_bulk (Koch & Dittmar, C-normalised)",
            f"{active_elem_sample.aromaticity_index:.3f}",
        )
        c_kpi7.metric("AI_mod_bulk", f"{getattr(active_elem_sample, 'aromaticity_index_mod', float('nan')):.3f}")
        c_kpi8.metric(
            "ВТС Менделеев (МДж/кг)" if lang == "ru" else "HHV Mendeleev (MJ/kg)",
            f"{getattr(active_elem_sample, 'hhv_mendeleev_mj_kg', active_elem_sample.combustion_heat_mj_kg):.2f}",
        )
        c_kpi9.metric(
            "ВТС Дюлонг (МДж/кг)" if lang == "ru" else "HHV Dulong (MJ/kg)",
            f"{getattr(active_elem_sample, 'hhv_dulong_mj_kg', float('nan')):.2f}",
        )
        st.caption(
            "AI_bulk — приближение для валовой пробы: выражение Koch & Dittmar (2006, 2016) на C-нормированной формуле "
            "C1HhNnOoSs без молекулярного члена «1+». ВТС (daf): Менделеев Q = 0.339C + 1.256H − 0.1088(O − S); "
            "Дюлонг HHV = 0.3383C + 1.443(H − O/8) + 0.0942S."
            if lang == "ru" else
            "AI_bulk is a bulk approximation: Koch & Dittmar (2006, 2016) expression on the C-normalised formula "
            "C1HhNnOoSs without the molecular '1+' term. HHV (daf): Mendeleev Q = 0.339C + 1.256H − 0.1088(O − S); "
            "Dulong HHV = 0.3383C + 1.443(H − O/8) + 0.0942S."
        )
        basis_val = getattr(active_elem_sample, "basis", "as_received")
        daf_sum = getattr(active_elem_sample, "daf_sum_pct", float("nan"))
        st.caption(
            (f"Базис исходных данных: **{basis_val}**; сумма C+H+N+S+O на daf = {daf_sum:.1f} %."
             if lang == "ru" else
             f"Input basis: **{basis_val}**; C+H+N+S+O on daf = {daf_sum:.1f} %.")
        )
        if getattr(active_elem_sample, "basis_warning", None):
            st.warning(active_elem_sample.basis_warning)

        st.markdown("---")
        st.write("#### " + ("Весовой элементный состав образца (% масс.):" if lang == "ru" else "Mass Elemental Composition (% wt):"))
        elem_weights_df = pd.DataFrame([{
            "Element / Component": "Углерод (C)" if lang == "ru" else "Carbon (C)", "Mass %": active_elem_sample.c_pct,
        }, {
            "Element / Component": "Водород (H)" if lang == "ru" else "Hydrogen (H)", "Mass %": active_elem_sample.h_pct,
        }, {
            "Element / Component": "Азот (N)" if lang == "ru" else "Nitrogen (N)", "Mass %": active_elem_sample.n_pct,
        }, {
            "Element / Component": "Сера (S)" if lang == "ru" else "Sulfur (S)", "Mass %": active_elem_sample.s_pct,
        }, {
            "Element / Component": "Кислород (O, расчет)" if lang == "ru" else "Oxygen (O, by diff)", "Mass %": active_elem_sample.o_pct,
        }, {
            "Element / Component": "Зольность (Ash)" if lang == "ru" else "Ash Content", "Mass %": active_elem_sample.ash_pct,
        }, {
            "Element / Component": "Влажность (Moisture)" if lang == "ru" else "Moisture", "Mass %": active_elem_sample.moisture_pct,
        }])
        st_df(elem_weights_df)

        fig_bar = px.bar(
            elem_weights_df.head(5),
            x="Element / Component",
            y="Mass %",
            color="Element / Component",
            color_discrete_sequence=["#1b4965", "#62b6cb", "#bee9e8", "#ffb703", "#e63946"],
            title=("Баланс элементов CHNS-O" if lang == "ru" else "CHNS-O Elemental Mass Balance"),
        )
        fig_bar.update_layout(height=380, plot_bgcolor="white", showlegend=False)
        st_plotly(fig_bar)

    # ТАБ 2: Валовая диаграмма Ван Кревелена
    with tab_el2:
        st.subheader(T[lang]["elem_tab2_title"])
        all_elem_records = [s.to_dict() for s in curr_elem_samples.values()]
        df_all_elem = pd.DataFrame(all_elem_records)

        fig_vk_bulk = px.scatter(
            df_all_elem,
            x="O_C",
            y="H_C",
            color="Sample_Type",
            size="C_pct",
            hover_name="Sample_ID",
            hover_data=["NOSC", "AI_bulk", "S_C"],
            text="Sample_ID",
            title=("Валовая диаграмма Ван Кревелена образцов (H/C vs O/C)" if lang == "ru" else "Bulk Van Krevelen Diagram (H/C vs O/C)"),
            labels={"O_C": "O/C (Атомное отношение)", "H_C": "H/C (Атомное отношение)"},
        )
        fig_vk_bulk.add_shape(type="rect", x0=0.0, y0=1.5, x1=0.3, y1=2.2, fillcolor="#a8dadc", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.add_shape(type="rect", x0=0.2, y0=0.7, x1=0.6, y1=1.5, fillcolor="#e9c46a", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.add_shape(type="rect", x0=0.6, y0=0.8, x1=1.0, y1=1.8, fillcolor="#f4a261", opacity=0.15, layer="below", line_width=1)
        fig_vk_bulk.update_traces(textposition="top center")
        fig_vk_bulk.update_layout(height=560, plot_bgcolor="white")
        st_plotly(fig_vk_bulk)

    # ТАБ 3: Кросс-валидация с FT-ICR MS
    with tab_el3:
        st.subheader(f"{T[lang]['elem_tab3_title']} — {active_s_id}")
        spectra_db = st.session_state.get("spectra_db", {})

        matched_spec_name = None
        for s_name in spectra_db:
            if active_s_id.lower() in s_name.lower() or s_name.lower() in active_s_id.lower():
                matched_spec_name = s_name
                break
        if matched_spec_name is None and spectra_db:
            matched_spec_name = list(spectra_db.keys())[0]

        spec_options = list(spectra_db.keys()) if spectra_db else []
        if not spec_options:
            st.info("В памяти сессии нет спектров FT-ICR MS с приписанными формулами. Загрузите спектр в Модуле 1 для кросс-валидации." if lang == "ru" else "No FT-ICR MS spectra available in session. Upload a spectrum in Module 1 to perform cross-validation.")
        else:
            def_idx = spec_options.index(matched_spec_name) if matched_spec_name in spec_options else 0
            sel_spec = st.selectbox(
                "Сопоставляемый спектр FT-ICR MS:" if lang == "ru" else "Matching FT-ICR MS Spectrum:",
                spec_options,
                index=def_idx,
                key="elem_crossval_spec_select",
            )
            spec_obj = spectra_db[sel_spec]
            as_df = spec_obj.get("assigned_df")

            if as_df is None or as_df.empty:
                st.warning("Для выбранного спектра еще не выполнено приписывание брутто-формул. Перейдите в Модуль 1 (Этап 2) и нажмите 'Приписать формулы'." if lang == "ru" else "No formulas assigned for selected spectrum. Please run formula assignment in Module 1 first.")
            else:
                cval_res = elemental_core.cross_validate_fticr_elemental(as_df, active_elem_sample)
                st.success(f"✓ **Concordance Score (Сходимость методов): {cval_res['concordance_score']}%** — {cval_res['interpretation']}")

                c_cv1, c_cv2 = st.columns(2)
                with c_cv1:
                    st.write("##### " + ("Элементный микроанализ (CHNS-O bulk):" if lang == "ru" else "Elemental Microanalysis (CHNS-O bulk):"))
                    st_df(pd.DataFrame([cval_res["elemental_bulk"]]))
                with c_cv2:
                    st.write("##### " + ("Средневзвешенное по FT-ICR MS (ESI-):" if lang == "ru" else "Weighted FT-ICR MS Average (ESI-):"))
                    st_df(pd.DataFrame([cval_res["ms_weighted"]]))

                st.write("##### " + ("Невязки и расхождения (MS - CHNS):" if lang == "ru" else "Residuals (MS - CHNS):"))
                st_df(pd.DataFrame([cval_res["differences"]]))

    # ТАБ 4: Экспорт дескрипторов и синхронизация ML
    with tab_el4:
        st.subheader(T[lang]["elem_tab4_title"])
        all_elem_records = [s.to_dict() for s in curr_elem_samples.values()]
        df_elem_desc = pd.DataFrame(all_elem_records)
        st_df(df_elem_desc)
        st.caption(
            "Q_combustion_MJ_kg — высшая теплота сгорания по формуле Менделеева, HHV_Dulong_MJ_kg — по формуле Дюлонга (обе на daf). "
            "QC_Warning — предупреждение о сумме C+H+N+S+O > 100.5 % после пересчета на daf."
            if lang == "ru" else
            "Q_combustion_MJ_kg — higher heating value by the Mendeleev formula, HHV_Dulong_MJ_kg — by the Dulong formula (both daf). "
            "QC_Warning flags C+H+N+S+O > 100.5 % after conversion to daf."
        )
        if "QC_Warning" in df_elem_desc.columns:
            n_qc = int((df_elem_desc["QC_Warning"].astype(str).str.len() > 0).sum())
            if n_qc:
                st.warning(
                    f"{n_qc} образц(а/ов): сумма элементов на daf > 100.5 % — проверьте выбранный базис данных."
                    if lang == "ru" else
                    f"{n_qc} sample(s): element sum on daf > 100.5 % — check the declared data basis."
                )

        c_ed1, c_ed2, c_ed3 = st.columns([1.5, 1.5, 2.5])
        with c_ed1:
            st.download_button(
                label=T[lang]["elem_dl_csv"],
                data=df_elem_desc.to_csv(index=False).encode("utf-8"),
                file_name="ChemoSuite_Elemental_CHNS_Descriptors.csv",
                mime="text/csv",
                key="dl_elem_desc_csv",
            )
        with c_ed2:
            excel_elem_buf = io.BytesIO()
            with pd.ExcelWriter(excel_elem_buf, engine="openpyxl") as writer:
                df_elem_desc.to_excel(writer, index=False, sheet_name="Elemental_CHNS")
            st.download_button(
                label=T[lang]["elem_dl_excel"],
                data=excel_elem_buf.getvalue(),
                file_name="ChemoSuite_Elemental_Descriptors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_elem_desc_excel",
            )
        with c_ed3:
            if st.button(T[lang]["elem_sync_ml_btn"], type="primary", key="btn_sync_elem_ml_descriptors"):
                st.session_state["elemental_ml_descriptors"] = df_elem_desc
                st.toast(T[lang]["elem_sync_success"].format(n=len(df_elem_desc)), icon="✅")
                st.rerun()
