"""
views/eem_view.py — EEM Fluorometry & UV-Vis Spectroscopy Studio View.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly
try:
    from modules import eem_core
    EEM_CORE_AVAILABLE = True
except ImportError:
    eem_core = None
    EEM_CORE_AVAILABLE = False


def render_eem_studio(T: dict, lang: str):
    """Render Module 2: Excitation-Emission Matrix Fluorometry (EEM) & UV-Vis Studio."""
    st.title(T[lang]["eem_title"])
    st.caption(T[lang]["eem_subtitle"])

    if not EEM_CORE_AVAILABLE or eem_core is None:
        st.error(T[lang]["eem_err_missing"])
        st.stop()

    with st.sidebar:
        st.header(T[lang]["eem_sidebar_hdr"])
        eem_data_src = st.radio(
            T[lang]["eem_src_label"],
            [T[lang]["eem_src_upload"], T[lang]["eem_src_synth"]],
            index=0,
            key="eem_data_source_radio",
        )
        if st.session_state.get("_prev_eem_data_src") != eem_data_src:
            st.session_state["_prev_eem_data_src"] = eem_data_src
            if "parafac_active_tables_multiselect" in st.session_state:
                del st.session_state["parafac_active_tables_multiselect"]
            if "_prev_parafac_all_sample_ids" in st.session_state:
                del st.session_state["_prev_parafac_all_sample_ids"]
            if "eem_active_sample" in st.session_state:
                del st.session_state["eem_active_sample"]

        eem_files = []
        uv_files = []
        if eem_data_src == T[lang]["eem_src_upload"]:
            eem_files = st.file_uploader(
                T[lang]["eem_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "txt", "dat"],
                key="eem_files_uploader",
            )
            # Загрузчик 1D спектров поглощения УФ-Вид
            uv_files = st.file_uploader(
                T[lang]["eem_uv_uploader_label"],
                accept_multiple_files=True,
                type=["csv", "txt", "dat"],
                key="uv_files_uploader",
            )

            # Карточка постоянного хранения EEM и UV образцов в текущей сессии
            n_eem = len(st.session_state.get("eem_stored_dict", {}))
            n_uv = len(st.session_state.get("uv_stored_dict", {}))
            if n_eem > 0 or n_uv > 0:
                with st.expander(
                    f"💾 {'Образцы в памяти сессии' if lang == 'ru' else 'Session Samples'}: {n_eem} EEM / {n_uv} UV",
                    expanded=True
                ):
                    st.caption(
                        "Файлы сохраняются в сессии при переключении любых вкладок."
                        if lang == "ru" else
                        "Files are stored in session and retained across tab switches."
                    )
                    if n_eem > 0:
                        st.markdown("**EEM матрицы:**")
                        for ename in list(st.session_state["eem_stored_dict"].keys()):
                            ce1, ce2 = st.columns([3, 1])
                            with ce1: st.markdown(f"• `{ename}`")
                            with ce2:
                                if st.button("🗑️", key=f"del_eem_m2_{ename}", help="Удалить" if lang == "ru" else "Delete"):
                                    del st.session_state["eem_stored_dict"][ename]
                                    st.session_state["eem_stored_samples"] = list(st.session_state["eem_stored_dict"].values())
                                    st.rerun()
                    if n_uv > 0:
                        st.markdown("**UV-Vis спектры:**")
                        for uname in list(st.session_state["uv_stored_dict"].keys()):
                            cu1, cu2 = st.columns([3, 1])
                            with cu1: st.markdown(f"• `{uname}`")
                            with cu2:
                                if st.button("🗑️", key=f"del_uv_m2_{uname}", help="Удалить" if lang == "ru" else "Delete"):
                                    del st.session_state["uv_stored_dict"][uname]
                                    st.session_state["uv_stored_samples"] = list(st.session_state["uv_stored_dict"].values())
                                    st.rerun()

        with st.expander(T[lang]["eem_doc_expander"], expanded=False):
            default_doc_val = st.number_input(
                T[lang]["eem_default_doc"],
                min_value=0.0,
                max_value=200.0,
                value=4.0,
                step=0.5,
                key="default_doc_input"
            )

        st.markdown("---")
        st.subheader(T[lang]["eem_preproc_hdr"])
        clean_scatter = st.checkbox(T[lang]["eem_clean_scatter"], value=True, key="eem_clean_scatter")
        norm_raman = st.checkbox(T[lang]["eem_norm_raman"], value=False, key="eem_norm_raman")
        water_blank = None
        if norm_raman:
            blank_file = st.file_uploader(
                "EEM бланка воды (Milli-Q) для нормировки R.U.:" if lang == "ru" else "Water blank EEM (Milli-Q) for R.U. normalisation:",
                type=["csv", "txt", "dat"],
                key="eem_water_blank_uploader",
                help=(
                    "Площадь Raman-пика воды интегрируется при Ex 350 нм, Em 381–426 нм (Lawaetz & Stedmon, 2009). "
                    "Без бланка данные НЕ нормируются и остаются в исходных единицах прибора."
                    if lang == "ru" else
                    "The water Raman peak is integrated at Ex 350 nm, Em 381–426 nm (Lawaetz & Stedmon, 2009). "
                    "Without a blank the data are NOT normalised and stay in instrument units."
                ),
            )
            if blank_file is not None:
                try:
                    water_blank = eem_core.load_eem_file(blank_file, sample_id="water_blank")
                except Exception as e:
                    st.error(f"Error reading water blank {blank_file.name}: {e}")
        palette = st.selectbox(T[lang]["eem_palette_label"], ["Viridis", "Plasma", "Inferno", "Turbo"], key="eem_palette_select")

    if "eem_stored_dict" not in st.session_state:
        st.session_state["eem_stored_dict"] = {}
    if "uv_stored_dict" not in st.session_state:
        st.session_state["uv_stored_dict"] = {}

    # Чтение EEM
    if eem_files:
        for f in eem_files:
            try:
                sample_name = f.name.rsplit(".", 1)[0]
                if hasattr(eem_core, "load_eem_file"):
                    parsed = eem_core.load_eem_file(f, sample_id=sample_name)
                else:
                    df = pd.read_csv(f)
                    parsed = eem_core.parse_eem_dataframe(df, sample_id=sample_name)

                if norm_raman:
                    blank_matrix = None
                    if (
                        water_blank is not None
                        and water_blank.data.shape == parsed.data.shape
                        and np.allclose(water_blank.ex, parsed.ex)
                        and np.allclose(water_blank.em, parsed.em)
                    ):
                        blank_matrix = water_blank.data
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        parsed.data, r_factor = eem_core.normalize_to_raman_units(
                            parsed.data, parsed.ex, parsed.em, water_blank=blank_matrix
                        )
                    parsed.raman_normalized = bool(np.isfinite(r_factor))
                    parsed.raman_factor = float(r_factor) if np.isfinite(r_factor) else None
                if clean_scatter:
                    parsed.data = eem_core.remove_scatter_bands(parsed.data, parsed.ex, parsed.em)
                st.session_state["eem_stored_dict"][sample_name] = parsed
            except Exception as e:
                st.error(f"Error reading EEM {f.name}: {e}")

    # Чтение 1D УФ-Вид спектров
    if uv_files:
        for uf in uv_files:
            try:
                u_name = uf.name.rsplit(".", 1)[0]
                parsed_uv = eem_core.parse_uv_vis_spectrum(uf, sample_id=u_name)
                parsed_uv.doc = default_doc_val if default_doc_val > 0 else None
                st.session_state["uv_stored_dict"][u_name] = parsed_uv
            except Exception as e:
                st.error(f"Error reading UV-Vis {uf.name}: {e}")

    # Синхронизация списков образцов
    st.session_state["eem_stored_samples"] = list(st.session_state.get("eem_stored_dict", {}).values())
    st.session_state["uv_stored_samples"] = list(st.session_state.get("uv_stored_dict", {}).values())

    if eem_data_src == T[lang]["eem_src_synth"]:
        samples = eem_core.generate_synthetic_chemometrics_dataset(n_samples=10)
        uv_samples = eem_core.generate_synthetic_uv_vis_dataset(eem_samples=samples)
    else:
        samples = st.session_state.get("eem_stored_samples", [])
        uv_samples = st.session_state.get("uv_stored_samples", [])

    # Автоматическая связка УФ-Вид спектров с EEM образцами
    if samples and uv_samples:
        doc_dict = {
            s.sample_id: (s.doc if s.doc is not None else default_doc_val)
            for s in samples
            if (s.doc is not None or default_doc_val > 0)
        }
        eem_core.link_uv_vis_to_eem(samples, uv_samples, doc_map=doc_dict if doc_dict else None)

    if not samples and not uv_samples:
        st.warning(T[lang]["eem_warn_no_data"])
    else:
        not_ru = [s.sample_id for s in samples if getattr(s, "raman_normalized", None) is False]
        if not_ru:
            st.warning(
                (f"⚠️ Нормировка в Raman Units НЕ выполнена для {len(not_ru)} образц(а/ов) "
                 f"({', '.join(not_ru[:5])}{'…' if len(not_ru) > 5 else ''}): не загружен бланк воды или его сетка "
                 "Ex/Em не совпадает с образцом. Интенсивности остаются в исходных единицах прибора (не R.U.).")
                if lang == "ru" else
                (f"⚠️ Raman-unit normalisation was NOT applied to {len(not_ru)} sample(s) "
                 f"({', '.join(not_ru[:5])}{'…' if len(not_ru) > 5 else ''}): no water blank uploaded or its Ex/Em grid "
                 "does not match. Intensities remain in instrument units (not R.U.).")
            )
        tab_eem1, tab_eem2, tab_eem3, tab_eem4 = st.tabs([
            T[lang]["eem_tab1"], T[lang]["eem_tab2"], T[lang]["eem_tab3"], T[lang]["eem_tab4"]
        ])

        # Экран 1: Контурные карты EEM
        with tab_eem1:
            if not samples:
                st.info("Нет загруженных матриц EEM." if lang == "ru" else "No EEM matrices loaded.")
            else:
                s_names = [s.sample_id for s in samples]
                selected_sname = st.selectbox(T[lang]["eem_select_sample"], s_names, key="eem_active_sample")
                s_obj = next((s for s in samples if s.sample_id == selected_sname), samples[0])

                fig_eem = go.Figure(data=go.Contour(
                    z=s_obj.data, x=s_obj.ex, y=s_obj.em, colorscale=palette.lower(),
                    contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color="white")),
                    colorbar=dict(title=(T[lang]["eem_int_label"] + (" (R.U.)" if getattr(s_obj, "raman_normalized", None) else ""))),
                ))
                fig_eem.update_layout(
                    title=T[lang]["eem_contour_title"].format(name=s_obj.sample_id),
                    xaxis_title=T[lang]["eem_ex_axis"],
                    yaxis_title=T[lang]["eem_em_axis"],
                    height=560,
                    template="plotly_dark",
                )
                st_plotly(fig_eem)

        # Экран 2: Индексы
        with tab_eem2:
            st.subheader(T[lang]["eem_indices_title"])
            if not samples:
                st.info("Нет загруженных матриц EEM." if lang == "ru" else "No EEM matrices loaded.")
            else:
                indices_list = [eem_core.calculate_spectral_indices(s) for s in samples]
                df_indices = pd.DataFrame(indices_list)

                c_idx1, c_idx2 = st.columns([3, 1])
                with c_idx1:
                    st_df(
                        df_indices.style.format(
                            {"FI": "{:.2f}", "HIX": "{:.2f}", "BIX": "{:.2f}", "A254": "{:.4f}", "DOC": "{:.2f}", "SUVA254": "{:.2f}"},
                            na_rep="—"
                        )
                    )
                with c_idx2:
                    st.download_button(
                        label=T[lang]["eem_dl_indices"],
                        data=df_indices.to_csv(index=False).encode("utf-8"),
                        file_name="EEM_Indices.csv",
                        mime="text/csv",
                        key="eem_dl_indices_btn",
                    )

        # Экран 3: PARAFAC
        with tab_eem3:
            st.subheader(T[lang]["eem_parafac_title"])
            if not samples or len(samples) < 2:
                st.warning("Для факторизации PARAFAC необходимо минимум 2 матрицы EEM." if lang == "ru" else "At least 2 EEM matrices required for PARAFAC.")
            else:
                all_sample_ids = [s.sample_id for s in samples]

                # Автоматическая синхронизация при смене источника или состава доступных таблиц
                prev_ids = st.session_state.get("_prev_parafac_all_sample_ids")
                current_selected = st.session_state.get("parafac_active_tables_multiselect")
                valid_selected = [s for s in current_selected if s in all_sample_ids] if current_selected else []

                # Если состав доступных образцов изменился или в session_state нет ни одной валидной таблицы для текущего датасета:
                if prev_ids != all_sample_ids or current_selected is None or (len(current_selected) > 0 and len(valid_selected) == 0):
                    st.session_state["_prev_parafac_all_sample_ids"] = list(all_sample_ids)
                    st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)

                col_sel1, col_sel2 = st.columns([3, 1])
                with col_sel1:
                    selected_parafac_ids = st.multiselect(
                        T[lang]["eem_parafac_select_tables"],
                        options=all_sample_ids,
                        key="parafac_active_tables_multiselect",
                    )
                    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
                    with c_btn1:
                        if st.button("Выбрать все" if lang == "ru" else "Select All", key="btn_sel_all_parafac", type="secondary"):
                            st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)
                            st.rerun()
                    with c_btn2:
                        if st.button("Сбросить" if lang == "ru" else "Deselect", key="btn_desel_parafac", type="secondary"):
                            st.session_state["parafac_active_tables_multiselect"] = []
                            st.rerun()

                with col_sel2:
                    n_components_sel = st.selectbox(
                        "Компонентов (R):" if lang == "ru" else "Components (R):",
                        [2, 3, 4],
                        index=1,
                        key="parafac_n_components_select"
                    )

                if len(selected_parafac_ids) < 2:
                    st.warning(T[lang]["eem_parafac_min_samples"])
                    if st.button("⚡ " + ("Выбрать все таблицы EEM" if lang == "ru" else "Select All EEM Tables"), key="btn_quick_select_all_parafac", type="primary"):
                        st.session_state["parafac_active_tables_multiselect"] = list(all_sample_ids)
                        st.rerun()
                else:
                    active_parafac_samples = [s for s in samples if s.sample_id in selected_parafac_ids]
                    with st.spinner(T[lang]["eem_parafac_spinner"]):
                        tensor_x, em_ax, ex_ax, s_ids = eem_core.build_eem_tensor(active_parafac_samples)
                        results = eem_core.fit_parafac(tensor_x, n_components=n_components_sel, random_state=42)

                    m_c1, m_c2, m_c3 = st.columns(3)
                    m_c1.metric(T[lang]["eem_m_exp_var"], f"{results['explained_variance']:.2f}%")
                    m_c2.metric(
                        T[lang]["eem_m_corcondia"],
                        f"{results['corcondia']:.1f}%",
                        delta=T[lang]["eem_corcondia_ok"] if results["corcondia"] > 85.0 else T[lang]["eem_corcondia_warn"],
                    )
                    m_c3.metric(T[lang]["eem_m_components"], f"{n_components_sel} флуорофора" if lang == "ru" else f"{n_components_sel} Components")

                    if results["corcondia"] >= 80.0:
                        st.success("✅ **CORCONDIA ≥ 80%**: Трилинейная модель валидирована (высокая степень однозначности разложения)." if lang == "ru" else "✅ **CORCONDIA ≥ 80%**: Trilinear model validated (high degree of uniqueness).")
                    elif results["corcondia"] >= 50.0:
                        st.warning("⚠️ **CORCONDIA 50–80%**: Умеренная адекватность трилинейной структуры. Рекомендуется сплит-хаф анализ." if lang == "ru" else "⚠️ **CORCONDIA 50–80%**: Moderate trilinearity. Split-half validation recommended.")
                    else:
                        st.error("❌ **CORCONDIA < 50%**: Низкая согласованность ядра. Высокий риск перефакторизации (overfitting)." if lang == "ru" else "❌ **CORCONDIA < 50%**: Low core consistency. Risk of overfactoring.")

                    with st.expander(T[lang]["eem_splithalf_expander"], expanded=False):
                        if len(active_parafac_samples) < 4:
                            st.info(T[lang]["eem_splithalf_need_samples"])
                        else:
                            if st.button(T[lang]["eem_splithalf_btn"], key="btn_run_split_half"):
                                with st.spinner(T[lang]["eem_splithalf_spinner"]):
                                    sh_res = eem_core.split_half_analysis(tensor_x, n_components=n_components_sel, random_state=42)
                                if sh_res.get("can_split", False):
                                    mean_tcc = sh_res["mean_tcc"]
                                    thr = float(sh_res.get("tcc_threshold", 0.95))
                                    comp_ok = sh_res.get("component_validated", [])
                                    n_ok = int(sum(bool(v) for v in comp_ok))
                                    if sh_res["is_validated"]:
                                        st.success(
                                            f"✅ Модель PARAFAC валидирована: все {n_components_sel} компонента воспроизводятся в обеих половинах (TCC Em и Ex ≥ {thr:.2f}; Murphy et al., 2013). Средний TCC = {mean_tcc:.3f}."
                                            if lang == "ru" else
                                            f"✅ PARAFAC model validated: all {n_components_sel} components reproduced in both halves (TCC Em and Ex ≥ {thr:.2f}; Murphy et al., 2013). Mean TCC = {mean_tcc:.3f}."
                                        )
                                    else:
                                        st.warning(
                                            f"⚠️ Модель PARAFAC НЕ валидирована: порог TCC ≥ {thr:.2f} по Em и Ex пройден для {n_ok} из {n_components_sel} компонентов (средний TCC = {mean_tcc:.3f} не является критерием). Возможна перефакторизация."
                                            if lang == "ru" else
                                            f"⚠️ PARAFAC model NOT validated: TCC ≥ {thr:.2f} (Em and Ex) met by {n_ok} of {n_components_sel} components (mean TCC = {mean_tcc:.3f} is not the criterion). The model may be overfactored."
                                        )

                                    st.markdown(f"**{T[lang]['eem_splithalf_tcc_title']}**")
                                    tcc_rows = []
                                    comp_names_sh = [T[lang]["eem_comp_c1"], T[lang]["eem_comp_c2"], T[lang]["eem_comp_c3"]] if n_components_sel == 3 else [f"C{r+1}" for r in range(n_components_sel)]
                                    for idx_c in range(n_components_sel):
                                        c_name = comp_names_sh[idx_c] if idx_c < len(comp_names_sh) else f"C{idx_c+1}"
                                        c_valid = bool(comp_ok[idx_c]) if idx_c < len(comp_ok) else False
                                        tcc_rows.append({
                                            "Component": c_name,
                                            "TCC Em": sh_res["tcc_em"][idx_c],
                                            "TCC Ex": sh_res["tcc_ex"][idx_c],
                                            "Status": f"✅ OK (≥{thr:.2f})" if c_valid else f"⚠️ Low (<{thr:.2f})",
                                        })
                                    st_df(pd.DataFrame(tcc_rows))

                    st.markdown("---")
                    col_p1, col_p2 = st.columns(2)
                    comp_names = [f"C{r+1}" for r in range(n_components_sel)]
                    if n_components_sel == 3:
                        comp_names = [T[lang]["eem_comp_c1"], T[lang]["eem_comp_c2"], T[lang]["eem_comp_c3"]]

                    with col_p1:
                        fig_em = go.Figure()
                        for r in range(n_components_sel):
                            fig_em.add_trace(go.Scatter(x=em_ax, y=results["em_profiles"][:, r], mode="lines", name=comp_names[r], line=dict(width=2.5)))
                        fig_em.update_layout(title=T[lang]["eem_em_prof_title"], xaxis_title=T[lang]["eem_em_axis"], yaxis_title=T[lang]["eem_rel_int"], template="plotly_white")
                        st_plotly(fig_em)

                    with col_p2:
                        fig_ex = go.Figure()
                        for r in range(n_components_sel):
                            fig_ex.add_trace(go.Scatter(x=ex_ax, y=results["ex_profiles"][:, r], mode="lines", name=comp_names[r], line=dict(width=2.5)))
                        fig_ex.update_layout(title=T[lang]["eem_ex_prof_title"], xaxis_title=T[lang]["eem_ex_axis"], yaxis_title=T[lang]["eem_rel_int"], template="plotly_white")
                        st_plotly(fig_ex)

                    ref_lib_label = getattr(
                        eem_core,
                        "REFERENCE_LIBRARY_LABEL_RU" if lang == "ru" else "REFERENCE_LIBRARY_LABEL_EN",
                        "Built-in approximate reference fluorophores, not the OpenFluor database",
                    )
                    with st.expander(
                        ("🏷️ Сопоставление со встроенными приближенными эталонными флуорофорами (не OpenFluor)"
                         if lang == "ru" else
                         "🏷️ Matching against built-in approximate reference fluorophores (not OpenFluor)"),
                        expanded=False,
                    ):
                        of_matches = eem_core.match_parafac_to_openfluor(
                            results["em_profiles"], results["ex_profiles"], em_ax, ex_ax, threshold_tcc=0.85, lang=lang
                        )
                        st.markdown(f"**{ref_lib_label}**")
                        st.caption(
                            "Эталоны — 6 гауссовых моделей типичных положений пиков (Coble A/C/M/T/B), а не спектры базы "
                            "OpenFluor. Для публикуемой идентификации сравните компоненты с базой openfluor.org."
                            if lang == "ru" else
                            "References are 6 Gaussian models of typical peak positions (Coble A/C/M/T/B), not spectra from "
                            "the OpenFluor database. For publishable identification compare components at openfluor.org."
                        )
                        st_df(
                            of_matches,
                            column_config={
                                "Component": st.column_config.TextColumn("Компонент" if lang == "ru" else "Component", width="small"),
                                "Best_Match_ID": st.column_config.TextColumn("ID эталона" if lang == "ru" else "Reference ID", width="small"),
                                "Fluorophore_Name": st.column_config.TextColumn("Флуорофор" if lang == "ru" else "Fluorophore", width="medium"),
                                "Category": st.column_config.TextColumn("Категория" if lang == "ru" else "Category", width="small"),
                                "TCC_Mean": st.column_config.NumberColumn("TCC Средний" if lang == "ru" else "TCC Mean", format="%.3f", width="small"),
                                "Confidence_Status": st.column_config.TextColumn("Статус сходства" if lang == "ru" else "Confidence Status", width="medium"),
                                "Ecological_Origin": st.column_config.TextColumn("Экологический генезис" if lang == "ru" else "Ecological Origin", width="large"),
                            }
                        )
                        st.markdown("##### " + ("Экологический генезис идентифицированных флуорофоров:" if lang == "ru" else "Ecological Origin of Identified Fluorophores:"))
                        for _, m_row in of_matches.iterrows():
                            st.markdown(
                                f"- **{m_row.get('Component')}** ({m_row.get('Fluorophore_Name')} | TCC = {m_row.get('TCC_Mean')}): "
                                f"*{m_row.get('Confidence_Status')}* — {m_row.get('Ecological_Origin')}"
                            )
                        st.download_button(
                            label=T[lang]["eem_openfluor_dl_btn"],
                            data=of_matches.to_csv(index=False).encode("utf-8"),
                            file_name="ChemoSuite_PARAFAC_BuiltinReference_Matches.csv",
                            mime="text/csv",
                            key="dl_openfluor_matches_btn",
                        )

                    st.subheader(T[lang]["eem_scores_title"])
                    score_col_names = [f"C{r+1}" for r in range(n_components_sel)]
                    if n_components_sel == 3:
                        score_col_names = ["C1_Fulvic", "C2_Lignin_Humic", "C3_Protein"]

                    scores_df = pd.DataFrame(results["scores"], columns=score_col_names)
                    scores_df.insert(0, "Sample_ID", s_ids)

                    fig_scores = px.bar(
                        scores_df, x="Sample_ID", y=score_col_names,
                        title=T[lang]["eem_scores_chart_title"], labels={"value": T[lang]["eem_int_label"], "variable": "Component"},
                        barmode="stack", template="plotly_white",
                    )
                    st_plotly(fig_scores)

                    # Экспорт объединенных оптических признаков
                    ml_export_df = pd.merge(df_indices[df_indices["Sample_ID"].isin(s_ids)], scores_df, on="Sample_ID")
                    st.session_state["eem_ml_descriptors"] = ml_export_df
                    st.download_button(
                        label=T[lang]["eem_dl_ml_btn"],
                        data=ml_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="ChemoSuite_EEM_ML_Features.csv",
                        mime="text/csv",
                        key="eem_dl_ml_btn",
                    )

        # Экран 4: УФ-Вид спектрофотометрия (UV-Vis)
        with tab_eem4:
            st.subheader(T[lang]["eem_uv_title"])
            if not uv_samples:
                st.info(T[lang]["eem_uv_no_data"])
            else:
                uv_names = [u.sample_id for u in uv_samples]
                col_uv1, col_uv2 = st.columns([3, 1])
                with col_uv1:
                    active_uv_name = st.selectbox(T[lang]["eem_uv_select_sample"], uv_names, key="active_uv_sample_select")
                active_uv = next((u for u in uv_samples if u.sample_id == active_uv_name), uv_samples[0])

                # График A(λ) с реперными точками
                fig_uv = go.Figure()
                fig_uv.add_trace(go.Scatter(
                    x=active_uv.wl, y=active_uv.absorbance,
                    mode="lines",
                    line=dict(color="#0b5394", width=2.0),
                    name="A(λ)",
                    hovertemplate="<b>λ</b>: %{x:.1f} нм<br><b>A</b>: %{y:.4f}<extra></extra>"
                ))

                for mw, col in zip([254.0, 280.0, 365.0], ["#d62728", "#e69138", "#2ca02c"]):
                    idx = np.argmin(np.abs(active_uv.wl - mw))
                    if abs(active_uv.wl[idx] - mw) <= 5.0:
                        fig_uv.add_trace(go.Scatter(
                            x=[active_uv.wl[idx]], y=[active_uv.absorbance[idx]],
                            mode="markers+text",
                            marker=dict(color=col, size=8),
                            text=[f"A_{int(mw)}={active_uv.absorbance[idx]:.3f}"],
                            textposition="top right",
                            name=f"A_{int(mw)}",
                            hoverinfo="skip"
                        ))

                fig_uv.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=0.8)
                fig_uv.update_layout(
                    title=T[lang]["eem_uv_plot_title"].format(name=active_uv.sample_id),
                    xaxis_title="Длина волны λ (нм)" if lang == "ru" else "Wavelength λ (nm)",
                    yaxis_title="Оптическая плотность A" if lang == "ru" else "Absorbance A",
                    plot_bgcolor="white",
                    height=450,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                st_plotly(fig_uv)

                # Таблица оптических дескрипторов УФ-Вид для всех загруженных проб
                uv_indices_list = [
                    eem_core.calculate_uv_vis_indices(u, doc=(u.doc if u.doc is not None else (default_doc_val if default_doc_val > 0 else None)))
                    for u in uv_samples
                ]
                df_uv_indices = pd.DataFrame(uv_indices_list)
                st.session_state["uv_ml_descriptors"] = df_uv_indices

                st.markdown(f"##### {T[lang]['eem_uv_tbl_title']}")
                c_tbl1, c_tbl2 = st.columns([3, 1])
                with c_tbl1:
                    st_df(
                        df_uv_indices.style.format({
                            "A254": "{:.4f}", "A280": "{:.4f}", "A365": "{:.4f}",
                            "E2_E3": "{:.2f}", "E4_E6": "{:.2f}",
                            "S_275_295": "{:.4f}", "S_350_400": "{:.4f}", "S_R": "{:.2f}",
                            "DOC": "{:.2f}", "SUVA254": "{:.2f}"
                        }, na_rep="—")
                    )
                    mw_hi = getattr(eem_core, "MW_UV_E2E3_RANGE", (0.0, 7.82))[1]
                    st.caption(
                        f"Mw_est — эмпирическая оценка Mw = 3450 − 390·E2/E3 (Да); первоисточник калибровки не подтвержден, "
                        f"значение ориентировочное. При E2/E3 > {mw_hi:.2f} (вне области калибровки) Mw_est не рассчитывается (—)."
                        if lang == "ru" else
                        f"Mw_est is an empirical estimate Mw = 3450 − 390·E2/E3 (Da) with an unverified calibration source; "
                        f"treat it as indicative only. For E2/E3 > {mw_hi:.2f} (outside the calibrated range) Mw_est is not reported (—)."
                    )
                with c_tbl2:
                    st.download_button(
                        label=T[lang]["eem_dl_uv_indices"],
                        data=df_uv_indices.to_csv(index=False).encode("utf-8"),
                        file_name="UV_Vis_Descriptors.csv",
                        mime="text/csv",
                        key="dl_uv_vis_descriptors_btn",
                    )
