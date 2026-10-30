"""
views/raman_view.py — Raman Spectroscopy Studio for Carbonaceous NOM & Lignin.
Analyzes D and G bands (800-2000 cm^-1), I_D/I_G, A_D/A_G deconvolution,
five-band Sadezky fit, crystallite size L_a (Cançado 2006 / Ferrari-Robertson 2000, laser-dependent)
and a heuristic order index from FWHM(G). Failed fits are shown as "—", never as numbers.
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


def _fmt(v, fmt: str = "{:.2f}", suffix: str = "") -> str:
    """Форматирует число; NaN/None (подгонка не удалась) — «—»."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(fv) else fmt.format(fv) + suffix


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

        default_laser = float(getattr(raman_core, "DEFAULT_LASER_NM", 532.0))
        laser_nm = float(st.number_input(
            "Длина волны лазера λ (нм):" if lang == "ru" else "Laser wavelength λ (nm):",
            min_value=200.0, max_value=1100.0, value=default_laser, step=1.0,
            key="raman_laser_nm",
            help=(
                "L_a зависит от λ⁴ (Cançado et al. 2006): одно и то же A_D/A_G дает разные L_a при 514.5, 532 и 633 нм."
                if lang == "ru" else
                "L_a scales with λ⁴ (Cançado et al. 2006): the same A_D/A_G gives different L_a at 514.5, 532 and 633 nm."
            ),
        ))

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
                    try:
                        spec = raman_core.parse_raman_spectrum(f.getvalue(), sample_id=s_id, laser_wavelength_nm=laser_nm)
                    except ValueError as exc:
                        st.error(f"{f.name}: {exc}")
                        continue
                    st.session_state.setdefault("raman_samples", {})[s_id] = spec
        else:
            if not st.session_state.get("raman_samples"):
                bench_dict = raman_core.generate_synthetic_raman_dataset()
                st.session_state["raman_samples"] = bench_dict

    curr_samples = st.session_state.get("raman_samples", {})
    if not curr_samples:
        st.info("Нет загруженных спектров комбинационного рассеяния (Raman). Загрузите файл или выберите бенчмарк в боковой панели.")
        st.stop()

    # L_a пересчитывается при смене длины волны лазера (подгонка не повторяется)
    for _s in curr_samples.values():
        if abs(float(getattr(_s, "laser_wavelength_nm", default_laser)) - laser_nm) > 1e-9:
            _s.estimate_crystallite_size(laser_nm)

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
            f'<span class="sample-chip"><b>I_D/I_G</b>: {_fmt(active_spec.id_ig_ratio)}</span>',
            f'<span class="sample-chip"><b>A_D/A_G</b>: {_fmt(active_spec.ad_ag_ratio)}</span>',
            f'<span class="sample-chip green"><b>L_a</b>: {_fmt(active_spec.la_crystallite_nm, "{:.1f}", " nm")}</span>',
            f'<span class="sample-chip orange"><b>G-FWHM</b>: {_fmt(active_spec.fwhm_g, "{:.1f}", " cm⁻¹")}</span>',
        ]
        order_v = active_spec.carbon_order_index
        ord_color = "blue" if not np.isfinite(order_v) else ("green" if order_v > 0.6 else ("orange" if order_v > 0.35 else "blue"))
        chips.append(f'<span class="sample-chip {ord_color}"><b>Order (heuristic, FWHM G)</b>: {_fmt(order_v)}</span>')

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
        if not getattr(active_spec, "fit_success", True):
            st.error(
                ("❌ Пятиполосная подгонка не удалась — дескрипторы не рассчитаны: " if lang == "ru"
                 else "❌ Five-band fit failed — no descriptors computed: ")
                + str(getattr(active_spec, "fit_message", ""))
            )
        elif not getattr(active_spec, "fit_ok", True):
            st.warning(
                ("⚠️ Качество подгонки неудовлетворительно, интерпретируйте с осторожностью: " if lang == "ru"
                 else "⚠️ Poor fit quality, interpret with care: ")
                + "; ".join(getattr(active_spec, "fit_warnings", []) or [])
            )
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("I_D1 / I_G (Высота)", _fmt(active_spec.id_ig_ratio))
        kpi2.metric("A_D1 / A_G (Площадь)", _fmt(active_spec.ad_ag_ratio))
        kpi3.metric("Размер кластеров L_a", _fmt(active_spec.la_crystallite_nm, "{:.1f}", " нм"))
        kpi4.metric("Позиция G-полосы", _fmt(active_spec.g_pos, "{:.1f}", " см⁻¹"))
        kpi5.metric("Позиция D1-полосы", _fmt(active_spec.d_pos, "{:.1f}", " см⁻¹"))
        st.caption(
            (f"Модель Sadezky et al. 2005 (D4, D1, D3, G, D2), R² = {_fmt(getattr(active_spec, 'fit_r2', float('nan')), '{:.4f}')}. "
             f"L_a: {getattr(active_spec, 'la_method', '') or '—'}; Cançado (TK) = {_fmt(getattr(active_spec, 'la_tk_nm', float('nan')), '{:.1f}', ' нм')}, "
             f"Ferrari–Robertson = {_fmt(getattr(active_spec, 'la_fr_nm', float('nan')), '{:.1f}', ' нм')}. "
             "I_D/I_G немонотонно зависит от порядка (Ferrari & Robertson 2000), поэтому режим нельзя однозначно определить по одному отношению; "
             "индекс упорядоченности — эвристическая шкала по FWHM(G), не литературный индекс.")
            if lang == "ru" else
            (f"Sadezky et al. 2005 model (D4, D1, D3, G, D2), R² = {_fmt(getattr(active_spec, 'fit_r2', float('nan')), '{:.4f}')}. "
             f"L_a: {getattr(active_spec, 'la_method', '') or '—'}; Cançado (TK) = {_fmt(getattr(active_spec, 'la_tk_nm', float('nan')), '{:.1f}', ' nm')}, "
             f"Ferrari–Robertson = {_fmt(getattr(active_spec, 'la_fr_nm', float('nan')), '{:.1f}', ' nm')}. "
             "I_D/I_G is non-monotonic in structural order (Ferrari & Robertson 2000), so the regime cannot be fixed from the ratio alone; "
             "the order index is a heuristic FWHM(G) scale, not a literature index.")
        )

        st.markdown("---")
        fig_fit = go.Figure()

        # Нормированный спектр на оси окна подгонки (analysis_wavenumbers, отсортирована)
        wn = np.asarray(active_spec.wavenumbers, dtype=float)
        mask = (wn >= 1000.0) & (wn <= 1800.0)
        wn_plot = getattr(active_spec, "analysis_wavenumbers", None)
        if wn_plot is None:
            wn_plot = wn[mask]
        y_plot = active_spec.corrected_intensities if active_spec.corrected_intensities is not None else np.asarray(active_spec.intensities)[mask]

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
                "D4_band": ("#64748b", "D4 полоса (~1200 см⁻¹: sp²–sp³, полиены)"),
                "D1_band": ("#e11d48", "D1 полоса (~1350 см⁻¹: дефекты колец)"),
                "D3_amorphous": ("#d97706", "D3 полоса (~1500 см⁻¹: аморфный C)"),
                "G_band": ("#0ea5e9", "G полоса (~1585 см⁻¹: sp² решетка)"),
                "D2_band": ("#16a34a", "D2 полоса (~1620 см⁻¹: поверхностные слои)"),
                "Residual_Background": ("#94a3b8", "Остаточный линейный фон (подгоняется вместе с полосами)"),
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
                w = np.asarray(s.wavenumbers, dtype=float)
                m = (w >= 1000.0) & (w <= 1800.0)
                x_s = getattr(s, "analysis_wavenumbers", None)
                if x_s is None:
                    x_s = w[m]
                y = s.corrected_intensities if s.corrected_intensities is not None else np.asarray(s.intensities)[m]
                fig_multi.add_trace(go.Scatter(
                    x=x_s,
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
            # Размер маркера: NaN (неудачная подгонка) недопустим для plotly
            df_raman_meta["Marker_Size"] = pd.to_numeric(df_raman_meta["Carbon_Order"], errors="coerce").fillna(0.0).clip(lower=0.05)

            fig_order = px.scatter(
                df_raman_meta,
                x="FWHM_G",
                y="ID_IG",
                color="Sample_Type",
                size="Marker_Size",
                size_max=16,
                hover_name="Sample_ID",
                template="plotly_white",
                labels={"FWHM_G": "Ширина G-полосы FWHM (см⁻¹)", "ID_IG": "Отношение I_D / I_G"},
            )
            fig_order.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)
            st_plotly(fig_order)
            n_failed = int(sum(1 for s in curr_samples.values() if not getattr(s, "fit_success", True)))
            if n_failed:
                st.warning(
                    f"{n_failed} спектр(а/ов): подгонка не удалась, на диаграмме не показаны." if lang == "ru"
                    else f"{n_failed} spectrum/spectra: fit failed, not shown on the diagram."
                )
            st.caption(
                "I_D/I_G немонотонно зависит от упорядоченности (Ferrari & Robertson 2000): читайте диаграмму вместе с FWHM(G). "
                "Референсные области классов не показаны — для них нет откалиброванных данных."
                if lang == "ru" else
                "I_D/I_G is non-monotonic in structural order (Ferrari & Robertson 2000): read it together with FWHM(G). "
                "No class reference regions are drawn, as no calibrated data back them."
            )

    # TAB 3: Мультимодальная связь (Thermal & FT-ICR)
    with tab3:
        st.subheader("🔥 " + ("Кросс-модальная валидация (Raman ↔ Py-GC-MS / TGA ↔ FT-ICR MS)" if lang == "ru" else "Cross-Modal Synergy"))
        st.write(
            "Параметр $I_D/I_G$ и размер кластеров $L_a$ прямо коррелируют с выходом коксового остатка (Char Yield при 600°C по данным ТГА) и долей конденсированных полиароматических структур ($AI_{mod} \\ge 0.67$) по данным FT-ICR масс-спектрометрии."
            if lang == "ru" else
            "Raman I_D/I_G ratio and La crystallite size directly correlate with TGA char yield and FT-ICR condensed aromatic formulas (AI_mod >= 0.67)."
        )
        st.info(
            "Связь Raman-дескрипторов с выходом кокса (ТГА) и долей конденсированных формул (FT-ICR) следует проверять корреляцией на собственных образцах: "
            "встроенные бенчмарк-спектры синтетические, и характерные диапазоны I_D/I_G или L_a для шлам-лигнина БЦБК по ним не устанавливаются."
            if lang == "ru" else
            "Links between Raman descriptors, TGA char yield and FT-ICR condensed formulas must be checked by correlation on your own samples: "
            "the built-in benchmark spectra are synthetic and do not establish characteristic I_D/I_G or L_a ranges for BCBK sludge lignin."
        )

    # TAB 4: ML Дескрипторы & Экспорт
    with tab4:
        st.subheader("📊 " + ("Хемометрическая матрица дескрипторов Raman" if lang == "ru" else "Raman Chemometric Descriptors"))
        desc_df = raman_core.calculate_raman_descriptors(list(curr_samples.values()))
        st_df(desc_df)
        if "fit_success" in desc_df.columns and not bool(desc_df["fit_success"].all()):
            st.warning(
                "Для части спектров подгонка не удалась: их дескрипторы — NaN." if lang == "ru"
                else "Some spectra failed to fit: their descriptors are NaN."
            )
        qc_cols = list(getattr(raman_core, "RAMAN_QC_COLUMNS", []))

        c_sync, c_exp_csv, c_exp_xlsx = st.columns([2.5, 1.5, 1.5])
        with c_sync:
            if st.button("🧬 " + ("Отправить в ChemoSuite ML (Data Fusion)" if lang == "ru" else "Push to ChemoSuite ML"), type="primary", use_container_width=True):
                st.session_state["raman_ml_descriptors"] = desc_df.drop(columns=qc_cols, errors="ignore").reset_index()
                st.success("Дескрипторы Raman успешно зарегистрированы в блоке слияния данных!")
        with c_exp_csv:
            csv_data = desc_df.to_csv().encode("utf-8")
            st.download_button("📥 Скачать CSV", csv_data, "raman_descriptors.csv", "text/csv", use_container_width=True)
        with c_exp_xlsx:
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                desc_df.to_excel(writer, sheet_name="Raman_Data")
            st.download_button("📊 Скачать Excel", out_buf.getvalue(), "raman_descriptors.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
