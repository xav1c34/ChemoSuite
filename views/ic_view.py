"""
views/ic_view.py — Ion Chromatography Studio (IC).
Analyzes inorganic anions (SO4(2-), Cl-, NO3-), major cations,
low-molecular-weight organic acids (LMWOA), charge balance, and Kraft effluent markers.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import st_df, st_plotly

try:
    from modules import ic_core
except ImportError:
    ic_core = None

try:
    from modules.report_generator import sanitize_excel_df
except ImportError:  # pragma: no cover
    def sanitize_excel_df(df):
        return df


def _fmt(val, spec=".1f", suffix="", na="n/a"):
    """Formats a number; NaN/None -> 'n/a' (never a fabricated 0)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(v):
        return na
    return f"{v:{spec}}{suffix}"


_CBE_FLAG_TEXT = {
    "HCO3_missing": ("нет HCO₃⁻/щёлочности", "no HCO₃⁻/alkalinity"),
    "major_ion_missing": ("не измерен главный ион", "major ion not measured"),
    "no_ions": ("нет данных", "no data"),
}


def render_ic_studio(T: dict, lang: str):
    """Render Module 10: Ion Chromatography Studio (IC)."""
    if ic_core is None:
        st.error("IC core module not found.")
        st.stop()

    st.title(T[lang].get("ic_title", "💧 Ионная хроматография (IC: Анионы & Кислоты)"))
    st.caption(T[lang].get("ic_subtitle", "Количественное определение минеральных анионов, низкомолекулярных органических кислот (LMWOA), катионов и баланса зарядов"))

    with st.sidebar:
        st.header(T[lang].get("ic_sidebar_hdr", "📁 Данные IC"))
        data_src = st.radio(
            T[lang].get("ic_src_label", "Источник данных:"),
            [
                T[lang].get("ic_src_upload", "Загрузить таблицу (CSV/TSV)"),
                T[lang].get("ic_src_synth", "Байкальский бенчмарк (БЦБК / Пелагиаль)")
            ],
            index=1,
            key="ic_data_src_radio",
        )

        if st.session_state.get("_prev_ic_data_src") != data_src:
            st.session_state["_prev_ic_data_src"] = data_src
            st.session_state["ic_samples"] = {}

        if data_src == T[lang].get("ic_src_upload", "Загрузить таблицу (CSV/TSV)"):
            up_file = st.file_uploader(
                T[lang].get("ic_up_label", "Загрузить таблицу ионов (.csv, .tsv, .txt):"),
                type=["csv", "tsv", "txt"],
                key="ic_uploader_file",
            )
            alk_opts = {
                ("Авто (по заголовку столбца)" if lang == "ru" else "Auto (from column header)"): None,
                "meq/L (мг-экв/л)" if lang == "ru" else "meq/L": "meq/L",
                "mg/L CaCO₃": "mg/L CaCO3",
            }
            alk_choice = st.selectbox(
                "Единицы щёлочности (если есть столбец Alkalinity):" if lang == "ru" else "Alkalinity unit (if an Alkalinity column is present):",
                list(alk_opts.keys()),
                index=0,
                key="ic_alk_unit_select",
            )
            if up_file is not None:
                samples = ic_core.parse_ic_table(up_file.getvalue(), alkalinity_unit=alk_opts[alk_choice])
                st.session_state["ic_samples"] = {s.sample_id: s for s in samples}
                if samples:
                    for w in samples[0].metadata.get("parse_warnings", []):
                        st.warning(w)
                    st.caption(
                        ("Распознанные столбцы: " if lang == "ru" else "Recognised columns: ")
                        + ", ".join(f"{v}→{k}" for k, v in samples[0].metadata.get("column_map", {}).items())
                    )
        else:
            if not st.session_state.get("ic_samples"):
                bench_df = ic_core.generate_synthetic_ic_dataset()
                samples = ic_core.parse_ic_table(bench_df)
                st.session_state["ic_samples"] = {s.sample_id: s for s in samples}

    curr_samples = st.session_state.get("ic_samples", {})
    if not curr_samples:
        st.info("Нет загруженных данных ионной хроматографии. Загрузите файл или используйте бенчмарк в боковой панели.")
        st.stop()

    sample_ids = list(curr_samples.keys())
    col_sel1, col_sel2 = st.columns([2.5, 3.5])
    with col_sel1:
        active_id = st.selectbox(
            T[lang].get("ic_active_sample", "Активный образец:"),
            sample_ids,
            index=0,
            key="ic_active_sample_select",
        )
    active_sample = curr_samples[active_id]

    cbe_val = active_sample.charge_balance_error_pct
    cbe_ok = cbe_val is not None and np.isfinite(cbe_val)
    cbe_flag = getattr(active_sample, "charge_balance_flag", "")
    cbe_flag_txt = _CBE_FLAG_TEXT.get(cbe_flag, ("", ""))[0 if lang == "ru" else 1]
    cbe_str = _fmt(cbe_val, "+.1f", "%") if cbe_ok else f"n/a ({cbe_flag_txt})" if cbe_flag_txt else "n/a"

    with col_sel2:
        chips = [
            f'<span class="sample-chip"><b>SO₄²⁻</b>: {active_sample.so4_2minus:.1f} mg/L</span>',
            f'<span class="sample-chip"><b>Cl⁻</b>: {active_sample.cl_minus:.1f} mg/L</span>',
            f'<span class="sample-chip"><b>HCO₃⁻</b>: {_fmt(getattr(active_sample, "hco3_eff_mg_l", np.nan), ".1f", " mg/L")}</span>',
            f'<span class="sample-chip"><b>SO₄/Cl</b>: {_fmt(active_sample.so4_to_cl_ratio, ".1f")}</span>',
            f'<span class="sample-chip green"><b>LMWOA</b>: {active_sample.total_lmwoa_mg_l:.1f} mg/L</span>',
        ]
        if cbe_ok:
            cbe_color = "green" if abs(cbe_val) <= 5.0 else ("orange" if abs(cbe_val) <= 15.0 else "red")
        else:
            cbe_color = "orange"
        chips.append(f'<span class="sample-chip {cbe_color}"><b>CBE</b>: {cbe_str}</span>')

        k_color = "red" if active_sample.kraft_effluent_indicator > 0.6 else ("orange" if active_sample.kraft_effluent_indicator > 0.25 else "green")
        chips.append(f'<span class="sample-chip {k_color}"><b>Kraft Effluent</b>: {active_sample.kraft_effluent_indicator*100:.0f}%</span>')

        st.markdown(
            f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 26px;">{"".join(chips)}</div>',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌊 Ионный профиль & Баланс" if lang == "ru" else "🌊 Ion Profiles & Balance",
        "🎯 Диагностика & Маркеры стока" if lang == "ru" else "🎯 Diagnostic Ratios & Effluent",
        "🍃 Органические кислоты (LMWOA)" if lang == "ru" else "🍃 Organic Acids (LMWOA)",
        "📊 ML Дескрипторы & Экспорт" if lang == "ru" else "📊 ML Descriptors & Export",
    ])

    # TAB 1: Ионный профиль & Баланс
    with tab1:
        st.subheader(f"{'Ионный состав и электронейтральность' if lang == 'ru' else 'Ion Balance'} — {active_id}")
        u = "мг/л" if lang == "ru" else "mg/L"
        kpi1, kpi2, kpi3, kpi6, kpi4, kpi5 = st.columns(6)
        kpi1.metric("Сульфат (SO₄²⁻)" if lang == "ru" else "Sulfate (SO₄²⁻)", f"{active_sample.so4_2minus:.1f} {u}")
        kpi2.metric("Хлорид (Cl⁻)" if lang == "ru" else "Chloride (Cl⁻)", f"{active_sample.cl_minus:.1f} {u}")
        kpi3.metric("Нитрат (NO₃⁻)" if lang == "ru" else "Nitrate (NO₃⁻)", f"{active_sample.no3_minus:.2f} {u}")
        basis = getattr(active_sample, "carbonate_basis", "")
        kpi6.metric(
            "Гидрокарбонат (HCO₃⁻)" if lang == "ru" else "Bicarbonate (HCO₃⁻)",
            _fmt(getattr(active_sample, "hco3_eff_mg_l", np.nan), ".1f", f" {u}"),
            help=(("Источник: " if lang == "ru" else "Basis: ") + (basis or "—")),
        )
        kpi4.metric("Сумма LMWOA" if lang == "ru" else "Total LMWOA", f"{active_sample.total_lmwoa_mg_l:.2f} {u}")
        if cbe_ok:
            cbe_delta = ("В норме (≤5%)" if lang == "ru" else "Within ±5%") if abs(cbe_val) <= 5.0 else ("Дисбаланс" if lang == "ru" else "Imbalance")
        else:
            cbe_delta = None
        kpi5.metric("Баланс зарядов (CBE)" if lang == "ru" else "Charge balance (CBE)", cbe_str, delta=cbe_delta)
        if np.isfinite(getattr(active_sample, "sum_cations_meq_l", np.nan)):
            st.caption(
                (f"Σ катионов = {_fmt(active_sample.sum_cations_meq_l, '.3f')} мг-экв/л; Σ анионов = {_fmt(active_sample.sum_anions_meq_l, '.3f')} мг-экв/л; "
                 f"средний заряд PO₄ = {_fmt(active_sample.po4_mean_charge, '.2f')}. CBE = (Σкат − Σан)/(Σкат + Σан)·100 (Standard Methods 1030 E).")
                if lang == "ru" else
                (f"Σ cations = {_fmt(active_sample.sum_cations_meq_l, '.3f')} meq/L; Σ anions = {_fmt(active_sample.sum_anions_meq_l, '.3f')} meq/L; "
                 f"mean PO₄ charge = {_fmt(active_sample.po4_mean_charge, '.2f')}. CBE = (Σcat − Σan)/(Σcat + Σan)·100 (Standard Methods 1030 E).")
            )
        for note in getattr(active_sample, "notes", []):
            st.warning(note)

        st.markdown("---")
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.write("#### " + ("Спектр анионов активного образца (мг/л):" if lang == "ru" else "Anion Concentrations (mg/L):"))
            anions_data = [
                {"Ион": "SO₄²⁻", "Концентрация (мг/л)": active_sample.so4_2minus, "Тип": "Минеральный"},
                {"Ион": "Cl⁻", "Концентрация (мг/л)": active_sample.cl_minus, "Тип": "Минеральный"},
                {"Ион": "NO₃⁻", "Концентрация (мг/л)": active_sample.no3_minus, "Тип": "Минеральный"},
                {"Ион": "F⁻", "Концентрация (мг/л)": active_sample.f_minus, "Тип": "Минеральный"},
                {"Ион": "PO₄³⁻", "Концентрация (мг/л)": active_sample.po4_3minus, "Тип": "Минеральный"},
                {"Ион": "HCO₃⁻", "Концентрация (мг/л)": getattr(active_sample, "hco3_eff_mg_l", np.nan), "Тип": "Карбонатный"},
                {"Ион": "Ацетат", "Концентрация (мг/л)": active_sample.acetate, "Тип": "LMWOA"},
                {"Ион": "Формиат", "Концентрация (мг/л)": active_sample.formate, "Тип": "LMWOA"},
                {"Ион": "Оксалат", "Концентрация (мг/л)": active_sample.oxalate, "Тип": "LMWOA"},
            ]
            fig_anions = px.bar(
                pd.DataFrame(anions_data),
                x="Ион",
                y="Концентрация (мг/л)",
                color="Тип",
                color_discrete_map={"Минеральный": "#0ea5e9", "LMWOA": "#10b981", "Карбонатный": "#64748b"},
                template="plotly_white",
            )
            fig_anions.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st_plotly(fig_anions)

        with c_chart2:
            st.write("#### " + ("Соотношение катионов активного образца (мг/л):" if lang == "ru" else "Cations Distribution (mg/L):"))
            cations_data = [
                {"Катион": "Na⁺", "Концентрация": active_sample.na_plus},
                {"Катион": "K⁺", "Концентрация": active_sample.k_plus},
                {"Катион": "Ca²⁺", "Концентрация": active_sample.ca_2plus},
                {"Катион": "Mg²⁺", "Концентрация": active_sample.mg_2plus},
                {"Катион": "NH₄⁺", "Концентрация": active_sample.nh4_plus},
            ]
            fig_cat = px.pie(
                pd.DataFrame(cations_data),
                names="Катион",
                values="Концентрация",
                hole=0.45,
                template="plotly_white",
            )
            fig_cat.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st_plotly(fig_cat)

    # TAB 2: Диагностика & Маркеры стока
    with tab2:
        st.subheader("🎯 " + ("Геохимические и техногенные индикаторы" if lang == "ru" else "Diagnostic Geochemical Ratios"))
        st.write(
            "Отношение $SO_4^{2-}/Cl^-$ и избыточный сульфат являются главными индикаторами воздействия стоков сульфат-целлюлозного производства (БЦБК) на ультраолиготрофные байкальские воды."
            if lang == "ru" else
            "SO4/Cl ratio and sulfate excess are prime markers of Kraft pulp mill effluent impact on oligotrophic waters."
        )

        all_ic_rows = []
        for s in curr_samples.values():
            all_ic_rows.append({
                "Sample_ID": s.sample_id,
                "SO4": s.so4_2minus,
                "Cl": s.cl_minus,
                "SO4_Cl_Ratio": s.so4_to_cl_ratio,
                "LMWOA_Total": s.total_lmwoa_mg_l,
                "HCO3": getattr(s, "hco3_eff_mg_l", np.nan),
                "CBE_pct": s.charge_balance_error_pct,
                "CBE_flag": getattr(s, "charge_balance_flag", ""),
                "Kraft_Index": s.kraft_effluent_indicator,
                "Sample_Type": s.sample_type,
            })
        ic_overview_df = pd.DataFrame(all_ic_rows)

        col_ratio1, col_ratio2 = st.columns(2)
        with col_ratio1:
            fig_so4_cl = px.scatter(
                ic_overview_df,
                x="Cl",
                y="SO4",
                color="Sample_Type",
                size="Kraft_Index",
                size_max=16,
                hover_name="Sample_ID",
                template="plotly_white",
                title="Сульфат vs Хлорид (Трассировка шлама)" if lang == "ru" else "SO4 vs Cl (Sludge Tracing)",
                labels={"SO4": "SO₄²⁻ (мг/л)", "Cl": "Cl⁻ (мг/л)"},
            )
            # Pelagic Baikal background SO4/Cl ≈ 13 by mass (SO4 ≈ 5.5, Cl ≈ 0.43 mg/L; Falkner et al. 1991)
            bg_ratio = getattr(ic_core, "BAIKAL_SO4_CL_BACKGROUND", 13.0)
            max_cl = max(15.0, ic_overview_df["Cl"].max())
            fig_so4_cl.add_trace(go.Scatter(
                x=[0, max_cl],
                y=[0, max_cl * bg_ratio],
                mode="lines",
                name=(f"Фон Байкала (SO₄/Cl ≈ {bg_ratio:.0f})" if lang == "ru" else f"Baikal background (SO₄/Cl ≈ {bg_ratio:.0f})"),
                line=dict(color="#0ea5e9", dash="dot"),
            ))
            fig_so4_cl.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
            st_plotly(fig_so4_cl)

        with col_ratio2:
            fig_kraft = px.bar(
                ic_overview_df,
                x="Sample_ID",
                y="Kraft_Index",
                color="Kraft_Index",
                color_continuous_scale="Reds",
                template="plotly_white",
                title="Индикатор техногенного стока БЦБК (Kraft Effluent Index)" if lang == "ru" else "Kraft Effluent Indicator",
                labels={"Kraft_Index": "Индекс воздействия (0-1)", "Sample_ID": "Образец"},
            )
            fig_kraft.add_hline(y=0.4, line_dash="dash", line_color="#e11d48", annotation_text="Порог выраженного техногенеза")
            fig_kraft.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
            st_plotly(fig_kraft)

    # TAB 3: Низкомолекулярные кислоты (LMWOA)
    with tab3:
        st.subheader("🍃 " + ("Низкомолекулярные органические кислоты (LMWOA)" if lang == "ru" else "Low-Molecular-Weight Organic Acids"))
        st.write(
            "Ацетат и формиат образуются при щелочной отварке древесины (peeling reactions), тогда как оксалат и цитрат характерны для корневых экссудатов и разложения почвенного гумуса."
            if lang == "ru" else
            "Acetate and formate are generated during alkaline Kraft wood cooking, whereas oxalate/citrate reflect soil humic leaching."
        )

        lmwoa_table = []
        for s in curr_samples.values():
            lmwoa_table.append({
                "Sample_ID": s.sample_id,
                "Ацетат (мг/л)": s.acetate,
                "Формиат (мг/л)": s.formate,
                "Оксалат (мг/л)": s.oxalate,
                "Лактат (мг/л)": s.lactate,
                "Сукцинат (мг/л)": s.succinate,
                "Сумма LMWOA": s.total_lmwoa_mg_l,
                "Доля LMWOA (%)": round(s.lmwoa_fraction * 100.0, 1),
            })
        st_df(pd.DataFrame(lmwoa_table))

    # TAB 4: ML Дескрипторы & Экспорт
    with tab4:
        st.subheader("📊 " + ("Хемометрическая матрица дескрипторов IC" if lang == "ru" else "IC Chemometric Descriptors"))
        desc_df = ic_core.calculate_ic_descriptors(list(curr_samples.values()))
        st_df(desc_df)

        c_sync, c_exp_csv, c_exp_xlsx = st.columns([2.5, 1.5, 1.5])
        with c_sync:
            if st.button("🧬 " + ("Отправить в ChemoSuite ML (Data Fusion)" if lang == "ru" else "Push to ChemoSuite ML"), type="primary", use_container_width=True):
                st.session_state["ic_ml_descriptors"] = desc_df.reset_index()
                st.success("Дескрипторы IC успешно зарегистрированы в блоке слияния данных!")
        with c_exp_csv:
            csv_data = desc_df.to_csv().encode("utf-8")
            st.download_button("📥 Скачать CSV", csv_data, "ic_descriptors.csv", "text/csv", use_container_width=True)
        with c_exp_xlsx:
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                sanitize_excel_df(desc_df.reset_index()).to_excel(writer, sheet_name="IC_Data", index=False)
            st.download_button("📊 Скачать Excel", out_buf.getvalue(), "ic_descriptors.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
