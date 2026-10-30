"""
views/sec_view.py — Size-Exclusion Chromatography (SEC / GPC) Studio View.
"""

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from views.common import st_df, st_plotly
try:
    from modules import sec_core
except ImportError:
    sec_core = None
try:
    from modules.report_generator import sanitize_excel_df
except ImportError:  # pragma: no cover
    def sanitize_excel_df(df):
        return df

_CUSTOM_CALIB_KEY = "__custom_fit__"


def _fmt(val, spec: str = ",.0f", suffix: str = "", na: str = "n/a") -> str:
    """Форматирует число; NaN/None показывается как n/a (никаких подставных значений)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(v):
        return na
    return f"{v:{spec}}{suffix}"


def render_sec_studio(T: dict, lang: str):
    """Render Module 7: Size-Exclusion & Gel Permeation Chromatography (SEC / GPC Studio)."""
    if sec_core is None:
        st.error(T[lang]["sec_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #063e36 0%, #0d5c4d 50%, #04241d 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["sec_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #BEE3DB; font-size: 13px;">{T[lang]["sec_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.popover("ℹ️ " + ("Руководство: Эксклюзионная хроматография (SEC / GPC) в анализе РОВ и лигнина" if lang == "ru" else "Guide: Size-Exclusion Chromatography (SEC / GPC) in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы гель-проникающей хроматографии (SEC / GPC) природного и техногенного органического вещества

**Эксклюзионная хроматография (HPSEC / GPC)** — фундаментальный физико-химический метод оценки молекулярно-массового распределения (ММР / MWD), полидисперсности и надмолекулярной ассоциации природного органического вещества (РОВ/DOM, гуминовые и фульвокислоты) и техногенных лигносульфонатов и шлам-лигнина.

Разделение компонентов происходит по их гидродинамическому объему ($V_h$) в порах неподвижной фазы при отсутствии сорбционных энтальпийных взаимодействий:

| Молекулярная фракция | Диапазон масс (MW) | Типичные компоненты биогеохимического пула | Экологическое / биосферное значение |
| :--- | :--- | :--- | :--- |
| **Фракция I (HMW)** | $> 20$ кДа | Высококонденсированные полимеры шлам-лигнина, биоколлоиды | Техногенный маркер стоков БЦБК, концентрированные комплексы |
| **Фракция II (MHMW)** | $5\text{--}20$ кДа | Гуминовые кислоты (HA), агрегаты ароматических доменов | Устойчивый гумусовый матрикс, связывание тяжелых металлов |
| **Фракция III (MMW)** | $1\text{--}5$ кДа | Фульвокислоты (FA), гидролизованный окисленный лигнин | Растворимый подвижный миграционный пул в воде озера Байкал |
| **Фракция IV (LMW)** | $< 1$ кДа | Низкомолекулярные мономеры, танины, простые органические кислоты | Биодоступный субстрат автохтонной микробиоты, микробный рециклинг |

---

### 📊 Среднемолекулярные массы и коэффициенты полидисперсности:
* **Среднечисловая масса ($M_n$):** $M_n = \frac{\sum h_i}{\sum (h_i / M_i)}$ — чувствительна к низкомолекулярным олигомерам и мономерам.
* **Среднемассовая масса ($M_w$):** $M_w = \frac{\sum (h_i \cdot M_i)}{\sum h_i}$ — отражает основной гидродинамический объем макромолекул.
* **z-Средняя масса ($M_z$):** $M_z = \frac{\sum (h_i \cdot M_i^2)}{\sum (h_i \cdot M_i)}$ — чувствительна к крупным полимерным сеткам и коллоидным агрегатам.
* **Индекс полидисперсности ($Đ$ или PDI):** $Đ = \frac{M_w}{M_n}$. Для монодисперсных биополимеров $Đ \approx 1.0$; для природных фульвокислот $Đ \approx 1.3–2.2$; для техногенного поликонденсированного лигнина $Đ > 3.0–8.0$.
* **Отношение $HMW/LMW$:** $\frac{\text{Frac I} + \text{Frac II}}{\text{Frac IV}}$ — чувствительный дискриминантный маркер техногенной трансформации водной экосистемы.
""" if lang == "ru" else r"""
### 🔬 Physicochemical Principles of Size-Exclusion Chromatography (SEC / GPC) in NOM & Lignin Analysis

**Size-Exclusion Chromatography (HPSEC / GPC)** separates macromolecules according to their hydrodynamic volume ($V_h$) in porous stationary phase pores without non-size interactions:

| Molecular Fraction | Mass Range (MW) | Representative Natural/Anthropogenic Components | Environmental & Geochemical Relevance |
| :--- | :--- | :--- | :--- |
| **Fraction I (HMW)** | $> 20$ kDa | Supramolecular kraft lignin clusters, biocolloids | Pulp mill sludge lignin marker, high persistence |
| **Fraction II (MHMW)**| $5\text{--}20$ kDa | Humic acids (HA), condensed polyphenolic networks | Stable soil/sediment humic pool, metal chelation |
| **Fraction III (MMW)**| $1\text{--}5$ kDa | Fulvic acids (FA), oxidized polyphenol fragments | Dominant mobile NOM fraction in Lake Baikal pelagic water |
| **Fraction IV (LMW)** | $< 1$ kDa | Low-MW organic acids, tannins, monomeric phenols | Labile autochthonous substrate for microbial uptake |

---

### 📊 Average Molecular Weights & Polydispersity Metrics:
* **Number-Average MW ($M_n$):** $M_n = \frac{\sum h_i}{\sum (h_i / M_i)}$ — sensitive to low-molecular weight oligomers.
* **Weight-Average MW ($M_w$):** $M_w = \frac{\sum (h_i \cdot M_i)}{\sum h_i}$ — represents average hydrodynamic bulk mass.
* **z-Average MW ($M_z$):** $M_z = \frac{\sum (h_i \cdot M_i^2)}{\sum (h_i \cdot M_i)}$ — sensitive to ultra-high mass supramolecular aggregates.
* **Polydispersity Index ($Đ$ or PDI):** $Đ = \frac{M_w}{M_n}$. Monodisperse standards $Đ \approx 1.0$; natural fulvic acids $Đ \approx 1.4–2.2$; industrial kraft lignin $Đ > 3.5–8.0$.
* **$HMW/LMW$ Ratio:** $\frac{\text{Frac I} + \text{Frac II}}{\text{Frac IV}}$ — primary discriminator of anthropogenic lignin pollution.
""")

    with st.sidebar:
        st.markdown(f"### {T[lang]['sec_sidebar_hdr']}")
        sec_src = st.radio(
            T[lang]["sec_src_label"],
            [T[lang]["sec_src_upload"], T[lang]["sec_src_synth"]],
            index=0,
            key="sec_data_source_radio"
        )

        if st.session_state.get("_prev_sec_src") != sec_src:
            st.session_state["_prev_sec_src"] = sec_src
            st.session_state["sec_samples"] = {}

        # Настройки хроматографической колонки
        custom_calib = None
        with st.expander("⚙️ " + ("Параметры колонки и калибровки" if lang == "ru" else "Column & Calibration Settings"), expanded=False):
            calib_preset_choices = list(sec_core.CALIBRATION_PRESETS.keys()) + [_CUSTOM_CALIB_KEY]
            calib_preset = st.selectbox(
                T[lang]["sec_calib_preset_label"],
                calib_preset_choices,
                index=0,
                format_func=lambda k: (
                    ("Своя калибровка (по стандартам)" if lang == "ru" else "Custom calibration (fit from standards)")
                    if k == _CUSTOM_CALIB_KEY
                    else sec_core.CALIBRATION_PRESETS[k]["name_ru" if lang == "ru" else "name_en"]
                ),
                key="sec_calib_preset_select"
            )
            if calib_preset == _CUSTOM_CALIB_KEY:
                std_text = st.text_area(
                    "Стандарты: V (мл), M (Да) — по одному на строку" if lang == "ru" else "Standards: V (mL), M (Da) — one per line",
                    value="10.25, 1100\n9.40, 3600\n8.55, 8200\n7.75, 17800\n7.15, 32000\n6.45, 65400\n5.85, 122000",
                    key="sec_custom_standards_text",
                )
                custom_deg = st.selectbox(
                    "Степень полинома log10(M) = f(V)" if lang == "ru" else "Polynomial degree log10(M) = f(V)",
                    [1, 2, 3], index=2, key="sec_custom_degree",
                )
                try:
                    pairs = []
                    for line in std_text.strip().splitlines():
                        parts = [p for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
                        if len(parts) >= 2:
                            pairs.append((float(parts[0]), float(parts[1])))
                    custom_calib = sec_core.fit_sec_calibration([p[0] for p in pairs], [p[1] for p in pairs], degree=int(custom_deg))
                    st.caption(
                        (f"R² = {custom_calib.r_squared:.4f}; диапазон V = {custom_calib.v_min:.2f}–{custom_calib.v_max:.2f} мл "
                         f"({custom_calib.mw_min:,.0f}–{custom_calib.mw_max:,.0f} Да). Вне диапазона масса не экстраполируется.")
                        if lang == "ru" else
                        (f"R² = {custom_calib.r_squared:.4f}; V range = {custom_calib.v_min:.2f}–{custom_calib.v_max:.2f} mL "
                         f"({custom_calib.mw_min:,.0f}–{custom_calib.mw_max:,.0f} Da). No extrapolation outside this range.")
                    )
                    if custom_calib.calibration_type.endswith("NonMonotonic"):
                        st.warning("Калибровочная кривая немонотонна в диапазоне стандартов — уменьшите степень полинома." if lang == "ru" else "The calibration curve is not monotonic within the standards' range — lower the polynomial degree.")
                except Exception as e_cal:
                    # Не подменяем пользовательскую калибровку пресетом — без калибровки массы не считаются
                    st.error((f"Не удалось построить калибровку: {e_cal}") if lang == "ru" else (f"Could not fit calibration: {e_cal}"))
                    st.stop()
            else:
                p_cal = sec_core.CALIBRATION_PRESETS[calib_preset]
                v_lo, v_hi = p_cal["v_range"]
                st.caption(
                    f"Калиброванный диапазон: V = {v_lo:.2f}–{v_hi:.2f} мл ({p_cal['mw_range'][0]:,.0f}–{p_cal['mw_range'][1]:,.0f} Да); сигнал вне диапазона исключается."
                    if lang == "ru" else
                    f"Calibrated range: V = {v_lo:.2f}–{v_hi:.2f} mL ({p_cal['mw_range'][0]:,.0f}–{p_cal['mw_range'][1]:,.0f} Da); signal outside it is excluded."
                )
            flow_rate = st.number_input(
                T[lang]["sec_flow_rate_label"],
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="sec_flow_rate_input"
            )

        # Пересчёт образцов при смене калибровки/скорости потока (иначе показываются массы старой калибровки)
        calib_sig = (
            calib_preset,
            tuple(custom_calib.coefficients) if custom_calib is not None else None,
            float(flow_rate),
        )
        if st.session_state.get("_prev_sec_calib_sig") != calib_sig:
            st.session_state["_prev_sec_calib_sig"] = calib_sig
            st.session_state["sec_samples"] = {}

        if sec_src == T[lang]["sec_src_upload"]:
            up_sec_files = st.file_uploader(
                T[lang]["sec_up_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="sec_up_files"
            )
            if up_sec_files:
                for uf in up_sec_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_SEC", "").replace("_sec", "")
                    try:
                        s_obj = sec_core.parse_sec_file(
                            b_content,
                            sample_id=s_id,
                            flow_rate_ml_min=flow_rate,
                            calibration_preset=calib_preset,
                            calibration=custom_calib,
                        )
                        st.session_state["sec_samples"][s_id] = s_obj
                    except Exception as e_sec:
                        st.warning(f"SEC error ({uf.name}): {e_sec}")
        else:
            if not st.session_state["sec_samples"]:
                demo_sec_dir = os.path.join("demo_data", "sec")
                if os.path.exists(demo_sec_dir) and any(f.endswith(".csv") for f in os.listdir(demo_sec_dir)):
                    all_files = os.listdir(demo_sec_dir)
                    sample_map = {}
                    for fn in all_files:
                        if not fn.endswith(".csv") or fn == "sec_descriptors_matrix.csv":
                            continue
                        fpath = os.path.join(demo_sec_dir, fn)
                        with open(fpath, "rb") as f:
                            b_content = f.read()
                        s_id = fn.replace("_SEC.csv", "").replace(".csv", "")
                        sample_map[s_id] = sec_core.parse_sec_file(
                            b_content,
                            sample_id=s_id,
                            flow_rate_ml_min=flow_rate,
                            calibration_preset=calib_preset,
                            calibration=custom_calib,
                        )
                    st.session_state["sec_samples"] = sample_map
                else:
                    synth_samples, _ = sec_core.generate_synthetic_sec_dataset(n_samples_per_class=6, save_dir=demo_sec_dir)
                    st.session_state["sec_samples"] = synth_samples

    if not st.session_state["sec_samples"]:
        st.info(T[lang]["sec_no_samples"])
    else:
        # Сборка матрицы дескрипторов по всем образцам
        sec_rows = []
        for s_id, s_obj in st.session_state["sec_samples"].items():
            row = {"Sample_ID": s_id}
            row.update(sec_core.extract_sec_descriptors(s_obj))
            row["Class_Target"] = 1 if any(k in str(s_id).lower() for k in ["lignin", "sludge", "шлам"]) else 0
            sec_rows.append(row)
        df_sec_desc = pd.DataFrame(sec_rows)

        sample_list = list(st.session_state["sec_samples"].keys())
        col_sel1, col_sel2 = st.columns([3, 2])
        with col_sel1:
            active_s_id = st.selectbox(
                T[lang]["sec_active_sample"],
                sample_list,
                index=0,
                key="select_active_sec_sample"
            )
        active_sample = st.session_state["sec_samples"][active_s_id]

        with col_sel2:
            mw_val = active_sample.indices.get("SEC_Mw", np.nan)
            pdi_val = active_sample.indices.get("SEC_Polydispersity", active_sample.indices.get("SEC_PDI", np.nan))
            n_peaks = len(active_sample.peaks)
            st.markdown(
                f"<div style='padding-top: 28px;'>"
                f"<span style='background: #008080; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Mw: {_fmt(mw_val, ',.0f', ' Da')}</span>"
                f"<span style='background: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Đ: {_fmt(pdi_val, '.2f')}</span>"
                f"<span style='background: #2980b9; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;'>Peaks: {n_peaks}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Калиброванный диапазон и исключённая площадь (без экстраполяции за пределы стандартов)
        excl_pct = active_sample.indices.get("SEC_Excluded_Area_pct", np.nan)
        above_pct = active_sample.indices.get("SEC_Above_Cal_Range_pct", np.nan)
        below_pct = active_sample.indices.get("SEC_Below_Cal_Range_pct", np.nan)
        cal_obj = active_sample.calibration
        for note in active_sample.metadata.get("calibration_notes", []):
            st.warning(note)
        if excl_pct is not None and np.isfinite(excl_pct) and excl_pct > 0.05:
            st.info(
                (f"Калибровка «{cal_obj.calibration_type}» действительна для V = {_fmt(cal_obj.v_min, '.2f')}–{_fmt(cal_obj.v_max, '.2f')} мл. "
                 f"{excl_pct:.1f}% площади сигнала вне этого диапазона исключено из Mn/Mw/Mz и долей фракций "
                 f"(раньше самого тяжёлого стандарта: {_fmt(above_pct, '.1f', '%')}; позже самого лёгкого: {_fmt(below_pct, '.1f', '%')}). "
                 "Фракции, чей интервал масс не покрыт стандартами, показаны как n/a.")
                if lang == "ru" else
                (f"Calibration «{cal_obj.calibration_type}» is valid for V = {_fmt(cal_obj.v_min, '.2f')}–{_fmt(cal_obj.v_max, '.2f')} mL. "
                 f"{excl_pct:.1f}% of the signal area lies outside it and is excluded from Mn/Mw/Mz and fraction shares "
                 f"(earlier than the heaviest standard: {_fmt(above_pct, '.1f', '%')}; later than the lightest: {_fmt(below_pct, '.1f', '%')}). "
                 "Fractions whose mass interval is not covered by the standards are shown as n/a.")
            )

        tab_sec1, tab_sec2, tab_sec3, tab_sec4 = st.tabs([
            T[lang]["sec_tab1_title"],
            T[lang]["sec_tab2_title"],
            T[lang]["sec_tab3_title"],
            T[lang]["sec_tab4_title"],
        ])

        # ----------------------------------------------------------------------
        # ТАБ 1: Хроматограмма элюирования и пики
        # ----------------------------------------------------------------------
        with tab_sec1:
            st.write(f"#### {T[lang]['sec_tab1_title']} — **{active_s_id}**")

            col_c1, col_c2 = st.columns([2, 2])
            with col_c1:
                x_axis_mode = st.radio(
                    T[lang]["sec_xaxis_label"],
                    [T[lang]["sec_xaxis_vol"], T[lang]["sec_xaxis_time"]],
                    horizontal=True,
                    key="sec_xaxis_toggle"
                )
            use_vol = (x_axis_mode == T[lang]["sec_xaxis_vol"])

            x_data = active_sample.elution_volume if use_vol else active_sample.retention_times
            x_title = "Elution Volume Ve (mL)" if use_vol else "Retention Time tR (min)"
            if lang == "ru":
                x_title = "Элюционный объем Ve (мл)" if use_vol else "Время удерживания tR (мин)"

            fig_chrom = go.Figure()

            # Сырой сигнал
            if len(active_sample.signal_raw) > 0 and len(x_data) == len(active_sample.signal_raw):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.signal_raw,
                    mode="lines",
                    line=dict(color="#95a5a6", width=1.2, dash="dash"),
                    name="Raw Signal (UV/RI)" if lang == "en" else "Сырой сигнал",
                ))

            # Базовая линия ALS
            if active_sample.baseline is not None and len(active_sample.baseline) == len(x_data):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.baseline,
                    mode="lines",
                    line=dict(color="#e74c3c", width=1.0, dash="dot"),
                    name="ALS Baseline" if lang == "en" else "Базовая линия ALS",
                ))

            # Скорректированный профиль
            if len(active_sample.signal_corrected) > 0 and len(x_data) == len(active_sample.signal_corrected):
                fig_chrom.add_trace(go.Scatter(
                    x=x_data,
                    y=active_sample.signal_corrected,
                    mode="lines",
                    line=dict(color="#008080", width=2.4),
                    name="Corrected Signal" if lang == "en" else "Скорректированный сигнал",
                ))

            # Идентифицированные пики
            fraction_palette = {
                "HMW (>20 kDa)": "#e74c3c",
                "MHMW (5-20 kDa)": "#e67e22",
                "MMW (1-5 kDa)": "#3498db",
                "LMW (<1 kDa)": "#2ecc71",
            }

            if active_sample.peaks:
                for p in active_sample.peaks:
                    p_x = p.elution_volume if use_vol else p.retention_time
                    p_color = fraction_palette.get(p.fraction_name, "#16a085")
                    fig_chrom.add_trace(go.Scatter(
                        x=[p_x],
                        y=[p.height],
                        mode="markers+text",
                        marker=dict(size=10, color=p_color, symbol="diamond", line=dict(width=1, color="black")),
                        text=[f"P{p.peak_id}: {_fmt(p.peak_mw, ',.0f', ' Da')}"],
                        textposition="top center",
                        name=p.fraction_name,
                        hoverinfo="text",
                        hovertext=(
                            f"Peak {p.peak_id} ({p.fraction_name})<br>"
                            f"Ve: {p.elution_volume:.3f} mL | tR: {p.retention_time:.2f} min<br>"
                            f"Mp: {_fmt(p.peak_mw, ',.1f', ' Da')}<br>"
                            f"Area: {p.area:.2f} ({p.area_pct:.1f}%)"
                        )
                    ))

            fig_chrom.update_layout(
                xaxis_title=x_title,
                yaxis_title="Detector Response (mAU / mV)" if lang == "en" else "Отклик детектора (mAU / mV)",
                template="plotly_white",
                height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st_plotly(fig_chrom)

            # Таблица пиков
            st.markdown(f"##### {T[lang]['sec_peaks_hdr']}")
            if active_sample.peaks:
                peak_records = []
                for p in active_sample.peaks:
                    peak_records.append({
                        "Peak_ID": p.peak_id,
                        "Retention_Time_min": p.retention_time,
                        "Elution_Volume_ml": p.elution_volume,
                        "Peak_MW_Mp_Da": p.peak_mw,
                        "Peak_Height": p.height,
                        "Peak_Area": p.area,
                        "Area_pct": p.area_pct,
                        "Fraction": p.fraction_name,
                    })
                df_peaks = pd.DataFrame(peak_records)
                st_df(df_peaks)

                st.download_button(
                    label=T[lang]["sec_peaks_dl_btn"],
                    data=df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name=f"{active_s_id}_SEC_Peaks.csv",
                    mime="text/csv",
                    key=f"dl_sec_peaks_{active_s_id}",
                )
            else:
                st.info("Пики не обнаружены выше установленного порога высоты." if lang == "ru" else "No peaks detected above the threshold.")

        # ----------------------------------------------------------------------
        # ТАБ 2: Молекулярно-массовое распределение (ММР / MWD)
        # ----------------------------------------------------------------------
        with tab_sec2:
            st.write(f"#### {T[lang]['sec_tab2_title']} — **{active_s_id}**")

            if len(active_sample.log_mw) > 0 and len(active_sample.mwd_diff) > 0:
                fig_mwd = make_subplots(specs=[[{"secondary_y": True}]])

                # Цветовые зоны фракций по молекулярным массам
                fig_mwd.add_vrect(x0=float(np.log10(20000)), x1=float(np.log10(200000)), fillcolor="#e74c3c", opacity=0.08, layer="below", line_width=0,
                                  annotation_text="Frac I: HMW (>20 kDa)" if lang == "en" else "Фракция I: HMW (>20 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(5000)), x1=float(np.log10(20000)), fillcolor="#e67e22", opacity=0.10, layer="below", line_width=0,
                                  annotation_text="Frac II: MHMW (5-20 kDa)" if lang == "en" else "Фракция II: MHMW (5–20 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(1000)), x1=float(np.log10(5000)), fillcolor="#3498db", opacity=0.08, layer="below", line_width=0,
                                  annotation_text="Frac III: MMW (1-5 kDa)" if lang == "en" else "Фракция III: MMW (1–5 кДа)", annotation_position="top left")
                fig_mwd.add_vrect(x0=float(np.log10(200)), x1=float(np.log10(1000)), fillcolor="#2ecc71", opacity=0.10, layer="below", line_width=0,
                                  annotation_text="Frac IV: LMW (<1 kDa)" if lang == "en" else "Фракция IV: LMW (<1 кДа)", annotation_position="top left")

                # Дифференциальная кривая ММР w(log M)
                fig_mwd.add_trace(
                    go.Scatter(
                        x=active_sample.log_mw,
                        y=active_sample.mwd_diff,
                        mode="lines",
                        line=dict(color="#008080", width=2.6),
                        name="Differential MWD w(log M)" if lang == "en" else "Дифференциальное ММР w(log M)",
                    ),
                    secondary_y=False
                )

                # Интегральная кривая ММР W(M)
                if len(active_sample.mwd_cum) == len(active_sample.log_mw):
                    fig_mwd.add_trace(
                        go.Scatter(
                            x=active_sample.log_mw,
                            y=active_sample.mwd_cum,
                            mode="lines",
                            line=dict(color="#8e44ad", width=2.0, dash="dash"),
                            name="Cumulative MWD W(M) %" if lang == "en" else "Интегральное ММР W(M) %",
                        ),
                        secondary_y=True
                    )

                # Линии средних масс: Mn, Mw, Mz, Mp
                mn_val = active_sample.indices.get("SEC_Mn", 0.0)
                mw_val = active_sample.indices.get("SEC_Mw", 0.0)
                mz_val = active_sample.indices.get("SEC_Mz", 0.0)
                mp_val = active_sample.indices.get("SEC_Mp", 0.0)

                if mn_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mn_val, 1.0))), line_width=1.5, line_dash="dash", line_color="#2980b9",
                                      annotation_text=f"Mn={mn_val:,.0f}", annotation_position="bottom left")
                if mw_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mw_val, 1.0))), line_width=2.0, line_dash="solid", line_color="#c0392b",
                                      annotation_text=f"Mw={mw_val:,.0f}", annotation_position="top left")
                if mz_val > 0:
                    fig_mwd.add_vline(x=float(np.log10(max(mz_val, 1.0))), line_width=1.5, line_dash="dot", line_color="#8e44ad",
                                      annotation_text=f"Mz={mz_val:,.0f}", annotation_position="top right")

                fig_mwd.update_xaxes(
                    title_text="log₁₀(Molecular Weight, Da)" if lang == "en" else "log₁₀(Молекулярная масса, Да)",
                )
                fig_mwd.update_yaxes(title_text="Differential Weight w(log M)" if lang == "en" else "Дифференциальная доля w(log M)", secondary_y=False)
                fig_mwd.update_yaxes(title_text="Cumulative Fraction (%)" if lang == "en" else "Кумулятивная доля (%)", secondary_y=True, range=[0, 105])
                fig_mwd.update_layout(template="plotly_white", height=470, margin=dict(l=40, r=40, t=30, b=40))
                st_plotly(fig_mwd)
            else:
                st.info("Недостаточно данных для расчета кривой ММР." if lang == "ru" else "Insufficient data to compute MWD curve.")

        # ----------------------------------------------------------------------
        # ТАБ 3: Полидисперсность и фракционирование
        # ----------------------------------------------------------------------
        with tab_sec3:
            st.write(f"#### {T[lang]['sec_tab3_title']}")

            col_km1, col_km2, col_km3, col_km4 = st.columns(4)
            col_km1.metric(T[lang]["sec_kpi_mn"], _fmt(active_sample.indices.get('SEC_Mn'), ',.0f', ' Da'), help="Среднечисловая масса (только калиброванный диапазон)" if lang == "ru" else "Number-average MW (calibrated range only)")
            col_km2.metric(T[lang]["sec_kpi_mw"], _fmt(active_sample.indices.get('SEC_Mw'), ',.0f', ' Da'), help="Среднемассовая масса (только калиброванный диапазон)" if lang == "ru" else "Weight-average MW (calibrated range only)")
            col_km3.metric(T[lang]["sec_kpi_mz"], _fmt(active_sample.indices.get('SEC_Mz'), ',.0f', ' Da'), help="z-Средняя масса (только калиброванный диапазон)" if lang == "ru" else "z-average MW (calibrated range only)")
            col_km4.metric(T[lang]["sec_kpi_mp"], _fmt(active_sample.indices.get('SEC_Mp'), ',.0f', ' Da'), help="Пиковая масса главного пика (n/a, если максимум вне калибровки)" if lang == "ru" else "Peak MW of the main peak (n/a if the maximum is outside the calibration)")

            col_kp1, col_kp2, col_kp3, col_kp4 = st.columns(4)
            col_kp1.metric(T[lang]["sec_kpi_pdi"], _fmt(active_sample.indices.get('SEC_Polydispersity', active_sample.indices.get('SEC_PDI')), '.2f'), help="Đ = Mw / Mn")
            col_kp2.metric(T[lang]["sec_kpi_hmw_ratio"], _fmt(active_sample.indices.get('SEC_HMW_LMW_ratio'), '.2f'), help="(Frac I + II) / Frac IV; n/a, если фракция IV вне калибровки" if lang == "ru" else "(Frac I + II) / Frac IV; n/a if Fraction IV is outside the calibration")
            col_kp3.metric(T[lang]["sec_kpi_frac1"], _fmt(active_sample.indices.get('SEC_Fraction_I_pct'), '.1f', '%'), help="Высокомолекулярная фракция (>20 кДа)" if lang == "ru" else "High-MW fraction (>20 kDa)")
            col_kp4.metric(T[lang]["sec_kpi_frac4"], _fmt(active_sample.indices.get('SEC_Fraction_IV_pct'), '.1f', '%'), help="Низкомолекулярная фракция (<1 кДа)" if lang == "ru" else "Low-MW fraction (<1 kDa)")

            st.markdown("---")

            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                st.write("##### " + ("Распределение молекулярных фракций (Активный образец)" if lang == "ru" else "Molecular Fraction Distribution (Active Sample)"))
                labels_all = ["HMW (>20 kDa)", "MHMW (5-20 kDa)", "MMW (1-5 kDa)", "LMW (<1 kDa)"]
                colors_all = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
                vals_all = [
                    active_sample.indices.get("SEC_Fraction_I_pct", np.nan),
                    active_sample.indices.get("SEC_Fraction_II_pct", np.nan),
                    active_sample.indices.get("SEC_Fraction_III_pct", np.nan),
                    active_sample.indices.get("SEC_Fraction_IV_pct", np.nan),
                ]
                keep = [i for i, v in enumerate(vals_all) if v is not None and np.isfinite(v)]
                labels_frac = [labels_all[i] for i in keep]
                vals_frac = [vals_all[i] for i in keep]
                if len(keep) < len(vals_all):
                    st.caption(
                        "Доли — в пределах калиброванного окна; фракции вне диапазона стандартов не показаны (n/a)."
                        if lang == "ru" else
                        "Shares are within the calibrated window; fractions outside the standards' range are not shown (n/a)."
                    )
                fig_donut_sec = go.Figure(data=[go.Pie(
                    labels=labels_frac,
                    values=vals_frac,
                    hole=0.45,
                    marker=dict(colors=[colors_all[i] for i in keep]),
                )])
                fig_donut_sec.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
                st_plotly(fig_donut_sec)

            with col_diag2:
                st.write("##### " + ("Сравнение фракций по образцам сессии" if lang == "ru" else "Fraction Comparison Across All Samples"))
                if len(df_sec_desc) > 0:
                    fig_bar_frac = go.Figure()
                    fig_bar_frac.add_trace(go.Bar(name="HMW (>20 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_I_pct"], marker_color="#e74c3c"))
                    fig_bar_frac.add_trace(go.Bar(name="MHMW (5-20 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_II_pct"], marker_color="#e67e22"))
                    fig_bar_frac.add_trace(go.Bar(name="MMW (1-5 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_III_pct"], marker_color="#3498db"))
                    fig_bar_frac.add_trace(go.Bar(name="LMW (<1 kDa)", x=df_sec_desc["Sample_ID"], y=df_sec_desc["SEC_Fraction_IV_pct"], marker_color="#2ecc71"))
                    fig_bar_frac.update_layout(barmode="stack", height=360, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20),
                                               yaxis_title="Fraction Share (%)" if lang == "en" else "Доля фракции (%)")
                    st_plotly(fig_bar_frac)

        # ----------------------------------------------------------------------
        # ТАБ 4: Мультимодальная интеграция и экспорт
        # ----------------------------------------------------------------------
        with tab_sec4:
            st.write(f"#### {T[lang]['sec_tab4_title']}")

            if len(df_sec_desc) > 1 and "SEC_Mw" in df_sec_desc.columns and "SEC_Polydispersity" in df_sec_desc.columns:
                df_plot_sec = df_sec_desc.copy()
                df_plot_sec["_size"] = df_plot_sec["SEC_Fraction_I_pct"].apply(lambda v: max(float(v), 4.0) if pd.notnull(v) else 8.0)
                fig_scatter_sec = px.scatter(
                    df_plot_sec,
                    x="SEC_Mw",
                    y="SEC_Polydispersity",
                    text="Sample_ID" if "Sample_ID" in df_plot_sec.columns else None,
                    size="_size",
                    color="Class_Target" if "Class_Target" in df_plot_sec.columns else None,
                    color_continuous_scale="Teal",
                    title="Molecular Weight (Mw) vs Polydispersity Index (Đ)" if lang == "en" else "Молекулярная масса (Mw) vs Полидисперсность (Đ)",
                    labels={
                        "SEC_Mw": "Weight-Average MW Mw (Da)",
                        "SEC_Polydispersity": "Polydispersity Index Đ (Mw/Mn)",
                        "Class_Target": "Class"
                    }
                )
                fig_scatter_sec.update_layout(height=420, template="plotly_white")
                st_plotly(fig_scatter_sec)

            st.write("##### " + ("Сводная матрица дескрипторов SEC/GPC по всем пробам" if lang == "ru" else "Summary Matrix of SEC/GPC Descriptors Across All Samples"))
            st_df(df_sec_desc)

            c_sd1, c_sd2, c_sd3 = st.columns([1.5, 1.5, 2.5])
            with c_sd1:
                st.download_button(
                    label=T[lang]["sec_dl_csv"],
                    data=df_sec_desc.to_csv(index=False).encode("utf-8"),
                    file_name="ChemoSuite_SEC_Descriptors.csv",
                    mime="text/csv",
                    key="dl_sec_descriptors_csv",
                )
            with c_sd2:
                excel_sec_buf = io.BytesIO()
                with pd.ExcelWriter(excel_sec_buf, engine="openpyxl") as writer:
                    sanitize_excel_df(df_sec_desc).to_excel(writer, index=False, sheet_name="SEC_Descriptors")
                st.download_button(
                    label=T[lang]["sec_dl_excel"],
                    data=excel_sec_buf.getvalue(),
                    file_name="ChemoSuite_SEC_Descriptors.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_sec_descriptors_excel",
                )
            with c_sd3:
                if st.button(T[lang]["sec_sync_ml_btn"], type="primary", key="btn_sync_sec_ml_descriptors"):
                    st.session_state["sec_ml_descriptors"] = df_sec_desc
                    st.toast(T[lang]["sec_sync_success"].format(n=len(df_sec_desc)), icon="✅")
                    st.rerun()
