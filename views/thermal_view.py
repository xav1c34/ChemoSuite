"""
views/thermal_view.py — Thermal Analysis Studio (Py-GC-MS & TGA/DTG).
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
    from modules import thermal_core
except ImportError:
    thermal_core = None


def render_thermal_studio(T: dict, lang: str):
    """Render Module 6: Analytical Pyrolysis (Py-GC-MS) & Thermogravimetry (TGA/DTG) Studio."""
    if thermal_core is None:
        st.error(T[lang]["thermal_err_missing"])
        st.stop()

    st.markdown(
        f'<div style="background: linear-gradient(90deg, #5c0617 0%, #870000 50%, #190a2c 100%); padding: 16px 20px; border-radius: 8px; color: white; margin-bottom: 20px;">'
        f'<h3 style="margin: 0; color: #FFFFFF; font-size: 22px;">{T[lang]["thermal_title"]}</h3>'
        f'<p style="margin: 4px 0 0 0; color: #F5D0D0; font-size: 13px;">{T[lang]["thermal_subtitle"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.popover("ℹ️ " + ("Руководство: Аналитический пиролиз (Py-GC-MS) и термогравиметрия (TGA/DTG) в анализе РОВ и лигнина" if lang == "ru" else "Guide: Analytical Pyrolysis (Py-GC-MS) & Thermogravimetry (TGA/DTG) in NOM & Lignin Analysis"), width="stretch"):
        st.markdown(r"""
### 🔬 Физико-химические основы термодеструкции природного и техногенного органического вещества

**Аналитический пиролиз (Flash Py-GC-MS, 500–600°C)** в бескислородной среде гелия вызывает термический разрыв наименее прочных связей ($\beta$-O-4, $\beta$-5, алифатические мостики) макромолекулярного скелета, высвобождая летучие мономерные фрагменты, отражающие структуру исходного полимера:

| Группа мономеров | Ключевые маркеры (Py-GC-MS) | Специфичность | Экологическое / биогеохимическое значение |
| :--- | :--- | :--- | :--- |
| **Гваяцильные (G)** | Гваякол, 4-Метилгваякол, Ванилин, Ванилиновая к-та | Хвойный лигнин (сосна, лиственница Прибайкалья) | Доминирующий маркер сульфатного шлам-лигнина БЦБК |
| **Сирингильные (S)** | Сирингол, 4-Метилсирингол, Сирингальдегид, Сиринговая к-та | Лиственный лигнин (береза, осина) | Источник аллохтонного терригенного стока |
| **п-Гидроксифенильные (H)** | Фенол, о/м/п-Крезолы, 4-Винилфенол | Травянистые растения, мхи, торфяники | Автохтонный и моховой гуминовый пул |
| **Углеводы (Carb)** | Фурфурол, 5-HMF, Левоглюкозан, 2-Фурилметилкетон | Полисахариды, целлюлоза, гемицеллюлозы | Доля биогенного углеводного матрикса |
| **Азотистые (N)** | Пиридин, Пиррол, Индол, Метилиндол | Белки, пептиды, микробная биомасса | Индикатор микробной активности и автохтонного РОВ |

---

### 📊 Хемометрические индексы пиролиза:
* **Индекс $S/G$:** Отношение сирингильных мономеров к гваяцильным. Для чистых хвойных лигнинов $S/G \approx 0.01–0.05$; для лиственных $S/G > 1.0$.
* **Индекс $H/G$:** Доля неметоксилированных фенолов. Возрастает при деметилировании и микробной биодеградации.
* **Индекс $L/C$:** Отношение суммарного пула лигнина $(G+S+H)$ к углеводам. Ключевой биомаркер зрелости и техногенного загрязнения лигнином.
* **Степень окисления $(Ac/Al)_G$ и $(Ac/Al)_S$:** Отношение карбоновых кислот к соответствующим альдегидам (ванилиновая к-та / ванилин). В интактном лигнине $(Ac/Al)_G < 0.20$; при техногенном щелочном окислении или глубоком выветривании возрастает до $0.50–1.20$.

---

### 🔥 Термогравиметрия (TGA) и дифференциальная термогравиметрия (DTG, $-dm/dT$):
Кривая убыли массы и скорость термодеструкции разделяются на 4 температурные зоны:
1. **Зона I ($T < 150^\circ\text{C}$):** Десорбция гигроскопической влаги, окклюдированных газов и сверхлетучих фрагментов.
2. **Зона II ($150–350^\circ\text{C}$):** Деструкция гемицеллюлоз, эфирных связей $\beta$-O-4 и алифатических боковых цепей ($\Delta m_{\text{aliph}}$).
3. **Зона III ($350–600^\circ\text{C}$):** Глубокий пиролиз ароматического углеродного скелета, конденсация и распад стабильных C–C связей лигнина ($\Delta m_{\text{arom}}$).
4. **Зона IV ($T > 600^\circ\text{C}$):** Обугливание высококонденсированного полиароматического кокса и термолиз минерального зольного остатка.
* **Thermal Index ($TI$):** $TI = \frac{\Delta m_{\text{III}}}{\Delta m_{\text{II}}} = \frac{\Delta m_{\text{arom}}}{\Delta m_{\text{aliph}}}$. Отражает термическую стабильность и долю тугоплавкого конденсированного лигнина.
""" if lang == "ru" else r"""
### 🔬 Physicochemical Principles of Thermal Degradation in NOM & Lignin

**Analytical Pyrolysis (Flash Py-GC-MS, 500–600°C)** under inert helium cleaves weaker ether ($\beta$-O-4) and aliphatic bonds, liberating volatile monomers that reflect structural precursors:

| Monomer Class | Representative Biomarkers | Biological Specificity | Environmental / Geochemical Relevance |
| :--- | :--- | :--- | :--- |
| **Guaiacyl (G)** | Guaiacol, 4-Methylguaiacol, Vanillin, Vanillic acid | Softwood lignin (Pine, Larch from Lake Baikal basin) | Primary chemical signature of kraft sludge lignin |
| **Syringyl (S)** | Syringol, 4-Methylsyringol, Syringaldehyde, Syringic acid | Hardwood lignin (Birch, Aspen) | Marker of angiosperm terrigenous runoff |
| **p-Hydroxyphenyl (H)**| Phenol, o/m/p-Cresols, 4-Vinylphenol | Non-woody tissues, mosses, peatlands | Autochthonous & bryophyte organic pool |
| **Carbohydrate (Carb)**| Furfural, 5-HMF, Levoglucosan, 2-Furylmethylketone | Polysaccharides, cellulose, hemicellulose | Biogenic carbohydrate matrix contribution |
| **Nitrogen (N)** | Pyridine, Pyrrole, Indole, Methylindole | Proteins, amino sugars, microbial biomass | Microbial activity and autochthonous NOM |

---

### 📊 Pyrolysis Chemometric Indices:
* **$S/G$ Ratio:** Syringyl-to-guaiacyl ratio ($S/G < 0.05$ for gymnosperms/softwood kraft lignin, $> 1.0$ for angiosperms).
* **$H/G$ Ratio:** p-Hydroxyphenyl to guaiacyl ratio (increases during demethylation & biodegradation).
* **$L/C$ Ratio:** Total lignin $(G+S+H)$ to carbohydrate ratio. Discriminates biogenic NOM from concentrated kraft lignin.
* **Oxidation State $(Ac/Al)_G$ & $(Ac/Al)_S$:** Acid-to-aldehyde ratios. Fresh wood $< 0.20$; degraded or pulping-oxidized lignin reaches $0.50–1.20$.

---

### 🔥 Thermogravimetry (TGA) & Derivative (DTG, $-dm/dT$):
Decomposition is separated into 4 kinetic temperature zones:
1. **Zone I ($T < 150^\circ\text{C}$):** Moisture desorption & volatile degasification.
2. **Zone II ($150–350^\circ\text{C}$):** Hemicellulose, aliphatic scission & $\beta$-O-4 cleavage ($\Delta m_{\text{aliph}}$).
3. **Zone III ($350–600^\circ\text{C}$):** Pyrolysis of cross-linked aromatic lignin core & refractory C-C linkages ($\Delta m_{\text{arom}}$).
4. **Zone IV ($T > 600^\circ\text{C}$):** Polycondensed char graphitization & inorganic mineral ash.
* **Thermal Index ($TI$):** $TI = \frac{\Delta m_{\text{III}}}{\Delta m_{\text{II}}} = \frac{\Delta m_{\text{arom}}}{\Delta m_{\text{aliph}}}$. Quantifies thermal stability and aromatic condensation.
""")

    with st.sidebar:
        st.markdown(f"### {T[lang]['thermal_sidebar_hdr']}")
        thermal_src = st.radio(
            T[lang]["thermal_src_label"],
            [T[lang]["thermal_src_upload"], T[lang]["thermal_src_synth"]],
            index=0,
            key="thermal_data_source_radio"
        )

        if st.session_state.get("_prev_thermal_src") != thermal_src:
            st.session_state["_prev_thermal_src"] = thermal_src
            st.session_state["thermal_samples"] = {}

        if thermal_src == T[lang]["thermal_src_upload"]:
            up_py_files = st.file_uploader(
                T[lang]["thermal_up_py_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="thermal_up_py_files"
            )
            up_tga_files = st.file_uploader(
                T[lang]["thermal_up_tga_label"],
                type=["csv", "txt", "tsv"],
                accept_multiple_files=True,
                key="thermal_up_tga_files"
            )

            if up_py_files:
                for uf in up_py_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_PyGCMS", "").replace("_pygcms", "")
                    try:
                        py_obj = thermal_core.parse_py_gcms_file(b_content, sample_id=s_id)
                        if s_id not in st.session_state["thermal_samples"]:
                            st.session_state["thermal_samples"][s_id] = thermal_core.ThermalSample(sample_id=s_id, py_gcms=py_obj)
                        else:
                            st.session_state["thermal_samples"][s_id].py_gcms = py_obj
                    except Exception as e_py:
                        st.warning(f"Py-GC-MS error ({uf.name}): {e_py}")

            if up_tga_files:
                for uf in up_tga_files:
                    b_content = uf.getvalue()
                    s_id = uf.name.rsplit(".", 1)[0].replace("_TGA", "").replace("_tga", "")
                    try:
                        tga_obj = thermal_core.parse_tga_file(b_content, sample_id=s_id)
                        if s_id not in st.session_state["thermal_samples"]:
                            st.session_state["thermal_samples"][s_id] = thermal_core.ThermalSample(sample_id=s_id, tga=tga_obj)
                        else:
                            st.session_state["thermal_samples"][s_id].tga = tga_obj
                    except Exception as e_tga:
                        st.warning(f"TGA error ({uf.name}): {e_tga}")
        else:
            if not st.session_state["thermal_samples"]:
                demo_th_dir = os.path.join("demo_data", "thermal")
                if os.path.exists(demo_th_dir) and any(f.endswith(".csv") for f in os.listdir(demo_th_dir)):
                    all_files = os.listdir(demo_th_dir)
                    sample_map = {}
                    for fn in all_files:
                        if not fn.endswith(".csv") or fn == "thermal_descriptors_matrix.csv":
                            continue
                        fpath = os.path.join(demo_th_dir, fn)
                        with open(fpath, "rb") as f:
                            b_content = f.read()
                        if "_PyGCMS" in fn:
                            s_id = fn.replace("_PyGCMS.csv", "")
                            if s_id not in sample_map:
                                sample_map[s_id] = thermal_core.ThermalSample(sample_id=s_id)
                            sample_map[s_id].py_gcms = thermal_core.parse_py_gcms_file(b_content, sample_id=s_id)
                        elif "_TGA" in fn:
                            s_id = fn.replace("_TGA.csv", "")
                            if s_id not in sample_map:
                                sample_map[s_id] = thermal_core.ThermalSample(sample_id=s_id)
                            sample_map[s_id].tga = thermal_core.parse_tga_file(b_content, sample_id=s_id)
                    st.session_state["thermal_samples"] = sample_map
                else:
                    synth_samples, _ = thermal_core.generate_synthetic_thermal_dataset(n_samples_per_class=6, save_dir=demo_th_dir)
                    st.session_state["thermal_samples"] = synth_samples

    if not st.session_state["thermal_samples"]:
        st.info(T[lang]["thermal_no_samples"])
    else:
        # Сборка матрицы дескрипторов по всем образцам
        th_rows = []
        for s_id, s_obj in st.session_state["thermal_samples"].items():
            row = {"Sample_ID": s_id}
            row.update(thermal_core.extract_thermal_descriptors(s_obj))
            row["Class_Target"] = 1 if any(k in str(s_id).lower() for k in ["lignin", "sludge", "шлам"]) else 0
            th_rows.append(row)
        df_thermal_desc = pd.DataFrame(th_rows)

        sample_list = list(st.session_state["thermal_samples"].keys())
        col_sel1, col_sel2 = st.columns([3, 1.5])
        with col_sel1:
            active_s_id = st.selectbox(
                T[lang]["thermal_active_sample"],
                sample_list,
                index=0,
                key="select_active_thermal_sample"
            )
        active_sample = st.session_state["thermal_samples"][active_s_id]
        has_py = active_sample.py_gcms is not None
        has_tga = active_sample.tga is not None

        with col_sel2:
            st.markdown(
                f"<div style='padding-top: 28px;'>"
                f"<span style='background: {'#27ae60' if has_py else '#7f8c8d'}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;'>Py-GC-MS: {'✓' if has_py else '✗'}</span>"
                f"<span style='background: {'#c0392b' if has_tga else '#7f8c8d'}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;'>TGA: {'✓' if has_tga else '✗'}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        tab_th1, tab_th2, tab_th3, tab_th4 = st.tabs([
            T[lang]["thermal_tab1_title"],
            T[lang]["thermal_tab2_title"],
            T[lang]["thermal_tab3_title"],
            T[lang]["thermal_tab4_title"],
        ])

        # ----------------------------------------------------------------------
        # ТАБ 1: Py-GC-MS Хроматограммы и мономеры
        # ----------------------------------------------------------------------
        with tab_th1:
            if not has_py:
                st.warning("Для активного образца отсутствуют данные Py-GC-MS. Загрузите соответствующий файл или используйте эталонный бенчмарк." if lang == "ru" else "No Py-GC-MS data available for the active sample.")
            else:
                py_data = active_sample.py_gcms
                st.write(f"#### {T[lang]['thermal_tab1_title']} — **{active_s_id}**")

                fig_tic = go.Figure()
                if len(py_data.retention_times) > 0 and len(py_data.tic_intensity) > 0:
                    fig_tic.add_trace(go.Scatter(
                        x=py_data.retention_times,
                        y=py_data.tic_intensity,
                        mode="lines",
                        line=dict(color="#1A365D", width=1.5),
                        name="TIC (Raw Signal)",
                    ))

                class_palette = {
                    "Guaiacyl": "#e67e22",
                    "Syringyl": "#8e44ad",
                    "p-Hydroxyphenyl": "#2980b9",
                    "Carbohydrate": "#27ae60",
                    "Nitrogen": "#c0392b",
                    "Aliphatic": "#f39c12",
                    "Other": "#7f8c8d"
                }

                if py_data.peaks:
                    total_peak_area = sum(getattr(p, "area", getattr(p, "peak_area", 0.0)) for p in py_data.peaks)
                    for p in py_data.peaks:
                        p_cls = getattr(p, "compound_class", getattr(p, "chemical_class", "Other"))
                        p_color = class_palette.get(p_cls, "#7f8c8d")
                        p_name = (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak")
                        p_rt = getattr(p, "rt", getattr(p, "retention_time", 0.0))
                        p_area = getattr(p, "area", getattr(p, "peak_area", 0.0))
                        p_area_pct = getattr(p, "area_pct", 0.0)
                        if p_area_pct <= 0.0 and total_peak_area > 0:
                            p_area_pct = (p_area / total_peak_area) * 100.0
                        p_monomer = getattr(p, "monomer_type", p_cls)
                        p_mw = getattr(p, "mw", 0.0)
                        p_formula = getattr(p, "formula", "") or "—"

                        fig_tic.add_trace(go.Scatter(
                            x=[p_rt],
                            y=[p_area if (len(py_data.tic_intensity) == 0) else np.interp(p_rt, py_data.retention_times, py_data.tic_intensity)],
                            mode="markers",
                            marker=dict(size=9, color=p_color, symbol="diamond", line=dict(width=1, color="black")),
                            name=p_cls,
                            legendgroup=p_cls,
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{p_name}</b><br>"
                                f"Класс: {p_cls} ({p_monomer})<br>"
                                f"RT: {p_rt:.2f} мин | Площадь: {p_area_pct:.2f}%<br>"
                                f"Формула: {p_formula} (MW {p_mw:.1f})"
                                if lang == "ru" else
                                f"<b>{p_name}</b><br>"
                                f"Class: {p_cls} ({p_monomer})<br>"
                                f"RT: {p_rt:.2f} min | Area: {p_area_pct:.2f}%<br>"
                                f"Formula: {p_formula} (MW {p_mw:.1f})"
                            )
                        ))

                fig_tic.update_layout(
                    xaxis_title="Retention Time (min)" if lang == "en" else "Время удерживания RT (мин)",
                    yaxis_title="Total Ion Current (TIC)" if lang == "en" else "Интенсивность ионного тока (TIC)",
                    template="plotly_white",
                    height=440,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st_plotly(fig_tic)

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.write(f"**{T[lang]['thermal_dist_pie_title']}**")
                    class_areas = {
                        "Guaiacyl (G)": py_data.indices.get("Py_G_pct", 0.0),
                        "Syringyl (S)": py_data.indices.get("Py_S_pct", 0.0),
                        "p-Hydroxyphenyl (H)": py_data.indices.get("Py_H_pct", 0.0),
                        "Carbohydrate": py_data.indices.get("Py_Carbohydrate_Area_pct", py_data.indices.get("Py_Carbohydrate_pct", 0.0)),
                        "Nitrogen / Protein": py_data.indices.get("Py_Nitrogen_Area_pct", py_data.indices.get("Py_Nitrogen_pct", 0.0)),
                        "Aliphatic / Lipid": py_data.indices.get("Py_Aliphatic_Area_pct", 0.0),
                        "Unassigned": py_data.indices.get("Py_Unassigned_Area_pct", 0.0),
                    }
                    class_areas_filtered = {k: v for k, v in class_areas.items() if v > 0.01}
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(class_areas_filtered.keys()),
                        values=list(class_areas_filtered.values()),
                        hole=0.45,
                        marker=dict(colors=[
                            class_palette.get(k.split()[0], "#7f8c8d") for k in class_areas_filtered.keys()
                        ])
                    )])
                    fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
                    st_plotly(fig_pie)

                with col_p2:
                    st.write(f"**{T[lang]['thermal_monomers_bar_title']}**")
                    if py_data.peaks:
                        def _get_pk_pct(pk):
                            pct = getattr(pk, "area_pct", 0.0)
                            if pct <= 0.0 and total_peak_area > 0:
                                pct = (getattr(pk, "area", getattr(pk, "peak_area", 0.0)) / total_peak_area) * 100.0
                            return pct

                        top_peaks = sorted(py_data.peaks, key=_get_pk_pct, reverse=True)[:10]
                        df_top = pd.DataFrame([
                            {
                                "Compound": (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak"),
                                "Class": getattr(p, "compound_class", getattr(p, "chemical_class", "Other")),
                                "Area_%": round(_get_pk_pct(p), 2)
                            }
                            for p in top_peaks
                        ])
                        fig_bar = px.bar(
                            df_top,
                            x="Area_%",
                            y="Compound",
                            orientation="h",
                            color="Class",
                            color_discrete_map=class_palette,
                            text=df_top["Area_%"].apply(lambda v: f"{v:.1f}%")
                        )
                        fig_bar.update_layout(
                            yaxis=dict(autorange="reversed"),
                            xaxis_title="Relative Area (%)" if lang == "en" else "Относительная площадь (%)",
                            template="plotly_white",
                            height=360,
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        st_plotly(fig_bar)

                with st.expander(f"📋 {T[lang]['thermal_peaks_hdr']} ({len(py_data.peaks)} пиков)", expanded=False):
                    if py_data.peaks:
                        peaks_table = []
                        for p in py_data.peaks:
                            c_name = (getattr(p, "name_ru", None) if lang == "ru" else getattr(p, "name_en", None)) or getattr(p, "compound_name", "Peak")
                            c_class = getattr(p, "compound_class", getattr(p, "chemical_class", "Other"))
                            c_rt = getattr(p, "rt", getattr(p, "retention_time", 0.0))
                            c_pct = getattr(p, "area_pct", 0.0)
                            if c_pct <= 0.0 and total_peak_area > 0:
                                c_pct = (getattr(p, "area", getattr(p, "peak_area", 0.0)) / total_peak_area) * 100.0
                            c_monomer = getattr(p, "monomer_type", c_class)
                            c_mw = getattr(p, "mw", 0.0)
                            c_formula = getattr(p, "formula", "") or "—"
                            c_cas = getattr(p, "cas", "") or "—"
                            peaks_table.append({
                                "RT (min)": round(c_rt, 2),
                                "Compound": c_name,
                                "Class": c_class,
                                "Monomer": c_monomer,
                                "Area (%)": round(c_pct, 2),
                                "Formula": c_formula,
                                "MW": round(float(c_mw), 1) if c_mw > 0 else np.nan,
                                "CAS": c_cas,
                            })
                        df_p_tbl = pd.DataFrame(peaks_table)
                        st_df(df_p_tbl)
                        st.download_button(
                            label=T[lang]["thermal_peaks_dl_btn"],
                            data=df_p_tbl.to_csv(index=False).encode("utf-8"),
                            file_name=f"{active_s_id}_PyGCMS_Peaks.csv",
                            mime="text/csv",
                            key=f"dl_peaks_{active_s_id}"
                        )

        # ----------------------------------------------------------------------
        # ТАБ 2: TGA / DTG Термограммы и температурные зоны
        # ----------------------------------------------------------------------
        with tab_th2:
            if not has_tga:
                st.warning("Для активного образца отсутствуют данные TGA/DTG. Загрузите файл термогравиметрии или выберите эталонный бенчмарк." if lang == "ru" else "No TGA/DTG data available for the active sample.")
            else:
                tga_data = active_sample.tga
                st.write(f"#### {T[lang]['thermal_tga_title']} — **{active_s_id}**")

                col_to1, col_to2 = st.columns(2)
                show_zones = col_to1.checkbox(T[lang]["thermal_tga_zones_chk"], value=True, key="chk_tga_zones")
                show_dtg = col_to2.checkbox(T[lang]["thermal_tga_dtg_chk"], value=True, key="chk_tga_dtg")

                fig_tga = make_subplots(specs=[[{"secondary_y": True}]])
                fig_tga.add_trace(
                    go.Scatter(
                        x=tga_data.temperature,
                        y=tga_data.mass_pct,
                        mode="lines",
                        line=dict(color="#C0392B", width=2.5),
                        name="TGA Mass (%)",
                    ),
                    secondary_y=False
                )

                if show_dtg:
                    fig_tga.add_trace(
                        go.Scatter(
                            x=tga_data.temperature,
                            y=tga_data.dtg,
                            mode="lines",
                            line=dict(color="#2980B9", width=2.0),
                            name="DTG (-dm/dT, %/°C)",
                        ),
                        secondary_y=True
                    )

                if show_zones:
                    fig_tga.add_vrect(x0=30, x1=150, fillcolor="#3498DB", opacity=0.10, layer="below", line_width=0,
                                      annotation_text="Zone I: Moisture" if lang == "en" else "Зона I: Влага (<150°C)", annotation_position="top left")
                    fig_tga.add_vrect(x0=150, x1=350, fillcolor="#F1C40F", opacity=0.12, layer="below", line_width=0,
                                      annotation_text="Zone II: Aliphatic" if lang == "en" else "Зона II: Алифатика (150–350°C)", annotation_position="top left")
                    fig_tga.add_vrect(x0=350, x1=600, fillcolor="#E74C3C", opacity=0.10, layer="below", line_width=0,
                                      annotation_text="Zone III: Lignin Core" if lang == "en" else "Зона III: Ядро лигнина (350–600°C)", annotation_position="top left")
                    fig_tga.add_vrect(x0=600, x1=800, fillcolor="#95A5A6", opacity=0.12, layer="below", line_width=0,
                                      annotation_text="Zone IV: Char" if lang == "en" else "Зона IV: Кокс (>600°C)", annotation_position="top left")

                fig_tga.update_xaxes(title_text="Temperature (°C)" if lang == "en" else "Температура (°C)")
                fig_tga.update_yaxes(title_text="TGA Mass Loss (%)" if lang == "en" else "Масса образца TGA (%)", secondary_y=False)
                fig_tga.update_yaxes(title_text="DTG (-dm/dT, %/°C)" if lang == "en" else "Скорость деструкции DTG (-dm/dT, %/°C)", secondary_y=True)
                fig_tga.update_layout(template="plotly_white", height=450, margin=dict(l=40, r=40, t=30, b=40))
                st_plotly(fig_tga)

                # Метрики зон деструкции
                st.markdown("##### " + ("Кинетические зоны термодеструкции" if lang == "ru" else "Thermal Decomposition Kinetic Zones"))
                col_zm1, col_zm2, col_zm3, col_zm4 = st.columns(4)
                col_zm1.metric("Зона I (<150°C)" if lang == "ru" else "Zone I (<150°C)", f"{tga_data.indices.get('TGA_Zone_I_pct', 0.0):.1f}%", help="Влага и десорбция")
                col_zm2.metric("Зона II (150–350°C)" if lang == "ru" else "Zone II (150-350°C)", f"{tga_data.indices.get('TGA_Zone_II_pct', 0.0):.1f}%", help="Гемицеллюлозы и алифатика")
                col_zm3.metric("Зона III (350–600°C)" if lang == "ru" else "Zone III (350-600°C)", f"{tga_data.indices.get('TGA_Zone_III_pct', 0.0):.1f}%", help="Ароматический каркас лигнина")
                col_zm4.metric("Зона IV (>600°C)" if lang == "ru" else "Zone IV (>600°C)", f"{tga_data.indices.get('TGA_Zone_IV_pct', 0.0):.1f}%", help="Кокс и минеральный остаток")

                col_tm1, col_tm2, col_tm3, col_tm4 = st.columns(4)
                col_tm1.metric(T[lang]["thermal_kpi_ti"], f"{tga_data.indices.get('TGA_Thermal_Index_TI', 0.0):.2f}", help="Δm(III) / Δm(II)")
                col_tm2.metric(T[lang]["thermal_kpi_t50"], f"{tga_data.indices.get('TGA_T50_degC', 0.0):.1f} °C", help="Температура 50% убыли массы")
                col_tm3.metric("T_max (DTG)", f"{tga_data.indices.get('DTG_Tmax_degC', 0.0):.1f} °C", help="Пик максимальной скорости деструкции")
                col_tm4.metric(T[lang]["thermal_kpi_residue"], f"{tga_data.indices.get('TGA_Residue_800C_pct', 0.0):.1f}%", help="Неразложившийся остаток при 800°C")

        # ----------------------------------------------------------------------
        # ТАБ 3: Хемометрические индексы (S/G, L/C, TI)
        # ----------------------------------------------------------------------
        with tab_th3:
            st.write(f"#### {T[lang]['thermal_tab3_title']}")

            col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
            sg_val = active_sample.py_gcms.indices.get("Py_S_G_ratio", 0.0) if has_py else 0.0
            hg_val = active_sample.py_gcms.indices.get("Py_H_G_ratio", 0.0) if has_py else 0.0
            lc_val = active_sample.py_gcms.indices.get("Py_L_C_ratio", 0.0) if has_py else 0.0
            ox_val = (active_sample.py_gcms.indices.get("Py_Acid_Aldehyde_G", active_sample.py_gcms.indices.get("Py_Ac_Al_G", 0.0))) if has_py else 0.0
            ti_val = (active_sample.tga.indices.get("TGA_Thermal_Index_TI", active_sample.tga.indices.get("TGA_Thermostability_Index", 0.0))) if has_tga else 0.0
            t50_val = (active_sample.tga.indices.get("TGA_T50_degC", active_sample.tga.indices.get("TGA_T_50pct", 0.0))) if has_tga else 0.0

            col_k1.metric("S/G Ratio", f"{sg_val:.3f}" if has_py else "—")
            col_k2.metric("H/G Ratio", f"{hg_val:.3f}" if has_py else "—")
            col_k3.metric("L/C Ratio", f"{lc_val:.2f}" if has_py else "—")
            col_k4.metric("(Ac/Al)_G", f"{ox_val:.2f}" if has_py else "—")
            col_k5.metric("TI (TGA)", f"{ti_val:.2f}" if has_tga else "—")
            col_k6.metric("T_50% (TGA)", f"{t50_val:.1f} °C" if has_tga else "—")

            st.markdown("---")
            st.write("##### " + ("Сводная таблица термических дескрипторов по всем пробам сессии" if lang == "ru" else "Summary Table of Thermal Descriptors Across All Samples"))
            st_df(df_thermal_desc)

            # Радар-диаграмма сравнения активного образца со средней картиной
            if len(df_thermal_desc) > 1 and has_py and has_tga:
                st.write("##### " + ("Мультипараметрический профиль образца (Fingerprint Radar)" if lang == "ru" else "Sample Thermal Fingerprint Radar"))
                radar_metrics = ["Py_S_G_ratio", "Py_L_C_ratio", "Py_Ac_Al_G", "TGA_Thermal_Index_TI", "Py_Total_Lignin_pct"]
                labels = ["S/G Ratio", "L/C Ratio", "(Ac/Al)_G", "Thermal Index TI", "Lignin Area (%)"]

                vals_sample = []
                vals_mean = []
                for m in radar_metrics:
                    m_lookup = m
                    if m not in df_thermal_desc.columns:
                        if m == "Py_Ac_Al_G" and "Py_Acid_Aldehyde_G" in df_thermal_desc.columns:
                            m_lookup = "Py_Acid_Aldehyde_G"
                        elif m == "TGA_Thermal_Index_TI" and "TGA_Thermostability_Index" in df_thermal_desc.columns:
                            m_lookup = "TGA_Thermostability_Index"
                        elif m == "Py_Total_Lignin_pct" and "Py_Total_Lignin_Area_pct" in df_thermal_desc.columns:
                            m_lookup = "Py_Total_Lignin_Area_pct"
                    col_vals = df_thermal_desc[m_lookup] if m_lookup in df_thermal_desc.columns else pd.Series([0.0])
                    v_min, v_max = col_vals.min(), col_vals.max()
                    denom = (v_max - v_min) if (v_max - v_min) > 1e-6 else 1.0
                    val_from_sample = (
                        active_sample.py_gcms.indices.get(m, active_sample.py_gcms.indices.get(m_lookup, 0.0))
                        if active_sample.py_gcms else
                        active_sample.tga.indices.get(m, active_sample.tga.indices.get(m_lookup, 0.0))
                    )
                    v_s = (val_from_sample - v_min) / denom
                    v_m = (col_vals.mean() - v_min) / denom
                    vals_sample.append(float(np.clip(v_s, 0.0, 1.0)))
                    vals_mean.append(float(np.clip(v_m, 0.0, 1.0)))

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_sample + [vals_sample[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    name=active_s_id,
                    line_color="#c0392b"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_mean + [vals_mean[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    name="Mean (All Samples)" if lang == "en" else "Среднее (Все пробы)",
                    line_color="#2980b9",
                    opacity=0.5
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=420,
                    margin=dict(l=40, r=40, t=30, b=30)
                )
                st_plotly(fig_radar)

        # ----------------------------------------------------------------------
        # ТАБ 4: Мультимодальная интеграция и экспорт
        # ----------------------------------------------------------------------
        with tab_th4:
            st.write(f"#### {T[lang]['thermal_tab4_title']}")

            tga_col = "TGA_Thermal_Index_TI" if "TGA_Thermal_Index_TI" in df_thermal_desc.columns else ("TGA_Thermostability_Index" if "TGA_Thermostability_Index" in df_thermal_desc.columns else None)
            lig_col = "Py_Total_Lignin_Area_pct" if "Py_Total_Lignin_Area_pct" in df_thermal_desc.columns else ("Py_Total_Lignin_pct" if "Py_Total_Lignin_pct" in df_thermal_desc.columns else None)

            if len(df_thermal_desc) > 1 and tga_col and "Py_L_C_ratio" in df_thermal_desc.columns:
                df_plot = df_thermal_desc.copy()
                if lig_col:
                    df_plot["_size"] = df_plot[lig_col].apply(lambda v: max(float(v), 5.0) if pd.notnull(v) else 10.0)
                else:
                    df_plot["_size"] = 10.0
                fig_scatter = px.scatter(
                    df_plot,
                    x=tga_col,
                    y="Py_L_C_ratio",
                    text="Sample_ID" if "Sample_ID" in df_plot.columns else None,
                    size="_size",
                    color="Class_Target" if "Class_Target" in df_plot.columns else None,
                    color_continuous_scale="Reds",
                    title="Thermal Stability (TI) vs Lignin/Carbohydrate Ratio (L/C)" if lang == "en" else "Термический индекс (TI) vs Лигнин/Углеводы (L/C)",
                    labels={
                        tga_col: "Thermal Index TI (Zone III / Zone II)",
                        "Py_L_C_ratio": "Lignin / Carbohydrate Ratio (L/C)",
                        "Class_Target": "Class"
                    }
                )
                fig_scatter.update_layout(height=420, template="plotly_white")
                st_plotly(fig_scatter)

            c_d1, c_d2, c_d3 = st.columns([1.5, 1.5, 2.5])
            with c_d1:
                st.download_button(
                    label=T[lang]["thermal_dl_csv"],
                    data=df_thermal_desc.to_csv(index=False).encode("utf-8"),
                    file_name="ChemoSuite_Thermal_Descriptors.csv",
                    mime="text/csv",
                    key="dl_thermal_descriptors_csv",
                )
            with c_d2:
                excel_th_buf = io.BytesIO()
                with pd.ExcelWriter(excel_th_buf, engine="openpyxl") as writer:
                    df_thermal_desc.to_excel(writer, index=False, sheet_name="Thermal_Descriptors")
                st.download_button(
                    label=T[lang]["thermal_dl_excel"],
                    data=excel_th_buf.getvalue(),
                    file_name="ChemoSuite_Thermal_Descriptors.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_thermal_descriptors_excel",
                )
            with c_d3:
                if st.button(T[lang]["thermal_sync_ml_btn"], type="primary", key="btn_sync_thermal_ml_descriptors"):
                    st.session_state["thermal_ml_descriptors"] = df_thermal_desc
                    st.toast(T[lang]["thermal_sync_success"].format(n=len(df_thermal_desc)), icon="✅")
                    st.rerun()
