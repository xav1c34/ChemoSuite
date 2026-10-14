"""
views/common.py — Shared GUI utilities and chart wrappers for ChemoSuite Pro.
"""

import io
import os
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st


def st_df(data, **kwargs):
    """Render modern styled DataFrame with automatic responsive column sizing."""
    if isinstance(data, pd.DataFrame):
        col_cfg = kwargs.get("column_config")
        if col_cfg is None:
            col_cfg = {}
        else:
            col_cfg = dict(col_cfg)
        for col in data.columns:
            if col not in col_cfg:
                col_s = str(col).lower()
                if any(k in col_s for k in ["origin", "генезис", "эколог", "desc", "описан", "status", "статус", "name", "назван", "smiles", "pathway", "trans"]):
                    col_cfg[col] = st.column_config.TextColumn(width="large")
                elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                    max_len = data[col].astype(str).str.len().max() if len(data) > 0 else 0
                    if max_len > 35:
                        col_cfg[col] = st.column_config.TextColumn(width="large")
                    elif max_len > 15:
                        col_cfg[col] = st.column_config.TextColumn(width="medium")
        if col_cfg:
            kwargs["column_config"] = col_cfg
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)


def st_plotly(fig, **kwargs):
    """Render interactive Plotly chart with modern instrument ergonomics."""
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    try:
        return st.plotly_chart(fig, width="stretch", config=config, **kwargs)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, config=config, **kwargs)


def get_plot_download_buttons(fig, base_filename: str, lang: str = "ru"):
    """Export Matplotlib or Plotly figure as PNG / SVG download buttons."""
    col1, col2, _ = st.columns([1.8, 1.8, 4.4])
    label_png = "📥 Скачать PNG (300 DPI)" if lang == "ru" else "📥 Download PNG (300 DPI)"
    label_svg = "📐 Скачать SVG (Вектор)" if lang == "ru" else "📐 Download SVG (Vector)"

    with col1:
        png_buf = io.BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label=label_png,
            data=png_buf.getvalue(),
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"dl_png_{base_filename}",
        )
    with col2:
        svg_buf = io.BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        st.download_button(
            label=label_svg,
            data=svg_buf.getvalue(),
            file_name=f"{base_filename}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{base_filename}",
        )


def ensure_sample_peaks(sample: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Гарантирует наличие валидного DataFrame пиков (parsed_peaks) у образца."""
    if sample is None:
        return None
    if sample.get("parsed_peaks") is not None and not sample["parsed_peaks"].empty:
        return sample["parsed_peaks"]

    b_content = sample.get("file_bytes")
    if not b_content and sample.get("raw_path") and os.path.exists(sample["raw_path"]):
        try:
            with open(sample["raw_path"], "rb") as f:
                b_content = f.read()
                sample["file_bytes"] = b_content
        except Exception:
            pass

    if not b_content:
        return None

    try:
        from modules import fticr_core
        raw_df = fticr_core.parse_uploaded_file(b_content, delimiter="Auto", decimal_sep=".", has_header=True)
        if raw_df is None or raw_df.empty:
            return None
        sample["raw_df"] = raw_df
        col_names = list(raw_df.columns)
        mz_col = col_names[0]
        int_col = col_names[1] if len(col_names) > 1 else col_names[0]
        for idx, cname in enumerate(col_names):
            cn_low = str(cname).lower()
            if any(k in cn_low for k in ["m/z", "mass", "mz", "m.z"]):
                mz_col = cname
            elif any(k in cn_low for k in ["int", "i", "count", "abund"]):
                int_col = cname
        clean_m = pd.to_numeric(raw_df[mz_col], errors="coerce")
        clean_i = pd.to_numeric(raw_df[int_col], errors="coerce")
        valid_mask = clean_m.notna() & clean_i.notna()
        v_df = pd.DataFrame({"mass": clean_m[valid_mask].astype(float), "intensity": clean_i[valid_mask].astype(float)})
        v_df = v_df[v_df["mass"] > 0].sort_values("mass").reset_index(drop=True)
        if not v_df.empty:
            v_df["norm_intensity"] = (v_df["intensity"] / max(1e-12, float(v_df["intensity"].max()))) * 100.0
            sample["parsed_peaks"] = v_df
            return v_df
    except Exception:
        pass
    return sample.get("parsed_peaks")
