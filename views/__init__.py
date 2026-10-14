"""
views/ — Modular GUI view components for ChemoSuite Pro studios.
"""
from views.elemental_view import render_elemental_studio
from views.sec_view import render_sec_studio
from views.thermal_view import render_thermal_studio
from views.nmr_view import render_nmr_studio
from views.ftir_view import render_ftir_studio
from views.chemo_ml_view import render_chemo_ml_studio
from views.eem_view import render_eem_studio
from views.fticr_view import render_fticr_studio

__all__ = [
    "render_elemental_studio",
    "render_sec_studio",
    "render_thermal_studio",
    "render_nmr_studio",
    "render_ftir_studio",
    "render_chemo_ml_studio",
    "render_eem_studio",
    "render_fticr_studio",
]
