"""
test_raman.py — Unit tests for Raman Spectroscopy core module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.raman_core import (
    RamanSpectrum,
    parse_raman_spectrum,
    generate_synthetic_raman_dataset,
    calculate_raman_descriptors,
    lorentzian_band,
    gaussian_band,
    CANCADO_COEFF_NM3,
)


def test_raman_spectrum_deconvolution_and_metrics():
    # Synthetic five-band (Sadezky 2005) spectrum: D1 and G of equal height (true I_D/I_G = 1)
    wn = np.linspace(800.0, 2000.0, 601)
    ints = (
        lorentzian_band(wn, 10.0, 1200.0, 150.0)
        + lorentzian_band(wn, 80.0, 1350.0, 110.0)
        + gaussian_band(wn, 15.0, 1500.0, 120.0)
        + lorentzian_band(wn, 80.0, 1585.0, 60.0)
        + lorentzian_band(wn, 12.0, 1620.0, 35.0)
        + 20.0 + 0.01 * (wn - 800.0)  # sloped background
    )

    spec = RamanSpectrum(
        sample_id="Synthetic_Carbon",
        wavenumbers=wn,
        intensities=ints,
    ).process()

    assert spec.fit_success and spec.fit_ok
    assert spec.corrected_intensities is not None
    assert 1335.0 <= spec.d_pos <= 1365.0
    assert 1570.0 <= spec.g_pos <= 1600.0
    assert 0.8 <= spec.id_ig_ratio <= 1.2
    # Cançado et al. 2006: L_a = 2.4e-10 * lambda^4 / (A_D/A_G), default lambda = 532 nm
    expected_la = CANCADO_COEFF_NM3 * 532.0 ** 4 / spec.ad_ag_ratio
    assert abs(spec.la_crystallite_nm - expected_la) < 1e-9
    assert "Cançado" in spec.la_method
    assert spec.fwhm_g > 10.0
    assert 0.0 <= spec.carbon_order_index <= 1.0
    for key in ("D1_band", "D2_band", "D3_amorphous", "D4_band", "G_band", "Cumulative_Fit"):
        assert key in spec.components
        assert len(spec.components[key]) == len(spec.analysis_wavenumbers)


def test_raman_benchmark_generation_and_parsing():
    spectra = generate_synthetic_raman_dataset()
    assert len(spectra) >= 7

    # Check Kraft sludge char vs pristine humic
    char_samples = [s for s in spectra.values() if s.sample_type == "Kraft_Sludge_Char"]
    humic_samples = [s for s in spectra.values() if s.sample_type == "Pristine_NOM"]

    assert len(char_samples) >= 1
    assert len(humic_samples) >= 1

    # Sludge hydrochar should have higher ID/IG than amorphous humics
    assert char_samples[0].id_ig_ratio > humic_samples[0].id_ig_ratio

    # Test parsing from DataFrame
    sample = list(spectra.values())[0]
    df = pd.DataFrame({"wavenumber": sample.wavenumbers, "intensity": sample.intensities})
    parsed_spec = parse_raman_spectrum(df, sample_id="Parsed_Test")
    assert parsed_spec.sample_id == "Parsed_Test"
    assert parsed_spec.id_ig_ratio > 0.0


def test_calculate_raman_descriptors():
    spectra = generate_synthetic_raman_dataset()
    desc_df = calculate_raman_descriptors(list(spectra.values()))

    assert isinstance(desc_df, pd.DataFrame)
    assert len(desc_df) == len(spectra)
    assert "id_ig_ratio" in desc_df.columns
    assert "la_crystallite_nm" in desc_df.columns
    assert "carbon_order_index" in desc_df.columns
    assert "fit_success" in desc_df.columns and bool(desc_df["fit_success"].all())
    assert desc_df.index.name == "sample_id"
