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
)


def test_raman_spectrum_deconvolution_and_metrics():
    # Synthetic spectrum with clear D (~1350) and G (~1585) bands
    wn = np.linspace(1000.0, 1800.0, 401)
    d_band = 80.0 * np.exp(-((wn - 1350.0) ** 2) / (2 * (40.0 ** 2)))
    g_band = 80.0 * np.exp(-((wn - 1585.0) ** 2) / (2 * (30.0 ** 2)))
    ints = d_band + g_band + 10.0  # constant baseline

    spec = RamanSpectrum(
        sample_id="Synthetic_Carbon",
        wavenumbers=wn,
        intensities=ints,
    ).process()

    assert spec.corrected_intensities is not None
    assert 1335.0 <= spec.d_pos <= 1365.0
    assert 1570.0 <= spec.g_pos <= 1600.0
    assert 0.8 <= spec.id_ig_ratio <= 1.2
    # Tuinstra-Koenig relation: La = 4.4 / (ID/IG)
    expected_la = 4.4 / spec.id_ig_ratio
    assert abs(spec.la_crystallite_nm - expected_la) < 0.2
    assert spec.fwhm_g > 10.0
    assert 0.0 <= spec.carbon_order_index <= 1.0
    assert "D1_band" in spec.components
    assert "G_band" in spec.components


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
    assert desc_df.index.name == "sample_id"
