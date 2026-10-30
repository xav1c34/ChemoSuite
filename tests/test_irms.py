"""
test_irms.py — Unit tests for Isotope Ratio Mass Spectrometry (IRMS) core module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.irms_core import (
    IRMSSample,
    STANDARDS,
    ENDMEMBERS,
    parse_irms_table,
    generate_synthetic_irms_dataset,
    calculate_irms_descriptors,
    calculate_two_source_mixing,
)


def test_irms_sample_metrics():
    # Organic matter (NOM): delta2H/delta18O of solids are not on the meteoric water line,
    # so the Dansgaard d-excess is not defined -> NaN plus a note (previously -243.6 was reported).
    sample = IRMSSample(
        sample_id="Baikal_Pelagic",
        d13C=-30.5,
        d15N=6.2,
        d34S=0.8,
        d2H=-154.0,
        d18O=11.2,
        c_n_atomic=11.5,
        sample_type="Pristine",
    ).compute_metrics()

    assert np.isnan(sample.deuterium_excess)
    assert any("d-excess" in n for n in sample.notes)
    # Pristine background should have low Kraft probability
    assert sample.kraft_lignin_prob < 0.2

    # Water sample: Dansgaard deuterium excess d = d2H - 8 * d18O (Dansgaard 1964)
    water = IRMSSample(
        sample_id="Baikal_Water",
        d13C=float("nan"),
        d15N=float("nan"),
        d2H=-123.0,
        d18O=-15.8,
        sample_type="Lake_Water",
    ).compute_metrics()
    assert abs(water.deuterium_excess - (-123.0 - 8.0 * -15.8)) < 1e-9


def test_kraft_lignin_probability_detection():
    # Heavy sulfur and less depleted carbon from Kraft cooking liquor
    sludge_sample = IRMSSample(
        sample_id="BPPM_Core_01",
        d13C=-27.1,
        d15N=2.5,
        d34S=7.5,
        d2H=-130.0,
        d18O=16.0,
    ).compute_metrics()

    assert sludge_sample.kraft_lignin_prob >= 0.7


def test_parse_irms_table_and_synthetic_dataset():
    df = generate_synthetic_irms_dataset()
    assert len(df) == 12
    assert "d13C" in df.columns and "d15N" in df.columns and "d34S" in df.columns

    # Parse DataFrame
    samples = parse_irms_table(df)
    assert len(samples) == 12
    assert samples[0].sample_id.startswith("BPPM")
    assert samples[0].d34S is not None

    # Parse CSV string
    csv_str = df.to_csv(index=False)
    samples_from_csv = parse_irms_table(csv_str)
    assert len(samples_from_csv) == 12

    # Parse TSV bytes
    tsv_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    samples_from_tsv = parse_irms_table(tsv_bytes)
    assert len(samples_from_tsv) == 12


def test_calculate_irms_descriptors():
    df = generate_synthetic_irms_dataset()
    samples = parse_irms_table(df)
    desc_df = calculate_irms_descriptors(samples)

    assert isinstance(desc_df, pd.DataFrame)
    assert len(desc_df) == 12
    assert "d13C" in desc_df.columns
    assert "d34S" in desc_df.columns
    assert "kraft_lignin_prob" in desc_df.columns
    assert desc_df.index.name == "sample_id"


def test_two_source_mixing():
    # Endmember A (Kraft): d34S = 7.0, Endmember B (Baikal): d34S = 1.0
    # Sample at d34S = 4.0 should be 50% Kraft
    f_mid = calculate_two_source_mixing(4.0, 7.0, 1.0)
    assert abs(f_mid - 0.50) < 1e-4

    # 100% Kraft
    f_high = calculate_two_source_mixing(7.0, 7.0, 1.0)
    assert abs(f_high - 1.0) < 1e-4

    # Beyond boundary clipped to 0.0 or 1.0
    f_under = calculate_two_source_mixing(0.0, 7.0, 1.0)
    assert f_under == 0.0
    f_over = calculate_two_source_mixing(9.0, 7.0, 1.0)
    assert f_over == 1.0
