"""
test_ic.py — Unit tests for Ion Chromatography (IC) core module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.ic_core import (
    ICSample,
    ION_SPECIES,
    parse_ic_table,
    generate_synthetic_ic_dataset,
    calculate_ic_descriptors,
)


def test_ic_sample_charge_balance_and_metrics():
    # Synthetic balanced water: 1 mmol/L NaCl (Na+ = 23 mg/L, Cl- = 35.5 mg/L)
    s = ICSample(
        sample_id="NaCl_Standard",
        na_plus=22.99,
        cl_minus=35.45,
        so4_2minus=0.0,
        alkalinity_meq_l=0.0,  # carbonate-free standard: alkalinity stated explicitly
    ).compute_metrics()

    # Molar charges are equal (1 meq/L Na+, 1 meq/L Cl-) -> CBE ~ 0.0%
    assert abs(s.charge_balance_error_pct) < 0.5
    # Ionic strength for 1 mmol/L NaCl = 0.5 * (1*1^2 + 1*1^2) = 1.0 mmol/L
    assert abs(s.ionic_strength_mmol_l - 1.0) < 0.1
    assert s.total_mineral_anions_mg_l > 35.0
    assert s.total_cations_mg_l > 22.0


def test_ic_kraft_effluent_indicator():
    # Kraft pulp mill leachate: elevated SO4 (>100 mg/L), high Na+, elevated acetate
    kraft_sample = ICSample(
        sample_id="BPPM_Leachate_Test",
        so4_2minus=150.0,
        cl_minus=10.0,
        acetate=25.0,
        formate=12.0,
        na_plus=85.0,
        ca_2plus=25.0,
    ).compute_metrics()

    assert kraft_sample.so4_to_cl_ratio == 15.0
    assert kraft_sample.total_lmwoa_mg_l >= 37.0
    assert kraft_sample.kraft_effluent_indicator >= 0.7


def test_ic_synthetic_dataset_and_parsing():
    df = generate_synthetic_ic_dataset()
    assert len(df) == 12
    assert "SO4" in df.columns and "Cl" in df.columns and "Acetate" in df.columns

    # Parse DataFrame
    samples = parse_ic_table(df)
    assert len(samples) == 12
    for s in samples:
        assert s.sample_id != ""
        assert s.total_mineral_anions_mg_l > 0

    # Parse CSV string
    csv_str = df.to_csv(index=False)
    samples_from_csv = parse_ic_table(csv_str)
    assert len(samples_from_csv) == 12

    # Parse TSV bytes
    tsv_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    samples_from_tsv = parse_ic_table(tsv_bytes)
    assert len(samples_from_tsv) == 12


def test_calculate_ic_descriptors():
    df = generate_synthetic_ic_dataset()
    samples = parse_ic_table(df)
    desc_df = calculate_ic_descriptors(samples)

    assert isinstance(desc_df, pd.DataFrame)
    assert len(desc_df) == 12
    assert "so4_mg_l" in desc_df.columns
    assert "so4_cl_ratio" in desc_df.columns
    assert "total_lmwoa" in desc_df.columns
    assert "kraft_effluent_indicator" in desc_df.columns
    assert desc_df.index.name == "sample_id"
