"""
test_elemental.py — Unit and integration tests for Elemental Analysis (CHNS-O) module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.elemental_core import (
    ElementalSample,
    parse_elemental_table,
    generate_synthetic_elemental_dataset,
    cross_validate_fticr_elemental,
)


def test_elemental_sample_stoichiometry():
    # Эталонная гуминовая кислота: C=52.0%, H=5.0%, O=39.0%, N=2.0%, S=1.0%
    s = ElementalSample(
        sample_id="Test_HA",
        c_pct=52.0,
        h_pct=5.0,
        n_pct=2.0,
        s_pct=1.0,
        o_pct=39.0,
        ash_pct=0.0,
        moisture_pct=0.0,
        sample_type="Humic",
    )
    s.compute_stoichiometry()

    # H/C ~ (5.0/1.008) / (52.0/12.011) = 4.9603 / 4.3294 ~ 1.146
    assert 1.10 <= s.h_c <= 1.20
    # O/C ~ (39.0/15.999) / 4.3294 = 2.4377 / 4.3294 ~ 0.563
    assert 0.52 <= s.o_c <= 0.60
    # NOSC = - (H - 3*N - 2*O - 2*S) / C
    assert isinstance(s.nosc, float)
    assert -1.5 <= s.nosc <= 1.5
    # AI_bulk
    assert 0.0 <= s.aromaticity_index <= 1.0


def test_synthetic_dataset_generation_and_parsing():
    df = generate_synthetic_elemental_dataset()
    assert len(df) == 12
    assert "C" in df.columns and "H" in df.columns and "O" in df.columns

    samples = parse_elemental_table(df)
    assert len(samples) == 12
    for s in samples:
        assert s.h_c > 0.0
        assert s.o_c > 0.0
        assert s.sample_id != ""


def test_cross_validate_fticr_elemental():
    s = ElementalSample(
        sample_id="Baikal_Water",
        c_pct=50.0,
        h_pct=5.0,
        n_pct=2.0,
        s_pct=0.5,
        o_pct=42.5,
    ).compute_stoichiometry()

    # Синтетический датафрейм приписанных формул FT-ICR MS
    assigned_df = pd.DataFrame({
        "Formula": ["C20H24O10", "C22H26O11", "C25H30O12"],
        "C": [20, 22, 25],
        "H": [24, 26, 30],
        "O": [10, 11, 12],
        "N": [0, 0, 0],
        "S": [0, 0, 0],
        "norm_intensity": [100.0, 80.0, 50.0],
    })

    res = cross_validate_fticr_elemental(assigned_df, s)
    assert res["status"] == "success"
    assert "concordance_score" in res
    assert 0.0 <= res["concordance_score"] <= 100.0
    assert "differences" in res
