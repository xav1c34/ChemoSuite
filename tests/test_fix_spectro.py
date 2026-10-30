"""
test_fix_spectro.py — regression tests for the spectroscopy fixes:
JCAMP-DX 4.24 parser (open_spectrometry_formats, shared by nmr_core), mzML/mzXML metadata and
compression, 13C NMR descriptors, FTIR band ratios / deconvolution / %T handling, Raman
five-band fit and crystallite size. Every test fails on the code before the fix.
"""
import base64
import struct
import zlib

import numpy as np
import pandas as pd
import pytest

from modules import ftir_core, nmr_core, raman_core
from modules import open_spectrometry_formats as osf


# ==============================================================================
# JCAMP-DX fixtures (McDonald & Wilks 1988): the same 8-point spectrum in every data form
# x = 1000..1007 cm^-1, y = 0.010..0.017 (raw ordinates 10..17, YFACTOR = 0.001)
# ==============================================================================
_JCAMP_HEADER = """##TITLE=fixture
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XFACTOR=1.0
##YFACTOR=0.001
##FIRSTX=1000
##LASTX=1007
##DELTAX=1
##NPOINTS=8
##FIRSTY=0.010
##XYDATA=(X++(Y..Y))
"""
_EXPECTED_X = np.arange(1000.0, 1008.0)
_EXPECTED_Y = np.arange(10, 18) * 0.001

JCAMP_BODIES = {
    "AFFN": "1000 10 11 12 13\n1004 14 15 16 17\n",
    "SQZ": "1000A0A1A2A3\n1004A4A5A6A7\n",
    # DIF with Y-check: line 2 repeats the last ordinate of line 1 (13) before continuing
    "DIF": "1000A0JJJ\n1003A3JJJJ\n",
    # DUP after DIF repeats the difference; DUP digits S..Z,s = 1..9 occurrences (U = 3, V = 4)
    "DIFDUP": "1000A0JU\n1003A3JV\n",
}


@pytest.mark.parametrize("form", sorted(JCAMP_BODIES))
def test_jcamp_xpp_data_forms_decode_identically(form):
    res = osf.parse_jcamp_dx(_JCAMP_HEADER + JCAMP_BODIES[form] + "##END=\n")
    np.testing.assert_allclose(res["x"], _EXPECTED_X)
    np.testing.assert_allclose(res["y"], _EXPECTED_Y)


def test_jcamp_per_point_x_uses_line_abscissa_and_xfactor_once():
    # Real x = raw·XFACTOR; FIRSTX/LASTX/DELTAX are already in real units (not multiplied again)
    text = """##TITLE=xfactor
##JCAMP-DX=4.24
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XFACTOR=0.5
##YFACTOR=0.001
##FIRSTX=500
##LASTX=503.5
##NPOINTS=8
##XYDATA=(X++(Y..Y))
1000 10 11 12 13
1004 14 15 16 17
##END=
"""
    res = osf.parse_jcamp_dx(text)
    np.testing.assert_allclose(res["x"], 500.0 + 0.5 * np.arange(8))
    assert not res["warnings"]
    np.testing.assert_allclose(res["y"], _EXPECTED_Y)


def test_jcamp_xy_pairs_peak_table_and_bytes_input():
    text = b"""##TITLE=ms
##JCAMP-DX=4.24
##DATA TYPE=MASS SPECTRUM
##XUNITS=M/Z
##YUNITS=RELATIVE ABUNDANCE
##XFACTOR=0.01
##YFACTOR=2
##NPOINTS=3
##PEAK TABLE=(XY..XY)
10050,10 20025,20
30000,5
##END=
"""
    res = osf.parse_jcamp_dx(text)  # bytes accepted
    np.testing.assert_allclose(res["x"], [100.5, 200.25, 300.0])
    np.testing.assert_allclose(res["y"], [20.0, 40.0, 10.0])
    assert osf.jcamp_is_mass_spectrum(res)


_NMR_JCAMP = """##TITLE=13C fixture
##JCAMP-DX=5.01
##DATA TYPE=NMR SPECTRUM
##.OBSERVE FREQUENCY=100.0
##.OBSERVE NUCLEUS=^13C
##XUNITS=HZ
##YUNITS=ARBITRARY UNITS
##XFACTOR=1.0
##YFACTOR=1.0
##FIRSTX=20000
##LASTX=0
##DELTAX=-2500
##NPOINTS=9
{extra}##XYDATA=(X++(Y..Y))
20000A0JJJ
12500A3jjj
5000A0jj
##END=
"""


def test_jcamp_nmr_hz_to_ppm_through_nmr_core():
    sample = nmr_core.parse_nmr_file(_NMR_JCAMP.format(extra=""), sample_id="c13", nucleus="13C")
    np.testing.assert_allclose(sample.chemical_shifts, np.arange(0.0, 201.0, 25.0))  # Hz / 100 MHz
    np.testing.assert_allclose(sample.intensity, [8, 9, 10, 11, 12, 13, 12, 11, 10])


def test_jcamp_nmr_shift_reference_applied():
    text = _NMR_JCAMP.format(extra="##.SHIFT REFERENCE=(INTERNAL, TMS, 9, 0.5)\n")
    shifts, _, _ = nmr_core.parse_jcamp_dx(text, sample_id="ref")
    np.testing.assert_allclose(shifts, np.arange(0.0, 201.0, 25.0) + 0.5)


def test_jcamp_nmr_hz_without_frequency_raises():
    text = _NMR_JCAMP.format(extra="").replace("##.OBSERVE FREQUENCY=100.0\n", "")
    with pytest.raises(ValueError):
        nmr_core.parse_nmr_file(text, sample_id="nofreq")


# ==============================================================================
# mzML / mzXML
# ==============================================================================
def _mzxml(peaks_b64: str, compression: str = "zlib", extra_attrs: str = 'polarity="-" retentionTime="PT12.5S"') -> bytes:
    return f"""<?xml version="1.0"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"><msRun>
<scan num="1" msLevel="1" {extra_attrs} peaksCount="3">
<peaks precision="32" byteOrder="network" pairOrder="m/z-int" compressionType="{compression}">{peaks_b64}</peaks>
</scan></msRun></mzXML>""".encode()


def test_mzxml_zlib_peaks_and_scan_metadata():
    raw = struct.pack(">6f", 100.0, 1.0, 200.0, 2.0, 300.0, 3.0)
    res = osf.parse_mzml_content(_mzxml(base64.b64encode(zlib.compress(raw)).decode()))
    np.testing.assert_allclose(res["mz"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(res["intensity"], [1.0, 2.0, 3.0])
    assert res["metadata"]["polarity"] == "negative"
    assert res["metadata"]["retention_time_sec"] == pytest.approx(12.5)


def _mzml(mz_vals, int_vals, cv_extra: str = "") -> bytes:
    def arr(vals, acc, name):
        b64 = base64.b64encode(struct.pack(f"<{len(vals)}f", *vals)).decode()
        return (f'<binaryDataArray><cvParam accession="MS:1000521" name="32-bit float"/>'
                f'<cvParam accession="{acc}" name="{name}"/><binary>{b64}</binary></binaryDataArray>')
    return f"""<?xml version="1.0"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml"><run><spectrumList count="1">
<spectrum id="s1" index="0">{cv_extra}<binaryDataArrayList count="2">
{arr(mz_vals, "MS:1000514", "m/z array")}{arr(int_vals, "MS:1000515", "intensity array")}
</binaryDataArrayList></spectrum></spectrumList></run></mzML>""".encode()


def test_mzml_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        osf.parse_mzml_content(_mzml([100.0, 200.0, 300.0], [1.0, 2.0]))


def test_mzml_polarity_and_rt_from_cvparams_or_none():
    cv = ('<cvParam accession="MS:1000129" name="negative scan"/>'
          '<scanList><scan><cvParam accession="MS:1000016" name="scan start time" value="2.5" '
          'unitAccession="UO:0000031" unitName="minute"/></scan></scanList>')
    meta = osf.parse_mzml_content(_mzml([100.0, 200.0], [1.0, 2.0], cv))["metadata"]
    assert meta["polarity"] == "negative"
    assert meta["retention_time_sec"] == pytest.approx(150.0)
    meta_none = osf.parse_mzml_content(_mzml([100.0, 200.0], [1.0, 2.0]))["metadata"]
    assert meta_none["polarity"] is None and meta_none["retention_time_sec"] is None


def test_mass_spectrum_to_peak_df_for_fticr_upload():
    df = osf.mass_spectrum_to_peak_df([300.0, 100.0, -1.0, np.nan, 200.0], [1.0, 4.0, 5.0, 6.0, 2.0])
    assert list(df.columns) == ["mass", "intensity", "norm_intensity"]
    np.testing.assert_allclose(df["mass"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(df["norm_intensity"], [100.0, 50.0, 25.0])
    with pytest.raises(ValueError):
        osf.mass_spectrum_to_peak_df([1.0, 2.0], [1.0])


# ==============================================================================
# 13C / 1H NMR
# ==============================================================================
def test_nmr_zero_spectrum_gives_nan_and_no_mw_regression():
    ppm = np.linspace(0.0, 220.0, 1101)
    desc = nmr_core.calculate_13c_nmr_descriptors(ppm, np.zeros_like(ppm))
    assert "NMR_estimated_mw" not in desc
    assert np.isnan(desc["NMR_fa_aromaticity"])
    assert np.isnan(desc["NMR_hb_hi_index"])


def test_nmr_acetal_bin_and_six_ihss_bins_sum_to_100():
    ppm = np.linspace(0.0, 220.0, 2201)
    y = (np.exp(-0.5 * ((ppm - 30) / 6) ** 2) + np.exp(-0.5 * ((ppm - 72) / 5) ** 2)
         + 0.6 * np.exp(-0.5 * ((ppm - 103) / 3) ** 2) + 0.8 * np.exp(-0.5 * ((ppm - 128) / 6) ** 2)
         + 0.3 * np.exp(-0.5 * ((ppm - 155) / 4) ** 2) + 0.5 * np.exp(-0.5 * ((ppm - 175) / 4) ** 2))
    desc = nmr_core.calculate_13c_nmr_descriptors(ppm, y)
    assert desc["NMR_C_acetal_pct"] > 5.0  # anomeric C of carbohydrates, 90–110 ppm
    six = nmr_core.ihss_6bin_fractions(desc)
    assert len(six) == 6
    assert sum(six.values()) == pytest.approx(100.0, abs=1e-6)


def test_nmr_dmso_reference_for_1h_is_2_50_ppm():
    # Gottlieb, Kotlyar & Nudelman, J. Org. Chem. 1997, 62, 7512: DMSO-d5 residual 1H = 2.50 ppm
    ppm = np.linspace(0.0, 10.0, 2001)
    y = np.exp(-0.5 * ((ppm - 2.60) / 0.01) ** 2)
    shifted, delta = nmr_core.auto_recalibrate_nmr_shifts(ppm, y, mode="dmso", nucleus="1H")
    assert delta == pytest.approx(-0.10, abs=0.01)
    assert shifted[int(np.argmax(y))] == pytest.approx(2.50, abs=0.01)


def test_nmr_aromatic_deconvolution_has_no_lignin_unit_ratios():
    for n in (3, 4):
        bands = nmr_core.NMR_AROMATIC_SUBBANDS[n]
        for a, b in zip(bands, bands[1:]):
            assert a["hi"] <= b["lo"]  # non-overlapping centre windows
    ppm = np.linspace(100.0, 175.0, 751)
    y = sum(np.exp(-0.5 * ((ppm - c) / 4.0) ** 2) for c in (116.0, 130.0, 147.0, 156.0))
    res = nmr_core.deconvolve_aromatic_region(ppm, y, n_peaks=4)
    assert res["success"] and res["fit_ok"]
    labels = " ".join(c["label"] for c in res["components"]).lower()
    assert "syringyl" not in labels and "guaiacyl" not in labels
    assert not any("s_g" in k.lower() or "sg_" in k.lower() or "syring" in k.lower() for k in res)


# ==============================================================================
# FTIR
# ==============================================================================
def _ftir_sample(bands, wn=None, offset=0.05):
    wn = np.linspace(400.0, 4000.0, 1801) if wn is None else wn
    a = offset + sum(h * np.exp(-0.5 * ((wn - c) / s) ** 2) for h, c, s in bands)
    return ftir_core.FTIRSample(sample_id="t", wavenumbers=wn, absorbance=a, raw_signal=a.copy())


def test_ftir_batch_indices_ignore_snv():
    bands = [(0.30, 1600, 20), (0.20, 1510, 12), (0.25, 2920, 15), (0.15, 1720, 15), (0.12, 1030, 25)]
    s_snv, s_ref = _ftir_sample(bands), _ftir_sample(bands)
    df = ftir_core.batch_process_ftir_samples([s_snv], norm_method="snv")
    ftir_core.subtract_baseline(s_ref, method="als")
    ref = ftir_core.calculate_ftir_indices(s_ref.wavenumbers, s_ref.corrected_absorbance, sample_id="t")
    assert df.iloc[0]["FTIR_Lignin_Idx"] == pytest.approx(ref["FTIR_Lignin_Idx"])
    # SNV would distort the ratio of two bands; the true height ratio is 0.20/0.30
    assert df.iloc[0]["FTIR_Lignin_Idx"] == pytest.approx(0.2 / 0.3, rel=0.1)


def test_ftir_gs_ratio_nan_when_1330_band_absent():
    wn = np.linspace(400.0, 4000.0, 1801)
    a = sum(h * np.exp(-0.5 * ((wn - c) / s) ** 2) for h, c, s in [(0.3, 1600, 20), (0.2, 1510, 12), (0.25, 2920, 15), (0.2, 1270, 15)])
    ind = ftir_core.calculate_ftir_indices(wn, a)
    assert np.isnan(ind["FTIR_GS_Ratio"])
    assert np.isfinite(ind["FTIR_Lignin_Idx"])


def _lignin_like_spectrum(wn, s_height, noise=0.004, seed=0):
    # сильная G-полоса 1270 (σ=22) рядом с 1330: ее хвост дает в окне 1330 ~10 % ее высоты
    bands = [(0.65, 1602, 25), (0.58, 1510, 16), (0.22, 1455, 20), (0.42, 1270, 22), (0.45, 1220, 25)]
    if s_height:
        bands.append((s_height, 1330, 15))
    a = 0.1 + sum(h * np.exp(-0.5 * ((wn - c) / s) ** 2) for h, c, s in bands)
    return a + np.random.default_rng(seed).normal(0.0, noise, size=wn.size)


@pytest.mark.parametrize("seed", range(5))
def test_ftir_gs_ratio_ignores_tail_of_neighbouring_band(seed):
    wn = np.arange(650.0, 4000.0, 2.0)
    ind = ftir_core.calculate_ftir_indices(wn, _lignin_like_spectrum(wn, s_height=0.0, seed=seed))
    assert np.isnan(ind["FTIR_GS_Ratio"])


@pytest.mark.parametrize("seed", range(5))
def test_ftir_gs_ratio_detects_s_shoulder_on_g_band_flank(seed):
    wn = np.arange(650.0, 4000.0, 2.0)
    ind = ftir_core.calculate_ftir_indices(wn, _lignin_like_spectrum(wn, s_height=0.15, seed=seed))
    assert np.isfinite(ind["FTIR_GS_Ratio"])
    assert 1.0 < ind["FTIR_GS_Ratio"] < 4.0


def test_ftir_deconvolution_failure_is_flagged(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")
    monkeypatch.setattr(ftir_core, "curve_fit", boom)
    wn = np.linspace(1400.0, 1900.0, 501)
    a = 0.5 * np.exp(-0.5 * ((wn - 1600) / 20) ** 2)
    res = ftir_core.deconvolve_carbonyl_region(wn, a)
    assert res["success"] is False
    assert np.isnan(res["r2"]) and np.all(np.isnan(res["fitted"]))
    assert res["peaks_df"]["Area"].isna().all()
    assert res["individual_curves"] == []


def _ftir_text(values, header=None):
    wn = np.linspace(4000.0, 400.0, 61)
    rows = [f"{w:.1f},{v:.6f}" for w, v in zip(wn, values)]
    return ("\n".join(([header] if header else []) + rows)).encode(), wn


def test_ftir_percent_t_not_converted_without_label():
    vals = np.linspace(20.0, 95.0, 61)
    content, wn = _ftir_text(vals)
    s = ftir_core.parse_ftir_file(content, sample_id="noheader")
    order = np.argsort(wn)
    np.testing.assert_allclose(s.absorbance, vals[order])  # kept as absorbance, not shifted
    assert s.signal_type == "absorbance"
    assert s.metadata.get("warnings")


def test_ftir_labelled_transmittance_converted_without_min_shift():
    vals = np.linspace(20.0, 104.0, 61)  # %T > 100 (reference drift) → A < 0 must stay negative
    content, wn = _ftir_text(vals, header="Wavenumber,%T")
    s = ftir_core.parse_ftir_file(content, sample_id="pct")
    order = np.argsort(wn)
    np.testing.assert_allclose(s.absorbance, 2.0 - np.log10(vals[order]), atol=1e-9)
    assert s.signal_type == "transmittance" and s.absorbance.min() < 0


def test_ftir_counts_scale_data_not_treated_as_percent_t():
    vals = np.linspace(1000.0, 30000.0, 61)
    content, wn = _ftir_text(vals)
    s = ftir_core.parse_ftir_file(content, sample_id="counts")
    np.testing.assert_allclose(s.absorbance, vals[np.argsort(wn)])


def test_ftir_1510_label_is_aromatic_skeletal_not_guaiacyl():
    band = [b for b in ftir_core.CHARACTERISTIC_IR_BANDS if b["min_wn"] <= 1510.0 <= b["max_wn"]]
    assert band and all("guaiacyl" not in b["name"].lower() for b in band)
    assert any("aromatic skeletal" in b["name"] for b in band)


def test_ftir_library_marked_synthetic():
    wn = np.linspace(400.0, 4000.0, 1801)
    a = 0.3 * np.exp(-0.5 * ((wn - 1600) / 20) ** 2)
    df = ftir_core.match_spectrum_against_library(wn, a, top_n=3)
    assert "Library_Type" in df.columns
    assert df["Library_Type"].str.contains("Synthetic").all()


# ==============================================================================
# Raman
# ==============================================================================
def _sadezky_spectrum(wn, d1=(80.0, 1350.0, 110.0), g=(80.0, 1585.0, 60.0), background=None):
    y = (raman_core.lorentzian_band(wn, 10.0, 1200.0, 150.0) + raman_core.lorentzian_band(wn, *d1)
         + raman_core.gaussian_band(wn, 15.0, 1500.0, 120.0) + raman_core.lorentzian_band(wn, *g)
         + raman_core.lorentzian_band(wn, 12.0, 1620.0, 35.0))
    return y + (20.0 + 0.01 * (wn - 800.0) if background is None else background)


def _five_band_carbon(wn, width_scale, seed):
    y = (raman_core.lorentzian_band(wn, 25.0, 1200.0, 180.0 * width_scale)
         + raman_core.lorentzian_band(wn, 100.0, 1350.0, 140.0 * width_scale)
         + raman_core.gaussian_band(wn, 30.0, 1510.0, 140.0 * width_scale)
         + raman_core.lorentzian_band(wn, 90.0, 1585.0, 70.0 * width_scale)
         + raman_core.lorentzian_band(wn, 25.0, 1620.0, 40.0 * width_scale))
    return y + np.random.default_rng(seed).normal(0.0, 1.5, wn.size)


def test_raman_ratio_standard_error_matches_noise_spread():
    # предсказанная по ковариации погрешность I_D/I_G согласуется с разбросом по реализациям шума
    wn = np.arange(800.0, 2000.0, 2.0)
    specs = [raman_core.RamanSpectrum("s", wn, _five_band_carbon(wn, 1.0, seed)).process() for seed in range(15)]
    ratios = np.array([s.id_ig_ratio for s in specs])
    empirical = ratios.std() / ratios.mean()
    predicted = float(np.median([s.id_ig_rel_se for s in specs]))
    assert 0.5 * empirical < predicted < 2.0 * empirical
    assert all(s.fit_ok for s in specs)


def test_raman_broad_bands_flagged_as_ill_conditioned():
    wn = np.arange(800.0, 2000.0, 2.0)
    spec = raman_core.RamanSpectrum("broad", wn, _five_band_carbon(wn, 1.6, seed=0)).process()
    assert spec.fit_success
    assert spec.id_ig_rel_se > raman_core.RAMAN_RATIO_MAX_REL_SE
    assert not spec.fit_ok
    assert any("poorly constrained" in w for w in spec.fit_warnings)
    desc = raman_core.calculate_raman_descriptors([spec])
    assert "id_ig_rel_se" in raman_core.RAMAN_QC_COLUMNS
    assert desc.loc["broad", "id_ig_rel_se"] == pytest.approx(spec.id_ig_rel_se)


def test_raman_zero_height_bands_do_not_inflate_ratio_error():
    # только D1 и G: пустые D2/D3/D4 на нулевой границе не должны раздувать погрешность D1/G
    wn = np.arange(800.0, 2000.0, 2.0)
    y = (raman_core.lorentzian_band(wn, 100.0, 1360.0, 260.0) + raman_core.lorentzian_band(wn, 90.0, 1580.0, 140.0)
         + np.random.default_rng(0).normal(0.0, 1.5, wn.size))
    spec = raman_core.RamanSpectrum("two", wn, y).process()
    assert spec.fit_ok
    assert spec.id_ig_rel_se < 0.1
    assert spec.id_ig_ratio == pytest.approx(100.0 / 90.0, rel=0.1)


def test_raman_fit_failure_gives_nan_and_flag(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")
    monkeypatch.setattr(raman_core, "curve_fit", boom)
    wn = np.linspace(800.0, 2000.0, 601)
    spec = raman_core.RamanSpectrum("fail", wn, _sadezky_spectrum(wn)).process()
    assert spec.fit_success is False
    for attr in ("id_ig_ratio", "ad_ag_ratio", "la_crystallite_nm", "fwhm_g", "carbon_order_index"):
        assert np.isnan(getattr(spec, attr))
    desc = raman_core.calculate_raman_descriptors([spec])
    assert np.isnan(desc.loc["fail", "id_ig_ratio"]) and not bool(desc.loc["fail", "fit_success"])


def test_raman_five_band_sadezky_fit_recovers_ratio():
    wn = np.linspace(800.0, 2000.0, 601)
    spec = raman_core.RamanSpectrum("five", wn, _sadezky_spectrum(wn)).process(baseline_method="asls")
    assert spec.fit_success and spec.fit_ok
    for key in ("D1_band", "D2_band", "D3_amorphous", "D4_band", "G_band"):
        assert key in spec.components
    assert set(spec.band_params) == {"D1", "D2", "D3", "D4", "G"}
    assert spec.id_ig_ratio == pytest.approx(1.0, abs=0.1)


def test_raman_la_cancado_depends_on_laser_wavelength():
    # Cançado et al., Appl. Phys. Lett. 2006, 88, 163106: L_a = 2.4e-10·λ⁴/(A_D/A_G)
    wn = np.linspace(800.0, 2000.0, 601)
    spec = raman_core.RamanSpectrum("la", wn, _sadezky_spectrum(wn)).process()
    la_532 = spec.estimate_crystallite_size(532.0)
    assert la_532 == pytest.approx(2.4e-10 * 532.0 ** 4 / spec.ad_ag_ratio)
    la_514 = spec.estimate_crystallite_size(514.5)
    assert la_514 / la_532 == pytest.approx((514.5 / 532.0) ** 4)
    assert la_532 != pytest.approx(4.4 / spec.id_ig_ratio, rel=0.05)  # not the old 4.4/(I_D/I_G)


def test_raman_ferrari_robertson_used_below_2nm():
    spec = raman_core.RamanSpectrum("fr", np.array([0.0]), np.array([0.0]))
    spec.fit_success = True
    spec.ad_ag_ratio = 15.0   # Cançado → 2.4e-10·532⁴/15 ≈ 1.3 nm < 2 nm: outside TK validity
    spec.id_ig_ratio = 0.5
    la = spec.estimate_crystallite_size(514.5)
    assert la == pytest.approx(np.sqrt(0.5 / 0.0055) / 10.0)  # Ferrari & Robertson 2000
    assert "Ferrari" in spec.la_method


def test_raman_single_column_file_rejected():
    with pytest.raises(ValueError):
        raman_core.parse_raman_spectrum("\n".join(str(v) for v in np.linspace(10, 100, 50)), sample_id="one")


def test_raman_order_index_graphite_vs_disordered():
    wn = np.linspace(800.0, 2000.0, 1201)
    graphite = raman_core.lorentzian_band(wn, 100.0, 1582.0, 16.0) + raman_core.lorentzian_band(wn, 5.0, 1350.0, 40.0) + 5.0
    g = raman_core.RamanSpectrum("graphite", wn, graphite).process()
    assert g.fit_success and g.carbon_order_index > 0.9
    disordered = raman_core.RamanSpectrum("dis", wn, _sadezky_spectrum(wn, g=(80.0, 1590.0, 120.0))).process()
    assert disordered.fit_success and disordered.carbon_order_index < 0.35


def test_raman_asls_baseline_handles_curved_background():
    wn = np.linspace(800.0, 2000.0, 601)
    curved = 40.0 + 2e-4 * (wn - 1400.0) ** 2   # fluorescence-like curvature, 40 → 112 at the edges
    spec = raman_core.RamanSpectrum("curved", wn, _sadezky_spectrum(wn, background=curved)).process(baseline_method="asls")
    assert spec.fit_success and spec.fit_ok
    assert spec.id_ig_ratio == pytest.approx(1.0, abs=0.2)
