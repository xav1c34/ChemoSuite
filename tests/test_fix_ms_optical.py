"""
tests/test_fix_ms_optical.py — Регрессионные тесты исправлений модулей MS / оптики / элементного анализа:
chemo_pubchem, structure_elucidation, fticr_core, eem_core, elemental_core.

Сеть не используется: requests.get подменяется фейковыми ответами, кэш PubChem — во временной папке.
"""
import io
import math
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
import requests

import chemo_pubchem
import structure_elucidation as se
import fticr_core
import eem_core
import elemental_core


# ==============================================================================
# Общие фикстуры: офлайн PubChem
# ==============================================================================
_PUBCHEM_MODULES = []
for _m in (chemo_pubchem, se.chemo_pubchem):
    if all(_m is not x for x in _PUBCHEM_MODULES):
        _PUBCHEM_MODULES.append(_m)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeGet:
    """Подмена requests.get: возвращает заданные ответы и записывает вызовы."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        result = self.responder(url, params or {})
        if isinstance(result, Exception):
            raise result
        return result


def _props_payload(props):
    return {"PropertyTable": {"Properties": props}}


_VANILLIN_PROPS = {
    "CID": 1183, "MolecularFormula": "C8H8O3", "MolecularWeight": "152.15",
    "MonoisotopicMass": "152.047344113", "SMILES": "COC1=C(C=CC(=C1)C=O)O",
    "IUPACName": "4-hydroxy-3-methoxybenzaldehyde", "Title": "Vanillin",
}
_ISOVANILLIN_PROPS = {
    "CID": 12127, "MolecularFormula": "C8H8O3", "MolecularWeight": "152.15",
    "MonoisotopicMass": "152.047344113", "SMILES": "COC1=C(C=C(C=C1)C=O)O",
    "IUPACName": "3-hydroxy-4-methoxybenzaldehyde", "Title": "Isovanillin",
}
_MEPARABEN_PROPS = {
    "CID": 7456, "MolecularFormula": "C8H8O3", "MolecularWeight": "152.15",
    "MonoisotopicMass": "152.047344113", "SMILES": "COC(=O)C1=CC=C(C=C1)O",
    "IUPACName": "methyl 4-hydroxybenzoate", "Title": "Methylparaben",
}


@pytest.fixture(autouse=True)
def _isolated_pubchem(tmp_path, monkeypatch):
    for i, m in enumerate(_PUBCHEM_MODULES):
        monkeypatch.setattr(m, "CACHE_FILE", str(tmp_path / f"pubchem_cache_{i}.json"))
        m.clear_cache()

    def _no_network(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network disabled in tests")

    monkeypatch.setattr(requests, "get", _no_network)
    yield
    for m in _PUBCHEM_MODULES:
        m.clear_cache()


def _install_get(monkeypatch, responder):
    fake = _FakeGet(responder)
    monkeypatch.setattr(requests, "get", fake)
    return fake


# ==============================================================================
# chemo_pubchem
# ==============================================================================
def test_fetch_compounds_by_formula_keys_and_monoisotopic_mass(monkeypatch):
    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload([_VANILLIN_PROPS])))
    res = chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=5)

    assert len(res) == 1
    item = res[0]
    assert set(item) >= {"cid", "name", "iupac_name", "smiles", "formula", "exact_mass", "source"}
    assert item["cid"] == 1183
    assert item["smiles"] == "COC1=C(C=CC(=C1)C=O)O"   # ключ "SMILES" ответа PUG-REST
    assert item["exact_mass"] == pytest.approx(152.047344113)  # MonoisotopicMass, не MolecularWeight
    assert len(fake.calls) == 1
    assert "fastformula/C8H8O3/" in fake.calls[0]["url"]
    assert fake.calls[0]["params"]["MaxRecords"] == 5


def test_pubchem_404_is_zero_hits_and_202_is_error(monkeypatch):
    _install_get(monkeypatch, lambda url, p: _FakeResponse(404, {"Fault": {"Code": "PUGREST.NotFound"}}))
    assert chemo_pubchem.fetch_compounds_by_formula("C50H2O3") == []

    _install_get(monkeypatch, lambda url, p: _FakeResponse(202, {"Waiting": {"ListKey": "123"}}))
    with pytest.raises(chemo_pubchem.PubChemError):
        chemo_pubchem.fetch_compounds_by_formula("C8H8O3", use_cache=False)

    _install_get(monkeypatch, lambda url, p: _FakeResponse(503, None))
    with pytest.raises(chemo_pubchem.PubChemError):
        chemo_pubchem.fetch_compounds_by_formula("C8H8O3", use_cache=False)


def test_pubchem_failures_are_not_cached(monkeypatch):
    _install_get(monkeypatch, lambda url, p: requests.exceptions.ConnectionError("offline"))
    with pytest.raises(chemo_pubchem.PubChemError):
        chemo_pubchem.fetch_compounds_by_formula("C8H8O3")
    res_off = chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=2)
    assert res_off.error  # ошибка сети явно возвращена
    assert all(k.split("|")[0] not in ("fetch", "lookup") for k in chemo_pubchem._get_cache())

    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload([_VANILLIN_PROPS])))
    res_on = chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=2)
    assert len(fake.calls) == 1  # после сбоя запрос реально повторяется
    assert res_on.source == "Online" and res_on.error is None
    assert res_on[0]["Source"] == "Online" and res_on[0]["Offline_Fallback"] is False
    assert res_on[0]["MonoisotopicMass"] == pytest.approx(152.047344113)


def test_pubchem_cache_ttl_expiry(monkeypatch):
    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload([_VANILLIN_PROPS])))
    chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=3)
    cached = chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=3)
    assert len(fake.calls) == 1
    assert cached[0]["source"] == "Cache"

    key = "fetch|C8H8O3|3"
    chemo_pubchem._get_cache()[key]["ts"] -= chemo_pubchem.CACHE_TTL_SECONDS + 60
    chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=3)
    assert len(fake.calls) == 2  # устаревшая запись не используется


def test_pubchem_cache_key_includes_max_records(monkeypatch):
    three = [_VANILLIN_PROPS, _ISOVANILLIN_PROPS, _MEPARABEN_PROPS]
    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload(three)))
    assert len(chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=1)) == 1
    assert len(chemo_pubchem.fetch_compounds_by_formula("C8H8O3", max_records=3)) == 3
    assert len(fake.calls) == 2

    assert len(chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=1)) == 1
    assert len(chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=3)) == 3


def test_offline_reference_is_flagged_fallback():
    # requests.get по умолчанию бросает ConnectionError (фикстура)
    res = chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=4)
    assert isinstance(res, list) and len(res) == 1
    assert res.offline_fallback is True and res.source == "Reference_Offline"
    assert "network disabled" in res.error
    assert res[0]["Source"] == "Reference_Offline" and res[0]["Offline_Fallback"] is True
    assert res[0]["Note"]

    res2 = chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", use_online=False)
    assert res2.offline_fallback is True and res2[0]["CID"] == 1183

    # формулы вне справочника: пустой результат + ошибка, без «эвристической» заглушки
    res3 = chemo_pubchem.lookup_formula_in_pubchem("C33H41O17N1")
    assert len(res3) == 0 and res3.error

    df = chemo_pubchem.annotate_formula_table(["C8H8O3"], max_top=1)
    assert df.iloc[0]["Source"] == "Reference_Offline"
    assert df.iloc[0]["Lookup_Note"]


def test_invalid_formula_rejected_without_request(monkeypatch):
    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload([_VANILLIN_PROPS])))
    for bad in ["H2O/../compound", "c8h8o3", "C10H12Zz3", "CH4;rm", ""]:
        assert not chemo_pubchem.is_valid_formula(chemo_pubchem.clean_formula(bad))
        with pytest.raises(ValueError):
            chemo_pubchem.fetch_compounds_by_formula(bad)
        res = chemo_pubchem.lookup_formula_in_pubchem(bad)
        assert len(res) == 0 and res.error
    assert fake.calls == []
    assert chemo_pubchem.is_valid_formula("C10H12O6S")
    assert not chemo_pubchem.is_valid_formula("C8H8O3/..")


def test_reference_cids_corrected():
    ref = chemo_pubchem.REFERENCE_NOM_STRUCTURES
    expected = {
        "C9H10O3": ("Acetovanillone", 2214),
        "C8H10O3": ("Syringol", 7041),
        "C9H10O5": ("Syringic", 10742),
        "C11H14O4": ("Sinapyl", 5280507),
        "C20H28O2": ("Dehydroabietic", 94391),
        "C8H8O3": ("Vanillin", 1183),
        "C7H8O2": ("Guaiacol", 460),
    }
    for formula, (name_part, cid) in expected.items():
        assert ref[formula]["CID"] == cid, formula
        assert name_part.lower() in ref[formula]["Title"].lower()
    # ошибочная запись «Lignosulfonate model» (CID 9832104 = C44H44N4O9) удалена
    assert "C9H10O7S" not in ref
    assert all(v["CID"] != 9832104 for v in ref.values())


# ==============================================================================
# structure_elucidation
# ==============================================================================
def test_elucidation_no_fake_candidates_on_error_or_empty(monkeypatch):
    cons = se.MultiModalConstraints(formula="C8H8O3", exact_mass=152.0473)
    original_fetch = se.chemo_pubchem.fetch_compounds_by_formula

    def _raise(*a, **k):
        raise se.chemo_pubchem.PubChemError("PubChem unreachable")

    monkeypatch.setattr(se.chemo_pubchem, "fetch_compounds_by_formula", _raise)
    res = se.elucidate_candidates(cons)
    assert isinstance(res, list) and len(res) == 0
    assert res.status == "error" and "unreachable" in res.message
    assert not any("Isomer" in getattr(c, "name", "") for c in res)

    monkeypatch.setattr(se.chemo_pubchem, "fetch_compounds_by_formula", lambda *a, **k: [])
    res = se.elucidate_candidates(cons)
    assert len(res) == 0 and res.status == "no_candidates" and res.message

    monkeypatch.setattr(se.chemo_pubchem, "fetch_compounds_by_formula", original_fetch)
    res = se.elucidate_candidates(se.MultiModalConstraints(formula="H2O/../x"))
    assert len(res) == 0 and res.status == "invalid_formula"


def test_elucidation_uses_pubchem_isomers(monkeypatch):
    three = [_VANILLIN_PROPS, _ISOVANILLIN_PROPS, _MEPARABEN_PROPS]
    fake = _install_get(monkeypatch, lambda url, p: _FakeResponse(200, _props_payload(three)))
    cons = se.MultiModalConstraints(formula="C8H8O3", exact_mass=151.0401, mass_tol_ppm=2.0,
                                    ftir_carboxyl=0, ftir_phenol_oh=1, ftir_carbonyl=1)
    res = se.elucidate_candidates(cons, max_results=5)
    assert res.status == "ok" and len(res) == 3
    assert fake.calls and fake.calls[0]["params"]["MaxRecords"] == 10
    assert {c.cid for c in res} == {1183, 12127, 7456}
    assert all(c.source for c in res)
    assert all(np.isclose(c.exact_mass, 152.047344113) for c in res)


def test_stoichiometry_score_against_targets_and_unset_excluded():
    cand = {"cid": 1183, "name": "Vanillin", "smiles": "COC1=C(C=CC(=C1)C=O)O",
            "formula": "C8H8O3", "exact_mass": 152.0473}
    # ничего не задано: ни одного компонента, MMCS = NaN (а не правдоподобная константа)
    mmcs, br, pf = se.compute_concordance_score(cand, se.MultiModalConstraints(formula="C8H8O3"))
    assert math.isnan(mmcs)
    assert all(math.isnan(v) for v in br.values())
    assert pf["scored_components"] == []

    # H/C = 1.0, O/C = 0.375, NOSC = 4 - (32 + 8 - 6)/8 = -0.25 для C8H8O3
    exact = se.MultiModalConstraints(formula="C8H8O3", target_h_c=1.0, target_o_c=0.375, target_nosc=-0.25)
    mmcs, br, pf = se.compute_concordance_score(cand, exact)
    assert pf["scored_components"] == ["stoich"]
    assert br["Bulk Stoichiometry"] == pytest.approx(100.0)
    assert mmcs == pytest.approx(100.0)

    off = se.MultiModalConstraints(formula="C8H8O3", target_h_c=1.25)  # |Δ| = 0.25 -> 100*(1 - 0.25/0.5)
    mmcs, br, _ = se.compute_concordance_score(cand, off)
    assert br["Bulk Stoichiometry"] == pytest.approx(50.0)
    assert mmcs == pytest.approx(50.0)

    # NMR: только заданные фракции, без штрафа за незаданные
    nmr_only = se.MultiModalConstraints(formula="C8H8O3", nmr_aromatic_pct=0.75)
    mmcs, br, pf = se.compute_concordance_score(cand, nmr_only)
    assert pf["scored_components"] == ["nmr"] and mmcs == pytest.approx(100.0)


def test_kekule_smiles_aromaticity_and_carbonyl_types():
    kek = se.detect_smiles_substructures("COC1=C(C=CC(=C1)C=O)O")     # ванилин (PubChem, Kekulé)
    aro = se.detect_smiles_substructures("COc1cc(C=O)ccc1O")          # ванилин (ароматическая запись)
    for f in (kek, aro):
        assert f["parse_ok"] and f["has_aromatic"] and f["has_phenol_oh"] and f["has_carbonyl"]
        assert f["has_ether"] and not f["has_carboxyl"]
    nmr = se.estimate_13c_nmr_distribution("COC1=C(C=CC(=C1)C=O)O")
    assert nmr["total_carbons"] == 8
    assert nmr["aromatic"] == pytest.approx(6 / 8)
    assert nmr["carboxyl"] == pytest.approx(1 / 8)
    assert nmr["o_alkyl"] == pytest.approx(1 / 8)

    msa = se.detect_smiles_substructures("CS(=O)(=O)O")               # метансульфоновая кислота
    assert not msa["has_carbonyl"] and not msa["has_carboxyl"]
    nitro = se.detect_smiles_substructures("C1=CC=C(C=C1)[N+](=O)[O-]")
    assert nitro["has_aromatic"] and not nitro["has_carbonyl"]
    ester = se.detect_smiles_substructures("CC(=O)OC")
    assert ester["has_carbonyl"] and not ester["has_carboxyl"]
    acid = se.detect_smiles_substructures("CC(=O)O")
    assert acid["has_carboxyl"]
    hexene = se.detect_smiles_substructures("C1=CCCCC1")               # одиночная C=C в цикле
    assert not hexene["has_aromatic"]

    bad = se.detect_smiles_substructures("C1CC(")
    assert bad["parse_ok"] is False
    assert math.isnan(se.estimate_13c_nmr_distribution("C1CC(")["aromatic"])


# ==============================================================================
# fticr_core
# ==============================================================================
def test_ai_koch_dittmar_erratum_and_ai_mod():
    df = pd.DataFrame({
        "C": [10, 7, 6], "H": [10, 6, 12], "O": [4, 2, 6], "N": [1, 0, 0], "S": [0, 0, 0],
    })
    out = fticr_core.calculate_descriptors(df)
    # C10H10O4N1: AI = (1+10-4-0.5*(10+1))/(10-4-1) = 0.3 ; AI_mod = (1+10-2-5.5)/(10-2-1) = 0.5
    assert out.loc[0, "AI"] == pytest.approx(0.3)
    assert out.loc[0, "AI_mod"] == pytest.approx(0.5)
    # C7H6O2 (бензойная кислота): AI = 3/5, AI_mod = 4/6
    assert out.loc[1, "AI"] == pytest.approx(0.6)
    assert out.loc[1, "AI_mod"] == pytest.approx(4 / 6)
    # C6H12O6: числитель < 0 -> 0
    assert out.loc[2, "AI"] == 0.0 and out.loc[2, "AI_mod"] == 0.0


def test_descriptors_without_n_s_columns():
    df = pd.DataFrame({"C": [10, 12], "H": [12, 14], "O": [4, 6]})
    out = fticr_core.calculate_descriptors(df)
    assert list(out["Hetero_Class"]) == ["CHO", "CHO"]
    assert np.all(np.isfinite(out["AI_mod"]))


def test_flux_type_exact_code_matching():
    base = 300.0
    peaks = pd.DataFrame({
        "mass": [base, base + 18.010565, base + 27.994915, base + 15.994915],
        "intensity": [100.0, 90.0, 80.0, 70.0],
    })
    res = fticr_core.compute_geochemical_vector_fluxes(peaks, tol_mda=1.0)
    ft = dict(zip(res["summary_df"]["Transformation"].str.split().str[0], res["summary_df"]["Flux_Type"]))
    assert ft["H2O"] == "Hydration"
    assert ft["CO"] == "Carbonylation"
    assert ft["O"] == "Oxidation"
    assert ft["CO2"] == "Mineralization"
    assert ft["NH3"] == "Amination"
    assert list(res["summary_df"]["Flux_Type"]).count("Oxidation") == 1


def _ion_mass(c, h, o, n=0, s=0, polarity=-1):
    em = fticr_core.EXACT_MASSES
    neutral = c * em["C"] + h * em["H"] + o * em["O"] + n * em["N"] + s * em["S"]
    return neutral + polarity * fticr_core.H_ION_MASS


def test_c13_check_uses_closest_peak():
    m12 = _ion_mass(20, 22, 8)
    exp13 = m12 + fticr_core.C13_DIFF
    tol = exp13 * 2.0e-6
    peaks = pd.DataFrame({
        # первый пик окна 13C (далекий, интенсивность не соответствует), затем ближайший (соответствует 20*1.08 %)
        "mass": [m12, exp13 - 0.9 * tol, exp13 + 0.05 * tol],
        "intensity": [100.0, 1.0, 21.6],
    })
    bounds = {"C": (18, 22), "H": (10, 40), "O": (5, 10), "N": (0, 0), "S": (0, 0)}
    res = fticr_core.fast_formula_assigner(peaks, bounds=bounds, ppm_tolerance=2.0, ion_mode="ESI(-)",
                                           iso_check=True, iso_strict=True)
    row = res[np.isclose(res["mass"], m12)]
    assert len(row) == 1
    assert row.iloc[0]["Formula"] == "C20H22O8"
    assert bool(row.iloc[0]["13C_confirmed"]) is True


def test_intensity_column_token_detection():
    csv = b"m/z,Resolution,S/N,Intensity\n300.10,400000,50,1000\n301.20,390000,40,500\n"
    df = fticr_core.parse_uploaded_file(csv)
    assert df.attrs["intensity_column"] == "Intensity"
    assert list(df["intensity"]) == [1000.0, 500.0]

    csv2 = b"m/z,Resolution,Abundance\n300.10,400000,7\n301.20,390000,9\n"
    df2 = fticr_core.parse_uploaded_file(csv2)
    assert list(df2["intensity"]) == [7.0, 9.0]

    csv3 = b"m/z,Resolution\n300.10,400000\n301.20,390000\n"   # Resolution не берется даже как резерв
    df3 = fticr_core.parse_uploaded_file(csv3)
    assert list(df3["intensity"]) == [100.0, 100.0]
    assert "missing" in df3.attrs["intensity_source"]


def test_unsupported_ion_mode_raises():
    assert fticr_core.ion_mode_polarity("ESI(-)") == -1
    assert fticr_core.ion_mode_polarity("ESI(+)") == 1
    assert fticr_core.ion_mode_polarity("[M-H]⁻") == -1
    assert fticr_core.ion_mode_polarity("Neutral [M]") == 0
    for bad in ["[M+Na]+", "[M+Cl]-", "APPI(+)", "MALDI"]:
        with pytest.raises(ValueError):
            fticr_core.ion_mode_polarity(bad)
    peaks = pd.DataFrame({"mass": [_ion_mass(10, 10, 4)], "intensity": [100.0]})
    with pytest.raises(ValueError):
        fticr_core.fast_formula_assigner(peaks, ion_mode="[M+Na]+")
    with pytest.raises(ValueError):
        fticr_core.get_calibrant_library("FA", "[M+Na]+")
    with pytest.raises(ValueError):
        fticr_core.batch_process_fticr_spectra({"a.csv": b"mass,intensity\n300,1\n"}, ion_mode="[M+Na]+")


def _spectrum_csv(formulas):
    lines = ["mass,intensity"]
    for i, (c, h, o) in enumerate(formulas):
        lines.append(f"{_ion_mass(c, h, o):.6f},{1000 - 10 * i}")
    return ("\n".join(lines) + "\n").encode()


def test_batch_errors_reported_and_duplicate_names_kept():
    good_a = _spectrum_csv([(10, 10, 4), (15, 14, 6), (20, 22, 8)])
    good_b = _spectrum_csv([(12, 12, 5), (16, 16, 7), (18, 20, 9)])
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("siteA/sample.csv", good_a)
        zf.writestr("siteB/sample.csv", good_b)
        zf.writestr("garbage.csv", b"foo,bar\nx,y\n")
        zf.writestr("lowmass.csv", b"mass,intensity\n50.0,100\n60.0,200\n")
    summary, spectra = fticr_core.batch_process_fticr_spectra(bio.getvalue(), ion_mode="ESI(-)")

    assert sorted(summary["Sample_ID"]) == ["siteA__sample", "siteB__sample"]
    assert set(summary.attrs["renamed"]) == {"siteA/sample.csv", "siteB/sample.csv"}
    assert spectra["siteA__sample.csv"]["archive_path"] == "siteA/sample.csv"
    assert "AI_mod" in summary.columns

    errs = {e["file"]: e for e in summary.attrs["errors"]}
    assert errs["garbage.csv"]["stage"] == "parse"
    assert errs["lowmass.csv"]["stage"] == "filter"
    assert all(e["error"] for e in errs.values())


# ==============================================================================
# eem_core
# ==============================================================================
def test_ife_absorbance_divided_by_pathlength():
    ex = np.array([250.0, 300.0, 350.0])
    em = np.array([350.0, 400.0, 450.0])
    eem = np.ones((3, 3))
    uv_wl = np.linspace(200, 500, 301)
    uv_a = np.full(301, 0.2)

    corr1, cf1, _ = eem_core.correct_inner_filter_effect(eem, ex, em, uv_wl, uv_a, pathlength_cm=1.0)
    corr5, cf5, _ = eem_core.correct_inner_filter_effect(eem, ex, em, uv_wl, uv_a, pathlength_cm=5.0)
    assert np.allclose(corr1, 10 ** 0.2)
    assert np.allclose(corr5, 10 ** (0.2 / 5.0))   # A_1cm = A / L
    assert np.allclose(cf5, 0.04)
    with pytest.raises(ValueError):
        eem_core.correct_inner_filter_effect(eem, ex, em, uv_wl, uv_a, pathlength_cm=0.0)


def test_ife_uncovered_wavelengths_masked_and_not_applied():
    ex = np.array([250.0, 300.0, 350.0])
    em = np.array([350.0, 400.0, 450.0])
    eem = np.ones((3, 3)) * 10.0
    uv_wl = np.linspace(200, 420, 221)          # Em 450 не покрыт
    uv_a = np.full(221, 0.1)

    corr, cf, warn = eem_core.correct_inner_filter_effect(eem, ex, em, uv_wl, uv_a)
    assert np.all(np.isnan(corr[2, :])) and np.all(np.isfinite(corr[:2, :]))
    assert warn is not None and "3 ячеек" in warn

    sample = eem_core.EEMSample(sample_id="S1", ex=ex, em=em, data=eem.copy())
    uv = eem_core.UVVisSample(sample_id="S1", wl=uv_wl, absorbance=uv_a, doc=3.0)
    logs = eem_core.link_uv_vis_to_eem([sample], [uv], apply_ife=True)
    assert logs[0]["IFE_Applied"] is False
    assert logs[0]["IFE_Uncovered_Cells"] == 3
    assert logs[0]["IFE_Warning"]
    assert np.array_equal(sample.data, eem)     # частичная коррекция не записана


def test_raman_first_order_band_masked():
    assert float(eem_core.raman_emission_wavelength(350.0)) == pytest.approx(397.3, abs=0.2)
    ex = np.arange(300.0, 405.0, 5.0)
    em = np.arange(300.0, 602.0, 2.0)
    ex_g, em_g = np.meshgrid(ex, em)
    data = np.ones_like(ex_g)
    data[np.abs(em_g - eem_core.raman_emission_wavelength(ex_g)) <= 2.0] += 100.0

    i_em = int(np.argmin(np.abs(em - 397.0)))
    i_ex = int(np.argmin(np.abs(ex - 350.0)))
    masked = eem_core.remove_scatter_bands(data, ex, em)
    unmasked = eem_core.remove_scatter_bands(data, ex, em, mask_raman1=False)
    assert masked[i_em, i_ex] == pytest.approx(1.0, abs=0.05)
    assert unmasked[i_em, i_ex] > 50.0


def test_raman_normalization_window_and_no_silent_factor():
    assert eem_core.RAMAN_EX_NM == 350.0
    assert eem_core.RAMAN_EM_WINDOW_NM == (381.0, 426.0)
    ex = np.arange(250.0, 455.0, 5.0)
    em = np.arange(300.0, 602.0, 2.0)
    eem = np.ones((len(em), len(ex))) * 5.0

    with pytest.warns(RuntimeWarning):
        out, factor = eem_core.normalize_to_raman_units(eem, ex, em)
    assert math.isnan(factor) and out is eem

    blank = np.zeros_like(eem)
    i350 = int(np.argmin(np.abs(ex - 350.0)))
    blank[:, i350] = 2.0 * np.exp(-0.5 * ((em - 397.0) / 8.0) ** 2) + 0.5
    from scipy.integrate import simpson
    win = (em >= 381.0) & (em <= 426.0)
    expected = float(simpson(blank[win, i350], x=em[win]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, factor = eem_core.normalize_to_raman_units(eem, ex, em, water_blank=blank)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert factor == pytest.approx(expected)
    assert np.allclose(out, eem / expected)

    ex_short = np.arange(250.0, 305.0, 5.0)             # нет Ex 350
    with pytest.warns(RuntimeWarning):
        _, f2 = eem_core.normalize_to_raman_units(np.ones((len(em), len(ex_short))), ex_short, em,
                                                  water_blank=np.ones((len(em), len(ex_short))))
    assert math.isnan(f2)


def test_bix_index():
    ex = np.arange(250.0, 455.0, 5.0)
    em = np.arange(300.0, 602.0, 2.0)
    data = np.ones((len(em), len(ex)))
    data[int(np.argmin(np.abs(em - 380.0))), int(np.argmin(np.abs(ex - 310.0)))] = 0.8
    ind = eem_core.calculate_spectral_indices(eem_core.EEMSample("S", ex, em, data))
    assert ind["BIX"] == pytest.approx(0.8)

    ex2 = np.arange(350.0, 455.0, 5.0)                   # нет Ex 310 (и Ex 254) -> NaN, а не значение «соседа»
    ind2 = eem_core.calculate_spectral_indices(eem_core.EEMSample("S", ex2, em, np.ones((len(em), len(ex2)))))
    assert math.isnan(ind2["BIX"]) and math.isnan(ind2["HIX"])


@pytest.mark.skipif(not eem_core.TENSORLY_AVAILABLE, reason="tensorly не установлен")
def test_corcondia_not_clipped():
    rng = np.random.default_rng(0)
    tensor = rng.random((5, 6, 7)) * 100.0
    factors = (rng.random((5, 2)), rng.random((6, 2)), rng.random((7, 2)))
    val = eem_core.compute_corcondia(tensor, factors)
    assert np.isfinite(val) and val < 0.0


@pytest.mark.skipif(not eem_core.TENSORLY_AVAILABLE, reason="tensorly не установлен")
def test_fit_parafac_multiple_starts_keeps_best():
    samples = eem_core.generate_synthetic_chemometrics_dataset(n_samples=5)
    tensor, _, _, _ = eem_core.build_eem_tensor(samples)
    res = eem_core.fit_parafac(tensor, n_components=3, random_state=1, n_starts=2)
    assert len(res["start_sse"]) == 3
    assert res["best_init"].split(":")[0] in ("svd", "random")
    model = np.einsum("ir,jr,kr->ijk", res["scores"], res["em_profiles"], res["ex_profiles"])
    sse = float(np.sum((tensor - model) ** 2))
    assert sse == pytest.approx(np.nanmin(res["start_sse"]), rel=1e-6)

    bad = tensor.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        eem_core.fit_parafac(bad, n_components=2)


def test_split_half_requires_every_component(monkeypatch):
    def _unit(n, k):
        v = np.zeros(n)
        v[k] = 1.0
        return v

    em_a = np.column_stack([_unit(10, 0), _unit(10, 1)])
    ex_a = np.column_stack([_unit(8, 0), _unit(8, 1)])
    t = 0.85
    em_b = np.column_stack([_unit(10, 0), t * _unit(10, 1) + math.sqrt(1 - t * t) * _unit(10, 2)])
    ex_b = np.column_stack([_unit(8, 0), t * _unit(8, 1) + math.sqrt(1 - t * t) * _unit(8, 2)])
    calls = []

    def _fake_fit(tensor, n_components=2, random_state=42, **kw):
        calls.append(random_state)
        em, ex = (em_a, ex_a) if len(calls) == 1 else (em_b, ex_b)
        return {"em_profiles": em, "ex_profiles": ex}

    monkeypatch.setattr(eem_core, "fit_parafac", _fake_fit)
    tensor = np.ones((6, 10, 8))
    res = eem_core.split_half_analysis(tensor, n_components=2)
    assert res["mean_tcc"] >= 0.90                     # старый критерий (среднее) прошел бы
    assert res["component_validated"] == [True, False]
    assert res["min_tcc"] == pytest.approx(0.85, abs=1e-3)
    assert res["is_validated"] is False

    calls.clear()
    res2 = eem_core.split_half_analysis(tensor, n_components=2, tcc_threshold=0.80)
    assert res2["is_validated"] is True

    small = eem_core.split_half_analysis(np.ones((3, 10, 8)), n_components=2)
    assert small["can_split"] is False and math.isnan(small["mean_tcc"])


def test_mw_estimate_nan_outside_calibration():
    assert eem_core.estimate_molecular_weight_uv(5.0) == pytest.approx(3450.0 - 390.0 * 5.0)
    hi = eem_core.MW_UV_E2E3_RANGE[1]
    assert eem_core.estimate_molecular_weight_uv(hi) == pytest.approx(400.0)
    for bad in (8.0, 12.0, 0.0, -1.0, float("nan"), None):
        assert math.isnan(eem_core.estimate_molecular_weight_uv(bad))


def test_reference_library_honest_naming():
    assert eem_core.OPENFLUOR_REFERENCE_LIBRARY is eem_core.BUILTIN_REFERENCE_FLUOROPHORES
    assert eem_core.match_parafac_to_reference_library is eem_core.match_parafac_to_openfluor
    assert all(v["id"].startswith("REF_") for v in eem_core.BUILTIN_REFERENCE_FLUOROPHORES.values())

    em_ax = np.arange(280.0, 600.0, 5.0)
    ex_ax = np.arange(240.0, 450.0, 5.0)
    em_p = np.column_stack([np.exp(-0.5 * ((em_ax - 450) / 40) ** 2), np.exp(-0.5 * ((em_ax - 340) / 25) ** 2)])
    ex_p = np.column_stack([np.exp(-0.5 * ((ex_ax - 330) / 40) ** 2), np.exp(-0.5 * ((ex_ax - 275) / 20) ** 2)])
    df = eem_core.match_parafac_to_reference_library(em_p, ex_p, em_ax, ex_ax, lang="en")
    assert len(df) == 2
    assert "Reference_Library" in df.columns
    assert df["Reference_Library"].str.contains("not the OpenFluor database").all()


# ==============================================================================
# elemental_core
# ==============================================================================
def test_hhv_mendeleev_and_dulong():
    s = elemental_core.ElementalSample("X", c_pct=50.0, h_pct=6.0, n_pct=1.0, s_pct=0.5, o_pct=42.5,
                                       basis="daf").compute_stoichiometry()
    mendeleev = 0.339 * 50 + 1.256 * 6 - 0.1088 * (42.5 - 0.5)
    dulong = 0.3383 * 50 + 1.443 * (6 - 42.5 / 8) + 0.0942 * 0.5
    assert s.hhv_mendeleev_mj_kg == pytest.approx(mendeleev)
    assert s.hhv_dulong_mj_kg == pytest.approx(dulong)
    assert s.combustion_heat_mj_kg == pytest.approx(mendeleev)
    d = s.to_dict()
    assert d["HHV_Dulong_MJ_kg"] == pytest.approx(round(dulong, 2))
    assert d["Q_combustion_MJ_kg"] == pytest.approx(round(mendeleev, 2))


def test_ai_bulk_c_normalised_with_n_s_and_basis_independent():
    aw = elemental_core.ATOMIC_WEIGHTS
    c, h, n, s_, o = 65.0, 4.0, 3.0, 1.0, 27.0
    daf = elemental_core.ElementalSample("D", c, h, n, s_, o, basis="daf").compute_stoichiometry()
    mc = c / aw["C"]
    hc, nc, sc, oc = h / aw["H"] / mc, n / aw["N"] / mc, s_ / aw["S"] / mc, o / aw["O"] / mc
    ai = (1 - oc - sc - 0.5 * (hc + nc)) / (1 - oc - sc - nc)
    ai_mod = (1 - 0.5 * oc - sc - 0.5 * (hc + nc)) / (1 - 0.5 * oc - sc - nc)
    assert daf.aromaticity_index == pytest.approx(ai)
    assert daf.aromaticity_index_mod == pytest.approx(ai_mod)
    assert 0.0 < daf.aromaticity_index < 1.0

    k = 0.85  # тот же состав на влажной зольной массе (зола 10 %, влага 5 %)
    ar = elemental_core.ElementalSample("A", c * k, h * k, n * k, s_ * k, o * k, ash_pct=10.0,
                                        moisture_pct=5.0, basis="as_received").compute_stoichiometry()
    assert ar.aromaticity_index == pytest.approx(daf.aromaticity_index)
    assert ar.daf_sum_pct == pytest.approx(100.0)

    no_ns = elemental_core.ElementalSample("B", c, h, 0.0, 0.0, o, basis="daf").compute_stoichiometry()
    assert no_ns.aromaticity_index != pytest.approx(daf.aromaticity_index)


def test_basis_daf_not_double_corrected_and_sum_warning():
    vals = dict(c_pct=55.0, h_pct=5.0, n_pct=1.0, s_pct=0.5, o_pct=38.5, ash_pct=10.0, moisture_pct=2.0)
    daf = elemental_core.ElementalSample("D", basis="daf", **vals).compute_stoichiometry()
    assert daf.daf_sum_pct == pytest.approx(100.0)
    assert daf.basis_warning is None

    wrong = elemental_core.ElementalSample("W", basis="as_received", **vals).compute_stoichiometry()
    assert wrong.daf_sum_pct == pytest.approx(100.0 * 100.0 / 88.0)
    assert wrong.basis_warning and "100.5" in wrong.basis_warning
    assert wrong.to_dict()["QC_Warning"]

    with pytest.raises(ValueError):
        elemental_core.ElementalSample("E", basis="wet", **vals).compute_stoichiometry()

    df = pd.DataFrame({"Sample_ID": ["P1", "P2"], "C": [55.0, 49.5], "H": [5.0, 4.5], "N": [1.0, 0.9],
                       "S": [0.5, 0.45], "Ash": [10.0, 10.0], "Basis": ["daf", "dry"]})
    samples = elemental_core.parse_elemental_table(df)
    assert [s.basis for s in samples] == ["daf", "dry"]
    # O по разности на заявленном базисе: daf 100-(C+H+N+S); dry 100-(C+H+N+S+A)
    assert samples[0].o_pct == pytest.approx(38.5)
    assert samples[1].o_pct == pytest.approx(100.0 - (49.5 + 4.5 + 0.9 + 0.45 + 10.0))
    assert all(s.basis_warning is None for s in samples)

    forced = elemental_core.parse_elemental_table(df, basis="daf")
    assert all(s.basis == "daf" for s in forced)
    with pytest.raises(ValueError):
        elemental_core.parse_elemental_table(df, basis="wet")
