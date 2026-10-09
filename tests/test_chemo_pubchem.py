"""
test_chemo_pubchem.py — Тестирование модуля хемоинформатики chemo_pubchem.
"""

import pytest
import chemo_pubchem


def test_clean_formula():
    assert chemo_pubchem.clean_formula("CHO C9H10O4") == "C9H10O4"
    assert chemo_pubchem.clean_formula("C7H8O2") == "C7H8O2"
    assert chemo_pubchem.clean_formula("  C20H28O2  ") == "C20H28O2"


def test_reference_lookup():
    """Проверка мгновенного извлечения из эталонного справочника (офлайн)."""
    res = chemo_pubchem.lookup_formula_in_pubchem("C8H8O3", max_records=1, use_online=False)
    assert len(res) == 1
    item = res[0]
    assert item["CID"] == 1183
    assert "Vanillin" in item["Title"]
    assert "C8H8O3" in chemo_pubchem.REFERENCE_NOM_STRUCTURES


def test_classification_heuristics():
    """Проверка биогеохимической классификации по химическим дескрипторам."""
    cls_resin = chemo_pubchem.classify_compound_by_name("Dehydroabietic Acid", "resin derivative", "C20H28O2")
    assert "Смоляная кислота" in cls_resin or "древесин" in cls_resin

    cls_lignin = chemo_pubchem.classify_compound_by_name("Vanillic acid", "", "C8H8O4")
    assert "лигнин" in cls_lignin.lower() or "кислота" in cls_lignin.lower()

    cls_sulf = chemo_pubchem.classify_compound_by_name("Lignosulfonate model", "", "C9H10O7S")
    assert "Лигносульфонат" in cls_sulf


def test_annotate_formula_table():
    """Проверка сводной таблицы аннотаций формул с ChEMBL и HMDB ссылками."""
    formulas = ["C7H8O2", "C8H8O3", "C20H28O2"]
    df = chemo_pubchem.annotate_formula_table(formulas, max_top=3)
    assert not df.empty
    assert len(df) == 3
    assert "Formula" in df.columns
    assert "Compound_Name" in df.columns
    assert "CID" in df.columns
    assert "ChEMBL_URL" in df.columns
    assert "HMDB_URL" in df.columns
    assert "Image_URL" in df.columns
    assert df.iloc[0]["CID"] == 460  # Guaiacol
    assert "chembl" in df.iloc[0]["ChEMBL_URL"]
    assert "hmdb" in df.iloc[0]["HMDB_URL"]


def test_expanded_reference_compounds():
    """Проверка наличия новых эталонных структур (кверцетин, галловая кислота, олеиновая кислота)."""
    assert "C15H10O7" in chemo_pubchem.REFERENCE_NOM_STRUCTURES
    assert "C7H6O5" in chemo_pubchem.REFERENCE_NOM_STRUCTURES
    assert "C18H34O2" in chemo_pubchem.REFERENCE_NOM_STRUCTURES
    q_data = chemo_pubchem.lookup_formula_in_pubchem("C15H10O7", use_online=False)
    assert len(q_data) > 0
    assert q_data[0]["Title"] == "Quercetin"
