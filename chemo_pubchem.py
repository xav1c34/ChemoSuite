"""
chemo_pubchem.py — Модуль хемоинформатики и структурной идентификации ChemoSuite.
Интеграция с базой данных PubChem (NCBI PUG-REST API) для аннотации формул FT-ICR MS,
поиска химических названий, SMILES, 2D структур и классификации природных/техногенных соединений.
"""

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
import pandas as pd

CACHE_FILE = ".cache_pubchem.json"

# Встроенный справочник эталонных маркеров природного РОВ и крафт-шлам-лигнина (офлайн fallback)
REFERENCE_NOM_STRUCTURES = {
    "C7H8O2": {
        "CID": 460,
        "Title": "Guaiacol",
        "IUPACName": "2-methoxyphenol",
        "SMILES": "COC1=CC=CC=C1O",
        "Class": "Мономер лигнина (Гваяциловый пул)",
    },
    "C8H8O3": {
        "CID": 1183,
        "Title": "Vanillin",
        "IUPACName": "4-hydroxy-3-methoxybenzaldehyde",
        "SMILES": "COC1=C(C=CC(=C1)C=O)O",
        "Class": "Продукт окисления лигнина",
    },
    "C8H8O4": {
        "CID": 8468,
        "Title": "Vanillic Acid",
        "IUPACName": "4-hydroxy-3-methoxybenzoic acid",
        "SMILES": "COC1=C(C=CC(=C1)C(=O)O)O",
        "Class": "Карбоновая кислота лигнина",
    },
    "C9H10O3": {
        "CID": 8469,
        "Title": "Acetovanillone",
        "IUPACName": "1-(4-hydroxy-3-methoxyphenyl)ethan-1-one",
        "SMILES": "CC(=O)C1=CC(=C(C=C1)O)OC",
        "Class": "Кето-производное лигнина",
    },
    "C9H10O4": {
        "CID": 1738,
        "Title": "Homovanillic Acid",
        "IUPACName": "2-(4-hydroxy-3-methoxyphenyl)acetic acid",
        "SMILES": "COC1=C(C=CC(=C1)CC(=O)O)O",
        "Class": "Маркер деструкции лигнина",
    },
    "C8H10O3": {
        "CID": 10787,
        "Title": "Syringol",
        "IUPACName": "2,6-dimethoxyphenol",
        "SMILES": "COC1=CC=CC(=C1O)OC",
        "Class": "Сирингиловый мономер лигнина",
    },
    "C9H10O5": {
        "CID": 10768,
        "Title": "Syringic Acid",
        "IUPACName": "4-hydroxy-3,5-dimethoxybenzoic acid",
        "SMILES": "COC1=CC(=CC(=C1O)OC)C(=O)O",
        "Class": "Карбоновая кислота лигнина (S-пул)",
    },
    "C10H10O4": {
        "CID": 445858,
        "Title": "Ferulic Acid",
        "IUPACName": "(E)-3-(4-hydroxy-3-methoxyphenyl)prop-2-enoic acid",
        "SMILES": "COC1=C(C=CC(=C1)/C=C/C(=O)O)O",
        "Class": "Оксикоричная кислота (Лигнин/CRAM)",
    },
    "C9H8O4": {
        "CID": 689043,
        "Title": "Caffeic Acid",
        "IUPACName": "(E)-3-(3,4-dihydroxyphenyl)prop-2-enoic acid",
        "SMILES": "C1=CC(=C(C=C1/C=C/C(=O)O)O)O",
        "Class": "Полифенольный антиоксидант",
    },
    "C10H12O3": {
        "CID": 1549095,
        "Title": "Coniferyl Alcohol",
        "IUPACName": "4-[(E)-3-hydroxyprop-1-enyl]-2-methoxyphenol",
        "SMILES": "COC1=C(C=CC(=C1)/C=C/CO)O",
        "Class": "Монолигнол (G-тип)",
    },
    "C11H14O4": {
        "CID": 5352119,
        "Title": "Sinapyl Alcohol",
        "IUPACName": "4-[(E)-3-hydroxyprop-1-enyl]-2,6-dimethoxyphenol",
        "SMILES": "COC1=CC(=CC(=C1O)OC)/C=C/CO",
        "Class": "Монолигнол (S-тип)",
    },
    "C20H30O2": {
        "CID": 10569,
        "Title": "Abietic Acid",
        "IUPACName": "(1R,4aR,4bR,10aR)-1,4a-dimethyl-7-propan-2-yl-2,3,4,4b,5,6,10,10a-octahydrophenanthrene-1-carboxylic acid",
        "SMILES": "CC(C)C1=CC2=CCC3C(C2CC1)(CCCC3(C)C(=O)O)C",
        "Class": "Смоляная кислота хвойных (Древесный маркер)",
    },
    "C20H28O2": {
        "CID": 9382,
        "Title": "Dehydroabietic Acid",
        "IUPACName": "(1R,4aS,10aR)-1,4a-dimethyl-7-propan-2-yl-1,2,3,4,9,10-hexahydrophenanthrene-1-carboxylic acid",
        "SMILES": "CC(C)C1=CC2=C(C=C1)C3(CCCC(C3CC2)(C)C(=O)O)C",
        "Class": "Техногенный маркер шлам-лигнина БЦБК (ДХАК)",
    },
    "C9H10O7S": {
        "CID": 9832104,
        "Title": "Guaiacyl Lignosulfonate Model",
        "IUPACName": "1-(4-hydroxy-3-methoxyphenyl)-1-sulfopropan-2-one model",
        "SMILES": "COC1=C(C=CC(=C1)C(C(=O)C)S(=O)(=O)O)O",
        "Class": "Лигносульфонат (Техногенный Kraft маркер)",
    },
    "C15H10O7": {
        "CID": 5280343,
        "Title": "Quercetin",
        "IUPACName": "2-(3,4-dihydroxyphenyl)-3,5,7-trihydroxychromen-4-one",
        "SMILES": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
        "Class": "Флавоноид / Полифенол (Танниновый пул)",
    },
    "C15H10O6": {
        "CID": 5280863,
        "Title": "Kaempferol",
        "IUPACName": "3,5,7-trihydroxy-2-(4-hydroxyphenyl)chromen-4-one",
        "SMILES": "C1=CC(=CC=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O",
        "Class": "Флавоноид / Растительный полифенол",
    },
    "C7H6O5": {
        "CID": 370,
        "Title": "Gallic Acid",
        "IUPACName": "3,4,5-trihydroxybenzoic acid",
        "SMILES": "C1=C(C=C(C(=C1O)O)O)C(=O)O",
        "Class": "Гидролизуемый таннин / Фенолокислота",
    },
    "C16H18O9": {
        "CID": 1794427,
        "Title": "Chlorogenic Acid",
        "IUPACName": "(1S,3R,4R,5R)-3-[(E)-3-(3,4-dihydroxyphenyl)prop-2-enoyl]oxy-1,4,5-trihydroxycyclohexane-1-carboxylic acid",
        "SMILES": "C1C(C(C(CC1(C(=O)O)O)OC(=O)/C=C/C2=CC(=C(C=C2)O)O)O)O",
        "Class": "Оксикоричный эфир / Природный антиоксидант",
    },
    "C18H34O2": {
        "CID": 445639,
        "Title": "Oleic Acid",
        "IUPACName": "(Z)-octadec-9-enoic acid",
        "SMILES": "CCCCCCCCC=CCCCCCCCC(=O)O",
        "Class": "Ненасыщенная жирная кислота (Липидный пул)",
    },
    "C16H32O2": {
        "CID": 985,
        "Title": "Palmitic Acid",
        "IUPACName": "hexadecanoic acid",
        "SMILES": "CCCCCCCCCCCCCCCC(=O)O",
        "Class": "Насыщенная жирная кислота (Липидный пул)",
    },
    "C14H12O3": {
        "CID": 445154,
        "Title": "Resveratrol",
        "IUPACName": "5-[(E)-2-(4-hydroxyphenyl)ethenyl]benzene-1,3-diol",
        "SMILES": "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
        "Class": "Стильбен / Растительный фитоалексин",
    },
}

_memory_cache: Dict[str, List[Dict[str, Any]]] = {}


def _load_disk_cache() -> Dict[str, List[Dict[str, Any]]]:
    """Загружает локальный кэш запросов PubChem с диска."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_disk_cache(cache: Dict[str, List[Dict[str, Any]]]) -> None:
    """Сохраняет кэш на диск."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clean_formula(formula: str) -> str:
    """Очищает строку формулы от посторонних символов (например 'CHO C9H10O4' -> 'C9H10O4')."""
    f = str(formula).strip()
    match = re.search(r"C\d+H\d+[A-Za-z0-9]*", f)
    if match:
        return match.group(0)
    return f.replace(" ", "")


def classify_compound_by_name(title: str, iupac: str, formula: str) -> str:
    """Эвристическая биогеохимическая классификация соединения на основе названия и формулы."""
    t = (title + " " + iupac).lower()

    if any(k in t for k in ["abietic", "pimaric", "resin acid", "dehydroabiet"]):
        return "Смоляная кислота / Экстрактивные в-ва древесины"
    if any(k in t for k in ["lignosulfon", "sulfon"]):
        return "Лигносульфонат (Техногенный маркер сульфитной/крафт варки)"
    if any(k in t for k in ["vanill", "guaiac", "syring", "ferul", "sinap", "conifer", "lignin", "coumar"]):
        return "Лигнин / Полифенольный мономер"
    if any(k in t for k in ["benzoic", "salicylic", "hydroxybenzoic", "phthal"]):
        return "Ароматическая карбоновая кислота"
    if any(k in t for k in ["fatty acid", "octadec", "hexadec", "palmit", "stear"]):
        return "Алифатическая жирная кислота (Липиды)"
    if any(k in t for k in ["amino acid", "peptide"]):
        return "Аминокислота / Пептид"
    if any(k in t for k in ["glucose", "fructose", "sugar", "furan", "pyran", "sacchar"]):
        return "Углевод / Сахарид"

    # Классификация по элементным отношениям
    c_m = re.search(r"C(\d+)", formula)
    h_m = re.search(r"H(\d+)", formula)
    o_m = re.search(r"O(\d+)", formula)
    if c_m and h_m:
        c_val = float(c_m.group(1))
        h_val = float(h_m.group(1))
        o_val = float(o_m.group(1)) if o_m else 0.0
        hc = h_val / c_val if c_val > 0 else 1.0
        oc = o_val / c_val if c_val > 0 else 0.0

        if hc < 0.7 and oc < 0.5:
            return "Конденсированная ароматика (CAS / Пирогенный пул)"
        elif 0.7 <= hc <= 1.5 and 0.2 <= oc <= 0.6:
            return "Лигниноподобное вещество / CRAM"
        elif hc > 1.5 and oc < 0.3:
            return "Алифатическое соединение (Липидоподобный пул)"
        elif oc >= 0.7:
            return "Высокоокисленное гуминовое вещество / Таннины"

    return "Органическое вещество природного происхождения"


def lookup_formula_in_pubchem(
    formula: str,
    max_records: int = 4,
    timeout: float = 4.0,
    use_online: bool = True
) -> List[Dict[str, Any]]:
    """
    Выполняет поиск структурных изомеров в PubChem по брутто-формуле.
    Сначала проверяет память и кэш, затем офлайн-справочник, затем PubChem PUG-REST API.

    Returns:
    --------
    Список словарей с метаданными:
      [
        {
          'CID': int,
          'Title': str,
          'IUPACName': str,
          'SMILES': str,
          'MolecularWeight': float,
          'Class': str,
          'PubChem_URL': str,
          'Image_URL': str,
          'Source': 'Cache' | 'Online' | 'Reference',
        },
        ...
      ]
    """
    global _memory_cache
    f_clean = clean_formula(formula)

    if not _memory_cache:
        _memory_cache = _load_disk_cache()

    if f_clean in _memory_cache and _memory_cache[f_clean]:
        return _memory_cache[f_clean][:max_records]

    # Проверка офлайн-справочника
    if f_clean in REFERENCE_NOM_STRUCTURES:
        ref = REFERENCE_NOM_STRUCTURES[f_clean].copy()
        ref["MolecularWeight"] = 0.0
        ref["PubChem_URL"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{ref['CID']}"
        ref["Image_URL"] = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ref['CID']}/PNG?record_type=2d&image_size=small"
        ref["Source"] = "Reference"
        res = [ref]
        _memory_cache[f_clean] = res
        _save_disk_cache(_memory_cache)
        return res

    if not use_online:
        return []

    # Запрос в NCBI PubChem PUG-REST API
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{urllib.parse.quote(f_clean)}/"
        f"property/IUPACName,Title,MolecularWeight,ConnectivitySMILES,CanonicalSMILES/JSON?MaxRecords={max_records}"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ChemoSuite/3.0 (Analytical Chem Dept, MSU; msu.chemosuite@gmail.com)"}
    )

    results: List[Dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            props = payload.get("PropertyTable", {}).get("Properties", [])
            for p in props:
                cid = p.get("CID")
                title = p.get("Title") or p.get("IUPACName") or f"Compound CID {cid}"
                iupac = p.get("IUPACName") or title
                smiles = p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or ""
                mw = float(p.get("MolecularWeight", 0.0))
                cls_annot = classify_compound_by_name(title, iupac, f_clean)

                results.append({
                    "CID": cid,
                    "Title": title,
                    "IUPACName": iupac,
                    "SMILES": smiles,
                    "MolecularWeight": mw,
                    "Class": cls_annot,
                    "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                    "Image_URL": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?record_type=2d&image_size=small",
                    "Source": "Online",
                })
    except Exception:
        # При отсутствии сети или тайм-ауте возвращаем обобщенную оценку на основе брутто-формулы
        fallback_cls = classify_compound_by_name("", "", f_clean)
        results = [{
            "CID": None,
            "Title": f"Изомерный ансамбль {f_clean}",
            "IUPACName": f"Молекулярный класс формулы {f_clean}",
            "SMILES": "N/A",
            "MolecularWeight": 0.0,
            "Class": fallback_cls,
            "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/#query={f_clean}",
            "Image_URL": None,
            "Source": "Formula_Heuristic",
        }]

    if results:
        _memory_cache[f_clean] = results
        _save_disk_cache(_memory_cache)

    return results


def annotate_formula_table(formulas: List[str], max_top: int = 15) -> pd.DataFrame:
    """
    Аннотирует список формул (например, топ VIP-маркеров) структурами PubChem.
    Возвращает сводный DataFrame для отображения в Streamlit и экспорта в CSV.
    """
    rows = []
    seen = set()

    for raw_f in formulas[:max_top]:
        f_clean = clean_formula(raw_f)
        if not f_clean or f_clean in seen:
            continue
        seen.add(f_clean)

        matches = lookup_formula_in_pubchem(f_clean, max_records=1, timeout=3.0)
        chembl_url = f"https://www.ebi.ac.uk/chembl/g/#search_results/all/query={urllib.parse.quote(f_clean)}"
        hmdb_url = f"https://hmdb.ca/unearth/q?query={urllib.parse.quote(f_clean)}&searcher=metabolites"

        if matches:
            best = matches[0]
            cid = best.get("CID")
            rows.append({
                "Formula": f_clean,
                "Compound_Name": best.get("Title", "N/A"),
                "CID": cid,
                "Chemical_Class": best.get("Class", "N/A"),
                "SMILES": best.get("SMILES", "N/A"),
                "PubChem_URL": best.get("PubChem_URL", f"https://pubchem.ncbi.nlm.nih.gov/#query={f_clean}"),
                "ChEMBL_URL": chembl_url,
                "HMDB_URL": hmdb_url,
                "Image_URL": best.get("Image_URL"),
            })
        else:
            rows.append({
                "Formula": f_clean,
                "Compound_Name": f"Формула {f_clean}",
                "CID": None,
                "Chemical_Class": classify_compound_by_name("", "", f_clean),
                "SMILES": "N/A",
                "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/#query={f_clean}",
                "ChEMBL_URL": chembl_url,
                "HMDB_URL": hmdb_url,
                "Image_URL": None,
            })

    return pd.DataFrame(rows)
