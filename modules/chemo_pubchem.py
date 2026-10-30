"""
chemo_pubchem.py — Модуль хемоинформатики и структурной идентификации ChemoSuite.
Интеграция с базой данных PubChem (NCBI PUG-REST API) для аннотации формул FT-ICR MS,
поиска химических названий, SMILES, 2D структур и классификации природных/техногенных соединений.

Принципы:
- Сначала выполняется онлайн-поиск изомеров в PubChem (PUG-REST ``fastformula``).
- Встроенный справочник REFERENCE_NOM_STRUCTURES используется ТОЛЬКО как явно помеченный
  офлайн-резерв (нет сети / use_online=False); он не является поиском изомеров.
- Ошибки сети/HTTP никогда не кэшируются и не подменяются «эвристическими» заглушками:
  возвращается пустой результат с атрибутом ``error`` (см. PubChemResult) либо
  возбуждается PubChemError (fetch_compounds_by_formula).
- Записи кэша имеют срок жизни (CACHE_TTL_SECONDS).
"""

import json
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover - requests входит в requirements.txt
    requests = None
    REQUESTS_AVAILABLE = False

CACHE_FILE = ".cache_pubchem.json"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 суток
PUBCHEM_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HTTP_HEADERS = {"User-Agent": "ChemoSuite/3.0 (Analytical Chem Dept, MSU; msu.chemosuite@gmail.com)"}

# Полный список символов химических элементов (IUPAC) для строгой валидации формул.
_ELEMENT_SYMBOLS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No
Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d{0,4})+$")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_MAX_FORMULA_LEN = 64

# Моноизотопные массы (для офлайн-записей справочника)
_MONOISOTOPIC = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
                 "S": 31.97207100, "P": 30.97376163}
# Стандартные атомные массы IUPAC (средние)
_AVERAGE = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974}


class PubChemError(RuntimeError):
    """Ошибка обращения к PubChem (сеть, тайм-аут, HTTP-ошибка, асинхронный ответ 202)."""


class PubChemResult(list):
    """
    Список результатов (полностью совместим с list) с диагностическими атрибутами:
      - source: 'Online' | 'Cache' | 'Reference_Offline' | 'None'
      - error: Optional[str] — текст ошибки сети/HTTP (если онлайн-поиск не удался)
      - offline_fallback: bool — True, если возвращена запись встроенного справочника
    """

    def __init__(self, items=(), source: str = "None", error: Optional[str] = None,
                 offline_fallback: bool = False):
        super().__init__(items)
        self.source = source
        self.error = error
        self.offline_fallback = offline_fallback


# Встроенный справочник эталонных маркеров природного РОВ и крафт-шлам-лигнина.
# ТОЛЬКО офлайн-резерв. Все пары CID / формула / SMILES / IUPAC сверены с PubChem
# (PUG-REST compound/cid/.../property/MolecularFormula,Title,IUPACName,SMILES; 2026-10).
# Исправлены CID: Acetovanillone 2214 (было 8469), Syringol 7041 (было 10787),
# Syringic acid 10742 (было 10768), Sinapyl alcohol 5280507 (было 5352119),
# Dehydroabietic acid 94391 (было 9382). Запись «Guaiacyl Lignosulfonate Model» удалена:
# CID 9832104 — C44H44N4O9, ключ C9H10O7S не совпадал с её SMILES (C10H12O6S),
# а сам SMILES в PubChem отсутствует (CID 0).
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
        "CID": 2214,
        "Title": "Acetovanillone",
        "IUPACName": "1-(4-hydroxy-3-methoxyphenyl)ethanone",
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
        "CID": 7041,
        "Title": "Syringol",
        "IUPACName": "2,6-dimethoxyphenol",
        "SMILES": "COC1=C(C(=CC=C1)OC)O",
        "Class": "Сирингиловый мономер лигнина",
    },
    "C9H10O5": {
        "CID": 10742,
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
        "CID": 5280507,
        "Title": "Sinapyl Alcohol",
        "IUPACName": "4-[(E)-3-hydroxyprop-1-enyl]-2,6-dimethoxyphenol",
        "SMILES": "COC1=CC(=CC(=C1O)OC)/C=C/CO",
        "Class": "Монолигнол (S-тип)",
    },
    "C20H30O2": {
        "CID": 10569,
        "Title": "Abietic Acid",
        "IUPACName": "(1R,4aR,4bR,10aR)-1,4a-dimethyl-7-propan-2-yl-2,3,4,4b,5,6,10,10a-octahydrophenanthrene-1-carboxylic acid",
        "SMILES": "CC(C)C1=CC2=CC[C@@H]3[C@@]([C@H]2CC1)(CCC[C@@]3(C)C(=O)O)C",
        "Class": "Смоляная кислота хвойных (Древесный маркер)",
    },
    "C20H28O2": {
        "CID": 94391,
        "Title": "Dehydroabietic Acid",
        "IUPACName": "(1R,4aS,10aR)-1,4a-dimethyl-7-propan-2-yl-2,3,4,9,10,10a-hexahydrophenanthrene-1-carboxylic acid",
        "SMILES": "CC(C)C1=CC2=C(C=C1)[C@]3(CCC[C@@]([C@@H]3CC2)(C)C(=O)O)C",
        "Class": "Техногенный маркер шлам-лигнина БЦБК (ДХАК)",
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
        "SMILES": "C1[C@H]([C@H]([C@@H](C[C@@]1(C(=O)O)O)OC(=O)/C=C/C2=CC(=C(C=C2)O)O)O)O",
        "Class": "Оксикоричный эфир / Природный антиоксидант",
    },
    "C18H34O2": {
        "CID": 445639,
        "Title": "Oleic Acid",
        "IUPACName": "(Z)-octadec-9-enoic acid",
        "SMILES": "CCCCCCCC/C=C\\CCCCCCCC(=O)O",
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

_memory_cache: Optional[Dict[str, Any]] = None
_memory_cache_path: Optional[str] = None


def _load_disk_cache() -> Dict[str, Any]:
    """Загружает локальный кэш запросов PubChem с диска."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_disk_cache(cache: Dict[str, Any]) -> None:
    """Сохраняет кэш на диск."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _get_cache() -> Dict[str, Any]:
    """Кэш в памяти (перечитывается с диска при смене CACHE_FILE, например в тестах)."""
    global _memory_cache, _memory_cache_path
    if _memory_cache is None or _memory_cache_path != CACHE_FILE:
        _memory_cache = _load_disk_cache()
        _memory_cache_path = CACHE_FILE
    return _memory_cache


def _cache_get(key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Возвращает кэшированный успешный ответ, если он не старше CACHE_TTL_SECONDS.
    Записи старого формата (без метки времени) и эвристические заглушки считаются недействительными.
    """
    entry = _get_cache().get(key)
    if not isinstance(entry, dict) or "ts" not in entry or "results" not in entry:
        return None
    try:
        age = time.time() - float(entry["ts"])
    except (TypeError, ValueError):
        return None
    if age < 0 or age > CACHE_TTL_SECONDS:
        return None
    results = entry["results"]
    if not isinstance(results, list):
        return None
    if any(isinstance(r, dict) and r.get("Source") == "Formula_Heuristic" for r in results):
        return None
    return [dict(r) for r in results]


def _cache_put(key: str, results: List[Dict[str, Any]]) -> None:
    """Сохраняет ТОЛЬКО успешный ответ PubChem (в т.ч. «0 совпадений») с меткой времени."""
    cache = _get_cache()
    cache[key] = {"ts": time.time(), "results": results}
    _save_disk_cache(cache)


def clear_cache(remove_file: bool = False) -> None:
    """Очищает кэш в памяти (и, при remove_file=True, файл кэша на диске)."""
    global _memory_cache, _memory_cache_path
    _memory_cache = {}
    _memory_cache_path = CACHE_FILE
    if remove_file and os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except OSError:
            pass


def is_valid_formula(formula: str) -> bool:
    """
    Строгая проверка брутто-формулы (например 'C9H10O4', 'C10H12O6S').
    Допускаются только символы реальных элементов и целые индексы; пробелы, '/', '%', '..', '_' и т.п.
    отклоняются, поэтому формула безопасна для подстановки в путь URL PUG-REST.
    """
    f = str(formula or "").strip()
    if not f or len(f) > _MAX_FORMULA_LEN or not _FORMULA_RE.match(f):
        return False
    tokens = _FORMULA_TOKEN_RE.findall(f)
    if "".join(sym + cnt for sym, cnt in tokens) != f:
        return False
    return all(sym in _ELEMENT_SYMBOLS for sym, _ in tokens)


def parse_formula_counts(formula: str) -> Dict[str, int]:
    """Разбирает валидную брутто-формулу в словарь {элемент: число атомов} ({} для невалидной)."""
    counts: Dict[str, int] = {}
    if not is_valid_formula(formula):
        return counts
    for sym, cnt in _FORMULA_TOKEN_RE.findall(str(formula).strip()):
        counts[sym] = counts.get(sym, 0) + (int(cnt) if cnt else 1)
    return counts


def _formula_mass(formula: str, table: Dict[str, float]) -> Optional[float]:
    counts = parse_formula_counts(formula)
    if not counts or any(el not in table for el in counts):
        return None
    return float(sum(table[el] * n for el, n in counts.items()))


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


def _pubchem_formula_property_search(
    formula: str,
    properties: str,
    max_records: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    """
    PUG-REST: compound/fastformula/{formula}/property/{properties}/JSON?MaxRecords=N.
    Возвращает PropertyTable.Properties; [] при ответе PubChem «нет совпадений» (HTTP 404 PUGREST.NotFound).
    Любая сетевая/HTTP ошибка или асинхронный ответ 202 -> PubChemError (никогда не подменяется данными).
    """
    if not is_valid_formula(formula):
        raise ValueError(f"Некорректная брутто-формула для запроса PubChem: {formula!r}")
    if not REQUESTS_AVAILABLE:
        raise PubChemError("Библиотека requests недоступна: онлайн-запрос к PubChem невозможен.")

    max_records = max(1, int(max_records))
    url = (
        f"{PUBCHEM_REST_BASE}/compound/fastformula/{urllib.parse.quote(formula, safe='')}/"
        f"property/{properties}/JSON"
    )
    try:
        resp = requests.get(url, params={"MaxRecords": max_records}, headers=HTTP_HEADERS, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise PubChemError(f"PubChem: превышен тайм-аут запроса ({timeout:g} с) для {formula}.") from e
    except requests.exceptions.RequestException as e:
        raise PubChemError(f"PubChem недоступен (ошибка сети) для {formula}: {e}") from e

    status = int(getattr(resp, "status_code", 0) or 0)
    if status == 404:
        # PUGREST.NotFound «Search returned no hits» — валидный ответ «0 изомеров»
        return []
    if status == 202:
        raise PubChemError(
            f"PubChem вернул асинхронный ответ 202 (запрос для {formula} поставлен в очередь). Повторите позже."
        )
    if status != 200:
        raise PubChemError(f"PubChem вернул HTTP {status} для {formula}.")
    try:
        payload = resp.json()
    except ValueError as e:
        raise PubChemError(f"PubChem вернул некорректный JSON для {formula}.") from e
    if not isinstance(payload, dict) or "PropertyTable" not in payload:
        fault = payload.get("Fault", {}) if isinstance(payload, dict) else {}
        raise PubChemError(f"PubChem: неожиданный ответ для {formula}: {fault.get('Message', 'нет PropertyTable')}")
    props = payload.get("PropertyTable", {}).get("Properties", [])
    return [p for p in props if isinstance(p, dict) and p.get("CID")]


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_compounds_by_formula(
    formula: str,
    max_records: int = 30,
    timeout: float = 10.0,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Поиск изомеров-кандидатов в PubChem по брутто-формуле для движка structure_elucidation.

    PUG-REST: compound/fastformula/{formula}/property/MonoisotopicMass,IsomericSMILES,IUPACName,Title,MolecularFormula/JSON

    Returns:
    --------
    Список словарей с ключами: cid, name, iupac_name, smiles, formula,
    exact_mass (= MonoisotopicMass PubChem, НЕ средняя MolecularWeight), source.
    Пустой список, если PubChem сообщает об отсутствии совпадений.

    Raises:
    -------
    ValueError    — некорректная формула (запрос не отправляется).
    PubChemError  — ошибка сети / тайм-аут / HTTP / 202; такие ответы не кэшируются.
    """
    f_clean = clean_formula(formula)
    if not is_valid_formula(f_clean):
        raise ValueError(f"Некорректная брутто-формула: {formula!r}")

    cache_key = f"fetch|{f_clean}|{int(max_records)}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            for c in cached:
                c["source"] = "Cache"
            return cached

    props = _pubchem_formula_property_search(
        f_clean, "MonoisotopicMass,IsomericSMILES,IUPACName,Title,MolecularFormula", max_records, timeout
    )
    out: List[Dict[str, Any]] = []
    for p in props[:max(1, int(max_records))]:
        cid = p.get("CID")
        # PubChem (2025+) возвращает запрошенный IsomericSMILES под ключом "SMILES"
        smiles = (p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES")
                  or p.get("ConnectivitySMILES") or "")
        title = p.get("Title") or p.get("IUPACName") or f"CID {cid}"
        out.append({
            "cid": int(cid) if cid is not None else None,
            "name": title,
            "iupac_name": p.get("IUPACName") or title,
            "smiles": smiles,
            "formula": p.get("MolecularFormula") or f_clean,
            "exact_mass": _to_float_or_none(p.get("MonoisotopicMass")),
            "source": "PubChem",
        })
    if use_cache:
        _cache_put(cache_key, out)
    return out


def _offline_reference_entry(f_clean: str, reason: str) -> List[Dict[str, Any]]:
    """Явно помеченная запись встроенного справочника (НЕ поиск изомеров PubChem)."""
    if f_clean not in REFERENCE_NOM_STRUCTURES:
        return []
    ref = REFERENCE_NOM_STRUCTURES[f_clean].copy()
    ref["MolecularFormula"] = f_clean
    ref["MolecularWeight"] = _formula_mass(f_clean, _AVERAGE)
    ref["MonoisotopicMass"] = _formula_mass(f_clean, _MONOISOTOPIC)
    ref["PubChem_URL"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{ref['CID']}"
    ref["Image_URL"] = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ref['CID']}/PNG?record_type=2d&image_size=small"
    ref["Source"] = "Reference_Offline"
    ref["Offline_Fallback"] = True
    ref["Note"] = (
        "Офлайн-резерв: одна запись встроенного справочника, а не поиск изомеров PubChem "
        f"(для этой формулы существуют и другие изомеры). Причина: {reason}"
    )
    return [ref]


def lookup_formula_in_pubchem(
    formula: str,
    max_records: int = 4,
    timeout: float = 4.0,
    use_online: bool = True
) -> List[Dict[str, Any]]:
    """
    Выполняет поиск структурных изомеров в PubChem по брутто-формуле.
    Порядок: кэш (TTL, ключ учитывает max_records) -> PubChem PUG-REST (fastformula) ->
    только при недоступности сети (или use_online=False) — явно помеченный офлайн-справочник.
    Ошибки не кэшируются и не подменяются заглушками.

    Returns:
    --------
    PubChemResult (подкласс list) словарей:
      [
        {
          'CID': int,
          'Title': str,
          'IUPACName': str,
          'SMILES': str,
          'MolecularWeight': float | None,
          'MonoisotopicMass': float | None,
          'MolecularFormula': str,
          'Class': str,
          'PubChem_URL': str,
          'Image_URL': str,
          'Source': 'Cache' | 'Online' | 'Reference_Offline',
          'Offline_Fallback': bool,
          'Note': str (только для офлайн-записей),
        },
        ...
      ]
    Атрибуты результата: .source, .error (текст ошибки сети или None), .offline_fallback.
    """
    f_clean = clean_formula(formula)
    if not is_valid_formula(f_clean):
        return PubChemResult([], source="None", error=f"Некорректная брутто-формула: {formula!r}")

    max_records = max(1, int(max_records))
    cache_key = f"lookup|{f_clean}|{max_records}"
    cached = _cache_get(cache_key)
    if cached is not None:
        for c in cached:
            c["Source"] = "Cache"
        return PubChemResult(cached, source="Cache")

    if not use_online:
        ref = _offline_reference_entry(f_clean, reason="онлайн-поиск отключен (use_online=False)")
        return PubChemResult(ref, source="Reference_Offline" if ref else "None", offline_fallback=bool(ref))

    try:
        props = _pubchem_formula_property_search(
            f_clean, "IUPACName,Title,MolecularWeight,MonoisotopicMass,MolecularFormula,IsomericSMILES",
            max_records, timeout,
        )
    except PubChemError as e:
        ref = _offline_reference_entry(f_clean, reason=str(e))
        return PubChemResult(ref, source="Reference_Offline" if ref else "None", error=str(e),
                             offline_fallback=bool(ref))

    results: List[Dict[str, Any]] = []
    for p in props[:max_records]:
        cid = p.get("CID")
        title = p.get("Title") or p.get("IUPACName") or f"Compound CID {cid}"
        iupac = p.get("IUPACName") or title
        smiles = (p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES")
                  or p.get("ConnectivitySMILES") or "")
        mol_formula = p.get("MolecularFormula") or f_clean
        results.append({
            "CID": cid,
            "Title": title,
            "IUPACName": iupac,
            "SMILES": smiles,
            "MolecularWeight": _to_float_or_none(p.get("MolecularWeight")),
            "MonoisotopicMass": _to_float_or_none(p.get("MonoisotopicMass")),
            "MolecularFormula": mol_formula,
            "Class": classify_compound_by_name(title, iupac, mol_formula),
            "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "Image_URL": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?record_type=2d&image_size=small",
            "Source": "Online",
            "Offline_Fallback": False,
        })

    _cache_put(cache_key, results)
    return PubChemResult(results, source="Online")


def annotate_formula_table(formulas: List[str], max_top: int = 15) -> pd.DataFrame:
    """
    Аннотирует список формул (например, топ VIP-маркеров) структурами PubChem.
    Возвращает сводный DataFrame для отображения в Streamlit и экспорта в CSV.
    Колонка 'Source' показывает происхождение аннотации ('Online', 'Cache',
    'Reference_Offline' — офлайн-резерв, 'Not found', 'Unavailable'); 'Lookup_Note' — пояснение/ошибка.
    """
    rows = []
    seen = set()

    for raw_f in formulas[:max_top]:
        f_clean = clean_formula(raw_f)
        if not f_clean or f_clean in seen:
            continue
        seen.add(f_clean)

        matches = lookup_formula_in_pubchem(f_clean, max_records=1, timeout=3.0)
        err = getattr(matches, "error", None)
        q = urllib.parse.quote(f_clean, safe="")
        chembl_url = f"https://www.ebi.ac.uk/chembl/search_results/{q}"
        hmdb_url = f"https://hmdb.ca/unearth/q?query={q}&searcher=metabolites"

        if matches:
            best = matches[0]
            cid = best.get("CID")
            rows.append({
                "Formula": f_clean,
                "Compound_Name": best.get("Title", "N/A"),
                "CID": cid,
                "Chemical_Class": best.get("Class", "N/A"),
                "SMILES": best.get("SMILES", "N/A"),
                "PubChem_URL": best.get("PubChem_URL", f"https://pubchem.ncbi.nlm.nih.gov/#query={q}"),
                "ChEMBL_URL": chembl_url,
                "HMDB_URL": hmdb_url,
                "Image_URL": best.get("Image_URL"),
                "Source": best.get("Source", "Online"),
                "Lookup_Note": best.get("Note") or err or "",
            })
        else:
            rows.append({
                "Formula": f_clean,
                "Compound_Name": f"Формула {f_clean}",
                "CID": None,
                "Chemical_Class": classify_compound_by_name("", "", f_clean),
                "SMILES": "N/A",
                "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/#query={q}",
                "ChEMBL_URL": chembl_url,
                "HMDB_URL": hmdb_url,
                "Image_URL": None,
                "Source": "Unavailable" if err else "Not found",
                "Lookup_Note": err or "PubChem: совпадений не найдено",
            })

    return pd.DataFrame(rows)
