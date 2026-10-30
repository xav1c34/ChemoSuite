"""
modules/open_spectrometry_formats.py — Open Science Spectrometry Standards Engine for ChemoSuite Pro.

Provides zero-dependency, pure-Python parsers and exporters for standard open scientific formats:
1. mzML / mzXML Parser: HUPO-PSI standard XML format with IEEE 32/64-bit float base64 decoding & zlib decompression.
2. JCAMP-DX Parser: IUPAC JCAMP-DX 4.24/5.x (McDonald & Wilks 1988) incl. ASDF compression
   (SQZ/DIF/DUP with Y-check), (XY..XY) peak tables, X/Y factors and NMR Hz→ppm conversion.
3. mzTab-M Exporter: Metabolomics Standards Initiative (MSI) compliance for multi-modal compound tables.
"""

import io
import re
import zlib
import base64
import struct
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd

# Предел распаковки одного бинарного массива mzML (защита от декомпрессионной бомбы)
MAX_DECOMPRESSED_ARRAY_BYTES = 200_000_000  # 200 MB


# Ускорители MS-Numpress (MS:1002312..1002314, MS:1002746..1002748) не поддерживаются
_NUMPRESS_ACCESSIONS = {
    "MS:1002312", "MS:1002313", "MS:1002314",
    "MS:1002746", "MS:1002747", "MS:1002748",
}


def _zlib_decompress_limited(data: bytes, what: str = "mzML binary array") -> bytes:
    """zlib-распаковка с ограничением MAX_DECOMPRESSED_ARRAY_BYTES (защита от декомпрессионной бомбы)."""
    d_obj = zlib.decompressobj()
    out = d_obj.decompress(data, MAX_DECOMPRESSED_ARRAY_BYTES)
    if d_obj.unconsumed_tail:
        raise ValueError(
            f"{what} exceeds {MAX_DECOMPRESSED_ARRAY_BYTES // 1_000_000} MB after decompression"
        )
    return out


def _iso8601_duration_to_seconds(value: str) -> Optional[float]:
    """mzXML retentionTime (xs:duration, напр. 'PT12.5S', 'PT1M3.2S') → секунды, либо None."""
    if not value:
        return None
    m = re.fullmatch(
        r"\s*-?P(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?\s*",
        value,
    )
    if not m or not any(m.groups()):
        try:
            return float(value)
        except ValueError:
            return None
    d, h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return d * 86400.0 + h * 3600.0 + mi * 60.0 + sec


def parse_mzml_content(raw_data: Union[bytes, str]) -> Dict[str, Any]:
    """
    Parse an open mzML or mzXML mass spectrometry file (first spectrum / scan).
    Decodes binary data arrays (m/z and intensity) with automatic precision detection (32-bit / 64-bit)
    and optional zlib decompression (mzML cvParam MS:1000574, mzXML compressionType="zlib").

    Returns:
    --------
    Dict with:
    - 'mz': np.ndarray of m/z values (sorted ascending)
    - 'intensity': np.ndarray of peak intensities
    - 'metadata': dict with format, id, title, polarity and retention_time_sec.
      Polarity ('positive'/'negative') and retention time (s) are read from the file
      (mzML: MS:1000130 / MS:1000129 / MS:1000016; mzXML: scan@polarity / scan@retentionTime)
      and are None when the file does not state them.

    Raises ValueError when the m/z and intensity arrays have different lengths,
    when a compression scheme is unsupported, or when a decompressed array is too large.
    """
    # Parse XML stream
    if isinstance(raw_data, str):
        raw_data = raw_data.encode("utf-8")

    tree = ET.fromstring(raw_data)

    # XML namespaces handling
    ns_match = re.match(r"\{.*\}", tree.tag)
    ns_prefix = ns_match.group(0) if ns_match else ""

    metadata: Dict[str, Any] = {
        "format": "mzML",
        "title": "Raw MS Spectrum",
        "polarity": None,
        "retention_time_sec": None,
    }

    mz_array = np.array([], dtype=np.float64)
    intensity_array = np.array([], dtype=np.float64)

    # Search for spectrum elements
    spectrum_elem = tree.find(f".//{ns_prefix}spectrum")
    is_mzxml = False
    if spectrum_elem is None:
        # Fallback to mzXML scans
        spectrum_elem = tree.find(f".//{ns_prefix}scan")
        if spectrum_elem is not None:
            metadata["format"] = "mzXML"
            is_mzxml = True

    if spectrum_elem is not None:
        metadata["id"] = spectrum_elem.attrib.get("id", spectrum_elem.attrib.get("num", "scan_1"))

        if is_mzxml:
            pol = spectrum_elem.attrib.get("polarity", "").strip()
            if pol == "+":
                metadata["polarity"] = "positive"
            elif pol == "-":
                metadata["polarity"] = "negative"
            metadata["retention_time_sec"] = _iso8601_duration_to_seconds(spectrum_elem.attrib.get("retentionTime", ""))
        else:
            # mzML: cvParams полярности и времени начала скана (scanList/scan)
            for cv in spectrum_elem.iter(f"{ns_prefix}cvParam"):
                acc = cv.attrib.get("accession", "")
                if acc == "MS:1000130":
                    metadata["polarity"] = "positive"
                elif acc == "MS:1000129":
                    metadata["polarity"] = "negative"
                elif acc == "MS:1000016":
                    try:
                        rt_val = float(cv.attrib.get("value", ""))
                    except ValueError:
                        continue
                    unit_acc = cv.attrib.get("unitAccession", "")
                    unit_name = cv.attrib.get("unitName", "").lower()
                    if unit_acc == "UO:0000031" or unit_name.startswith("minute"):
                        rt_val *= 60.0
                    elif unit_acc == "UO:0000032" or unit_name.startswith("hour"):
                        rt_val *= 3600.0
                    metadata["retention_time_sec"] = rt_val

        # Parse binaryDataArrayList in mzML
        binary_arrays = spectrum_elem.findall(f".//{ns_prefix}binaryDataArray")
        for b_arr in binary_arrays:
            # Check data type (m/z vs intensity)
            is_mz = False
            is_intensity = False
            is_64bit = False
            is_zlib = False

            for cv in b_arr.findall(f"{ns_prefix}cvParam"):
                acc = cv.attrib.get("accession", "")
                name = cv.attrib.get("name", "").lower()
                if acc == "MS:1000514" or "m/z array" in name:
                    is_mz = True
                elif acc == "MS:1000515" or "intensity array" in name:
                    is_intensity = True
                elif acc == "MS:1000523" or "64-bit float" in name:
                    is_64bit = True
                elif acc == "MS:1000521" or "32-bit float" in name:
                    is_64bit = False
                elif acc == "MS:1000574" or "zlib compression" in name:
                    is_zlib = True
                elif acc in _NUMPRESS_ACCESSIONS or "numpress" in name:
                    raise ValueError("MS-Numpress compressed mzML arrays are not supported; re-export with zlib or no compression")

            binary_elem = b_arr.find(f"{ns_prefix}binary")
            if binary_elem is not None and binary_elem.text:
                raw_b64 = binary_elem.text.strip().encode("ascii")
                decoded = base64.b64decode(raw_b64)
                if is_zlib:
                    decoded = _zlib_decompress_limited(decoded, "mzML binary array")

                dtype = np.dtype("<f8") if is_64bit else np.dtype("<f4")  # mzML: little-endian
                if len(decoded) % dtype.itemsize:
                    raise ValueError("mzML binary array length is not a multiple of the declared float size")
                data_unpacked = np.frombuffer(decoded, dtype=dtype).astype(np.float64)

                if is_mz:
                    mz_array = data_unpacked
                elif is_intensity:
                    intensity_array = data_unpacked

        # If mzXML peaks format
        if len(mz_array) == 0:
            peaks_elem = spectrum_elem.find(f".//{ns_prefix}peaks")
            if peaks_elem is not None and peaks_elem.text:
                decoded = base64.b64decode(peaks_elem.text.strip().encode("ascii"))
                comp_type = peaks_elem.attrib.get("compressionType", "none").strip().lower()
                if comp_type == "zlib":
                    decoded = _zlib_decompress_limited(decoded, "mzXML peaks array")
                elif comp_type not in ("", "none"):
                    raise ValueError(f"Unsupported mzXML compressionType='{comp_type}'")
                is_32 = peaks_elem.attrib.get("precision", "32") == "32"
                # mzXML always uses network (big-endian) byte order
                dtype_str = ">f4" if is_32 else ">f8"
                if len(decoded) % np.dtype(dtype_str).itemsize:
                    raise ValueError("mzXML peaks array length is not a multiple of the declared float size")
                pairs = np.frombuffer(decoded, dtype=dtype_str).astype(np.float64)
                if len(pairs) % 2 != 0:
                    raise ValueError("mzXML peaks array holds an odd number of values (m/z-intensity pairs expected)")
                mz_array = pairs[0::2]
                intensity_array = pairs[1::2]

    if len(mz_array) != len(intensity_array):
        raise ValueError(
            f"m/z array ({len(mz_array)} values) and intensity array ({len(intensity_array)} values) differ in length"
        )

    # Clean and sort by m/z
    if len(mz_array) > 0:
        sort_order = np.argsort(mz_array, kind="stable")
        mz_array = mz_array[sort_order]
        intensity_array = intensity_array[sort_order]

    return {
        "mz": mz_array,
        "intensity": intensity_array,
        "metadata": metadata
    }


# ==============================================================================
# JCAMP-DX 4.24 (McDonald & Wilks, Appl. Spectrosc. 1988, 42, 151–162)
# NMR extension: Davies & Lancashire, Pure Appl. Chem. 1993, 65, 1085–1095
# ==============================================================================

# ASDF (ASCII Squeezed Difference Form) character tables
_SQZ_DIGITS = {"@": 0, **{c: i + 1 for i, c in enumerate("ABCDEFGHI")}, **{c: -(i + 1) for i, c in enumerate("abcdefghi")}}
_DIF_DIGITS = {"%": 0, **{c: i + 1 for i, c in enumerate("JKLMNOPQR")}, **{c: -(i + 1) for i, c in enumerate("jklmnopqr")}}
_DUP_DIGITS = {**{c: i + 1 for i, c in enumerate("STUVWXYZ")}, "s": 9}
# Символы сжатия, кроме 'E'/'e' (которые в AFFN могут означать экспоненту)
_COMPRESSION_CHARS_RE = re.compile(r"[@A-DF-Ia-df-i%J-Rj-rS-Zs]")


def _decode_text(content: Union[str, bytes]) -> str:
    if isinstance(content, (bytes, bytearray)):
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return bytes(content).decode(enc)
            except UnicodeDecodeError:
                continue
        return bytes(content).decode("utf-8", errors="replace")
    return str(content)


def _norm_label(label: str) -> str:
    """JCAMP-DX: имена меток сравниваются без пробелов, '-', '/', '_' и без учета регистра."""
    return re.sub(r"[\s\-/_]", "", label.upper())


def _asdf_tokenize(line: str, allow_exponent: bool) -> List[Tuple[str, float]]:
    """
    Разбивает строку данных JCAMP-DX на токены ('AFFN'|'SQZ'|'DIF'|'DUP'|'MISSING', value).
    AFFN-числа могут разделяться пробелами, запятыми, ';' или знаком (PAC).
    """
    tokens: List[Tuple[str, float]] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t,;":
            i += 1
            continue
        if c == "?":
            tokens.append(("MISSING", float("nan")))
            i += 1
            continue
        if c in _SQZ_DIGITS or c in _DIF_DIGITS or c in _DUP_DIGITS:
            if c in _SQZ_DIGITS:
                kind, lead = "SQZ", _SQZ_DIGITS[c]
            elif c in _DIF_DIGITS:
                kind, lead = "DIF", _DIF_DIGITS[c]
            else:
                kind, lead = "DUP", _DUP_DIGITS[c]
            j = i + 1
            while j < n and (line[j].isdigit() or (line[j] == "." and kind != "DUP")):
                j += 1
            digits = line[i + 1:j]
            if kind == "DUP":
                tokens.append((kind, float(int(f"{lead}{digits}"))))
            else:
                val = float(f"{abs(lead)}{digits}")
                tokens.append((kind, -val if lead < 0 else val))
            i = j
            continue
        if c in "+-" or c.isdigit() or c == ".":
            j = i + 1 if c in "+-" else i
            while j < n and (line[j].isdigit() or line[j] == "."):
                j += 1
            if allow_exponent and j < n and line[j] in "Ee":
                k = j + 1
                if k < n and line[k] in "+-":
                    k += 1
                if k < n and line[k].isdigit():
                    while k < n and line[k].isdigit():
                        k += 1
                    j = k
            num = line[i:j]
            try:
                tokens.append(("AFFN", float(num)))
            except ValueError:
                raise ValueError(f"Invalid number '{num}' in JCAMP-DX data line: {line.strip()[:60]}")
            i = j
            continue
        raise ValueError(f"Unexpected character '{c}' in JCAMP-DX data line: {line.strip()[:60]}")
    return tokens


def _asdf_decode_ordinates(tokens: List[Tuple[str, float]]) -> Tuple[List[float], bool]:
    """
    Декодирует ординаты (AFFN/SQZ абсолютные, DIF — разности, DUP — повторы).
    DUP после DIF повторяет разность (JCAMP-DX 4.24, §5.9).
    Returns (values, ends_in_dif_mode).
    """
    vals: List[float] = []
    last_dif: Optional[float] = None
    for kind, v in tokens:
        if kind in ("AFFN", "SQZ"):
            vals.append(v)
            last_dif = None
        elif kind == "MISSING":
            vals.append(float("nan"))
            last_dif = None
        elif kind == "DIF":
            if not vals:
                raise ValueError("JCAMP-DX DIF value without a preceding absolute ordinate")
            vals.append(vals[-1] + v)
            last_dif = v
        elif kind == "DUP":
            if not vals:
                raise ValueError("JCAMP-DX DUP count without a preceding ordinate")
            for _ in range(int(v) - 1):
                vals.append(vals[-1] + last_dif if last_dif is not None else vals[-1])
    return vals, last_dif is not None


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).split(",")[0].strip())
    except ValueError:
        return None


def _ntuple_list(val: Optional[str]) -> List[str]:
    return [p.strip() for p in str(val).split(",")] if val else []


def _parse_xpp_block(
    data_lines: List[str],
    x_factor: float,
    y_factor: float,
    first_x: Optional[float],
    last_x: Optional[float],
    n_points: Optional[int],
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Декодирование (X++(Y..Y)) / (X++(R..R)): AFFN, SQZ, DIF (с Y-check), DUP."""
    # Решение о сжатии принимается для всего блока (иначе 'E' в SQZ-строке спутать с экспонентой)
    block_compressed = any(_COMPRESSION_CHARS_RE.search(ln) for ln in data_lines)

    records: List[Tuple[float, List[float], bool]] = []
    prev_ended_dif = False
    prev_last: Optional[float] = None
    for ln in data_lines:
        toks = _asdf_tokenize(ln, allow_exponent=not block_compressed)
        if not toks:
            continue
        x_kind, x_raw = toks[0]
        if x_kind not in ("AFFN", "SQZ"):
            raise ValueError(f"JCAMP-DX (X++(Y..Y)) line does not start with an abscissa: {ln.strip()[:60]}")
        ys, ends_dif = _asdf_decode_ordinates(toks[1:])
        if not ys:
            continue
        has_check = bool(prev_ended_dif and prev_last is not None)
        if has_check and not np.isclose(ys[0], prev_last, rtol=1e-9, atol=1e-6):
            warnings.append(
                f"Y-check failed at abscissa {x_raw:g}: {ys[0]:g} != {prev_last:g} (possible data corruption)"
            )
        records.append((x_raw, ys, has_check))
        prev_ended_dif = ends_dif
        prev_last = ys[-1]

    if not records:
        raise ValueError("JCAMP-DX data table contains no decodable ordinates")

    # Шаг по X: ΔX = (LASTX − FIRSTX)/(NPOINTS − 1); FIRSTX/LASTX уже в реальных единицах
    dx: Optional[float] = None
    if first_x is not None and last_x is not None and n_points and n_points > 1:
        dx = (last_x - first_x) / (n_points - 1)
    else:
        steps_dx: List[float] = []
        for (x_i, ys_i, _), (x_next, _, check_next) in zip(records[:-1], records[1:]):
            steps = len(ys_i) - (1 if check_next else 0)
            if steps > 0:
                steps_dx.append((x_next - x_i) * x_factor / steps)
        if steps_dx:
            dx = float(np.median(steps_dx))
            warnings.append("FIRSTX/LASTX/NPOINTS missing: abscissa spacing estimated from line abscissas")
    total_new = sum(len(ys) - (1 if chk else 0) for _, ys, chk in records)
    if dx is None:
        if total_new == 1:
            dx = 0.0
        else:
            raise ValueError("Cannot determine the X spacing of the (X++(Y..Y)) table (FIRSTX/LASTX/NPOINTS missing)")

    xs: List[float] = []
    ys_out: List[float] = []
    for x_raw, ys, has_check in records:
        x0 = x_raw * x_factor
        for k in range(1 if has_check else 0, len(ys)):
            xs.append(x0 + k * dx)
            ys_out.append(ys[k] * y_factor)

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys_out, dtype=np.float64)

    if n_points and len(x_arr) != n_points:
        warnings.append(f"NPOINTS={n_points} but {len(x_arr)} ordinates were decoded")
    tol = 0.5 * abs(dx) if dx else 1e-9
    first_ok = first_x is None or abs(x_arr[0] - first_x) <= max(tol, 1e-6 * abs(first_x))
    last_ok = last_x is None or abs(x_arr[-1] - last_x) <= max(tol, 1e-6 * abs(last_x))
    if not first_ok:
        warnings.append(f"First abscissa {x_arr[0]:g} disagrees with FIRSTX={first_x:g}; data-line abscissas were used")
    if not last_ok:
        warnings.append(f"Last abscissa {x_arr[-1]:g} disagrees with LASTX={last_x:g}; data-line abscissas were used")
    # Если заголовок и данные согласованы, берём точную равномерную сетку (без накопления округлений X0)
    if first_ok and last_ok and first_x is not None and last_x is not None and n_points == len(x_arr):
        x_arr = np.linspace(first_x, last_x, n_points)
    return x_arr, y_arr


def _parse_xy_block(
    data_lines: List[str],
    group: int,
    x_factor: float,
    y_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Декодирование (XY..XY), (XYW..XYW), (XYM..XYM): пары/тройки AFFN-значений."""
    flat: List[str] = []
    for ln in data_lines:
        flat.extend(t for t in re.split(r"[\s,;]+", ln.strip()) if t)
    if len(flat) % group != 0:
        raise ValueError(f"JCAMP-DX XY table holds {len(flat)} values, not a multiple of {group}")

    def _f(tok: str) -> float:
        if tok == "?":
            return float("nan")
        return float(tok)

    try:
        x_arr = np.array([_f(flat[i]) for i in range(0, len(flat), group)], dtype=np.float64) * x_factor
        y_arr = np.array([_f(flat[i + 1]) for i in range(0, len(flat), group)], dtype=np.float64) * y_factor
    except ValueError as exc:
        raise ValueError(f"Non-numeric value in JCAMP-DX XY table: {exc}")
    return x_arr, y_arr


def parse_jcamp_dx(content: Union[str, bytes]) -> Dict[str, Any]:
    """
    Parse a JCAMP-DX file (IR, Raman, UV, MS, 1D NMR) — JCAMP-DX 4.24 (McDonald & Wilks 1988),
    5.x and the NMR extension (Davies & Lancashire 1993). Accepts str or bytes.

    Supported data tables:
    - ##XYDATA=(X++(Y..Y)) and NTUPLES ##DATA TABLE=(X++(R..R)) (first page only):
      AFFN, PAC, and ASDF compressed forms SQZ (@, A–I, a–i), DIF (%, J–R, j–r) with the
      Y-value check at line starts, DUP (S–Z, s). Every ordinate gets its own abscissa
      X = X0 + k·ΔX, ΔX = (LASTX − FIRSTX)/(NPOINTS − 1); X0 = line abscissa · XFACTOR.
    - ##XYPOINTS / ##PEAK TABLE / ##DATA TABLE = (XY..XY), (XYW..XYW), (XYM..XYM).
    Raw values are multiplied by XFACTOR / YFACTOR; FIRSTX/LASTX are taken as real units.
    '?' ordinates become NaN.

    Returns dict: x, y (file order), metadata (LDR → value, upper-cased labels),
    title, x_units, y_units, data_form, warnings (list of str).
    Raises ValueError when no data table can be decoded.
    """
    text = _decode_text(content)
    metadata: Dict[str, str] = {}
    norm_meta: Dict[str, str] = {}
    warnings: List[str] = []

    data_label: Optional[str] = None
    data_form = ""
    data_lines: List[str] = []
    data_done = False
    extra_tables = 0

    for raw_line in text.splitlines():
        line = raw_line.split("$$", 1)[0]  # '$$' — комментарий до конца строки
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            parts = stripped[2:].split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip() if len(parts) > 1 else ""
            nkey = _norm_label(key)
            if data_label is not None:
                data_done = True  # любая новая метка завершает таблицу данных
            if nkey in ("XYDATA", "XYPOINTS", "PEAKTABLE", "DATATABLE"):
                if data_label is None:
                    data_label = nkey
                    data_form = val
                    data_done = False
                else:
                    extra_tables += 1
                continue
            if nkey == "END" or data_done:
                continue
            metadata[key] = val
            norm_meta[nkey] = val
        elif data_label is not None and not data_done:
            data_lines.append(stripped)

    if extra_tables:
        warnings.append(f"{extra_tables} additional data table(s)/page(s) ignored; only the first was read")
    if data_label is None:
        raise ValueError("No JCAMP-DX data table (##XYDATA, ##XYPOINTS, ##PEAK TABLE, ##DATA TABLE) found")

    # Параметры абсциссы; для NTUPLES — из списков ##FIRST/##LAST/##FACTOR/##VAR_DIM/##UNITS
    is_ntuples = "NTUPLES" in norm_meta
    x_factor = _to_float(norm_meta.get("XFACTOR"))
    y_factor = _to_float(norm_meta.get("YFACTOR"))
    first_x = _to_float(norm_meta.get("FIRSTX"))
    last_x = _to_float(norm_meta.get("LASTX"))
    n_pts_f = _to_float(norm_meta.get("NPOINTS"))
    x_units = norm_meta.get("XUNITS", "")
    y_units = norm_meta.get("YUNITS", "")
    if is_ntuples:
        firsts, lasts = _ntuple_list(norm_meta.get("FIRST")), _ntuple_list(norm_meta.get("LAST"))
        factors, dims = _ntuple_list(norm_meta.get("FACTOR")), _ntuple_list(norm_meta.get("VARDIM"))
        units = _ntuple_list(norm_meta.get("UNITS"))
        if first_x is None and firsts:
            first_x = _to_float(firsts[0])
        if last_x is None and lasts:
            last_x = _to_float(lasts[0])
        if n_pts_f is None and dims:
            n_pts_f = _to_float(dims[0])
        if x_factor is None and factors:
            x_factor = _to_float(factors[0])
        if y_factor is None and len(factors) > 1:
            y_factor = _to_float(factors[1])
        if not x_units and units:
            x_units = units[0]
        if not y_units and len(units) > 1:
            y_units = units[1]
    x_factor = 1.0 if x_factor is None else x_factor
    y_factor = 1.0 if y_factor is None else y_factor
    n_points = int(round(n_pts_f)) if n_pts_f else None

    form_match = re.search(r"\(([^)]*)\)", data_form)
    form_core = form_match.group(1).replace(" ", "").upper() if form_match else data_form.replace(" ", "").upper()
    if "++" in form_core:
        x_arr, y_arr = _parse_xpp_block(data_lines, x_factor, y_factor, first_x, last_x, n_points, warnings)
    else:
        if not form_core:
            form_core = "XY..XY"
            warnings.append("Data table form not stated; (XY..XY) pairs assumed")
        group = 3 if re.match(r"XY[A-Z]\.\.XY[A-Z]", form_core) else 2
        x_arr, y_arr = _parse_xy_block(data_lines, group, x_factor, y_factor)

    if len(x_arr) == 0:
        raise ValueError("JCAMP-DX data table is empty")

    return {
        "x": x_arr,
        "y": y_arr,
        "metadata": metadata,
        "title": metadata.get("TITLE", "JCAMP Spectrum"),
        "x_units": x_units,
        "y_units": y_units,
        "data_form": data_form,
        "warnings": warnings,
    }


def jcamp_metadata_value(metadata: Dict[str, Any], label: str) -> Optional[str]:
    """Поиск значения JCAMP-метки без учёта пробелов/регистра ('.OBSERVE FREQUENCY' == '.OBSERVEFREQUENCY')."""
    target = _norm_label(label)
    for k, v in metadata.items():
        if _norm_label(str(k)) == target:
            return v
    return None


def jcamp_nmr_x_to_ppm(x: np.ndarray, metadata: Dict[str, Any], x_units: str = "") -> Tuple[np.ndarray, List[str]]:
    """
    Convert a JCAMP-DX NMR abscissa to ppm (x must be in file order).

    - XUNITS=PPM: returned unchanged.
    - XUNITS=HZ: δ(ppm) = ν(Hz) / SF(MHz), with SF = ##$SF (Bruker spectral reference
      frequency) when present, otherwise ##.OBSERVE FREQUENCY (Davies & Lancashire 1993).
      If ##.SHIFT REFERENCE=(type, compound, point, δ) is present, the scale is shifted so that
      data point `point` (1-based) equals δ.
    Raises ValueError for Hz data without a spectrometer frequency or for unknown units.
    """
    notes: List[str] = []
    units = (x_units or jcamp_metadata_value(metadata, "XUNITS") or "").strip().upper()
    x = np.asarray(x, dtype=np.float64)
    if units in ("PPM", ""):
        if not units:
            notes.append("XUNITS not stated; abscissa assumed to be ppm")
        return x.copy(), notes
    if units != "HZ":
        raise ValueError(f"Unsupported NMR abscissa units XUNITS={units} (expected PPM or HZ)")

    sf = _to_float(jcamp_metadata_value(metadata, "$SF"))
    sf_src = "##$SF"
    if not sf:
        sf = _to_float(jcamp_metadata_value(metadata, ".OBSERVE FREQUENCY"))
        sf_src = "##.OBSERVE FREQUENCY"
    if not sf or sf <= 0:
        raise ValueError("JCAMP-DX NMR data in Hz but no ##.OBSERVE FREQUENCY / ##$SF given: cannot convert to ppm")
    ppm = x / sf
    notes.append(f"Hz → ppm using {sf_src} = {sf:g} MHz")

    shift_ref = jcamp_metadata_value(metadata, ".SHIFT REFERENCE")
    if shift_ref:
        parts = [p.strip() for p in shift_ref.strip().strip("()").split(",")]
        if len(parts) >= 4:
            try:
                point = int(float(parts[2]))
                delta_ref = float(parts[3])
                if 1 <= point <= len(ppm):
                    ppm = ppm + (delta_ref - ppm[point - 1])
                    notes.append(f"Re-referenced with ##.SHIFT REFERENCE: point {point} = {delta_ref:g} ppm")
            except ValueError:
                notes.append("##.SHIFT REFERENCE could not be parsed; ignored")
    return ppm, notes


def mass_spectrum_to_peak_df(mz: Any, intensity: Any) -> pd.DataFrame:
    """
    Converts parallel m/z and intensity arrays (from parse_mzml_content or a JCAMP-DX mass
    spectrum) into the peak table used by the FT-ICR studio: columns mass, intensity,
    norm_intensity (% of the base peak). Non-finite values and m/z ≤ 0 are dropped, rows are
    sorted by mass. Raises ValueError if the arrays differ in length or nothing valid remains.
    """
    m = np.asarray(mz, dtype=float).ravel()
    i = np.asarray(intensity, dtype=float).ravel()
    if len(m) != len(i):
        raise ValueError(f"m/z array ({len(m)} values) and intensity array ({len(i)} values) differ in length")
    ok = np.isfinite(m) & np.isfinite(i) & (m > 0)
    df = pd.DataFrame({"mass": m[ok], "intensity": i[ok]}).sort_values("mass", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid (m/z > 0, finite intensity) data points")
    base = float(df["intensity"].max())
    df["norm_intensity"] = df["intensity"] / base * 100.0 if base > 0 else np.nan
    return df


def jcamp_is_mass_spectrum(parsed: Dict[str, Any]) -> bool:
    """True if a parse_jcamp_dx result declares a mass spectrum (##DATA TYPE or ##XUNITS = M/Z)."""
    meta = parsed.get("metadata", {}) or {}
    data_type = str(jcamp_metadata_value(meta, "DATA TYPE") or "").upper()
    x_units = str(parsed.get("x_units", "") or "").upper().replace(" ", "")
    return "MASS" in data_type or x_units in ("M/Z", "MZ", "M/E")


def export_mztab_m(
    arg1: Any = None,
    arg2: Any = None,
    *,
    sample_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    features_df: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> str:
    """
    Generate Metabolomics Standards Initiative (MSI) compliant mzTab-M table.

    Flexibly accepts:
    - export_mztab_m(metadata, features_df)
    - export_mztab_m(features_df, sample_id="...")
    - export_mztab_m(features_df, metadata)
    - export_mztab_m(features_df=df, sample_id=...)
    """
    actual_meta: Dict[str, Any] = {}
    actual_df: Optional[pd.DataFrame] = None

    if isinstance(arg1, pd.DataFrame):
        actual_df = arg1
        if isinstance(arg2, dict):
            actual_meta = dict(arg2)
    elif isinstance(arg1, dict):
        actual_meta = dict(arg1)
        if isinstance(arg2, pd.DataFrame):
            actual_df = arg2

    if features_df is not None:
        actual_df = features_df
    if metadata is not None:
        actual_meta.update(metadata)

    if sample_id:
        actual_meta.setdefault("sample_id", sample_id)
        if "description" not in actual_meta:
            actual_meta["description"] = f"ChemoSuite Multi-Modal Spectrometry Export - {sample_id}"

    if actual_df is None:
        actual_df = pd.DataFrame()

    out = io.StringIO()

    # 1. MTD Section (Metadata)
    out.write("MTD\tmzTab-version\t2.0.0-M\n")
    out.write("MTD\tmzTab-mode\tSummary\n")
    out.write("MTD\tmzTab-type\tIdentification\n")
    desc = actual_meta.get("description") or f"ChemoSuite Export for {actual_meta.get('sample_id', 'Sample')}"
    out.write(f"MTD\tdescription\t{desc}\n")
    out.write("MTD\tsoftware[1]\t[MS, MS:1003100, ChemoSuite Pro, 3.1.0]\n")

    # 2. SML Section (Small Molecule Summary header)
    headers = [
        "SML_ID",
        "chemical_formula",
        "smiles",
        "theoretical_neutral_mass",
        "adduct",
        "reliability_score",
        "best_search_engine_score[1]",
    ]
    out.write("SMH\t" + "\t".join(headers) + "\n")

    # 3. SML Rows
    for idx, row in actual_df.iterrows():
        sml_id = str(idx + 1)
        formula = str(row.get("Formula", row.get("formula", "N/A")))
        smiles = str(row.get("SMILES", row.get("smiles", "N/A")))
        mass = str(row.get("m/z", row.get("mz", row.get("mass", row.get("theor_mass", row.get("exact_mass", "0.0"))))))
        adduct = str(row.get("adduct", "[M-H]-"))
        raw_score = row.get("VIP", row.get("concordance_score", row.get("reliability_score", 1.0)))
        try:
            score = str(round(float(raw_score), 3))
        except (ValueError, TypeError):
            score = "1.0"

        row_data = [
            sml_id,
            formula,
            smiles,
            mass,
            adduct,
            "1",
            score
        ]
        out.write("SML\t" + "\t".join(row_data) + "\n")

    return out.getvalue()
