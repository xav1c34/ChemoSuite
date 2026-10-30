"""
modules/open_spectrometry_formats.py — Open Science Spectrometry Standards Engine for ChemoSuite Pro.

Provides zero-dependency, pure-Python parsers and exporters for standard open scientific formats:
1. mzML / mzXML Parser: HUPO-PSI standard XML format with IEEE 32/64-bit float base64 decoding & zlib decompression.
2. JCAMP-DX Parser: IUPAC Joint Committee on Atomic and Molecular Physical Data exchange standard.
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


def parse_mzml_content(raw_data: bytes) -> Dict[str, Any]:
    """
    Parse an open mzML or mzXML mass spectrometry file.
    Decodes binary data arrays (m/z and intensity) with automatic precision detection (32-bit / 64-bit)
    and optional zlib decompression.

    Returns:
    --------
    Dict with:
    - 'mz': np.ndarray of m/z values
    - 'intensity': np.ndarray of peak intensities
    - 'metadata': dict with instrument, scan number, polarity, and title
    """
    # Parse XML stream
    if isinstance(raw_data, str):
        raw_data = raw_data.encode("utf-8")

    tree = ET.fromstring(raw_data)

    # XML namespaces handling
    ns_match = re.match(r"\{.*\}", tree.tag)
    ns_prefix = ns_match.group(0) if ns_match else ""

    metadata = {
        "format": "mzML",
        "title": "Raw MS Spectrum",
        "polarity": "positive",
        "retention_time_sec": 0.0,
    }

    mz_array = np.array([], dtype=np.float64)
    intensity_array = np.array([], dtype=np.float64)

    # Search for spectrum elements
    spectrum_elem = tree.find(f".//{ns_prefix}spectrum")
    if spectrum_elem is None:
        # Fallback to mzXML scans
        spectrum_elem = tree.find(f".//{ns_prefix}scan")
        if spectrum_elem is not None:
            metadata["format"] = "mzXML"

    if spectrum_elem is not None:
        metadata["id"] = spectrum_elem.attrib.get("id", "scan_1")

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

            binary_elem = b_arr.find(f"{ns_prefix}binary")
            if binary_elem is not None and binary_elem.text:
                raw_b64 = binary_elem.text.strip().encode("ascii")
                decoded = base64.b64decode(raw_b64)
                if is_zlib:
                    d_obj = zlib.decompressobj()
                    decoded = d_obj.decompress(decoded, MAX_DECOMPRESSED_ARRAY_BYTES)
                    if d_obj.unconsumed_tail:
                        raise ValueError(
                            f"mzML binary array exceeds {MAX_DECOMPRESSED_ARRAY_BYTES // 1_000_000} MB after decompression"
                        )

                dtype = np.float64 if is_64bit else np.float32
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
                is_32 = peaks_elem.attrib.get("precision", "32") == "32"
                # mzXML defaults to big-endian network byte order '>f' or '>d'
                dtype_str = ">f4" if is_32 else ">f8"
                pairs = np.frombuffer(decoded, dtype=dtype_str).astype(np.float64)
                if len(pairs) % 2 == 0:
                    mz_array = pairs[0::2]
                    intensity_array = pairs[1::2]

    # Clean and sort by m/z
    if len(mz_array) > 0 and len(mz_array) == len(intensity_array):
        sort_order = np.argsort(mz_array)
        mz_array = mz_array[sort_order]
        intensity_array = intensity_array[sort_order]

    return {
        "mz": mz_array,
        "intensity": intensity_array,
        "metadata": metadata
    }


def parse_jcamp_dx(content: str) -> Dict[str, Any]:
    """
    Parse standard IUPAC JCAMP-DX formatted files for FTIR, NMR, or MS.
    Extracts headers and (X, Y) spectral coordinates.
    """
    metadata: Dict[str, str] = {}
    x_vals: List[float] = []
    y_vals: List[float] = []

    lines = content.splitlines()
    in_data_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("$$"):
            continue  # Comment

        if stripped.startswith("##"):
            parts = stripped[2:].split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip() if len(parts) > 1 else ""

            if key in ("XYDATA", "PEAK TABLE", "XYPOINTS"):
                in_data_block = True
                continue
            elif key == "END":
                in_data_block = False
                continue

            metadata[key] = val
        elif in_data_block:
            # Data row: X Y1 Y2 ... or X, Y
            # Clean comma or whitespace separation
            cleaned = stripped.replace(",", " ")
            tokens = cleaned.split()
            if len(tokens) >= 2:
                try:
                    # First token is X, remaining are Y values
                    cur_x = float(tokens[0])
                    for t in tokens[1:]:
                        cur_y = float(t)
                        x_vals.append(cur_x)
                        y_vals.append(cur_y)
                except ValueError:
                    pass

    return {
        "x": np.array(x_vals, dtype=np.float64),
        "y": np.array(y_vals, dtype=np.float64),
        "metadata": metadata,
        "title": metadata.get("TITLE", "JCAMP Spectrum"),
        "x_units": metadata.get("XUNITS", "1/CM" if "FTIR" in metadata.get("DATA TYPE", "") else "PPM"),
        "y_units": metadata.get("YUNITS", "ABSORBANCE"),
    }


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
