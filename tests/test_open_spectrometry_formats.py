"""
tests/test_open_spectrometry_formats.py — Unit tests for open spectrometry standards (mzML, JCAMP-DX, mzTab-M).
"""
import base64
import zlib
import numpy as np
import pandas as pd
import pytest
from modules.open_spectrometry_formats import (
    parse_mzml_content,
    parse_jcamp_dx,
    export_mztab_m,
)


def test_parse_mzml_synthetic():
    # Build a valid synthetic mzML XML snippet with 32-bit floats and zlib compression
    mz_raw = np.array([150.05, 200.10, 350.25], dtype=np.float32)
    int_raw = np.array([1000.0, 50000.0, 2500.0], dtype=np.float32)

    b64_mz = base64.b64encode(zlib.compress(mz_raw.tobytes())).decode("ascii")
    b64_int = base64.b64encode(zlib.compress(int_raw.tobytes())).decode("ascii")

    mzml_xml = f"""<?xml version="1.0" encoding="utf-8"?>
    <mzML xmlns="http://psi.hupo.org/ms/mzml">
      <run id="run_1">
        <spectrumList count="1">
          <spectrum id="scan=100">
            <binaryDataArrayList count="2">
              <binaryDataArray>
                <cvParam accession="MS:1000514" name="m/z array"/>
                <cvParam accession="MS:1000521" name="32-bit float"/>
                <cvParam accession="MS:1000574" name="zlib compression"/>
                <binary>{b64_mz}</binary>
              </binaryDataArray>
              <binaryDataArray>
                <cvParam accession="MS:1000515" name="intensity array"/>
                <cvParam accession="MS:1000521" name="32-bit float"/>
                <cvParam accession="MS:1000574" name="zlib compression"/>
                <binary>{b64_int}</binary>
              </binaryDataArray>
            </binaryDataArrayList>
          </spectrum>
        </spectrumList>
      </run>
    </mzML>
    """

    res = parse_mzml_content(mzml_xml.encode("utf-8"))
    assert len(res["mz"]) == 3
    assert len(res["intensity"]) == 3
    assert np.isclose(res["mz"][0], 150.05, atol=1e-2)
    assert np.isclose(res["intensity"][1], 50000.0, atol=1.0)
    assert res["metadata"]["format"] == "mzML"


def test_parse_jcamp_dx():
    jcamp_content = """##TITLE=Standard Humic Acid FTIR
##JCAMP-DX=5.00
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=400.0
##LASTX=4000.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
1700.0 0.85
1600.0 0.42
1200.0 0.31
1000.0 0.15
##END=
"""
    parsed = parse_jcamp_dx(jcamp_content)
    assert len(parsed["x"]) == 4
    assert len(parsed["y"]) == 4
    assert parsed["title"] == "Standard Humic Acid FTIR"
    assert parsed["x_units"] == "1/CM"
    assert np.isclose(parsed["x"][0], 1700.0)
    assert np.isclose(parsed["y"][0], 0.85)


def test_export_mztab_m():
    df_features = pd.DataFrame({
        "Formula": ["C15H18O7", "C20H24O10"],
        "SMILES": ["c1cc(O)ccc1C(=O)O", "c1c(O)c(O)cc(C(=O)O)c1"],
        "m/z": [310.105, 424.137],
        "VIP": [2.45, 1.82]
    })

    mztab_str = export_mztab_m({"description": "Test Multi-Modal Run"}, df_features)
    assert "MTD\tmzTab-version\t2.0.0-M" in mztab_str
    assert "SMH\tSML_ID\tchemical_formula" in mztab_str
    assert "C15H18O7" in mztab_str
    assert "C20H24O10" in mztab_str
    assert "2.45" in mztab_str
