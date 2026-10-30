"""
tests/test_security.py — Регрессионные тесты защиты веб-развёртывания ChemoSuite.
Path Traversal в именах спектров, изоляция хранилища, бюджет ИИ-агента, zlib-бомбы в mzML.
"""

import base64
import io
import json
import os
import threading
import zipfile

import numpy as np
import pytest

import chemo_agent
import open_spectrometry_formats
from project_io import load_chemo_project, safe_spectrum_name


# ==============================================================================
# 1. ИМЕНА СПЕКТРОВ И ПРОЕКТЫ .chemo
# ==============================================================================

@pytest.mark.parametrize("raw, expected", [
    ("sample.csv", "sample.csv"),
    ("../app.py", "app.py"),
    ("..\\..\\Windows\\win.ini", "win.ini"),
    ("/etc/passwd", "passwd"),
    ("C:\\data\\spec.txt", "spec.txt"),
    ("..", "spectrum"),
    ("", "spectrum"),
    (None, "spectrum"),
    ("bad\x00name.csv", "badname.csv"),
])
def test_safe_spectrum_name_strips_directories(raw, expected):
    assert safe_spectrum_name(raw) == expected


def _build_chemo(spectra_meta, raw_bins):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("chemo_metadata.json", json.dumps({"project_name": "evil"}))
        zf.writestr("spectra_db/index.json", json.dumps(spectra_meta))
        for idx, payload in raw_bins.items():
            zf.writestr(f"spectra_db/{idx}_raw.bin", payload)
    return buf.getvalue()


def test_chemo_project_names_cannot_escape_storage():
    """Имя '../app.py' из index.json не должно давать путь за пределами хранилища."""
    data = _build_chemo(
        [
            {"id": 0, "name": "../PWNED_app.py", "raw_path": "/etc/passwd"},
            {"id": 1, "name": "..", "raw_path": "C:\\Windows\\win.ini"},
        ],
        {0: b"print('pwned')", 1: b"x"},
    )
    restored = load_chemo_project(data)
    names = list(restored["spectra_db"].keys())
    assert names == ["PWNED_app.py", "spectrum_1"]
    for entry in restored["spectra_db"].values():
        # Путь из архива не доверяется и не используется для удаления/записи
        assert entry["raw_path"] == ""


def test_storage_path_stays_inside_session_dir(tmp_path, monkeypatch):
    from views import common
    monkeypatch.setattr(common, "get_storage_dir", lambda: str(tmp_path))
    base = os.path.realpath(str(tmp_path))
    for name in ["../app.py", "..\\..\\x.csv", "/abs/y.txt", "ok.csv"]:
        p = common.storage_path(name)
        assert os.path.commonpath([base, p]) == base


def test_remove_stored_file_ignores_outside_paths(tmp_path, monkeypatch):
    from views import common
    storage = tmp_path / "storage"
    storage.mkdir()
    outside = tmp_path / "keep_me.txt"
    outside.write_text("important")
    inside = storage / "spec.csv"
    inside.write_text("1,2")
    monkeypatch.setattr(common, "get_storage_dir", lambda: str(storage))

    common.remove_stored_file(str(outside))
    common.remove_stored_file(str(storage / ".." / "keep_me.txt"))
    assert outside.exists()

    common.remove_stored_file(str(inside))
    assert not inside.exists()


def test_web_mode_is_default(monkeypatch):
    """Без переменной окружения доступ к каталогам сервера выключен."""
    from views import chemo_ml_view, common
    if os.getenv("CHEMOSUITE_LOCAL_MODE"):
        pytest.skip("Local mode enabled in environment")
    assert common.LOCAL_MODE is False
    assert chemo_ml_view._is_allowed_batch_dir("demo_data")
    assert not chemo_ml_view._is_allowed_batch_dir("/")
    assert not chemo_ml_view._is_allowed_batch_dir(os.path.join("demo_data", "..", ".."))


# ==============================================================================
# 2. БЮДЖЕТ ИИ-АГЕНТА
# ==============================================================================

def test_daily_usage_counter_is_thread_safe(tmp_path, monkeypatch):
    """Параллельные сессии не должны превышать дневной лимит из-за гонки записи."""
    monkeypatch.setattr(chemo_agent, "USAGE_FILE", str(tmp_path / "usage.json"))
    limit = 30
    allowed = []

    def worker():
        for _ in range(10):
            ok, _, _ = chemo_agent.check_and_increment_daily_usage(limit)
            allowed.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(allowed) == limit
    assert chemo_agent.get_daily_usage(limit)["count"] == limit


def test_session_request_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(chemo_agent, "USAGE_FILE", str(tmp_path / "usage.json"))
    fake_state = {}
    monkeypatch.setattr(chemo_agent, "_get_session_state", lambda: fake_state)

    results = [chemo_agent.check_and_increment_daily_usage(1000)[0] for _ in range(chemo_agent.SESSION_REQUEST_LIMIT + 5)]
    assert sum(results) == chemo_agent.SESSION_REQUEST_LIMIT
    assert results[-1] is False


# ==============================================================================
# 3. mzML: ДЕКОМПРЕССИОННАЯ БОМБА
# ==============================================================================

def _mzml_with_array(payload: bytes, zlib_flag: bool) -> bytes:
    import zlib
    data = zlib.compress(payload) if zlib_flag else payload
    b64 = base64.b64encode(data).decode("ascii")
    comp = '<cvParam accession="MS:1000574" name="zlib compression"/>' if zlib_flag else ""
    return f"""<?xml version="1.0"?>
<mzML><run><spectrumList><spectrum id="s1">
<binaryDataArrayList>
<binaryDataArray><cvParam accession="MS:1000514" name="m/z array"/>
<cvParam accession="MS:1000523" name="64-bit float"/>{comp}<binary>{b64}</binary></binaryDataArray>
<binaryDataArray><cvParam accession="MS:1000515" name="intensity array"/>
<cvParam accession="MS:1000523" name="64-bit float"/>{comp}<binary>{b64}</binary></binaryDataArray>
</binaryDataArrayList></spectrum></spectrumList></run></mzML>""".encode("utf-8")


def test_mzml_normal_zlib_array_parses():
    arr = np.array([100.0, 200.0, 300.0])
    res = open_spectrometry_formats.parse_mzml_content(_mzml_with_array(arr.tobytes(), True))
    assert np.allclose(res["mz"], arr)


def test_mzml_zlib_bomb_rejected(monkeypatch):
    monkeypatch.setattr(open_spectrometry_formats, "MAX_DECOMPRESSED_ARRAY_BYTES", 1024)
    bomb = _mzml_with_array(b"\x00" * 64_000, True)
    with pytest.raises(ValueError):
        open_spectrometry_formats.parse_mzml_content(bomb)
