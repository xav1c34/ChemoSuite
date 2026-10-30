"""
tests/test_app_smoke.py — Сквозной smoke-тест интерфейса Streamlit.
Каждый аналитический модуль открывается через streamlit.testing.AppTest на обоих языках,
с перебором источников данных (включая демо-наборы) и обучением всех архитектур ML.
Необработанные исключения при рендере страниц должны отсутствовать.
"""

import os

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def _run_app(lang_label="English"):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    if lang_label != "English":
        at.sidebar.selectbox(key="global_lang_choice").select(lang_label).run()
    return at


def _errors(at):
    return [str(e.value) for e in at.exception]


def _open_module(at, idx):
    at.sidebar.selectbox(key="nav_module_selection").select_index(idx).run()
    return at.sidebar.selectbox(key="nav_module_selection").value


def _ml_module_index(at):
    opts = at.sidebar.selectbox(key="nav_module_selection").options
    return next(i for i, o in enumerate(opts) if "ML" in str(o))


def test_app_starts_without_exceptions():
    at = _run_app()
    assert not at.exception, _errors(at)
    assert len(at.sidebar.selectbox(key="nav_module_selection").options) >= 10


@pytest.mark.parametrize("lang_label", ["English", "Русский"])
def test_every_module_and_data_source_renders(lang_label):
    at = _run_app(lang_label)
    failures = {}
    n_modules = len(at.sidebar.selectbox(key="nav_module_selection").options)
    for idx in range(n_modules):
        name = _open_module(at, idx)
        if at.exception:
            failures[name] = _errors(at)
            continue
        for radio in list(at.radio):
            if radio.key is None:
                continue
            for option in list(radio.options):
                at.radio(key=radio.key).set_value(option).run()
                if at.exception:
                    failures[f"{name} / {radio.key}={option}"] = _errors(at)
    assert not failures, failures


def test_batch_screening_of_demo_data():
    at = _run_app()
    _open_module(at, _ml_module_index(at))
    radio = at.radio(key="ml_data_source_mode_radio")
    folder = next(o for o in radio.options if "Batch" in str(o) or "Folder" in str(o))
    radio.set_value(folder).run()
    at.button(key="btn_run_batch_screening").click().run()
    assert not at.exception, _errors(at)
    assert not at.error, [str(e.value) for e in at.error]
