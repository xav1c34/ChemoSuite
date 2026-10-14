"""
tests/test_chemo_agent.py — Comprehensive unit tests for ChemoSuite AI Copilot.

Tests API key resolution, scientific prompt synthesis, Groq REST calls,
streaming token generation, and error handling with full mocking.
"""

import os
import json
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

import modules.chemo_agent as chemo_agent
from views.agent_view import save_groq_api_key_to_env


def test_get_groq_api_key_explicit():
    """Explicit key should take highest precedence."""
    key = chemo_agent.get_groq_api_key("gsk_explicit_test_key_123")
    assert key == "gsk_explicit_test_key_123"


def test_get_groq_api_key_env_var(monkeypatch):
    """Environment variable GROQ_API_KEY should be read if explicit key is None."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_test_key_456")
    key = chemo_agent.get_groq_api_key(None)
    assert key == "gsk_env_test_key_456"


def test_get_groq_api_key_missing(monkeypatch, tmp_path):
    """When no key is configured anywhere, returns None."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("os.path.exists", return_value=False):
        key = chemo_agent.get_groq_api_key(None)
        assert key is None


def test_build_sample_analysis_prompt_fticr():
    """Prompt construction should calculate and format key physicochemical metrics."""
    df = pd.DataFrame({
        "mass": [350.12, 420.15, 510.22],
        "intensity": [100.0, 50.0, 25.0],
        "H/C": [1.12, 0.95, 1.30],
        "O/C": [0.45, 0.60, 0.35],
        "AI": [0.35, 0.68, 0.15],
        "DBE": [10.0, 14.0, 8.0],
        "NOSC": [-0.1, 0.2, -0.4],
        "Hetero_Class": ["CHO", "CHON", "CHOS"],
    })
    sample_data = {
        "assigned_df": df,
        "eem": {"C1_Humic_like": 45.2, "C2_Fulvic_like": 30.1, "HIX": 8.4},
        "nmr": {"fa (aromaticity)": 0.32, "aliphatic": 0.45},
        "ftir": {"1510 cm-1": "Strong", "1720 cm-1": "Medium"},
        "elemental": {"C": 52.3, "H": 4.8, "N": 1.2, "O": 41.7},
    }

    prompt_ru = chemo_agent.build_sample_analysis_prompt("Baikal_Water_01", sample_data, lang="ru")
    assert "Baikal_Water_01" in prompt_ru
    assert "FT-ICR MS" in prompt_ru
    assert "H/C (среднее)" in prompt_ru
    assert "EEM-PARAFAC" in prompt_ru
    assert "13C" in prompt_ru
    assert "ATR-FTIR" in prompt_ru
    assert "CHNS-O" in prompt_ru

    prompt_en = chemo_agent.build_sample_analysis_prompt("Sample_EN", sample_data, lang="en")
    assert "Analyze sample: **Sample_EN**" in prompt_en


def test_call_groq_chat_no_key(monkeypatch):
    """call_groq_chat must return success=False with helpful error when no key is present."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("os.path.exists", return_value=False):
        res = chemo_agent.call_groq_chat(
            messages=[{"role": "user", "content": "ping"}],
            api_key=None,
        )
        assert res["success"] is False
        assert "GROQ_API_KEY не найден" in res["error"]


@patch("requests.post")
def test_call_groq_chat_success(mock_post):
    """Verifies successful Groq chat completion parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Образец представляет собой природное РОВ с умеренной ароматичностью.",
                }
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180},
    }
    mock_post.return_value = mock_resp

    messages = [{"role": "user", "content": "Опиши образец."}]
    res = chemo_agent.call_groq_chat(messages, api_key="gsk_mock_valid_key")

    assert res["success"] is True
    assert "природное РОВ" in res["content"]
    assert res["model"] == "llama-3.3-70b-versatile"
    assert res["usage"]["total_tokens"] == 180


@patch("requests.post")
def test_call_groq_chat_http_error(mock_post):
    """Verifies graceful handling of HTTP 401/429 error responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Invalid API Key"
    mock_post.return_value = mock_resp

    res = chemo_agent.call_groq_chat([{"role": "user", "content": "hi"}], api_key="gsk_invalid")
    assert res["success"] is False
    assert "HTTP 401" in res["error"]


@patch("requests.post")
def test_stream_groq_chat_tokens(mock_post):
    """Verifies SSE line parsing in stream_groq_chat."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Simulate Server-Sent Events (SSE) data lines
    mock_resp.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}}]}',
        b'data: {"choices":[{"delta":{"content":" \xd0\xbc\xd0\xb8\xd1\x80"}}]}',
        b'data: [DONE]',
    ]
    mock_post.return_value = mock_resp

    messages = [{"role": "user", "content": "Привет"}]
    tokens = list(chemo_agent.stream_groq_chat(messages, api_key="gsk_mock_stream_key"))

    assert len(tokens) == 2
    assert "".join(tokens) == "Привет мир"


def test_save_groq_api_key_to_env(tmp_path, monkeypatch):
    """Verifies that save_groq_api_key_to_env correctly writes to .env."""
    monkeypatch.chdir(tmp_path)
    success = save_groq_api_key_to_env("gsk_persistence_test_999")
    assert success is True

    env_path = tmp_path / ".env"
    assert env_path.exists()
    content = env_path.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=gsk_persistence_test_999" in content
    assert os.environ.get("GROQ_API_KEY") == "gsk_persistence_test_999"

    # Updating key in existing .env
    save_groq_api_key_to_env("gsk_updated_key_000")
    updated_content = env_path.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=gsk_updated_key_000" in updated_content
    assert "gsk_persistence_test_999" not in updated_content


@patch("requests.get")
def test_get_available_groq_models(mock_get):
    """Verifies that available models are filtered and ranked properly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "whisper-large-v3"},
            {"id": "meta-llama/llama-prompt-guard-2-86m"},
            {"id": "openai/gpt-oss-120b"},
            {"id": "qwen/qwen3.8-27b"},
            {"id": "openai/gpt-oss-20b"},
        ]
    }
    mock_get.return_value = mock_resp

    models = chemo_agent.get_available_groq_models("gsk_test")
    assert "whisper-large-v3" not in models
    assert "meta-llama/llama-prompt-guard-2-86m" not in models
    assert "openai/gpt-oss-120b" in models
    assert models[0] == "qwen/qwen3.8-27b"
