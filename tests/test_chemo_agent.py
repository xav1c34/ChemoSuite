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
from views.agent_view import save_groq_api_key_to_env, save_gemini_api_key_to_env


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


def test_build_sample_analysis_prompt_raw_unassigned():
    """Prompt construction must NOT crash with KeyError if H/C is missing (raw peak list)."""
    raw_df = pd.DataFrame({
        "m/z": [245.0512, 312.1124, 489.2319],
        "intensity": [10500.0, 54000.0, 23000.0],
    })
    sample_data = {
        "assigned_df": raw_df,
    }
    prompt = chemo_agent.build_sample_analysis_prompt("Raw_Peak_List_01", sample_data, lang="ru")
    assert "Raw_Peak_List_01" in prompt
    assert "Число детектированных пиков: 3" in prompt
    assert "245.05" in prompt or "348" in prompt or "Средняя масса" in prompt


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
    assert "401" in res["error"]


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


def test_normalize_latex():
    """Verifies that various LLM LaTeX formats are standardized for KaTeX."""
    # 1. Inline parenthesis conversion
    assert chemo_agent.normalize_latex(r"Index \( H/C \)") == "Index $H/C$"
    # 2. Block bracket conversion
    raw_block = r"Formula: \[ x = \frac{1}{2} \]"
    norm_block = chemo_agent.normalize_latex(raw_block)
    assert "$$\n" in norm_block
    assert r"x = \frac{1}{2}" in norm_block
    # 3. Strip inner spaces
    assert chemo_agent.normalize_latex("Sample $ H/C $ and $ O/C $") == "Sample $H/C$ and $O/C$"
    # 4. AI_mod formatting
    assert chemo_agent.normalize_latex("Shows $AI_mod$ index") == r"Shows $AI_{\text{mod}}$ index"
    assert chemo_agent.normalize_latex("Shows $AI_{mod}$ index") == r"Shows $AI_{\text{mod}}$ index"


def test_repair_mojibake():
    """Verifies that mojibake from improper decoding is restored to proper unicode symbols."""
    # Em-dash mojibake
    assert "—" in chemo_agent.repair_mojibake("FT-ICR MS â€” Baikal")
    assert "—" in chemo_agent.repair_mojibake("FT-ICR MS â≡≡ Baikal")
    # En-dash mojibake
    assert "200–500" in chemo_agent.repair_mojibake("range 200â€“500 Da")
    # Math symbols
    assert "≈" in chemo_agent.repair_mojibake("AI â‰ˆ 0.45")
    # Clean text unchanged
    assert chemo_agent.repair_mojibake("Чистый текст — тире") == "Чистый текст — тире"


def test_format_groq_error_429_otpm():
    """Verifies that 429 rate limit errors (OTPM) produce helpful messages with collapsible details."""
    raw_error = json.dumps({
        "error": {
            "message": "Request too large for model qwen/qwen3.8-27b on output tokens per minute (OTPM): Limit 1000, Requested 2048.",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    })
    formatted = chemo_agent.format_groq_error(429, raw_error)
    assert "Превышен минутный лимит токенов Groq" in formatted
    assert "<details>" in formatted
    assert "</details>" in formatted
    assert "Limit 1000" in formatted


def test_format_groq_error_404():
    """Verifies that 404 model errors produce clear guidance."""
    formatted = chemo_agent.format_groq_error(404, '{"error":{"message":"model not found"}}')
    assert "Модель недоступна в Groq" in formatted
    assert "<details>" in formatted


def test_format_groq_error_401():
    """Verifies that 401 unauthorized errors guide the user to check API key."""
    formatted = chemo_agent.format_groq_error(401, "Invalid API Key")
    assert "Неверный GROQ_API_KEY" in formatted


@patch("requests.post")
def test_max_tokens_clamping(mock_post):
    """Verifies that requested max_tokens is clamped sensibly per model architecture."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Ok"}}],
    }
    mock_post.return_value = mock_resp

    chemo_agent.call_groq_chat(
        messages=[{"role": "user", "content": "ping"}],
        api_key="gsk_test_key",
        model="qwen/qwen3.8-27b",
        max_tokens=4096,  # Intentionally large
    )

    call_kwargs = mock_post.call_args.kwargs
    sent_payload = call_kwargs["json"]
    assert sent_payload["max_tokens"] <= 1200


def test_get_active_fticr_data_selection():
    """Verifies that _get_active_fticr_data supports explicit selection and fallback resolution."""
    import streamlit as st
    from views.agent_view import _get_active_fticr_data

    sample1_df = pd.DataFrame({"m/z": [200.1], "intensity": [100.0]})
    sample2_df = pd.DataFrame({"m/z": [300.2], "intensity": [200.0]})

    st.session_state["spectra_db"] = {
        "Sample_A.csv": {"assigned_df": sample1_df},
        "Sample_B.csv": {"assigned_df": sample2_df},
    }
    st.session_state["active_spectrum_name"] = "Sample_A.csv"

    # 1. Explicit selection of Sample_B
    res_b = _get_active_fticr_data("Sample_B.csv")
    assert res_b is not None
    assert res_b["name"] == "Sample_B.csv"
    assert len(res_b["assigned_df"]) == 1
    assert res_b["assigned_df"]["m/z"].iloc[0] == 300.2

    # 2. Default fallback to active_spectrum_name
    res_a = _get_active_fticr_data(None)
    assert res_a is not None
    assert res_a["name"] == "Sample_A.csv"

    # 3. Fallback to first available if active_spectrum_name not in db
    st.session_state["active_spectrum_name"] = "Non_Existent.csv"
    res_first = _get_active_fticr_data(None)
    assert res_first is not None
    assert res_first["name"] == "Sample_A.csv"

    # 4. Empty db
    st.session_state["spectra_db"] = {}
    assert _get_active_fticr_data("Sample_A.csv") is None


def test_normalize_sample_id():
    """Verifies that sample IDs are normalized by stripping extensions and analytical technique suffixes."""
    assert chemo_agent._normalize_sample_id("Baikal_Pristine_01_fticr.csv") == "baikal_pristine_01"
    assert chemo_agent._normalize_sample_id("Baikal_Pristine_01_nmr.jdx") == "baikal_pristine_01"
    assert chemo_agent._normalize_sample_id("Baikal_Pristine_01_eem.dat") == "baikal_pristine_01"
    assert chemo_agent._normalize_sample_id("Sample-1-pygcms.txt") == "sample_1"


def test_match_sample_in_container():
    """Verifies multi-level sample matching in dictionaries and DataFrames."""
    dict_container = {
        "Baikal_Deep_01_nmr": {"fa": 0.35},
        "Other_Sample": {"fa": 0.12},
    }
    # Stem match
    m1 = chemo_agent.match_sample_in_container("Baikal_Deep_01_fticr.csv", dict_container)
    assert m1 is not None and m1["fa"] == 0.35

    # Single-item fallback
    single_dict = {"Lignin_Control": {"lignin_idx": 0.88}}
    m_single = chemo_agent.match_sample_in_container("Any_Query.csv", single_dict)
    assert m_single is not None and m_single["lignin_idx"] == 0.88

    # DataFrame match
    df_container = pd.DataFrame([
        {"Sample_ID": "Baikal_Deep_01_eem", "HIX": 8.5},
        {"Sample_ID": "Other_01", "HIX": 2.1},
    ])
    m_df = chemo_agent.match_sample_in_container("Baikal_Deep_01_fticr.csv", df_container)
    assert m_df is not None and m_df["HIX"] == 8.5


def test_harvest_and_cross_modal_synthesis():
    """Tests end-to-end multi-modal data harvesting, insight computation, and prompt synthesis."""
    fticr_df = pd.DataFrame({
        "m/z": [340.1, 420.2, 510.3],
        "H/C": [1.12, 0.95, 1.25],
        "O/C": [0.45, 0.60, 0.35],
        "AI": [0.35, 0.58, 0.20],
        "DBE": [11.0, 15.0, 9.0],
        "NOSC": [-0.1, 0.2, -0.3],
        "Hetero_Class": ["CHO", "CHON", "CHOS"],
    })

    spectra_db = {
        "Baikal_01.csv": {"assigned_df": fticr_df}
    }
    eem_samples = {
        "Baikal_01_eem": {"SUVA254": 3.4, "HIX": 8.1, "C1_Fulvic": 45.0, "C2_Humic_Lignin": 32.0, "C3_Protein": 8.0}
    }
    nmr_samples = {
        "Baikal_01_nmr": {"fa": 0.32, "aliphatic": 44.0, "aromatic": 25.0, "carboxyl": 14.0}
    }
    ftir_samples = {
        "Baikal_01_ftir": {"FTIR_Lignin_Idx": 0.42, "FTIR_Aromaticity_Idx": 0.52}
    }
    sec_samples = {
        "Baikal_01_sec": {"SEC_Mw": 3800.0, "SEC_Mn": 2100.0, "SEC_Polydispersity": 1.81}
    }
    thermal_samples = {
        "Baikal_01_thermal": {"py_guaiacols": 12.5, "tga_labile_pct": 52.0}
    }
    elemental_samples = {
        "Baikal_01_elem": {"c_pct": 52.4, "h_pct": 4.8, "n_pct": 1.5, "s_pct": 0.85, "o_pct": 40.4}
    }

    harvested = chemo_agent.harvest_multimodal_descriptors(
        sample_name="Baikal_01.csv",
        spectra_db=spectra_db,
        eem_samples=eem_samples,
        nmr_samples=nmr_samples,
        ftir_samples=ftir_samples,
        sec_samples=sec_samples,
        thermal_samples=thermal_samples,
        elemental_samples=elemental_samples,
    )

    assert len(harvested["active_modalities"]) == 7
    assert "fticr" in harvested["active_modalities"]
    assert "eem" in harvested["active_modalities"]
    assert "nmr" in harvested["active_modalities"]
    assert "ftir" in harvested["active_modalities"]
    assert "sec" in harvested["active_modalities"]
    assert "thermal" in harvested["active_modalities"]
    assert "elemental" in harvested["active_modalities"]

    # Compute insights
    insights = chemo_agent.compute_cross_modal_insights(harvested)
    assert len(insights) >= 2
    # Verify aggregation ratio (SEC Mw ~3800 / FT-ICR Mn ~423 Da -> ~9.0x)
    assert any("Коэффициент агрегации" in ins["status"] for ins in insights)

    # Prompt generation in Russian and English
    prompt_ru = chemo_agent.build_cross_modal_synthesis_prompt("Baikal_01.csv", harvested, lang="ru")
    assert "кросс-модальный аналитический синтез" in prompt_ru
    assert "FT-ICR MS" in prompt_ru
    assert "EEM-PARAFAC" in prompt_ru
    assert "ЯМР 13C" in prompt_ru
    assert "ATR-FTIR" in prompt_ru
    assert "SEC/GPC" in prompt_ru
    assert "CHNS-O" in prompt_ru
    assert "Коэффициент агрегации" in prompt_ru

    prompt_en = chemo_agent.build_cross_modal_synthesis_prompt("Baikal_01.csv", harvested, lang="en")
    assert "cross-modal analytical synthesis" in prompt_en


def test_get_gemini_api_key_explicit():
    """Explicit Gemini key takes precedence."""
    key = chemo_agent.get_gemini_api_key("gemini_explicit_test_key_999")
    assert key == "gemini_explicit_test_key_999"


def test_get_gemini_api_key_env_var(monkeypatch):
    """Environment variable GEMINI_API_KEY is read if explicit key is None."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_env_key_888")
    key = chemo_agent.get_gemini_api_key(None)
    assert key == "gemini_env_key_888"


def test_get_gemini_api_key_missing(monkeypatch):
    """Returns None when no key is present anywhere."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("os.path.exists", return_value=False):
        key = chemo_agent.get_gemini_api_key(None)
        assert key is None


def test_save_gemini_api_key_to_env(tmp_path, monkeypatch):
    """save_gemini_api_key_to_env writes key into .env and updates os.environ."""
    test_env = tmp_path / ".env"
    test_env.write_text("SOME_VAR=123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    ok = save_gemini_api_key_to_env("AQ_test_gemini_key_xyz")
    assert ok is True
    content = test_env.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=AQ_test_gemini_key_xyz" in content
    assert os.environ.get("GEMINI_API_KEY") == "AQ_test_gemini_key_xyz"


def test_get_available_gemini_models_success():
    """Dynamically parses and filters Gemini chat models."""
    mock_payload = {
        "data": [
            {"id": "models/gemini-2.5-flash"},
            {"id": "models/gemini-2.5-flash-lite"},
            {"id": "models/gemini-3.5-flash"},
            {"id": "models/gemini-embedding-001"},
            {"id": "models/gemini-3.1-flash-tts-preview"},
            {"id": "models/veo-3.1-generate-preview"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        models = chemo_agent.get_available_gemini_models("test_key")
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-flash-lite" in models
        assert "gemini-3.5-flash" in models
        # Excluded non-chat models
        assert not any("embedding" in m for m in models)
        assert not any("tts" in m for m in models)
        assert not any("veo" in m for m in models)


def test_call_gemini_chat_success():
    """Mock Gemini chat completion returns parsed message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gemini-2.5-flash",
        "choices": [
            {"message": {"role": "assistant", "content": "Молекулярный индекс $H/C = 1.25$."}}
        ],
        "usage": {"total_tokens": 42},
    }

    with patch("requests.post", return_value=mock_resp):
        res = chemo_agent.call_gemini_chat(
            messages=[{"role": "user", "content": "test"}],
            api_key="valid_gemini_key",
            model="gemini-2.5-flash",
        )
        assert res["success"] is True
        assert "$H/C = 1.25$" in res["content"]
        assert res["model"] == "gemini-2.5-flash"


def test_call_gemini_chat_missing_key(monkeypatch):
    """call_gemini_chat fails gracefully if key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("os.path.exists", return_value=False):
        res = chemo_agent.call_gemini_chat([{"role": "user", "content": "hi"}], api_key=None)
        assert res["success"] is False
        assert "GEMINI_API_KEY не найден" in res["error"]


def test_stream_gemini_chat_mock():
    """stream_gemini_chat handles SSE data chunks correctly."""
    lines = [
        b'data: {"choices":[{"delta":{"content":"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}}]}',
        b'data: {"choices":[{"delta":{"content":" \xd0\xbc\xd0\xb8\xd1\x80"}}]}',
        b"data: [DONE]",
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines

    with patch("requests.post", return_value=mock_resp):
        tokens = list(chemo_agent.stream_gemini_chat([{"role": "user", "content": "hi"}], api_key="gem_key"))
        assert "".join(tokens) == "Привет мир"


def test_format_gemini_error():
    """format_gemini_error builds clean collapsible markdown for various error codes."""
    err_429 = chemo_agent.format_gemini_error(429, '{"error": {"message": "Resource has been exhausted (e.g. check quota)."}}')
    assert "Превышен лимит запросов Google Gemini API" in err_429

    err_401 = chemo_agent.format_gemini_error(401, '{"error": {"message": "API key not valid."}}')
    assert "Неверный GEMINI_API_KEY" in err_401

    err_503 = chemo_agent.format_gemini_error(503, '{"error": {"message": "high demand"}}')
    assert "Высокая нагрузка на модель Google Gemini" in err_503


def test_resolve_provider():
    """resolve_provider accurately routes provider based on explicit choice, model prefix, or credentials."""
    assert chemo_agent.resolve_provider("gemini") == "gemini"
    assert chemo_agent.resolve_provider("groq") == "groq"
    assert chemo_agent.resolve_provider("auto", model="gemini-2.5-flash") == "gemini"
    assert chemo_agent.resolve_provider("auto", model="models/gemini-3.5-flash") == "gemini"
    assert chemo_agent.resolve_provider("auto", model="qwen/qwen3.8-27b") == "groq"
    assert chemo_agent.resolve_provider("auto", model="openai/gpt-oss-120b") == "groq"


def test_unified_call_and_stream_chat(monkeypatch):
    """call_chat and stream_chat route properly to Gemini or Groq."""
    with patch("modules.chemo_agent.call_gemini_chat", return_value={"success": True, "content": "gemini reply"}) as mock_gem:
        res = chemo_agent.call_chat([{"role": "user", "content": "hi"}], provider="gemini", api_key="k", model="gemini-2.5-flash")
        assert res["content"] == "gemini reply"
        assert mock_gem.called

    with patch("modules.chemo_agent.call_groq_chat", return_value={"success": True, "content": "groq reply"}) as mock_groq:
        res = chemo_agent.call_chat([{"role": "user", "content": "hi"}], provider="groq", api_key="k", model="qwen/qwen3.8-27b")
        assert res["content"] == "groq reply"
        assert mock_groq.called




