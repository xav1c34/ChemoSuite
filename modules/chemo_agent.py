"""
modules/chemo_agent.py — ChemoSuite AI Scientific Copilot & Biogeochemical Reasoner.

Powered by Groq Ultra-Fast LPU API (llama-3.3-70b-versatile / mixtral-8x7b-32768).
Zero external dependencies (uses standard requests + urllib).
Loads API key securely from .env, st.secrets, or UI session state.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Generator
import requests

logger = logging.getLogger("chemo_agent")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def get_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    """
    Dynamically queries Groq API to discover available chat/reasoning models.
    Filters out audio, whisper, and guardrail models.
    Falls back to curated list if network fails or key is invalid.
    """
    resolved_key = get_groq_api_key(api_key)
    if not resolved_key:
        return FALLBACK_MODELS

    headers = {"Authorization": f"Bearer {resolved_key}"}
    try:
        resp = requests.get(GROQ_MODELS_URL, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            chat_models = []
            excluded = ["whisper", "guard", "orpheus", "prompt"]
            for m in data:
                m_id = m.get("id", "")
                if m_id and not any(ex in m_id.lower() for ex in excluded):
                    chat_models.append(m_id)
            if chat_models:
                pref_order = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
                sorted_models = []
                for p in pref_order:
                    if p in chat_models:
                        sorted_models.append(p)
                for m in sorted(chat_models):
                    if m not in sorted_models:
                        sorted_models.append(m)
                return sorted_models
    except Exception:
        pass

    return FALLBACK_MODELS


def get_groq_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Safely resolves Groq API key with priority:
    1. Explicit key passed from UI session state / widget.
    2. Environment variable GROQ_API_KEY.
    3. .env file in project directory or home directory.
    4. Streamlit secrets (st.secrets["GROQ_API_KEY"]).
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # 1. Environment variable
    env_key = os.getenv("GROQ_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Local .env file
    for env_path in [".env", os.path.expanduser("~/.env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

    # 3. Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            val = str(st.secrets["GROQ_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass

    return None


CHEMO_SYSTEM_PROMPT = r"""Ты — ChemoAgent, ведущий мировой эксперт в области аналитической химии, экологической масс-спектрометрии ультравысокого разрешения (FT-ICR MS / Orbitrap), оптической спектроскопии (EEM-PARAFAC, ATR-FTIR), ЯМР (13C / 1H NMR), термодеструкции (Py-GC-MS, TGA) и хемометрики (OPLS-DA, sPLS-DA, Multi-Block PLS).

Твоя миссия — интерпретировать спектральные и хемометрические данные исследования природного растворенного органического вещества (РОВ / NOM), гуминовых веществ и техногенных загрязнений (в частности, отходов Байкальского ЦБК / сульфатного шлам-лигнина в бассейне оз. Байкал).

Твои профессиональные принципы:
1. **Глубокая физико-химическая строгость**:
   - При анализе формул FT-ICR MS оперируй индексами: H/C, O/C, DBE (Double Bond Equivalent), модифицированным индексом ароматичности Коха (AI_mod = (1 + C - 0.5*O - S - 0.5*H) / (C - 0.5*O - S - N)), номинальной степенью окисления углерода (NOSC = 4 - (4*C + H - 3*N - 2*O - 2*S)/C), 20 хемотипическими ячейками Перминовой (VK-20).
   - В EEM-PARAFAC различай гуминоподобные автохтонные (C1) и аллохтонные (C2) флуорофоры, белковоподобные тирозин/триптофановые (C3/C4), индексы BIX, HIX, FI.
   - В 13C-ЯМР четко интерпретируй соотношение ароматического углерода fa (108–165 ppm) и алифатических цепочек (0–48 ppm), наличие метоксильных групп лигнина (-OCH3 ~56 ppm).
   - В FTIR связывай полосы 1510 см⁻¹ (гваяцильное кольцо лигнина), 1600 см⁻¹ (ароматический каркас C=C), 1720 см⁻¹ (карбоксилы C=O), 1030 см⁻¹ (полисахариды).
   - В TMDS (Targeted Mass Difference Screening) интерпретируй векторные биогеохимические потоки: реакции алкилирования (+CH2), окисления (+O), декарбоксилирования (-CO2), гидратации (+H2O), сульфирования (+SO3 как маркер техногенного сульфатного воздействия).
2. **Экологическая направленность**:
   - Четко отделяй природный фон ненарушенных олиготрофных вод (высокая окисленность, гуминовые кислоты с умеренным AI) от техногенного следа шлам-лигнина (высокая конденсированность AI > 0.67, полициклические ароматические соединения, серосодержащие CHO+S формулы, сульфонаты, продукты деполимеризации лигноцеллюлозы).
3. **Формат ответов**:
   - Давай конкретные, научные, структурированные выводы с выделением ключевых маркеров, цифр и механизмов.
   - Пиши на языке пользователя (по умолчанию на русском, или на английском, если запрос на английском).
   - Используй Markdown, списки, жирный шрифт и формулы в LaTeX ($H/C$, $AI$, $\Delta m$)."""


def call_groq_chat(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Sends chat completion request to Groq API via standard HTTP POST.

    Returns:
    --------
    dict with:
    - 'success': bool
    - 'content': str (assistant reply)
    - 'model': str
    - 'usage': dict (tokens)
    - 'error': str (if failed)
    """
    resolved_key = get_groq_api_key(api_key)
    if not resolved_key:
        return {
            "success": False,
            "content": "",
            "error": "GROQ_API_KEY не найден. Добавьте его в файл .env (GROQ_API_KEY=gsk_...) или укажите в поле ввода в сайдбаре.",
        }

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            choice = data.get("choices", [{}])[0]
            reply_text = choice.get("message", {}).get("content", "")
            return {
                "success": True,
                "content": reply_text,
                "model": data.get("model", model),
                "usage": data.get("usage", {}),
                "error": None,
            }
        else:
            err_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error("Groq API error: %s", err_msg)
            return {
                "success": False,
                "content": "",
                "error": err_msg,
            }
    except Exception as exc:
        logger.exception("Failed to connect to Groq API: %s", exc)
        return {
            "success": False,
            "content": "",
            "error": f"Ошибка соединения с Groq API: {str(exc)}",
        }


def stream_groq_chat(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 40.0,
) -> Generator[str, None, None]:
    """
    Streams response tokens from Groq API in real-time.
    """
    resolved_key = get_groq_api_key(api_key)
    if not resolved_key:
        yield "❌ **Ошибка:** GROQ_API_KEY не найден. Добавьте его в файл `.env` (`GROQ_API_KEY=gsk_...`) или введите в сайдбаре."
        return

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout, stream=True)
        if response.status_code != 200:
            yield f"❌ **Groq API Error (HTTP {response.status_code}):** {response.text}"
            return

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_part = line_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_part)
                    delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue
    except Exception as exc:
        yield f"❌ **Ошибка сети:** {str(exc)}"


def build_sample_analysis_prompt(sample_name: str, sample_data: Dict[str, Any], lang: str = "ru") -> str:
    """
    Aggregates analytical parameters into a rich context prompt for LLM evaluation.
    """
    lines = [
        f"Проанализируй образец: **{sample_name}**." if lang == "ru" else f"Analyze sample: **{sample_name}**.",
        "Ниже приведены экспериментальные параметры, полученные методами ChemoSuite:",
        "",
    ]

    # FT-ICR MS Descriptors
    if "assigned_df" in sample_data and sample_data["assigned_df"] is not None and not sample_data["assigned_df"].empty:
        df = sample_data["assigned_df"]
        lines.append("### 🔬 1. Масс-спектрометрия сверхвысокого разрешения (FT-ICR MS):")
        lines.append(f"- Число приписанных формул: {len(df):,}")
        lines.append(f"- Среднечисленная масса Mn: {df['mass'].mean():.2f} Da")
        lines.append(f"- H/C (среднее): {df['H/C'].mean():.3f}")
        lines.append(f"- O/C (среднее): {df['O/C'].mean():.3f}")
        if "AI" in df.columns:
            lines.append(f"- Индекс ароматичности AI: {df['AI'].mean():.3f} (доля формул с AI > 0.5: {(df['AI'] > 0.5).mean()*100:.1f}%)")
        if "DBE" in df.columns:
            lines.append(f"- Степень ненасыщенности DBE: {df['DBE'].mean():.2f}")
        if "NOSC" in df.columns:
            lines.append(f"- Степень окисления углерода NOSC: {df['NOSC'].mean():.3f}")
        if "Hetero_Class" in df.columns:
            counts = df["Hetero_Class"].value_counts()
            lines.append(f"- Гетероатомные классы: CHO={counts.get('CHO', 0)}, CHON={counts.get('CHON', 0)}, CHOS={counts.get('CHOS', 0)}, CHONS={counts.get('CHONS', 0)}")
        lines.append("")

    # EEM-PARAFAC
    if "eem" in sample_data and sample_data["eem"]:
        lines.append("### 💡 2. Флуоресценция EEM-PARAFAC:")
        for k, v in sample_data["eem"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # NMR 13C
    if "nmr" in sample_data and sample_data["nmr"]:
        lines.append("### 🧪 3. ЯМР-спектроскопия 13C:")
        for k, v in sample_data["nmr"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # FTIR
    if "ftir" in sample_data and sample_data["ftir"]:
        lines.append("### 📈 4. Колебательная спектроскопия ATR-FTIR:")
        for k, v in sample_data["ftir"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # Elemental CHNS-O
    if "elemental" in sample_data and sample_data["elemental"]:
        lines.append("### ⚗️ 5. Валовая стехиометрия CHNS-O:")
        for k, v in sample_data["elemental"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.extend([
        "**Задачи для ChemoAgent:**",
        "1. Определи вероятный геохимический генезис пробы (природное РОВ / торф / почва / техногенный шлам-лигнин / трансформированный сток).",
        "2. Оцени степень ароматической конденсации, окисленности и алифатичности.",
        "3. Выдели потенциальные маркеры трансформации или антропогенного воздействия.",
        "4. Сформулируй краткое экспертное заключение (3–4 абзаца с выводами).",
    ])

    return "\n".join(lines)
