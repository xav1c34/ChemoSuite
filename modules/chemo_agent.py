"""
modules/chemo_agent.py — ChemoSuite AI Scientific Copilot & Biogeochemical Reasoner.

Powered by Polza.ai API (OpenAI-compatible) using DeepSeek v4.1 Flash (deepseek/deepseek-v4.1-flash-1017).
Built-in web integration: works out-of-the-box for all users via server environment/secrets.
Hard budget guard: max 100 requests per day, token capping (2500 max), anti-spam rate limiting.
Zero external heavy dependencies (uses standard requests).
"""

import math
import os
import re
import json
import logging
import datetime
import threading
from typing import Dict, Any, List, Optional, Generator, Tuple
import requests
import pandas as pd

logger = logging.getLogger("chemo_agent")

POLZA_API_URL = "https://polza.ai/api/v1/chat/completions"
POLZA_MODELS_URL = "https://polza.ai/api/v1/models"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
AVAILABLE_MODELS = [
    "deepseek/deepseek-v4.1-flash-1017",
    "deepseek/deepseek-chat",
]
FALLBACK_MODELS = AVAILABLE_MODELS

DEFAULT_MAX_TOKENS = 2048
MAX_TOKEN_CAP = 2500
DAILY_REQUEST_LIMIT = 100
SESSION_REQUEST_LIMIT = 20  # один посетитель не может израсходовать общий дневной лимит
USAGE_FILE = ".agent_usage.json"

# Сессии Streamlit — потоки одного процесса: счётчик читается и пишется под блокировкой.
# globals().get сохраняет блокировку при importlib.reload модуля.
_USAGE_LOCK = globals().get("_USAGE_LOCK") or threading.Lock()


def _get_today_str() -> str:
    """Returns today's date formatted as YYYY-MM-DD."""
    return datetime.date.today().isoformat()


def get_daily_usage(max_limit: int = DAILY_REQUEST_LIMIT) -> Dict[str, Any]:
    """
    Returns current daily usage statistics:
    {'date': 'YYYY-MM-DD', 'count': int, 'limit': int, 'remaining': int}
    """
    today = _get_today_str()
    data = {
        "date": today,
        "count": 0,
        "limit": max_limit,
        "remaining": max_limit,
    }
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if saved.get("date") == today:
                    cnt = int(saved.get("count", 0))
                    data["count"] = cnt
                    data["remaining"] = max(0, max_limit - cnt)
        except Exception:
            pass
    return data


def check_and_increment_daily_usage(max_limit: int = DAILY_REQUEST_LIMIT) -> Tuple[bool, int, str]:
    """
    Budget Guard: Checks if today's request limit has been reached.
    If under limit, increments and persists to disk.
    Returns (allowed: bool, current_count: int, message: str).
    """
    session_state = _get_session_state()
    if session_state is not None:
        sess_cnt = int(session_state.get("_agent_session_requests", 0))
        if sess_cnt >= SESSION_REQUEST_LIMIT:
            msg = f"⚠️ Превышен лимит сессии: {SESSION_REQUEST_LIMIT} запросов. Общий дневной лимит защищён от исчерпания одним пользователем."
            return False, sess_cnt, msg

    today = _get_today_str()
    with _USAGE_LOCK:
        usage = get_daily_usage()
        if usage["count"] >= max_limit:
            msg = f"⚠️ Превышен защитный лимит: {max_limit} запросов в день. Лимит обновится в полночь для защиты баланса."
            return False, usage["count"], msg

        new_count = usage["count"] + 1
        try:
            tmp_file = f"{USAGE_FILE}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"date": today, "count": new_count, "limit": max_limit}, f, indent=2)
            os.replace(tmp_file, USAGE_FILE)
        except Exception as exc:
            logger.warning("Could not persist daily usage: %s", exc)

    if session_state is not None:
        session_state["_agent_session_requests"] = int(session_state.get("_agent_session_requests", 0)) + 1

    return True, new_count, "OK"


def _get_session_state():
    """Возвращает st.session_state текущей сессии или None вне Streamlit (тесты, скрипты)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return None
        import streamlit as st
        return st.session_state
    except Exception:
        return None


def get_polza_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Safely resolves Polza.ai API key with priority:
    1. Explicit key passed from UI session state / widget.
    2. Environment variable POLZA_AI_KEY.
    3. Streamlit secrets (st.secrets["POLZA_AI_KEY"]) for web hosting.
    4. Local .env file in project directory or home directory.
    """
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()

    # 1. Environment variable
    env_key = os.getenv("POLZA_AI_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "POLZA_AI_KEY" in st.secrets:
            val = str(st.secrets["POLZA_AI_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass

    # 3. Local .env file
    for env_path in [".env", os.path.expanduser("~/.env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("POLZA_AI_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

    return None


def get_available_models(api_key: Optional[str] = None) -> List[str]:
    """Returns curated list of chat/reasoning models supported by Polza.ai."""
    return AVAILABLE_MODELS



CHEMO_SYSTEM_PROMPT = r"""Ты — Chemo Agent, ведущий мировой эксперт в области аналитической химии, экологической масс-спектрометрии ультравысокого разрешения (FT-ICR MS / Orbitrap), оптической спектроскопии (EEM-PARAFAC, ATR-FTIR), ЯМР (13C / 1H NMR), термодеструкции (Py-GC-MS, TGA) и хемометрики (OPLS-DA, sPLS-DA, Multi-Block PLS).

Твоя миссия — интерпретировать спектральные и хемометрические данные исследования природного растворенного органического вещества (РОВ / NOM), гуминовых веществ и техногенных загрязнений (в частности, отходов Байкальского ЦБК / сульфатного шлам-лигнина в бассейне оз. Байкал).

Твои профессиональные принципы:
1. **Глубокая физико-химическая строгость**:
   - При анализе формул FT-ICR MS оперируй индексами: H/C, O/C, DBE (Double Bond Equivalent), модифицированным индексом ароматичности Коха и Диттмара (AI_mod = (1 + C - 0.5*O - S - 0.5*(H + N + P)) / (C - 0.5*O - S - N - P); AI_mod > 0.5 — ароматические, ≥ 0.67 — конденсированные ароматические структуры), номинальной степенью окисления углерода (NOSC = 4 - (4*C + H - 3*N - 2*O - 2*S)/C), 20 хемотипическими ячейками Перминовой (VK-20).
   - В EEM-PARAFAC различай гуминоподобные автохтонные (C1) и аллохтонные (C2) флуорофоры, белковоподобные тирозин/триптофановые (C3/C4), индексы BIX, HIX, FI.
   - В 13C-ЯМР четко интерпретируй соотношение ароматического углерода fa (108–165 ppm) и алифатических цепочек (0–48 ppm), наличие метоксильных групп лигнина (-OCH3 ~56 ppm).
   - В FTIR связывай полосы 1510 см⁻¹ (гваяцильное кольцо лигнина), 1600 см⁻¹ (ароматический каркас C=C), 1720 см⁻¹ (карбоксилы C=O), 1030 см⁻¹ (полисахариды).
   - В TMDS (Targeted Mass Difference Screening) интерпретируй векторные биогеохимические потоки: реакции алкилирования (+CH2), окисления (+O), декарбоксилирования (-CO2), гидратации (+H2O), сульфирования (+SO3 как маркер техногенного сульфатного воздействия).
2. **Экологическая направленность**:
   - Четко отделяй природный фон ненарушенных олиготрофных вод (высокая окисленность, гуминовые кислоты с умеренным AI) от техногенного следа шлам-лигнина (высокая конденсированность AI > 0.67, полициклические ароматические соединения, серосодержащие CHO+S формулы, сульфонаты, продукты деполимеризации лигноцеллюлозы).
3. **Формат ответов и математика (LaTeX / KaTeX)**:
   - Давай конкретные, научные, структурированные выводы с выделением ключевых маркеров, цифр и механизмов.
   - Пиши на языке пользователя (по умолчанию на русском, или на английском, если запрос на английском).
   - Для строчных формул используй СТРОГО одиночные доллары БЕЗ пробелов внутри: `$H/C$`, `$O/C$`, `$AI_{\text{mod}}$`, `$NOSC$`, `$f_a$`, `$\Delta m$`, `$C_{10}H_{12}O_5$`. Никогда не используй скобки вида \(...\).
   - Для блочных выключных формул используй СТРОГО двойные доллары:
     $$
     AI_{\text{mod}} = \frac{1 + C - 0.5O - S - 0.5H}{C - 0.5O - S - N}
     $$
     Никогда не используй скобки вида \[...\].
   - Подстрочные текстовые индексы оформляй через \\text{}, например $AI_{\text{mod}}$, $f_{\text{a}}$.
   - Не ставь пробелы сразу после открывающего $ и перед закрывающим $.
4. **Служебная идентичность**:
   - Ты — специализированный научно-аналитический ИИ-ассистент Chemo Agent платформы ChemoSuite Pro.
   - Ты работаешь на базе современной нейросетевой модели глубокого анализа и рассуждений (DeepSeek v4.1).
   - При любых вопросах пользователя о твоей модели или создателе сообщай, что ты — встроенный Chemo Agent платформы ChemoSuite Pro на базе DeepSeek v4.1."""


def repair_mojibake(text: str) -> str:
    """
    Repairs unintended Latin-1 / CP1252 misinterpretations of UTF-8 multi-byte characters:
    - 'â\x80\x94' / 'â\u2261\u2261' / 'â€”' -> '—' (em-dash)
    - 'â\x80\x93' / 'â€“' -> '–' (en-dash)
    - 'â‰ˆ' -> '≈' (almost equal)
    - 'â‰¤' -> '≤', 'â‰¥' -> '≥'
    - 'Â°' -> '°'
    """
    if not text or ("â" not in text and "Â" not in text and "\xe2" not in text):
        return text

    # 1. Try standard CP1252 / Latin-1 round-trip decode
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if fixed and len(fixed) >= len(text) * 0.7:
                return fixed
        except Exception:
            pass

    # 2. Targeted fallback replacements for glyph-corrupted font outputs (like â≡≡)
    text = re.sub(r'â[\u2261\u2262\u2260\x80-\x9f]{2}', '—', text)
    text = re.sub(r'\xe2[\u20ac\x80-\x9f][\u2014\u2013\u201d\u201c\x80-\x9f]', '—', text)
    text = text.replace("â€”", "—").replace("â€“", "–").replace("â‰ˆ", "≈")
    text = text.replace("â‰¤", "≤").replace("â‰¥", "≥").replace("Â°", "°")
    return text


def normalize_latex(text: str) -> str:
    """
    Normalizes LaTeX syntax so Streamlit KaTeX renders it cleanly:
    1. Repairs accidental mojibake (e.g. â€” -> —)
    2. Converts block math \\[ ... \\] to \\n\\n$$\\n...\\n$$\\n\\n
    3. Converts inline math \\( ... \\) to $...$
    4. Normalizes single dollar math with inner whitespace: '$ H/C $' -> '$H/C$'
    5. Normalizes subscript in AI_mod: '$AI_{mod}$' -> '$AI_{\\text{mod}}$'
    """
    if not text:
        return ""
    text = repair_mojibake(text)
    # 1. Block math: \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n\n$$\n\1\n$$\n\n', text, flags=re.DOTALL)
    # 2. Inline math: \( ... \) -> $...$
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)
    # 3. Strip inner spaces inside $ ... $ (e.g. '$ H/C $' -> '$H/C$')
    text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', lambda m: f"${m.group(1).strip()}$", text)
    # 4. Standardize AI_mod subscript to \text{mod} for proper text formatting in KaTeX
    text = re.sub(r'\$AI_\{?mod\}?\$', r'$AI_{\\text{mod}}$', text)
    return text


def format_polza_error(status_code: int, raw_text: str) -> str:
    """
    Formats Polza.ai API errors into a concise, polite, human-readable notification.
    Hides raw technical JSON dumps inside an interactive collapsible <details> block.
    """
    msg = ""
    try:
        data = json.loads(raw_text)
        err = data.get("error", {})
        msg = err.get("message", "")
    except Exception:
        msg = str(raw_text)

    msg_lower = msg.lower()
    if status_code == 402 or "balance" in msg_lower or "резерв" in msg_lower or "funds" in msg_lower:
        header = "💳 **Сервис временно недоступен (HTTP 402)**"
        hint = "Лимит тарифного плана на сервере исчерпан. Пожалуйста, обратитесь к администратору платформы."
    elif status_code == 429 or "rate" in msg_lower or "too_many_requests" in msg_lower:
        header = "⚠️ **Превышена частота запросов к ИИ-сервису (HTTP 429)**"
        hint = "Слишком много запросов в минуту. Пожалуйста, подождите несколько секунд и повторите."
    elif status_code in (401, 403) or "unauthorized" in msg_lower or "api_key" in msg_lower:
        header = "❌ **Ошибка авторизации сервиса ИИ (HTTP 401/403)**"
        hint = "Серверный ключ доступа недействителен или не настроен. Обратитесь к администратору."
    elif status_code == 404:
        header = "❌ **Модель ИИ временно недоступна (HTTP 404)**"
        hint = "Выбранная модель анализа временно недоступна на сервере."
    elif status_code in (500, 502, 503):
        header = f"⚠️ **Сервер ИИ временно недоступен (HTTP {status_code})**"
        hint = "Вычислительный узел ИИ временно перегружен. Повторите запрос через минуту."
    else:
        header = f"❌ **Ошибка сервиса ИИ (HTTP {status_code})**"
        hint = "Не удалось завершить генерацию ответа. Попробуйте повторить запрос."

    clean_snippet = str(raw_text).strip().replace("```", "'''")
    if len(clean_snippet) > 350:
        clean_snippet = clean_snippet[:350] + "..."

    return (
        f"{header}\n\n"
        f"*{hint}*\n\n"
        f"<details><summary style='font-size:12px; color:#94a3b8; cursor:pointer;'>🔍 Технические подробности ошибки</summary>\n\n"
        f"```json\n{clean_snippet}\n```\n"
        f"</details>"
    )


def call_polza_chat(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 40.0,
) -> Dict[str, Any]:
    """
    Sends chat completion request to Polza.ai (OpenAI-compatible) API.
    Enforces daily budget ceiling (100 req/day) and hard token cap (2500 tokens).
    """
    # 1. Budget Guard
    allowed, cur_cnt, guard_msg = check_and_increment_daily_usage(DAILY_REQUEST_LIMIT)
    if not allowed:
        return {
            "success": False,
            "content": "",
            "error": guard_msg,
        }

    # 2. Resolve Key
    resolved_key = get_polza_api_key(api_key)
    if not resolved_key:
        return {
            "success": False,
            "content": "",
            "error": "❌ POLZA_AI_KEY не найден в файле `.env` или `st.secrets`. Добавьте ключ для работы Chemo Agent.",
        }

    target_model = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    clamped_tokens = min(int(max_tokens or DEFAULT_MAX_TOKENS), MAX_TOKEN_CAP)
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": clamped_tokens,
    }

    try:
        response = requests.post(POLZA_API_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            choice = data.get("choices", [{}])[0]
            reply_text = choice.get("message", {}).get("content", "")
            return {
                "success": True,
                "content": reply_text,
                "model": data.get("model", target_model),
                "usage": data.get("usage", {}),
                "error": None,
            }
        else:
            err_msg = format_polza_error(response.status_code, response.text)
            logger.error("Polza.ai API error: HTTP %d", response.status_code)
            return {
                "success": False,
                "content": "",
                "error": err_msg,
            }
    except Exception as exc:
        logger.exception("Failed to connect to Polza.ai API: %s", exc)
        return {
            "success": False,
            "content": "",
            "error": f"Ошибка соединения с Polza.ai API: {str(exc)}",
        }


def stream_polza_chat(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 60.0,
) -> Generator[str, None, None]:
    """
    Streams response tokens from Polza.ai in real-time.
    Enforces daily budget ceiling (100 req/day) and hard token cap (2500 tokens).
    Smoothly filters out raw reasoning chunks and yields clean scientific content.
    """
    # 1. Budget Guard
    allowed, cur_cnt, guard_msg = check_and_increment_daily_usage(DAILY_REQUEST_LIMIT)
    if not allowed:
        yield guard_msg
        return

    # 2. Resolve Key
    resolved_key = get_polza_api_key(api_key)
    if not resolved_key:
        yield "❌ **Ошибка:** `POLZA_AI_KEY` не найден в файле `.env` или `st.secrets`."
        return

    target_model = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    clamped_tokens = min(int(max_tokens or DEFAULT_MAX_TOKENS), MAX_TOKEN_CAP)
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": clamped_tokens,
        "stream": True,
    }

    response = None
    yielded_any = False
    finish_reason = None
    try:
        response = requests.post(POLZA_API_URL, headers=headers, json=payload, timeout=(10.0, 90.0), stream=True)
        if response.status_code != 200:
            yield format_polza_error(response.status_code, response.text)
            return

        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            if line.startswith(b"data: "):
                data_part = line[6:].strip()
                if data_part == b"[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_part.decode("utf-8"))
                except Exception:
                    continue

                if "error" in chunk_json:
                    err_text = chunk_json.get("error", {}).get("message", str(chunk_json["error"]))
                    yield f"\n\n{format_polza_error(429 if 'rate' in err_text.lower() else 500, json.dumps(chunk_json))}"
                    return

                c0 = chunk_json.get("choices", [{}])[0]
                if c0.get("finish_reason"):
                    finish_reason = c0.get("finish_reason")
                delta = c0.get("delta", {})
                token = delta.get("content")
                if token:
                    yield repair_mojibake(token)
                    yielded_any = True

        if finish_reason == "length":
            yield "\n\n*(Ответ был ограничен защитным лимитом токенов модели)*"
    except Exception as exc:
        err_str = str(exc)
        logger.warning("Polza stream ended with exception: %s", err_str)
        if yielded_any and any(k in err_str.lower() for k in ["eof", "incompleteread", "connection broken"]):
            yield "\n\n*(Генерация завершена)*"
        else:
            yield f"\n\n❌ **Ошибка сети Polza.ai:** {err_str}"
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def call_chat(
    messages: List[Dict[str, str]],
    provider: str = "polza",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = 40.0,
) -> Dict[str, Any]:
    """Unified AI chat completion dispatcher routing directly to Polza.ai."""
    return call_polza_chat(
        messages=messages,
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        timeout=timeout,
    )


def stream_chat(
    messages: List[Dict[str, str]],
    provider: str = "polza",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> Generator[str, None, None]:
    """Unified real-time streaming dispatcher routing directly to Polza.ai."""
    yield from stream_polza_chat(
        messages=messages,
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        timeout=timeout,
    )


# Backward compatibility aliases
get_groq_api_key = get_polza_api_key
get_gemini_api_key = get_polza_api_key
resolve_provider = lambda *args, **kwargs: "polza"
format_groq_error = format_polza_error
format_gemini_error = format_polza_error



def build_sample_analysis_prompt(sample_name: str, sample_data: Dict[str, Any], lang: str = "ru") -> str:
    """
    Aggregates analytical parameters into a compact, high-density context prompt for LLM evaluation.
    Keeps token length minimal to fit comfortably within free-tier Groq rate limits.
    """
    if lang == "ru":
        lines = [
            f"Проанализируй образец: **{sample_name}**.",
            "Ниже приведены экспериментальные параметры ChemoSuite:",
            "",
        ]
    else:
        lines = [
            f"Analyze sample: **{sample_name}**.",
            "Below are experimental parameters from ChemoSuite:",
            "",
        ]

    # FT-ICR MS Descriptors
    if "assigned_df" in sample_data and sample_data["assigned_df"] is not None and not sample_data["assigned_df"].empty:
        raw_df = sample_data["assigned_df"]
        df = raw_df.copy() if hasattr(raw_df, "copy") else pd.DataFrame(raw_df)

        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # Auto-compute molecular descriptors if formulas (C, H) are present but H/C is not
        if ("c" in cols_lower and "h" in cols_lower) and not any(k in cols_lower for k in ["h/c", "hc", "h_c"]):
            try:
                import modules.fticr_core as fticr_core
                df = fticr_core.calculate_descriptors(df)
                cols_lower = {str(c).lower().strip(): c for c in df.columns}
            except Exception:
                pass

        has_formulas = any(k in cols_lower for k in ["h/c", "hc", "h_c", "formula", "formula_str", "c"])

        lines.append("🔬 **FT-ICR MS:**")
        if has_formulas:
            count_line = f"- Число формул: {len(df):,}"
        else:
            count_line = f"- Число детектированных пиков: {len(df):,}"

        # Gather metric snippets
        metrics = []

        # Mass / m/z
        mass_col = None
        for cand in ["mass", "m/z", "mz", "mw", "calculated_mass", "experimental_mass"]:
            if cand in cols_lower:
                mass_col = cols_lower[cand]
                break
        if mass_col:
            m_s = pd.to_numeric(df[mass_col], errors="coerce").dropna()
            if not m_s.empty:
                metrics.append(f"Mn: {m_s.mean():.1f} Da ({m_s.min():.0f}–{m_s.max():.0f})")

        # H/C
        hc_col = None
        for cand in ["h/c", "hc", "h_c"]:
            if cand in cols_lower:
                hc_col = cols_lower[cand]
                break
        if hc_col:
            hc_s = pd.to_numeric(df[hc_col], errors="coerce").dropna()
            if not hc_s.empty:
                metrics.append(f"H/C (среднее): {hc_s.mean():.2f}")

        # O/C
        oc_col = None
        for cand in ["o/c", "oc", "o_c"]:
            if cand in cols_lower:
                oc_col = cols_lower[cand]
                break
        if oc_col:
            oc_s = pd.to_numeric(df[oc_col], errors="coerce").dropna()
            if not oc_s.empty:
                metrics.append(f"O/C: {oc_s.mean():.2f}")

        # AI (Modified Aromaticity Index)
        ai_col = None
        # AI_mod (Koch & Dittmar) предпочтительнее: пороги 0.5 / 0.67 определены для него
        for cand in ["ai_mod", "ai", "ai_koch", "aromaticity_index"]:
            if cand in cols_lower:
                ai_col = cols_lower[cand]
                break
        if ai_col:
            ai_s = pd.to_numeric(df[ai_col], errors="coerce").dropna()
            if not ai_s.empty:
                high_ai = (ai_s > 0.5).mean() * 100.0
                condensed = (ai_s >= 0.67).mean() * 100.0
                ai_label = "AI_mod" if str(ai_col).lower() == "ai_mod" else "AI"
                metrics.append(f"{ai_label}: {ai_s.mean():.3f} (доля >0.5: {high_ai:.0f}%, конденс.: {condensed:.0f}%)")

        # DBE
        dbe_col = None
        for cand in ["dbe", "dbe-o"]:
            if cand in cols_lower:
                dbe_col = cols_lower[cand]
                break
        if dbe_col:
            dbe_s = pd.to_numeric(df[dbe_col], errors="coerce").dropna()
            if not dbe_s.empty:
                metrics.append(f"DBE: {dbe_s.mean():.1f}")

        # NOSC
        nosc_col = None
        for cand in ["nosc"]:
            if cand in cols_lower:
                nosc_col = cols_lower[cand]
                break
        if nosc_col:
            nosc_s = pd.to_numeric(df[nosc_col], errors="coerce").dropna()
            if not nosc_s.empty:
                metrics.append(f"NOSC: {nosc_s.mean():.2f}")

        if metrics:
            lines.append(f"{count_line} | " + ", ".join(metrics))
        else:
            lines.append(count_line)

        # Hetero_Class
        hetero_col = None
        for cand in ["hetero_class", "class", "heteroclass"]:
            if cand in cols_lower:
                hetero_col = cols_lower[cand]
                break
        if hetero_col and hetero_col in df.columns:
            counts = df[hetero_col].value_counts().to_dict()
            pools_str = ", ".join(f"{k}:{v}" for k, v in list(counts.items())[:5])
            lines.append(f"- Классы: {pools_str}")
        lines.append("")

    # EEM-PARAFAC
    if "eem" in sample_data and sample_data["eem"]:
        eem_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["eem"].items())[:5])
        lines.append(f"💡 **EEM-PARAFAC:** {eem_items}")

    # NMR 13C
    if "nmr" in sample_data and sample_data["nmr"]:
        nmr_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["nmr"].items())[:5])
        lines.append(f"🧪 **ЯМР 13C:** {nmr_items}")

    # FTIR
    if "ftir" in sample_data and sample_data["ftir"]:
        ftir_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["ftir"].items())[:5])
        lines.append(f"📈 **ATR-FTIR:** {ftir_items}")

    # SEC / GPC
    if "sec" in sample_data and sample_data["sec"]:
        sec_d = sample_data["sec"]
        if isinstance(sec_d, dict):
            sec_items = ", ".join(f"{k}: {v}" for k, v in list(sec_d.items())[:4])
            lines.append(f"📏 **SEC/GPC:** {sec_items}")

    # Thermal Analysis
    if "thermal" in sample_data and sample_data["thermal"]:
        therm_d = sample_data["thermal"]
        if isinstance(therm_d, dict):
            therm_items = ", ".join(f"{k}: {v}" for k, v in list(therm_d.items())[:4])
            lines.append(f"🔥 **Термодеструкция (Py-GC & TGA):** {therm_items}")

    # Elemental CHNS-O
    if "elemental" in sample_data and sample_data["elemental"]:
        elem_items = ", ".join(f"{k}: {v}" for k, v in sample_data["elemental"].items())
        lines.append(f"⚗️ **CHNS-O:** {elem_items}")

    # IRMS Stable Isotopes
    if "irms" in sample_data and sample_data["irms"]:
        irms_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["irms"].items())[:6])
        lines.append(f"🎯 **IRMS (Изотопы):** {irms_items}")

    # IC (Ion Chromatography)
    if "ic" in sample_data and sample_data["ic"]:
        ic_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["ic"].items())[:6])
        lines.append(f"💧 **IC (Ионная хроматография):** {ic_items}")

    # Raman Spectroscopy
    if "raman" in sample_data and sample_data["raman"]:
        raman_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["raman"].items())[:5])
        lines.append(f"💎 **Raman (D/G полосы):** {raman_items}")

    lines.append("")
    if lang == "ru":
        lines.append("**Задача:** Дай краткое экспертное заключение строго в 2–3 абзаца (до 250 слов): вероятный генезис (природное РОВ / торф / шлам-лигнин / сток), степень окисления/ароматизации и ключевые маркеры трансформации. Обязательно закончи вывод кратким резюме.")
    else:
        lines.append("**Task:** Provide a concise expert interpretation in strictly 2–3 short paragraphs (under 250 words): likely geochemical origin, oxidation/aromaticity state, and transformation markers. Conclude with a clear brief takeaway.")

    return "\n".join(lines)


# ==============================================================================
# КРОСС-МОДАЛЬНЫЙ СИНТЕЗ (CROSS-MODAL SYNTHESIS & MULTI-OMICS FUSION)
# ==============================================================================

def _is_finite_number(val: Any) -> bool:
    """True для конечного числа (NaN, None и строки — False): NaN в Python истинно в условиях."""
    try:
        return val is not None and math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _normalize_sample_id(s: str) -> str:
    """Normalizes sample identifier across different studio file naming schemes."""
    if not s:
        return ""
    base = str(s).rsplit(".", 1)[0].lower()
    for tag in [
        "_fticr", "-fticr", "_ft-icr", "-ft-icr",
        "_ms", "-ms", "_eem", "-eem",
        "_nmr", "-nmr", "_13c", "-13c", "_1h", "-1h",
        "_ftir", "-ftir", "_atr", "-atr",
        "_sec", "-sec", "_gpc", "-gpc",
        "_thermal", "-thermal", "_pygcms", "-pygcms", "_py-gcms", "-py-gcms", "_tga", "-tga",
        "_elem", "-elem", "_chns", "-chns", "_chnso", "-chnso",
        "_irms", "-irms", "_ic", "-ic", "_ion", "-ion", "_raman", "-raman"
    ]:
        base = base.replace(tag, "")
    base = re.sub(r'[\s_\-]+', '_', base).strip('_')
    return base


def match_sample_in_container(sample_name: str, container: Any) -> Optional[Any]:
    """
    Finds matching sample in dict or DataFrame with multi-level key resolution:
    1. Exact string match
    2. Base name match without extension
    3. Normalized ID match
    4. Single-sample fallback if container has exactly 1 element
    """
    if container is None:
        return None

    norm_target = _normalize_sample_id(sample_name)
    base_target = str(sample_name).rsplit(".", 1)[0].lower()

    # 1. Container is a dictionary (sample_id -> object or dict)
    if isinstance(container, dict):
        if not container:
            return None
        if sample_name in container:
            return container[sample_name]
        for k, v in container.items():
            if str(k).rsplit(".", 1)[0].lower() == base_target:
                return v
        for k, v in container.items():
            if _normalize_sample_id(str(k)) == norm_target:
                return v
        if len(container) == 1:
            return list(container.values())[0]
        return None

    # 2. Container is a pandas DataFrame
    if isinstance(container, pd.DataFrame):
        if container.empty:
            return None
        id_col = None
        for cand in ["Sample_ID", "sample_id", "Sample", "sample", "ID", "id"]:
            if cand in container.columns:
                id_col = cand
                break
        if id_col is not None:
            col_series = container[id_col].astype(str)
            m_exact = container[col_series.str.lower() == sample_name.lower()]
            if not m_exact.empty:
                return m_exact.iloc[0].to_dict()
            m_stem = container[col_series.apply(lambda x: str(x).rsplit(".", 1)[0].lower()) == base_target]
            if not m_stem.empty:
                return m_stem.iloc[0].to_dict()
            m_norm = container[col_series.apply(_normalize_sample_id) == norm_target]
            if not m_norm.empty:
                return m_norm.iloc[0].to_dict()
        if len(container) == 1:
            return container.iloc[0].to_dict()

    return None


def harvest_multimodal_descriptors(
    sample_name: str,
    spectra_db: Optional[Dict[str, Any]] = None,
    eem_samples: Optional[Any] = None,
    nmr_samples: Optional[Any] = None,
    ftir_samples: Optional[Any] = None,
    sec_samples: Optional[Any] = None,
    thermal_samples: Optional[Any] = None,
    elemental_samples: Optional[Any] = None,
    irms_samples: Optional[Any] = None,
    ic_samples: Optional[Any] = None,
    raman_samples: Optional[Any] = None,
    eem_ml: Optional[pd.DataFrame] = None,
    nmr_ml: Optional[pd.DataFrame] = None,
    ftir_ml: Optional[pd.DataFrame] = None,
    sec_ml: Optional[pd.DataFrame] = None,
    thermal_ml: Optional[pd.DataFrame] = None,
    irms_ml: Optional[pd.DataFrame] = None,
    ic_ml: Optional[pd.DataFrame] = None,
    raman_ml: Optional[pd.DataFrame] = None,
    eem_db: Optional[Dict[str, Any]] = None,
    nmr_db: Optional[Dict[str, Any]] = None,
    ftir_db: Optional[Dict[str, Any]] = None,
    elemental_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Harvests and unifies multi-omics analytical descriptors across all 11 ChemoSuite studios
    for a chosen sample name.
    """
    result: Dict[str, Any] = {
        "name": sample_name,
        "fticr": None,
        "eem": None,
        "nmr": None,
        "ftir": None,
        "sec": None,
        "thermal": None,
        "elemental": None,
        "irms": None,
        "ic": None,
        "raman": None,
        "active_modalities": [],
    }

    # 1. FT-ICR MS
    if spectra_db and sample_name in spectra_db:
        spec = spectra_db[sample_name]
        df_assigned = spec.get("assigned_df")
        if df_assigned is None:
            df_assigned = spec.get("raw_df", spec.get("parsed_peaks"))

        if df_assigned is not None and not df_assigned.empty:
            df = df_assigned.copy() if hasattr(df_assigned, "copy") else pd.DataFrame(df_assigned)
            cols_lower = {str(c).lower().strip(): c for c in df.columns}

            # Calculate basic metrics
            fticr_data: Dict[str, Any] = {
                "assigned_df": df,
                "n_peaks": len(df),
                "has_formulas": any(k in cols_lower for k in ["h/c", "hc", "formula", "formula_str", "c"]),
            }

            # Mn
            for cand in ["mass", "m/z", "mz", "mw", "calculated_mass"]:
                if cand in cols_lower:
                    s_m = pd.to_numeric(df[cols_lower[cand]], errors="coerce").dropna()
                    if not s_m.empty:
                        fticr_data["mn"] = round(float(s_m.mean()), 1)
                        fticr_data["mass_range"] = (round(float(s_m.min()), 0), round(float(s_m.max()), 0))
                    break

            # H/C
            for cand in ["h/c", "hc", "h_c"]:
                if cand in cols_lower:
                    s_hc = pd.to_numeric(df[cols_lower[cand]], errors="coerce").dropna()
                    if not s_hc.empty:
                        fticr_data["hc_mean"] = round(float(s_hc.mean()), 2)
                    break

            # O/C
            for cand in ["o/c", "oc", "o_c"]:
                if cand in cols_lower:
                    s_oc = pd.to_numeric(df[cols_lower[cand]], errors="coerce").dropna()
                    if not s_oc.empty:
                        fticr_data["oc_mean"] = round(float(s_oc.mean()), 2)
                    break

            # AI_mod
            for cand in ["ai_mod", "ai", "ai_koch", "aromaticity_index"]:
                if cand in cols_lower:
                    s_ai = pd.to_numeric(df[cols_lower[cand]], errors="coerce").dropna()
                    if not s_ai.empty:
                        fticr_data["ai_mod"] = round(float(s_ai.mean()), 3)
                        fticr_data["high_ai_pct"] = round(float((s_ai > 0.5).mean() * 100.0), 1)
                        fticr_data["condensed_pct"] = round(float((s_ai >= 0.67).mean() * 100.0), 1)
                    break

            # DBE
            for cand in ["dbe", "dbe-o"]:
                if cand in cols_lower:
                    s_dbe = pd.to_numeric(df[cols_lower[cand]], errors="coerce").dropna()
                    if not s_dbe.empty:
                        fticr_data["dbe_mean"] = round(float(s_dbe.mean()), 1)
                    break

            # NOSC
            if "nosc" in cols_lower:
                s_nosc = pd.to_numeric(df[cols_lower["nosc"]], errors="coerce").dropna()
                if not s_nosc.empty:
                    fticr_data["nosc_mean"] = round(float(s_nosc.mean()), 2)

            # Heteroatom Classes
            for cand in ["hetero_class", "class", "heteroclass"]:
                if cand in cols_lower and cols_lower[cand] in df.columns:
                    fticr_data["hetero_classes"] = df[cols_lower[cand]].value_counts().to_dict()
                    break

            result["fticr"] = fticr_data
            result["active_modalities"].append("fticr")

    # 2. EEM / UV-Vis
    eem_obj = match_sample_in_container(sample_name, eem_samples) or match_sample_in_container(sample_name, eem_ml) or (eem_db.get(sample_name) if eem_db else None)
    if eem_obj is not None:
        eem_data: Dict[str, Any] = {}
        if isinstance(eem_obj, dict):
            for k in ["FI", "fi", "HIX", "hix", "BIX", "bix", "SUVA254", "suva254", "DOC", "doc", "A254", "a254", "C1", "C2", "C3", "C4", "C1_Fulvic", "C2_Humic_Lignin", "C3_Protein"]:
                if k in eem_obj and pd.notnull(eem_obj[k]):
                    eem_data[k.lower()] = round(float(eem_obj[k]), 2)
        elif hasattr(eem_obj, "data"):
            try:
                import modules.eem_core as eem_core
                indices = eem_core.calculate_spectral_indices(eem_obj)
                for k, v in indices.items():
                    if pd.notnull(v):
                        eem_data[k.lower()] = round(float(v), 2)
            except Exception:
                pass
        if eem_data:
            result["eem"] = eem_data
            result["active_modalities"].append("eem")

    # 3. NMR 13C
    nmr_obj = match_sample_in_container(sample_name, nmr_samples) or match_sample_in_container(sample_name, nmr_ml) or (nmr_db.get(sample_name) if nmr_db else None)
    if nmr_obj is not None:
        nmr_data: Dict[str, Any] = {}
        if isinstance(nmr_obj, dict):
            for k in ["fa", "aliphatic", "carbohydrate", "aromatic", "phenolic", "carboxyl", "carbonyl", "c_alk_c_oalk", "hydrophobicity_ratio"]:
                if k in nmr_obj and pd.notnull(nmr_obj[k]):
                    nmr_data[k] = round(float(nmr_obj[k]), 2)
        elif hasattr(nmr_obj, "indices") or hasattr(nmr_obj, "integrals"):
            if getattr(nmr_obj, "indices", None):
                for k, v in nmr_obj.indices.items():
                    if pd.notnull(v):
                        nmr_data[k] = round(float(v), 2)
            if getattr(nmr_obj, "integrals", None):
                for k, v in nmr_obj.integrals.items():
                    if pd.notnull(v):
                        nmr_data[k] = round(float(v), 2)
        if nmr_data:
            result["nmr"] = nmr_data
            result["active_modalities"].append("nmr")

    # 4. ATR-FTIR
    ftir_obj = match_sample_in_container(sample_name, ftir_samples) or match_sample_in_container(sample_name, ftir_ml) or (ftir_db.get(sample_name) if ftir_db else None)
    if ftir_obj is not None:
        ftir_data: Dict[str, Any] = {}
        if isinstance(ftir_obj, dict):
            for k in ["FTIR_Aromaticity_Idx", "FTIR_Lignin_Idx", "FTIR_Oxidation_Idx", "FTIR_Branching_Idx", "FTIR_GS_Ratio", "FTIR_Polysaccharide_Idx", "FTIR_Phenolic_Idx", "aromaticity_idx", "lignin_idx", "oxidation_idx"]:
                if k in ftir_obj and pd.notnull(ftir_obj[k]):
                    clean_k = k.replace("FTIR_", "").lower()
                    ftir_data[clean_k] = round(float(ftir_obj[k]), 3)
        elif hasattr(ftir_obj, "indices"):
            for k, v in getattr(ftir_obj, "indices", {}).items():
                if pd.notnull(v):
                    ftir_data[k.lower()] = round(float(v), 3)
        if ftir_data:
            result["ftir"] = ftir_data
            result["active_modalities"].append("ftir")

    # 5. SEC / GPC
    sec_obj = match_sample_in_container(sample_name, sec_samples) or match_sample_in_container(sample_name, sec_ml)
    if sec_obj is not None:
        sec_data: Dict[str, Any] = {}
        if isinstance(sec_obj, dict):
            for k in ["SEC_Mw", "SEC_Mn", "SEC_Polydispersity", "SEC_Fraction_I_pct", "SEC_Fraction_II_pct", "SEC_Fraction_III_pct", "mw", "mn", "pdi"]:
                if k in sec_obj and pd.notnull(sec_obj[k]):
                    clean_k = k.replace("SEC_", "").lower()
                    sec_data[clean_k] = round(float(sec_obj[k]), 2)
        elif hasattr(sec_obj, "descriptors"):
            for k, v in getattr(sec_obj, "descriptors", {}).items():
                if pd.notnull(v):
                    sec_data[k.lower()] = round(float(v), 2)
        if sec_data:
            result["sec"] = sec_data
            result["active_modalities"].append("sec")

    # 6. Thermal Analysis (Py-GC/MS & TGA)
    therm_obj = match_sample_in_container(sample_name, thermal_samples) or match_sample_in_container(sample_name, thermal_ml)
    if therm_obj is not None:
        therm_data: Dict[str, Any] = {}
        if isinstance(therm_obj, dict):
            for k in ["py_phenols", "py_guaiacols", "py_syringols", "py_aromatics", "py_lipids", "py_polysaccharides", "tga_labile_pct", "tga_refractory_pct", "tga_char_pct", "T_max"]:
                if k in therm_obj and pd.notnull(therm_obj[k]):
                    therm_data[k] = round(float(therm_obj[k]), 1)
        elif hasattr(therm_obj, "py_gcms") or hasattr(therm_obj, "tga"):
            try:
                import modules.thermal_core as thermal_core
                desc = thermal_core.extract_thermal_descriptors(therm_obj)
                for k, v in desc.items():
                    if pd.notnull(v):
                        therm_data[k] = round(float(v), 1)
            except Exception:
                pass
        if therm_data:
            result["thermal"] = therm_data
            result["active_modalities"].append("thermal")

    # 7. Elemental CHNS-O
    elem_obj = match_sample_in_container(sample_name, elemental_samples) or (elemental_db.get(sample_name) if elemental_db else None)
    if elem_obj is not None:
        elem_data: Dict[str, Any] = {}
        if isinstance(elem_obj, dict):
            for k in ["c_pct", "h_pct", "n_pct", "s_pct", "o_pct", "c_n", "h_c", "o_c", "nosc"]:
                if k in elem_obj and pd.notnull(elem_obj[k]):
                    elem_data[k] = round(float(elem_obj[k]), 2)
        elif hasattr(elem_obj, "c_pct"):
            for k in ["c_pct", "h_pct", "n_pct", "s_pct", "o_pct", "h_c", "o_c", "n_c", "s_c", "nosc"]:
                v = getattr(elem_obj, k, None)
                if v is not None and pd.notnull(v):
                    elem_data[k] = round(float(v), 2)
        if elem_data:
            result["elemental"] = elem_data
            result["active_modalities"].append("elemental")

    # 8. IRMS Stable Isotopes
    irms_obj = match_sample_in_container(sample_name, irms_samples) or match_sample_in_container(sample_name, irms_ml)
    if irms_obj is not None:
        irms_data: Dict[str, Any] = {}
        if isinstance(irms_obj, dict):
            for k in ["d13C", "d15N", "d34S", "d2H", "d18O", "d_excess", "kraft_lignin_prob"]:
                if k in irms_obj and pd.notnull(irms_obj[k]):
                    irms_data[k] = round(float(irms_obj[k]), 2)
        elif hasattr(irms_obj, "d13C"):
            for k in ["d13C", "d15N", "d34S", "d2H", "d18O", "deuterium_excess", "kraft_lignin_prob"]:
                v = getattr(irms_obj, k, None)
                if v is not None and pd.notnull(v):
                    irms_data[k] = round(float(v), 2)
        if irms_data:
            result["irms"] = irms_data
            result["active_modalities"].append("irms")

    # 9. IC (Ion Chromatography)
    ic_obj = match_sample_in_container(sample_name, ic_samples) or match_sample_in_container(sample_name, ic_ml)
    if ic_obj is not None:
        ic_data: Dict[str, Any] = {}
        if isinstance(ic_obj, dict):
            for k in ["so4_mg_l", "cl_mg_l", "so4_cl_ratio", "total_lmwoa", "charge_balance_err_pct", "kraft_effluent_indicator", "acetate_mg_l", "oxalate_mg_l"]:
                if k in ic_obj and pd.notnull(ic_obj[k]):
                    ic_data[k] = round(float(ic_obj[k]), 2)
        elif hasattr(ic_obj, "so4_2minus"):
            for k, attr in [("so4_mg_l", "so4_2minus"), ("cl_mg_l", "cl_minus"), ("so4_cl_ratio", "so4_to_cl_ratio"), ("total_lmwoa", "total_lmwoa_mg_l"), ("cbe_pct", "charge_balance_error_pct"), ("kraft_effluent_indicator", "kraft_effluent_indicator")]:
                v = getattr(ic_obj, attr, None)
                if v is not None and pd.notnull(v):
                    ic_data[k] = round(float(v), 2)
        if ic_data:
            result["ic"] = ic_data
            result["active_modalities"].append("ic")

    # 10. Raman Spectroscopy
    raman_obj = match_sample_in_container(sample_name, raman_samples) or match_sample_in_container(sample_name, raman_ml)
    if raman_obj is not None:
        raman_data: Dict[str, Any] = {}
        if isinstance(raman_obj, dict):
            for k in ["id_ig_ratio", "ad_ag_ratio", "la_crystallite_nm", "fwhm_g", "carbon_order_index"]:
                if k in raman_obj and pd.notnull(raman_obj[k]):
                    raman_data[k] = round(float(raman_obj[k]), 2)
        elif hasattr(raman_obj, "id_ig_ratio"):
            for k in ["id_ig_ratio", "ad_ag_ratio", "la_crystallite_nm", "fwhm_g", "carbon_order_index"]:
                v = getattr(raman_obj, k, None)
                if v is not None and pd.notnull(v):
                    raman_data[k] = round(float(v), 2)
        if raman_data:
            result["raman"] = raman_data
            result["active_modalities"].append("raman")

    return result


def compute_cross_modal_insights(multimodal_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Computes rigorous analytical cross-modal correlations, identifying convergence
    and potential analytical divergence (such as ESI ionization selectivity vs bulk NMR).
    """
    insights: List[Dict[str, str]] = []
    fticr = multimodal_data.get("fticr") or {}
    eem = multimodal_data.get("eem") or {}
    nmr = multimodal_data.get("nmr") or {}
    ftir = multimodal_data.get("ftir") or {}
    sec = multimodal_data.get("sec") or {}
    thermal = multimodal_data.get("thermal") or {}
    elemental = multimodal_data.get("elemental") or {}
    irms = multimodal_data.get("irms") or {}
    ic = multimodal_data.get("ic") or {}
    raman = multimodal_data.get("raman") or {}

    # 1. Aromaticity & Condensation Coherence
    ai = fticr.get("ai_mod")
    fa = nmr.get("fa")
    ftir_arom = ftir.get("aromaticity_idx")
    suva = eem.get("suva254")

    active_arom = [v for v in [ai, fa, ftir_arom, suva] if v is not None]
    if len(active_arom) >= 2:
        if ai is not None and fa is not None and ai >= 0.45 and fa < 0.18:
            insights.append({
                "category": "Ароматичность / Селективность ESI(-)",
                "status": "⚠️ Аналитическое расхождение (Селективность ESI)",
                "detail": f"В FT-ICR MS наблюдается высокий индекс $AI_{{\\text{{mod}}}} = {ai:.3f}$, тогда как в объёмном ЯМР доля $f_a = {fa:.2f}$ низка. Это типичный признак селективной ионизации полифенолов в ESI(-) на фоне объемного алифатического матрикса.",
            })
        elif (ai is not None and ai >= 0.35) and (fa is None or fa >= 0.25) and (suva is None or suva >= 2.8):
            insights.append({
                "category": "Ароматичность / Конвергенция",
                "status": "✅ Высокая сходимость методов (Ароматический кор)",
                "detail": "FT-ICR, ЯМР и оптические спектры согласованно подтверждают развитый ароматический каркас высокой окислительной устойчивости.",
            })
        elif (ai is not None and ai < 0.28) and (fa is None or fa < 0.20):
            insights.append({
                "category": "Ароматичность / Конвергенция",
                "status": "✅ Высокая сходимость методов (Алифатическое/биогенное РОВ)",
                "detail": "Методы независимо подтверждают низкую ароматичность и преобладание алифатических структур.",
            })

    # 2. Supramolecular Aggregation State (SEC vs FT-ICR)
    sec_mw = sec.get("mw")
    fticr_mn = fticr.get("mn")
    if sec_mw and fticr_mn and fticr_mn > 0:
        ar = sec_mw / fticr_mn
        insights.append({
            "category": "Супрамолекулярная структура",
            "status": f"🧬 Коэффициент агрегации: {ar:.1f}x",
            "detail": f"Кажущаяся гидродинамическая масса в растворе (SEC $M_w = {sec_mw:.0f}$ Da) превышает массу индивидуальных ионизированных мономеров (FT-ICR $M_n = {fticr_mn:.0f}$ Da) в {ar:.1f} раз. Это подтверждает супрамолекулярную ассоциацию олигомеров РОВ.",
        })

    # 3. Technogenic Lignin & Sulfate Markers
    lignin_flags = []
    if fticr.get("hetero_classes", {}).get("CHOS", 0) > 0:
        lignin_flags.append(f"CHOS формулы ({fticr['hetero_classes']['CHOS']} шт)")
    if ftir.get("lignin_idx") and ftir["lignin_idx"] > 0.35:
        lignin_flags.append(f"FTIR полоса 1510 см⁻¹ (индекс {ftir['lignin_idx']:.2f})")
    if thermal.get("py_guaiacols") and thermal["py_guaiacols"] > 10.0:
        lignin_flags.append(f"Py-GC/MS гваяцилы ({thermal['py_guaiacols']:.1f}%)")
    if elemental.get("s_pct") and elemental["s_pct"] > 0.7:
        lignin_flags.append(f"Элементная сера S = {elemental['s_pct']:.2f}%")
    if irms.get("d34S") and irms["d34S"] >= 4.0:
        lignin_flags.append(f"тяжелая изотопная сера IRMS δ³⁴S = {irms['d34S']:.2f}‰")
    if ic.get("so4_cl_ratio") and ic["so4_cl_ratio"] >= 4.0:
        lignin_flags.append(f"избыток сульфата IC SO₄/Cl = {ic['so4_cl_ratio']:.1f}")

    if len(lignin_flags) >= 2:
        insights.append({
            "category": "Техногенный след / Шлам-лигнин",
            "status": "🚨 Обнаружен устойчивый маркерный кластер лигнина/сульфонатов",
            "detail": "Кросс-модальное совпадение: " + "; ".join(lignin_flags) + ". Высокая вероятность техногенного вклада сульфатного лигнина.",
        })

    # 4. Raman Carbonaceous Condensation
    if _is_finite_number(raman.get("id_ig_ratio")) and _is_finite_number(raman.get("la_crystallite_nm")):
        id_ig = raman["id_ig_ratio"]
        la = raman["la_crystallite_nm"]
        insights.append({
            "category": "Рамановская конденсированность",
            "status": f"💎 D/G отношение: {id_ig:.2f} (L_a = {la:.1f} нм)",
            "detail": f"Спектроскопия Raman фиксирует размер графеноподобных кластеров $L_a = {la:.1f}$ нм при $I_D/I_G = {id_ig:.2f}$, что указывает на {'высококонденсированный углеродный скелет (пирогенный чар / шлам)' if id_ig >= 0.9 else 'рыхлую полиароматическую сетку природного гумуса'}.",
        })

    # 4. Fresh Biogenic / Microbial Organic Matter
    bio_flags = []
    if fticr.get("hetero_classes", {}).get("CHON", 0) > 0:
        bio_flags.append("пептидоподобные CHON")
    if eem.get("c3_protein") and eem["c3_protein"] > 10.0:
        bio_flags.append(f"белковоподобный флуорофор C3 ({eem['c3_protein']:.1f}%)")
    if elemental.get("c_n_ratio") and elemental["c_n_ratio"] < 15.0:
        bio_flags.append(f"обогащение азотом (C/N = {elemental['c_n_ratio']:.1f})")

    if len(bio_flags) >= 2:
        insights.append({
            "category": "Биогенный автохтонный пул",
            "status": "🌱 Свежая микробная / протеиновая продукция",
            "detail": "Согласованные маркеры: " + "; ".join(bio_flags) + ". Свидетельствует об активном автохтонном продуцировании РОВ планктоном/бактериями.",
        })

    return insights


def build_cross_modal_synthesis_prompt(
    sample_name: str,
    multimodal_data: Dict[str, Any],
    lang: str = "ru",
) -> str:
    """
    Builds a rigorous, multi-omics scientific synthesis prompt integrating all
    available analytical techniques for ChemoAgent.
    """
    lines: List[str] = []
    active = multimodal_data.get("active_modalities", [])
    mod_str = ", ".join(m.upper() for m in active) if active else "FT-ICR MS"

    if lang == "ru":
        lines = [
            f"Проведи глубокий **кросс-модальный аналитический синтез** образца: **{sample_name}**.",
            f"В исследовании объединены данные {len(active)} аналитических методов: **{mod_str}**.",
            "",
            "### 🔬 Экспериментальные данные по модальностям:",
        ]
    else:
        lines = [
            f"Perform a deep **cross-modal analytical synthesis** for sample: **{sample_name}**.",
            f"The analysis fuses {len(active)} analytical modalities: **{mod_str}**.",
            "",
            "### 🔬 Experimental Multi-Modal Data:",
        ]

    # FT-ICR MS
    fticr = multimodal_data.get("fticr")
    if fticr:
        f_lines = []
        if fticr.get("n_peaks"):
            f_lines.append(f"Число формул/пиков: {fticr['n_peaks']:,}")
        if fticr.get("mn"):
            m_min, m_max = fticr.get("mass_range", (200, 800))
            f_lines.append(f"Mn: {fticr['mn']} Da ({m_min:.0f}–{m_max:.0f})")
        if fticr.get("hc_mean") is not None:
            f_lines.append(f"H/C: {fticr['hc_mean']}")
        if fticr.get("oc_mean") is not None:
            f_lines.append(f"O/C: {fticr['oc_mean']}")
        if fticr.get("ai_mod") is not None:
            f_lines.append(f"AI_mod: {fticr['ai_mod']} (доля >0.5: {fticr.get('high_ai_pct', 0)}%, конд.: {fticr.get('condensed_pct', 0)}%)")
        if fticr.get("dbe_mean") is not None:
            f_lines.append(f"DBE: {fticr['dbe_mean']}")
        if fticr.get("nosc_mean") is not None:
            f_lines.append(f"NOSC: {fticr['nosc_mean']}")
        lines.append(f"1. **FT-ICR MS:** {', '.join(f_lines)}")
        if fticr.get("hetero_classes"):
            top_classes = ", ".join(f"{k}:{v}" for k, v in list(fticr["hetero_classes"].items())[:5])
            lines.append(f"   - Классы: {top_classes}")

    # EEM
    eem = multimodal_data.get("eem")
    if eem:
        e_items = ", ".join(f"{k}: {v}" for k, v in eem.items())
        lines.append(f"2. **Оптическая спектроскопия (EEM-PARAFAC / UV):** {e_items}")

    # NMR
    nmr = multimodal_data.get("nmr")
    if nmr:
        n_items = ", ".join(f"{k}: {v}" for k, v in nmr.items())
        lines.append(f"3. **Спектроскопия ЯМР 13C:** {n_items}")

    # FTIR
    ftir = multimodal_data.get("ftir")
    if ftir:
        f_items = ", ".join(f"{k}: {v}" for k, v in ftir.items())
        lines.append(f"4. **ИК-Фурье (ATR-FTIR):** {f_items}")

    # SEC
    sec = multimodal_data.get("sec")
    if sec:
        s_items = ", ".join(f"{k}: {v}" for k, v in sec.items())
        lines.append(f"5. **Эксклюзионная хроматография (SEC/GPC):** {s_items}")

    # Thermal
    thermal = multimodal_data.get("thermal")
    if thermal:
        t_items = ", ".join(f"{k}: {v}" for k, v in thermal.items())
        lines.append(f"6. **Термодеструкция (Py-GC/MS & TGA):** {t_items}")

    # Elemental
    elemental = multimodal_data.get("elemental")
    if elemental:
        elem_items = ", ".join(f"{k}: {v}" for k, v in elemental.items())
        lines.append(f"7. **Элементный микроанализ (CHNS-O):** {elem_items}")

    # Cross-Modal Correlations
    insights = compute_cross_modal_insights(multimodal_data)
    if insights:
        lines.append("")
        lines.append("### ⚖️ Предварительные кросс-модальные сопоставления ChemoSuite:")
        for ins in insights:
            lines.append(f"- **{ins['status']}** ({ins['category']}): {ins['detail']}")

    lines.append("")
    if lang == "ru":
        lines.extend([
            "### 🎯 Твоя задача:",
            "Сформируй исчерпывающее, строгое экспертное заключение в 4 структурированных разделах (суммарно до 350 слов):",
            "1. **🧬 Комплексный хемотип и генезис**: синтез происхождения (природное РОВ / торф / шлам-лигнин / микробный автохтон) с опорой на ВСЕ представленные методы.",
            "2. **⚖️ Кросс-модальная валидация**: прямое сопоставление данных ЯМР, FTIR, EEM, SEC и FT-ICR (согласуются ли оптические, спектроскопические и масс-спектрометрические маркеры).",
            "3. **🔬 Инструментальные расхождения**: выявление эффектов селективности ионизации ESI(-), супрамолекулярной ассоциации в растворе против депротонированных мономеров в вакууме.",
            "4. **🎯 Итоговый вердикт и ключевые маркеры**: однозначный вывод об экологическом состоянии и генезисе пробы.",
        ])
    else:
        lines.extend([
            "### 🎯 Task:",
            "Provide a rigorous expert interpretation in 4 structured sections (up to 350 words total):",
            "1. **🧬 Multi-Modal Chemotype & Origin**: synthesis across all available techniques.",
            "2. **⚖️ Cross-Modal Validation**: direct comparison of NMR, FTIR, EEM, SEC, and FT-ICR.",
            "3. **🔬 Analytical Discrepancies**: ionization selectivity of ESI(-) vs bulk spectroscopy, supramolecular aggregation.",
            "4. **🎯 Definitive Takeaway & Key Biomarkers**.",
        ])

    return "\n".join(lines)

