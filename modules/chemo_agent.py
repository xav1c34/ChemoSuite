"""
modules/chemo_agent.py — ChemoSuite AI Scientific Copilot & Biogeochemical Reasoner.

Powered by Groq Ultra-Fast LPU API (llama-3.3-70b-versatile / mixtral-8x7b-32768).
Zero external dependencies (uses standard requests + urllib).
Loads API key securely from .env, st.secrets, or UI session state.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Generator
import requests
import pandas as pd

logger = logging.getLogger("chemo_agent")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
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
                pref_order = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
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
   - Не ставь пробелы сразу после открывающего $ и перед закрывающим $."""


def normalize_latex(text: str) -> str:
    """
    Normalizes LaTeX syntax so Streamlit KaTeX renders it cleanly:
    1. Converts block math \\[ ... \\] to \\n\\n$$\\n...\\n$$\\n\\n
    2. Converts inline math \\( ... \\) to $...$
    3. Normalizes single dollar math with inner whitespace: '$ H/C $' -> '$H/C$'
    4. Normalizes subscript in AI_mod: '$AI_{mod}$' -> '$AI_{\\text{mod}}$'
    """
    if not text:
        return ""
    # 1. Block math: \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n\n$$\n\1\n$$\n\n', text, flags=re.DOTALL)
    # 2. Inline math: \( ... \) -> $...$
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)
    # 3. Strip inner spaces inside $ ... $ (e.g. '$ H/C $' -> '$H/C$')
    text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', lambda m: f"${m.group(1).strip()}$", text)
    # 4. Standardize AI_mod subscript to \text{mod} for proper text formatting in KaTeX
    text = re.sub(r'\$AI_\{?mod\}?\$', r'$AI_{\\text{mod}}$', text)
    return text


DEFAULT_MAX_TOKENS = 1200


def _get_model_max_tokens(model: str, requested: Optional[int] = None) -> int:
    """
    Returns optimal token budget:
    - Reasoning models (e.g. gpt-oss-120b, gpt-oss-20b) generate hidden/chain-of-thought
      reasoning tokens that consume the output budget. They need 1500-2048 tokens.
    - Direct generation models (e.g. qwen3.8-27b) output 100% visible tokens and thrive on 1000-1200 tokens,
      comfortably avoiding token starvation and rate limits.
    """
    m = model.lower() if model else ""
    if "gpt-oss" in m:
        return min(int(requested or 2048), 2048)
    return min(int(requested or 1200), 1200)


def format_groq_error(status_code: int, raw_text: str) -> str:
    """
    Formats Groq API errors into a concise, polite, human-readable notification.
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

    if status_code == 429:
        if any(w in msg_lower for w in ["otpm", "limit 1000", "expected output", "output tokens per minute"]):
            header = "⚠️ **Превышен минутный лимит токенов Groq (429 Rate Limit)**"
            hint = "Бесплатный тариф Groq On-Demand временно исчерпал лимит токенов генерации (OTPM). Лимит восстанавливается за 15–30 секунд. Пожалуйста, подождите полминуты и повторите запрос."
        else:
            header = "⚠️ **Превышен лимит запросов к Groq (429 Too Many Requests)**"
            hint = "Слишком много запросов в минуту. Пожалуйста, подождите 15–20 секунд перед следующим запросом."
    elif status_code == 404:
        header = "❌ **Модель недоступна в Groq (404 Not Found)**"
        hint = "Выбранная модель устарела или недоступна на вашем ключе. Выберите доступную модель в настройках."
    elif status_code == 401:
        header = "❌ **Неверный GROQ_API_KEY (401 Unauthorized)**"
        hint = "Проверьте правильность API ключа в настройках боковой панели или в файле `.env`."
    elif status_code == 400:
        header = "❌ **Некорректный запрос (400 Bad Request)**"
        hint = "Запрос превышает контекст или содержит неподдерживаемые параметры."
    else:
        header = f"❌ **Ошибка Groq API (HTTP {status_code})**"
        hint = "Не удалось завершить генерацию ответа нейросетью. Попробуйте повторить запрос."

    # Clean technical details snippet for collapsible block
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


def _sanitize_model(model: Optional[str]) -> str:
    """Ensures model is valid, remapping deprecated or unpermitted models to DEFAULT_MODEL."""
    if not model:
        return DEFAULT_MODEL
    s = str(model).strip()
    if any(bad in s.lower() for bad in ["llama", "mixtral", "gemma", "versatile", "instant"]):
        return DEFAULT_MODEL
    return s


def call_groq_chat(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Sends chat completion request to Groq API via standard HTTP POST.
    Automatically falls back to DEFAULT_MODEL if model returns 404.
    Self-heals on OTPM 429 errors by dynamically scaling down max_tokens.
    """
    resolved_key = get_groq_api_key(api_key)
    if not resolved_key:
        return {
            "success": False,
            "content": "",
            "error": "GROQ_API_KEY не найден. Добавьте его в файл .env (GROQ_API_KEY=gsk_...) или укажите в поле ввода в сайдбаре.",
        }

    target_model = _sanitize_model(model)
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    clamped_tokens = _get_model_max_tokens(target_model, max_tokens)
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": clamped_tokens,
        "reasoning_effort": "low",
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 404 and target_model != DEFAULT_MODEL:
            logger.warning("Groq model %s returned 404, falling back to %s", target_model, DEFAULT_MODEL)
            payload["model"] = DEFAULT_MODEL
            payload["max_tokens"] = _get_model_max_tokens(DEFAULT_MODEL, max_tokens)
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)

        # Self-heal on OTPM 429 if Groq reports a lower token limit
        if response.status_code == 429 and "limit" in response.text.lower():
            m_limit = re.search(r"Limit\s+(\d+)", response.text, re.IGNORECASE)
            if m_limit and payload["max_tokens"] > 750:
                safe_limit = max(int(m_limit.group(1)) - 50, 600)
                logger.warning("Groq OTPM limit exceeded. Retrying with reduced max_tokens=%d", safe_limit)
                payload["max_tokens"] = safe_limit
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            choice = data.get("choices", [{}])[0]
            reply_text = choice.get("message", {}).get("content", "")
            return {
                "success": True,
                "content": reply_text,
                "model": data.get("model", payload["model"]),
                "usage": data.get("usage", {}),
                "error": None,
            }
        else:
            err_msg = format_groq_error(response.status_code, response.text)
            logger.error("Groq API error: HTTP %d", response.status_code)
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 60.0,
) -> Generator[str, None, None]:
    """
    Streams response tokens from Groq API in real-time with resilient connection handling.
    Automatically falls back to DEFAULT_MODEL if model returns 404.
    Recovers from OTPM rate limits and handles unexpected socket drops gracefully.
    """
    resolved_key = get_groq_api_key(api_key)
    if not resolved_key:
        yield "❌ **Ошибка:** GROQ_API_KEY не найден. Добавьте его в файл `.env` (`GROQ_API_KEY=gsk_...`) или введите в сайдбаре."
        return

    target_model = _sanitize_model(model)
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }

    clamped_tokens = _get_model_max_tokens(target_model, max_tokens)
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": clamped_tokens,
        "reasoning_effort": "low",
        "stream": True,
    }

    response = None
    yielded_any = False
    finish_reason = None
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=(10.0, 90.0), stream=True)
        if response.status_code == 404 and target_model != DEFAULT_MODEL:
            logger.warning("Groq stream model %s returned 404, falling back to %s", target_model, DEFAULT_MODEL)
            payload["model"] = DEFAULT_MODEL
            payload["max_tokens"] = _get_model_max_tokens(DEFAULT_MODEL, max_tokens)
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=(10.0, 90.0), stream=True)

        # Self-heal on OTPM 429 if Groq reports a lower limit
        if response.status_code == 429 and "limit" in response.text.lower():
            m_limit = re.search(r"Limit\s+(\d+)", response.text, re.IGNORECASE)
            if m_limit and payload["max_tokens"] > 750:
                safe_limit = max(int(m_limit.group(1)) - 50, 600)
                logger.warning("Groq stream OTPM limit exceeded. Retrying with max_tokens=%d", safe_limit)
                payload["max_tokens"] = safe_limit
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=(10.0, 90.0), stream=True)

        if response.status_code != 200:
            yield format_groq_error(response.status_code, response.text)
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line_str = line if isinstance(line, str) else line.decode("utf-8", errors="replace")
            if line_str.startswith("data: "):
                data_part = line_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_part)
                except Exception:
                    continue

                if "error" in chunk_json:
                    err_text = chunk_json.get("error", {}).get("message", str(chunk_json["error"]))
                    yield f"\n\n{format_groq_error(429 if 'rate' in err_text.lower() else 500, json.dumps(chunk_json))}"
                    return

                c0 = chunk_json.get("choices", [{}])[0]
                if c0.get("finish_reason"):
                    finish_reason = c0.get("finish_reason")
                delta = c0.get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
                    yielded_any = True

        if finish_reason == "length":
            yield "\n\n*(Ответ был ограничен лимитом токенов модели. Вы можете написать: «Продолжи»)*"
    except Exception as exc:
        err_str = str(exc)
        logger.warning("Groq stream ended with exception: %s", err_str)
        # If tokens were already yielded and socket closed (e.g. unexpected EOF / IncompleteRead)
        if yielded_any and any(k in err_str.lower() for k in ["eof", "incompleteread", "connection broken", "chunked"]):
            if finish_reason != "length":
                yield "\n\n*(Генерация завершена сервером)*"
        else:
            yield f"\n\n❌ **Ошибка сети:** {err_str}"
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


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
        for cand in ["ai", "ai_mod", "ai_koch", "aromaticity_index"]:
            if cand in cols_lower:
                ai_col = cols_lower[cand]
                break
        if ai_col:
            ai_s = pd.to_numeric(df[ai_col], errors="coerce").dropna()
            if not ai_s.empty:
                high_ai = (ai_s > 0.5).mean() * 100.0
                condensed = (ai_s >= 0.67).mean() * 100.0
                metrics.append(f"AI_mod: {ai_s.mean():.3f} (доля >0.5: {high_ai:.0f}%, конденс.: {condensed:.0f}%)")

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
        eem_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["eem"].items())[:4])
        lines.append(f"💡 **EEM-PARAFAC:** {eem_items}")

    # NMR 13C
    if "nmr" in sample_data and sample_data["nmr"]:
        nmr_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["nmr"].items())[:4])
        lines.append(f"🧪 **ЯМР 13C:** {nmr_items}")

    # FTIR
    if "ftir" in sample_data and sample_data["ftir"]:
        ftir_items = ", ".join(f"{k}: {v}" for k, v in list(sample_data["ftir"].items())[:4])
        lines.append(f"📈 **ATR-FTIR:** {ftir_items}")

    # Elemental CHNS-O
    if "elemental" in sample_data and sample_data["elemental"]:
        elem_items = ", ".join(f"{k}: {v}" for k, v in sample_data["elemental"].items())
        lines.append(f"⚗️ **CHNS-O:** {elem_items}")

    lines.append("")
    if lang == "ru":
        lines.append("**Задача:** Дай краткое экспертное заключение строго в 2–3 абзаца (до 250 слов): вероятный генезис (природное РОВ / торф / шлам-лигнин / сток), степень окисления/ароматизации и ключевые маркеры трансформации. Обязательно закончи вывод кратким резюме.")
    else:
        lines.append("**Task:** Provide a concise expert interpretation in strictly 2–3 short paragraphs (under 250 words): likely geochemical origin, oxidation/aromaticity state, and transformation markers. Conclude with a clear brief takeaway.")

    return "\n".join(lines)
