"""
views/agent_view.py — Interactive ChemoAgent AI Copilot for ChemoSuite Pro.

Powered by Groq Ultra-Fast LPU API (llama-3.3-70b-versatile).
Provides dialog chat, quick sample biogeochemical interpretation,
and secure API key management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

import modules.chemo_agent as chemo_agent


def save_groq_api_key_to_env(key: str) -> bool:
    """Safely saves or updates GROQ_API_KEY in the local .env file."""
    key = key.strip()
    if not key:
        return False
    env_file = Path(".env")
    lines = []
    found = False
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        lines.append(f"GROQ_API_KEY={key}\n")
                        found = True
                    else:
                        lines.append(line)
        except Exception:
            pass

    if not found:
        lines.append(f"GROQ_API_KEY={key}\n")

    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.environ["GROQ_API_KEY"] = key
        return True
    except Exception as exc:
        st.error(f"Не удалось записать в .env: {exc}")
        return False


def _stream_with_loader(stream_gen, loader_placeholder):
    """
    Wraps the stream generator: clears the loader_placeholder as soon as the
    first real token chunk is received, then yields all tokens.
    """
    cleared = False
    for chunk in stream_gen:
        if not cleared:
            loader_placeholder.empty()
            cleared = True
        yield chunk
    if not cleared:
        loader_placeholder.empty()


def _get_active_fticr_data() -> Optional[Dict[str, Any]]:
    """Extracts currently active FT-ICR MS sample from session state."""
    db = st.session_state.get("spectra_db", {})
    active_name = st.session_state.get("active_spectrum_name")
    if not active_name and db:
        active_name = list(db.keys())[0]

    if active_name and active_name in db:
        spec = db[active_name]
        assigned_df = spec.get("assigned_df")
        if assigned_df is None and "raw_df" in spec:
            assigned_df = spec.get("raw_df")
        return {
            "name": active_name,
            "assigned_df": assigned_df,
            "spec": spec,
        }
    return None


def render_chemo_agent_sidebar(T: Dict[str, Any], lang: str = "ru"):
    """Renders ChemoAgent status and launch trigger in the sidebar."""
    try:
        api_key = chemo_agent.get_groq_api_key(st.session_state.get("groq_api_key"))
        has_key = bool(api_key)

        st.markdown("---")
        
        # Header with status indicator
        col_t, col_s = st.columns([2.2, 1.0])
        with col_t:
            st.markdown("**🤖 ChemoAgent (AI)**")
        with col_s:
            if has_key:
                st.markdown('<span style="color:#10b981; font-size:12px; font-weight:700;">🟢 Groq LPU</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#94a3b8; font-size:12px; font-weight:700;">⚪ No Key</span>', unsafe_allow_html=True)

        btn_label = "💬 Открыть AI-Копилот" if lang == "ru" else "💬 Open AI Copilot"
        if hasattr(st, "dialog"):
            if st.button(btn_label, key="btn_open_agent_dialog", use_container_width=True):
                _show_agent_dialog(T, lang)
        else:
            with st.expander(btn_label, expanded=False):
                _render_agent_content(T, lang, is_dialog=False)
    except Exception as exc:
        st.caption(f"🤖 ChemoAgent: {exc}")


if hasattr(st, "dialog"):
    @st.dialog("🤖 ChemoAgent — AI-копилот ChemoSuite", width="large")
    def _show_agent_dialog(T: Dict[str, Any], lang: str = "ru"):
        _render_agent_content(T, lang, is_dialog=True)
else:
    def _show_agent_dialog(T: Dict[str, Any], lang: str = "ru"):
        pass


def _render_agent_content(T: Dict[str, Any], lang: str = "ru", is_dialog: bool = True):
    """Core interactive chat and reasoning panel."""
    # 1. State initialization
    if "chemo_agent_messages" not in st.session_state:
        greeting = (
            "Здравствуйте! Я **ChemoAgent** — ваш специализированный научный ассистент по масс-спектрометрии "
            "ультравысокого разрешения (FT-ICR MS), флуоресценции (EEM-PARAFAC), ЯМР, ИК-спектроскопии и хемометрике.\n\n"
            "Я могу помочь:\n"
            "- Проанализировать активный образец и выявить маркеры шлам-лигнина или ненарушенного РОВ.\n"
            "- Интерпретировать молекулярные индексы ($H/C$, $O/C$, $AI_{\text{mod}}$, $NOSC$, VK-20).\n"
            "- Оценить векторные потоки реакций TMDS (+CH2, +O, -CO2, +SO3).\n"
            "- Сформулировать выводы для научной публикации или паспорта образца."
            if lang == "ru"
            else "Hello! I am **ChemoAgent** — your domain-specialized AI scientific assistant for ultrahigh-resolution "
            "mass spectrometry (FT-ICR MS), fluorescence (EEM-PARAFAC), NMR, FTIR, and chemometrics.\n\n"
            "I can interpret molecular descriptors, identify contamination markers (e.g. pulp mill lignin vs. natural NOM), "
            "and assist in geochemical classification."
        )
        st.session_state["chemo_agent_messages"] = [
            {"role": "assistant", "content": greeting}
        ]

    # 2. Key Check and Settings Expander
    current_key = chemo_agent.get_groq_api_key(st.session_state.get("groq_api_key"))
    
    with st.expander("⚙️ Конфигурация Groq API & Модели" if lang == "ru" else "⚙️ Groq API & Model Settings", expanded=not bool(current_key)):
        col_k1, col_k2 = st.columns([2.5, 1.0])
        with col_k1:
            key_input = st.text_input(
                "Groq API Key (gsk_...):",
                value=st.session_state.get("groq_api_key", current_key or ""),
                type="password",
                help="Бесплатный ключ LPU доступен мгновенно на console.groq.com/keys",
                key="agent_key_input_field",
            )
        with col_k2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Сохранить в .env" if lang == "ru" else "💾 Save to .env", key="btn_save_key_env", use_container_width=True):
                if key_input.strip():
                    st.session_state["groq_api_key"] = key_input.strip()
                    if save_groq_api_key_to_env(key_input.strip()):
                        st.success("Ключ сохранен в .env и окружение!" if lang == "ru" else "Key saved to .env & environment!")
                        st.rerun()
                else:
                    st.warning("Введите ключ перед сохранением." if lang == "ru" else "Enter key before saving.")

        col_m1, col_m2 = st.columns([1.5, 1.0])
        with col_m1:
            raw_options = chemo_agent.get_available_groq_models(current_key)
            model_options = [
                m for m in raw_options
                if not any(d in m.lower() for d in ["llama-3.3", "llama-3.1", "mixtral", "gemma"])
            ]
            if not model_options:
                model_options = chemo_agent.FALLBACK_MODELS

            # Force purge stale deprecated selections from session_state
            if st.session_state.get("agent_model_select") not in model_options:
                st.session_state["agent_model_select"] = model_options[0]

            def_idx = model_options.index(st.session_state["agent_model_select"])

            selected_model = st.selectbox(
                "Модель нейросети:" if lang == "ru" else "AI Model:",
                model_options,
                index=def_idx,
                key="agent_model_select",
            )
        with col_m2:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.2, 0.05, key="agent_temp_slider", help="0.1–0.3 для строгой науки")

    # 3. Quick Action: Load Active Sample into Agent Context
    active_data = _get_active_fticr_data()
    col_act1, col_act2, col_act3 = st.columns([2.0, 1.5, 1.0])
    with col_act1:
        if active_data:
            st.caption(f"📍 **Активный образец:** `{active_data['name']}`" if lang == "ru" else f"📍 **Active sample:** `{active_data['name']}`")
        else:
            st.caption("ℹ️ Спектры FT-ICR еще не загружены" if lang == "ru" else "ℹ️ No FT-ICR spectra loaded yet")
    with col_act2:
        btn_analyze_sample = st.button(
            "🔬 Экспертный анализ пробы" if lang == "ru" else "🔬 Analyze Sample",
            key="btn_agent_analyze_sample",
            disabled=(active_data is None or not bool(current_key)),
            use_container_width=True,
            type="primary",
        )
    with col_act3:
        if st.button("🧹 Очистить чат" if lang == "ru" else "🧹 Clear Chat", key="btn_agent_clear_chat", use_container_width=True):
            st.session_state["chemo_agent_messages"] = [
                {"role": "assistant", "content": "История очищена. Задайте новый вопрос или выберите образец для анализа." if lang == "ru" else "Chat history cleared. Ask a question or select a sample."}
            ]
            st.rerun()

    # 4. Display Chat Messages
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state["chemo_agent_messages"]:
            avatar = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                display_text = msg.get("display_content", msg["content"])
                st.markdown(chemo_agent.normalize_latex(display_text), unsafe_allow_html=True)

    # 5. Handle Sample Analysis Button Trigger
    if btn_analyze_sample and active_data:
        prompt_text = chemo_agent.build_sample_analysis_prompt(
            sample_name=active_data["name"],
            sample_data=active_data,
            lang=lang,
        )
        short_display = f"🔬 **Экспертный анализ образца:** `{active_data['name']}`" if lang == "ru" else f"🔬 **Expert sample analysis:** `{active_data['name']}`"
        st.session_state["chemo_agent_messages"].append({
            "role": "user",
            "content": prompt_text,
            "display_content": short_display,
        })
        with chat_box:
            with st.chat_message("user", avatar="👤"):
                st.markdown(short_display)

            with st.chat_message("assistant", avatar="🤖"):
                loader_ph = st.empty()
                loader_msg = "ChemoAgent формулирует ответ..." if lang == "ru" else "ChemoAgent generating response..."
                loader_ph.markdown(
                    f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
                    unsafe_allow_html=True,
                )
                req_messages = [{"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chemo_agent_messages"][-6:]
                ]
                stream_gen = chemo_agent.stream_groq_chat(
                    messages=req_messages,
                    api_key=current_key,
                    model=selected_model,
                    temperature=temperature,
                )
                if hasattr(st, "write_stream"):
                    full_response = st.write_stream(_stream_with_loader(stream_gen, loader_ph))
                else:
                    msg_ph = st.empty()
                    full_response = ""
                    for chunk in _stream_with_loader(stream_gen, loader_ph):
                        full_response += chunk
                        msg_ph.markdown(full_response + "▌")
                    msg_ph.markdown(full_response)
        st.session_state["chemo_agent_messages"].append({"role": "assistant", "content": chemo_agent.normalize_latex(full_response)})

    # 6. Interactive Chat Input
    user_prompt = st.chat_input(
        "Спросите ChemoAgent о формулах, индексах или реакциях..." if lang == "ru" else "Ask ChemoAgent about formulas, indices, or reactions...",
        key="agent_user_chat_input",
    )
    if user_prompt:
        if not current_key:
            st.error("Пожалуйста, укажите GROQ_API_KEY в настройках выше!" if lang == "ru" else "Please specify GROQ_API_KEY in the settings above!")
        else:
            st.session_state["chemo_agent_messages"].append({"role": "user", "content": user_prompt})
            with chat_box:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    loader_ph = st.empty()
                    loader_msg = "ChemoAgent формулирует ответ..." if lang == "ru" else "ChemoAgent generating response..."
                    loader_ph.markdown(
                        f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
                        unsafe_allow_html=True,
                    )
                    req_messages = [{"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT}] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chemo_agent_messages"][-8:]
                    ]
                    stream_gen = chemo_agent.stream_groq_chat(
                        messages=req_messages,
                        api_key=current_key,
                        model=selected_model,
                        temperature=temperature,
                    )
                    if hasattr(st, "write_stream"):
                        full_response = st.write_stream(_stream_with_loader(stream_gen, loader_ph))
                    else:
                        msg_ph = st.empty()
                        full_response = ""
                        for chunk in _stream_with_loader(stream_gen, loader_ph):
                            full_response += chunk
                            msg_ph.markdown(full_response + "▌")
                        msg_ph.markdown(full_response)
            st.session_state["chemo_agent_messages"].append({"role": "assistant", "content": chemo_agent.normalize_latex(full_response)})


def render_sample_ai_card(sample_name: str, sample_data: Dict[str, Any], lang: str = "ru"):
    """
    Renders an on-demand AI Analysis card inside FT-ICR MS or ML Studio.
    Caches results per sample in st.session_state so repeated views do not re-query the API.
    """
    api_key = chemo_agent.get_groq_api_key(st.session_state.get("groq_api_key"))
    cache_key = f"ai_summary_{sample_name}"

    st.markdown("### 🤖 Экспертное AI-заключение (ChemoAgent)" if lang == "ru" else "### 🤖 AI Expert Interpretation (ChemoAgent)")
    
    col_info, col_btn = st.columns([3.0, 1.2])
    with col_info:
        st.caption(
            "Глубокая биогеохимическая интерпретация на базе нейросети Groq LPU (GPT-OSS 120B / Qwen 27B): генезис пробы, степень ароматической конденсации, техногенные маркеры."
            if lang == "ru"
            else "In-depth biogeochemical reasoning powered by Groq LPU (GPT-OSS 120B / Qwen 27B): origin, aromatic condensation, contamination markers."
        )
    with col_btn:
        run_btn = st.button(
            "⚡ Сгенерировать анализ" if lang == "ru" else "⚡ Generate Analysis",
            key=f"btn_run_ai_{sample_name}",
            use_container_width=True,
            type="primary" if not st.session_state.get(cache_key) else "secondary",
        )

    if run_btn:
        if not api_key:
            st.warning("Укажите GROQ_API_KEY в боковой панели ChemoAgent для генерации заключения." if lang == "ru" else "Specify GROQ_API_KEY in the sidebar ChemoAgent to generate interpretation.")
            return

        prompt = chemo_agent.build_sample_analysis_prompt(sample_name, sample_data, lang=lang)
        messages = [
            {"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        loader_ph = st.empty()
        loader_msg = "ChemoAgent анализирует спектральные дескрипторы..." if lang == "ru" else "Analyzing spectral descriptors..."
        loader_ph.markdown(
            f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
            unsafe_allow_html=True,
        )
        res = chemo_agent.call_groq_chat(
            messages,
            api_key=api_key,
            model=st.session_state.get("agent_model_select", chemo_agent.DEFAULT_MODEL),
        )
        loader_ph.empty()
        if res.get("success"):
            st.session_state[cache_key] = res.get("content", "")
        else:
            st.markdown(res.get("error", "Произошла ошибка"), unsafe_allow_html=True)

    cached_content = st.session_state.get(cache_key)
    if cached_content:
        with st.container(border=True):
            st.markdown(chemo_agent.normalize_latex(cached_content))
