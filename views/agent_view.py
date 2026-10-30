"""
views/agent_view.py — Interactive ChemoAgent AI Copilot for ChemoSuite Pro.

Powered by Groq Ultra-Fast LPU API (llama-3.3-70b-versatile).
Provides dialog chat, quick sample biogeochemical interpretation,
and secure API key management.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

import modules.chemo_agent as chemo_agent


def save_polza_api_key_to_env(key: str) -> bool:
    """Safely saves or updates POLZA_AI_KEY in the local .env file."""
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
                    if line.strip().startswith("POLZA_AI_KEY="):
                        lines.append(f"POLZA_AI_KEY={key}\n")
                        found = True
                    else:
                        lines.append(line)
        except Exception:
            pass

    if not found:
        lines.append(f"POLZA_AI_KEY={key}\n")

    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.environ["POLZA_AI_KEY"] = key
        return True
    except Exception as exc:
        st.error(f"Не удалось записать в .env: {exc}")
        return False


# Backward compatibility aliases
save_groq_api_key_to_env = save_polza_api_key_to_env
save_gemini_api_key_to_env = save_polza_api_key_to_env


def _get_active_ai_config():
    """
    Returns (provider_code, active_key, selected_model, temperature, polza_key, polza_key)
    based on session state and environment.
    """
    polza_key = chemo_agent.get_polza_api_key(st.session_state.get("polza_api_key"))
    model = chemo_agent.DEFAULT_MODEL
    temp = float(st.session_state.get("agent_temp_slider", 0.2))
    return "polza", polza_key, model, temp, polza_key, polza_key


def _check_debounce_allowed() -> bool:
    """Checks if enough time (2.0s) has passed since the last AI query to prevent double-clicks."""
    now = time.time()
    last_ts = st.session_state.get("_last_agent_query_ts", 0.0)
    if now - last_ts < 2.0:
        return False
    st.session_state["_last_agent_query_ts"] = now
    return True


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


def _get_active_multimodal_data(selected_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extracts unified multi-modal analytical data across all active studios in session state."""
    db = st.session_state.get("spectra_db", {})
    if not db:
        return None

    name = selected_name
    if not name or name not in db:
        name = st.session_state.get("active_spectrum_name")
        if not name or name not in db:
            name = list(db.keys())[0]

    if not name or name not in db:
        return None

    harvested = chemo_agent.harvest_multimodal_descriptors(
        sample_name=name,
        spectra_db=db,
        eem_samples=st.session_state.get("eem_samples"),
        nmr_samples=st.session_state.get("nmr_samples"),
        ftir_samples=st.session_state.get("ftir_samples"),
        sec_samples=st.session_state.get("sec_samples"),
        thermal_samples=st.session_state.get("thermal_samples"),
        elemental_samples=st.session_state.get("elemental_samples"),
        irms_samples=st.session_state.get("irms_samples"),
        ic_samples=st.session_state.get("ic_samples"),
        raman_samples=st.session_state.get("raman_samples"),
        eem_ml=st.session_state.get("eem_ml_descriptors"),
        nmr_ml=st.session_state.get("nmr_ml_descriptors"),
        ftir_ml=st.session_state.get("ftir_ml_descriptors"),
        sec_ml=st.session_state.get("sec_ml_descriptors"),
        thermal_ml=st.session_state.get("thermal_ml_descriptors"),
        irms_ml=st.session_state.get("irms_ml_descriptors"),
        ic_ml=st.session_state.get("ic_ml_descriptors"),
        raman_ml=st.session_state.get("raman_ml_descriptors"),
        eem_db=st.session_state.get("eem_db"),
        nmr_db=st.session_state.get("nmr_db"),
        ftir_db=st.session_state.get("ftir_db"),
        elemental_db=st.session_state.get("elemental_db"),
    )

    if harvested.get("fticr"):
        harvested["assigned_df"] = harvested["fticr"].get("assigned_df")
        harvested["spec"] = db.get(name, {})
    return harvested


def _get_active_fticr_data(selected_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper for FT-ICR sample retrieval."""
    return _get_active_multimodal_data(selected_name)


def render_chemo_agent_sidebar(T: Dict[str, Any], lang: str = "ru"):
    """Renders ChemoAgent status and launch trigger in the sidebar."""
    try:
        provider_code, active_key, model, temp, *extra = _get_active_ai_config()
        has_key = bool(active_key)
        usage = chemo_agent.get_daily_usage(100)
        used_count = usage["count"]
        limit = usage["limit"]
        is_exhausted = used_count >= limit

        st.markdown("---")
        
        # Header with status indicator
        col_t, col_s = st.columns([1.8, 1.4])
        with col_t:
            st.markdown("**🤖 Chemo Agent**")
        with col_s:
            if not has_key:
                st.markdown('<span style="color:#94a3b8; font-size:12px; font-weight:700;">⚪ No Key</span>', unsafe_allow_html=True)
            elif is_exhausted:
                st.markdown(f'<span style="color:#f43f5e; font-size:12px; font-weight:700;" title="Лимит исчерпан">🔴 {used_count}/{limit}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color:#10b981; font-size:12px; font-weight:700;" title="Встроенный AI">🟢 {used_count}/{limit}</span>', unsafe_allow_html=True)

        st.caption("Встроенный ИИ" if lang == "ru" else "Built-in AI")

        btn_label = "💬 Открыть Chemo Agent" if lang == "ru" else "💬 Open Chemo Agent"
        if hasattr(st, "dialog"):
            if st.button(btn_label, key="btn_open_agent_dialog", use_container_width=True):
                _show_agent_dialog(T, lang)
        else:
            with st.expander(btn_label, expanded=False):
                _render_agent_content(T, lang, is_dialog=False)
    except Exception as exc:
        st.caption(f"🤖 Chemo Agent: {exc}")


if hasattr(st, "dialog"):
    @st.dialog("🤖 Chemo Agent", width="large")
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
            "Здравствуйте! Я **Chemo Agent** — ваш специализированный научный ассистент по масс-спектрометрии "
            "ультравысокого разрешения (FT-ICR MS), флуоресценции (EEM-PARAFAC), ЯМР, ИК-спектроскопии и хемометрике.\n\n"
            "Работаю через встроенный серверный API. Сервис готов к работе для всех исследователей!\n\n"
            "Я могу помочь:\n"
            "- Проанализировать активный образец и выявить маркеры шлам-лигнина или ненарушенного РОВ.\n"
            "- Интерпретировать молекулярные индексы ($H/C$, $O/C$, $AI_{\\text{mod}}$, $NOSC$, VK-20).\n"
            "- Оценить векторные потоки реакций TMDS (+CH2, +O, -CO2, +SO3).\n"
            "- Сформулировать выводы для научной публикации или паспорта образца."
            if lang == "ru"
            else "Hello! I am **Chemo Agent** — your domain-specialized AI scientific assistant for ultrahigh-resolution "
            "mass spectrometry (FT-ICR MS), fluorescence (EEM-PARAFAC), NMR, FTIR, and chemometrics.\n\n"
            "Powered by built-in server API. Ready out-of-the-box for all researchers!\n\n"
            "I can interpret molecular descriptors, identify contamination markers (e.g. pulp mill lignin vs. natural NOM), "
            "and assist in geochemical classification."
        )
        st.session_state["chemo_agent_messages"] = [
            {"role": "assistant", "content": greeting}
        ]

    # 2. Daily Quota Counter and Status
    provider_code, active_key, selected_model, temperature, *extra = _get_active_ai_config()
    usage = chemo_agent.get_daily_usage(100)
    used_count = usage["count"]
    limit = usage["limit"]
    is_exhausted = used_count >= limit

    col_stat1, col_stat2 = st.columns([2.6, 1.4])
    with col_stat1:
        if active_key:
            st.markdown(
                '<div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 6px 12px; border-radius: 4px; font-size: 13px;">'
                '🟢 <b>Chemo Agent активен</b>'
                '</div>' if lang == "ru" else
                '<div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 6px 12px; border-radius: 4px; font-size: 13px;">'
                '🟢 <b>Chemo Agent active</b>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("⚠️ Сервис ИИ временно недоступен. Обратитесь к администратору платформы." if lang == "ru" else "⚠️ AI service is currently unavailable. Please contact the platform administrator.")
    with col_stat2:
        badge_color = "#f43f5e" if is_exhausted else ("#f59e0b" if used_count > 80 else "#10b981")
        st.markdown(
            f'<div style="text-align: right; font-size: 13px; padding-top: 6px;">'
            f'Дневной лимит: <b style="color: {badge_color};">{used_count} / {limit}</b>'
            f'</div>' if lang == "ru" else
            f'<div style="text-align: right; font-size: 13px; padding-top: 6px;">'
            f'Daily quota: <b style="color: {badge_color};">{used_count} / {limit}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if is_exhausted:
        st.error(
            "⚠️ Достигнут суточный лимит запросов. Лимит автоматически сбросится в 00:00 UTC."
            if lang == "ru" else
            "⚠️ Daily query quota reached. Quota will automatically reset at 00:00 UTC."
        )

    # 3. Spectrum Selection & Quick Action: Load Selected Sample into Agent Context
    spectra_db = st.session_state.get("spectra_db", {})
    available_samples = list(spectra_db.keys())

    if available_samples:
        current_active = (
            st.session_state.get("agent_spectrum_selector_mm")
            or st.session_state.get("agent_spectrum_selector_sm")
            or st.session_state.get("agent_spectrum_selector")
            or st.session_state.get("active_spectrum_name")
        )
        if current_active not in available_samples:
            current_active = available_samples[0]
        default_idx = available_samples.index(current_active)

        active_data = _get_active_multimodal_data(current_active)
        active_mods = active_data.get("active_modalities", ["fticr"]) if active_data else []
        has_multimodal = len(active_mods) >= 2

        btn_cross_modal = False
        btn_analyze_sample = False

        if has_multimodal:
            col_act1, col_act2, col_act3, col_act4 = st.columns([1.7, 1.4, 1.1, 0.7])
            with col_act1:
                selected_sample_name = st.selectbox(
                    "Спектр для анализа:" if lang == "ru" else "Spectrum to analyze:",
                    options=available_samples,
                    index=default_idx,
                    key="agent_spectrum_selector_mm",
                    label_visibility="collapsed",
                    help="Выберите любой из загруженных спектров для передачи данных Chemo Agent" if lang == "ru" else "Select any loaded spectrum to pass data to Chemo Agent",
                )
            with col_act2:
                btn_cross_modal = st.button(
                    "🧬 Кросс-модальный синтез" if lang == "ru" else "🧬 Multi-Modal Synthesis",
                    key="btn_agent_cross_modal",
                    disabled=(active_data is None or not bool(active_key) or is_exhausted),
                    use_container_width=True,
                    type="primary",
                    help=f"Комплексный синтез данных ({len(active_mods)} активных методов: {', '.join(m.upper() for m in active_mods)})" if lang == "ru" else f"Fused synthesis ({len(active_mods)} modalities: {', '.join(m.upper() for m in active_mods)})",
                )
            with col_act3:
                btn_analyze_sample = st.button(
                    "🔬 FT-ICR анализ" if lang == "ru" else "🔬 FT-ICR Analysis",
                    key="btn_agent_analyze_sample_mm",
                    disabled=(active_data is None or not bool(active_key) or is_exhausted),
                    use_container_width=True,
                    type="secondary",
                    help="Сформировать экспертное биогеохимическое заключение строго по FT-ICR MS" if lang == "ru" else "Generate single-spectrum FT-ICR MS interpretation",
                )
            with col_act4:
                if st.button("🧹 Очистить" if lang == "ru" else "🧹 Clear", key="btn_agent_clear_chat_mm", use_container_width=True, help="Очистить историю сообщений"):
                    st.session_state["chemo_agent_messages"] = [
                        {"role": "assistant", "content": "История очищена. Задайте новый вопрос или выберите образец для анализа." if lang == "ru" else "Chat history cleared. Ask a question or select a sample."}
                    ]
                    st.rerun()
        else:
            col_act1, col_act2, col_act3 = st.columns([2.2, 1.4, 0.9])
            with col_act1:
                selected_sample_name = st.selectbox(
                    "Спектр для анализа:" if lang == "ru" else "Spectrum to analyze:",
                    options=available_samples,
                    index=default_idx,
                    key="agent_spectrum_selector_sm",
                    label_visibility="collapsed",
                    help="Выберите любой из загруженных спектров для передачи данных Chemo Agent" if lang == "ru" else "Select any loaded spectrum to pass data to Chemo Agent",
                )
            with col_act2:
                btn_analyze_sample = st.button(
                    "🔬 Отправить в ИИ" if lang == "ru" else "🔬 Send to AI",
                    key="btn_agent_analyze_sample_sm",
                    disabled=(active_data is None or not bool(active_key) or is_exhausted),
                    use_container_width=True,
                    type="primary",
                    help="Сформировать экспертное биогеохимическое заключение для выбранного спектра" if lang == "ru" else "Generate biogeochemical expert interpretation for selected spectrum",
                )
            with col_act3:
                if st.button("🧹 Очистить" if lang == "ru" else "🧹 Clear", key="btn_agent_clear_chat_sm", use_container_width=True, help="Очистить историю сообщений"):
                    st.session_state["chemo_agent_messages"] = [
                        {"role": "assistant", "content": "История очищена. Задайте новый вопрос или выберите образец для анализа." if lang == "ru" else "Chat history cleared. Ask a question or select a sample."}
                    ]
                    st.rerun()

        if selected_sample_name != current_active:
            active_data = _get_active_multimodal_data(selected_sample_name)
            active_mods = active_data.get("active_modalities", ["fticr"]) if active_data else []

        # Multi-Modal Availability Matrix Ribbon
        if active_data:
            pills = []
            # 1. FT-ICR
            if "fticr" in active_mods:
                n_p = active_data.get("fticr", {}).get("n_peaks", 0)
                lbl = "формул" if active_data.get("fticr", {}).get("has_formulas") else "пиков"
                pills.append(f"<span class='chemo-mod-pill active'>🔬 FT-ICR ({n_p:,} {lbl})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>🔬 FT-ICR</span>")

            # 2. EEM
            if "eem" in active_mods:
                suva = active_data.get("eem", {}).get("suva254")
                v_str = f"SUVA {suva}" if suva else "OK"
                pills.append(f"<span class='chemo-mod-pill active'>💡 EEM ({v_str})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>💡 EEM</span>")

            # 3. NMR
            if "nmr" in active_mods:
                fa = active_data.get("nmr", {}).get("fa")
                v_str = f"fa {fa}" if fa else "OK"
                pills.append(f"<span class='chemo-mod-pill active'>🧪 ЯМР 13C ({v_str})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>🧪 ЯМР 13C</span>")

            # 4. FTIR
            if "ftir" in active_mods:
                lig = active_data.get("ftir", {}).get("lignin_idx")
                v_str = f"Лигнин {lig}" if lig else "OK"
                pills.append(f"<span class='chemo-mod-pill active'>📈 FTIR ({v_str})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>📈 FTIR</span>")

            # 5. SEC
            if "sec" in active_mods:
                mw = active_data.get("sec", {}).get("mw")
                v_str = f"Mw {mw:.0f} Da" if mw else "OK"
                pills.append(f"<span class='chemo-mod-pill active'>📏 SEC ({v_str})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>📏 SEC</span>")

            # 6. Thermal
            if "thermal" in active_mods:
                pills.append("<span class='chemo-mod-pill active'>🔥 Термо (Py-GC)</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>🔥 Термо</span>")

            # 7. Elemental
            if "elemental" in active_mods:
                c_n = active_data.get("elemental", {}).get("c_n")
                v_str = f"C/N {c_n}" if c_n else "OK"
                pills.append(f"<span class='chemo-mod-pill active'>⚗️ CHNS-O ({v_str})</span>")
            else:
                pills.append("<span class='chemo-mod-pill inactive'>⚗️ CHNS-O</span>")

            st.markdown(f"<div class='chemo-mod-ribbon'>{' '.join(pills)}</div>", unsafe_allow_html=True)
            if not has_multimodal:
                st.caption(
                    "ℹ️ *Совет: загрузите спектры EEM, ЯМР, FTIR или SEC в соответствующих студиях для полной кросс-модальной верификации.*"
                    if lang == "ru"
                    else "ℹ️ *Tip: upload EEM, NMR, FTIR or SEC spectra in their studios to unlock full cross-modal synthesis.*"
                )
    else:
        active_data = None
        btn_cross_modal = False
        btn_analyze_sample = False
        col_act1, col_act3 = st.columns([3.5, 1.0])
        with col_act1:
            st.caption("ℹ️ Спектры FT-ICR еще не загружены. Загрузите файлы в студии FT-ICR MS." if lang == "ru" else "ℹ️ No FT-ICR spectra loaded yet. Upload data in FT-ICR MS Studio.")
        with col_act3:
            if st.button("🧹 Очистить" if lang == "ru" else "🧹 Clear", key="btn_agent_clear_chat_none", use_container_width=True):
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
                st.markdown(chemo_agent.normalize_latex(display_text))

    # 5. Handle Analysis & Synthesis Triggers
    is_triggered = (btn_cross_modal or btn_analyze_sample) and bool(active_data)
    if is_triggered:
        if is_exhausted:
            st.error("⚠️ Достигнут суточный лимит в 100 запросов. Сброс в 00:00 UTC." if lang == "ru" else "⚠️ Daily limit of 100 queries reached. Resets at midnight.")
            st.stop()
        if not _check_debounce_allowed():
            st.warning("Пожалуйста, подождите 2 секунды перед следующим запросом." if lang == "ru" else "Please wait 2 seconds between queries.")
            st.stop()

        st.session_state["active_spectrum_name"] = active_data["name"]
        if btn_cross_modal:
            prompt_text = chemo_agent.build_cross_modal_synthesis_prompt(
                sample_name=active_data["name"],
                multimodal_data=active_data,
                lang=lang,
            )
            mod_names = [m.upper() for m in active_mods]
            short_display = (
                f"🧬 **Кросс-модальный синтез:** `{active_data['name']}` ({len(active_mods)} метода: {', '.join(mod_names)})"
                if lang == "ru"
                else f"🧬 **Multi-Modal Synthesis:** `{active_data['name']}` ({len(active_mods)} modalities: {', '.join(mod_names)})"
            )
        else:
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
                loader_msg = "Chemo Agent выполняет синтез данных..." if btn_cross_modal else ("Chemo Agent формулирует ответ..." if lang == "ru" else "Chemo Agent generating response...")
                loader_ph.markdown(
                    f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
                    unsafe_allow_html=True,
                )
                req_messages = [{"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chemo_agent_messages"][-6:]
                ]
                stream_gen = chemo_agent.stream_chat(
                    messages=req_messages,
                    provider=provider_code,
                    api_key=active_key,
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
        "Спросите Chemo Agent о формулах, индексах или реакциях..." if lang == "ru" else "Ask Chemo Agent about formulas, indices, or reactions...",
        key="agent_user_chat_input",
    )
    if user_prompt:
        if not active_key:
            st.error("⚠️ Сервис ИИ временно недоступен. Обратитесь к администратору платформы." if lang == "ru" else "⚠️ AI service is currently unavailable. Please contact the platform administrator.")
        elif is_exhausted:
            st.error("⚠️ Достигнут суточный лимит запросов. Лимит автоматически сбросится в 00:00 UTC." if lang == "ru" else "⚠️ Daily query quota reached. Quota will automatically reset at 00:00 UTC.")
        elif not _check_debounce_allowed():
            st.warning("Пожалуйста, подождите 2 секунды перед следующим запросом." if lang == "ru" else "Please wait 2 seconds between queries.")
        else:
            context_hint = ""
            if active_data and not any(active_data["name"] in m.get("content", "") for m in st.session_state["chemo_agent_messages"][-3:]):
                context_hint = f"[Контекст: выбран образец {active_data['name']}]\n"

            st.session_state["chemo_agent_messages"].append({
                "role": "user",
                "content": f"{context_hint}{user_prompt}" if context_hint else user_prompt,
                "display_content": user_prompt,
            })
            with chat_box:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    loader_ph = st.empty()
                    loader_msg = "Chemo Agent формулирует ответ..." if lang == "ru" else "Chemo Agent generating response..."
                    loader_ph.markdown(
                        f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
                        unsafe_allow_html=True,
                    )
                    req_messages = [{"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT}] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chemo_agent_messages"][-8:]
                    ]
                    stream_gen = chemo_agent.stream_chat(
                        messages=req_messages,
                        provider=provider_code,
                        api_key=active_key,
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
    provider_code, active_key, model, temp, *extra = _get_active_ai_config()
    usage = chemo_agent.get_daily_usage(100)
    used_count = usage["count"]
    limit = usage["limit"]
    is_exhausted = used_count >= limit
    cache_key = f"ai_summary_{sample_name}"

    st.markdown("### 🤖 Chemo Agent" if lang == "ru" else "### 🤖 Chemo Agent")
    
    col_info, col_btn = st.columns([3.0, 1.2])
    with col_info:
        st.caption(
            f"Глубокая биогеохимическая интерпретация Chemo Agent: генезис пробы, степень ароматической конденсации, техногенные маркеры. (Использовано сегодня: {used_count}/{limit})"
            if lang == "ru"
            else f"In-depth biogeochemical reasoning powered by Chemo Agent: origin, aromatic condensation, contamination markers. (Daily usage: {used_count}/{limit})"
        )
    with col_btn:
        run_btn = st.button(
            "⚡ Сгенерировать анализ" if lang == "ru" else "⚡ Generate Analysis",
            key=f"btn_run_ai_{sample_name}",
            disabled=is_exhausted,
            use_container_width=True,
            type="primary" if not st.session_state.get(cache_key) else "secondary",
        )

    if run_btn:
        if is_exhausted:
            st.error("⚠️ Достигнут суточный лимит в 100 запросов. Сброс в 00:00 UTC." if lang == "ru" else "⚠️ Daily limit of 100 queries reached. Resets at midnight.")
            return

        if not _check_debounce_allowed():
            st.warning("Пожалуйста, подождите 2 секунды перед следующим запросом." if lang == "ru" else "Please wait 2 seconds between queries.")
            return

        if not active_key:
            st.warning("Сервис ИИ временно недоступен. Обратитесь к администратору платформы." if lang == "ru" else "AI service is currently unavailable. Please contact the platform administrator.")
            return

        prompt = chemo_agent.build_sample_analysis_prompt(sample_name, sample_data, lang=lang)
        messages = [
            {"role": "system", "content": chemo_agent.CHEMO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        loader_ph = st.empty()
        loader_msg = "Chemo Agent анализирует спектральные дескрипторы..." if lang == "ru" else "Analyzing spectral descriptors..."
        loader_ph.markdown(
            f'<div class="chemo-loader-badge"><span class="chemo-loader-icon"></span><span>{loader_msg}</span></div>',
            unsafe_allow_html=True,
        )
        res = chemo_agent.call_chat(
            messages,
            provider=provider_code,
            api_key=active_key,
            model=model,
            temperature=temp,
        )
        loader_ph.empty()
        if res.get("success"):
            st.session_state[cache_key] = res.get("content", "")
            st.rerun()
        else:
            st.markdown(res.get("error", "Произошла ошибка"))

    cached_content = st.session_state.get(cache_key)
    if cached_content:
        with st.container(border=True):
            st.markdown(chemo_agent.normalize_latex(cached_content))
