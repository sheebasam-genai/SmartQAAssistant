import re
from datetime import datetime

import streamlit as st

from src.ai_chatbot.chatbot import ChatBot, ChatResponse
from src.repository.defect_creator import (
    ALLOWED_FILE_TYPES,
    DefectCreator,
    MAX_FILE_SIZE,
    validate_attachment,
)
from src.repository.history_manager import HistoryManager


# ────────────────────────────────────────────────
# Text formatting helpers
# ────────────────────────────────────────────────
def clean_markdown_asterisks(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"\*\*(\s*)", "", text)
    text = re.sub(r"\*(\s*)", "", text)
    return text


def add_line_breaks(text: str) -> str:
    for h in HEADINGS:
        pattern = rf"(?<!^)\b{re.escape(h)}\b\s*:"
        text = re.sub(pattern, f"<strong>{h}:</strong> ", text)
    return text


def extract_first_line_from_messages(messages):
    if not messages:
        return "Empty chat"
    for m in messages:
        text = m if isinstance(m, str) else m.get("content", "")
        if not text:
            continue
        text = re.sub(r"<[^>]+>", "", text)
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if first_line:
            return first_line[:30] + "..." if len(first_line) > 30 else first_line
    return "Empty chat"


def _short(text: str, size: int = 110) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:size] + "..." if len(text) > size else text


def _build_loaded_assistant_message(item: dict) -> dict:
    """Convert a persisted conversation record into assistant chat message format."""
    assistant_msg = {
        "role": "assistant",
        "content": item.get("assistant_response", ""),
        "follow_ups": item.get("follow_ups", []) or item.get("suggested_followups", []) or [],
    }

    token_info = item.get("tokens_used") or {}
    if token_info:
        assistant_msg["token_info"] = {
            "total": int(token_info.get("total_tokens", token_info.get("total", 0)) or 0),
            "prompt": int(token_info.get("prompt_tokens", token_info.get("prompt", 0)) or 0),
            "completion": int(token_info.get("completion_tokens", token_info.get("completion", 0)) or 0),
            "llm_used": bool(token_info.get("llm_used", False)),
        }
    return assistant_msg


def _load_history_conversation(item: dict):
    messages = item.get("messages", [])

    chat_messages = []
    for m in messages:
        chat_messages.append({"role": "user", "content": m.get("user_query", "")})
        chat_messages.append({
            "role": "assistant",
            "content": m.get("assistant_response", "")
        })

    st.session_state.messages = chat_messages

# ────────────────────────────────────────────────
# Session initialization
# ────────────────────────────────────────────────
def _init_session_state():
    defaults = {
        "messages": [],
        "history": [],
        "chat_name": "New chat",
        "show_create_form": False,
        "pending_followup": None,
        # token stats
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        # defect form state
        "defect_title": "",
        "defect_description": "",
        "defect_severity": "P1",
        "defect_opco": "DLL",
        "defect_tcode": "",
        "defect_release": "R3.1.0",
        "defect_test_sets": "",
        "defect_test_cases": "",
        # AI title assistant state
        "ai_issue_details": "",
        "ai_requirement_id": "",
        "ai_username": "qa_user",
        "ai_suggested_title": "",
        "ai_title_generated_once": False,
        "ai_title_last_tokens": None,
        # interactive history state
        "loaded_history_conversation_id": None,
        "loaded_history_conversation": None,
        "history_load_feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatBot()

    if "history_manager" not in st.session_state:
        st.session_state.history_manager = HistoryManager()


# ────────────────────────────────────────────────
# Page setup
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Defect Intelligence Assistant",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()
bot: ChatBot = st.session_state.chatbot
history_manager: HistoryManager = st.session_state.history_manager


# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ➕ Create New Defect")
    if st.button("🐞 Create Defect", key="create_defect_btn", use_container_width=True):
        st.session_state.show_create_form = not st.session_state.show_create_form

    if st.session_state.show_create_form:
        # Feature 1: AI Title Assistant (optional)
        st.markdown("#### 🤖 AI Title Assistant")
        st.caption("Optional: Generate a suggested defect title from issue details. You can edit it before creating the defect.")

        st.text_area(
            "Issue Details*",
            key="ai_issue_details",
            height=90,
            help="Describe what failed, where it failed, and expected vs actual behavior.",
            placeholder="Example: During transaction ME23N, screen load times out after save.",
        )
        st.text_input(
            "Requirement ID",
            key="ai_requirement_id",
            help="Optional requirement reference used to improve title context.",
            placeholder="e.g., REQ-1234",
        )
        st.text_input(
            "Username",
            key="ai_username",
            help="Optional. Auto-filled from session; used for form context and defect history entry.",
            placeholder="qa_user",
        )

        if st.button("Generate Title with AI 🤖", use_container_width=True, key="generate_ai_title_btn"):
            if not st.session_state.ai_issue_details.strip():
                st.warning("Please enter Issue Details first to generate a title.")
            else:
                try:
                    with st.spinner("Generating title with AI..."):
                        suggested_title, usage = bot.llm.generate_defect_title(
                            st.session_state.ai_issue_details,
                            st.session_state.ai_requirement_id,
                        )
                except Exception as exc:
                    suggested_title, usage = "", None
                    st.error(f"Title generation failed: {exc}")
                else:
                    if suggested_title and suggested_title.strip():
                        st.session_state.ai_suggested_title = suggested_title.strip()
                        st.session_state.defect_title = suggested_title.strip()
                        st.session_state.ai_title_generated_once = True
                        st.session_state.ai_title_last_tokens = usage

                        if usage and usage.total_tokens > 0:
                            st.session_state.total_tokens += usage.total_tokens
                            st.session_state.prompt_tokens += usage.prompt_tokens
                            st.session_state.completion_tokens += usage.completion_tokens
                            st.session_state.llm_calls += 1

                        st.success("AI title generated successfully. It has been copied to Title* below.")
                    else:
                        st.session_state.ai_title_last_tokens = usage
                        st.error("AI could not generate a title. Please try again or enter Title* manually.")

        if st.session_state.ai_suggested_title:
            st.info(f"Generated Title: {st.session_state.ai_suggested_title}")

        if st.session_state.ai_title_last_tokens:
            u = st.session_state.ai_title_last_tokens
            st.caption(f"🔢 Title generation tokens: {u.total_tokens} (prompt {u.prompt_tokens} · completion {u.completion_tokens})")

        st.markdown("---")
        st.markdown("**New Defect Details**")

        st.text_input(
            "Title*",
            key="defect_title",
            help="Editable title. You can manually enter one or use the AI generated title.",
            placeholder="e.g., [ME23N] Timeout occurs during purchase order save",
        )
        st.text_area(
            "Description/Issue*",
            key="defect_description",
            height=100,
            placeholder="Describe the defect in detail...",
        )
        st.selectbox(
            "Severity*",
            options=["P1", "P2", "P3", "P4"],
            key="defect_severity",
            help="P1: Highest, P2: High, P3: Medium, P4: Low",
        )
        st.selectbox(
            "OpCo*",
            options=["DLL", "MI", "AB", "DS"],
            key="defect_opco",
            help="Operating Company",
        )
        st.text_input("TCode*", key="defect_tcode", placeholder="e.g., FB60, VF01")
        st.selectbox(
            "Release*",
            options=["R3.1.0", "R3.1.2", "R3.2.2", "R3.3.2", "R3.4.1", "R3.4.2", "R3.5.2"],
            key="defect_release",
        )

        st.markdown("**Optional Fields**")
        st.text_input("Impacted Test Sets", key="defect_test_sets", placeholder="e.g., Set_1, Set_2")
        st.text_input("Impacted Test Cases", key="defect_test_cases", placeholder="e.g., TC_101, TC_102")

        # Existing attachment support
        st.markdown("**Attachments**")
        uploaded_files = st.file_uploader(
            "Upload files (images, logs, docs — max 5 MB each)",
            accept_multiple_files=True,
            type=list(ALLOWED_FILE_TYPES),
            key="defect_attachments",
            help=f"Allowed: {', '.join(sorted(ALLOWED_FILE_TYPES))}. Max size per file: {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )

        if uploaded_files:
            for uf in uploaded_files:
                file_bytes = uf.read()
                uf.seek(0)
                result = validate_attachment(file_bytes, uf.name)
                size_kb = len(file_bytes) / 1024
                if result.valid:
                    st.caption(f"✅ {uf.name} ({size_kb:.0f} KB)")
                else:
                    st.caption(f"❌ {uf.name} — {result.error}")

        if st.button("✅ Create Defect", use_container_width=True, key="create_defect_submit_btn"):
            missing_fields = []
            if not st.session_state.defect_title.strip():
                missing_fields.append("Title")
            if not st.session_state.defect_description.strip():
                missing_fields.append("Description")
            if not st.session_state.defect_severity:
                missing_fields.append("Severity")
            if not st.session_state.defect_opco:
                missing_fields.append("OpCo")
            if not st.session_state.defect_tcode.strip():
                missing_fields.append("TCode")
            if not st.session_state.defect_release:
                missing_fields.append("Release")

            if missing_fields:
                st.error(f"Please fill in the mandatory fields: {', '.join(missing_fields)}")
            else:
                defect_data = {
                    "Title": st.session_state.defect_title,
                    "AI_Generated_Title": st.session_state.ai_suggested_title,
                    "Description": st.session_state.defect_description,
                    "Severity": st.session_state.defect_severity,
                    "OpCo": st.session_state.defect_opco,
                    "TCode": st.session_state.defect_tcode,
                    "Release": st.session_state.defect_release,
                    "ImpactedTestSets": st.session_state.defect_test_sets,
                    "ImpactedTestCases": st.session_state.defect_test_cases,
                }

                attachment_tuples = []
                if uploaded_files:
                    for uf in uploaded_files:
                        file_bytes = uf.read()
                        uf.seek(0)
                        attachment_tuples.append((file_bytes, uf.name))

                creator = DefectCreator(csv_path="db/defects_kedb.csv")
                success, message, issue_key = creator.create_defect(defect_data, attachments=attachment_tuples)

                if success:
                    st.success(message)
                    st.balloons()

                    att_note = ""
                    if attachment_tuples:
                        names = [t[1] for t in attachment_tuples]
                        att_note = f"\n**Attachments:** {', '.join(names)}"

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"🎉 New defect created successfully!\n\n"
                                f"**Issue Key:** {issue_key}\n"
                                f"**Title:** {st.session_state.defect_title}\n"
                                f"**Severity:** {st.session_state.defect_severity}\n"
                                f"**OpCo:** {st.session_state.defect_opco}\n"
                                f"**Release:** {st.session_state.defect_release}"
                                f"{att_note}"
                            ),
                        }
                    )

                    # Feature 2: save defect creation history
                    history_manager.save_defect_creation(
                        {
                            "defect_id": issue_key,
                            "created_at": datetime.utcnow().isoformat(),
                            "title": st.session_state.defect_title,
                            "description": st.session_state.defect_description,
                            "created_by": st.session_state.ai_username,
                            "ai_generated_title": st.session_state.ai_title_generated_once,
                        }
                    )

                    # reset form state
                    st.session_state.show_create_form = False
                    st.session_state.ai_title_generated_once = False
                    st.session_state.ai_title_last_tokens = None
                    st.session_state.ai_suggested_title = ""
                else:
                    st.error(message)

    st.markdown("---")

    # Existing local chat thread reset/history
    if st.button("📝 New chat", key="new_chat_btn"):
        if st.session_state.messages:
            st.session_state.history.append({"messages": st.session_state.messages.copy()})
        st.session_state.messages = []
        st.session_state.chat_name = "New chat"
        st.session_state.total_tokens = 0
        st.session_state.prompt_tokens = 0
        st.session_state.completion_tokens = 0
        st.session_state.llm_calls = 0
        st.session_state.pending_followup = None
        st.session_state.loaded_history_conversation_id = None
        st.session_state.loaded_history_conversation = None
        st.session_state.history_load_feedback = None
        bot.reset_session_stats()

    st.markdown("---")

    search_q = st.text_input("Search chats", key="search_chats")
    filtered_chats = []
    for idx, entry in enumerate(st.session_state.history):
        messages = entry.get("messages", [])
        label = extract_first_line_from_messages(messages)
        if not search_q or search_q.lower() in label.lower():
            filtered_chats.append((idx, label, messages))

    for idx, label, messages in filtered_chats:
        if st.button(label, key=f"chat_{idx}"):
            st.session_state.messages = messages.copy()
            st.session_state.chat_name = label

    st.markdown("---")
    st.markdown(f"**Current:** {st.session_state.chat_name}")

    st.markdown("---")
    st.markdown("### 📊 Session Token Stats")
    col1, col2 = st.columns(2)
    col1.metric("Total Tokens", st.session_state.total_tokens)
    col2.metric("Prompt Tokens", st.session_state.prompt_tokens)
    col3, col4 = st.columns(2)
    col3.metric("LLM Calls", st.session_state.llm_calls)
    col4.metric("Completion Tokens", st.session_state.completion_tokens)

    # Feature 2: Persistent history section
    st.markdown("---")
    st.markdown("### 🕘 History")

    st.markdown(
        """
        <style>
        .history-card {
            border: 1px solid #d7dce5;
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 8px;
            background: #f9fbff;
            cursor: pointer;
        }
        .history-card:hover {
            border-color: #7f56d9;
            background: #f3edff;
        }
        .history-click-hint {
            font-size: 12px;
            color: #6b7280;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history_load_feedback:
        st.success(st.session_state.history_load_feedback)
        st.session_state.history_load_feedback = None

    with st.expander("Recent Conversations (last 10)", expanded=True):
        recent = history_manager.get_recent_conversations(limit=10)
        if not recent:
            st.caption("No conversation history yet.")
        else:
            for idx, item in enumerate(recent):
                item_id = item.get("id", item.get("timestamp", f"row_{idx}"))
                is_active = st.session_state.loaded_history_conversation_id == item_id
                ts = item.get("timestamp", "")

                st.markdown("<div class='history-card'>", unsafe_allow_html=True)
                st.markdown(f"**{ts}**")
                messages = item.get("messages", [])

                if messages:
                    last_msg = messages[-1]
                    st.caption(f"Q: {_short(last_msg.get('user_query', ''), 72)}")
                    st.caption(f"A: {_short(last_msg.get('assistant_response', ''), 72)}")
                else:
                    st.caption("No messages")

                if is_active:
                    st.caption("✅ Currently loaded")
                else:
                    st.caption("↗️ Click View to load this conversation")

                if st.button(
                    "👁️ View Conversation",
                    key=f"view_history_{idx}_{item_id}",
                    use_container_width=True,
                    help="Load this conversation into the main chat area",
                ):
                    with st.spinner("Loading conversation..."):
                        _load_history_conversation(item)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    history_keyword = st.text_input("Search History", key="history_keyword", placeholder="Keyword...")
    if history_keyword:
        matches = history_manager.search_history(keyword=history_keyword)
        conv = matches.get("conversations", [])
        defects = matches.get("defect_history", [])

        st.caption(f"Found {len(conv)} conversation matches and {len(defects)} defect matches")

        if conv:
            st.markdown("**Conversation Matches**")
            for item in conv[:10]:
                st.markdown(f"- `{item.get('timestamp', '')}` · {_short(item.get('user_query', ''), 80)}")

        if defects:
            st.markdown("**Defect Matches**")
            for item in defects[:10]:
                did = item.get("defect_id", "")
                st.markdown(f"- `{did}` · {_short(item.get('title', ''), 80)}")


# ────────────────────────────────────────────────
# Header + styles
# ────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title {
            background: linear-gradient(90deg, #1A5276, #2471A3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }
        .main-title h2 {
            color: #FDFEFE;
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 5px;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title p {
            color: #EAF2F8;
            font-size: 17px;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
        }
    </style>
    <div class="main-title">
        <h2>🐞 Defect Intelligence Assistant</h2>
        <p>
            Your smart QA companion to search, analyze, and summarize software defects.<br>
            Just ask naturally — e.g., "show P1 defects in release R3.5.2" or "test cases for D010".
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.assistant-box {
    background-color: #f3e8ff;
    border: 1px solid #b26bff;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
    color: #4a148c;
    font-size: 16px;
    line-height: 1.5;
}
.token-badge {
    display: inline-block;
    background: #e0f7fa;
    border: 1px solid #00acc1;
    border-radius: 8px;
    padding: 2px 10px;
    margin: 4px 4px 4px 0;
    font-size: 13px;
    color: #006064;
}
.llm-badge {
    display: inline-block;
    background: #fff3e0;
    border: 1px solid #fb8c00;
    border-radius: 8px;
    padding: 2px 10px;
    margin: 4px;
    font-size: 13px;
    color: #e65100;
}
</style>
""",
    unsafe_allow_html=True,
)


HEADINGS = [
    "DefectID", "Defect ID", "Title", "Summary", "RootCause", "Root Cause", "Resolution",
    "TCode", "T-Code", "Release", "Status", "OpCo", "Severity",
    "Impacted Test Cases", "Impacted Test Sets", "Test Cases", "Test Case",
    "Key Details", "ImpactedTestSets", "ImpactedTestCases",
    "What happened", "Severity/Impact", "ProbableRoot Cause", "ResolutionStatus", "Next steps",
]


def bold_headings(content: str) -> str:
    if not isinstance(content, str):
        return content
    html = content
    for h in HEADINGS:
        pattern = re.compile(rf"(\*{{0,2}}\s*)\b{re.escape(h)}\b\s*:", flags=re.IGNORECASE)
        html = pattern.sub(f"<strong>{h}:</strong>", html)
    html = html.replace("\n", "<br>")
    return html


# ────────────────────────────────────────────────
# Display chat messages
# ────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            cleaned = clean_markdown_asterisks(msg["content"])
            with_breaks = add_line_breaks(cleaned)
            content_html = bold_headings(with_breaks)
            st.markdown(f"<div class='assistant-box'>{content_html}</div>", unsafe_allow_html=True)

            token_info = msg.get("token_info")
            if token_info:
                total = token_info.get("total", 0)
                prompt = token_info.get("prompt", 0)
                completion = token_info.get("completion", 0)
                llm_used = token_info.get("llm_used", False)
                badge_html = f"<span class='token-badge'>🔢 Tokens: {total} (prompt {prompt} · completion {completion})</span>"
                if llm_used:
                    badge_html += " <span class='llm-badge'>🤖 LLM used</span>"
                st.markdown(badge_html, unsafe_allow_html=True)

            follow_ups = msg.get("follow_ups", [])
            if follow_ups:
                cols = st.columns(len(follow_ups))
                for j, fu in enumerate(follow_ups):
                    if cols[j].button(f"💡 {fu}", key=f"fu_{i}_{j}"):
                        st.session_state.pending_followup = fu
                        st.rerun()
        else:
            st.markdown(msg["content"])


# ────────────────────────────────────────────────
# User input + chatbot call
# ────────────────────────────────────────────────
pending = st.session_state.pop("pending_followup", None)
user_input = pending or st.chat_input("💬 Ask me about defects, test cases, releases, or patterns...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Chatbot automatically saves conversation to HistoryManager
    response: ChatResponse = bot.ask_with_tokens(user_input)

    msg_dict = {
        "role": "assistant",
        "content": response.answer,
        "follow_ups": response.follow_ups,
    }

    if response.token_usage and response.token_usage.total_tokens > 0:
        msg_dict["token_info"] = {
            "total": response.token_usage.total_tokens,
            "prompt": response.token_usage.prompt_tokens,
            "completion": response.token_usage.completion_tokens,
            "llm_used": response.llm_used,
        }
        st.session_state.total_tokens += response.token_usage.total_tokens
        st.session_state.prompt_tokens += response.token_usage.prompt_tokens
        st.session_state.completion_tokens += response.token_usage.completion_tokens
        st.session_state.llm_calls += 1
    elif response.llm_used:
        msg_dict["token_info"] = {"total": 0, "prompt": 0, "completion": 0, "llm_used": True}

    st.session_state.messages.append(msg_dict)

    with st.chat_message("assistant"):
        cleaned = clean_markdown_asterisks(response.answer)
        with_breaks = add_line_breaks(cleaned)
        content_html = bold_headings(with_breaks)
        st.markdown(f"<div class='assistant-box'>{content_html}</div>", unsafe_allow_html=True)

        if msg_dict.get("token_info"):
            ti = msg_dict["token_info"]
            badge_html = f"<span class='token-badge'>🔢 Tokens: {ti['total']} (prompt {ti['prompt']} · completion {ti['completion']})</span>"
            if ti.get("llm_used"):
                badge_html += " <span class='llm-badge'>🤖 LLM used</span>"
            st.markdown(badge_html, unsafe_allow_html=True)

        if response.follow_ups:
            cols = st.columns(len(response.follow_ups))
            new_msg_index = len(st.session_state.messages) - 1
            for j, fu in enumerate(response.follow_ups):
                # Use stable unique keys per assistant message to avoid replaying stale button state.
                btn_key = f"fu_{new_msg_index}_{j}"
                if cols[j].button(f"💡 {fu}", key=btn_key):
                    st.session_state.pending_followup = fu
                    st.rerun()

    st.rerun()
