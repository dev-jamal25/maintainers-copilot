"""Maintainer's Copilot — internal Streamlit app.

Authenticated surface for maintainers/admins: login, chat with the tool-calling assistant, inspect
long-term memory, and (admins) manage widget configs + invite users. Calls the backend API only —
never the database.
"""

from __future__ import annotations

import os

import streamlit as st
from api_client import ApiClient, ApiError

API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000")
client = ApiClient(API_BASE_URL)

st.set_page_config(page_title="Maintainer's Copilot", page_icon="🛠️", layout="wide")


def _ensure_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("messages", [])


def login_view() -> None:
    st.title("🛠️ Maintainer's Copilot")
    st.caption("Sign in to triage issues with the assistant.")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            token = client.login(email, password)
            st.session_state.token = token
            st.session_state.user = client.me(token)
            st.rerun()
        except ApiError as exc:
            st.error("Login failed." if exc.status_code in (400, 401) else exc.message)


def chat_view() -> None:
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about an issue, or paste issue text…")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            result = client.chat(st.session_state.token, prompt, st.session_state.conversation_id)
        except ApiError as exc:
            st.error(f"Chat failed: {exc.message}")
            return
        st.session_state.conversation_id = result["conversation_id"]
        st.markdown(result["answer"])
        tool_events = [e for e in result.get("events", []) if e["type"] in ("tool_call", "tool_result")]
        if tool_events:
            with st.expander("Tool activity"):
                st.json(tool_events)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def memory_view() -> None:
    st.header("Memory inspector")
    st.caption("Long-term episodic memories saved for you (write happens only via the assistant).")
    try:
        memories = client.list_memories(st.session_state.token)
    except ApiError as exc:
        st.error(f"Could not load memories: {exc.message}")
        return
    if not memories:
        st.info("No memories yet.")
        return
    for memory in memories:
        st.markdown(f"**{memory['event_type']}** — {memory.get('subject') or ''}")
        st.write(memory["content"])
        st.caption(memory["created_at"])
        st.divider()


def admin_view() -> None:
    st.header("Admin")
    if (st.session_state.user or {}).get("role") != "admin":
        st.warning("Admin role required for these actions.")
        return

    st.subheader("Invite a user")
    with st.form("invite"):
        invite_email = st.text_input("Email to invite")
        role = st.selectbox("Role", ["user", "admin"])
        if st.form_submit_button("Create invite") and invite_email:
            try:
                invite = client.create_invite(st.session_state.token, invite_email, role)
                st.success("Invite created. Share this token with the user:")
                st.code(invite["token"])
            except ApiError as exc:
                st.error(exc.message)

    st.subheader("Widgets")
    with st.form("widget"):
        origins = st.text_area("Allowed origins (one per line)", "http://localhost:8080")
        greeting = st.text_input("Greeting", "Hi! Ask me about this project's issues.")
        theme = st.selectbox("Theme", ["light", "dark"])
        if st.form_submit_button("Create widget"):
            payload = {
                "allowed_origins": [o.strip() for o in origins.splitlines() if o.strip()],
                "greeting": greeting,
                "theme": theme,
            }
            try:
                widget = client.create_widget(st.session_state.token, payload)
                st.success(f"Created widget {widget['widget_id']}")
            except ApiError as exc:
                st.error(exc.message)

    try:
        widgets = client.list_widgets(st.session_state.token)
        st.dataframe(
            [
                {
                    "widget_id": w["widget_id"],
                    "origins": ", ".join(w["allowed_origins"]),
                    "active": w["is_active"],
                }
                for w in widgets
            ],
            use_container_width=True,
        )
    except ApiError as exc:
        st.error(exc.message)


def main() -> None:
    _ensure_state()
    if st.session_state.token is None:
        login_view()
        return
    with st.sidebar:
        st.write(f"Signed in as **{(st.session_state.user or {}).get('email', '')}**")
        page = st.radio("Navigate", ["Chat", "Memory", "Admin"])
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
    {"Chat": chat_view, "Memory": memory_view, "Admin": admin_view}[page]()


main()
