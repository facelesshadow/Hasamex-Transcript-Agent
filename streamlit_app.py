import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "checkpoints.db"
CHAT_INDEX_TABLE = "streamlit_chats"
CHAT_MESSAGES_TABLE = "streamlit_messages"


os.chdir(BASE_DIR)


@st.cache_resource(show_spinner=False)
def load_graph():
    from graph import graph

    return graph


def checkpoint_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def init_storage() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHAT_INDEX_TABLE} (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHAT_MESSAGES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_chat(thread_id: str, title: str | None = None) -> None:
    init_storage()

    clean_title = (title or f"Chat {short_thread_id(thread_id)}").strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            INSERT INTO {CHAT_INDEX_TABLE} (thread_id, title)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = CASE
                    WHEN excluded.title LIKE 'Chat %' THEN {CHAT_INDEX_TABLE}.title
                    ELSE excluded.title
                END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, clean_title),
        )


def save_visible_message(thread_id: str, role: str, content: str) -> None:
    init_storage()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            INSERT INTO {CHAT_MESSAGES_TABLE} (thread_id, role, content)
            VALUES (?, ?, ?)
            """,
            (thread_id, role, content),
        )


def fetch_threads() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []

    init_storage()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            WITH checkpoint_threads AS (
                SELECT
                    thread_id,
                    COUNT(*) AS checkpoints,
                    MAX(checkpoint_id) AS latest_checkpoint
                FROM checkpoints
                GROUP BY thread_id
            ),
            visible_threads AS (
                SELECT
                    thread_id,
                    MAX(created_at) AS latest_message
                FROM streamlit_messages
                GROUP BY thread_id
            )
            SELECT
                c.thread_id,
                c.title,
                COALESCE(t.checkpoints, 0) AS checkpoints,
                COALESCE(v.latest_message, t.latest_checkpoint, c.updated_at) AS latest_checkpoint
            FROM streamlit_chats c
            LEFT JOIN checkpoint_threads t ON t.thread_id = c.thread_id
            LEFT JOIN visible_threads v ON v.thread_id = c.thread_id

            UNION

            SELECT
                t.thread_id,
                NULL AS title,
                t.checkpoints,
                t.latest_checkpoint
            FROM checkpoint_threads t
            LEFT JOIN streamlit_chats c ON c.thread_id = t.thread_id
            WHERE c.thread_id IS NULL

            ORDER BY latest_checkpoint DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def short_thread_id(thread_id: str) -> str:
    return thread_id.split("-")[0] if "-" in thread_id else thread_id[:8]


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


def display_name(thread_id: str, fallback: str) -> str:
    try:
        state = load_graph().get_state(checkpoint_config(thread_id))
    except Exception:
        return fallback

    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            text = message_text(message)
            if text:
                return text[:42] + ("..." if len(text) > 42 else "")

    return fallback


def read_chat_messages(thread_id: str) -> list[tuple[str, str]]:
    init_storage()

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT role, content
            FROM {CHAT_MESSAGES_TABLE}
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()

    if rows:
        return [(role, content) for role, content in rows]

    try:
        state = load_graph().get_state(checkpoint_config(thread_id))
    except Exception as exc:
        return [("assistant", f"Could not load this chat: {exc}")]

    visible_messages: list[tuple[str, str]] = []

    for message in state.values.get("messages", []):
        text = message_text(message)
        if not text:
            continue

        if isinstance(message, HumanMessage):
            visible_messages.append(("user", text))

    answer = state.values.get("answer")
    if answer:
        visible_messages.append(("assistant", str(answer)))

    return visible_messages


def create_chat() -> str:
    thread_id = str(uuid.uuid4())
    save_chat(thread_id)
    return thread_id


def select_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id


def run_agent(prompt: str, thread_id: str) -> str:
    save_chat(thread_id, prompt[:42] + ("..." if len(prompt) > 42 else ""))
    save_visible_message(thread_id, "user", prompt)

    result = load_graph().invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=checkpoint_config(thread_id),
    )

    answer = result.get("answer")
    if answer:
        clean_answer = str(answer)
        save_visible_message(thread_id, "assistant", clean_answer)
        return clean_answer

    fallback = "I did not get a final answer from the graph."
    save_visible_message(thread_id, "assistant", fallback)
    return fallback


st.set_page_config(page_title="Hasamex Agent", page_icon=":material/chat:", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
    .small-muted {
        color: #667085;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "thread_id" not in st.session_state:
    threads = fetch_threads()
    st.session_state.thread_id = threads[0]["thread_id"] if threads else create_chat()

with st.sidebar:
    st.title("Chats")

    if st.button("New chat", use_container_width=True):
        select_thread(create_chat())
        st.rerun()

    st.divider()

    threads = fetch_threads()
    if threads:
        for thread in threads:
            thread_id = thread["thread_id"]
            label = thread.get("title") or display_name(thread_id, f"Chat {short_thread_id(thread_id)}")
            selected = thread_id == st.session_state.thread_id
            button_label = f"{'> ' if selected else ''}{label}"

            if st.button(button_label, key=f"thread-{thread_id}", use_container_width=True):
                select_thread(thread_id)
                st.rerun()
    else:
        st.caption("No saved chats yet.")

st.title("Hasamex Agent")

for role, content in read_chat_messages(st.session_state.thread_id):
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask about France, Germany, or the United Kingdom..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(prompt, st.session_state.thread_id)
            except Exception as exc:
                response = f"Something went wrong while running the agent: {exc}"
                save_visible_message(st.session_state.thread_id, "assistant", response)
            st.markdown(response)

    st.rerun()
