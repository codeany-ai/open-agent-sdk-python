"""Session management - persistence for conversation history."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any

# Default storage location
SESSION_DIR = os.path.join(os.path.expanduser("~"), ".open-agent-sdk", "sessions")


def _session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, session_id)


def _transcript_path(session_id: str) -> str:
    return os.path.join(_session_path(session_id), "transcript.json")


async def save_session(
    session_id: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save session to disk."""
    path = _session_path(session_id)
    os.makedirs(path, exist_ok=True)

    data = {
        "metadata": {
            "id": session_id,
            "updatedAt": datetime.now().isoformat(),
            "messageCount": len(messages),
            **(metadata or {}),
        },
        "messages": messages,
    }

    with open(_transcript_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


async def load_session(session_id: str) -> dict[str, Any] | None:
    """Load session from disk."""
    path = _transcript_path(session_id)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


async def list_sessions() -> list[dict[str, Any]]:
    """List all sessions, sorted by most recent."""
    if not os.path.exists(SESSION_DIR):
        return []

    sessions = []
    for entry in os.listdir(SESSION_DIR):
        path = os.path.join(SESSION_DIR, entry, "transcript.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(data.get("metadata", {"id": entry}))
            except Exception:
                pass

    sessions.sort(key=lambda s: s.get("updatedAt", ""), reverse=True)
    return sessions


async def fork_session(source_session_id: str, new_session_id: str | None = None) -> str | None:
    """Copy a session with a new ID."""
    source_path = _session_path(source_session_id)
    if not os.path.exists(source_path):
        return None

    new_id = new_session_id or str(uuid.uuid4())
    new_path = _session_path(new_id)

    shutil.copytree(source_path, new_path)

    # Update metadata
    transcript = _transcript_path(new_id)
    if os.path.exists(transcript):
        with open(transcript, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["metadata"]["id"] = new_id
        data["metadata"]["updatedAt"] = datetime.now().isoformat()
        data["metadata"]["forkedFrom"] = source_session_id
        with open(transcript, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return new_id


async def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Get messages from a session."""
    data = await load_session(session_id)
    if data:
        return data.get("messages", [])
    return []


async def get_session_info(session_id: str) -> dict[str, Any] | None:
    """Get session metadata."""
    data = await load_session(session_id)
    if data:
        return data.get("metadata")
    return None


async def rename_session(session_id: str, title: str) -> None:
    """Rename a session."""
    data = await load_session(session_id)
    if data:
        data["metadata"]["title"] = title
        data["metadata"]["updatedAt"] = datetime.now().isoformat()
        with open(_transcript_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


async def tag_session(session_id: str, tags: list[str]) -> None:
    """Tag a session."""
    data = await load_session(session_id)
    if data:
        data["metadata"]["tags"] = tags
        data["metadata"]["updatedAt"] = datetime.now().isoformat()
        with open(_transcript_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


async def append_to_session(session_id: str, message: dict[str, Any]) -> None:
    """Append a message to an existing session."""
    data = await load_session(session_id)
    if data:
        data["messages"].append(message)
        data["metadata"]["messageCount"] = len(data["messages"])
        data["metadata"]["updatedAt"] = datetime.now().isoformat()
        with open(_transcript_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


async def delete_session(session_id: str) -> bool:
    """Delete a session."""
    path = _session_path(session_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False
