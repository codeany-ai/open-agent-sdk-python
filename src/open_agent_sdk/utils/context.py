"""Git and project context utilities."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

# Cache for context results
_context_cache: dict[str, str] = {}


async def _run_command(cmd: list[str], cwd: str) -> str:
    """Run a command and return stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


async def get_git_status(cwd: str) -> dict[str, str]:
    """Get git status information for the working directory."""
    result: dict[str, str] = {}

    # Check if it's a git repo
    git_dir = await _run_command(["git", "rev-parse", "--git-dir"], cwd)
    if not git_dir:
        return result

    result["is_git"] = "true"

    # Get current branch
    branch = await _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch:
        result["branch"] = branch

    # Get default branch
    for candidate in ["main", "master"]:
        check = await _run_command(["git", "rev-parse", "--verify", f"refs/heads/{candidate}"], cwd)
        if check:
            result["default_branch"] = candidate
            break

    # Get status summary
    status = await _run_command(["git", "status", "--short"], cwd)
    if status:
        result["status"] = status
    else:
        result["status"] = "(clean)"

    # Get recent commits
    log = await _run_command(
        ["git", "log", "--oneline", "-5", "--no-decorate"],
        cwd,
    )
    if log:
        result["recent_commits"] = log

    # Get git user
    user = await _run_command(["git", "config", "user.name"], cwd)
    if user:
        result["user"] = user

    return result


async def get_system_context(cwd: str) -> str:
    """Build system context string with git status and project info."""
    cache_key = f"system:{cwd}"
    if cache_key in _context_cache:
        return _context_cache[cache_key]

    parts: list[str] = []

    # Working directory
    parts.append(f"Primary working directory: {cwd}")

    # Git status
    git = await get_git_status(cwd)
    if git.get("is_git"):
        parts.append(f"  Is a git repository: true")
        if git.get("branch"):
            parts.append(f"  Current branch: {git['branch']}")
        if git.get("default_branch"):
            parts.append(f"  Main branch: {git['default_branch']}")
        if git.get("user"):
            parts.append(f"  Git user: {git['user']}")
        if git.get("status"):
            parts.append(f"  Status: {git['status']}")
        if git.get("recent_commits"):
            parts.append(f"  Recent commits:\n{git['recent_commits']}")

    # Platform info
    import platform

    parts.append(f"Platform: {platform.system().lower()}")
    parts.append(f"OS Version: {platform.platform()}")

    context = "\n".join(parts)
    _context_cache[cache_key] = context
    return context


async def get_user_context(cwd: str) -> str:
    """Get user-defined context from AGENT.md or similar files."""
    parts: list[str] = []

    # Check for AGENT.md or .claude/prompt.md
    context_files = [
        os.path.join(cwd, "AGENT.md"),
        os.path.join(cwd, ".claude", "prompt.md"),
        os.path.join(cwd, "CLAUDE.md"),
    ]

    for filepath in context_files:
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    parts.append(content.strip())
            except Exception:
                pass

    # Add current date
    parts.append(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

    return "\n\n".join(parts)


def clear_context_cache() -> None:
    """Clear cached context data."""
    _context_cache.clear()
