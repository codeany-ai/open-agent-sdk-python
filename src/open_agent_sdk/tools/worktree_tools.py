"""Git worktree management tools."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class EnterWorktreeTool(BaseTool):
    _name = "EnterWorktree"
    _description = "Create a git worktree or isolated workspace for safe experimentation."
    _input_schema = ToolInputSchema(
        properties={
            "branch": {"type": "string", "description": "Branch name for the worktree"},
            "path": {"type": "string", "description": "Custom path for the worktree"},
        },
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        branch = input.get("branch", f"worktree-{uuid.uuid4().hex[:8]}")
        wt_path = input.get("path", "")

        if not wt_path:
            wt_path = os.path.join(tempfile.gettempdir(), f"agent-worktree-{uuid.uuid4().hex[:8]}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "add", "-b", branch, wt_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace")
                return ToolResult(tool_use_id="", content=f"Error creating worktree: {error}", is_error=True)

            return ToolResult(
                tool_use_id="",
                content=f"Created worktree at {wt_path} on branch {branch}",
            )

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)


class ExitWorktreeTool(BaseTool):
    _name = "ExitWorktree"
    _description = "Exit worktree with keep or remove action."
    _input_schema = ToolInputSchema(
        properties={
            "path": {"type": "string", "description": "Worktree path"},
            "action": {
                "type": "string",
                "description": "Action: 'keep' or 'remove'",
                "enum": ["keep", "remove"],
            },
        },
        required=["path"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        wt_path = input.get("path", "")
        action = input.get("action", "remove")

        if not wt_path:
            return ToolResult(tool_use_id="", content="Error: path is required", is_error=True)

        if action == "remove":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "worktree", "remove", "--force", wt_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.cwd,
                )
                await proc.communicate()
                return ToolResult(tool_use_id="", content=f"Removed worktree at {wt_path}")
            except Exception as e:
                return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)

        return ToolResult(tool_use_id="", content=f"Keeping worktree at {wt_path}")
