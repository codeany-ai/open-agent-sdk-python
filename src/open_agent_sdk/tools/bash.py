"""Bash tool - execute shell commands."""

from __future__ import annotations

import asyncio
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class BashTool(BaseTool):
    """Execute shell commands and return stdout/stderr."""

    _name = "Bash"
    _description = (
        "Executes a given bash command and returns its output. "
        "The working directory persists between commands."
    )
    _input_schema = ToolInputSchema(
        properties={
            "command": {
                "type": "string",
                "description": "The command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout in milliseconds (max 600000)",
            },
            "description": {
                "type": "string",
                "description": "Clear, concise description of what this command does",
            },
        },
        required=["command"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        if input:
            cmd = input.get("command", "")
            read_only_prefixes = [
                "ls", "cat", "head", "tail", "grep", "find", "which", "whoami",
                "pwd", "echo", "date", "git status", "git log", "git diff",
                "git branch", "git show", "git remote",
            ]
            cmd_stripped = cmd.strip()
            for prefix in read_only_prefixes:
                if cmd_stripped.startswith(prefix):
                    return True
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "")
        timeout_ms = input.get("timeout", 120000)

        if not command:
            return ToolResult(
                tool_use_id="",
                content="Error: command is required",
                is_error=True,
            )

        # Cap timeout
        timeout_ms = min(timeout_ms, 600000)
        timeout_s = timeout_ms / 1000.0

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
                env={**dict(__import__("os").environ), **context.env} if context.env else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    tool_use_id="",
                    content=f"Command timed out after {timeout_ms}ms",
                    is_error=True,
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate large output
            max_output = 100 * 1024  # 100KB
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + f"\n... [truncated, total {len(stdout_str)} bytes]"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + f"\n... [truncated, total {len(stderr_str)} bytes]"

            output = ""
            if stdout_str:
                output += stdout_str
            if stderr_str:
                if output:
                    output += "\n"
                output += stderr_str

            exit_code = proc.returncode or 0
            if exit_code != 0:
                output += f"\n(exit code: {exit_code})"

            return ToolResult(
                tool_use_id="",
                content=output if output else "(no output)",
                is_error=exit_code != 0,
            )

        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Error executing command: {e}",
                is_error=True,
            )
