"""Grep tool - regex content search using ripgrep."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class GrepTool(BaseTool):
    """Search file contents with regex patterns using ripgrep."""

    _name = "Grep"
    _description = (
        "A powerful search tool built on ripgrep. "
        "Supports full regex syntax and multiple output modes."
    )
    _input_schema = ToolInputSchema(
        properties={
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (defaults to cwd)",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py')",
            },
            "type": {
                "type": "string",
                "description": "File type to search (e.g. 'py', 'js', 'rust')",
            },
            "output_mode": {
                "type": "string",
                "description": "Output mode: 'content', 'files_with_matches', or 'count'",
                "enum": ["content", "files_with_matches", "count"],
            },
            "-i": {
                "type": "boolean",
                "description": "Case insensitive search",
            },
            "-n": {
                "type": "boolean",
                "description": "Show line numbers",
            },
            "-A": {
                "type": "number",
                "description": "Lines to show after each match",
            },
            "-B": {
                "type": "number",
                "description": "Lines to show before each match",
            },
            "-C": {
                "type": "number",
                "description": "Context lines around each match",
            },
            "context": {
                "type": "number",
                "description": "Alias for -C context lines",
            },
            "head_limit": {
                "type": "number",
                "description": "Limit output to first N lines (default 250)",
            },
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline matching",
            },
        },
        required=["pattern"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        search_path = input.get("path", context.cwd)
        glob_pattern = input.get("glob")
        file_type = input.get("type")
        output_mode = input.get("output_mode", "files_with_matches")
        case_insensitive = input.get("-i", False)
        show_line_numbers = input.get("-n", True)
        after_context = input.get("-A")
        before_context = input.get("-B")
        context_lines = input.get("-C") or input.get("context")
        head_limit = input.get("head_limit", 250)
        multiline = input.get("multiline", False)

        if not pattern:
            return ToolResult(tool_use_id="", content="Error: pattern is required", is_error=True)

        # Build command
        rg_path = shutil.which("rg")
        if rg_path:
            cmd = [rg_path]
        else:
            # Fallback to grep
            cmd = ["grep", "-r"]

        # Output mode
        if rg_path:
            if output_mode == "files_with_matches":
                cmd.append("--files-with-matches")
            elif output_mode == "count":
                cmd.append("--count")
            # content mode is default

            if case_insensitive:
                cmd.append("-i")
            if show_line_numbers and output_mode == "content":
                cmd.append("-n")
            if multiline:
                cmd.extend(["-U", "--multiline-dotall"])
            if glob_pattern:
                cmd.extend(["--glob", glob_pattern])
            if file_type:
                cmd.extend(["--type", file_type])
            if after_context and output_mode == "content":
                cmd.extend(["-A", str(int(after_context))])
            if before_context and output_mode == "content":
                cmd.extend(["-B", str(int(before_context))])
            if context_lines and output_mode == "content":
                cmd.extend(["-C", str(int(context_lines))])

            cmd.extend(["--", pattern, search_path])
        else:
            if case_insensitive:
                cmd.append("-i")
            if show_line_numbers and output_mode == "content":
                cmd.append("-n")
            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            cmd.extend([pattern, search_path])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            output = stdout.decode("utf-8", errors="replace")

            # Apply head_limit
            if head_limit and head_limit > 0:
                lines = output.split("\n")
                if len(lines) > head_limit:
                    output = "\n".join(lines[:head_limit])
                    output += f"\n\n... (output limited to {head_limit} lines)"

            if not output.strip():
                return ToolResult(tool_use_id="", content="No matches found.")

            return ToolResult(tool_use_id="", content=output.strip())

        except asyncio.TimeoutError:
            return ToolResult(tool_use_id="", content="Search timed out after 30 seconds", is_error=True)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error in grep: {e}", is_error=True)
