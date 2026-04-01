"""Hook system for lifecycle event handling."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class HookEvent(str, Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PERMISSION_REQUEST = "PermissionRequest"
    PERMISSION_DENIED = "PermissionDenied"
    TASK_CREATED = "TaskCreated"
    TASK_COMPLETED = "TaskCompleted"
    CONFIG_CHANGE = "ConfigChange"
    CWD_CHANGED = "CwdChanged"
    FILE_CHANGED = "FileChanged"
    NOTIFICATION = "Notification"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"
    TEAMMATE_IDLE = "TeammateIdle"


HOOK_EVENTS = list(HookEvent)


@dataclass
class HookInput:
    event: HookEvent
    tool_name: str = ""
    tool_input: Any = None
    tool_output: Any = None
    tool_use_id: str = ""
    session_id: str = ""
    cwd: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookOutput:
    message: str = ""
    permission_update: dict[str, str] | None = None
    block: bool = False
    notification: dict[str, str] | None = None


@dataclass
class HookDefinition:
    command: str = ""
    handler: Callable[[HookInput], Awaitable[HookOutput | None]] | None = None
    matcher: str = ""  # Regex pattern for tool names
    timeout: int = 10000  # Milliseconds


HookConfig = dict[str, list[dict[str, Any]]]


class HookRegistry:
    """Registry for lifecycle hooks."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[HookDefinition]] = {}

    def register_from_config(self, config: HookConfig) -> None:
        """Register hooks from configuration dict."""
        for event_name, definitions in config.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                continue
            for defn in definitions:
                hook_def = HookDefinition(
                    command=defn.get("command", ""),
                    matcher=defn.get("matcher", ""),
                    timeout=defn.get("timeout", 10000),
                )
                self.register(event, hook_def)

    def register(self, event: HookEvent, definition: HookDefinition) -> None:
        """Register a hook for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(definition)

    async def execute(self, event: HookEvent, input: HookInput) -> list[HookOutput]:
        """Execute all hooks for an event."""
        hooks = self._hooks.get(event, [])
        results: list[HookOutput] = []

        for hook in hooks:
            # Check matcher
            if hook.matcher and input.tool_name:
                if not re.match(hook.matcher, input.tool_name):
                    continue

            try:
                if hook.handler:
                    output = await asyncio.wait_for(
                        hook.handler(input),
                        timeout=hook.timeout / 1000,
                    )
                    if output:
                        results.append(output)

                elif hook.command:
                    # Execute shell command
                    env_data = json.dumps({
                        "event": event.value,
                        "tool_name": input.tool_name,
                        "tool_input": input.tool_input,
                        "session_id": input.session_id,
                        "cwd": input.cwd,
                    }, default=str)

                    proc = await asyncio.create_subprocess_shell(
                        hook.command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env={**__import__("os").environ, "HOOK_INPUT": env_data},
                    )

                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(env_data.encode()),
                        timeout=hook.timeout / 1000,
                    )

                    if stdout:
                        try:
                            output_data = json.loads(stdout.decode())
                            results.append(HookOutput(
                                message=output_data.get("message", ""),
                                block=output_data.get("block", False),
                            ))
                        except json.JSONDecodeError:
                            results.append(HookOutput(message=stdout.decode().strip()))

            except asyncio.TimeoutError:
                results.append(HookOutput(message=f"Hook timed out for {event.value}"))
            except Exception as e:
                results.append(HookOutput(message=f"Hook error: {e}"))

        return results

    def clear(self) -> None:
        self._hooks.clear()


def create_hook_registry() -> HookRegistry:
    return HookRegistry()
