"""Config tool - configuration management."""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class ConfigTool(BaseTool):
    _name = "Config"
    _description = "Get, set, or clear configuration values."
    _input_schema = ToolInputSchema(
        properties={
            "action": {
                "type": "string",
                "description": "Action: 'get', 'set', 'clear', 'list'",
                "enum": ["get", "set", "clear", "list"],
            },
            "key": {"type": "string", "description": "Config key"},
            "value": {"description": "Config value (for set)"},
        },
        required=["action"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return input.get("action") in ("get", "list") if input else False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _config_store, set_config, clear_config

        action = input.get("action", "")
        key = input.get("key", "")

        if action == "get":
            if not key:
                return ToolResult(tool_use_id="", content="Error: key is required for get", is_error=True)
            value = _config_store.get(key)
            if value is None:
                return ToolResult(tool_use_id="", content=f"Config '{key}' not set.")
            return ToolResult(tool_use_id="", content=json.dumps(value, default=str))

        elif action == "set":
            if not key:
                return ToolResult(tool_use_id="", content="Error: key is required for set", is_error=True)
            set_config(key, input.get("value"))
            return ToolResult(tool_use_id="", content=f"Set config '{key}'.")

        elif action == "clear":
            if key:
                _config_store.pop(key, None)
                return ToolResult(tool_use_id="", content=f"Cleared config '{key}'.")
            else:
                clear_config()
                return ToolResult(tool_use_id="", content="Cleared all config.")

        elif action == "list":
            if not _config_store:
                return ToolResult(tool_use_id="", content="No configuration set.")
            return ToolResult(tool_use_id="", content=json.dumps(_config_store, indent=2, default=str))

        return ToolResult(tool_use_id="", content=f"Unknown action: {action}", is_error=True)
