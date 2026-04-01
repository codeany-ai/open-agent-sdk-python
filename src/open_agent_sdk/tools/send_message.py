"""SendMessage tool - inter-agent messaging."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class SendMessageTool(BaseTool):
    """Send messages between agents via mailboxes."""

    _name = "SendMessage"
    _description = "Send a message to another agent or broadcast to all agents."
    _input_schema = ToolInputSchema(
        properties={
            "to": {
                "type": "string",
                "description": "Agent name to send to, or '*' for broadcast",
            },
            "content": {
                "type": "string",
                "description": "Message content",
            },
            "type": {
                "type": "string",
                "description": "Message type",
                "enum": ["text", "shutdown_request", "shutdown_response", "plan_approval_response"],
            },
        },
        required=["to", "content"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import write_to_mailbox, _mailboxes

        to = input.get("to", "")
        content = input.get("content", "")
        msg_type = input.get("type", "text")

        if not to or not content:
            return ToolResult(tool_use_id="", content="Error: 'to' and 'content' are required", is_error=True)

        message = {"type": msg_type, "content": content, "from": "agent"}

        if to == "*":
            # Broadcast to all known agents
            for agent_name in list(_mailboxes.keys()):
                write_to_mailbox(agent_name, message)
            return ToolResult(tool_use_id="", content="Message broadcast to all agents.")
        else:
            write_to_mailbox(to, message)
            return ToolResult(tool_use_id="", content=f"Message sent to {to}.")
