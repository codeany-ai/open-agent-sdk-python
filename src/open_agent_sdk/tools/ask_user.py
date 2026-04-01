"""AskUser tool - prompt user for input."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult

QuestionHandler = Callable[[str], Awaitable[str]]

_question_handler: QuestionHandler | None = None


def set_question_handler(handler: QuestionHandler | None) -> None:
    global _question_handler
    _question_handler = handler


class AskUserQuestionTool(BaseTool):
    _name = "AskUserQuestion"
    _description = "Ask the user a question and wait for their response."
    _input_schema = ToolInputSchema(
        properties={
            "question": {"type": "string", "description": "The question to ask the user"},
        },
        required=["question"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        question = input.get("question", "")
        if not question:
            return ToolResult(tool_use_id="", content="Error: question is required", is_error=True)

        if _question_handler is None:
            return ToolResult(
                tool_use_id="",
                content="Error: no question handler configured",
                is_error=True,
            )

        try:
            answer = await _question_handler(question)
            return ToolResult(tool_use_id="", content=answer)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error getting user input: {e}", is_error=True)
