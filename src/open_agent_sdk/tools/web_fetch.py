"""WebFetch tool - HTTP fetch with content extraction."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class WebFetchTool(BaseTool):
    """Fetch content from a URL."""

    _name = "WebFetch"
    _description = "Fetches content from a URL and returns the text content."
    _input_schema = ToolInputSchema(
        properties={
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers",
            },
        },
        required=["url"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = input.get("url", "")
        headers = input.get("headers", {})

        if not url:
            return ToolResult(tool_use_id="", content="Error: url is required", is_error=True)

        try:
            def _fetch():
                req = Request(url)
                req.add_header("User-Agent", "OpenAgentSDK/0.1")
                for k, v in headers.items():
                    req.add_header(k, v)

                with urlopen(req, timeout=30) as response:
                    content_type = response.headers.get("Content-Type", "")
                    body = response.read(512 * 1024)  # Max 512KB
                    return body, content_type

            body, content_type = await asyncio.get_event_loop().run_in_executor(None, _fetch)

            text = body.decode("utf-8", errors="replace")

            # Simple HTML to text conversion
            if "html" in content_type.lower():
                # Remove script/style
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                # Remove tags
                text = re.sub(r"<[^>]+>", " ", text)
                # Collapse whitespace
                text = re.sub(r"\s+", " ", text).strip()

            # Truncate
            if len(text) > 100000:
                text = text[:100000] + "\n\n... [truncated]"

            return ToolResult(tool_use_id="", content=text)

        except URLError as e:
            return ToolResult(tool_use_id="", content=f"Error fetching URL: {e}", is_error=True)
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)
