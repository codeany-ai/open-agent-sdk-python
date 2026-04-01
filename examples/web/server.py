"""
Web Chat Server

A lightweight HTTP server providing:
  GET  /           — serves the chat UI
  POST /api/chat   — SSE stream of agent events
  POST /api/new    — resets the session

Run: python examples/web/server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from open_agent_sdk import Agent, AgentOptions, SDKMessageType  # noqa: E402

PORT = int(os.environ.get("PORT", "8083"))
INDEX_HTML = Path(__file__).parent / "index.html"

# Shared agent instance
_agent: Agent | None = None
_loop: asyncio.AbstractEventLoop | None = None


def get_or_create_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(AgentOptions(
            model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5"),
            api_key=os.environ.get("CODEANY_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
            base_url=os.environ.get("CODEANY_BASE_URL", ""),
            max_turns=20,
        ))
    return _agent


def reset_agent() -> None:
    global _agent
    if _agent is not None:
        try:
            _loop and _loop.run_until_complete(_agent.close())
        except Exception:
            pass
        _agent = None


async def run_chat(prompt: str, write_event):
    """Run agent query and emit SSE events."""
    agent = get_or_create_agent()
    import time
    start_ms = time.monotonic()

    try:
        async for ev in agent.query(prompt):
            if ev.type == SDKMessageType.ASSISTANT:
                # Extract text blocks and tool_use blocks from message
                msg = ev.message
                if isinstance(msg, dict):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text" and block.get("text"):
                                    write_event("text", {"text": block["text"]})
                                elif block.get("type") == "tool_use":
                                    write_event("tool_use", {
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                        "input": block.get("input", {}),
                                    })
                                elif block.get("type") == "thinking":
                                    write_event("thinking", {
                                        "thinking": block.get("thinking", ""),
                                    })
                elif ev.text:
                    write_event("text", {"text": ev.text})

            elif ev.type == SDKMessageType.TOOL_RESULT:
                write_event("tool_result", {
                    "tool_use_id": ev.tool_use_id,
                    "content": ev.result_content[:5000] if ev.result_content else "",
                    "is_error": ev.is_error,
                })

            elif ev.type == SDKMessageType.RESULT:
                elapsed_ms = int((time.monotonic() - start_ms) * 1000)
                usage = ev.total_usage
                write_event("result", {
                    "num_turns": ev.num_turns,
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "cost": ev.total_cost,
                    "duration_ms": elapsed_ms,
                })

    except Exception as e:
        write_event("error", {"message": str(e)})

    write_event("done", None)


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for the web chat server."""

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_index()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        elif self.path == "/api/new":
            self._handle_new_session()
        else:
            self.send_error(404)

    def _serve_index(self):
        try:
            html = INDEX_HTML.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        except FileNotFoundError:
            self.send_error(404, "index.html not found")

    def _handle_chat(self):
        # Read body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        prompt = (data.get("message") or "").strip()
        if not prompt:
            self.send_error(400, "Empty message")
            return

        # SSE headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def write_event(event: str, data):
            payload = json.dumps({"event": event, "data": data}, default=str)
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

        # Run the async agent query in the event loop
        try:
            _loop.run_until_complete(run_chat(prompt, write_event))
        except Exception as e:
            write_event("error", {"message": str(e)})
            write_event("done", None)

    def _handle_new_session(self):
        reset_agent()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())


def main():
    global _loop
    _loop = asyncio.new_event_loop()

    server = HTTPServer(("0.0.0.0", PORT), ChatHandler)
    print(f"\n  Open Agent SDK — Web Chat (Python)")
    print(f"  http://localhost:{PORT}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        reset_agent()
        server.server_close()


if __name__ == "__main__":
    main()
