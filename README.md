# Open Agent SDK (Python)

[![PyPI version](https://img.shields.io/pypi/v/open-agent-sdk)](https://pypi.org/project/open-agent-sdk/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-brightgreen)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

Open-source Agent SDK that runs the full agent loop **in-process** — no subprocess or CLI required. Deploy anywhere: cloud, serverless, Docker, CI/CD.

Also available in **TypeScript**: [open-agent-sdk-typescript](https://github.com/codeany-ai/open-agent-sdk-typescript) · **Go**: [open-agent-sdk-go](https://github.com/codeany-ai/open-agent-sdk-go)

## Get started

```bash
pip install open-agent-sdk
```

Set your API key:

```bash
export CODEANY_API_KEY=your-api-key
```

Third-party providers (e.g. OpenRouter) are supported via `CODEANY_BASE_URL`:

```bash
export CODEANY_BASE_URL=https://openrouter.ai/api
export CODEANY_API_KEY=sk-or-...
export CODEANY_MODEL=anthropic/claude-sonnet-4
```

## Quick start

### One-shot query (streaming)

```python
import asyncio
from open_agent_sdk import query, AgentOptions, SDKMessageType

async def main():
    async for message in query(
        prompt="Read pyproject.toml and tell me the project name.",
        options=AgentOptions(
            allowed_tools=["Read", "Glob"],
            permission_mode="bypassPermissions",
        ),
    ):
        if message.type == SDKMessageType.ASSISTANT:
            print(message.text)

asyncio.run(main())
```

### Simple blocking prompt

```python
import asyncio
from open_agent_sdk import create_agent

async def main():
    agent = create_agent(AgentOptions(model="claude-sonnet-4-6"))
    result = await agent.prompt("What files are in this project?")

    print(result.text)
    print(f"Turns: {result.num_turns}, Tokens: {result.usage.input_tokens + result.usage.output_tokens}")
    await agent.close()

asyncio.run(main())
```

### Multi-turn conversation

```python
import asyncio
from open_agent_sdk import create_agent, AgentOptions

async def main():
    agent = create_agent(AgentOptions(max_turns=5))

    r1 = await agent.prompt('Create a file /tmp/hello.txt with "Hello World"')
    print(r1.text)

    r2 = await agent.prompt("Read back the file you just created")
    print(r2.text)

    print(f"Session messages: {len(agent.get_messages())}")
    await agent.close()

asyncio.run(main())
```

### Custom tools (Pydantic schema)

```python
import asyncio
from pydantic import BaseModel
from open_agent_sdk import query, create_sdk_mcp_server, AgentOptions, SDKMessageType
from open_agent_sdk.tool_helper import tool, CallToolResult

class CityInput(BaseModel):
    city: str

async def get_weather_handler(input: CityInput, ctx):
    return CallToolResult(
        content=[{"type": "text", "text": f"{input.city}: 22°C, sunny"}]
    )

get_weather = tool("get_weather", "Get the temperature for a city", CityInput, get_weather_handler)
server = create_sdk_mcp_server("weather", tools=[get_weather])

async def main():
    async for msg in query(
        prompt="What is the weather in Tokyo?",
        options=AgentOptions(mcp_servers={"weather": server}),
    ):
        if msg.type == SDKMessageType.RESULT:
            print(f"Done: ${msg.total_cost:.4f}")

asyncio.run(main())
```

### Custom tools (low-level)

```python
import asyncio
from open_agent_sdk import create_agent, get_all_base_tools, AgentOptions
from open_agent_sdk.tool_helper import define_tool
from open_agent_sdk.types import ToolResult, ToolContext

async def calc_handler(input: dict, ctx: ToolContext) -> ToolResult:
    result = eval(input["expression"], {"__builtins__": {}})
    return ToolResult(tool_use_id="", content=f"{input['expression']} = {result}")

calculator = define_tool(
    name="Calculator",
    description="Evaluate a math expression",
    input_schema={
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    handler=calc_handler,
    read_only=True,
)

async def main():
    agent = create_agent(AgentOptions(tools=[*get_all_base_tools(), calculator]))
    r = await agent.prompt("Calculate 2**10 * 3")
    print(r.text)
    await agent.close()

asyncio.run(main())
```

### MCP server integration

```python
import asyncio
from open_agent_sdk import create_agent, AgentOptions, McpStdioConfig

async def main():
    agent = create_agent(AgentOptions(
        mcp_servers={
            "filesystem": McpStdioConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            ),
        },
    ))

    result = await agent.prompt("List files in /tmp")
    print(result.text)
    await agent.close()

asyncio.run(main())
```

### Subagents

```python
import asyncio
from open_agent_sdk import query, AgentOptions, AgentDefinition, SDKMessageType

async def main():
    async for msg in query(
        prompt="Use the code-reviewer agent to review src/",
        options=AgentOptions(
            agents={
                "code-reviewer": AgentDefinition(
                    description="Expert code reviewer",
                    prompt="Analyze code quality. Focus on security and performance.",
                    tools=["Read", "Glob", "Grep"],
                ),
            },
        ),
    ):
        if msg.type == SDKMessageType.RESULT:
            print("Done")

asyncio.run(main())
```

### Permissions

```python
import asyncio
from open_agent_sdk import query, AgentOptions, SDKMessageType

async def main():
    # Read-only agent — can only analyze, not modify
    async for msg in query(
        prompt="Review the code in src/ for best practices.",
        options=AgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="dontAsk",
        ),
    ):
        pass

asyncio.run(main())
```

### Web UI

A built-in web chat interface is included for testing:

```bash
python examples/web/server.py
# Open http://localhost:8083
```

## API reference

### Top-level functions

| Function                                  | Description                                          |
| ----------------------------------------- | ---------------------------------------------------- |
| `query(prompt, options)`                  | One-shot streaming query, returns `AsyncGenerator`   |
| `create_agent(options)`                   | Create a reusable agent with session persistence     |
| `tool(name, desc, model, handler)`        | Create a tool with Pydantic schema validation        |
| `create_sdk_mcp_server(name, tools)`      | Bundle tools into an in-process MCP server           |
| `define_tool(name, ...)`                  | Low-level tool definition helper                     |
| `get_all_base_tools()`                    | Get all 34 built-in tools                            |
| `list_sessions()`                         | List persisted sessions                              |
| `get_session_messages(id)`                | Retrieve messages from a session                     |
| `fork_session(id)`                        | Fork a session for branching                         |

### Agent methods

| Method                                   | Description                                         |
| ---------------------------------------- | --------------------------------------------------- |
| `await agent.query(prompt)`              | Streaming query, returns `AsyncGenerator[SDKMessage]`|
| `await agent.prompt(text)`               | Blocking query, returns `QueryResult`               |
| `agent.get_messages()`                   | Get conversation history                            |
| `agent.clear()`                          | Reset session                                       |
| `await agent.interrupt()`                | Abort current query                                 |
| `await agent.set_model(model)`           | Change model mid-session                            |
| `await agent.set_permission_mode(mode)`  | Change permission mode                              |
| `await agent.close()`                    | Close MCP connections, persist session               |

### Options (`AgentOptions`)

| Option               | Type                                | Default                       | Description                                                             |
| -------------------- | ----------------------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| `model`              | `str`                               | `claude-sonnet-4-5-20250514`  | LLM model ID                                                           |
| `api_key`            | `str`                               | `CODEANY_API_KEY`             | API key                                                                 |
| `base_url`           | `str`                               | —                             | Custom API endpoint                                                     |
| `cwd`                | `str`                               | `os.getcwd()`                 | Working directory                                                       |
| `system_prompt`      | `str`                               | —                             | System prompt override                                                  |
| `append_system_prompt` | `str`                             | —                             | Append to default system prompt                                         |
| `tools`              | `list[BaseTool]`                    | All built-in                  | Available tools                                                         |
| `allowed_tools`      | `list[str]`                         | —                             | Tool allow-list                                                         |
| `disallowed_tools`   | `list[str]`                         | —                             | Tool deny-list                                                          |
| `permission_mode`    | `PermissionMode`                    | `bypassPermissions`           | `default` / `acceptEdits` / `dontAsk` / `bypassPermissions` / `plan`    |
| `can_use_tool`       | `CanUseToolFn`                      | —                             | Custom permission callback                                              |
| `max_turns`          | `int`                               | `10`                          | Max agentic turns                                                       |
| `max_budget_usd`     | `float`                             | —                             | Spending cap                                                            |
| `max_tokens`         | `int`                               | `16000`                       | Max output tokens                                                       |
| `thinking`           | `ThinkingConfig`                    | —                             | Extended thinking                                                       |
| `mcp_servers`        | `dict[str, McpServerConfig]`        | —                             | MCP server connections                                                  |
| `agents`             | `dict[str, AgentDefinition]`        | —                             | Subagent definitions                                                    |
| `hooks`              | `dict[str, list[dict]]`             | —                             | Lifecycle hooks                                                         |
| `resume`             | `str`                               | —                             | Resume session by ID                                                    |
| `continue_session`   | `bool`                              | `False`                       | Continue most recent session                                            |
| `persist_session`    | `bool`                              | `False`                       | Persist session to disk                                                 |
| `session_id`         | `str`                               | auto                          | Explicit session ID                                                     |
| `json_schema`        | `dict`                              | —                             | Structured output                                                       |
| `sandbox`            | `bool`                              | `False`                       | Filesystem/network sandbox                                              |
| `env`                | `dict[str, str]`                    | —                             | Environment variables                                                   |
| `debug`              | `bool`                              | `False`                       | Enable debug output                                                     |

### Environment variables

| Variable             | Description            |
| -------------------- | ---------------------- |
| `CODEANY_API_KEY`    | API key (required)     |
| `CODEANY_MODEL`      | Default model override |
| `CODEANY_BASE_URL`   | Custom API endpoint    |

## Built-in tools

| Tool                                       | Description                                  |
| ------------------------------------------ | -------------------------------------------- |
| **Bash**                                   | Execute shell commands                       |
| **Read**                                   | Read files with line numbers                 |
| **Write**                                  | Create / overwrite files                     |
| **Edit**                                   | Precise string replacement in files          |
| **Glob**                                   | Find files by pattern                        |
| **Grep**                                   | Search file contents with regex              |
| **WebFetch**                               | Fetch and parse web content                  |
| **WebSearch**                              | Search the web                               |
| **NotebookEdit**                           | Edit Jupyter notebook cells                  |
| **Agent**                                  | Spawn subagents for parallel work            |
| **TaskCreate/List/Update/Get/Stop/Output** | Task management system                       |
| **TeamCreate/Delete**                      | Multi-agent team coordination                |
| **SendMessage**                            | Inter-agent messaging                        |
| **EnterWorktree/ExitWorktree**             | Git worktree isolation                       |
| **EnterPlanMode/ExitPlanMode**             | Structured planning workflow                 |
| **AskUserQuestion**                        | Ask the user for input                       |
| **ToolSearch**                             | Discover lazy-loaded tools                   |
| **ListMcpResources/ReadMcpResource**       | MCP resource access                          |
| **CronCreate/Delete/List**                 | Scheduled task management                    |
| **RemoteTrigger**                          | Remote agent triggers                        |
| **LSP**                                    | Language Server Protocol (code intelligence) |
| **Config**                                 | Dynamic configuration                        |
| **TodoWrite**                              | Session todo list                            |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Your Application                    │
│                                                       │
│   from open_agent_sdk import create_agent              │
└────────────────────────┬─────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │       Agent         │  Session state, tool pool,
              │ query() / prompt()  │  MCP connections
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    QueryEngine      │  Agentic loop:
              │  submit_message()   │  API call → tools → repeat
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
   │  LLM API  │  │  34 Tools │  │    MCP     │
   │  Client   │  │ Bash,Read │  │  Servers   │
   │(streaming)│  │ Edit,...  │  │ stdio/SSE/ │
   └───────────┘  └───────────┘  │ HTTP/SDK   │
                                  └───────────┘
```

**Key internals:**

| Component             | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| **QueryEngine**       | Core agentic loop with auto-compact, retry, tool orchestration   |
| **Auto-compact**      | Summarizes conversation when context window fills up             |
| **Micro-compact**     | Truncates oversized tool results                                 |
| **Retry**             | Exponential backoff for rate limits and transient errors         |
| **Token estimation**  | Rough token counting for budget and compaction thresholds        |
| **File cache**        | LRU cache for file reads                                         |
| **Hook system**       | 20 lifecycle events (PreToolUse, PostToolUse, SessionStart, ...) |
| **Session storage**   | Persist / resume / fork sessions on disk                         |
| **Context injection** | Git status + AGENT.md automatically injected into system prompt  |

## Examples

| #   | File                                    | Description                            |
| --- | --------------------------------------- | -------------------------------------- |
| 01  | `examples/01_simple_query.py`           | Streaming query with event handling    |
| 02  | `examples/02_multi_tool.py`             | Multi-tool orchestration (Glob + Bash) |
| 03  | `examples/03_multi_turn.py`             | Multi-turn session persistence         |
| 04  | `examples/04_custom_tools.py`           | Custom tools with Pydantic + `tool()`  |
| 05  | `examples/05_permissions.py`            | Permission-based access control        |
| 06  | `examples/06_session_persistence.py`    | Session save / resume / fork           |
| web | `examples/web/`                         | Web chat UI for testing                |

Run any example:

```bash
python examples/01_simple_query.py
```

Start the web UI:

```bash
python examples/web/server.py
# Open http://localhost:8083
```

## Project structure

```
open-agent-sdk-python/
├── src/open_agent_sdk/
│   ├── __init__.py         # Public exports
│   ├── agent.py            # Agent high-level API
│   ├── engine.py           # QueryEngine agentic loop
│   ├── types.py            # Core type definitions
│   ├── session.py          # Session persistence
│   ├── hooks.py            # Hook system (20 lifecycle events)
│   ├── tool_helper.py      # Pydantic-based tool creation
│   ├── sdk_mcp_server.py   # In-process MCP server factory
│   ├── mcp/
│   │   └── client.py       # MCP client (stdio/SSE/HTTP)
│   ├── tools/              # 34 built-in tools
│   │   ├── bash.py, read.py, write.py, edit.py
│   │   ├── glob_tool.py, grep.py, web_fetch.py, web_search.py
│   │   ├── agent_tool.py, send_message.py, task_tools.py
│   │   ├── team_tools.py, worktree_tools.py, plan_tools.py
│   │   ├── cron_tools.py, lsp_tool.py, config_tool.py
│   │   └── ...
│   └── utils/
│       ├── messages.py     # Message creation & normalization
│       ├── tokens.py       # Token estimation & cost calculation
│       ├── compact.py      # Auto-compaction logic
│       ├── retry.py        # Exponential backoff retry
│       ├── context.py      # Git & project context injection
│       └── file_cache.py   # LRU file state cache
├── tests/                  # 220 tests
├── examples/               # 6 examples + web UI
└── pyproject.toml
```

## Links

- Website: [codeany.ai](https://codeany.ai)
- TypeScript SDK: [github.com/codeany-ai/open-agent-sdk-typescript](https://github.com/codeany-ai/open-agent-sdk-typescript)
- Go SDK: [github.com/codeany-ai/open-agent-sdk-go](https://github.com/codeany-ai/open-agent-sdk-go)
- Issues: [github.com/codeany-ai/open-agent-sdk-python/issues](https://github.com/codeany-ai/open-agent-sdk-python/issues)

## License

MIT
