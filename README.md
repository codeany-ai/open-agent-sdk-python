# Open Agent SDK (Python)

Open-source Agent SDK for Python. Runs the full agent loop in-process — no local CLI required. Deploy anywhere: cloud, serverless, Docker, CI/CD.

## Installation

```bash
pip install open-agent-sdk
```

With MCP support:
```bash
pip install open-agent-sdk[mcp]
```

## Quick Start

```python
import asyncio
from open_agent_sdk import Agent, AgentOptions

async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        api_key="your-api-key",
    ))

    result = await agent.prompt("What is 2 + 2?")
    print(result.text)
    await agent.close()

asyncio.run(main())
```

## Features

- **34 Built-in Tools**: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch, Agent, MCP, and more
- **Complete Agent Loop**: Multi-turn tool calling with streaming events
- **MCP Support**: Connect to MCP servers via stdio, SSE, or HTTP
- **Session Persistence**: Save/load/fork conversation sessions
- **Context Management**: Auto-compaction, token estimation, cost tracking
- **Permission System**: Configurable tool access control
- **Hook System**: 20 lifecycle events for extensibility
- **Custom Tools**: Define tools with Pydantic models or raw JSON schemas

## License

MIT
