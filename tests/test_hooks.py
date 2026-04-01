"""Tests for hook system."""

import pytest
from open_agent_sdk.hooks import (
    HookEvent,
    HookInput,
    HookOutput,
    HookDefinition,
    HookRegistry,
    create_hook_registry,
    HOOK_EVENTS,
)


class TestHookEvents:
    def test_all_events_defined(self):
        assert len(HOOK_EVENTS) >= 17
        assert HookEvent.PRE_TOOL_USE in HOOK_EVENTS
        assert HookEvent.POST_TOOL_USE in HOOK_EVENTS
        assert HookEvent.SESSION_START in HOOK_EVENTS
        assert HookEvent.STOP in HOOK_EVENTS


class TestHookRegistry:
    def test_create(self):
        registry = create_hook_registry()
        assert isinstance(registry, HookRegistry)

    @pytest.mark.asyncio
    async def test_register_and_execute_handler(self):
        registry = create_hook_registry()

        async def my_hook(input: HookInput) -> HookOutput:
            return HookOutput(message=f"Handled {input.tool_name}")

        registry.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(handler=my_hook),
        )

        results = await registry.execute(
            HookEvent.PRE_TOOL_USE,
            HookInput(event=HookEvent.PRE_TOOL_USE, tool_name="Bash"),
        )
        assert len(results) == 1
        assert results[0].message == "Handled Bash"

    @pytest.mark.asyncio
    async def test_matcher_filtering(self):
        registry = create_hook_registry()
        calls = []

        async def my_hook(input: HookInput) -> HookOutput:
            calls.append(input.tool_name)
            return HookOutput()

        registry.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(handler=my_hook, matcher="Bash"),
        )

        # Should match
        await registry.execute(
            HookEvent.PRE_TOOL_USE,
            HookInput(event=HookEvent.PRE_TOOL_USE, tool_name="Bash"),
        )
        assert len(calls) == 1

        # Should not match
        await registry.execute(
            HookEvent.PRE_TOOL_USE,
            HookInput(event=HookEvent.PRE_TOOL_USE, tool_name="Read"),
        )
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_no_hooks_returns_empty(self):
        registry = create_hook_registry()
        results = await registry.execute(
            HookEvent.POST_TOOL_USE,
            HookInput(event=HookEvent.POST_TOOL_USE),
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_register_from_config(self):
        registry = create_hook_registry()
        config = {
            "PreToolUse": [{"command": "echo ok", "matcher": ".*"}],
        }
        registry.register_from_config(config)
        # Just verify no errors during registration

    @pytest.mark.asyncio
    async def test_blocking_hook(self):
        registry = create_hook_registry()

        async def blocking_hook(input: HookInput) -> HookOutput:
            return HookOutput(block=True, message="Blocked!")

        registry.register(HookEvent.PRE_TOOL_USE, HookDefinition(handler=blocking_hook))

        results = await registry.execute(
            HookEvent.PRE_TOOL_USE,
            HookInput(event=HookEvent.PRE_TOOL_USE, tool_name="Write"),
        )
        assert results[0].block is True

    def test_clear(self):
        registry = create_hook_registry()

        async def noop(input: HookInput) -> HookOutput:
            return HookOutput()

        registry.register(HookEvent.STOP, HookDefinition(handler=noop))
        registry.clear()
        # No way to directly check, but clear should not raise
