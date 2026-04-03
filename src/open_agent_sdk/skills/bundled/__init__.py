"""Bundled Skills Initialization.

Registers all built-in skills at SDK startup.
"""

from __future__ import annotations

from open_agent_sdk.skills.bundled.simplify import register_simplify_skill
from open_agent_sdk.skills.bundled.commit import register_commit_skill
from open_agent_sdk.skills.bundled.review import register_review_skill
from open_agent_sdk.skills.bundled.debug import register_debug_skill
from open_agent_sdk.skills.bundled.test import register_test_skill

_initialized = False


def init_bundled_skills() -> None:
    """Initialize all bundled skills.

    Safe to call multiple times (idempotent).
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    register_simplify_skill()
    register_commit_skill()
    register_review_skill()
    register_debug_skill()
    register_test_skill()
