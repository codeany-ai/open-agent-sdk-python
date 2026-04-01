"""Tests for file state cache."""

import pytest
from open_agent_sdk.utils.file_cache import FileState, FileStateCache, create_file_state_cache


class TestFileStateCache:
    def test_get_set(self):
        cache = create_file_state_cache()
        state = FileState(path="/tmp/test.py", content="hello", mtime=1000.0, size=5)
        cache.set("/tmp/test.py", state)

        result = cache.get("/tmp/test.py")
        assert result is not None
        assert result.path == "/tmp/test.py"
        assert result.content == "hello"
        assert result.mtime == 1000.0

    def test_get_missing(self):
        cache = create_file_state_cache()
        assert cache.get("/nonexistent") is None

    def test_lru_eviction(self):
        cache = create_file_state_cache(max_size=2)
        cache.set("a", FileState(path="a", content="", mtime=0, size=0))
        cache.set("b", FileState(path="b", content="", mtime=0, size=0))
        cache.set("c", FileState(path="c", content="", mtime=0, size=0))

        # "a" should be evicted
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_lru_access_updates_order(self):
        cache = create_file_state_cache(max_size=2)
        cache.set("a", FileState(path="a", content="", mtime=0, size=0))
        cache.set("b", FileState(path="b", content="", mtime=0, size=0))

        # Access "a" to make it most recently used
        cache.get("a")

        # Add "c", "b" should be evicted
        cache.set("c", FileState(path="c", content="", mtime=0, size=0))
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_delete(self):
        cache = create_file_state_cache()
        cache.set("x", FileState(path="x", content="", mtime=0, size=0))
        cache.delete("x")
        assert cache.get("x") is None

    def test_clear(self):
        cache = create_file_state_cache()
        cache.set("a", FileState(path="a", content="", mtime=0, size=0))
        cache.set("b", FileState(path="b", content="", mtime=0, size=0))
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        cache = create_file_state_cache(max_size=100)
        cache.set("a", FileState(path="a", content="", mtime=0, size=0))
        stats = cache.get_stats()
        assert stats["size"] == 100
        assert stats["entries"] == 1

    def test_update_existing(self):
        cache = create_file_state_cache()
        cache.set("a", FileState(path="a", content="old", mtime=0, size=3))
        cache.set("a", FileState(path="a", content="new", mtime=1, size=3))
        result = cache.get("a")
        assert result.content == "new"
        assert result.mtime == 1
