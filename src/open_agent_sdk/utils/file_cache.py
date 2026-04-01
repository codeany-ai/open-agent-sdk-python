"""LRU file state cache for tracking file reads."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class FileState:
    path: str
    content: str
    mtime: float
    size: int


class FileStateCache:
    """LRU cache for file state tracking (staleness detection)."""

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._cache: OrderedDict[str, FileState] = OrderedDict()

    def get(self, path: str) -> FileState | None:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        return None

    def set(self, path: str, state: FileState) -> None:
        if path in self._cache:
            self._cache.move_to_end(path)
        self._cache[path] = state
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def delete(self, path: str) -> None:
        self._cache.pop(path, None)

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> dict[str, int]:
        return {"size": self._max_size, "entries": len(self._cache)}


def create_file_state_cache(max_size: int = 1000) -> FileStateCache:
    return FileStateCache(max_size)
