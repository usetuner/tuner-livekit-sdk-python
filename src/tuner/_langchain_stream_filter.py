"""Filters ToolMessage chunks from LangGraph streams in LiveKit pipelines.

Workaround for a bug in livekit-plugins-langchain where _to_chat_chunk()
matches BaseMessageChunk instead of AIMessageChunk, causing tool results
to reach TTS as raw JSON.

Upstream file:
  livekit-plugins/livekit-plugins-langchain/livekit/plugins/langchain/langgraph.py
  Function: _to_chat_chunk(), line ~240

Remove this file once upstream changes:
  isinstance(msg, BaseMessageChunk) → isinstance(msg, AIMessageChunk)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


class _ToolMessageFilter:
    """Drop-in wrapper that strips ToolMessage/ToolMessageChunk from a stream."""

    __slots__ = ("_wrapped",)

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def astream(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        return _filtered_stream(self._wrapped.astream(input, config, **kwargs))

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> Any:
        return await self._wrapped.ainvoke(input, config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


async def _filtered_stream(aiter: Any) -> AsyncIterator[Any]:
    try:
        from langchain_core.messages import ToolMessage, ToolMessageChunk
        _tool_types: tuple[type, ...] = (ToolMessage, ToolMessageChunk)
    except ImportError:
        _tool_types = ()

    async for item in aiter:
        if not (_tool_types and isinstance(item, tuple)):
            yield item
            continue

        if item and isinstance(item[0], _tool_types):
            continue

        if (
            len(item) == 2
            and item[0] == "messages"
            and isinstance(item[1], tuple)
            and item[1]
            and isinstance(item[1][0], _tool_types)
        ):
            continue

        yield item