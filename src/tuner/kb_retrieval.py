"""KB retrievals recorded by the customer: validation, turn linking and transcript placement.

Pure functions with no I/O. Log messages carry ids only, never chunk text or queries.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("tuner.mapper")

# livekit-agents creates the preemptive user message when the final transcript arrives and
# new_message at end of turn, so they differ by the endpointing delay (default max 3.0 s).
PREEMPTIVE_MATCH_WINDOW_S = 5.0


class InvalidKbRetrieval(ValueError):
    """Raised for unusable record_kb_retrieval input. Messages never include input values."""


@dataclass(frozen=True)
class KbRetrieval:
    user_turn_id: str
    user_text: str | None
    user_created_at: float | None
    recorded_at: float
    chunks: tuple[dict[str, Any], ...]
    query: str | None = None
    chunk_text_omitted: bool = False

    def __repr__(self) -> str:  # keep chunk text and queries out of logs and tracebacks
        return f"KbRetrieval(user_turn_id={self.user_turn_id!r}, chunks={len(self.chunks)})"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize_chunk(index: int, chunk: Any, capture_text: bool) -> dict[str, Any]:
    if isinstance(chunk, str):
        return {"text": chunk} if capture_text else {}
    if not isinstance(chunk, Mapping):
        raise InvalidKbRetrieval(f"chunk {index} must be a str or a dict")
    text = chunk.get("text")
    if not isinstance(text, str):
        raise InvalidKbRetrieval(f"chunk {index} needs a str 'text'")
    out: dict[str, Any] = {"text": text} if capture_text else {}
    source_id = chunk.get("source_id")
    if source_id is not None:
        if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
            raise InvalidKbRetrieval(f"chunk {index} 'source_id' must be a str or int")
        out["source_id"] = source_id
    score = chunk.get("score")
    if score is not None:
        if not _is_number(score):
            raise InvalidKbRetrieval(f"chunk {index} 'score' must be a finite number")
        out["score"] = score
    return out


def build_kb_retrieval(
    new_message: Any,
    chunks: Any,
    query: Any,
    *,
    capture_text: bool,
    recorded_at: float,
) -> KbRetrieval:
    """Copy what the customer passed into a KbRetrieval. Reads ``new_message`` only."""
    turn_id = getattr(new_message, "id", None)
    if not isinstance(turn_id, str) or not turn_id:
        raise InvalidKbRetrieval("new_message has no id")
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise InvalidKbRetrieval("chunks must be a list")
    if query is not None and not isinstance(query, str):
        raise InvalidKbRetrieval("query must be a str")

    user_text = getattr(new_message, "text_content", None)
    created_at: Any = getattr(new_message, "created_at", None)
    return KbRetrieval(
        user_turn_id=turn_id,
        user_text=user_text if isinstance(user_text, str) else None,
        user_created_at=float(created_at) if _is_number(created_at) else None,
        recorded_at=recorded_at,
        chunks=tuple(_normalize_chunk(i, c, capture_text) for i, c in enumerate(chunks)),
        query=query,
        chunk_text_omitted=not capture_text,
    )


def resolve_user_turn_id(
    retrieval: KbRetrieval,
    user_messages: Sequence[Any],
    claimed_ids: frozenset[str] = frozenset(),
) -> str | None:
    """Id of the history user message this retrieval belongs to, or None.

    With preemptive generation, history can keep a copy of the user message with a
    different id but identical text. Fall back to that copy only when exactly one
    unclaimed message has the same text within PREEMPTIVE_MATCH_WINDOW_S.
    """
    if any(m.id == retrieval.user_turn_id for m in user_messages):
        return retrieval.user_turn_id
    if not retrieval.user_text:
        return None
    at = retrieval.user_created_at if retrieval.user_created_at is not None else retrieval.recorded_at
    candidates = [
        m
        for m in user_messages
        if m.text_content == retrieval.user_text
        and m.id not in claimed_ids
        and m.created_at is not None
        and abs(m.created_at - at) <= PREEMPTIVE_MATCH_WINDOW_S
    ]
    return candidates[0].id if len(candidates) == 1 else None


def _to_segment(retrieval: KbRetrieval, user_turn_id: str, session_start_ts: float) -> dict[str, Any]:
    seg: dict[str, Any] = {
        "role": "kb_retrieval",
        "user_turn_id": user_turn_id,
        "start_ms": max(0, int((retrieval.recorded_at - session_start_ts) * 1000)),
    }
    if retrieval.query is not None:
        seg["query"] = retrieval.query
    seg["chunks"] = [dict(c) for c in retrieval.chunks]
    if retrieval.chunk_text_omitted:
        seg["chunk_text_omitted"] = True
    return seg


def insert_kb_retrievals(
    segments: list[dict],
    retrievals: Sequence[KbRetrieval],
    user_messages: Sequence[Any],
    session_start_ts: float,
) -> list[dict]:
    """Place each retrieval right after its user turn, in recorded order.

    The turn link decides placement; timestamps never move a retrieval to another turn.
    Unlinked retrievals go at the end with their original id (the backend flags them).
    """
    history_ids = {m.id for m in user_messages}
    claimed = frozenset(r.user_turn_id for r in retrievals if r.user_turn_id in history_ids)
    by_turn: dict[str, list[dict]] = {}
    unlinked: list[dict] = []
    for r in retrievals:
        turn_id = resolve_user_turn_id(r, user_messages, claimed)
        if turn_id is None:
            logger.warning(
                "kb_retrieval for user turn %s matches no single user message in history "
                "(same text within %.1fs); sending it unlinked",
                r.user_turn_id,
                PREEMPTIVE_MATCH_WINDOW_S,
            )
            unlinked.append(_to_segment(r, r.user_turn_id, session_start_ts))
        else:
            by_turn.setdefault(turn_id, []).append(_to_segment(r, turn_id, session_start_ts))

    out: list[dict] = []
    for seg in segments:
        out.append(seg)
        if seg.get("role") == "user":
            out.extend(by_turn.pop(str(seg.get("id")), []))
    for leftover in by_turn.values():  # user turn in history but not mapped (e.g. skipped)
        out.extend(leftover)
    out.extend(unlinked)
    return out
