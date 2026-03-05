from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collector import SessionState
    from .config import TimingStrategy, TunerConfig

logger = logging.getLogger("livekit_agents_tuner.mapper")


def _apply_timing(
    seg: dict,
    current: Any,
    next_msg: Any,
    strategy: "TimingStrategy",
) -> None:
    """
    Apply a timing strategy to a user/agent segment in-place.

    Built-in strategies:
      "word_count" — estimate duration_ms = word_count * 250 ms, end_ms = start_ms + duration_ms

    Custom strategy (callable):
      def my_strategy(current: ChatMessage, next_msg: ChatMessage | None) -> tuple[int | None, int | None]:
          # return (end_ms, duration_ms); use None to leave a field unset
          ...
    """
    from .config import MS_PER_WORD, TIMING_WORD_COUNT

    if strategy == TIMING_WORD_COUNT:
        text = seg.get("text", "")
        word_count = len(text.split()) if text.strip() else 0
        if word_count > 0:
            duration_ms = word_count * MS_PER_WORD
            seg["duration_ms"] = duration_ms
            seg["end_ms"] = seg["start_ms"] + duration_ms
        return

    if callable(strategy):
        try:
            end_ms, duration_ms = strategy(current, next_msg)
            if end_ms is not None:
                seg["end_ms"] = end_ms
            if duration_ms is not None:
                seg["duration_ms"] = duration_ms
        except Exception:
            logger.warning(
                "Custom timing_strategy raised an error; timing fields left unset",
                exc_info=True,
            )
        return

    logger.warning("Unknown timing_strategy %r; timing fields left unset", strategy)


def map_history_to_segments(
    items: list[Any],
    timing_strategy: "TimingStrategy" = "none",
) -> list[dict]:
    """
    Map LiveKit ChatContext items to Tuner PublicTranscriptSegment dicts.

    Roles produced:
      - user          → ChatMessage(role="user")
      - agent         → ChatMessage(role="assistant")
      - agent_function → FunctionCall (merged with FunctionCallOutput)
      - agent_result  → FunctionCallOutput with no matching FunctionCall
    """
    from livekit.agents.llm.chat_context import (
        ChatMessage,
        FunctionCall,
        FunctionCallOutput,
    )

    segments: list[dict] = []
    # Track (segment_index, source ChatMessage) for timing post-processing
    timed: list[tuple[int, Any]] = []

    for item in items:
        if isinstance(item, ChatMessage):
            if item.role not in ("user", "assistant"):
                continue  # Skip system / developer instruction messages

            role = "user" if item.role == "user" else "agent"
            text = item.text_content or ""
            start_ms = int(item.created_at * 1000)

            seg: dict = {
                "role": role,
                "text": text,
                "start_ms": start_ms,
                # end_ms and duration_ms are not available from LiveKit's ChatMessage.
                # LiveKit only exposes created_at (when the message was committed),
                # not per-segment speech start/end times.
                # Populate them by passing timing_strategy= to map_history_to_segments.
                # "end_ms": ...,
                # "duration_ms": ...,
                "metadata": {
                    "id": item.id,
                    "interrupted": item.interrupted,
                },
            }
            timed.append((len(segments), item))
            segments.append(seg)

        elif isinstance(item, FunctionCall):
            try:
                params = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                params = item.arguments  # Pass raw string if not valid JSON

            segments.append(
                {
                    "role": "agent_function",
                    "tool": {
                        "name": item.name,
                        "request_id": item.call_id,
                        "params": params,
                        "result": None,
                        "is_error": False,
                        "error": None,
                    },
                }
            )

        elif isinstance(item, FunctionCallOutput):
            # Merge into the nearest matching agent_function segment
            merged = False
            for seg in reversed(segments):
                if (
                    seg.get("role") == "agent_function"
                    and seg["tool"]["request_id"] == item.call_id
                    and seg["tool"]["result"] is None
                ):
                    seg["tool"]["result"] = item.output
                    seg["tool"]["is_error"] = item.is_error
                    if item.is_error:
                        seg["tool"]["error"] = item.output
                    merged = True
                    break

            if not merged:
                # Orphan output — no matching function call found
                segments.append(
                    {
                        "role": "agent_result",
                        "tool": {
                            "name": item.name,
                            "request_id": item.call_id,
                            "result": item.output,
                            "is_error": item.is_error,
                            "error": item.output if item.is_error else None,
                        },
                    }
                )

    # Post-processing: apply timing strategy to user/agent message segments
    for i, (seg_idx, current_msg) in enumerate(timed):
        next_msg = timed[i + 1][1] if i + 1 < len(timed) else None
        _apply_timing(segments[seg_idx], current_msg, next_msg, timing_strategy)

    return segments


def build_plain_transcript(items: list[Any]) -> str:
    """Build a plain-text diarized transcript from conversation history."""
    from livekit.agents.llm.chat_context import ChatMessage

    lines: list[str] = []
    for item in items:
        if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
            speaker = "user" if item.role == "user" else "agent"
            text = item.text_content or ""
            if text:
                lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def to_create_call_request(
    state: "SessionState",
    history_items: list[Any],
    config: "TunerConfig",
    ctx: Any,
) -> dict:
    """
    Build the Tuner CreateCallRequest payload from session state and history.

    Args:
        state:         Finalized SessionState from the collector.
        history_items: List of ChatContext items (session.history.items).
        config:        TunerConfig instance.
        ctx:           LiveKit JobContext.

    Returns:
        Dict ready to POST to the Tuner API.
    """
    from livekit import rtc

    # --- Required fields ---
    call_id = str(ctx.job.id)

    # Determine call_type: explicit config > SIP detection > fallback
    if config.call_type:
        call_type = config.call_type
    elif state.is_sip:
        call_type = "phone_call"
    else:
        # Final fallback: inspect current room participants
        call_type = "web_call"
        try:
            for participant in ctx.room.remote_participants.values():
                if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                    call_type = "phone_call"
                    break
        except Exception:
            logger.debug("Could not inspect room participants for call_type detection")

    segments = map_history_to_segments(history_items, config.timing_strategy)
    plain_transcript = build_plain_transcript(history_items)

    # --- Usage summary for metadata ---
    usage = state.get_usage_summary()
    usage_dict = {
        "llm_prompt_tokens": usage.llm_prompt_tokens,
        "llm_prompt_cached_tokens": usage.llm_prompt_cached_tokens,
        "llm_completion_tokens": usage.llm_completion_tokens,
        "tts_characters_count": usage.tts_characters_count,
        "stt_audio_duration_s": usage.stt_audio_duration,
    }

    general_meta: dict = {
        "livekit_job_id": str(ctx.job.id),
        "livekit_room_name": ctx.room.name,
        "usage_summary": usage_dict,
    }
    if config.extra_metadata:
        general_meta.update(config.extra_metadata)

    end_ts = state.end_timestamp or time.time()

    payload: dict = {
        "call_id": call_id,
        "call_type": call_type,
        "transcript_with_tool_calls": segments,
        "start_timestamp": int(state.start_timestamp),
        "end_timestamp": int(end_ts),
        "duration_ms": state.duration_ms,
        "call_status": state.call_status,
        "general_meta_data_raw": general_meta,
    }

    # Optional fields — omit when empty/None to keep payload clean
    if plain_transcript:
        payload["transcript"] = plain_transcript

    if state.caller_phone_number:
        payload["caller_phone_number"] = state.caller_phone_number

    if state.shutdown_reason:
        payload["disconnection_reason"] = state.shutdown_reason

    if state.close_error is not None:
        payload["call_successful"] = False

    return payload
