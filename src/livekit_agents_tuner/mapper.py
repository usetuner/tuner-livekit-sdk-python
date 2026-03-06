from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collector import SessionState
    from .config import TunerConfig

logger = logging.getLogger("livekit_agents_tuner.mapper")

def map_history_to_segments(
    items: list[Any],
    session_start_ts: float = 0.0,
    session_end_ts: float | None = None,
) -> list[dict]:
    """
    Map LiveKit ChatContext items to Tuner PublicTranscriptSegment dicts.

    Args:
        items:            List of ChatContext items (ChatMessage, FunctionCall, etc.).
        session_start_ts: Session start time (epoch seconds). Used to compute
                          start_ms/end_ms relative to session start.
        session_end_ts:   Session end time (epoch seconds). Used as the end_ms
                          for the last turn. Falls back to the last message's
                          created_at if not provided.

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

    # Log all items passed to the function
    logger.info(f"map_history_to_segments called with {len(items)} items")
    for idx, item in enumerate(items):
        logger.info(f"  Item {idx}: {type(item).__name__} - {item}")

    segments: list[dict] = []

    for i, item in enumerate(items):
            
        if isinstance(item, ChatMessage):
            if item.role not in ("user", "assistant"):
                continue  # Skip system / developer instruction messages

            role = "user" if item.role == "user" else "agent"
            text = item.text_content or ""

            seg: dict = {
                "role": role,
                "text": text,
                "start_ms": max(0, int((item.metrics["started_speaking_at"] - session_start_ts) * 1000)),
                "end_ms": max(0, int((item.metrics["stopped_speaking_at"] - session_start_ts) * 1000)),
                "metadata": {
                    "id": item.id,
                    "interrupted": item.interrupted,
                    "llm_node_ttft": item.metrics.get("llm_node_ttft"),
                    "tts_node_ttfb": item.metrics.get("tts_node_ttfb"),
                    "e2e_latency": item.metrics.get("e2e_latency"),
                },
            }
            segments.append(seg)

        elif isinstance(item, FunctionCall):
            start_ms = max(0, int((item.created_at - session_start_ts) * 1000))
            end_ms = start_ms
            result: Any = None
            is_error = False
            error: Any = None
            if i + 1 < len(items):
                next_item = items[i + 1]
                if isinstance(next_item, FunctionCallOutput):
                    end_ms = max(0, int((next_item.created_at - session_start_ts) * 1000))
                    is_error = bool(next_item.is_error)
                    if is_error:
                        result = next_item.output
                        error = next_item.output
                    else:
                        if isinstance(next_item.output, str):
                            try:
                                result = json.loads(next_item.output)
                            except (json.JSONDecodeError, TypeError):
                                result = (
                                    next_item.output
                                    if len(next_item.output.split()) <= 1
                                    else {"message": next_item.output}
                                )
                        else:
                            result = next_item.output

            try:
                params = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                params = item.arguments  # Pass raw string if not valid JSON

            segments.append(
                {
                    "role": "agent_function",
                    "tool": {
                        "name": item.name,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "request_id": item.call_id,
                        "params": params,
                        "result": result,
                        "is_error": is_error,
                        "error": error,
                    },
                }
            )

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

    end_ts = state.end_timestamp or time.time()

    segments = map_history_to_segments(
        history_items,
        session_start_ts=state.start_timestamp,
        session_end_ts=end_ts,
    )
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
    if config.cost_calculator is not None:
        try:
            payload["call_cost"] = config.cost_calculator()
        except Exception:
            logger.warning("cost_calculator function raised an error", exc_info=True)

    if plain_transcript:
        payload["transcript"] = plain_transcript

    if state.caller_phone_number:
        payload["caller_phone_number"] = state.caller_phone_number

    if state.shutdown_reason:
        payload["disconnection_reason"] = state.shutdown_reason

    if state.close_error is not None:
        payload["call_successful"] = False

    return payload
