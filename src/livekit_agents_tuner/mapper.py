from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from livekit.agents import AgentSession

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
    """
    from livekit.agents.llm.chat_context import (
        ChatMessage,
        FunctionCall,
        FunctionCallOutput,
    )

    logger.debug("map_history_to_segments called with %d items", len(items))
    for idx, item in enumerate(items):
        logger.debug("  Item %d: %s - %s", idx, type(item).__name__, item)

    segments: list[dict] = []

    for i, item in enumerate(items):
        role = None
        if isinstance(item, ChatMessage):
            if item.role not in ("user", "assistant"):
                continue  # Skip system / developer instruction messages

            role = "user" if item.role == "user" else "agent"
            text = item.text_content or ""

            started = item.metrics.get("started_speaking_at", session_start_ts)
            stopped = item.metrics.get("stopped_speaking_at", started)
            seg: dict = {
                "role": role,
                "text": text,
                "start_ms": max(0, int((started - session_start_ts) * 1000)),
                "end_ms": max(0, int((stopped - session_start_ts) * 1000)),
                "metadata": {
                    "id": item.id,
                    "interrupted": item.interrupted,
                    "llm_node_ttft": item.metrics.get("llm_node_ttft"),
                    "tts_node_ttfb": item.metrics.get("tts_node_ttfb"),
                    "stt_node_ttfb": item.metrics.get("transcription_delay"),
                    "e2e_latency": item.metrics.get("e2e_latency"),
                    "transcript_confidence": item.transcript_confidence,
                },
            }
            segments.append(seg)

        elif isinstance(item, (FunctionCall, FunctionCallOutput)):
            tool_payload: dict[str, Any] = {
                "name": item.name,
                "request_id": item.call_id,
            }

            if isinstance(item, FunctionCall):
                role = "agent_function"
                try:
                    params = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    params = item.arguments  # Pass raw string if not valid JSON
                tool_payload["params"] = params
            elif isinstance(item, FunctionCallOutput):
                role = "agent_result"
                tool_payload["is_error"] = item.is_error
                if item.is_error:
                    tool_payload["error"] = item.output
                else:
                    tool_payload["result"] = {"value": item.output}

            segments.append(
                {
                    "role": role,
                    "start_ms": max(0, int((item.created_at - session_start_ts) * 1000)),
                    "tool": tool_payload,
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

def _model_name(component: Any) -> str | None:
    # Handles None or components without a .model attribute
    return getattr(component, "model", None)

def to_create_call_request(
    session: "AgentSession",
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
        "llm_token": usage.llm_prompt_tokens + usage.llm_completion_tokens + usage.llm_prompt_cached_tokens,
        "tts_characters_count": usage.tts_characters_count,
        "stt_duration_seconds": usage.stt_audio_duration,
    }

    ai_models = {
        "stt_model": _model_name(getattr(session, "stt", None)),
        "llm_model": _model_name(getattr(session, "llm", None)),
        "tts_model": _model_name(getattr(session, "tts", None)),
    }

    general_meta: dict = {
        "livekit_job_id": str(ctx.job.id),
        "livekit_room_name": ctx.room.name,
        "usage_token": usage_dict,
        "ai_models":ai_models,
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
            payload["call_cost"] = config.cost_calculator(usage)
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
