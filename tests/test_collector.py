"""Tests for tuner.collector"""

import time

from tuner.collector import SessionState


def test_initial_state():
    state = SessionState(start_timestamp=1000.0)
    assert state.is_sip is False
    assert state.caller_phone_number is None
    assert state.close_error is None
    assert state.end_timestamp is None
    assert state.shutdown_reason == ""


def test_call_status_completed_by_default():
    state = SessionState()
    assert state.call_status == "completed"


def test_call_status_error_on_close_error():
    state = SessionState()
    state.record_close(Exception("LLM crashed"))
    assert state.call_status == "error"
    assert state.close_error is not None


def test_record_sip_participant():
    state = SessionState()
    state.record_sip_participant("+14155552671")
    assert state.is_sip is True
    assert state.caller_phone_number == "+14155552671"


def test_record_sip_no_phone():
    state = SessionState()
    state.record_sip_participant(None)
    assert state.is_sip is True
    assert state.caller_phone_number is None


def test_finalize_sets_end_timestamp():
    state = SessionState(start_timestamp=1000.0)
    before = time.time()
    state.finalize("user_disconnected")
    after = time.time()

    assert state.end_timestamp is not None
    assert before <= state.end_timestamp <= after
    assert state.shutdown_reason == "user_disconnected"


def test_duration_ms():
    state = SessionState(start_timestamp=1000.0)
    state.end_timestamp = 1001.5
    assert state.duration_ms == 1500


def test_duration_ms_before_finalize():
    state = SessionState()
    assert state.duration_ms == 0


def test_record_metrics_aggregates():
    state = SessionState()
    # Create a minimal mock metrics object that UsageCollector accepts
    from livekit.agents.metrics import LLMMetrics

    metrics = LLMMetrics(
        label="llm",
        request_id="req1",
        timestamp=time.time(),
        ttft=0.1,
        duration=0.5,
        cancelled=False,
        completion_tokens=10,
        prompt_tokens=50,
        prompt_cached_tokens=0,
        total_tokens=60,
        tokens_per_second=20.0,
    )
    state.record_metrics(metrics)
    summary = state.get_usage_summary()
    assert summary.llm_completion_tokens == 10
    assert summary.llm_prompt_tokens == 50


def test_get_usage_summary_empty():
    state = SessionState()
    summary = state.get_usage_summary()
    assert summary.llm_prompt_tokens == 0
    assert summary.llm_completion_tokens == 0
    assert summary.tts_characters_count == 0
    assert summary.stt_audio_duration == 0.0
