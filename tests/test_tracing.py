"""Tests for tuner.tracing (ENG-1233).

The behaviour worth pinning is what happens around a customer who already has tracing, and
what happens when the optional OTel extra is not installed. Both are silent failure modes
if we get them wrong.
"""

from unittest.mock import MagicMock, patch

from tuner.config import TunerConfig
from tuner.tracing import CALL_ID_ATTRIBUTE, setup_call_tracing


def _config(**overrides) -> TunerConfig:
    defaults = {
        "api_key": "tr_api_test",
        "workspace_id": 42,
        "agent_id": "agent-1",
        "base_url": "https://api.usetuner.ai",
    }
    defaults.update(overrides)
    return TunerConfig(**defaults)


def test_returns_false_when_otel_is_not_installed():
    """The extra is optional, so its absence must be a no-op and not an error."""
    with patch("tuner.tracing._import_otel", return_value=None):
        assert setup_call_tracing(config=_config(), call_id="call-1") is False


def test_attaches_to_an_existing_provider_without_replacing_it():
    """A customer who already exports traces must keep doing so.

    LiveKit's `set_tracer_provider` swaps the provider outright, so calling it here would
    silently drop the customer's own exporter.
    """
    from opentelemetry.sdk.trace import TracerProvider

    existing = TracerProvider()
    before = len(existing._active_span_processor._span_processors)

    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=existing),
        patch("livekit.agents.telemetry.set_tracer_provider") as set_provider,
        # Stubbed so the suite never POSTs to the real Tuner API.
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            MagicMock(),
        ),
    ):
        assert setup_call_tracing(config=_config(), call_id="call-1") is True

    # Ours were added to theirs...
    after = len(existing._active_span_processor._span_processors)
    assert after == before + 2, "should add the exporter and the call-id stamper"
    # ...and LiveKit was never told to swap providers.
    set_provider.assert_not_called()


def test_creates_and_registers_a_provider_when_none_exists():
    """With nothing configured, the default is a no-op proxy, so we must supply one."""
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=MagicMock()),
        patch("livekit.agents.telemetry.set_tracer_provider") as set_provider,
    ):
        assert setup_call_tracing(config=_config(), call_id="call-1") is True

    set_provider.assert_called_once()


def test_the_call_id_is_stamped_on_every_span():
    """Tuner needs it on only one span, but stamping all of them removes the ordering
    dependency entirely — no reliance on which span happens to be exported first."""
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=provider),
        patch("livekit.agents.telemetry.set_tracer_provider"),
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            MagicMock(),
        ),
    ):
        setup_call_tracing(config=_config(), call_id="call-abc-123")

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("agent_turn") as span:
        assert span.attributes[CALL_ID_ATTRIBUTE] == "call-abc-123"

    # A second, unrelated span gets it too.
    with tracer.start_as_current_span("llm") as span:
        assert span.attributes[CALL_ID_ATTRIBUTE] == "call-abc-123"


def test_the_endpoint_and_auth_header_are_built_from_config():
    with patch("tuner.tracing._import_otel") as import_otel:
        exporter_cls = MagicMock()
        import_otel.return_value = (
            exporter_cls,
            MagicMock,
            MagicMock(),
            MagicMock(return_value=MagicMock()),
        )
        setup_call_tracing(
            config=_config(base_url="https://staging.usetuner.ai/"), call_id="call-1"
        )

    kwargs = exporter_cls.call_args.kwargs
    # Trailing slash on base_url must not produce a double slash.
    assert kwargs["endpoint"] == "https://staging.usetuner.ai/api/v1/traces"
    assert kwargs["headers"] == {"authorization": "Bearer tr_api_test"}


def test_a_failure_is_swallowed_rather_than_breaking_the_call():
    """Traces are a debugging aid; they must never take a session down."""
    with patch("tuner.tracing._import_otel", side_effect=None) as import_otel:
        import_otel.return_value = (
            MagicMock(side_effect=RuntimeError("exporter blew up")),
            MagicMock,
            MagicMock(),
            MagicMock(),
        )
        assert setup_call_tracing(config=_config(), call_id="call-1") is False


def test_config_defaults_traces_on():
    assert _config().forward_traces is True


def test_traces_can_be_turned_off():
    assert _config(forward_traces=False).forward_traces is False


# --- plugin wiring -------------------------------------------------------------------


def _plugin_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.job.id = "AJ_livekit_job_9f2c"
    ctx.room.remote_participants = {}
    return ctx


def test_the_plugin_sets_up_tracing_with_the_livekit_job_id():
    """The job id is what the SDK reports to Tuner as the call id, so spans must carry it.

    Also pins the timing: this happens in the constructor, before any span is emitted.
    Setting it up later would miss the spans from the start of the call.
    """
    from tuner.plugin import TunerPlugin

    ctx = _plugin_ctx()
    with patch("tuner.plugin.setup_call_tracing") as setup:
        TunerPlugin(
            session=MagicMock(),
            ctx=ctx,
            api_key="tr_api_test",
            workspace_id=42,
            agent_id="agent-1",
        )

    setup.assert_called_once()
    assert setup.call_args.kwargs["call_id"] == str(ctx.job.id)


def test_the_plugin_skips_tracing_when_it_is_turned_off():
    from tuner.plugin import TunerPlugin

    with patch("tuner.plugin.setup_call_tracing") as setup:
        TunerPlugin(
            session=MagicMock(),
            ctx=_plugin_ctx(),
            api_key="tr_api_test",
            workspace_id=42,
            agent_id="agent-1",
            forward_traces=False,
        )

    setup.assert_not_called()


def test_a_disabled_plugin_does_not_set_up_tracing():
    """`enabled=False` returns before config exists, so tracing must not be attempted."""
    from tuner.plugin import TunerPlugin

    with patch("tuner.plugin.setup_call_tracing") as setup:
        TunerPlugin(
            session=MagicMock(),
            ctx=_plugin_ctx(),
            api_key="tr_api_test",
            workspace_id=42,
            agent_id="agent-1",
            enabled=False,
        )

    setup.assert_not_called()
