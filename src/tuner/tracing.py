"""OpenTelemetry trace forwarding to Tuner (ENG-1233).

LiveKit Agents already instruments every session with OTel spans. This wires those spans
to Tuner's OTLP endpoint and tags them with the call id, so the Trace tab on the call
details page can show the tree.

Without this, a customer configures it by hand: build a TracerProvider, pick the HTTP
exporter, set the endpoint and auth header, and stamp `tuner.call_id` on the spans. That
works but the failure is silent — traces simply never appear. Everything needed is already
on TunerConfig plus the LiveKit job id, so the SDK can just do it.

Optional: the OTel packages are an extra (`pip install tuner-livekit-sdk[traces]`). If they
are absent this degrades to a no-op with a debug log rather than failing the session.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import TunerConfig

logger = logging.getLogger("tuner.tracing")

# The attribute Tuner correlates on. It only needs to appear once in a trace, but stamping
# every span costs nothing and removes any dependence on which span happens to arrive first.
CALL_ID_ATTRIBUTE = "tuner.call_id"

# Tuner accepts OTLP over HTTP with protobuf encoding. gRPC is not supported.
_TRACES_PATH = "/api/v1/traces"


def _import_otel() -> Any | None:
    """Import the optional OTel pieces, or return None when the extra is not installed."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import get_tracer_provider

        return (OTLPSpanExporter, TracerProvider, BatchSpanProcessor, get_tracer_provider)
    except ImportError:
        return None


def _build_call_id_processor(call_id: str) -> Any:
    """A SpanProcessor that stamps the call id onto every span as it starts.

    Written here rather than reusing LiveKit's `metadata=` hook on `set_tracer_provider`:
    that hook installs a private processor class, and it is only applied when we own the
    provider. Doing it ourselves means the same code path works whether we created the
    provider or attached to the customer's.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    class _CallIdSpanProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            # on_start, not on_end: attributes are frozen once a span ends.
            span.set_attribute(CALL_ID_ATTRIBUTE, call_id)

        def on_end(self, span: Any) -> None:  # pragma: no cover - nothing to do
            return None

        def shutdown(self) -> None:  # pragma: no cover - nothing to own
            return None

        def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
            return True

    return _CallIdSpanProcessor()


def setup_call_tracing(*, config: "TunerConfig", call_id: str) -> bool:
    """Forward this session's OTel spans to Tuner, tagged with the call id.

    Deliberately additive rather than replacing anything. LiveKit's tracer defaults to the
    global OTel provider, and `set_tracer_provider` swaps it outright — so a customer who
    already exports traces to their own backend would silently lose them. Instead: if a
    real provider already exists, attach to it; only create and register one when nothing
    is configured. Either way the customer keeps whatever they had.

    Never raises. Traces are a debugging aid and must not be able to fail a call.

    Args:
        config: The plugin's configuration, for the API key and base URL
        call_id: The id this call is reported to Tuner under, so spans can be matched to it

    Returns:
        True if span forwarding was set up, False if it was skipped
    """
    otel = _import_otel()
    if otel is None:
        logger.debug(
            "Skipping trace forwarding: OpenTelemetry packages are not installed. "
            "Install the extra with: pip install 'tuner-livekit-sdk[traces]'"
        )
        return False

    otlp_span_exporter, tracer_provider_cls, batch_span_processor, get_tracer_provider = otel

    try:
        exporter = otlp_span_exporter(
            endpoint=f"{config.base_url.rstrip('/')}{_TRACES_PATH}",
            headers={"authorization": f"Bearer {config.api_key}"},
        )

        existing = get_tracer_provider()
        if isinstance(existing, tracer_provider_cls):
            # The customer already configured tracing. Add ours alongside theirs.
            provider = existing
            owns_provider = False
        else:
            # Nothing configured — the default is a no-op proxy provider.
            provider = tracer_provider_cls()
            owns_provider = True

        provider.add_span_processor(batch_span_processor(exporter))
        provider.add_span_processor(_build_call_id_processor(call_id))

        if owns_provider:
            from livekit.agents.telemetry import set_tracer_provider

            set_tracer_provider(provider)

        logger.debug(
            "Forwarding OTel spans to Tuner for call %s (%s provider)",
            call_id,
            "new" if owns_provider else "existing",
        )
        return True

    except Exception:
        logger.warning("Could not set up trace forwarding to Tuner", exc_info=True)
        return False
