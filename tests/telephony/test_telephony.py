import logging

import pytest

from tuner.telephony import LiveKitLeg, TunerTelephony, register_provider
from tuner.telephony.providers import registry

LIVEKIT = "sip:agent@xxxx.sip.livekit.cloud"


class FakeProvider:
    """A header-style carrier: reads ``{"token": ...}``, passes headers, not URI params."""

    name = "fake"

    def read_correlation_id(self, event):
        return event.get("token")

    def build_leg(self, livekit_uri, correlation_id):
        return LiveKitLeg(sip_uri=livekit_uri, correlation_id=correlation_id)


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr(registry, "_PROVIDERS", dict(registry._PROVIDERS))
    register_provider(FakeProvider())


def test_correlate_returns_token_on_the_leg():
    leg = TunerTelephony(provider="fake").correlate({"token": "abc123"}, livekit_uri=LIVEKIT)
    assert leg.correlation_id == "abc123"
    assert leg.headers == {"X-Correlation-Id": "abc123"}
    assert leg.custom_headers == [{"name": "X-Correlation-Id", "value": "abc123"}]


def test_no_token_leaves_the_leg_untouched():
    leg = TunerTelephony(provider="fake").correlate({}, livekit_uri=LIVEKIT)
    assert leg == LiveKitLeg(sip_uri=LIVEKIT)
    assert leg.headers == {}
    assert leg.custom_headers == []


@pytest.mark.parametrize("token", ["has space", "a" * 129, "x&X-Other=1"])
def test_malformed_token_is_dropped_and_never_logged(token, caplog):
    caplog.set_level(logging.DEBUG)
    leg = TunerTelephony(provider="fake").correlate({"token": token}, livekit_uri=LIVEKIT)
    assert leg.correlation_id is None
    assert token not in caplog.text


def test_valid_token_is_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    TunerTelephony(provider="fake").correlate({"token": "secret123"}, livekit_uri=LIVEKIT)
    assert "secret123" not in caplog.text
    assert "forwarded" in caplog.text


def test_provider_names_are_case_insensitive():
    assert TunerTelephony(provider="FAKE").provider.name == "fake"


def test_unknown_provider_lists_available_ones():
    with pytest.raises(ValueError, match="Unknown provider 'nope'. Available: .*fake"):
        TunerTelephony(provider="nope")
