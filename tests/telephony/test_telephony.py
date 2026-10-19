import logging

import pytest

from tuner.telephony import Forward, SipInfo, extract_sip, forward_to_sip, register_provider
from tuner.telephony.providers import registry

LIVEKIT = "sip:agent@xxxx.sip.livekit.cloud"


class FakeProvider:
    """A header-style carrier: reads ``{"token": ...}``, passes headers, not URI params."""

    name = "fake"

    def extract(self, payload):
        return SipInfo(sip_headers={"X-Correlation-Id": payload.get("token") or ""})

    def forward_to_sip(self, info, sip_uri):
        return Forward(sip_uri=sip_uri, correlation_id=info.correlation_id or None)


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr(registry, "_PROVIDERS", dict(registry._PROVIDERS))
    register_provider(FakeProvider())


def forward(payload):
    return forward_to_sip(extract_sip(payload, provider="fake"), sip_uri=LIVEKIT)


def test_custom_provider_end_to_end():
    info = extract_sip({"token": "17f03659db514827908c1eacaf8b7274"}, provider="fake")
    assert info.provider == "fake"
    fwd = forward_to_sip(info, sip_uri=LIVEKIT)
    assert fwd.correlation_id == "17f03659db514827908c1eacaf8b7274"
    assert fwd.headers == {"X-Correlation-Id": "17f03659db514827908c1eacaf8b7274"}
    assert fwd.custom_headers == [{"name": "X-Correlation-Id", "value": "17f03659db514827908c1eacaf8b7274"}]
    assert fwd.transfer == {"to": LIVEKIT, "custom_headers": fwd.custom_headers}


def test_no_token_leaves_the_leg_untouched():
    fwd = forward({})
    assert fwd == Forward(sip_uri=LIVEKIT)
    assert fwd.headers == {}
    assert fwd.custom_headers == []
    assert fwd.transfer == {"to": LIVEKIT}


def test_twiml_escapes_the_uri():
    fwd = Forward(sip_uri="sip:a@b?X-A=1&X-B=2")
    assert fwd.twiml == "<Response><Dial><Sip>sip:a@b?X-A=1&amp;X-B=2</Sip></Dial></Response>"


@pytest.mark.parametrize("token", ["has space", "a" * 129, "x&X-Other=1"])
def test_malformed_token_is_dropped_and_never_logged(token, caplog):
    caplog.set_level(logging.DEBUG)
    assert forward({"token": token}).correlation_id is None
    assert token not in caplog.text
    assert "malformed" in caplog.text


def test_valid_token_is_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    forward({"token": "17f03659db514827908c1eacaf8b7274"})
    assert "17f03659db514827908c1eacaf8b7274" not in caplog.text
    assert "forwarded" in caplog.text


def test_provider_names_are_case_insensitive():
    assert extract_sip({}, provider="FAKE").provider == "fake"


def test_provider_object_and_override():
    info = extract_sip({"token": "17f03659db514827908c1eacaf8b7274"}, provider=FakeProvider())
    assert forward_to_sip(SipInfo(), sip_uri=LIVEKIT, provider="twilio").sip_uri == LIVEKIT
    assert info.correlation_id == "17f03659db514827908c1eacaf8b7274"


def test_forward_needs_a_provider():
    with pytest.raises(ValueError, match="provider"):
        forward_to_sip(SipInfo(), sip_uri=LIVEKIT)


def test_unknown_provider_lists_available_ones():
    with pytest.raises(ValueError, match="Unknown provider 'nope'. Available: .*fake"):
        extract_sip({}, provider="nope")
