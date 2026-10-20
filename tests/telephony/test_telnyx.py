import pytest

from tuner.telephony import extract_sip, forward_to_sip

LIVEKIT = "sip:sim-nova-clinic@project.sip.livekit.cloud"
TOKEN = "aed6b562f7ab4d1e96021b17f28d6ee2"


def envelope(custom_headers, **payload):
    return {
        "data": {
            "event_type": "call.initiated",
            "payload": {"call_control_id": "v3:abc", "custom_headers": custom_headers, **payload},
        }
    }


def forward(event):
    return forward_to_sip(extract_sip(event, provider="telnyx"), sip_uri=LIVEKIT)


def test_returns_token_as_custom_headers_and_keeps_the_uri():
    fwd = forward(
        envelope(
            [
                {"name": "X-Correlation-Id", "value": TOKEN},
                {"name": "X-LiveKit-Room", "value": "room-1"},
            ]
        )
    )
    assert fwd.sip_uri == LIVEKIT
    assert fwd.custom_headers == [{"name": "X-Correlation-Id", "value": TOKEN}]
    assert fwd.transfer == {"to": LIVEKIT, "custom_headers": fwd.custom_headers}


def test_accepts_the_bare_payload():
    payload = envelope([{"name": "x-correlation-id", "value": TOKEN}])["data"]["payload"]
    assert forward(payload).correlation_id == TOKEN


def test_extracts_sip_call_id_and_carrier_id():
    info = extract_sip(envelope([], sip_call_id="sip@host"), provider="telnyx")
    assert info.sip_call_id == "sip@host"
    assert info.carrier_call_id == "v3:abc"
    assert not info.is_fallback
    assert extract_sip(envelope([]), provider="telnyx").is_fallback


def test_outgoing_call_reports_recipient():
    info = extract_sip(envelope([], direction="outgoing", to="sip:t@tuner"), provider="telnyx")
    assert info.recipient == "sip:t@tuner"
    assert extract_sip(envelope([], direction="incoming", to="+1"), provider="telnyx").recipient is None


@pytest.mark.parametrize(
    "event",
    [
        envelope([]),
        envelope(None),
        envelope([{"name": "X-Other", "value": "1"}]),
        envelope(["not-a-dict", {"name": "X-Correlation-Id"}]),
        {"data": "nope"},
        None,
    ],
)
def test_production_or_malformed_event_has_no_token(event):
    fwd = forward(event)
    assert fwd.sip_uri == LIVEKIT
    assert fwd.custom_headers == []
