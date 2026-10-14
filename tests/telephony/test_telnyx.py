import pytest

from tuner.telephony import TunerTelephony

LIVEKIT = "sip:sim-nova-clinic@project.sip.livekit.cloud"
TOKEN = "aed6b562f7ab4d1e96021b17f28d6ee2"


def envelope(custom_headers):
    return {
        "data": {
            "event_type": "call.initiated",
            "payload": {"call_control_id": "v3:abc", "custom_headers": custom_headers},
        }
    }


def correlate(event):
    return TunerTelephony(provider="telnyx").correlate(event, livekit_uri=LIVEKIT)


def test_returns_token_as_custom_headers_and_keeps_the_uri():
    leg = correlate(
        envelope(
            [
                {"name": "X-Correlation-Id", "value": TOKEN},
                {"name": "X-LiveKit-Room", "value": "room-1"},
            ]
        )
    )
    assert leg.sip_uri == LIVEKIT
    assert leg.custom_headers == [{"name": "X-Correlation-Id", "value": TOKEN}]


def test_accepts_the_bare_payload():
    payload = envelope([{"name": "x-correlation-id", "value": TOKEN}])["data"]["payload"]
    assert correlate(payload).correlation_id == TOKEN


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
    leg = correlate(event)
    assert leg.sip_uri == LIVEKIT
    assert leg.custom_headers == []
