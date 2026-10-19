"""Every built-in provider must leave production calls (no token) untouched."""

import pytest

from tuner.telephony import Forward, SipInfo
from tuner.telephony.providers import registry

LIVEKIT = "sip:agent@xxxx.sip.livekit.cloud"


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
def test_no_token_means_unchanged_leg(name):
    provider = registry.get_provider(name)
    assert provider.forward_to_sip(SipInfo(), LIVEKIT) == Forward(sip_uri=LIVEKIT)


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
@pytest.mark.parametrize("payload", [{}, None, "x"])
def test_empty_payload_has_no_token(name, payload):
    assert registry.get_provider(name).extract(payload).correlation_id == ""


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
def test_token_reaches_the_leg(name):
    info = SipInfo(sip_headers={"X-Correlation-Id": "17f03659db514827908c1eacaf8b7274"})
    fwd = registry.get_provider(name).forward_to_sip(info, LIVEKIT)
    assert fwd.correlation_id == "17f03659db514827908c1eacaf8b7274"
    assert fwd.headers == {"X-Correlation-Id": "17f03659db514827908c1eacaf8b7274"}
