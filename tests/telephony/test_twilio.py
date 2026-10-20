import pytest

from tuner.telephony import extract_sip, forward_to_sip

LIVEKIT = "sip:sim-nova-clinic@project.sip.livekit.cloud"
TOKEN = "17f03659db514827908c1eacaf8b7274"


def forward(params, sip_uri=LIVEKIT):
    return forward_to_sip(extract_sip(params, provider="twilio"), sip_uri=sip_uri)


def test_appends_token_to_the_livekit_uri():
    fwd = forward({"CallSid": "CA1", "SipHeader_X-Correlation-Id": TOKEN})
    assert fwd.sip_uri == f"{LIVEKIT}?X-Correlation-Id={TOKEN}"
    assert fwd.correlation_id == TOKEN
    assert fwd.twiml == (
        f"<Response><Dial><Sip>{LIVEKIT}?X-Correlation-Id={TOKEN}</Sip></Dial></Response>"
    )


def test_reads_the_param_case_insensitively():
    assert forward({"sipheader_x-correlation-id": TOKEN}).correlation_id == TOKEN


def test_extracts_sip_call_id_headers_and_recipient():
    info = extract_sip(
        {
            "CallSid": "CA1",
            "SipCallId": "sip@host",
            "SipHeader_X-Correlation-Id": TOKEN,
            "Direction": "outbound-api",
            "To": "sip:t@tuner",
        },
        provider="twilio",
    )
    assert info.sip_call_id == "sip@host"
    assert info.carrier_call_id == "CA1"
    assert info.sip_headers == {"SipHeader_X-Correlation-Id": TOKEN, "X-Correlation-Id": TOKEN}
    assert info.recipient == "sip:t@tuner"
    assert extract_sip({"CallSid": "CA1", "To": "+1"}, provider="twilio").recipient is None


def test_keeps_existing_uri_params():
    fwd = forward({"SipHeader_X-Correlation-Id": TOKEN}, sip_uri=f"{LIVEKIT};transport=tcp?X-A=1")
    assert fwd.sip_uri == f"{LIVEKIT};transport=tcp?X-A=1&X-Correlation-Id={TOKEN}"


@pytest.mark.parametrize("params", [{}, {"CallSid": "CA1"}, {"SipHeader_X-Correlation-Id": ""}])
def test_production_call_gets_the_uri_unchanged(params):
    assert forward(params).sip_uri == LIVEKIT


def test_malformed_token_is_not_forwarded():
    assert forward({"SipHeader_X-Correlation-Id": "x&X-Other=1"}).sip_uri == LIVEKIT


def test_non_mapping_event_is_ignored():
    assert forward(None).sip_uri == LIVEKIT


def test_works_with_a_multidict_like_form():
    class Form(dict):
        pass

    assert forward(Form({"SipHeader_X-Correlation-Id": TOKEN})).correlation_id == TOKEN
