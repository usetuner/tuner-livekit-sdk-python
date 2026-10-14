import pytest

from tuner.telephony import TunerTelephony

LIVEKIT = "sip:sim-nova-clinic@project.sip.livekit.cloud"
TOKEN = "17f03659db514827908c1eacaf8b7274"


def correlate(params, livekit_uri=LIVEKIT):
    return TunerTelephony(provider="twilio").correlate(params, livekit_uri=livekit_uri)


def test_appends_token_to_the_livekit_uri():
    leg = correlate({"CallSid": "CA1", "SipHeader_X-Correlation-Id": TOKEN})
    assert leg.sip_uri == f"{LIVEKIT}?X-Correlation-Id={TOKEN}"
    assert leg.correlation_id == TOKEN


def test_reads_the_param_case_insensitively():
    assert correlate({"sipheader_x-correlation-id": TOKEN}).correlation_id == TOKEN


def test_keeps_existing_uri_params():
    leg = correlate(
        {"SipHeader_X-Correlation-Id": TOKEN}, livekit_uri=f"{LIVEKIT};transport=tcp?X-A=1"
    )
    assert leg.sip_uri == f"{LIVEKIT};transport=tcp?X-A=1&X-Correlation-Id={TOKEN}"


@pytest.mark.parametrize("params", [{}, {"CallSid": "CA1"}, {"SipHeader_X-Correlation-Id": ""}])
def test_production_call_gets_the_uri_unchanged(params):
    assert correlate(params).sip_uri == LIVEKIT


def test_malformed_token_is_not_forwarded():
    assert correlate({"SipHeader_X-Correlation-Id": "x&X-Other=1"}).sip_uri == LIVEKIT


def test_non_mapping_event_is_ignored():
    assert correlate(None).sip_uri == LIVEKIT


def test_works_with_a_multidict_like_form():
    class Form(dict):
        pass

    assert correlate(Form({"SipHeader_X-Correlation-Id": TOKEN})).correlation_id == TOKEN
