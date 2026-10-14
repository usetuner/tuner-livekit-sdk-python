import pytest

from tuner.telephony.correlation import valid_correlation_id
from tuner.telephony.sip_uri import split_sip_uri, with_header


def test_split_sip_uri_keeps_user_and_host():
    assert split_sip_uri("sip:Exact.User@tuner.example;transport=tcp?X-A=1") == (
        "Exact.User",
        "tuner.example",
    )


@pytest.mark.parametrize("raw", ["", "+15551234567", "sip:nohost", 42])
def test_split_sip_uri_rejects_non_sip(raw):
    with pytest.raises(ValueError):
        split_sip_uri(raw)


def test_with_header_appends_and_escapes():
    assert with_header("sip:a@b", "X-K", "v 1") == "sip:a@b?X-K=v%201"
    assert with_header("sip:a@b?X-A=1", "X-K", "v") == "sip:a@b?X-A=1&X-K=v"


@pytest.mark.parametrize("token", ["17f03659db514827908c1eacaf8b7274", "a.b_c-d"])
def test_valid_correlation_id_keeps_token_shapes(token):
    assert valid_correlation_id(token) == token


@pytest.mark.parametrize("token", [None, "", "has space", "a" * 129, "x&X-Other=1"])
def test_valid_correlation_id_drops_anything_else(token):
    assert valid_correlation_id(token) == ""
