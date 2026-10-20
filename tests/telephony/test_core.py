import pytest

from tuner.telephony.correlation import valid_correlation_id
from tuner.telephony.sip_uri import with_header


def test_with_header_appends_and_escapes():
    assert with_header("sip:a@b", "X-K", "v 1") == "sip:a@b?X-K=v%201"
    assert with_header("sip:a@b?X-A=1", "X-K", "v") == "sip:a@b?X-A=1&X-K=v"


def test_with_header_replaces_an_existing_header():
    assert with_header("sip:a@b?x-k=old&X-A=1", "X-K", "v") == "sip:a@b?X-A=1&X-K=v"


def test_with_header_keeps_angle_brackets():
    assert with_header("<sip:a@b;transport=tcp>", "X-K", "v") == "<sip:a@b;transport=tcp?X-K=v>"


@pytest.mark.parametrize("token", ["17f03659db514827908c1eacaf8b7274", "C7YDMWNXKFEJ7EIMHKWK7C3SOQ"])
def test_valid_correlation_id_keeps_token_shapes(token):
    assert valid_correlation_id(token) == token


@pytest.mark.parametrize("token", [None, "", "has space", "a" * 129, "x&X-Other=1", "trace-abc", "abc123", "a" * 33])
def test_valid_correlation_id_drops_anything_else(token):
    assert valid_correlation_id(token) == ""
