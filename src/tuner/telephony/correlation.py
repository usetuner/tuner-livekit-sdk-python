"""Tuner's simulation correlation header: its name and the token shapes forwarded."""

from __future__ import annotations

import re

CORRELATION_HEADER = "X-Correlation-Id"

# The two Tuner id shapes: inbound uuid4().hex, outbound 26-char Base32. Anything else is dropped.
_TOKEN_RE = re.compile(r"[0-9a-fA-F]{32}|[A-Za-z2-7]{26}")


def valid_correlation_id(value: str | None) -> str:
    """Return ``value`` if it is safe to forward, else ``""``. Never log the value."""
    if value and _TOKEN_RE.fullmatch(value):
        return value
    return ""
