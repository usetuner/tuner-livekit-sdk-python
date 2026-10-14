"""SIP URI parsing and header appending."""

from __future__ import annotations

import re
from urllib.parse import quote

SIP_URI_RE = re.compile(
    r"^sips?:([^@\s;?]+)@([^@\s;?]+)(?:;[^?\s]*)?(?:\?[^\s]*)?$",
    re.IGNORECASE,
)


def split_sip_uri(raw: str) -> tuple[str, str]:
    """Return ``(user, host)`` of ``sip:user@host``; raise ValueError otherwise."""
    if not isinstance(raw, str):
        raise ValueError("SIP URI must be a string")
    match = SIP_URI_RE.fullmatch(raw)
    if not match:
        raise ValueError("SIP URI must be sip:user@host")
    return match.group(1), match.group(2)


def with_header(sip_uri: str, name: str, value: str) -> str:
    """Append a URI header; carriers send URI headers as SIP headers on the INVITE."""
    separator = "&" if "?" in sip_uri else "?"
    return f"{sip_uri}{separator}{name}={quote(value, safe='')}"
