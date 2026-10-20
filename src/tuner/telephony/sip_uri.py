"""SIP URI header appending."""

from __future__ import annotations

from urllib.parse import quote


def with_header(sip_uri: str, name: str, value: str) -> str:
    """Set a URI header; carriers send URI headers as SIP headers on the INVITE.

    An existing header of the same name is replaced, and ``<sip:...>`` keeps its brackets.
    """
    uri = sip_uri.strip()
    bracketed = uri.startswith("<") and uri.endswith(">")
    if bracketed:
        uri = uri[1:-1]
    base, _, query = uri.partition("?")
    params = [p for p in query.split("&") if p and p.split("=", 1)[0].lower() != name.lower()]
    params.append(f"{name}={quote(value, safe='')}")
    result = f"{base}?{'&'.join(params)}"
    return f"<{result}>" if bracketed else result
