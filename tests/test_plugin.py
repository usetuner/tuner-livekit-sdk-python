"""Tests for SIP participant capture in tuner.plugin"""

from __future__ import annotations

from types import SimpleNamespace

from livekit import rtc

from tuner.plugin import TunerPlugin

SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD

SIP_ATTRIBUTES = {
    "sip.phoneNumber": "+14155552671",
    "sip.callIDFull": "call-id-full",
    "sip.h.x-correlation-id": "corr-1",
}


class FakeEmitter:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event: str):
        def register(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return register

    def emit(self, event: str, *args) -> None:
        for fn in self.handlers.get(event, []):
            fn(*args)


class FakeRoom(FakeEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.remote_participants: dict[str, SimpleNamespace] = {}


def participant(kind=SIP, attributes=None, identity="sip-caller") -> SimpleNamespace:
    return SimpleNamespace(identity=identity, kind=kind, attributes=dict(attributes or {}))


def make_plugin() -> tuple[TunerPlugin, FakeRoom]:
    room = FakeRoom()
    ctx = SimpleNamespace(room=room, job=SimpleNamespace(id="job_1"), add_shutdown_callback=lambda _cb: None)
    plugin = TunerPlugin(FakeEmitter(), ctx, api_key="k", workspace_id=1, agent_id="agent")
    return plugin, room


def test_captures_sip_attributes_present_at_join():
    plugin, room = make_plugin()
    room.emit("participant_connected", participant(attributes=SIP_ATTRIBUTES))

    state = plugin._state
    assert state.is_sip is True
    assert state.caller_phone_number == "+14155552671"
    assert state.sip_call_id == "call-id-full"
    assert state.correlation_id == "corr-1"


def test_captures_attributes_set_after_join_when_caller_leaves_before_shutdown():
    plugin, room = make_plugin()
    caller = participant()
    room.remote_participants[caller.identity] = caller
    room.emit("participant_connected", caller)

    caller.attributes = dict(SIP_ATTRIBUTES)
    room.emit("participant_attributes_changed", dict(SIP_ATTRIBUTES), caller)
    del room.remote_participants[caller.identity]
    plugin._backfill_sip_state_from_room()

    assert plugin._state.sip_call_id == "call-id-full"
    assert plugin._state.correlation_id == "corr-1"


def test_second_sip_participant_does_not_replace_caller_identifiers():
    plugin, room = make_plugin()
    room.emit("participant_connected", participant(attributes=SIP_ATTRIBUTES))
    room.emit(
        "participant_connected",
        participant(
            attributes={"sip.phoneNumber": "+15550000000", "sip.callIDFull": "transfer-leg"},
            identity="sip-transfer",
        ),
    )

    state = plugin._state
    assert state.caller_phone_number == "+14155552671"
    assert state.sip_call_id == "call-id-full"
    assert state.correlation_id == "corr-1"


def test_ignores_attribute_changes_from_non_sip_participants():
    plugin, room = make_plugin()
    web_user = participant(kind=STANDARD, attributes=SIP_ATTRIBUTES, identity="web")
    room.emit("participant_attributes_changed", dict(SIP_ATTRIBUTES), web_user)

    assert plugin._state.is_sip is False
    assert plugin._state.correlation_id is None


def test_backfill_reads_caller_still_in_room():
    plugin, room = make_plugin()
    caller = participant(attributes=SIP_ATTRIBUTES)
    room.remote_participants[caller.identity] = caller
    plugin._backfill_sip_state_from_room()

    assert plugin._state.sip_call_id == "call-id-full"
    assert plugin._state.correlation_id == "corr-1"


def test_attach_sip_overrides_captured_values():
    from tuner.telephony import SipInfo

    plugin, room = make_plugin()
    room.emit("participant_connected", participant(attributes=SIP_ATTRIBUTES))
    plugin.attach_sip(
        SipInfo(
            sip_call_id="sip@host",
            sip_headers={"X-Correlation-Id": "tok123", "X-Other": "o"},
            recipient="sip:t@tuner",
        )
    )
    assert plugin._config.sip_call_id == "sip@host"
    assert plugin._config.recipient == "sip:t@tuner"
    assert plugin._state.sip_headers == {"X-Correlation-Id": "tok123", "X-Other": "o"}

    plugin.attach_sip(recipient="explicit", sip_call_id="explicit@host")
    assert plugin._config.recipient == "explicit"
    assert plugin._config.sip_call_id == "explicit@host"


def test_attach_sip_ignores_carrier_ids_and_malformed_tokens():
    from tuner.telephony import SipInfo

    plugin, _ = make_plugin()
    plugin.attach_sip(
        SipInfo(sip_call_id="CA1", carrier_call_id="CA1"),
        sip_headers={"X-Correlation-Id": "bad token!"},
    )
    assert plugin._config.sip_call_id is None
    assert plugin._state.sip_headers == {}
