import asyncio
from types import SimpleNamespace

import pytest

from tuner.collector import SessionState
from tuner.config import TunerConfig
from tuner.plugin import TunerPlugin


class DummyRoom:
    def __init__(self):
        self.name = "test-room"
        self._handlers = {}

    def on(self, event):
        def decorator(fn):
            self._handlers[event] = fn
            return fn

        return decorator


class DummySession:
    def __init__(self):
        self.history = SimpleNamespace(items=[])
        self._handlers = {}

    def on(self, event):
        def decorator(fn):
            self._handlers[event] = fn
            return fn

        return decorator


class DummyCtx:
    def __init__(self):
        self.room = DummyRoom()
        self.job = SimpleNamespace(id="job-123")
        self.shutdown_callbacks = []

    def add_shutdown_callback(self, callback):
        self.shutdown_callbacks.append(callback)


def build_plugin() -> TunerPlugin:
    plugin = TunerPlugin.__new__(TunerPlugin)
    plugin._session = DummySession()
    plugin._ctx = DummyCtx()
    plugin._state = SessionState()

    async def recording_url_resolver(room_name, job_id):
        return "http://example.com/recording"

    plugin._config = TunerConfig(
        api_key="tr_api_test",
        workspace_id=1,
        agent_id="agent-1",
        recording_url_resolver=recording_url_resolver,
        timeout_seconds=1.0,
        max_retries=0,
    )

    plugin._submitted = False
    return plugin


@pytest.mark.asyncio
async def test_on_shutdown_submitted_guard(monkeypatch):
    plugin = build_plugin()

    submitted_calls = []
    
    async def fake_submit_call(payload, config):
        submitted_calls.append(payload)

    monkeypatch.setattr("tuner.plugin.submit_call", fake_submit_call)
    monkeypatch.setattr("tuner.plugin.to_create_call_request", lambda *args, **kwargs: {"call_id": "abc"})

    await plugin._on_shutdown("first")
    assert plugin._submitted is True
    assert len(submitted_calls) == 1
    assert plugin._state.shutdown_reason == "first"

    await plugin._on_shutdown("second")
    assert len(submitted_calls) == 1
    assert plugin._state.shutdown_reason == "first"


@pytest.mark.asyncio
async def test_close_event_triggers_on_shutdown(monkeypatch):
    plugin = build_plugin()

    shutdown_calls = []

    async def fake_shutdown(reason):
        shutdown_calls.append(reason)

    plugin._on_shutdown = fake_shutdown

    def fake_ensure_future(coro):
        task = asyncio.create_task(coro)
        return task

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    plugin._setup_event_listeners()

    # Trigger close event handler directly
    plugin._session._handlers["close"](SimpleNamespace(error=None))
    await asyncio.sleep(0)

    assert shutdown_calls == ["session_closed"]
