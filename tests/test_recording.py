"""Tests for tuner.recording.upload_recording: path, bytes and stream sources."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tuner.config import TunerConfig
from tuner.recording import upload_recording


class FakeS3:
    """Local server standing in for the presigned POST endpoint; records raw bodies."""

    def __init__(self, status: int = 204) -> None:
        self.status = status
        self.bodies: list[bytes] = []

    async def handle(self, request: web.Request) -> web.Response:
        self.bodies.append(await request.read())
        return web.Response(status=self.status)

    @property
    def called(self) -> bool:
        return bool(self.bodies)

    def sent_filename(self) -> str:
        body = self.bodies[-1].decode("utf-8", "replace")
        marker = 'name="file"; filename="'
        start = body.index(marker) + len(marker)
        return body[start : body.index('"', start)]


@pytest.fixture
def config() -> TunerConfig:
    return TunerConfig(api_key="test-key", agent_id="agent-456", workspace_id=1)


@pytest.fixture
async def s3():
    fake = FakeS3()
    app = web.Application()
    app.router.add_post("/bucket", fake.handle)
    server = TestServer(app)
    await server.start_server()
    fake.url = str(server.make_url("/bucket"))
    yield fake
    await server.close()


def _response(s3: Any, **target_extra: Any) -> dict[str, Any]:
    return {
        "recording_upload": {
            "url": s3.url,
            "fields": {"key": "calls/call-123.wav", "policy": "abc"},
            "content_type": "audio/wav",
            "key": "calls/call-123.wav",
            **target_extra,
        }
    }


async def test_uploads_from_path(config, s3, tmp_path: Path) -> None:
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFFfake")

    assert await upload_recording(_response(s3), audio, config) is True
    assert s3.sent_filename() == "call.wav"


async def test_uploads_from_bytes(config, s3) -> None:
    assert await upload_recording(_response(s3), b"RIFFfake", config) is True
    # No path to borrow a name from: falls back to the target key.
    assert s3.sent_filename() == "call-123.wav"


async def test_fields_come_before_file(config, s3) -> None:
    assert await upload_recording(_response(s3), b"RIFFfake", config) is True
    body = s3.bodies[-1]
    assert body.index(b'name="policy"') < body.index(b'name="file"')
    assert body.index(b'name="key"') < body.index(b'name="file"')


async def test_uploads_from_stream_with_explicit_filename(config, s3) -> None:
    stream = io.BytesIO(b"RIFFfake")

    result = await upload_recording(
        _response(s3), stream, config, filename="stream.wav"
    )

    assert result is True
    assert s3.sent_filename() == "stream.wav"


async def test_bytes_over_max_bytes_are_skipped(config, s3) -> None:
    result = await upload_recording(_response(s3, max_bytes=4), b"RIFFfake", config)

    assert result is False
    assert not s3.called


async def test_seekable_stream_over_max_bytes_is_skipped(config, s3) -> None:
    result = await upload_recording(
        _response(s3, max_bytes=4), io.BytesIO(b"RIFFfake"), config
    )

    assert result is False
    assert not s3.called


async def test_unmeasurable_stream_uploads_whole_body(config, s3) -> None:
    """A non-seekable stream skips the local guard; S3's policy enforces it."""

    class Unseekable(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, b: Any) -> int:
            return self._buf.readinto(b)

    result = await upload_recording(
        _response(s3, max_bytes=4), Unseekable(b"RIFFfake"), config, filename="s.wav"
    )

    assert result is True
    assert b"RIFFfake" in s3.bodies[-1]


async def test_seekable_stream_is_sent_whole_and_measured_whole(config, s3) -> None:
    stream = io.BytesIO(b"SKIPRIFFfake")
    stream.seek(4)

    # 8 bytes remain from the current position, but 12 will be uploaded.
    assert await upload_recording(_response(s3, max_bytes=10), stream, config) is False
    assert not s3.called

    stream.seek(4)
    assert await upload_recording(_response(s3), stream, config) is True
    assert b"SKIPRIFFfake" in s3.bodies[-1]


async def test_rejected_upload_returns_false(config, s3) -> None:
    s3.status = 403
    assert await upload_recording(_response(s3), b"RIFFfake", config) is False


async def test_missing_path_is_reported(config, s3, tmp_path: Path) -> None:
    result = await upload_recording(_response(s3), tmp_path / "nope.wav", config)
    assert result is False


async def test_response_without_target_is_skipped(config) -> None:
    assert await upload_recording({}, b"RIFFfake", config) is False
    assert await upload_recording(None, b"RIFFfake", config) is False


def test_recording_options_are_last_so_positional_args_keep_working():
    import dataclasses
    import inspect

    from tuner import TunerConfig

    new = [
        "recording_file_extension",
        "recording_s3_multi_channel",
        "recording_s3_caller_channel",
    ]
    assert [f.name for f in dataclasses.fields(TunerConfig)][-3:] == new
    assert list(inspect.signature(TunerConfig.from_env).parameters)[-3:] == new
