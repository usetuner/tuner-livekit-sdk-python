"""Upload a call recording to Tuner's storage.

Setting ``recording_file_extension`` opts the call into a presigned upload target,
returned as ``recording_upload`` in the submit response. The target is an S3
presigned POST: multipart/form-data with the policy fields first and the file
last (S3 ignores fields sent after the file).

Usage::

    response = await plugin.wait_for_submit()
    await upload_recording(response, "call.wav", plugin.config)
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import IO, Any

import aiohttp

logger = logging.getLogger("tuner.recording")


def _measure(stream: IO[bytes]) -> int | None:
    """Total length of a seekable stream (from 0, since it is rewound before sending)."""
    if not stream.seekable():
        return None
    start = stream.tell()
    end = stream.seek(0, io.SEEK_END)
    stream.seek(start)
    return end


async def upload_recording(
    submit_response: dict[str, Any] | None,
    audio: Path | str | bytes | bytearray | IO[bytes],
    config: Any,
    *,
    filename: str | None = None,
) -> bool:
    """POST a recording to the presigned target in a submit response.

    Args:
        submit_response: What ``wait_for_submit`` / ``submit_call`` returned. No
            ``recording_upload`` key means the call was not opted in; the upload
            is skipped.
        audio: A filesystem path, raw bytes, or an open binary stream. A seekable
            stream is rewound and sent whole.
        config: TunerConfig, for the request timeout.
        filename: Name for the uploaded part. Defaults to the path's name, or to
            the target key's name.

    Returns:
        True if the recording was uploaded. Never raises.
    """
    if not submit_response:
        return False

    target = submit_response.get("recording_upload")
    if not target:
        return False

    url = target.get("url")
    if not url:
        logger.error("Tuner upload target has no url; skipping")
        return False

    key = target.get("key")
    max_bytes = target.get("max_bytes")

    handle: IO[bytes] | None = None
    payload: bytes | IO[bytes]
    size: int | None
    part_name = filename

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.is_file():
            logger.error("Recording not found for upload: %s", path)
            return False
        size = path.stat().st_size
        handle = path.open("rb")
        payload = handle
        part_name = part_name or path.name
    elif isinstance(audio, (bytes, bytearray)):
        payload = bytes(audio)
        size = len(payload)
    else:
        payload = audio
        # Non-seekable: size unknown, S3's content-length-range policy enforces it.
        size = _measure(audio)
        if size is not None:
            audio.seek(0)

    part_name = part_name or (Path(key).name if key else "recording")

    try:
        if max_bytes and size is not None and size > max_bytes:
            logger.error(
                "Recording is %d bytes, over the %d byte upload limit; skipping",
                size,
                max_bytes,
            )
            return False

        # The file part must come last: S3 ignores fields that follow it.
        form = aiohttp.FormData()
        for name, value in (target.get("fields") or {}).items():
            form.add_field(name, str(value))
        form.add_field(
            "file",
            payload,
            filename=part_name,
            content_type=target.get("content_type") or "application/octet-stream",
        )

        timeout = aiohttp.ClientTimeout(total=getattr(config, "timeout_seconds", 30.0))
        async with (
            aiohttp.ClientSession(timeout=timeout) as http_session,
            http_session.post(url, data=form) as resp,
        ):
            # A presigned POST answers 204, or 201 when the policy asks for it.
            if resp.status not in (200, 201, 204):
                body = await resp.text()
                logger.error("Recording upload rejected %d: %s", resp.status, body[:300])
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        logger.error("Recording upload failed: %s", exc)
        return False
    finally:
        if handle is not None:
            handle.close()

    logger.info(
        "Uploaded recording (%s bytes) to %s",
        size if size is not None else "unknown",
        key or "?",
    )
    return True
