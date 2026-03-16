from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from .config import TunerConfig

logger = logging.getLogger("livekit_agents_tuner.client")

# HTTP status codes that warrant a retry
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


async def submit_call(payload: dict, config: "TunerConfig") -> None:
    """
    POST the call payload to the Tuner API with exponential-backoff retries.

    Retry policy:
      - Retries on HTTP 5xx and connection errors.
      - Does NOT retry on HTTP 4xx (log and abandon).
      - Delays: 1s, 2s, 4s (+ 0–500ms jitter) between attempts.
    """
    url = (
        f"{config.base_url.rstrip('/')}/api/v1/public/call"
        f"?workspace_id={config.workspace_id}"
        f"&agent_remote_identifier={config.agent_id}"
    )
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    logger.debug("Tuner API request: POST %s\nPayload: %s", url, payload)

    last_exc: Exception | None = None

    async with aiohttp.ClientSession() as http_session:
        for attempt in range(config.max_retries + 1):
            if attempt > 0:
                delay = float(2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning(
                    "Tuner submission attempt %d/%d failed, retrying in %.1fs",
                    attempt,
                    config.max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                async with http_session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                ) as resp:
                    if resp.status in (200, 201):
                        try:
                            data = await resp.json()
                        except Exception:
                            data = {}
                        logger.info(
                            "Call submitted to Tuner (call_id=%s, tuner_id=%s, is_new=%s)",
                            payload.get("call_id"),
                            data.get("id"),
                            data.get("is_new"),
                        )
                        return

                    if resp.status in _RETRY_STATUSES:
                        body = await resp.text()
                        last_exc = RuntimeError(
                            f"Tuner API returned {resp.status}: {body[:200]}"
                        )
                        continue  # retry

                    # 4xx or unexpected — do not retry
                    body = await resp.text()
                    logger.error(
                        "Tuner API returned non-retryable status %d for call_id=%s: %s",
                        resp.status,
                        payload.get("call_id"),
                        body[:500],
                    )
                    return

            except aiohttp.ClientError as exc:
                last_exc = exc
            except asyncio.TimeoutError as exc:
                last_exc = exc

    logger.error(
        "Tuner submission failed after %d attempt(s) for call_id=%s: %s",
        config.max_retries + 1,
        payload.get("call_id"),
        last_exc,
    )
