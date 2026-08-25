"""Behavioral evals for the example receptionist agent.

These use the LiveKit Agents testing framework and call a real LLM through
LiveKit Inference, so they need LIVEKIT_URL / LIVEKIT_API_KEY /
LIVEKIT_API_SECRET in the environment.

    uv run pytest

See https://docs.livekit.io/agents/build/testing/
"""

import pytest
from livekit.agents import AgentSession, inference, llm

from agent import Assistant


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


@pytest.mark.asyncio
async def test_greets_the_caller() -> None:
    """The receptionist opens the call politely."""
    async with (
        _llm() as judge_llm,
        AgentSession(llm=judge_llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="Hello")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="""
                Greets the caller in a friendly, professional manner as a restaurant receptionist.

                Optional context that may or may not be included:
                - An offer to help with a reservation
                - A question about the booking details
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_collects_missing_booking_details() -> None:
    """A partial request should be met with a question, not a guess."""
    async with (
        _llm() as judge_llm,
        AgentSession(llm=judge_llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="I'd like to book a table.")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="""
                Asks the caller for at least one of the missing booking details:
                the number of guests, the date, or the time.

                The response must not invent a date, time or guest count on the
                caller's behalf, and must not claim the booking is confirmed.
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_checks_availability_with_a_tool_call() -> None:
    """With all details supplied, the agent calls check_table_availability."""
    async with (
        _llm() as judge_llm,
        AgentSession(llm=judge_llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(
            user_input=(
                "I'd like a table for four people on 2025-11-20 at 19:30, please."
            )
        )

        result.expect.next_event().is_function_call(
            name="check_table_availability",
            arguments={"date": "2025-11-20", "time": "19:30", "guests": 4},
        )
        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="Tells the caller the table is available for the requested date, time and party size.",
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_declines_oversized_party() -> None:
    """Parties over 8 are routed to the events team rather than confirmed."""
    async with (
        _llm() as judge_llm,
        AgentSession(llm=judge_llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(
            user_input="We're a group of twelve, can we come on 2025-12-01 at 20:00?"
        )

        result.expect.next_event().is_function_call(name="check_table_availability")
        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="""
                Explains that a party this large cannot be booked here and points the
                caller to the events team. Does not confirm the reservation.
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_stays_on_topic() -> None:
    """Off-topic questions are declined without inventing an answer."""
    async with (
        _llm() as judge_llm,
        AgentSession(llm=judge_llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="What city was I born in?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="""
                Does not claim to know or provide the caller's birthplace.

                The response should not:
                - State a specific city where the caller was born
                - Claim to have access to the caller's personal information

                It may explain the lack of access, say it doesn't know, or steer the
                conversation back to the reservation.
                """,
            )
        )

        result.expect.no_more_events()
