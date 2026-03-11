import logging

from livekit.agents import AgentSession

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tuner_livekit_sdk import TunerPlugin, TunerConfig

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful resturant receptionist who receive calls for booking tables, you should ask for number of guests, date and time. be nice and professional and asnwer any question related to booking by the customer.""",
        )

    @function_tool
    async def check_table_availability(
        self,
        context: RunContext,
        date: str,
        time: str,
        guests: int,
    ) -> str:
        """Check if the restaurant can accept a booking for the requested date/time and guest count.

        Args:
            date: Reservation date in YYYY-MM-DD format.
            time: Reservation time in HH:MM format.
            guests: Number of guests.
        """
        logger.info(
            "Checking availability",
            extra={"date": date, "time": time, "guests": guests},
        )

        if guests <= 0:
            return "Invalid guest count. Please provide a number greater than 0."
        if guests > 8:
            return "For parties larger than 8, please contact our events team."

        return f"Table is available on {date} at {time} for {guests} guests."

    @function_tool
    async def create_booking_summary(
        self,
        context: RunContext,
        customer_name: str,
        date: str,
        time: str,
        guests: int,
    ) -> str:
        """Create a booking confirmation summary.

        Args:
            customer_name: Guest full name.
            date: Reservation date in YYYY-MM-DD format.
            time: Reservation time in HH:MM format.
            guests: Number of guests.
        """
        return (
            "Booking summary:\n"
            f"- Name: {customer_name}\n"
            f"- Date: {date}\n"
            f"- Time: {time}\n"
            f"- Guests: {guests}"
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

def calculate_cost(usage):
    return 2 * usage.llm_prompt_tokens / 1000 + 4 * usage.llm_completion_tokens / 1000

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-3", language="en-US"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    TunerPlugin(                                                                                                                                                                                                                                                                  
      session,                                                                                                                                                                                                                                                                
      ctx,
      cost_calculator=calculate_cost,  # Example cost function (USD)
  )
    
    
    # Create the assistant with configured function tools.
    assistant = Assistant()

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
