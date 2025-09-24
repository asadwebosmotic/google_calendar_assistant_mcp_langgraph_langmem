import logging
from dotenv import load_dotenv
import os
import sys
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import cartesia, deepgram, google, silero
from livekit.agents.llm import mcp
# Import our Gemini + MCP runner from mcp_client.py
from mcp_client import run_with_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("livekit.plugins.cartesia").setLevel(logging.DEBUG)
logger = logging.getLogger("agent")

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

    async def on_session_started(self, session: AgentSession):
        """Called when the session starts — greet the user."""
        logger.info("Session started, greeting user...")
        await session.say("Hello! How can I assist you with calendar events today?")

    async def on_user_message(self, message: str, session: AgentSession):
        """Called whenever the user speaks (after STT)."""
        logger.info(f"User said: {message}")
        # await session.generate_reply(message)

    async def on_user_message(self, message: str, session: AgentSession):
        logger.info(f"User said: {message}")

        try:
            # Send text through Gemini + MCP pipeline
            reply = await run_with_gemini(message)
            logger.info(f"Assistant reply: {reply}")
            await session.say(reply)
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            await session.say("Sorry, I encountered an error while processing that.")

async def entrypoint(ctx: agents.JobContext):
    logger.info("Starting agent session...")

    try:

        mcp_tool = mcp.MCPServerStdio(
            command=sys.executable,
            args=["mcp_server.py"],
        )

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-3", language="multi"),
            # llm=google.LLM(model="gemini-2.5-flash"),
            tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            allow_interruptions=True,
            mcp_servers=[mcp_tool],
        )
    except Exception as e:
        logger.error(f"Failed to initialize AgentSession: {e}")
        return

    @session.on("transcript")
    def _on_transcript(ev):
        logger.info(f"STT → {ev.alternatives[0].text}")

    @session.on("tts_audio")
    def _on_tts_audio(ev):
        logger.info("TTS → Audio frame generated")

    try:
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            )
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))