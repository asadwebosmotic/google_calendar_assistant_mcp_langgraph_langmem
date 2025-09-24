# langflow.py
import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END, MessagesState

from prompts import (
    INTENT_PROMPT, DATA_PROMPT, VALIDATION_PROMPT,
    ACTION_PROMPT, FEEDBACK_PROMPT
)

from mcp_wrapper import MCPClient
from utils_serialization import serialize_tool_result
import google.generativeai as genai
import os
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Configure Gemini from environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning("[gemini] failed to configure API key: %s", e)
else:
    logger.warning("[gemini] GEMINI_API_KEY not set; calls will likely fail")

# ----------------- State Schema -----------------
class CalendarState(MessagesState):
    intent: str | None = None
    data: Dict[str, Any] | None = None
    validation: Dict[str, Any] | None = None
    action_result: Dict[str, Any] | None = None
    message: str | None = None


# ----------------- Helper -----------------
def ask_gemini(prompt: str, context: Dict[str, Any]) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    text = prompt + f"\n\nContext:\n{json.dumps(context, indent=2)}"
    response = model.generate_content(text)
    return response.text.strip()


def _last_message_content(messages) -> str:
    """Return the content of the last message.
    Handles both dict-based messages and LangChain Message objects (e.g., HumanMessage).
    """
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return last.get("content", "")
    # Fallback to attribute-style used by LangChain messages
    content = getattr(last, "content", None)
    if content is not None:
        return content
    # As a last resort, string-cast
    return str(last)


def _safe_json_loads(s: str) -> dict:
    """Best-effort JSON parsing.
    - Try direct json.loads
    - If fails, try to locate the first '{' and the last '}' and parse that substring
    - If still fails, return an empty dict
    """
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start : end + 1])
        except Exception:
            pass
    logger.debug("[_safe_json_loads] failed to parse JSON from: %s", s[:200])
    return {}


# ----------------- Nodes -----------------
def intent_node(state: dict) -> dict:
    logger.debug("[intent_node] running with messages=%s", state.get("messages"))
    messages = state.get("messages", [])
    query = _last_message_content(messages)
    intent_json = ask_gemini(INTENT_PROMPT, {"query": query})
    parsed = _safe_json_loads(intent_json)
    state["intent"] = parsed.get("intent")
    return state


def data_node(state: dict) -> dict:
    logger.debug("[data_node] extracting data for intent=%s", state.get("intent"))
    query = _last_message_content(state.get("messages", []))
    data_json = ask_gemini(DATA_PROMPT, {"intent": state.get("intent"), "query": query})
    state["data"] = _safe_json_loads(data_json)
    return state


def validation_node(state: dict) -> dict:
    logger.debug("[validation_node] validating data=%s", state.get("data"))
    val_json = ask_gemini(VALIDATION_PROMPT, {"intent": state.get("intent"), "data": state.get("data")})
    state["validation"] = _safe_json_loads(val_json)
    return state


async def action_node(state: dict) -> dict:
    logger.debug("[action_node] executing intent=%s", state.get("intent"))
    validation = state.get("validation", {}) or {}
    if not validation.get("valid", False):
        state["action_result"] = {"status": "error", "details": validation.get("errors")}
        return state

    async with MCPClient() as client:
        result = await client.call_tool(state.get("intent"), state.get("data"))
        state["action_result"] = serialize_tool_result(result)
    return state


def feedback_node(state: dict) -> dict:
    logger.debug("[feedback_node] generating feedback")
    fb_text = ask_gemini(FEEDBACK_PROMPT, {"result": state.get("action_result")})
    state["message"] = fb_text
    return state

# Initialize in-memory checkpointer for LangGraph
checkpointer = MemorySaver()
# ----------------- Graph Builder -----------------
def build_graph():
    logger.debug("[build_graph] compiling graph")
    workflow = StateGraph(state_schema=CalendarState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("data", data_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("action", action_node)
    workflow.add_node("feedback", feedback_node)

    workflow.add_edge(START, "intent")
    workflow.add_edge("intent", "data")
    workflow.add_edge("data", "validation")
    workflow.add_edge("validation", "action")
    workflow.add_edge("action", "feedback")
    workflow.add_edge("feedback", END)

    return workflow.compile(checkpointer=checkpointer)
