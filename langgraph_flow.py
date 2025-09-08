# langgraph_calendar.py

from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.config import get_config
from langmem import create_memory_store_manager

# Initialize configuration and models
config = get_config()
chat_model = init_chat_model(config)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # pretrained embedding model

# Create memory manager
memory_store = create_memory_store_manager(config)

# Define memory specifications
persistent_memory = memory_store.get_store("persistent_memory")
episodic_memory = memory_store.get_store("episodic_memory")

# Define graph states and nodes
graph = StateGraph(name="calendar_assistant")

# --- Intent Node ---
@graph.state(name="intent_state")
async def intent_state(messages: MessagesState, persistent_memory= persistent_memory, episodic_memory= episodic_memory):
    prompt = """
    You are an assistant helping to schedule events. Classify the user's intent into one of these: list, create, update, delete. Extract details like date, time, title, attendees, and timezone.
    If any required information is missing, clearly list the missing fields and ask for them in the next message.
    """
    response = await chat_model.ask(prompt, messages)
    return {"intent": response.intent, "parameters": response.parameters, "missing_fields": response.missing_fields}

# --- Data Node ---
@graph.state(name="data_state")
async def data_state(messages: MessagesState, intent: str, parameters: dict, persistent_memory=persistent_memory):
    prompt = """
    Based on the intent and provided parameters, fetch existing events that may overlap. Also, suggest relevant calendar preferences based on past behavior stored in memory.
    """
    response = await chat_model.ask(prompt, messages)
    return {"existing_events": response.existing_events, "available_data": response.available_data}

# --- Validation Node ---
@graph.state(name="validation_state")
async def validation_state(messages: MessagesState, existing_events: list, parameters: dict):
    prompt = """
    Check if the proposed event conflicts with existing events or if any information is missing.
    Provide a validation status: valid, conflict, or incomplete.
    Suggest recommendations to resolve conflicts or fill missing fields.
    """
    response = await chat_model.ask(prompt, messages)
    return {"conflicts": response.conflicts, "validation_status": response.validation_status, "recommendations": response.recommendations}

# --- Action Node ---
@graph.state(name="action_state")
async def action_state(messages: MessagesState, validation_status: str, parameters: dict, recommendations: list, user_confirmation: bool):
    prompt = f"""
    Based on the validation status and recommendations, decide whether to proceed with the action.
    If conflicts exist, ask the user for confirmation before finalizing.
    Respond with the outcome of the action and next steps.
    """
    response = await chat_model.ask(prompt, messages)
    return {"action_result": response.action_result}

# --- Feedback Node ---
@graph.state(name="feedback_state")
async def feedback_state(messages: MessagesState, action_result: dict):
    prompt = """
    Summarize the final outcome of the request in a user-friendly way.
    Provide suggestions or ask for further actions if needed.
    """
    response = await chat_model.ask(prompt, messages)
    return {"message": response.message}

# --- Memory Saver Checkpoints ---
@graph.checkpoint(name="save_memory")
async def save_memory(messages: MessagesState, persistent_memory=persistent_memory, episodic_memory=episodic_memory):
    await MemorySaver(persistent_memory).save(messages)
    await MemorySaver(episodic_memory).save(messages)

# --- Entry Point ---
@entrypoint(graph, start_state=START)
async def run_calendar_assistant():
    return {
        "messages": MessagesState(),
        "persistent_memory": persistent_memory,
        "episodic_memory": episodic_memory
    }
