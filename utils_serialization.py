# utils_serialization.py
import json

def serialize_tool_result(result):
    """Convert CallToolResult or MCP content into JSON-serializable form."""
    if result is None:
        return None

    # If it's already a simple JSON type
    if isinstance(result, (dict, str, int, float, bool)):
        return result

    # If it's a list, serialize each element
    if isinstance(result, list):
        return [serialize_tool_result(r) for r in result]

    # If it has .to_dict() (MCP objects often do)
    if hasattr(result, "to_dict"):
        return result.to_dict()

    # If it's a Pydantic-like object
    if hasattr(result, "__dict__"):
        return {k: serialize_tool_result(v) for k, v in result.__dict__.items()}

    # Special handling for TextContent-like objects
    if hasattr(result, "text"):
        return result.text

    if hasattr(result, "_pb"):  # protobuf fallback
        return json.loads(json.dumps(result, default=str))

    # Fallback: string representation
    return str(result)
