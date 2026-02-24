import inspect
from pydantic import create_model

# -- Reflect function signatures into OpenAI tool schemas --
def fn_to_tool(fn, name: str = None) -> dict:
    """Convert a function to an OpenAI tool definition via inspect + pydantic."""
    sig = inspect.signature(fn)
    fields = {}
    for pname, param in sig.parameters.items():
        ann = param.annotation if param.annotation != inspect.Parameter.empty else str
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[pname] = (ann, default)
    schema = create_model(fn.__name__, **fields).model_json_schema()
    return {"type": "function", "function": {
        "name": name or fn.__name__,
        "description": fn.__doc__ or "",
        "parameters": schema,
    }}
    