"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import json
import os
import subprocess

import inspect

import litellm
from pathlib import Path
from dotenv import load_dotenv
from pydantic import create_model

load_dotenv(override=True)

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """Run a shell command."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    """Read file contents."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """Write content to file."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, args: dict) -> str:
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    return f"Unknown tool: {name}"


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

def run_task(prompt: str, description: str = "") -> str:
    """Spawn a subagent with fresh context. It shares the filesystem but not conversation history."""

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
    fn_to_tool(run_bash, "bash"),
    fn_to_tool(run_read, "read_file"),
    fn_to_tool(run_write, "write_file"),
    fn_to_tool(run_edit, "edit_file"),
]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for _ in range(30):  # safety limit
        print(f"  subagent> {messages[-1]['content'][:80]}")
        response = litellm.completion(
            model=MODEL, messages=messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls

        if not tool_calls:
            return content or "(no summary)"

        # Append assistant message with tool calls
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else (fn.arguments or {})
            output = execute_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)[:50000]})

    return content or "(no summary)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [fn_to_tool(run_task, "task")]


def agent_loop(messages: list):
    while True:
        response = litellm.completion(
            model=MODEL, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls

        if content:
            print(content)

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            return

        # Append assistant message with tool calls
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else (fn.arguments or {})

            if name == "task":
                desc = args.get("description", "subtask")
                print(f"> task ({desc}): {args['prompt'][:80]}")
                output = run_subagent(args["prompt"])
            else:
                output = execute_tool(name, args)

            print(f"  {str(output)[:200]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    query = "Delegate: read all .py files and summarize what each one does"
    history.append({"role": "user", "content": query})
    agent_loop(history)
    print()
