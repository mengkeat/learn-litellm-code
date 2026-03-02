#!/usr/bin/env python3
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

from utils import fn_to_tool

load_dotenv(override=True)

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_msg) for all tool-role messages
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "tool":
            tool_results.append((msg_idx, msg))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_call_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
                name = fn.name if hasattr(fn, "name") else fn.get("name", "unknown")
                if tc_id:
                    tool_name_map[tc_id] = name
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, msg in to_clear:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            tool_id = msg.get("tool_call_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            msg["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.choices[0].message.content
    # Replace all messages with compressed summary
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]


# -- Tool implementations --
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

def compact(focus: str = None) -> str:
    """Trigger manual conversation compression."""
    return "Manual compression requested."


def execute_tool(name: str, args: dict) -> str:
    handlers = {
        "bash": lambda a: run_bash(a["command"]),
        "read_file": lambda a: run_read(a["path"], a.get("limit")),
        "write_file": lambda a: run_write(a["path"], a["content"]),
        "edit_file": lambda a: run_edit(a["path"], a["old_text"], a["new_text"]),
        "compact": lambda a: compact(a.get("focus")),
    }
    handler = handlers.get(name)
    return handler(a=args) if handler else f"Unknown tool: {name}"


TOOLS = [
    fn_to_tool(run_bash, "bash"),
    fn_to_tool(run_read, "read_file"),
    fn_to_tool(run_write, "write_file"),
    fn_to_tool(run_edit, "edit_file"),
    fn_to_tool(compact, "compact"),
]


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = litellm.completion(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            return

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        manual_compact = False
        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else (fn.arguments or {})
            if name == "compact":
                manual_compact = True
            try:
                output = execute_tool(name, args)
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {name}: {str(output)[:200]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})

        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    query = "Read every Python file content in directory one by one"
    history.append({"role": "user", "content": query})
    agent_loop(history)
    print()
