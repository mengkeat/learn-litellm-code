"""
v0_bash_agent.py - Mini LiteLLM Code: Bash is All You Need (~50 lines core)

Core Philosophy: "Bash is All You Need"
======================================
This is the ULTIMATE simplification of a coding agent. After building v1-v3,
we ask: what is the ESSENCE of an agent?

The answer: ONE tool (bash) + ONE loop = FULL agent capability.

Why Bash is Enough:
------------------
Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep                  |
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python v0_bash_agent.py "task"         |

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.

How Subagents Work:
------------------
    Main Agent
      |-- bash: python v0_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting

Usage:
    # Interactive mode
    python v0_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python v0_bash_agent.py "explore src/ and summarize"
"""

import llm_helper
import litellm
from dotenv import load_dotenv
import subprocess
import sys
import os

load_dotenv(override=True)
MODEL = llm_helper.get_model()

# The ONE tool that does everything (OpenAI function calling format for litellm)
# Notice how the description teaches the model common patterns AND how to spawn subagents
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v0_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
            "required": ["command"]
        }
    }
}]

# System prompt teaches the model HOW to use bash effectively
# Notice the subagent guidance - this is how we get hierarchical task decomposition
SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python v0_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""


def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.

    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)

    Returns:
        Final text response from the model
    """
    import json

    if history is None:
        history = []

    # Build messages with system prompt as first message (OpenAI format)
    messages = [{"role": "system", "content": SYSTEM}] + history
    messages.append({"role": "user", "content": prompt})
    history.append({"role": "user", "content": prompt})

    while True:
        # 1. Call the model with tools via litellm
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000
        )

        assistant_message = response.choices[0].message

        # 2. Append assistant message to history
        history.append(assistant_message.model_dump())
        messages.append(assistant_message.model_dump())

        # 3. If model didn't call tools, we're done
        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        # 4. Execute each tool call and collect results
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            cmd = args["command"]
            print(f"\033[33m$ {cmd}\033[0m")  # Yellow color for commands

            try:
                out = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.getcwd()
                )
                output = out.stdout + out.stderr
            except subprocess.TimeoutExpired:
                output = "(timeout after 300s)"

            print(output or "(empty)")

            # 5. Append tool result (OpenAI format)
            tool_result = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output[:50000]  # Truncate very long outputs
            }
            history.append(tool_result)
            messages.append(tool_result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        # This is how parent agents spawn children via bash
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        history = []
        while True:
            try:
                query = input("\033[36m>> \033[0m")  # Cyan prompt
            except (EOFError, KeyboardInterrupt):
                break
            if query in ("q", "exit", ""):
                break
            print(chat(query, history))
