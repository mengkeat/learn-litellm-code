# Agents Project Context

## Overview

This project contains minimal AI coding agents that demonstrate the "Bash is All You Need" philosophy - a single bash tool combined with an agentic loop provides full agent capability.

## UV Python Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Quick Start

```bash
# Run the agent (uv automatically creates venv and installs deps)
uv run python v0_bash_agent.py

# Or activate the environment manually
source .venv/bin/activate
python v0_bash_agent.py
```

### Common UV Commands

```bash
# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Sync environment with pyproject.toml
uv sync

# Run any command in the project environment
uv run <command>

# Show installed packages
uv pip list
```

### Environment Variables

Create a `.env` file with your LLM provider credentials:

```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# For other providers, see litellm docs
# https://docs.litellm.ai/docs/providers

# Optional: Override the default model
MODEL_ID=gpt-4o-mini
```

## Architecture

### v0_bash_agent.py

A ~60 line agent demonstrating the core agentic loop pattern:

```
while not done:
    response = model(messages, tools)
    if no tool calls: return
    execute tools, append results
```

**Key Features:**
- Uses litellm for multi-provider LLM support (OpenAI, Anthropic, Azure, etc.)
- Single tool: `bash` - executes any shell command
- Subagent support via self-invocation: `python v0_bash_agent.py "subtask"`
- Interactive REPL mode or one-shot subagent mode

**Usage Modes:**

```bash
# Interactive mode
uv run python v0_bash_agent.py

# Subagent/one-shot mode
uv run python v0_bash_agent.py "find all Python files and count lines"
```

## LiteLLM Model Support

The agent uses litellm which supports 100+ LLM providers with a unified interface. Set `MODEL_ID` to any supported model:

```bash
# OpenAI models
MODEL_ID=gpt-4o
MODEL_ID=gpt-4o-mini
MODEL_ID=o1-preview

# Anthropic models
MODEL_ID=claude-sonnet-4-20250514
MODEL_ID=claude-3-5-haiku-20241022

# Azure OpenAI
MODEL_ID=azure/gpt-4o

# Local models via Ollama
MODEL_ID=ollama/llama3.2

# See full list: https://docs.litellm.ai/docs/providers
```

## Dependencies

Managed via `pyproject.toml`:
- `litellm` - Unified LLM API interface
- `python-dotenv` - Environment variable loading from .env files
