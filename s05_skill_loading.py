"""
s05_skill_loading.py - Skills

Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - git: Git workflow helpers        |  <-- Layer 1: metadata only
    |   - test: Testing best practices     |
    +--------------------------------------+

    When model calls load_skill("git"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full git workflow instructions...  |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

Key insight: "Don't put everything in the system prompt. Load on demand."
"""

import json
import os
import re
import subprocess
from pathlib import Path
import pprint

import litellm
from dotenv import load_dotenv

from utils import fn_to_tool

load_dotenv(override=True)

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"


# -- SkillLoader: parse .skills/*.md files with YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        print(f"Loading skills from {self.skills_dir}...")
        for f in sorted(self.skills_dir.rglob('SKILL.md')):
            name = f.parent.name
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(Path(SKILLS_DIR))

# Layer 1: skill metadata injected into system prompt
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


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

def load_skill(name: str) -> str:
    """Load specialized knowledge by name."""
    return SKILL_LOADER.get_content(name)


def execute_tool(name: str, args: dict) -> str:
    handlers = {
        "bash": lambda a: run_bash(a["command"]),
        "read_file": lambda a: run_read(a["path"], a.get("limit")),
        "write_file": lambda a: run_write(a["path"], a["content"]),
        "edit_file": lambda a: run_edit(a["path"], a["old_text"], a["new_text"]),
        "load_skill": lambda a: load_skill(a["name"]),
    }
    handler = handlers.get(name)
    return handler(a=args) if handler else f"Unknown tool: {name}"


TOOLS = [
    fn_to_tool(run_bash, "bash"),
    fn_to_tool(run_read, "read_file"),
    fn_to_tool(run_write, "write_file"),
    fn_to_tool(run_edit, "edit_file"),
    fn_to_tool(load_skill, "load_skill"),
]


def agent_loop(messages: list):
    while True:
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

        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else (fn.arguments or {})
            try:
                output = execute_tool(name, args)
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {name}: {str(output)[:200]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    query = "What skills are available?"
    history.append({"role": "user", "content": query})
    agent_loop(history)

    for h in history:
        pprint.pprint(h)
