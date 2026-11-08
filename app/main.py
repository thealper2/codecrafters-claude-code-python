import argparse
import os
import sys
import json
import subprocess
import re

from openai import OpenAI
from app.lsp_client import LSPClient
from pathlib import Path
from urllib.parse import urlparse, unquote

API_KEY = os.getenv("OPENROUTER_API_KEY")
CLIENT = None
LSP = None
LSP_OPENED = set()
LSP_TOOL = {
    "type": "function",
    "function": {
        "name": "LSP",
        "description": "Finds where a symbol is defined (goToDefinition) or all places it is used (findReferences). Line and character are 1-based.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["goToDefinition", "findReferences"]},
                "filePath": {
                    "type": "string",
                    "description": "Absolute path, or a path relative to the agent's current working directory.",
                },
                "line": {"type": "integer", "minimum": 1, "description": "1-based line number of the symbol."},
                "character": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based character offset in UTF-16 code units (same as character count for ASCII).",
                },
            },
            "required": ["operation", "filePath", "line", "character"],
        },
    },
}
MODEL = "anthropic/claude-haiku-4.5"
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },  
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "Load a skill's instructions into the conversation",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "The name of the skill to use"},
                    "args": {"type": "string", "description": "Optional arguments for the skill"},
                },
            },
        },
    },
]
SKILLS_DIR = os.path.join(".claude", "skills")


def parse_skill(path):
    with open(path) as f:
        text = f.read()
    meta = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip("\"'")
            body = text[end + 4:].lstrip("\n")
    return meta, body


def load_skills():
    skills = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for name in sorted(os.listdir(SKILLS_DIR)):
        path = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if os.path.isfile(path):
            meta, body = parse_skill(path)
            skills[name] = {"meta": meta, "body": body}
    return skills


def lsp_open(path):
    uri = Path(path).as_uri()
    if uri not in LSP_OPENED:
        with open(path) as f:
            text = f.read()
        ext = os.path.splitext(path)[1]
        LSP.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": LSP.extension_to_language.get(ext, ""),
                "version": 1,
                "text": text,
            }
        })
        LSP_OPENED.add(uri)
    return uri


def uri_to_rel(uri):
    return os.path.relpath(unquote(urlparse(uri).path))


def lsp_references(a):
    uri = lsp_open(os.path.abspath(a["filePath"]))
    res = LSP.request("textDocument/references", {
        "textDocument": {"uri": uri},
        "position": {"line": a["line"] - 1, "character": a["character"] - 1},
        "context": {"includeDeclaration": True},
    })
    if isinstance(res, dict) and "result" in res:
        res = res["result"]
    if not res:
        return "No references found"
    groups = {}
    for loc in res:
        s = loc["range"]["start"]
        groups.setdefault(uri_to_rel(loc["uri"]), []).append(
            "  Line " + str(s["line"] + 1) + ":" + str(s["character"] + 1)
        )
    out = ["Found " + str(len(res)) + " references across " + str(len(groups)) + " files:"]
    for f, lines in groups.items():
        out.append("")
        out.append(f + ":")
        out.extend(lines)
    return "\n".join(out)


def lsp_definition(a):
    uri = lsp_open(os.path.abspath(a["filePath"]))
    res = LSP.request("textDocument/definition", {
        "textDocument": {"uri": uri},
        "position": {"line": a["line"] - 1, "character": a["character"] - 1},
    })
    if isinstance(res, dict) and "result" in res:
        res = res["result"]
    if isinstance(res, list):
        res = res[0] if res else None
    if not res:
        return "No definition found"
    target = res.get("uri") or res.get("targetUri")
    rng = res.get("range") or res.get("targetSelectionRange")
    start = rng["start"]
    rel = uri_to_rel(target)
    return "Defined in " + rel + ":" + str(start["line"] + 1) + ":" + str(start["character"] + 1)

def execute_tool(name, arguments):
    a = json.loads(arguments)
    if name == "Read":
        with open(a["file_path"]) as f:
            return f.read()
    if name == "Write":
        d = os.path.dirname(a["file_path"])
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a["file_path"], "w") as f:
            f.write(a["content"])
        return "Wrote " + a["file_path"]
    if name == "Bash":
        r = subprocess.run(a["command"], shell=True, capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            out += "\nExit code: " + str(r.returncode)
        return out or "(no output)"
    if name == "Skill":
        skills = load_skills()
        sname = a["name"].lstrip("/")
        if sname not in skills:
            return "Unknown skill: " + sname
        folder = os.path.join(SKILLS_DIR, sname)
        header = (
            "Skill: " + sname + " (located at " + folder + ")\n"
            "Paths in the instructions below are relative to that folder.\n\n"
        )
        body = header + fill_skill(skills[sname]["body"].strip(), a.get("args", "") or "")
        if skills[sname]["meta"].get("context") == "fork":
            answer = run_agent([{"role": "user", "content": body}])
            return "Skill " + sname + " ran in a separate context and returned: " + (answer or "")
        return body
    if name == "LSP":
        if a.get("operation") == "goToDefinition":
            return lsp_definition(a)
        if a.get("operation") == "findReferences":
            return lsp_references(a)
        return "Unsupported LSP operation: " + str(a.get("operation"))
    return "Unknown tool: " + name


def fill_skill(body, rest):
    argv = rest.split()
    pick = lambda m: argv[int(m.group(1))] if int(m.group(1)) < len(argv) else ""
    has_ph = re.search(r"\$ARGUMENTS|\$\d+", body)
    body = re.sub(r"\$ARGUMENTS\[(\d+)\]", pick, body)
    body = body.replace("$ARGUMENTS", rest)
    body = re.sub(r"\$(\d+)", pick, body)
    if not has_ph and rest:
        body += "\n\n" + rest
    return body


def run_agent(messages):
    while True:
        chat = CLIENT.chat.completions.create(
            model=MODEL, 
            messages=messages, 
            tools=TOOLS + ([LSP_TOOL] if LSP else [])
        )
        if not chat.choices:
            raise RuntimeError("no choices in response")
        msg = chat.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in msg.tool_calls
            ],
        })
        for c in msg.tool_calls:
            try:
                result = execute_tool(c.function.name, c.function.arguments)
            except Exception as e:
                result = "Error: " + type(e).__name__ + ": " + str(e)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    global CLIENT, LSP
    CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    LSP = LSPClient.from_environment()
    if LSP:
        print("LSP client connected", file=sys.stderr)
    
    skills = load_skills()
    messages = []
    
    if skills:
        lines = ["- " + n + ": " + s["meta"].get("description", "") for n, s in skills.items()]
        messages.append({
            "role": "system",
            "content": "You have access to the following skills:\n\n" + "\n".join(lines)
            + "\n\nIf a skill matches the user's request, call the Skill tool with its name"
            " and follow the instructions it returns.",
        })
    
    tokens = args.p.split()
    invoked = []
    i = 0
    while i < len(tokens) and tokens[i].startswith("/") and tokens[i][1:] in skills:
        invoked.append(tokens[i][1:])
        i += 1
    if invoked:
        rest = " ".join(tokens[i:])
        for name in invoked:
            folder = os.path.join(SKILLS_DIR, name)
            header = (
                "Skill: " + name + " (located at " + folder + ")\n"
                "Paths in the instructions below are relative to that folder.\n\n"
            )
            body = fill_skill(skills[name]["body"].strip(), rest)
            messages.append({"role": "user", "content": header + body})
    else:
        messages.append({"role": "user", "content": args.p})
    
    print(run_agent(messages))


if __name__ == "__main__":
    main()
