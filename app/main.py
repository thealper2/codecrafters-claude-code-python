import argparse
import os
import sys
import json
import subprocess
import re

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    skills = load_skills()
    messages = []
    
    if skills:
        lines = ["- " + n + ": " + s["meta"].get("description", "") for n, s in skills.items()]
        messages.append({
            "role": "system",
            "content": "You have access to the following skills:\n\n" + "\n".join(lines),
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
    
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )
        if not chat.choices:
            raise RuntimeError("no choices in response")
        msg = chat.choices[0].message
        if not msg.tool_calls:
            print(msg.content)
            break
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
            messages.append({
                "role": "tool",
                "tool_call_id": c.id,
                "content": execute_tool(c.function.name, c.function.arguments),
            })


if __name__ == "__main__":
    main()
