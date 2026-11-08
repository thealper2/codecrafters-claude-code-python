import argparse
import os
import sys
import json

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
    }
]

def execute_tool(name, arguments):
    a = json.loads(arguments)
    if name == "Read":
        with open(a["file_path"]) as f:
            return f.read()
    return "Unknown tool: " + name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    messages = [{"role": "user", "content": args.p}]
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
