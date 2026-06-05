"""
SCRIPT 02 — AGENT (tools)
================================================================================
The leap from chatbot to AGENT. Here the model can call REAL Python functions.

What you'll learn here:
  • A "tool" = a function you let the model call. You DESCRIBE it in JSON; the
    model never sees your code, only the description — so good descriptions are
    everything (this is prompt engineering).
  • The agent loop: model decides → your code runs the tool → result goes back →
    model continues. Repeat until it has a final answer.
  • Three message roles now: "user", "assistant" (which may request tools), and
    "tool" (your function's output, tied to the request by tool_call_id).
  • Two nested loops: OUTER = conversation turns, INNER = the agent loop per turn.

This is, in miniature, exactly what Claude Code does (its tools are Bash/Read/Edit).

Run it:   python 02_agent.py
================================================================================
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b:free")   # override via the MODEL env var
console = Console()


# --- 1. The real functions the model is allowed to call ---------------------
# Pick things the model CAN'T do reliably on its own: it doesn't know the current
# time, and it's bad at exact arithmetic. Tools give it those abilities.
def get_time() -> str:
    """Return the current local time."""
    return datetime.now().strftime("%H:%M:%S")


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# Maps the tool NAME (the string the model says) -> the actual function to run.
AVAILABLE_TOOLS = {
    "get_time": get_time,
    "multiply": multiply,
}

# --- 2. The SCHEMA: how we describe those tools to the model ----------------
# The model only ever sees this JSON — names, descriptions, and argument types.
# It decides which tool to call based purely on these words.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time.",
            "parameters": {"type": "object", "properties": {}},   # takes no arguments
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together.",
            "parameters": {                                        # describes the arguments
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "first number"},
                    "b": {"type": "number", "description": "second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


# --- 3. The agent loop ------------------------------------------------------
def take_turn(messages: list) -> None:
    """Handle ONE user turn: let the model call tools until it has a final answer.

    `messages` is passed in and mutated in place, so everything (the model's
    decisions AND the tool results) is recorded in the shared conversation.
    """
    while True:
        with console.status("[bold cyan]Thinking…[/]"):       # simple spinner (no streaming here)
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,                            # <- tell the model which tools exist
            )
        msg = response.choices[0].message
        messages.append(msg)                                  # record what the model decided

        # If the model didn't ask for a tool, its message IS the final answer.
        if not msg.tool_calls:
            console.print(f"[bold green]Bot[/] {msg.content}")
            return

        # Otherwise it requested one or more tools. Run each, then loop again so
        # the model can see the results and continue (or answer).
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)        # arguments arrive as a JSON string
            result = AVAILABLE_TOOLS[name](**args)            # run the REAL function
            console.print(f"  [dim]⚙ {name}({args}) → {result}[/]")
            messages.append({
                "role": "tool",                               # the new role: a tool's output
                "tool_call_id": call.id,                      # ties this result to the exact request
                "content": str(result),
            })
        # back to the top: the model now sees the tool result(s)


def main() -> None:
    messages = [
        {"role": "system", "content":
            "You are a concise, friendly assistant with tools. Use them when they help. "
            "Keep answers short and to the point."}
    ]
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({"prompt": "bold cyan"}),
    )

    try:
        while True:                                           # OUTER loop: one pass per user turn
            user = session.prompt(
                HTML("<prompt>❯ </prompt>"),
                placeholder=HTML("<ansibrightblack>Ask me anything…  (type 'exit' or press Ctrl+C to quit)</ansibrightblack>"),
            ).strip()
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue
            messages.append({"role": "user", "content": user})
            take_turn(messages)                               # INNER loop lives in here
    except (KeyboardInterrupt, EOFError):
        pass

    console.print("\n[dim]👋 Bye![/]")


if __name__ == "__main__":
    main()
