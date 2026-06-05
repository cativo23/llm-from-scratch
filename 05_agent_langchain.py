"""
SCRIPT 05 — THE AGENT, IN LANGCHAIN
================================================================================
The SAME interactive agent as script 02, rebuilt with the LangChain framework.
Read them side by side — that's the point.

What LangChain removes (the wins):
  • the hand-written JSON tool schema  ->  the @tool decorator builds it from the
    function's type hints + docstring. (Compare the big TOOLS_SCHEMA in 02.)
  • json.loads on the arguments        ->  call["args"] arrives already parsed.
  • raw {"role": ...} dicts            ->  typed SystemMessage/HumanMessage/ToolMessage.

What stays the same: the agent loop. No framework can hide it — because that loop
IS the concept. LangChain just hands you prebuilt pieces to fill it with.

A framework is "prebuilt plumbing + a universal adapter": swap models, vector
stores, or tools behind one interface. Convenient — but it churned a lot
historically, and it hides what's happening (matters for auditing in healthcare).

Run it:   python 05_agent_langchain.py
================================================================================
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool                 # the decorator that auto-builds tool schemas
from langchain_openai import ChatOpenAI               # LangChain's wrapper over the OpenAI API
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console

load_dotenv()
console = Console()

MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b:free")   # override via the MODEL env var


# --- Tools: @tool reads the signature + docstring and generates the schema ---
# Compare this to script 02, where we wrote ~15 lines of JSON per tool by hand.
@tool
def get_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%H:%M:%S")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together (exact, arbitrary precision)."""
    return a * b


TOOLS = {"get_time": get_time, "multiply": multiply}

# ChatOpenAI points at OpenRouter via base_url. .bind_tools(...) attaches the
# tools so every call offers them to the model (like tools=... in script 02).
llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
).bind_tools(list(TOOLS.values()))


def take_turn(messages: list) -> None:
    """Agent loop for one user turn — structurally identical to script 02's."""
    while True:
        with console.status("[bold cyan]Thinking…[/]"):
            ai_msg = llm.invoke(messages)                  # .invoke replaces client.chat.completions.create
        messages.append(ai_msg)                            # ai_msg is a typed AIMessage object

        if not ai_msg.tool_calls:                          # no tool -> final answer
            console.print(f"[bold green]Bot[/] {ai_msg.content}")
            return

        for call in ai_msg.tool_calls:                     # call["args"] is already a dict (no json.loads!)
            result = TOOLS[call["name"]].invoke(call["args"])   # a @tool is invoked with .invoke(args)
            console.print(f"  [dim]⚙ {call['name']}({call['args']}) → {result}[/]")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


def main() -> None:
    messages = [SystemMessage(                             # typed message instead of a raw dict
        "You are a concise, friendly assistant with tools. Use them when they help. "
        "Keep answers short and to the point."
    )]
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({"prompt": "bold cyan"}),
    )

    try:
        while True:
            user = session.prompt(
                HTML("<prompt>❯ </prompt>"),
                placeholder=HTML("<ansibrightblack>Ask me anything…  (type 'exit' or press Ctrl+C to quit)</ansibrightblack>"),
            ).strip()
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue
            messages.append(HumanMessage(user))
            take_turn(messages)
    except (KeyboardInterrupt, EOFError):
        pass

    console.print("\n[dim]👋 Bye![/]")


if __name__ == "__main__":
    main()
