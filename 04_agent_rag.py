"""
SCRIPT 04 — AGENTIC RAG (everything combined)
================================================================================
The capstone: an AGENT (memory + tools) whose tool is SEARCHING your documents.

This fuses script 02 (agent loop + memory) and script 03 (RAG). The difference
from plain RAG is huge:

  • Plain RAG (03) always searches with your literal query and has no memory.
    A follow-up like "opens?" searches for "opens?" and grabs the wrong doc.
  • Agentic RAG (this) gives the model MEMORY and makes search a TOOL it controls.
    So for "opens?" it can rewrite the query to "clinic opening time" using the
    conversation, THEN search — and get the right answer.

The model decides WHEN to search and WITH WHAT. That's the "agentic" part, and
it's the architecture of a real conversational assistant.

Run it:   python 04_agent_rag.py
================================================================================
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console

load_dotenv()
console = Console()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b:free")   # override via the MODEL env var

DOCS = [
    "Password reset: go to Settings > Security > Reset Password. The emailed link expires in 15 minutes.",
    "Clinic hours: Monday to Friday 8am-5pm. Closed weekends and national holidays.",
    "To book an appointment, use the patient portal under 'Appointments'. Same-day slots open at 7am.",
    "Lab results appear in the patient portal within 48 hours. Critical results are phoned by a nurse.",
    "Billing questions: the finance office is at extension 220, Monday to Thursday 9am-4pm.",
]

embedder = TextEmbedding()
DOC_VECTORS = list(embedder.embed(DOCS))


def cosine(a, b) -> float:
    """Cosine similarity between two vectors (closer to 1.0 = more similar meaning)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- RAG retrieval, but now packaged as a TOOL the model can call -----------
def search_docs(query: str) -> str:
    """Search Blue Medical's knowledge base and return the most relevant passages.

    Same cosine search as script 03 — the difference is the MODEL invokes this,
    choosing the `query` text itself (so it can reformulate vague follow-ups).
    """
    q_vec = list(embedder.embed([query]))[0]
    scored = sorted(
        ((cosine(q_vec, vec), doc) for vec, doc in zip(DOC_VECTORS, DOCS)),
        key=lambda pair: pair[0],
        reverse=True,
    )[:2]
    for score, doc in scored:
        console.print(f"  [dim]🔎 search('{query}')  {score:.2f}  {doc}[/]")
    return "\n".join(doc for _, doc in scored)


AVAILABLE_TOOLS = {"search_docs": search_docs}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search Blue Medical's knowledge base. Use this for ANY question about the "
                "clinic, hours, appointments, billing, lab results, or passwords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        # This hint nudges the model to resolve "opens?" -> a full query first.
                        "description": "a focused, standalone search query (resolve pronouns/follow-ups first)",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def take_turn(messages: list) -> None:
    """Agent loop for one turn: the model searches as needed, then answers."""
    while True:
        with console.status("[bold cyan]Thinking…[/]"):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
            )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:                                # no search needed -> final answer
            console.print(f"[bold green]Bot[/] {msg.content}")
            return

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = AVAILABLE_TOOLS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # loop: the model now has the search results and can answer


def main() -> None:
    # The system prompt does two jobs: force a search before answering (grounding),
    # and tell the model to rewrite short follow-ups using the conversation so far.
    messages = [{"role": "system", "content": (
        "You are an assistant for Blue Medical. For any question about the clinic, ALWAYS "
        "use the search_docs tool before answering. Answer ONLY from the search results; if "
        "the answer isn't there, say you don't know. When the user sends a short follow-up "
        "(like 'opens?'), first rewrite it into a complete, standalone search query using the "
        "conversation so far, THEN search."
    )}]
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({"prompt": "bold cyan"}),
    )

    try:
        while True:
            user = session.prompt(
                HTML("<prompt>❯ </prompt>"),
                placeholder=HTML("<ansibrightblack>Ask about Blue Medical…  (type 'exit' or press Ctrl+C to quit)</ansibrightblack>"),
            ).strip()
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue
            messages.append({"role": "user", "content": user})   # memory: lets follow-ups work
            take_turn(messages)
    except (KeyboardInterrupt, EOFError):
        pass

    console.print("\n[dim]👋 Bye![/]")


if __name__ == "__main__":
    main()
