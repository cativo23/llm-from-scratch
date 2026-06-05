"""
SCRIPT 03 — RAG (Retrieval-Augmented Generation)
================================================================================
Make the model answer over YOUR documents, not just its training data.

The idea in one line: before answering, SEARCH your documents for the relevant
bits, paste them into the prompt, and tell the model to answer only from them.

What you'll learn here:
  • Embeddings: turning text into a vector (a list of numbers) that captures
    MEANING. Similar meaning -> vectors that point the same direction.
  • Semantic search: find relevant text by comparing vector directions
    (cosine similarity), NOT by matching keywords.
  • Grounding: instructing the model to answer ONLY from the retrieved context
    and to say "I don't know" otherwise — so it can't hallucinate. Critical in
    healthcare and anywhere wrong answers are dangerous.

Pipeline:  DOCS -> embed once -> store vectors
           query -> embed -> cosine vs every doc -> top-k -> stuff into prompt -> answer

Note: this script has NO memory — each question is independent. That's a real
limitation (try a follow-up like "opens?"); script 04 fixes it.

Run it:   python 03_rag.py
================================================================================
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding          # runs a small embedding model LOCALLY (free, no API)
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console

load_dotenv()
console = Console()

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-120b:free"

# --- Our tiny knowledge base (pretend these are Blue Medical's docs) --------
DOCS = [
    "Password reset: go to Settings > Security > Reset Password. The emailed link expires in 15 minutes.",
    "Clinic hours: Monday to Friday 8am-5pm. Closed weekends and national holidays.",
    "To book an appointment, use the patient portal under 'Appointments'. Same-day slots open at 7am.",
    "Lab results appear in the patient portal within 48 hours. Critical results are phoned by a nurse.",
    "Billing questions: the finance office is at extension 220, Monday to Thursday 9am-4pm.",
]

# Embed every document ONCE, up front (this is the cheap, do-it-ahead-of-time step).
# DOC_VECTORS[i] is the vector for DOCS[i] — they stay aligned by index.
embedder = TextEmbedding()                    # downloads a ~small model on first run
DOC_VECTORS = list(embedder.embed(DOCS))


def create_client() -> OpenAI:
    """Create the OpenRouter client, validating that the API key exists."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Missing OPENROUTER_API_KEY.[/] Set it in your [bold].env[/] file."
        )
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def cosine(a, b) -> float:
    """Cosine similarity: how aligned two vectors are, ignoring their length.

    = dot(a, b) / (|a| * |b|). Returns ~1.0 for same meaning, ~0 for unrelated.
    We compare DIRECTION (not distance) because meaning lives in a vector's
    direction; its length is mostly noise (text size, word frequency).
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, k: int = 2) -> list[tuple[float, str]]:
    """Return the k documents closest in meaning to `query`, as (score, doc) pairs."""
    q_vec = list(embedder.embed([query]))[0]                  # embed the question the same way
    scored = [(cosine(q_vec, vec), doc) for vec, doc in zip(DOC_VECTORS, DOCS)]
    scored.sort(key=lambda pair: pair[0], reverse=True)       # highest similarity first
    return scored[:k]


def build_prompt(query: str, context: list[str]) -> list[dict]:
    """Assemble the messages, injecting the retrieved context and GROUNDING the model.

    The whole point of RAG lives in this system prompt: we paste the retrieved
    passages in and forbid the model from using anything else. That's what keeps
    it honest ("I don't know") instead of inventing answers.
    """
    context_block = "\n".join(f"- {line}" for line in context)
    system = (
        "You are a precise assistant for Blue Medical. Answer the question using ONLY "
        "the context below. If the answer is not in the context, say you don't know — "
        "never guess or use outside knowledge.\n\n"
        f"Context:\n{context_block}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def main() -> None:
    client = create_client()
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({"prompt": "bold cyan"}),
    )

    try:
        while True:
            query = session.prompt(
                HTML("<prompt>❯ </prompt>"),
                placeholder=HTML("<ansibrightblack>Ask about Blue Medical…  (type 'exit' or press Ctrl+C to quit)</ansibrightblack>"),
            ).strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            results = retrieve(query)
            for score, doc in results:                        # show WHAT was retrieved and how close
                console.print(f"  [dim]🔎 {score:.2f}  {doc}[/]")

            # Note: we send build_prompt(...) fresh each time — no history is kept.
            with console.status("[bold cyan]Thinking…[/]"):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=build_prompt(query, [doc for _, doc in results]),
                )
            console.print(f"[bold green]Bot[/] {resp.choices[0].message.content}")
    except (KeyboardInterrupt, EOFError):
        pass

    console.print("\n[dim]👋 Bye![/]")


if __name__ == "__main__":
    main()
