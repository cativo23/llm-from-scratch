"""
SCRIPT 01 — CHAT
================================================================================
The foundation. A conversational chat client over an LLM, built from scratch.

What you'll learn here:
  • An LLM is just a function: text goes in, text comes out. Nothing more.
  • Streaming: printing the reply token-by-token as it's generated.
  • A live "thinking" indicator (the spinner) using a background thread.
  • Memory: an LLM remembers NOTHING on its own — YOU keep the history and
    re-send it every turn. The growing `messages` list IS the memory.
  • The system prompt: a special first message that sets the assistant's behavior.

Run it:   python 01_chat.py
================================================================================
"""

import itertools
import os
import sys
import threading
import time

from dotenv import load_dotenv               # reads the .env file into environment variables
from openai import OpenAI                     # the client; OpenRouter speaks the OpenAI dialect
from prompt_toolkit import PromptSession      # a real CLI input box (history, suggestions)
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console              # pretty terminal output (colors, the spinner)

load_dotenv()                                 # so os.environ can see OPENROUTER_API_KEY

# OpenRouter is a single gateway to hundreds of models. It mimics the OpenAI API,
# so the OpenAI client works against it just by changing the base_url.
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-120b:free"            # free, reliable tool calling + number handling

# Purely cosmetic: the words the spinner rotates through while we wait.
THINKING_WORDS = [
    "Flibbertigibbeting", "Cogitating", "Percolating", "Ruminating",
    "Noodling", "Conjuring", "Marinating", "Pondering",
]

console = Console()


def create_client() -> OpenAI:
    """Create the OpenRouter client, failing early and clearly if the key is missing.

    Reading the key from the environment (never hard-coding it) keeps secrets out
    of the source. We use .get() + a friendly message instead of letting Python
    throw a cryptic KeyError.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Missing OPENROUTER_API_KEY.[/] "
            "Set it in your [bold].env[/] file."
        )
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def _drive_indicator(status, stop_event: threading.Event, start: float) -> None:
    """Animate the indicator's TEXT (rotating word + elapsed seconds) until told to stop.

    Why a separate thread? The main thread is BLOCKED waiting on the network call
    to the model — it can't update the screen while it waits. So this runs in
    parallel, refreshing the text, until the main thread signals `stop_event`.
    """
    words = itertools.cycle(THINKING_WORDS)   # endless loop over the word list
    word = next(words)
    last_swap = start
    while not stop_event.is_set():            # keep going until the main thread says stop
        now = time.monotonic()
        if now - last_swap >= 1.5:            # change the word every 1.5 seconds
            word = next(words)
            last_swap = now
        status.update(f"[magenta]✻[/] [bold]{word}…[/] [dim]({now - start:.0f}s)[/]")
        stop_event.wait(0.1)                  # sleep 0.1s, but wake instantly if signaled


def answer(client: OpenAI, messages: list) -> str:
    """Send the whole conversation, stream the reply to screen, and RETURN its text.

    Returning the text matters: main() appends it to `messages` so the model can
    "remember" what it said on later turns.
    """
    start = time.monotonic()

    # Start the spinner BEFORE the request. create() blocks while the connection
    # opens and the model warms up, so the spinner must already be running to
    # cover that initial wait (otherwise you'd see a frozen cursor first).
    status = console.status("")
    status.start()
    spinner_active = True
    stop_event = threading.Event()
    driver = threading.Thread(target=_drive_indicator, args=(status, stop_event, start), daemon=True)
    driver.start()

    tokens = None
    full_reply = []                           # collected here so the return is always safe
    try:
        # stream=True turns the reply into a sequence of small "chunks" we loop over,
        # instead of waiting for the whole thing. include_usage asks OpenRouter to
        # append a final chunk with token counts.
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:                   # the final stats chunk (carries no text)
                tokens = chunk.usage.completion_tokens
            if not chunk.choices:             # some chunks have no content — skip them
                continue
            text = chunk.choices[0].delta.content   # NOTE: .delta (a piece), not .message
            if not text:
                continue
            if spinner_active:                # first real token arrived -> kill the spinner
                stop_event.set()
                driver.join()                 # wait for the thread to actually stop
                status.stop()
                spinner_active = False
            full_reply.append(text)
            print(text, end="", flush=True)   # end="" = no newline between pieces; flush = show now
    finally:
        # Guarantee cleanup even if the request throws — no spinner left spinning forever.
        stop_event.set()
        if spinner_active:
            status.stop()

    # A past-tense summary line, à la "Cogitated for 46s".
    elapsed = time.monotonic() - start
    summary = f"\n[dim]✦ done in {elapsed:.0f}s"
    if tokens is not None:
        summary += f" · {tokens} tokens"
    summary += "[/]"
    console.print(summary)

    return "".join(full_reply)                # the full text, for main() to store in memory


def main() -> None:
    client = create_client()

    # `messages` is the memory. It starts with ONE special message:
    #   role "system" = instructions that shape how the assistant behaves.
    # Every turn we append the user's message and the assistant's reply, and
    # re-send the whole list — that's the entire trick behind "remembering".
    messages = [
        {"role": "system", "content":
            "You are a concise, friendly assistant. Answer in a few sentences. "
            "Do not lecture or add long disclaimers. Get to the point."}
    ]

    # A real CLI input: ↑/↓ history, greyed-out auto-suggestions, a styled prompt.
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=Style.from_dict({"prompt": "bold cyan"}),
    )

    try:
        while True:
            question = session.prompt(
                HTML("<prompt>❯ </prompt>"),
                placeholder=HTML("<ansibrightblack>Ask me anything…  (type 'exit' or press Ctrl+C to quit)</ansibrightblack>"),
            ).strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:                  # ignore empty input (just Enter)
                continue
            messages.append({"role": "user", "content": question})   # remember what you said
            reply = answer(client, messages)
            messages.append({"role": "assistant", "content": reply})  # remember what it said
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C raises KeyboardInterrupt; Ctrl+D / closed input raises EOFError.
        # Catching both turns an ugly traceback into a clean exit.
        pass

    console.print("\n[dim]👋 Bye![/]")


if __name__ == "__main__":
    main()
