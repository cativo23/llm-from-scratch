# LLM From Scratch

> A hands-on, runnable tour of how LLM applications actually work — built from a
> single API call up to an **agentic RAG** assistant, with **no magic**. Then the
> same agent is rebuilt in LangChain so you can see exactly what a framework gives
> you and what it hides.

Most tutorials start with a framework and hide the mechanics. This repo does the
opposite: you build chat, streaming, memory, tool-calling, the agent loop, and
RAG **by hand first** — so you genuinely understand them — and only then look at
LangChain/LangGraph with the judgment to evaluate the trade-offs.

Every script is short, runnable, and **heavily commented for learning**.

---

## What's inside (read in order)

| # | Script | What it teaches |
|---|--------|-----------------|
| 01 | [`01_chat.py`](01_chat.py) | The basics: an LLM is text-in/text-out. Streaming, a live "thinking" indicator (background thread), conversation **memory**, and the **system prompt**. |
| 02 | [`02_agent.py`](02_agent.py) | **Tools** + the **agent loop**: the model calls real Python functions. Decide → run → feed result back → repeat. This is, in miniature, what coding agents do. |
| 03 | [`03_rag.py`](03_rag.py) | **RAG**: answer over your own documents. Embeddings, semantic search via cosine similarity, and a **grounded** prompt that refuses to hallucinate. |
| 04 | [`04_agent_rag.py`](04_agent_rag.py) | **Agentic RAG**: search as a *tool* + memory, so the agent rewrites vague follow-ups ("opens?" → "clinic opening time") before searching. Everything combined. |
| 05 | [`05_agent_langchain.py`](05_agent_langchain.py) | The agent from #02, rebuilt in **LangChain**. See what disappears: hand-written schemas, `json.loads`, raw dicts. |
| 06 | [`06_agent_langgraph.py`](06_agent_langgraph.py) | The entire agent loop in **one line** with LangGraph's `create_agent`. Less code — but now the loop is a black box. |

`01`–`04` are built from scratch (the learning core). `05`–`06` are the framework
versions, for comparison.

---

## Concepts covered

- An LLM is **stateless** — it remembers nothing; *you* carry the history and re-send it.
- **Streaming** token-by-token, with cleanup that survives errors and Ctrl+C.
- **Tool / function calling**: you describe functions in JSON; the model decides when to call them. Good descriptions *are* the steering wheel.
- **The agent loop**: the difference between a chatbot and an agent.
- **Embeddings & semantic search**: meaning lives in a vector's *direction* (hence cosine similarity, not keyword match).
- **Grounding / anti-hallucination**: instruct the model to answer only from retrieved context.
- **Query rewriting** for conversational RAG (why a follow-up like "opens?" breaks plain RAG, and how an agent fixes it).
- **Frameworks**: what LangChain/LangGraph automate, and the cost (less control, more churn, harder to audit).

---

## Setup

Requires **Python 3.11+** and a free [OpenRouter](https://openrouter.ai) API key.

```bash
git clone https://github.com/cativo23/llm-from-scratch.git
cd llm-from-scratch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your OpenRouter key into .env

python 01_chat.py                  # start here, then 02, 03, ...
```

> The scripts default to `openai/gpt-oss-120b:free` — a free, tool-capable model
> on OpenRouter. Free models are rate-limited and occasionally busy; swap the
> `MODEL` line for any model id from <https://openrouter.ai/models>.

---

## Key takeaways

- **There is no magic.** Every LLM app is: build a `messages` list, send it to an API, read the reply, maybe run a tool, repeat.
- **A framework is convenience, not a concept.** Learn the concepts and you can use *any* framework — or none.
- **Trust is a chain.** An agent's answer is only as reliable as the model's decision, its argument-passing, *and* your tool's code. Know where each can fail.

---

## Where this could go next

Turning this into a production-grade service would add: a [FastAPI](https://fastapi.tiangolo.com/)
wrapper, a real vector database (pgvector / Qdrant), document chunking & PDF
loading, an **evaluation harness** (measure retrieval and answer quality),
guardrails (PII redaction, source citations), and tracing/cost observability.

---

## License

[MIT](LICENSE) © Carlos Cativo

*Built as a learning project — clear over clever, comments over cleverness.*
