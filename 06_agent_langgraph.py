"""
SCRIPT 06 — THE AGENT LOOP IN ONE LINE (create_agent)
================================================================================
The highest level of abstraction: LangGraph's prebuilt agent.

The entire `take_turn` loop you wrote by hand (invoke -> check tool_calls -> run
tool -> feed result back -> repeat) is exactly what `create_agent` builds for you.
One function call.

  • LangChain  = the components (models, tools, messages).  -> Lego bricks.
  • create_agent / LangGraph = the orchestration engine that wires them into a
    runnable, stateful agent (loops, branches, persistence, pauses). -> the
    blueprint tool. (create_agent is built ON TOP of LangGraph.)

The trade-off: far less code, but the loop is now a black box. When the agent
misbehaves you're debugging inside the framework — which is why building it by
hand first (scripts 02 & 04) matters: you know what this is hiding.

This script is a minimal one-shot demo (no chat loop) to keep the "one line"
point front and center. For multi-turn memory you'd give the agent a
checkpointer — but that's the framework's job to manage, not yours to write.

Run it:   python 06_agent_langgraph.py
================================================================================
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent          # NOTE: moved here in LangChain 1.x
from langchain_core.tools import tool               # (was langgraph.prebuilt.create_react_agent)
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%H:%M:%S")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together (exact, arbitrary precision)."""
    return a * b


llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# This single line replaces the whole take_turn() loop from scripts 02/04/05.
agent = create_agent(llm, [get_time, multiply])


if __name__ == "__main__":
    # You hand it a list of messages; it runs the full loop internally and returns
    # the final state. The answer is the last message.
    result = agent.invoke({"messages": [("user", "What is 347 times 891?")]})
    print(result["messages"][-1].content)
