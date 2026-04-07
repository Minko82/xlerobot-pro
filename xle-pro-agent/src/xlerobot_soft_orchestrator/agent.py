"""LangGraph ReAct agent for robot manipulation tasks.

Architecture:
  User directive → reason → act → observe → reason → ... → respond

The agent runs a ReAct (Reason + Act) loop:
  1. reason   — LLM thinks about what to do next, chooses a capability
  2. act      — execute the chosen capability via the registry
  3. The result feeds back into reason for the next step

The loop terminates when the LLM decides the task is done or max_iterations
is reached.  Every step is recorded in the trace for the UI.

Capability discovery pattern:
  The system prompt tells the agent to ALWAYS start by calling
  list_capabilities, then read_capability for any capability it plans
  to use, then run_capability to execute.  This keeps the prompt stable
  while capability docs evolve independently.
"""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, AsyncGenerator

from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .capability import TOOLS
from .config import Settings, configure_langsmith, get_settings


# ---------------------------------------------------------------------------
# Protocols & shared types
# ---------------------------------------------------------------------------

class LLMLike(Protocol):
    def invoke(self, input_data: Any) -> Any: ...


class TraceEvent(TypedDict):
    step: int
    stage: str
    capability: str
    reasoning: str
    args: dict[str, Any]
    result_summary: str


class AgentState(TypedDict):
    directive: str
    messages: Annotated[list[dict[str, str]], operator.add]   # running conversation
    trace: Annotated[list[TraceEvent], operator.add]
    step: int
    done: bool
    final_response: str


@dataclass
class AgentRunResult:
    final_response: str
    trace: list[TraceEvent]
    steps_taken: int
    graph_mermaid: str


# ---------------------------------------------------------------------------
# System prompt  (stable — capability docs live in capabilities.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a robot manipulation agent controlling an XLErobot arm.
Your job is to translate a high-level human directive into a sequence of robot
capability calls that accomplish the task.

=== HOW TO WORK ===
You operate in a ReAct loop: Reason → Act → Observe → Reason → ...

Before calling ANY capability you must:
1. Call list_capabilities to see what is available.
2. Call read_capability("<id>") for each capability you plan to use.
3. Then call run_capability("<id>", {args}) to execute it.

=== RESPONSE FORMAT ===
You can call tools to perform actions.
When you need to take an action, call the appropriate tool.
Do not output JSON. Use tool calls instead.

Use capability_id "DONE" when the task is fully complete.
The "args" field:
  - For list_capabilities: {}
  - For read_capability: {"capability_id": "<id>"}
  - For run_capability:   {"capability_id": "<id>", "args": {<capability args>}}
  - For DONE:            {"summary": "<what was accomplished>"}

=== SAFETY RULES ===
- Always call get_observation before any arm movement.
- Never invent object positions — use values from get_observation results.
- If a capability returns an error, read its doc again and correct your args.
- Do not loop more than {{MAX_ITERATIONS}} times.
"""

tool_node = ToolNode(TOOLS)

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "END"

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reason_node(state: AgentState, llm: LLMLike, max_iterations: int) -> dict[str, Any]:
    system = SYSTEM_PROMPT.replace("{{MAX_ITERATIONS}}", str(max_iterations))
    messages = [SystemMessage(content=system)] + state["messages"]

    response = llm.invoke(messages)

    tool_calls = getattr(response, "tool_calls", [])
    trace_event = TraceEvent(
        step=state["step"] + 1,
        stage="reason",
        capability=tool_calls[0]["name"] if tool_calls else "final_answer",
        reasoning=str(response.content),
        args=tool_calls[0]["args"] if tool_calls else {},
        result_summary=""
    )

    return {
        "messages": [response],
        "trace": [trace_event],
        "step": state["step"] + 1,
    }

# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def get_vertex_llm(settings: Settings, model_name: str | None = None) -> ChatVertexAI:
    return ChatVertexAI(
        model_name=model_name or settings.vertex_model,
        project=settings.vertex_project_id,
        location=settings.vertex_location,
        temperature=0,
    )

def get_ollama_llm(settings: Settings, model_name: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

def get_llm() -> LLMLike:
    settings = get_settings()
    configure_langsmith(settings)

    provider = settings.llm_provider.strip().lower()
    model_name = settings.llm_model.strip() or None
    if provider == "vertex":
        model = get_vertex_llm(settings, model_name)
    elif provider == "ollama":
        model = get_ollama_llm(settings, model_name)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'vertex' or 'ollama'.")

    return model.bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(llm: LLMLike, max_iterations: int = 10):
    workflow = StateGraph(AgentState)
    workflow.add_node("reason", lambda s: reason_node(s, llm, max_iterations))
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "reason")
    workflow.add_edge("reason", "tools")
    workflow.add_edge("tools", "reason")
    workflow.add_conditional_edges(
        "reason",
        should_continue,
        {
            "tools": "tools",
            "END": END
        }
    )
    return workflow.compile()

async def astream_directive(directive: str) -> AsyncGenerator[dict[str, Any], None]:
    settings = get_settings()
    model = get_llm()
    graph = build_graph(model, settings.max_iterations)
    
    initial_state = {
        "directive": directive,
        "messages": [{"role": "user", "content": directive}],
        "trace": [],
        "step": 0,
        "done": False,
        "final_response": ""
    }
    
    async for update in graph.astream(initial_state):
        yield update

def run_directive(directive: str) -> AgentRunResult:
    # Legacy sync support
    return asyncio.run(_run_async_wrapper(directive))

async def _run_async_wrapper(directive: str):
    model = get_llm()
    settings = get_settings()
    graph = build_graph(model, settings.max_iterations)
    res = await graph.ainvoke({
        "directive": directive,
        "messages": [{"role": "user", "content": directive}],
        "trace": [],
        "step": 0,
        "done": False,
        "final_response": ""
    })
    return AgentRunResult(
        final_response=res.get("final_response", ""),
        trace=res.get("trace", []),
        steps_taken=res.get("step", 0),
        graph_mermaid=""
    )
