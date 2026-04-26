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
from typing import Annotated, Any, Protocol, TypedDict, AsyncGenerator, Callable

from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .capability import TOOLS
from .config import Settings, configure_langsmith, get_settings

_FRAME_B64_KEYS = ("frame_b64", "base_frame_b64", "wrist_frame_b64")

def _strip_frame_b64_from_tool_messages(messages: list) -> list:
    """Remove all *_frame_b64 keys from tool results before they re-enter the context window.

    Base64 image strings can be 300k+ tokens and are only needed by the UI renderer.
    """
    import json
    cleaned = []
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            try:
                payload = json.loads(msg.content)
                if any(k in payload for k in _FRAME_B64_KEYS):
                    for k in _FRAME_B64_KEYS:
                        if k in payload:
                            payload[k] = "<stripped>"
                    from langchain_core.messages import ToolMessage
                    msg = ToolMessage(
                        content=json.dumps(payload),
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
            except (json.JSONDecodeError, TypeError):
                pass
        cleaned.append(msg)
    return cleaned


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
Before calling any tool, include a brief explanation of your reasoning in plain text. This must be included in the message content.
Do not output JSON. Use tool calls instead.

The "args" field:
  - For list_capabilities: {}
  - For read_capability: {"capability_id": "<id>"}
  - For run_capability:   {"capability_id": "<id>", "params": {<capability args>}}
  - If your model uses "kwargs" or "params" wrappers for tool inputs, that is
    also accepted and will be normalized.

When the task is complete, stop calling tools and provide a concise final
summary in plain text describing success or failure.

=== CAMERA RULES ===
You have two distinct camera views:
  - base camera  → wide-angle scene overview (object presence, workspace state)
  - wrist camera → close-up end-effector view (grasp quality, held object)

Use the correct camera for the task:
  - observe_with_base_camera:  scene understanding, "what's on the table?"
  - observe_with_wrist_camera: grasp confirmation, "what's in the gripper?"
  - observe_with_both_cameras: when you need full context simultaneously
  - yolo_base_camera:          locate objects on the table by bounding box
  - yolo_wrist_camera:         detect objects in close range or verify grip

Always pass a specific question when calling an observe capability.
Do not rely on frame images in your reasoning — the VLM description is your observation.

=== VLA POLICY EXECUTION ===
To run a manipulation policy:
1. Call start_vla_policy with the appropriate policy_id.
   This call BLOCKS until the model is loaded and the robot is already executing —
   no extra wait needed for model download.
2. Call wait (typically 30-60 seconds) to let the policy run further.
3. Call observe_with_base_camera or observe_with_wrist_camera to check success.
4. If not successful, call wait again and re-observe. Repeat until success or max attempts.
5. Call stop_vla_policy when the task is confirmed complete.

Never call start_vla_policy twice in a row — a policy is already running if SUCCESS was returned.

=== SAFETY RULES ===
- Always observe the scene before triggering any policy.
- Never invent object positions — use values from yolo_base_camera results.
- If a capability returns an error, read its doc again and correct your args.
- Do not loop more than {{MAX_ITERATIONS}} times.
"""

tool_node = ToolNode(TOOLS)

def _extract_text_content(content: Any) -> str:
    """Normalize AI message content into plain text for final summaries."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(p for p in parts if p).strip()
    return ""


def _extract_done_summary_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Back-compat shim: if model emits run_capability(DONE), terminate gracefully."""
    for call in tool_calls:
        if call.get("name") != "run_capability_tool":
            continue
        call_args = call.get("args") or {}
        cap_id = str(call_args.get("capability_id", "")).strip().upper()
        if cap_id != "DONE":
            continue
        payload = call_args.get("args") or call_args.get("kwargs") or {}
        if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
            return payload["summary"].strip()
        if isinstance(call_args.get("summary"), str):
            return call_args["summary"].strip()
        return "Task completed."
    return ""


def should_continue_factory(max_iterations: int) -> Callable[[AgentState], str]:
    def should_continue(state: AgentState) -> str:
        if state.get("done"):
            return "END"
        if state.get("step", 0) >= max_iterations:
            return "END"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "END"

    return should_continue

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reason_node(state: AgentState, llm: LLMLike, max_iterations: int) -> dict[str, Any]:
    system = SYSTEM_PROMPT.replace("{{MAX_ITERATIONS}}", str(max_iterations))
    # Scrub frame_b64 from any tool messages before they hit the LLM context window
    clean_messages = _strip_frame_b64_from_tool_messages(state["messages"])
    messages = [SystemMessage(content=system)] + clean_messages

    response = llm.invoke(messages)

    tool_calls = getattr(response, "tool_calls", [])

    # Extract thinking/reasoning from the response.
    # - Ollama (reasoning=True): lands in additional_kwargs["reasoning_content"]
    # - Vertex structured thinking: lands as a {"type": "thinking"} block in content list
    reasoning = ""
    additional = getattr(response, "additional_kwargs", {}) or {}
    if additional.get("reasoning_content"):
        # Ollama path
        reasoning = additional["reasoning_content"]
    elif isinstance(response.content, list):
        # Vertex path
        parts: list[str] = []
        for b in response.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "thinking" and isinstance(b.get("thinking"), str):
                parts.append(b["thinking"])
            elif b.get("type") == "reasoning" and isinstance(b.get("reasoning"), str):
                parts.append(b["reasoning"])
            elif b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        reasoning = "\n\n".join(filter(None, parts))
    elif isinstance(response.content, str):
        # some gemini models will simply output reasoning as invoked response content
        reasoning = str(response.content)

    # Vertex/Gemini tool-call turns can expose thought metadata/signatures
    # without plaintext thought text. Surface a bubble anyway for visibility.
    if not reasoning:
        response_meta = getattr(response, "response_metadata", {}) or {}
        usage_meta = response_meta.get("usage_metadata", {}) if isinstance(response_meta, dict) else {}
        thoughts_tokens = usage_meta.get("thoughts_token_count") if isinstance(usage_meta, dict) else 0

        usage = getattr(response, "usage_metadata", {}) or {}
        output_details = usage.get("output_token_details", {}) if isinstance(usage, dict) else {}
        reasoning_tokens = output_details.get("reasoning") if isinstance(output_details, dict) else 0

        has_vertex_signature = bool(additional.get("__gemini_function_call_thought_signatures__"))
        token_count = reasoning_tokens or thoughts_tokens
        if has_vertex_signature or token_count:
            token_note = f" ({token_count} reasoning tokens)" if token_count else ""
            reasoning = (
                "_Internal model thinking was used"
                f"{token_note}, but this provider does not expose plaintext "
                "thought content for this step._"
            )

    done_from_done_tool = _extract_done_summary_from_tool_calls(tool_calls)
    final_response = ""
    done = False

    # If model provided a terminal summary directly (no tool calls), end cleanly.
    if not tool_calls:
        final_response = _extract_text_content(getattr(response, "content", ""))
        done = True

    # Backward compatibility: accept legacy DONE call and terminate without executing it.
    if done_from_done_tool:
        final_response = done_from_done_tool
        done = True

    # Hard stop guardrail when max iterations reached.
    next_step = state["step"] + 1
    if next_step >= max_iterations and not done:
        final_response = (
            "Stopped after reaching max iterations before a clear completion "
            "summary was produced."
        )
        done = True

    trace_event = TraceEvent(
        step=state["step"] + 1,
        stage="reason",
        capability=tool_calls[0]["name"] if tool_calls else "final_answer",
        reasoning=reasoning,
        args=tool_calls[0]["args"] if tool_calls else {},
        result_summary=final_response if final_response else ""
    )

    # Do not forward tool-call message if it is the synthetic DONE call.
    messages_update = [] if done_from_done_tool else [response]

    return {
        "messages": messages_update,
        "trace": [trace_event],
        "step": next_step,
        "done": done,
        "final_response": final_response,
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
        include_thoughts = True,
    )

def get_ollama_llm(settings: Settings, model_name: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        reasoning=True,
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
    workflow.add_edge("tools", "reason")
    workflow.add_conditional_edges(
        "reason",
        should_continue_factory(max_iterations),
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
