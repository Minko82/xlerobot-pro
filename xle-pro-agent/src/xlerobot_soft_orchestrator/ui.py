"""Streamlit UI for the XLErobot ReAct orchestrator."""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

import streamlit as st

# Setup Pathing
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xlerobot_soft_orchestrator.agent import astream_directive

# ---------------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------------

def render_tool_status(step_num: int, cap_id: str, args: dict, result_summary: str, is_running: bool = False):
    """Renders a tool call as a status card."""
    label = f"🛠️ Step {step_num}: {cap_id}"
    state = "running" if is_running else "complete"
    
    status = st.status(label, state=state)
    with status:
        st.write("**Input (Args):**")
        st.json(args)
        if not is_running:
            st.write("**Observation Result:**")
            if "ERROR" in result_summary:
                st.error(result_summary)
            else:
                st.code(result_summary, language="text")
    return status

# TODO: this works well for vertex models, doesn't seem to be showing up for ollama. look into this
def render_reasoning(reasoning: str, expanded: bool = False):
    """Renders the 'Thinking' expander block."""
    if reasoning:
        with st.expander("💭 Thinking", expanded=expanded):
            st.markdown(reasoning)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="XLErobot-Pro Orchestrator", layout="wide", page_icon="🤖")

st.title("🤖 XLErobot-Pro Orchestrator")
st.caption("ReAct loop: Reason → Act → Observe → Reason")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar / Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    expand_thinking = st.toggle("Expand thinking by default", value=False)
    
# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

async def main():
    directive = st.text_area("Robot directive", placeholder='e.g. "Take the screw out of the battery pack."', height=100)
    run_clicked = st.button("▶ Run Orchestrator", type="primary", use_container_width=True)

    if run_clicked and directive.strip():
        st.chat_message("human").write(directive.strip())
        
        with st.chat_message("ai"):
            # We use a placeholder for the final response or state updates
            container = st.container()
            
            try:
                # We'll keep track of rendered steps to avoid duplicates in the stream
                last_step_rendered = 0
                
                async for update in astream_directive(directive.strip()):
                    # LangGraph .astream yields dictionaries like {'node_name': {state_updates}}
                    node_name = list(update.keys())[0]
                    node_data = update[node_name]
                    
                    # 1. Handle Reasoning (from 'reason' node)
                    if node_name == "reason":
                        if "trace" in node_data:
                            event = node_data["trace"][-1]
                            render_reasoning(event.get("reasoning"), expanded=expand_thinking)
                        
                        # Also check for tool calls issued by the reasoner
                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for call in msg.tool_calls:
                                        render_tool_status(
                                            step_num=last_step_rendered + 1,
                                            cap_id=call["name"],
                                            args=call.get("args", {}),
                                            result_summary="Executing capability...",
                                            is_running=True
                                        )

                    # 2. Handle Tool Results (from 'tools' node)
                    elif node_name == "tools":
                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                # ToolNode returns ToolMessage objects
                                if msg.type == "tool":
                                    render_tool_status(
                                        step_num=last_step_rendered + 1,
                                        cap_id=msg.name,
                                        args={}, 
                                        result_summary=msg.content,
                                        is_running=False
                                    )
                                    last_step_rendered += 1
                    
                    # If the node produced a final response
                    if "final_response" in node_data and node_data["final_response"]:
                        st.info(f"**Final Summary:** {node_data['final_response']}")

            except Exception as exc:
                st.error(f"Agent error during execution: {exc}")
                st.code(traceback.format_exc())

    # History display
    if st.session_state.history:
        st.divider()
        st.subheader("Previous Runs")
        for entry in st.session_state.history:
            with st.expander(f"Directive: {entry['directive'][:50]}..."):
                st.write(entry['final_response'])

if __name__ == "__main__":
    asyncio.run(main())