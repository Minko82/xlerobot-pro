"""Streamlit UI for the XLErobot ReAct orchestrator."""

from __future__ import annotations

import asyncio
import base64
import json
import re
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

def _parse_tool_result(result_str: str) -> dict | None:
    """Try to parse a tool result string as JSON. Returns dict or None."""
    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None


def render_tool_status(
    step_num: int,
    cap_id: str,
    args: dict,
    result_summary: str,
    is_running: bool = False,
):
    """Renders a tool call as a status card.

    If the result JSON contains a 'frame_b64' key with a non-empty base64 PNG,
    the image is displayed inline inside the card.
    """
    label = f"🛠️ Step {step_num}: {cap_id}"
    state = "running" if is_running else "complete"

    status = st.status(label, state=state)
    with status:
        st.write("**Input (Args):**")
        st.json(args)

        if not is_running:
            st.write("**Observation Result:**")

            # Try to parse the result as JSON so we can handle images specially
            parsed = _parse_tool_result(result_summary)

            if parsed is not None:
                frame_b64: str = parsed.pop("frame_b64", "") or ""

                # Show the camera image first if present
                if frame_b64:
                    try:
                        img_bytes = base64.b64decode(frame_b64)
                        st.image(
                            img_bytes,
                            caption=f"Camera frame — {parsed.get('camera_info', '')}",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Could not decode camera frame: {exc}")

                # Show remaining fields as JSON (frame_b64 already popped)
                if parsed.get("error"):
                    st.error(f"⚠️ {parsed['error']}")

                # Detected objects table if present
                detected = parsed.get("detected")
                if detected:
                    st.write(f"**Detected objects ({parsed.get('count', len(detected))}):**")
                    rows = []
                    for obj in detected:
                        rows.append({
                            "ID": obj.get("id", ""),
                            "Label": obj.get("label", ""),
                            "X (m)": round(obj.get("x", 0), 4),
                            "Y (m)": round(obj.get("y", 0), 4),
                            "Z (m)": round(obj.get("z", 0), 4),
                            "Conf": round(obj.get("confidence", 0), 3),
                        })
                    st.dataframe(rows, use_container_width=True)

                # Remaining metadata (note, camera_info, etc.)
                meta = {
                    k: v for k, v in parsed.items()
                    if k not in ("detected", "count", "error", "frame_b64")
                    and v not in (None, "", [], {})
                }
                if meta:
                    with st.expander("Details", expanded=False):
                        st.json(meta)

            else:
                # Fallback: plain text / error display
                if "ERROR" in result_summary.upper():
                    st.error(result_summary)
                else:
                    st.code(result_summary, language="text")

    return status


def render_reasoning(reasoning: str, expanded: bool = False):
    """Renders the 'Thinking' expander block."""
    if reasoning and reasoning.strip():
        with st.expander("💭 Thinking", expanded=expanded):
            st.markdown(reasoning)


def extract_reasoning_from_message_content(content: object) -> str:
    """Extract model reasoning from structured content blocks."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking" and isinstance(block.get("thinking"), str):
            parts.append(block["thinking"])
        elif block_type == "reasoning" and isinstance(block.get("reasoning"), str):
            parts.append(block["reasoning"])
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


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

    st.divider()
    st.subheader("Camera")
    st.caption(
        "Set `USE_REALSENSE=true` in your `.env` to use the Intel RealSense. "
        "Otherwise OpenCV camera index 0 is used.\n\n"
        "Set `OPENCV_CAMERA_INDEX=<n>` to pick a different USB camera."
    )

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

async def main():
    directive = st.text_area(
        "Robot directive",
        placeholder='e.g. "Look at the workspace and tell me what you see."',
        height=100,
    )
    run_clicked = st.button("▶ Run Orchestrator", type="primary", use_container_width=True)

    if run_clicked and directive.strip():
        st.chat_message("human").write(directive.strip())

        with st.chat_message("ai"):
            container = st.container()

            try:
                last_step_rendered = 0

                async for update in astream_directive(directive.strip()):
                    node_name = list(update.keys())[0]
                    node_data = update[node_name]

                    # 1. Reasoning node
                    if node_name == "reason":
                        if "trace" in node_data:
                            event = node_data["trace"][-1]
                            reasoning = event.get("reasoning", "")

                            if not reasoning and "messages" in node_data:
                                for msg in node_data["messages"]:
                                    content_obj = getattr(msg, "content", "")
                                    reasoning = extract_reasoning_from_message_content(content_obj)
                                    if reasoning:
                                        break
                                    raw = str(content_obj or "")
                                    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
                                    if m:
                                        reasoning = m.group(1).strip()
                                        break

                            render_reasoning(reasoning, expanded=expand_thinking)

                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for call in msg.tool_calls:
                                        render_tool_status(
                                            step_num=last_step_rendered + 1,
                                            cap_id=call["name"],
                                            args=call.get("args", {}),
                                            result_summary="Executing capability...",
                                            is_running=True,
                                        )

                    # 2. Tools node — this is where camera frames arrive
                    elif node_name == "tools":
                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if msg.type == "tool":
                                    render_tool_status(
                                        step_num=last_step_rendered + 1,
                                        cap_id=msg.name,
                                        args={},
                                        result_summary=msg.content,
                                        is_running=False,
                                    )
                                    last_step_rendered += 1

                    # Final response
                    if "final_response" in node_data and node_data["final_response"]:
                        st.info(f"**Final Summary:** {node_data['final_response']}")

            except Exception as exc:
                st.error(f"Agent error during execution: {exc}")
                st.code(traceback.format_exc())

    if st.session_state.history:
        st.divider()
        st.subheader("Previous Runs")
        for entry in st.session_state.history:
            with st.expander(f"Directive: {entry['directive'][:50]}..."):
                st.write(entry["final_response"])


if __name__ == "__main__":
    asyncio.run(main())