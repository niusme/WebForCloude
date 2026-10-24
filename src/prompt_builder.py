"""Prompt assembly — converts Anthropic API requests to DeepSeek text prompts."""

import json
import logging
import re

logger = logging.getLogger(__name__)

_MAX_HISTORY = 30


def _tools_to_text(tools: list) -> str:
    """Convert Anthropic tool definitions into a text description."""
    if not tools:
        return ""
    lines = []
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "") or ""
        schema = t.get("input_schema") or t.get("parameters") or {}
        lines.append(f"Tool: {name}")
        if desc:
            lines.append(f"  Description: {desc}")
        props = schema.get("properties", {})
        required = set(schema.get("required", []) or [])
        if props:
            lines.append("  Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                req = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", "")
                lines.append(f"    {pname} ({ptype}){req}")
                if pdesc:
                    lines.append(f"      - {pdesc}")
        lines.append("")
    return "\n".join(lines)


def _example_argument(ptype: str):
    """Produce a plausible JSON value for a parameter type."""
    if ptype in ("integer", "number"):
        return 0
    if ptype == "boolean":
        return False
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return "..."


def _build_tool_examples(tools: list) -> str:
    """Build <tool_call> examples from the request's real tool names.

    The examples must reuse the actual names: models copy them verbatim, and
    a copied name like ``Read`` would not match a tool exposed as ``read``.
    """
    blocks = []
    for tool in tools[:3]:
        name = tool.get("name")
        if not name:
            continue
        schema = tool.get("input_schema") or tool.get("parameters") or {}
        props = schema.get("properties") or {}
        args = {}
        for pname, pinfo in list(props.items())[:2]:
            ptype = (pinfo or {}).get("type", "string")
            args[pname] = _example_argument(ptype)
        blocks.append(
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            + "\n</tool_call>"
        )
    return "\n\n".join(blocks)


_TOOL_SYSTEM_PROMPT = """\
You have access to a set of tools you can use to accomplish tasks.

Available tools:
{tools_text}

IMPORTANT: Tool calls MUST use a <tool_call> tag with valid JSON inside.
The "name" value MUST be copied EXACTLY from the Available tools list above
(same spelling, same letter case) — do not invent names and do not translate
them.

{examples}

WRONG ❌: <Read>{{"file_path": "x"}}</Read>  ← wrong tag name
WRONG ❌: <tool_call>Read {{"file_path": "x"}}</tool_call>  ← tool name outside JSON
WRONG ❌: Read({{"file_path": "x"}})  ← no XML tag at all
WRONG ❌: <tool_call>{{"name": "some_tool"}}</tool_call>  ← missing arguments field

Rules:
- To use a tool, reply with ONLY the <tool_call> block(s), no prose around them.
- Emit one <tool_call> block per tool call.
- After the tool runs, the result appears between --- TOOL EXECUTED --- markers.
  Read it, analyze it, then either answer the user or call another tool.
- Never repeat the same tool call with the same arguments.

Begin."""


def _extract_text(content) -> str:
    """Extract plain text from Anthropic content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            p["text"]
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(texts)
    return ""


def build_prompt(body: dict) -> str:
    """Build a single prompt string from the full Anthropic request.

    - Injects tool definitions as system instructions
    - Injects tool_results back into conversation as text blocks
    - Caps conversation history to prevent unbounded growth
    """
    system = body.get("system", "") or ""
    tools = body.get("tools", [])
    messages = body.get("messages", [])
    logger.debug("[提示词] 消息数=%d 工具数=%d 系统指令长度=%d",
                 len(messages), len(tools), len(system))

    parts: list[str] = []

    # ── 1. system instructions ──
    if system:
        parts.append(f"[System Instructions]\n{system}\n")

    # ── 2. tool definitions (if any) ──
    if tools:
        tools_text = _tools_to_text(tools)
        tool_block = _TOOL_SYSTEM_PROMPT.format(
            tools_text=tools_text,
            examples=_build_tool_examples(tools) or "(see the tool list above)",
        )
        parts.append(f"[Tool Definitions]\n{tool_block}\n")

    # ── 3. conversation history (capped) ──
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    history = [m for m in messages if m.get("role") != "system"]
    dropped = max(0, len(history) - _MAX_HISTORY)
    if dropped:
        history = history[-_MAX_HISTORY:]
        logger.info("[提示词] 丢弃 %d 条旧消息（上限 %d 条）", dropped, _MAX_HISTORY)
    ordered = sys_msgs + history

    for msg in ordered:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            text = _extract_text(content)
            if text:
                parts.append(f"[System]\n{text}\n")
            continue

        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                btype = block.get("type", "text")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(f"User: {text}")
                elif btype == "tool_result":
                    result_text = block.get("content", "")
                    if isinstance(result_text, list):
                        result_text = "\n".join(
                            p.get("text", "") for p in result_text
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    parts.append(
                        f"--- TOOL EXECUTED ---\n"
                        f"{result_text}\n"
                        f"--- END TOOL OUTPUT ---\n"
                        f"[The tool ran successfully. Now analyze the results above. DO NOT call the same tool again.]"
                    )
            continue

        if role == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                btype = block.get("type", "text")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(f"Assistant: {text}")
                elif btype == "tool_use":
                    pass
            continue

    if not parts:
        raise ValueError("No conversation content found")

    result = "\n\n".join(parts)
    if dropped:
        logger.info("[提示词] 构建完成: %d 字符（%d 条消息截断至 %d 条）",
                     len(result), len(messages), len(ordered))
    return result
