"""工具调用解析、校验和格式修复检测。

从 DeepSeek 的回复中提取 <tool_call> / <function_call> 块，
验证 JSON 合法性，生成 Anthropic 格式的 tool_use 块。
"""

import json
import logging
import re
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Regex matching both <function_call> and <tool_call>/<tool_calls> tags
_FUNCTION_CALL_RE = re.compile(
    r"<(?:function_call|tool_calls?)>\s*(.*?)\s*</(?:function_call|tool_calls?)>",
    re.DOTALL,
)

_FIX_MAX_RETRIES = 3


# ── JSON parsing ──────────────────────────────────────────────────────────


def _try_parse_json(raw: str) -> Optional[dict]:
    """Strict JSON parse. Only accepts valid JSON directly, NO nested fallback."""
    raw = raw.replace("\\", "/")
    try:
        tc = json.loads(raw)
        if tc.get("name") and tc.get("arguments"):
            return tc
    except json.JSONDecodeError:
        pass
    return None


def _get_json_error_text(raw: str) -> str:
    """Try to parse JSON and return a human-readable error message on failure."""
    raw = raw.replace("\\", "/")
    try:
        json.loads(raw)
        return ""
    except json.JSONDecodeError as e:
        start = max(0, e.pos - 15)
        end = min(len(e.doc), e.pos + 15)
        context = repr(e.doc[start:end])
        return (
            f"JSON parse error at line {e.lineno} col {e.colno}: "
            f"{e.msg}. Context: {context}"
        )


# ── Validation ─────────────────────────────────────────────────────────────


def _has_valid_tool_calls(text: str) -> bool:
    """Check if text contains at least one valid <tool_call> with parseable JSON."""
    for match in _FUNCTION_CALL_RE.finditer(text):
        if _try_parse_json(match.group(1)):
            return True
    return False


def _needs_fix(response: str, prompt: str) -> bool:
    """Check if a response needs format fix.

    Returns True if the response has <tool_call>/<function_call> tags
    but none of them contain valid JSON.
    """
    if "Tool:" not in prompt:
        return False
    if not re.search(r"<(?:function_call|tool_calls?)>", response):
        return False
    if _has_valid_tool_calls(response):
        return False
    return True


# ── Response parsing ───────────────────────────────────────────────────────


def _parse_response(text: str) -> tuple[str, list[dict]]:
    """Parse DeepSeek's response for tool_call tags.

    Returns (clean_text, tool_use_blocks).
    """
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    # Strip verbatim tool output blocks that DeepSeek sometimes regurgitates
    text = re.sub(
        r"--- TOOL EXECUTED ---.*?--- END TOOL OUTPUT ---",
        "", text, flags=re.DOTALL,
    ).strip()
    tool_blocks: list[dict] = []
    last_end = 0
    clean_parts: list[str] = []

    for match in _FUNCTION_CALL_RE.finditer(text):
        before = text[last_end: match.start()].strip()
        if before:
            clean_parts.append(before)

        raw = match.group(1).strip()
        tc = _try_parse_json(raw)

        if tc is None:
            full = match.group(0)
            name_match = re.search(r'name\s*=\s*"(\w+)"', full)
            params = dict(re.findall(
                r'<parameter\s+name\s*=\s*"(\w+)"\s*>\s*"?([^"<\n]+)"?\s*</parameter>',
                full,
            ))
            if name_match and params:
                tc = {"name": name_match.group(1), "arguments": params}

        if tc is None:
            clean_parts.append(match.group(0))
            continue

        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        if name:
            tool_blocks.append({
                "type": "tool_use",
                "id": f"tu_{uuid.uuid4().hex[:16]}",
                "name": name,
                "input": arguments,
            })
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        clean_parts.append(remaining)

    clean_text = "\n".join(clean_parts).strip()
    return clean_text, tool_blocks


def _build_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"
