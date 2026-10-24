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

# DeepSeek's native tool markup uses special tokens that arrive as text like
#   <｜｜DSML｜｜ invoke name="read"> … </｜｜DSML｜｜ invoke>
# (sometimes with ASCII pipes instead of the full-width bar).
_DSML_MARKER_RE = re.compile(r"<(/?)[｜|]+\s*DSML\s*[｜|]+\s*")
_INVOKE_BLOCK_RE = re.compile(
    r'<invoke\s+name\s*=\s*"([^"]+)"\s*>(.*?)</invoke>', re.DOTALL
)
_PARAM_BLOCK_RE = re.compile(
    r'<parameter\s+name\s*=\s*"([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL
)

_FIX_MAX_RETRIES = 3


def _normalize_dsml(text: str) -> str:
    """Rewrite DeepSeek's DSML / Anthropic-style invoke blocks as tool_calls.

    DeepSeek emits ``<｜｜DSML｜｜ invoke name="read">`` with
    ``<｜｜DSML｜｜ parameter name="file_path" string="true">`` children. After
    stripping the special-token markers that is just the Anthropic
    ``<invoke>``/``<parameter>`` shape, so convert it into the <tool_call>
    JSON this module already understands.
    """
    if "DSML" not in text and "<invoke" not in text:
        return text

    text = _DSML_MARKER_RE.sub(lambda m: "<" + m.group(1), text)

    if "<invoke" in text:
        def _convert(match: re.Match) -> str:
            name = match.group(1).strip()
            args = {}
            for pname, pvalue in _PARAM_BLOCK_RE.findall(match.group(2)):
                args[pname.strip()] = pvalue.strip()
            return "<tool_call>" + json.dumps(
                {"name": name, "arguments": args}, ensure_ascii=False
            ) + "</tool_call>"

        text = _INVOKE_BLOCK_RE.sub(_convert, text)

    # Drop the enclosing <calls> wrapper left behind by the conversion.
    text = re.sub(r"</?calls\s*>", "", text)
    return text


# ── JSON parsing ──────────────────────────────────────────────────────────


def _try_parse_json(raw: str) -> Optional[dict]:
    """Parse a tool-call payload, repairing common model mistakes.

    The raw text is tried verbatim first: blanket-replacing backslashes (the
    old behaviour) corrupted Windows paths and escaped newlines. Only when
    the JSON is invalid do we retry with backslashes normalised, which fixes
    bare ``\p``-style escapes that models emit.
    """
    for candidate in (raw, raw.replace("\\", "/")):
        try:
            tc = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(tc, dict):
            continue

        name = tc.get("name") or tc.get("tool") or tc.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            continue

        args = tc.get("arguments")
        if args is None:
            args = tc.get("parameters")
        if args is None:
            args = tc.get("input")
        if args is None:
            args = {}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}

        return {"name": name.strip(), "arguments": args}

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
    # Rewrite DeepSeek's DSML / <invoke> markup into <tool_call> JSON first.
    text = _normalize_dsml(text).strip()
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
            name_match = re.search(r'name\s*=\s*"([^"]+)"', full)
            # Tolerate extra attributes, e.g. <parameter name="x" string="true">
            params = {
                pname.strip(): pvalue.strip()
                for pname, pvalue in _PARAM_BLOCK_RE.findall(full)
            }
            if name_match:
                tc = {
                    "name": name_match.group(1).strip(),
                    "arguments": params,
                }

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
