"""Anthropic-format API server — powered by DeepSeek Web Chat

Wraps DeepSeekClient into an Anthropic Messages API-compatible HTTP server
so that tools like Claude Code can talk to DeepSeek's free web version
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests

# ---- ensure project root is on sys.path ----
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.pw_env import apply_playwright_browsers_path

apply_playwright_browsers_path()

from playwright.sync_api import sync_playwright

from src.exceptions import (
    DeepSeekError,
    SessionExpiredError,
    ResponseTimeoutError,
)
from src.timing import _tic, _toc
from src.tool_parser import (
    _FIX_MAX_RETRIES,
    _build_msg_id,
    _has_valid_tool_calls,
    _needs_fix,
    _parse_response,
    _try_parse_json,
)
from src.prompt_builder import build_prompt

logger = logging.getLogger("server")

# (timing helpers imported from src.timing)


# ═══════════════════════════════════════════════════════════════════
#  HTTP DIRECT WORKER (no Playwright for chat, only for PoW)
# ═══════════════════════════════════════════════════════════════════

import queue
import threading

_DS_BASE = "https://chat.deepseek.com"
_DS_CHALLENGE = f"{_DS_BASE}/api/v0/chat/create_pow_challenge"
_DS_SESSION = f"{_DS_BASE}/api/v0/chat_session/create"
_DS_COMPLETION = f"{_DS_BASE}/api/v0/chat/completion"
_DS_SESSION_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "session", "storage_state.json")
)

_http_sess = requests.Session()
_http_sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/148.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
})


def _load_ds_token() -> Optional[str]:
    if not os.path.isfile(_DS_SESSION_FILE):
        return None
    try:
        with open(_DS_SESSION_FILE) as f:
            data = json.load(f)
        for origin in data.get("origins", []):
            for item in origin.get("localStorage", []):
                if item.get("name") == "userToken":
                    raw = item.get("value", "")
                    try:
                        return json.loads(raw).get("value") or None
                    except (TypeError, ValueError):
                        return raw or None
    except Exception as exc:
        logger.warning("[令牌] %s", exc)
    return None


def _extract_session_id(payload) -> Optional[str]:
    """Find the chat session id in a create_session response.

    DeepSeek has returned the id at several depths over time, e.g.
    ``data.biz_data.id`` (old) and ``data.biz_data.chat_session.id`` (new),
    so search the small known set of shapes instead of indexing blindly.
    """
    seen = 0

    def walk(node, in_session: bool = False) -> Optional[str]:
        nonlocal seen
        seen += 1
        if seen > 200:
            return None
        if isinstance(node, list):
            for item in node:
                found = walk(item, in_session)
                if found:
                    return found
            return None
        if not isinstance(node, dict):
            return None

        session = node.get("chat_session")
        if isinstance(session, dict) and session.get("id"):
            return str(session["id"])

        # Inside biz_data / chat_session, a plain "id" is the session id.
        if in_session:
            value = node.get("id")
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                return str(value)

        # A dict that otherwise looks like the session object itself.
        if node.get("id") and ("seq_id" in node or "title" in node):
            return str(node["id"])

        for key in ("data", "biz_data", "chat_session", "sessions"):
            if key in node:
                found = walk(
                    node[key],
                    in_session or key in ("biz_data", "chat_session"),
                )
                if found:
                    return found
        return None

    return walk(payload)


_THINK_FRAGMENT_TYPES = {"THINK"}
_ANSWER_FRAGMENT_TYPES = {"RESPONSE", "ANSWER", "CONTENT", "TEXT"}


def _extract_completion_text(resp_text: str) -> tuple:
    """Rebuild the assistant answer from DeepSeek's SSE patch stream.

    The stream begins with a full response snapshot and then sends
    JSON-patch style updates::

        data: {"v": {"response": {..., "fragments": [{"type": "THINK", ...}]}}}
        data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": " Hi"}
        data: {"v": " there"}        # path inherited from the previous event

    Thinking fragments (``type == "THINK"``) are skipped so only the final
    answer survives. Returns
    ``(answer, paths_seen, fragment_types, final_status)`` — the extra values
    are diagnostics used when the answer looks truncated.
    """
    fragments: list = []
    legacy_parts: list = []
    paths_seen: set = set()
    current_path = ""
    final_status = ""

    for line in resp_text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        value = event.get("v")
        # A patch with no "o" is an APPEND (that is how DeepSeek streams text
        # deltas).  Treating it as SET made every delta overwrite the previous
        # one, leaving only the final character.
        op = str(event.get("o") or "").upper()

        # ── full snapshot (no path) ──
        if not event.get("p") and isinstance(value, dict) and "response" in value:
            response = value.get("response") or {}
            frags = response.get("fragments")
            if isinstance(frags, list):
                fragments = [f for f in frags if isinstance(f, dict)]
            continue

        path = event.get("p") or current_path
        if event.get("p"):
            current_path = event["p"]
            paths_seen.add(current_path)
        if not path:
            continue

        # ── legacy flat paths ──
        if path == "response/content":
            if isinstance(value, str):
                legacy_parts.append(value)
            continue
        if path == "response/status":
            if isinstance(value, str):
                final_status = value
            continue
        if not path.startswith("response/fragments"):
            continue

        rest = path[len("response/fragments"):].lstrip("/")

        # whole-list updates
        if not rest:
            if isinstance(value, list):
                new_frags = [f for f in value if isinstance(f, dict)]
                if op == "APPEND":
                    fragments.extend(new_frags)
                else:
                    fragments = new_frags
            continue

        idx_raw, _, field = rest.partition("/")
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            idx = len(fragments) + idx

        # Direct write of a whole fragment object: response/fragments/N
        if not field:
            if isinstance(value, dict) and 0 <= idx <= len(fragments):
                if idx == len(fragments):
                    fragments.append(value)
                else:
                    fragments[idx] = value
            continue

        if not 0 <= idx < len(fragments):
            continue
        frag = fragments[idx]

        if field == "content":
            if op in ("", "APPEND"):
                if isinstance(value, str):
                    frag["content"] = (frag.get("content") or "") + value
            elif isinstance(value, str):
                frag["content"] = value
        elif field == "type":
            if isinstance(value, str):
                old_type = str(frag.get("type") or "").upper()
                frag["type"] = value
                # THINK → RESPONSE flip reuses the slot; drop the reasoning.
                if (
                    old_type in _THINK_FRAGMENT_TYPES
                    and value.upper() in _ANSWER_FRAGMENT_TYPES
                ):
                    frag["content"] = ""
        elif field == "id" and isinstance(value, int):
            frag["id"] = value

    frag_types = [
        str(f.get("type") or "?")
        for f in fragments
        if (f.get("content") or "")
    ]
    frag_detail = " | ".join(
        "{i}:{t}(len={n}) {pre!r}".format(
            i=i,
            t=f.get("type") or "?",
            n=len(f.get("content") or ""),
            pre=(f.get("content") or "")[:40],
        )
        for i, f in enumerate(fragments)
    )[:1500]

    def _join(predicate) -> str:
        return "".join(
            f.get("content") or "" for f in fragments if predicate(f)
        ).strip()

    answer = _join(
        lambda f: str(f.get("type") or "").upper() in _ANSWER_FRAGMENT_TYPES
    )
    if answer:
        return answer, paths_seen, frag_types, final_status, frag_detail

    answer = _join(
        lambda f: str(f.get("type") or "").upper() not in _THINK_FRAGMENT_TYPES
    )
    if answer:
        return answer, paths_seen, frag_types, final_status, frag_detail

    # Only thinking fragments arrived. A finished response that never emitted
    # an answer fragment is worth surfacing; an INCOMPLETE one means the
    # generation was cut mid-reasoning — returning the reasoning as if it were
    # the answer is what made the proxy look "chat only, no tools".
    if (final_status or "").upper() == "FINISHED":
        answer = _join(lambda f: True)
        if answer:
            return answer, paths_seen, frag_types, final_status, frag_detail

    return (
        "".join(legacy_parts).strip(),
        paths_seen,
        frag_types,
        final_status,
        frag_detail,
    )


class _HTTPWorker:
    """Dedicated thread: PoW via hidden browser, chat via HTTP."""

    def __init__(self):
        self._token: Optional[str] = None
        self._request_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ask(self, prompt: str, timeout: float = 300) -> str:
        if not self._token:
            self._token = _load_ds_token()
        if not self._token:
            raise SessionExpiredError("No session. 请先执行 --mode login 登录")
        q: queue.Queue = queue.Queue()
        self._request_queue.put(("ask", prompt, q))
        status, value = q.get(timeout=timeout)
        if status == "error":
            raise DeepSeekError(value)
        return value

    def shutdown(self):
        self._request_queue.put((None, None, None))

    def _run(self):
        if not os.path.isfile(_DS_SESSION_FILE):
            logger.error("[HTTP] 未找到会话文件. 请先执行 --mode login 登录.")
            while True:
                cmd, _, rq = self._request_queue.get()
                if cmd is None:
                    break
                rq.put(("error", "No session. 请先执行 --mode login 登录"))
            return

        logger.info("[HTTP] 启动 PoW 浏览器 ...")
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                storage_state=_DS_SESSION_FILE,
                viewport={"width": 1280, "height": 720},
            )
            page = ctx.new_page()
            pow_ev = threading.Event()
            pow_val: list[Optional[str]] = [None]

            def interceptor(route, req):
                if "chat/completion" in req.url:
                    h = req.headers
                    p = h.get("x-ds-pow-response")
                    if p:
                        pow_val[0] = p
                        pow_ev.set()
                route.abort()

            page.route("**/chat/completion**", interceptor)
            page.goto(_DS_BASE)
            page.wait_for_timeout(10000)
            logger.info("[HTTP] PoW 浏览器就绪")
        except Exception as exc:
            logger.error("[HTTP] PoW 浏览器启动失败: %s", exc)
            while True:
                cmd, _, rq = self._request_queue.get()
                if cmd is None:
                    break
                rq.put(("error", f"Startup: {exc}"))
            return

        while True:
            cmd, arg, rq = self._request_queue.get()
            if cmd is None:
                try: browser.close()
                except: pass
                try: pw.stop()
                except: pass
                if rq:
                    rq.put(("ok", None))
                break
            if cmd == "ask":
                logger.info("[HTTP] 请求 (%d chars)", len(arg))
                _tic()
                try:
                    result, complete = self._http_chat(
                        arg, page, pow_ev, pow_val
                    )
                    _toc(f"[HTTP] 完成 ({len(result)} chars)")

                    # Retry truncated/incomplete generations — DeepSeek ends
                    # the stream mid-reasoning when it gives up on thinking.
                    for attempt in range(_FIX_MAX_RETRIES):
                        if complete and not _needs_fix(result, arg):
                            break
                        reason = (
                            "incomplete" if not complete else "format fix"
                        )
                        logger.info("[修复] 重试 %d (%s) ...", attempt + 1, reason)
                        result, complete = self._http_chat(
                            arg, page, pow_ev, pow_val
                        )
                        _toc(
                            f"[修复] 重试 {attempt + 1} returned "
                            f"({len(result)} chars, complete={complete})"
                        )

                    if not result.strip():
                        raise RuntimeError(
                            "DeepSeek 返回空内容（生成被中断，已重试）"
                        )

                    rq.put(("ok", result))
                except Exception as exc:
                    _toc("[HTTP] 失败")
                    rq.put(("error", str(exc)))

    def _http_chat(self, prompt: str, page, pow_ev, pow_val) -> str:
        # Session
        r = _http_sess.post(_DS_SESSION, json={},
                            headers={"Authorization": f"Bearer {self._token}"})
        r.raise_for_status()
        try:
            j = r.json()
        except ValueError:
            raise RuntimeError(f"Session create 返回非 JSON: {r.text[:500]}")

        sid = _extract_session_id(j)
        if not sid:
            raise RuntimeError(
                "Session create 返回里找不到会话 id。原始响应: "
                + json.dumps(j, ensure_ascii=False)[:800]
            )

        # PoW
        pow_ev.clear()
        pow_val[0] = None
        ta = page.query_selector('textarea[name="search"]')
        if ta:
            ta.fill("_")
            page.wait_for_timeout(500)
            ta.press("Enter")
            if not pow_ev.wait(timeout=30):
                raise RuntimeError("PoW timed out after 30s")
        else:
            raise RuntimeError("PoW browser textarea not found")
        if not pow_val[0]:
            raise RuntimeError("PoW header not captured")
        nc = page.query_selector('[class*="_5a8ac7a"]')
        if nc:
            nc.click()
            page.wait_for_timeout(1000)

        # Completion (non-streaming to avoid SSE parsing issues)
        # thinking_enabled is OFF on purpose: with it on, DeepSeek spends the
        # whole stream reasoning and frequently ends with status=INCOMPLETE,
        # yielding only a THINK fragment (i.e. no answer, no tool call).
        body = {
            "chat_session_id": sid, "parent_message_id": None,
            "model_type": "default", "prompt": prompt,
            "ref_file_ids": [], "thinking_enabled": False,
            "search_enabled": False, "action": None, "preempt": False,
        }
        resp = _http_sess.post(
            _DS_COMPLETION, json=body, stream=False, timeout=180,
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Ds-Pow-Response": pow_val[0],
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Origin": _DS_BASE,
                "Referer": _DS_BASE + "/",
            },
        )
        resp.raise_for_status()
        resp_text = resp.text

        answer, paths_seen, frag_types, final_status, frag_detail = (
            _extract_completion_text(resp_text)
        )
        complete = (final_status or "").upper() == "FINISHED"

        try:
            dump_path = os.path.join("logs", "last_completion.sse")
            with open(dump_path, "w", encoding="utf-8") as fh:
                fh.write(resp_text)
        except OSError as exc:
            logger.warning("[HTTP] 保存原始流失败: %s", exc)

        logger.info(
            "[HTTP] 回复 %d 字符 | http=%s status=%s complete=%s len=%d",
            len(answer), resp.status_code, final_status or "?",
            complete, len(resp_text),
        )
        logger.info("[HTTP] 片段明细: %s", frag_detail or "none")

        if not complete or len(answer) < 20:
            logger.warning(
                "[HTTP] 原始响应尾部 40 行:\n%s",
                "\n".join(resp_text.splitlines()[-40:]),
            )

        return answer, complete


_worker: _HTTPWorker | None = None


def _get_worker() -> _HTTPWorker:
    global _worker
    if _worker is None:
        _worker = _HTTPWorker()
    return _worker


# ── app lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("[HTTP] 启动工作线程 ...")
    _get_worker()
    logger.info("[HTTP] 服务器就绪.")
    yield
    global _worker
    if _worker is not None:
        logger.info("[HTTP] 正在关闭")
        _worker.shutdown()
        _worker = None


app = FastAPI(lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════════

#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/v1/models")
async def list_models():
    logger.info("[请求] 查询模型列表")
    return {
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek-web",
            }
        ]
    }


@app.post("/v1/messages")
async def create_message(request: Request):
    _tic()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON 请求体无效"}, status_code=400)

    stream = body.get("stream", False)
    model = body.get("model", "deepseek-chat")
    tools = body.get("tools", [])
    msg_count = len(body.get("messages", []))
    tool_names = [t.get("name", "?") for t in tools]
    last_role = body.get("messages", [{}])[-1].get("role", "?") if body.get("messages") else "?"
    logger.info(
        "[请求] model=%s stream=%s tools=%s msgs=%d last_role=%s",
        model, stream, tool_names or "none", msg_count, last_role,
    )
    _toc("[服务器] 解析请求体")

    _tic()
    try:
        prompt_text = build_prompt(body)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    _toc(f"[服务器] 构建提示词 ({len(prompt_text)} chars)")

    if tools:
        logger.info("[服务器] 请求中的工具: %s", tool_names)

    _tic()
    try:
        worker = _get_worker()
    except SessionExpiredError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    except DeepSeekError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    _toc("[服务器] 获取工作线程")

    if stream:
        logger.info("[服务器] 流式响应 ...")
        return StreamingResponse(
            _stream_events(worker, prompt_text, model, tool_names),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    _tic()
    try:
        response_text = worker.ask(prompt_text)
    except (ResponseTimeoutError, DeepSeekError) as exc:
        logger.warning("[服务器] DeepSeek 错误: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    _toc(f"[服务器] DeepSeek 返回 ({len(response_text)} chars)")
    logger.info("[原始] DeepSeek 回复:\n%s", response_text)

    clean_text, tool_uses = _parse_response(response_text)
    _toc(f"[服务器] 解析响应 ({len(tool_uses)} tool_uses)")
    tool_uses = _normalize_tool_names(tool_uses, tool_names)

    if not tool_uses and tool_names:
        logger.info("[原始] 未发现 tool_call 标签.")

    content = []
    if clean_text:
        content.append({"type": "text", "text": clean_text})
    content.extend(tool_uses)
    stop_reason = "tool_use" if tool_uses else "end_turn"

    logger.info(
        "[响应] stop_reason=%s text_len=%d tool_count=%d",
        stop_reason, len(clean_text or ""), len(tool_uses),
    )
    return {
        "id": _build_msg_id(),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ═══════════════════════════════════════════════════════════════════
#  STREAMING
# ═══════════════════════════════════════════════════════════════════

def _normalize_tool_names(tool_uses: list, tool_names: list) -> list:
    """Map model-emitted tool names onto the request's real tool names.

    Chat models copy example spellings (``Read`` vs ``read``); a mismatch
    means the client cannot resolve the tool, so fix case here.
    """
    if not tool_uses:
        return tool_uses
    lookup = {str(n).lower(): n for n in tool_names if isinstance(n, str)}
    normalized = []
    for tu in tool_uses:
        name = str(tu.get("name") or "")
        canonical = lookup.get(name.lower())
        if canonical and canonical != name:
            logger.info("[工具] 名称规范化: %r → %r", name, canonical)
            tu = dict(tu, name=canonical)
        elif not canonical and tool_names:
            logger.warning(
                "[工具] 模型调用了未声明的工具 %r；已声明: %s",
                name, tool_names[:10],
            )
        normalized.append(tu)
    return normalized


async def _stream_events(worker, prompt: str, model: str, tool_names=None):
    """Anthropic Messages SSE stream.

    Each event needs BOTH an ``event:`` field and a ``data:`` payload: the
    Anthropic SDK (and every client built on it) dispatches on the SSE event
    name and silently drops data-only frames, which renders as an empty
    message.  The stream must also terminate with ``message_stop``.
    """
    msg_id = _build_msg_id()

    def _event(event_type: str, data: dict) -> str:
        payload = json.dumps(
            {"type": event_type, **data}, ensure_ascii=False
        )
        return f"event: {event_type}\ndata: {payload}\n\n"

    yield _event("message_start", {
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    try:
        response_text = worker.ask(prompt)
    except (ResponseTimeoutError, DeepSeekError) as exc:
        logger.warning("[服务器] DeepSeek 错误: %s", exc)
        response_text = f"[Error: {exc}]"

    clean_text, tool_uses = _parse_response(response_text)
    tool_uses = _normalize_tool_names(tool_uses, tool_names or [])
    logger.info(
        "[流式] DeepSeek 原始回复 %d 字符, 解析出 %d 个 tool_call: %s",
        len(response_text), len(tool_uses),
        [tu.get("name") for tu in tool_uses],
    )
    if tool_names and not tool_uses:
        logger.info(
            "[流式] 未发现 tool_call 标签；原始回复前 600 字符:\n%s",
            response_text[:600],
        )

    if clean_text:
        yield _event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": clean_text},
        })
        yield _event("content_block_stop", {"index": 0})

    tool_offset = 1 if clean_text else 0
    for i, tu in enumerate(tool_uses):
        idx = tool_offset + i
        yield _event("content_block_start", {
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": {},
            },
        })
        # Arguments must arrive as input_json_delta: clients rebuild the
        # input from the accumulated partial_json and overwrite whatever
        # was in content_block_start when the block stops.
        yield _event("content_block_delta", {
            "index": idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    tu.get("input") or {}, ensure_ascii=False
                ),
            },
        })
        yield _event("content_block_stop", {"index": idx})

    stop_reason = "tool_use" if tool_uses else "end_turn"
    yield _event("message_delta", {
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield _event("message_stop", {})


# ═══════════════════════════════════════════════════════════════════
#  OPENAI-COMPATIBLE ENDPOINTS  (/v1/chat/completions)
# ═══════════════════════════════════════════════════════════════════
#
# The core pipeline speaks Anthropic. Most third-party clients (Cherry
# Studio, ChatBox, Continue, LangChain, openai SDK pointed at a base_url)
# speak the OpenAI chat-completions dialect instead, so translate on the
# way in and out rather than forcing callers to switch APIs.

def _flatten_oai_content(content) -> str:
    """Flatten an OpenAI message content (str or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text"):
                    texts.append(part.get("text") or "")
        return "\n".join(t for t in texts if t)
    return str(content)


def _oai_tools_to_anthropic(tools) -> list:
    """Convert OpenAI ``tools`` into Anthropic tool definitions."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description", "") or "",
            "input_schema": fn.get("parameters")
            or {"type": "object", "properties": {}},
        })
    return out


def _oai_messages_to_anthropic(messages) -> tuple:
    """Split OpenAI messages into (system_text, Anthropic messages)."""
    system_parts: list = []
    converted: list = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "developer"):
            text = _flatten_oai_content(content)
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or "call_0",
                    "content": _flatten_oai_content(content),
                }],
            })
            continue

        if role == "assistant":
            blocks = []
            text = _flatten_oai_content(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        args = json.loads(raw_args)
                    except (TypeError, ValueError):
                        args = {"_raw": raw_args}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") or "call_0",
                    "name": fn.get("name") or "?",
                    "input": args,
                })
            if blocks:
                converted.append({"role": "assistant", "content": blocks})
            continue

        # user (and anything unknown treated as user input)
        text = _flatten_oai_content(content)
        if text:
            converted.append({
                "role": "user",
                "content": [{"type": "text", "text": text}],
            })

    return "\n\n".join(system_parts), converted


def _oai_error(message: str, err_type: str = "api_error", code: Optional[str] = None):
    """OpenAI-shaped error body."""
    return {"error": {"message": message, "type": err_type, "code": code}}


def _tool_uses_to_oai_calls(tool_uses) -> list:
    return [
        {
            "id": tu.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": tu.get("name", ""),
                "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False),
            },
        }
        for tu in tool_uses
    ]


def _anthropic_to_oai(anthropic: dict, model: str) -> dict:
    """Convert an Anthropic message payload into an OpenAI completion."""
    text_parts = []
    tool_uses = []
    for block in anthropic.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_uses.append(block)

    message: dict = {"role": "assistant", "content": "".join(text_parts)}
    if tool_uses:
        message["tool_calls"] = _tool_uses_to_oai_calls(tool_uses)
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    return {
        "id": anthropic.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _oai_stream(worker, prompt: str, model: str):
    """OpenAI ``chat.completion.chunk`` SSE stream.

    The underlying DeepSeek call returns the whole answer at once, so the
    text is emitted as a single delta — chunked enough for clients that
    require SSE framing, without pretending to be token-by-token.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _chunk(delta: dict, finish: Optional[str] = None) -> str:
        return "data: " + json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }, ensure_ascii=False) + "\n\n"

    yield _chunk({"role": "assistant", "content": ""})

    try:
        response_text = worker.ask(prompt)
    except (ResponseTimeoutError, DeepSeekError) as exc:
        logger.warning("[OpenAI] DeepSeek 错误: %s", exc)
        yield _chunk({"content": f"[Error: {exc}]"})
        yield _chunk({}, "stop")
        yield "data: [DONE]\n\n"
        return

    clean_text, tool_uses = _parse_response(response_text)
    if clean_text:
        yield _chunk({"content": clean_text})
    if tool_uses:
        yield _chunk({"tool_calls": _tool_uses_to_oai_calls(tool_uses)})
    yield _chunk({}, "tool_calls" if tool_uses else "stop")
    yield "data: [DONE]\n\n"


async def _openai_chat(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse(
            _oai_error("JSON 请求体必须是对象", "invalid_request_error"),
            status_code=400,
        )

    model = body.get("model") or "deepseek-chat"
    stream = bool(body.get("stream", False))
    logger.info(
        "[OpenAI] model=%s stream=%s msgs=%d tools=%d",
        model, stream, len(body.get("messages") or []),
        len(body.get("tools") or []),
    )

    try:
        system_text, messages = _oai_messages_to_anthropic(body.get("messages"))
        anthropic_body = {
            "system": system_text,
            "messages": messages,
            "tools": _oai_tools_to_anthropic(body.get("tools")),
        }
        prompt_text = build_prompt(anthropic_body)
    except ValueError as exc:
        return JSONResponse(
            _oai_error(str(exc), "invalid_request_error"), status_code=400
        )

    try:
        worker = _get_worker()
    except DeepSeekError as exc:
        return JSONResponse(_oai_error(str(exc)), status_code=500)

    if stream:
        return StreamingResponse(
            _oai_stream(worker, prompt_text, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        response_text = worker.ask(prompt_text)
    except SessionExpiredError as exc:
        return JSONResponse(
            _oai_error(str(exc), "authentication_error", "no_session"),
            status_code=401,
        )
    except (ResponseTimeoutError, DeepSeekError) as exc:
        logger.warning("[OpenAI] DeepSeek 错误: %s", exc)
        return JSONResponse(_oai_error(str(exc)), status_code=502)

    clean_text, tool_uses = _parse_response(response_text)
    content = []
    if clean_text:
        content.append({"type": "text", "text": clean_text})
    content.extend(tool_uses)

    anthropic = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "content": content,
    }
    payload = _anthropic_to_oai(anthropic, model)
    logger.info(
        "[OpenAI] 响应完成: text_len=%d tool_calls=%d",
        len(clean_text or ""), len(tool_uses),
    )
    return payload


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    return await _openai_chat(request)


@app.post("/chat/completions")
async def openai_chat_completions_short(request: Request):
    """Same handler without the /v1 prefix (some clients omit it)."""
    return await _openai_chat(request)


@app.get("/models")
async def list_models_short():
    return await list_models()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Web -> Anthropic API proxy"
    )
    parser.add_argument("--port", type=int, default=8080, help="Listen port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log file directory")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    args = parser.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "webai.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root_logger.addHandler(console)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10_485_760, backupCount=5, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(fh)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    print(f"\n  DeepSeek Web -> Anthropic / OpenAI API proxy")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Models:  GET  /v1/models")
    print(f"  Claude:  POST /v1/messages")
    print(f"  OpenAI:  POST /v1/chat/completions  (alias: /chat/completions)")
    print(f"  Logs: {os.path.abspath(log_path)}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
