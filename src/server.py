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
from playwright.sync_api import sync_playwright

# ---- ensure project root is on sys.path ----
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

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

os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "browsers")),
)

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
                    return json.loads(raw).get("value") or raw
    except Exception as exc:
        logger.warning("[令牌] %s", exc)
    return None


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
                    result = self._http_chat(arg, page, pow_ev, pow_val)
                    _toc(f"[HTTP] 完成 ({len(result)} chars)")

                    # Format fix loop (inline, has access to fresh PoW via _http_chat)
                    for attempt in range(_FIX_MAX_RETRIES):
                        if not _needs_fix(result, arg):
                            break
                        logger.info("[修复] 尝试 %d ...", attempt + 1)
                        result = self._http_chat(arg, page, pow_ev, pow_val)
                        _toc(f"[修复] 尝试 {attempt + 1} returned ({len(result)} chars)")

                    rq.put(("ok", result))
                except Exception as exc:
                    _toc("[HTTP] 失败")
                    rq.put(("error", str(exc)))

    def _http_chat(self, prompt: str, page, pow_ev, pow_val) -> str:
        # Session
        r = _http_sess.post(_DS_SESSION, json={},
                            headers={"Authorization": f"Bearer {self._token}"})
        r.raise_for_status()
        j = r.json()
        data = j.get("data")
        if not data or not data.get("biz_data"):
            raise RuntimeError(f"Session create failed: {j.get('code', '?')}")
        sid = data["biz_data"]["id"]

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
        body = {
            "chat_session_id": sid, "parent_message_id": None,
            "model_type": "default", "prompt": prompt,
            "ref_file_ids": [], "thinking_enabled": True,
            "search_enabled": True, "action": None, "preempt": False,
        }
        resp = _http_sess.post(
            _DS_COMPLETION, json=body, stream=False, timeout=180,
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Ds-Pow-Response": pow_val[0],
            },
        )
        resp.raise_for_status()
        resp_text = resp.text

        # Parse SSE with path inheritance
        chunks: list[str] = []
        current_path = ""
        for line in resp_text.split("\n"):
            if not line or not line.startswith("data: "):
                continue
            s = line[6:].strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            if d.get("p"):
                current_path = d["p"]
            v = d.get("v")
            o = d.get("o")
            if current_path == "response/content" and isinstance(v, str):
                if o == "APPEND" or not o:
                    chunks.append(v)
        return "".join(chunks).strip()


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
            _stream_events(worker, prompt_text, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-vercel-ai-data-stream": "v1",
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

async def _stream_events(worker, prompt: str, model: str):
    msg_id = _build_msg_id()
    event_id = 0

    def _event(data: dict):
        nonlocal event_id
        event_id += 1
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield _event({
        "type": "message_start",
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

    if clean_text:
        yield _event({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _event({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": clean_text},
        })
        yield _event({
            "type": "content_block_stop",
            "index": 0,
        })

    tool_offset = 1 if clean_text else 0
    for i, tu in enumerate(tool_uses):
        idx = tool_offset + i
        yield _event({
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu["input"],
            },
        })
        yield _event({
            "type": "content_block_stop",
            "index": idx,
        })

    stop_reason = "tool_use" if tool_uses else "end_turn"
    yield _event({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield "data: [DONE]\n\n"


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

    print(f"\n  DeepSeek Web -> Anthropic API proxy")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Models: GET /v1/models")
    print(f"  Chat:   POST /v1/messages")
    print(f"  Logs: {os.path.abspath(log_path)}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
