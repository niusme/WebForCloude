"""Direct DeepSeek API — PoW solved by hidden browser, chat via HTTP."""
import json, logging, os, sys, threading, time
from typing import Optional
import requests

# ---- ensure project root is on sys.path (script may run standalone) ----
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.pw_env import apply_playwright_browsers_path

apply_playwright_browsers_path()

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

BASE_URL = "https://chat.deepseek.com"
CHALLENGE_URL = f"{BASE_URL}/api/v0/chat/create_pow_challenge"
SESSION_URL = f"{BASE_URL}/api/v0/chat_session/create"
COMPLETION_URL = f"{BASE_URL}/api/v0/chat/completion"
SESSION_FILE = "session/storage_state.json"


def _load_token() -> Optional[str]:
    if not os.path.isfile(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE) as f:
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
        logger.warning("[令牌] 读取失败: %s", exc)
    return None


_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/148.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
})

_pow_ready = threading.Event()
_pow_header: Optional[str] = None


def _make_interceptor():
    """Return a route handler that captures X-Ds-Pow-Response then aborts."""

    def handler(route, request):
        global _pow_header
        if "chat/completion" in request.url:
            h = request.headers
            pov = h.get("x-ds-pow-response")
            if pov:
                _pow_header = pov
                _pow_ready.set()
                logger.info("[验证] 已从浏览器捕获 PoW 响应")
        route.abort()

    return handler


class DeepSeekAPI:
    """PoW solved via hidden browser page; chat via direct HTTP."""

    def __init__(self):
        self._token = _load_token()
        if not self._token:
            raise RuntimeError("请先执行 --mode login 登录")
        self._pw = None
        self._browser = None
        self._page = None
        self._start_browser()

    def _start_browser(self):
        logger.info("[验证] 启动隐藏浏览器计算 PoW ...")
        t0 = time.monotonic()
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        ctx = self._browser.new_context(
            storage_state=SESSION_FILE,
            viewport={"width": 1280, "height": 720},
        )
        self._page = ctx.new_page()
        self._page.route("**/chat/completion**", _make_interceptor())
        self._page.goto(BASE_URL)
        self._page.wait_for_timeout(10000)
        logger.info("[验证] PoW 浏览器就绪（%.1fs）", time.monotonic() - t0)

    def close(self):
        try:
            if self._browser:
                self._browser.close()
        except:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except:
            pass

    def ask(self, message: str, timeout: int = 120):
        """Yield raw Markdown chunks."""
        global _pow_header, _pow_ready
        _pow_header = None
        _pow_ready.clear()

        sid = self._create_session()

        # Trigger PoW solve via the hidden browser
        ta = self._page.query_selector('textarea[name="search"]')
        if ta:
            ta.fill("_")
            self._page.wait_for_timeout(500)
            ta.press("Enter")
            _pow_ready.wait(timeout=30)
            # Start new conversation to reset state
            nc = self._page.query_selector('[class*="_5a8ac7a"]')
            if nc:
                nc.click()
                self._page.wait_for_timeout(2000)

        if not _pow_header:
            raise RuntimeError("PoW solving failed")

        body = {
            "chat_session_id": sid,
            "parent_message_id": None,
            "model_type": "default",
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": True,
            "search_enabled": True,
            "action": None,
            "preempt": False,
        }

        logger.info("[API] 发送消息（%d 字符） ...", len(message))
        t0 = time.monotonic()

        resp = _session.post(
            COMPLETION_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Ds-Pow-Response": _pow_header,
            },
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        full = ""
        last_p = ""

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            s = line[6:].strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except:
                continue
            p = d.get("p") or last_p
            v = d.get("v")
            if p:
                last_p = p
            if p == "response/content" and isinstance(v, str):
                full += v
                yield v

        logger.info("[API] 请求完成（%.1fs，%d 字符）", time.monotonic() - t0, len(full))

    def _create_session(self) -> str:
        r = _session.post(SESSION_URL, json={},
                          headers={"Authorization": f"Bearer {self._token}"})
        r.raise_for_status()
        return r.json()["data"]["biz_data"]["id"]
