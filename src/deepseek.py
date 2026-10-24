"""Core DeepSeek Web Chat automation client

Provides a high-level API (DeepSeekClient) for sending messages and
receiving responses via chat.deepseek.com using Playwright
"""

import hashlib
import html2text
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PwTimeout

from src.browser import BrowserManager
from src.exceptions import (
    ChatInputError,
    DeepSeekError,
    LoginError,
    PoWTimeoutError,
    ResponseExtractionError,
    ResponseTimeoutError,
    SessionExpiredError,
)
from src.timing import _tic, _toc

# HTML → Markdown converter
_md_converter = html2text.HTML2Text()
_md_converter.body_width = 0       # don't wrap lines
_md_converter.ignore_links = False
_md_converter.ignore_images = False
_md_converter.ignore_emphasis = False
_md_converter.protect_links = True
_md_converter.unicode_snob = True  # use Unicode instead of ASCII
from src.config import (
    BASE_URL,
    CHAT_INPUT,
    LOGIN_LINK,
    LOGIN_EMAIL_TAB,
    EMAIL_INPUT,
    PASSWORD_INPUT,
    LOGIN_SUBMIT,
    USER_AVATAR,
    PROSE_DIV,
    STOP_BUTTON,
    NEW_CHAT_BUTTON,
    DEFAULT_WAIT,
    RESPONSE_POLL_INTERVAL,
    MAX_RESPONSE_WAIT,
    POW_TIMEOUT,
    LOGIN_TIMEOUT,
    TYPING_DELAY_MIN,
    TYPING_DELAY_MAX,
    POST_SEND_DELAY,
    STABLE_POLLS_REQUIRED,
    SESSION_FILE,
)

logger = logging.getLogger(__name__)

# ---- timing helper ----
# ===== Client =====


class DeepSeekClient:
    """Automation client for chat.deepseek.com

    Usage (context manager — recommended):
        with DeepSeekClient(headless=True) as client:
            resp = client.ask("Hello!")
            print(resp)

    Usage (manual lifecycle):
        client = DeepSeekClient()
        try:
            client.start()
            resp = client.ask("Hello!")
        finally:
            client.close()
    """

    def __init__(
        self,
        headless: bool = True,
        session_path: str = SESSION_FILE,
        timeout: int = MAX_RESPONSE_WAIT,
    ):
        self.headless = headless
        self.session_path = session_path
        self.timeout = timeout
        self._browser_mgr: Optional[BrowserManager] = None
        self.page: Optional[Page] = None
        self._message_count = 0

    # ---- Context manager support ----

    def __enter__(self) -> "DeepSeekClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---- Lifecycle ----

    def start(self) -> None:
        """Launch browser and navigate to DeepSeek."""
        _tic()
        self._browser_mgr = BrowserManager(headless=self.headless)
        self.page = self._browser_mgr.start()
        self._message_count = 0
        _toc("浏览器启动完成")

        _tic()
        logger.info("[导航] 正在导航到 %s", BASE_URL)
        self.page.goto(BASE_URL, wait_until="domcontentloaded")
        _toc("初始导航")

        self._wait_for_page_ready()
        logger.info("[导航] 首次页面加载完成. URL: %s", self.page.url)

    def close(self) -> None:
        """Save session state and release browser resources."""
        if self._browser_mgr and self.page:
            _tic()
            try:
                self._browser_mgr.save_session(self.session_path)
            except Exception as exc:
                logger.warning("关闭时保存会话失败: %s", exc)
            _toc("会话已保存")
        if self._browser_mgr:
            self._browser_mgr.close()
            self._browser_mgr = None
            self.page = None
        logger.info("[生命周期] 客户端已关闭")

    # ---- Authentication ----

    #: localStorage key DeepSeek uses for the session token.
    TOKEN_KEY = "userToken"

    @staticmethod
    def _token_from_raw(raw: Optional[str]) -> Optional[str]:
        """Normalise a localStorage value into a non-empty token string."""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return raw.strip() or None
        if isinstance(parsed, dict):
            value = parsed.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None
        return str(parsed).strip() or None

    def _read_user_token(self) -> Optional[str]:
        """Return DeepSeek's auth token from localStorage, or None.

        Checks ``userToken`` first, then falls back to any other localStorage
        key that looks like a token — DeepSeek has renamed/added keys before.
        """
        if not self.page:
            return None

        try:
            raw = self.page.evaluate(
                "() => { try { return localStorage.getItem('userToken'); }"
                " catch (e) { return null; } }"
            )
        except Exception as exc:
            logger.debug("[认证] 读取 userToken 失败: %s", exc)
            return None

        token = self._token_from_raw(raw)
        if token:
            return token

        # Fallback: any token-ish key other than the one we already checked.
        try:
            entries = self.page.evaluate(
                "() => Object.entries(localStorage)"
                ".filter(([k, v]) => /token/i.test(k) && v"
                " && k !== 'userToken')"
            )
        except Exception:
            return None

        for key, value in entries or []:
            token = self._token_from_raw(value)
            if token:
                logger.debug("[认证] 备用令牌键: %s", key)
                return token
        return None

    def _page_closed(self) -> bool:
        """True when the browser page/window is gone."""
        try:
            return self.page is None or self.page.is_closed()
        except Exception:
            return True

    def _localstorage_keys(self) -> list:
        """List localStorage keys (for diagnostics only)."""
        try:
            return list(self.page.evaluate("() => Object.keys(localStorage)") or [])
        except Exception:
            return []

    def is_logged_in(self, strict: bool = False) -> bool:
        """Check if the current session appears authenticated.

        ``strict=True`` trusts only a real auth token in localStorage.  The
        other heuristics (avatar / visible chat input) are also true on the
        public logged-out page, which made ``--mode login`` believe the user
        had already signed in and close the window immediately.
        """
        if not self.page:
            return False

        # Method 1: localStorage token — the only reliable signal
        _tic()
        if self._read_user_token():
            _toc("[认证] 检测到 localStorage 令牌")
            return True

        if strict:
            logger.debug("[认证] strict 模式：未发现 userToken")
            return False

        # Method 2: Avatar element
        try:
            avatar = self.page.query_selector(USER_AVATAR)
            if avatar and avatar.is_visible():
                logger.debug("[认证] 头像检查通过")
                return True
        except Exception:
            pass

        # Method 3: Chat input visible
        try:
            inp = self.page.query_selector(CHAT_INPUT)
            if inp and inp.is_visible() and inp.is_enabled():
                logger.debug("[认证] 输入框检查通过")
                return True
        except Exception:
            pass

        logger.debug("[认证] 未检测到登录状态")
        return False


    def login(self, timeout: Optional[int] = None) -> bool:
        """Open the browser and let the user log in manually."""
        if not self.page:
            raise LoginError("No active page. Call start() first.")

        timeout_ms = (timeout or LOGIN_TIMEOUT // 1000) * 1000
        deadline = time.monotonic() + timeout_ms / 1000

        current_url = self.page.url
        if "chat.deepseek.com" not in current_url:
            self.page.goto(BASE_URL, wait_until="domcontentloaded")
            self._wait_for_page_ready()

        print()
        print("=" * 60)
        print("  ** DeepSeek Login **")
        print()
        print("  Please log in manually in the browser window.")
        print("  Supported: WeChat scan / Google / Email etc.")
        print("  提示：DeepSeek 允许游客匿名使用，页面可能直接显示聊天框；")
        print("        请点击右上角 / 侧栏的「登录」完成账号登录。")
        print()
        print(f"  Timeout: {timeout_ms // 1000} seconds")
        print("  The window stays open until a real login token is found.")
        print("  Session will be saved automatically after login.")
        print("=" * 60)
        print()

        _tic()
        heartbeat_at = 0.0
        while time.monotonic() < deadline:
            if self._page_closed():
                self._debug_screenshot("login_window_closed")
                raise LoginError(
                    "浏览器窗口已被关闭，登录未完成。请重新运行 --mode login。"
                )

            # Only a real auth token counts — the logged-out page also shows
            # a chat box, so heuristics would end the wait too early.
            if self.is_logged_in(strict=True):
                _toc("[登录] 用户完成登录")
                logger.info("检测到登录 — saving session.")
                self._browser_mgr.save_session(self.session_path)
                print()
                print("=" * 60)
                print("  ** Login successful! 会话已保存. **")
                print("  You may close the browser window now.")
                print("=" * 60)
                print()
                return True

            now = time.monotonic()
            if now - heartbeat_at >= 15:
                heartbeat_at = now
                try:
                    url = self.page.url
                except Exception:
                    url = "?"
                print(
                    f"  [等待登录] 已等待 {int(now - (deadline - timeout_ms / 1000))}s"
                    f" / {timeout_ms // 1000}s  当前页面: {url}"
                )

            try:
                self.page.wait_for_timeout(1000)
            except Exception as exc:
                if self._page_closed():
                    raise LoginError(
                        "浏览器窗口已被关闭，登录未完成。请重新运行 --mode login。"
                    ) from exc
                logger.debug("[登录] 轮询失败: %s", exc)

        self._debug_screenshot("login_timeout")
        keys = self._localstorage_keys()
        if keys:
            logger.warning("[登录] 超时未发现 userToken，localStorage 键: %s", keys)
        raise LoginError(
            f"Login timeout ({timeout_ms // 1000}s). "
            "Run --mode login again and complete the login."
        )

    def ensure_logged_in(self) -> None:
        """Verify login state; raise SessionExpiredError if not logged in."""
        if not self.page:
            raise SessionExpiredError("No active page. Call start() first.")

        _tic()
        if self.is_logged_in():
            _toc("[认证] 确认登录状态")
            return

        if self.headless:
            raise SessionExpiredError(
                "Session expired or not logged in. "
                "Run with --mode login (headed) to re-authenticate."
            )

        logger.info(
            "未登录，请在浏览器窗口中手动登录。"
        )
        deadline = time.monotonic() + LOGIN_TIMEOUT / 1000
        while time.monotonic() < deadline:
            if self.is_logged_in():
                logger.info("检测到手动登录.")
                self._browser_mgr.save_session(self.session_path)
                return
            self.page.wait_for_timeout(1000)

        raise SessionExpiredError("Manual login not completed in time.")

    # ---- Core chat actions ----

    def send_message(self, message: str) -> None:
        """Type a message into the chat input and send it via Enter

        Raises ChatInputError if the textarea is not available
        """
        if not self.page:
            raise ChatInputError("No active page. Call start() first.")

        logger.debug("[发送] 消息长度: %d chars", len(message))
        _tic()

        self.ensure_logged_in()

        try:
            chat_input = self.page.wait_for_selector(
                CHAT_INPUT, state="visible", timeout=DEFAULT_WAIT
            )
            if not chat_input:
                raise ChatInputError("Chat input not found on page.")
        except PwTimeout as exc:
            self._debug_screenshot("chat_input_missing")
            raise ChatInputError(
                "Chat input (#chat-input) not found after waiting."
            ) from exc
        _toc("[发送] wait_for_selector(textarea)")

        _tic()
        chat_input.click()
        chat_input.fill("")
        self.page.wait_for_timeout(200)
        chat_input.fill(message)
        _toc(f"[发送] fill text ({len(message)} chars)")

        _tic()
        pause = random.randint(200, 600)
        self.page.wait_for_timeout(pause)

        chat_input.press("Enter")
        self._message_count += 1
        _toc(f"[发送] press Enter (after {pause}ms pause)")

        _tic()
        self.page.wait_for_timeout(POST_SEND_DELAY)
        _toc("[发送] post-send delay")

    def wait_for_response(self) -> str:
        """Wait for the AI response to finish generating and return its text

        Uses a polling algorithm that detects content stability:
        1. Records current prose div count as baseline
        2. Polls for new content
        3. Once new content stops changing (STABLE_POLLS_REQUIRED checks),
           considers the response complete

        Returns the extracted response text
        Raises ResponseTimeoutError or ResponseExtractionError on failure
        """
        if not self.page:
            raise ResponseTimeoutError("No active page. Call start() first.")

        logger.info("[等待] 等待 DeepSeek 响应 ...")
        _tic()

        prose_count_before = self._count_prose_divs()
        logger.debug("[等待] 发送前 prose 容器数: %d", prose_count_before)

        if not self._wait_for_generation_start(prose_count_before):
            logger.debug(
                "[等待] 未检测到生成开始; polling anyway."
            )

        # Phase 2: Poll until content stable
        deadline = time.monotonic() + self.timeout / 1000
        stable_count = 0
        last_text_hash = ""
        last_text = ""
        poll_round = 0
        total_polls = 0
        first_content_seen = False
        first_content_time = 0.0
        generation_end_time = 0.0
        stop_btn_seen = False
        stop_btn_lost_time = 0.0

        logger.info("[等待] 开始轮询 | prose_before=%d deadline=%.1fs",
                     prose_count_before, deadline - time.monotonic())

        while time.monotonic() < deadline:
            poll_round += 1
            total_polls += 1
            try:
                prose_divs = self.page.query_selector_all(PROSE_DIV)
                current_count = len(prose_divs)
                now = time.monotonic()

                # Track stop button lifecycle
                stop_btn = self.page.query_selector(STOP_BUTTON)
                if stop_btn and stop_btn.is_visible():
                    if not stop_btn_seen:
                        stop_btn_seen = True
                        logger.info("[等待] 停止按钮出现 at poll %d (%.1fs)",
                                    total_polls, now - (deadline - self.timeout / 1000))
                elif stop_btn_seen and stop_btn_lost_time == 0:
                    stop_btn_lost_time = now
                    logger.info("[等待] 停止按钮消失 at poll %d (%.1fs) — generation may be done",
                                total_polls, now - (deadline - self.timeout / 1000))

                if current_count > prose_count_before:
                    current_text = (prose_divs[-1].inner_text() or "").strip()
                    text_hash = hashlib.md5(current_text.encode()).hexdigest()

                    if not first_content_seen:
                        first_content_seen = True
                        first_content_time = now
                        logger.info("[等待] 首次检测到内容 at poll %d (%.1fs, len=%d)",
                                    total_polls, now - (deadline - self.timeout / 1000),
                                    len(current_text))

                    if text_hash == last_text_hash and current_text:
                        stable_count += 1
                        if stable_count == 1:
                            logger.info("[等待] 内容停止变化 at poll %d (%.1fs, len=%d, %.1fs after first content)",
                                        total_polls, now - (deadline - self.timeout / 1000),
                                        len(current_text), now - first_content_time)
                        if stable_count >= STABLE_POLLS_REQUIRED:
                            generation_time = now - first_content_time
                            elapsed = now - (deadline - self.timeout / 1000)
                            logger.info(
                                "[等待] 稳定: %d次轮询 | 总耗时%.1fs 生成耗时%.1fs 稳定等待%.1fs 文本长度=%d",
                                total_polls, elapsed, generation_time,
                                STABLE_POLLS_REQUIRED * (RESPONSE_POLL_INTERVAL / 1000),
                                len(current_text)
                            )
                            _toc("[等待] 总等待时间")
                            # Convert rendered HTML back to Markdown
                            try:
                                html_content = (prose_divs[-1].inner_html() or "")
                                if html_content.strip():
                                    markdown = _md_converter.handle(html_content).strip()
                                    if markdown:
                                        return markdown
                            except Exception:
                                pass
                            return current_text
                    else:
                        if stable_count > 0:
                            logger.debug(
                                "[等待] 内容发生变化 (was stable for %d polls)",
                                stable_count
                            )
                        stable_count = 0
                        last_text_hash = text_hash
                        last_text = current_text
                else:
                    # No new prose div yet
                    if stop_btn_seen:
                        stable_count = 0  # 仍在生成

                # Log every 10 polls
                if total_polls % 10 == 0:
                    logger.info("[等待] 持续轮询 ... round=%d stable=%d text_len=%d elapsed=%.1fs",
                                total_polls, stable_count, len(last_text),
                                now - (deadline - self.timeout / 1000))

                self.page.wait_for_timeout(RESPONSE_POLL_INTERVAL)
            except Exception as exc:
                logger.warning(
                    "[等待] 轮询 %d 次时页面交互失败: %s",
                    total_polls, exc
                )
                break

        elapsed = time.monotonic() - (deadline - self.timeout / 1000)
        logger.warning(
            "[等待] 轮询超时: %.1fs（%d 次轮询）", elapsed, total_polls
        )

        fallback_text = self._fallback_extract(prose_count_before)
        if fallback_text:
            logger.info("[等待] 后备提取成功.")
            return fallback_text

        self._debug_screenshot("response_timeout")
        raise ResponseTimeoutError(
            f"No response received within {self.timeout // 1000}s. "
            f"Screenshot saved to session/debug_response_timeout_*.png"
        )

    def ask(self, message: str) -> str:
        """Convenience: send a message and return the response."""
        logger.info("[对话] 开始请求 (%d chars)", len(message))
        _tic()
        result = self._ask_raw_markdown(message)
        if result is not None:
            total = _toc("[对话] 总往返耗时（SSE）")
            logger.info("[对话] 响应长度: %d 字符（SSE）", len(result))
            return result
        # Fallback to DOM polling
        self.send_message(message)
        result = self.wait_for_response()
        total = _toc("[对话] 总往返耗时（DOM）")
        logger.info("[对话] 响应长度: %d 字符（DOM）", len(result))
        return result

    def _ask_raw_markdown(self, message: str) -> Optional[str]:
        """Send message and capture raw Markdown via SSE stream

        Intercepts the /api/v0/chat/completion SSE response on the
        DeepSeek page to get the raw model output (with Markdown intact)
        instead of reading rendered HTML from the DOM
        """
        if not self.page:
            raise ChatInputError("No active page. Call start() first.")

        self.ensure_logged_in()

        try:
            chat_input = self.page.wait_for_selector(
                CHAT_INPUT, state="visible", timeout=DEFAULT_WAIT
            )
            if not chat_input:
                raise ChatInputError("Chat input not found.")
        except PwTimeout as exc:
            self._debug_screenshot("chat_input_missing")
            return None

        # Clear and fill
        try:
            chat_input.click()
            chat_input.fill("")
            self.page.wait_for_timeout(200)
            chat_input.fill(message)

            # Set up SSE capture BEFORE pressing Enter
            with self.page.expect_response(
                lambda r: "chat/completion" in r.url
            ) as resp_info:
                chat_input.press("Enter")
                self._message_count += 1

            # Wait for SSE stream to complete
            body = resp_info.value.body()
            raw = body.decode("utf-8", errors="replace")

            # Parse RESPONSE fragment content
            chunks: list[str] = []
            in_response = False
            last_p = ""

            for line in raw.split("\n"):
                if not line.startswith("data: "):
                    continue
                s = line[6:].strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                except json.JSONDecodeError:
                    continue

                p = data.get("p") or last_p
                v = data.get("v")
                o = data.get("o", "")
                if p:
                    last_p = p

                if o == "APPEND" and p == "response/fragments" and isinstance(v, list):
                    for frag in v:
                        if isinstance(frag, dict) and frag.get("type") == "RESPONSE":
                            in_response = True
                            init = frag.get("content", "")
                            if init:
                                chunks.append(init)

                if in_response and isinstance(v, str) and p and "/content" in p:
                    chunks.append(v)

            result = "".join(chunks).strip()
            if result:
                return result
            return None

        except Exception as exc:
            logger.warning("[SSE流] 失败: %s", exc)
            return None

    def start_new_conversation(self) -> None:
        """Start a fresh conversation by clicking the New Chat sidebar button

        Much faster than navigating to BASE_URL (~3s vs ~15s)
        """
        if not self.page:
            raise DeepSeekError("No active page. Call start() first.")

        _tic()
        try:
            btn = self.page.wait_for_selector(
                NEW_CHAT_BUTTON, timeout=10_000
            )
            if btn and btn.is_visible():
                btn.click()
                self.page.wait_for_timeout(2000)
                _toc("[导航] 新对话按钮已点击")
                self._message_count = 0
                logger.info("[导航] 通过按钮新建对话")
                return
        except Exception:
            logger.debug("[导航] 未找到按钮，降级为导航")

        self.page.goto(BASE_URL, wait_until="domcontentloaded")
        _toc("[导航] goto BASE_URL (fallback)")

        self._wait_for_page_ready()
        self._message_count = 0
        self.page.wait_for_timeout(500)
        logger.info("[导航] 通过导航新建对话")

    # ---- Private helpers ----

    def _type_humanlike(self, element, text: str) -> None:
        """Type text into an element with random delays between characters."""
        for char in text:
            element.type(char, delay=random.randint(TYPING_DELAY_MIN, TYPING_DELAY_MAX))

    def _count_prose_divs(self) -> int:
        """Return the number of prose divs currently on the page."""
        try:
            return len(self.page.query_selector_all(PROSE_DIV))
        except Exception:
            return 0

    def _wait_for_generation_start(self, prose_count_before: int) -> bool:
        """Poll briefly to detect when AI generation begins."""
        _tic()
        deadline = time.monotonic() + 10
        signals = {"stop_btn": False, "prose_div": False, "disabled": False}

        while time.monotonic() < deadline:
            stop_btn = self.page.query_selector(STOP_BUTTON)
            if stop_btn and stop_btn.is_visible():
                signals["stop_btn"] = True
                _toc("[等待] 生成开始（停止按钮）")
                return True

            if self._count_prose_divs() > prose_count_before:
                signals["prose_div"] = True
                _toc("[等待] 生成开始（prose 容器）")
                return True

            inp = self.page.query_selector(CHAT_INPUT)
            if inp and not inp.is_enabled():
                signals["disabled"] = True
                _toc("[等待] 生成开始（输入框禁用）")
                return True

            self.page.wait_for_timeout(RESPONSE_POLL_INTERVAL)

        _toc("[等待] 未检测到生成开始（10秒超时）")
        logger.debug("[等待] 信号: %s", signals)
        return False

    def _fallback_extract(self, prose_count_before: int) -> Optional[str]:
        """Fallback: try to extract response text from the DOM."""
        try:
            prose_divs = self.page.query_selector_all(PROSE_DIV)
            if len(prose_divs) > prose_count_before:
                # Try Markdown first
                try:
                    html = (prose_divs[-1].inner_html() or "")
                    if html.strip():
                        md = _md_converter.handle(html).strip()
                        if md:
                            return md
                except Exception:
                    pass
                text = (prose_divs[-1].inner_text() or "").strip()
                if text:
                    return text
        except Exception:
            pass

        try:
            text = self.page.evaluate(
                """() => {
                    const messages = document.querySelectorAll(
                        'div[class*="message"]'
                    );
                    if (messages.length === 0) return '';
                    const last = messages[messages.length - 1];
                    return last.innerText || '';
                }"""
            )
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        return None

    def _wait_for_page_ready(self) -> None:
        """Wait for the page to fully load after PoW challenge."""
        _tic()
        logger.info("[导航] 等待页面稳定（PoW + JS） (PoW + JS)...")
        try:
            self.page.wait_for_load_state("networkidle", timeout=POW_TIMEOUT)
            _toc("[导航] networkidle reached")
        except Exception:
            _toc("[导航] networkidle timeout (continuing)")
            logger.debug(
                "网络空闲状态未达到 within %dms; continuing.",
                POW_TIMEOUT,
            )
        self.page.wait_for_timeout(2000)
        logger.info(
            "[导航] 页面已稳定 (URL: %s)", self.page.url
        )

    def _debug_screenshot(self, name: str = "debug") -> str:
        """Save a screenshot for debugging. Returns the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("session", exist_ok=True)
        path = f"session/debug_{name}_{timestamp}.png"
        try:
            if self.page:
                self.page.screenshot(path=path, full_page=True)
                logger.info("调试截图已保存 to %s", path)
        except Exception as exc:
            logger.warning("保存调试截图失败: %s", exc)
        return path
