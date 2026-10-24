"""Browser lifecycle manager for DeepSeek Web Chat automation.

Handles Playwright launch/teardown, anti-detection patches,
and session persistence via storage_state.
"""

import os
import logging
from typing import Optional

from src.pw_env import (
    PROJECT_BROWSERS_PATH,
    apply_playwright_browsers_path,
    describe_playwright_browsers_path,
)
from src.exceptions import DeepSeekError
from src.timing import _tic, _toc

# Prefer project-local browser binaries, but fall back to Playwright's default
# install when they are missing (otherwise launch dies with "Executable doesn't
# exist"). Must run before any browser object is launched.
apply_playwright_browsers_path()

from playwright.sync_api import (
    Playwright,
    Browser,
    BrowserContext,
    Page,
    sync_playwright,
)

from src.config import (
    VIEWPORT,
    USER_AGENT,
    LAUNCH_ARGS,
    SESSION_DIR,
    SESSION_FILE,
    NAVIGATION_TIMEOUT,
)

logger = logging.getLogger(__name__)

ANTI_DETECTION_SCRIPT = """
// Hide Playwright automation indicators
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// Restore non-empty plugins array
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
        { name: 'Native Client', filename: 'pnacl' },
    ],
});

// Set realistic language preferences
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
});

// Patch chrome.runtime to prevent detection
window.chrome = window.chrome || {};
window.chrome.runtime = {};

// Override permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) => (
    params.name === 'notifications'
        ? Promise.resolve({ state: 'prompt' })
        : originalQuery(params)
);
"""


class BrowserManager:
    """Manages a Playwright Chromium instance with anti-detection and session persistence.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # ---- Public API ----

    def start(self) -> Page:
        """Launch the browser, create a context & page, and apply anti-detection."""
        logger.info(
            "[浏览器] 启动中 (headless=%s, 浏览器路径=%s)",
            self.headless, describe_playwright_browsers_path(),
        )
        _tic()
        self._playwright = sync_playwright().start()
        _toc("Playwright 引擎启动")

        _tic()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=LAUNCH_ARGS,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise DeepSeekError(
                    "Chromium 未安装或路径无效。请任选一种方式修复：\n"
                    "  1) 安装到 Playwright 默认目录:  python -m playwright install chromium\n"
                    "  2) 安装到项目内目录:          "
                    '$env:PLAYWRIGHT_BROWSERS_PATH="'
                    f"{PROJECT_BROWSERS_PATH}"
                    '"; python -m playwright install chromium\n'
                    f"原始错误: {message}"
                ) from exc
            raise DeepSeekError(f"浏览器启动失败: {message}") from exc
        _toc("Chromium 浏览器启动")

        _tic()
        storage_state = self._resolve_storage_state()
        self._context = self._browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            storage_state=storage_state or None,
            no_viewport=False,
        )
        _toc("浏览器上下文创建")

        self._page = self._context.new_page()
        self._page.set_default_timeout(NAVIGATION_TIMEOUT)
        logger.debug("[浏览器] 默认超时时间: %dms", NAVIGATION_TIMEOUT)

        _tic()
        self._page.add_init_script(ANTI_DETECTION_SCRIPT)
        self._page.add_init_script("""
            try {
                localStorage.setItem('thinkingEnabled',
                    JSON.stringify({"value": true, "__version": "2"}));
            } catch(e) {}
        """)
        _toc("反检测脚本注入 x2")

        logger.info(
            "[浏览器] 启动完成 (UA: %s)", USER_AGENT[:60]
        )
        return self._page

    def close(self) -> None:
        """Shut down the browser and release resources."""
        logger.info("[浏览器] 正在关闭 ...")
        _tic()

        try:
            if self._page:
                self._page.close()
                logger.debug("[浏览器] 页面已关闭")
        except Exception as exc:
            logger.warning("关闭页面失败: %s", exc)

        try:
            if self._context:
                self._context.close()
                logger.debug("[浏览器] 上下文已关闭")
        except Exception as exc:
            logger.warning("关闭上下文失败: %s", exc)

        try:
            if self._browser:
                self._browser.close()
                logger.debug("[浏览器] 浏览器已关闭")
        except Exception as exc:
            logger.warning("关闭浏览器失败: %s", exc)

        try:
            if self._playwright:
                self._playwright.stop()
                logger.debug("[浏览器] Playwright 引擎已停止")
        except Exception as exc:
            logger.warning("[浏览器] 停止 Playwright 引擎失败: %s", exc)

        _toc("[浏览器] 关闭完成")

    def save_session(self, path: str = SESSION_FILE) -> None:
        """Persist the current browser context's storage state to disk."""
        if not self._context:
            logger.warning("[会话] 无活动上下文，无法保存会话")
            return
        _tic()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._context.storage_state(path=path)
        _toc(f"[会话] 已保存至 {path}")

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def context(self) -> Optional[BrowserContext]:
        return self._context

    # ---- Private helpers ----

    def _resolve_storage_state(self) -> Optional[dict]:
        """Load saved storage state from disk if it exists."""
        if os.path.isfile(SESSION_FILE):
            size = os.path.getsize(SESSION_FILE)
            logger.info(
                "[会话] 发现已保存的登录状态: %s (%d 字节)",
                SESSION_FILE, size,
            )
            return SESSION_FILE
        logger.info("[会话] 未找到已保存的登录状态（首次运行？）")
        return None
