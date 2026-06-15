"""DeepSeek Web Chat automation configuration.

Centralized constants for selectors, timeouts, anti-detection settings,
and other tunable parameters. Update this file when the DeepSeek web UI changes.
"""

# ===== URLs =====
BASE_URL = "https://chat.deepseek.com"
LOGIN_URL = f"{BASE_URL}/auth/login"
NEW_CHAT_URL = f"{BASE_URL}/"

# ===== Selectors =====

# Chat input — no 'id' attribute on DeepSeek's textarea, use name="search"
CHAT_INPUT = 'textarea[name="search"]'

# Send button — fallback to Enter key in code (more reliable)
SEND_BUTTON = 'button[aria-label="Send"]'

# Login page
LOGIN_LINK = 'a[href*="login"]'
LOGIN_EMAIL_TAB = 'button:has-text("Email")'
EMAIL_INPUT = 'input[type="email"], input[name="email"], input[placeholder*="Email"]'
PASSWORD_INPUT = (
    'input[type="password"], input[name="password"], '
    'input[placeholder*="Password"]'
)
LOGIN_SUBMIT = 'button[type="submit"], button:has-text("Log in")'

# Post-login indicators
USER_AVATAR = 'img[alt*="avatar"], div[class*="avatar"]'

# Response / conversation elements
PROSE_DIV = 'div.ds-markdown, div[class*="ds-markdown"]'
MESSAGE_CONTAINER = 'div[class*="ds-message"], div[class*="conversation"]'

# Generation state indicators
STOP_BUTTON = 'button:has-text("Stop"), button[aria-label="Stop"]'
STREAMING_INDICATOR = 'div[class*="typing"], div[class*="generating"]'

# New conversation trigger (sidebar "新建对话" button)
NEW_CHAT_BUTTON = '[class*="_5a8ac7a"], button:has-text("New Chat")'

# ===== Timeouts (milliseconds) =====
NAVIGATION_TIMEOUT = 60_000
DEFAULT_WAIT = 30_000
RESPONSE_POLL_INTERVAL = 300
MAX_RESPONSE_WAIT = 180_000
POW_TIMEOUT = 60_000
LOGIN_TIMEOUT = 120_000
STABLE_POLLS_REQUIRED = 2  # consecutive stable polls before content is "done"

# ===== Anti-detection =====
VIEWPORT = {"width": 1280, "height": 720}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-web-security",
    "--disable-features=IsolateOrigins,site-per-process",
]

# ===== Session persistence =====
SESSION_DIR = "session"
SESSION_FILE = "session/storage_state.json"

# ===== Human mimicry =====
TYPING_DELAY_MIN = 30
TYPING_DELAY_MAX = 120
POST_SEND_DELAY = 200
