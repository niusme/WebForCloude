"""Resolve which Playwright browsers directory to use.

Priority:
  1. ``PLAYWRIGHT_BROWSERS_PATH`` already present in the environment.
  2. A project-local ``browsers/`` folder that actually contains browsers.
  3. Playwright's own default resolution (``%USERPROFILE%\\AppData\\Local\\ms-playwright``
     on Windows, or the platform equivalent).

The previous behaviour unconditionally pointed Playwright at ``<project>/browsers``,
which broke startup with "Executable doesn't exist" whenever that folder was
absent — even though the default Playwright install was perfectly usable.
"""

import os
from typing import Optional

#: Project-local browser binaries (created by
#: ``PLAYWRIGHT_BROWSERS_PATH=<this> playwright install``).
PROJECT_BROWSERS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "browsers")
)


def _contains_browsers(path: str) -> bool:
    """True when *path* exists and holds at least one entry."""
    try:
        if not os.path.isdir(path):
            return False
        with os.scandir(path) as entries:
            return any(True for _ in entries)
    except OSError:
        return False


def apply_playwright_browsers_path() -> Optional[str]:
    """Set ``PLAYWRIGHT_BROWSERS_PATH`` only when a local install exists.

    Returns the path that was applied, or ``None`` when Playwright should be
    left to resolve its default location.
    """
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_path:
        return env_path

    if _contains_browsers(PROJECT_BROWSERS_PATH):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = PROJECT_BROWSERS_PATH
        return PROJECT_BROWSERS_PATH

    return None


def describe_playwright_browsers_path() -> str:
    """Human-readable description of the effective browsers path (for logs)."""
    return os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "(Playwright 默认路径)"
