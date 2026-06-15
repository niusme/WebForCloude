"""共享计时工具 — 单一事实来源。"""

import time
import logging

logger = logging.getLogger(__name__)

_t0: float = 0.0


def _tic():
    """开始计时，记录当前时间点。"""
    global _t0
    _t0 = time.monotonic()


def _toc(label: str):
    """停止计时并输出耗时日志。

    Args:
        label: 阶段标识，用于区分不同计时点
    Returns:
        float: 经过的秒数
    """
    elapsed = time.monotonic() - _t0
    logger.info("[计时] %s 耗时: %.2fs", label, elapsed)
    return elapsed
