"""
输入预处理模块

对用户输入进行规范化、截断、刷屏检测等预处理。
"""

import re
import hashlib
from loguru import logger

from backend.config import settings


class InputCleaner:
    """输入清洗与安全检测"""

    def __init__(self):
        self.max_length = settings.max_sentence_length
        self.spam_threshold = settings.spam_threshold
        # 刷屏检测：记录每个会话的最近消息哈希
        self._recent_hashes: dict[str, list[str]] = {}

    def clean(self, raw_input: str, session_id: str = "") -> dict:
        """
        清洗用户输入

        Args:
            raw_input: 原始输入
            session_id: 会话 ID（用于刷屏检测）

        Returns:
            dict: {"cleaned": str, "is_spam": bool, "tokens": int}
        """
        text = raw_input.strip()

        # 空输入检测
        if not text:
            return {"cleaned": "", "is_spam": False, "tokens": 0}

        # 去重/刷屏检测
        is_spam = self._detect_spam(text, session_id)

        # 超长截断
        if len(text) > self.max_length * 3:
            text = text[:self.max_length * 3] + "..."

        # 规范空白
        text = re.sub(r"\s+", " ", text)

        # 移除控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 估算 token 数
        tokens = self._estimate_tokens(text)

        return {
            "cleaned": text,
            "is_spam": is_spam,
            "tokens": tokens,
        }

    def _detect_spam(self, text: str, session_id: str) -> bool:
        """刷屏检测"""
        if not session_id:
            return False

        text_hash = hashlib.md5(text.encode()).hexdigest()

        if session_id not in self._recent_hashes:
            self._recent_hashes[session_id] = []

        history = self._recent_hashes[session_id]
        history.append(text_hash)

        if len(history) > self.spam_threshold * 2:
            history = history[-(self.spam_threshold * 2):]
            self._recent_hashes[session_id] = history

        recent = history[-self.spam_threshold:]
        if len(recent) >= self.spam_threshold and len(set(recent)) == 1:
            logger.warning(f"[输入] 检测到刷屏: session={session_id}")
            return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)