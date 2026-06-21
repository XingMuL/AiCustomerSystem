"""
记忆融合智能体（Memory Merge Agent）

将多段记忆（来自不同 Agent 的上下文）融合为一致的上下文。
"""

from loguru import logger
from openai import OpenAI

from backend.config import settings


class MemoryMergeAgent:
    """记忆融合智能体"""

    def __init__(self):
        self._client: OpenAI = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
        return self._client

    def merge_memory(
        self,
        session_memory: str,
        rag_memory: str,
        degradation_level: int = 0,
    ) -> str:
        """
        融合会话记忆和 RAG 召回记忆

        Args:
            session_memory: 会话记忆文本
            rag_memory: RAG 召回记忆文本
            degradation_level: 降级等级

        Returns:
            str: 融合后的上下文
        """
        # L3/4: RAG 不可用，只用会话记忆
        if degradation_level >= 3:
            logger.info("[Memory Merge] RAG 不可用，仅使用会话记忆")
            return session_memory

        if not rag_memory:
            return session_memory

        if not session_memory:
            return rag_memory

        # 简单拼接（大多数情况下不需要 LLM 融合）
        return session_memory + "\n\n" + rag_memory