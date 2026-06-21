"""
记忆管理器

管理会话记忆、滑动窗口压缩、记忆与 RAG 结果融合。
"""

from loguru import logger

from backend.config import settings


class MemoryManager:
    """会话记忆管理器"""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self.max_tokens = settings.max_memory_tokens
        self.min_turns = settings.min_history_turns

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史"""
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, user_message: str, assistant_message: str):
        """添加一轮对话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": "user",
            "content": user_message,
        })
        self._sessions[session_id].append({
            "role": "assistant",
            "content": assistant_message,
        })

        # 压缩超长的历史
        self._compress_if_needed(session_id)

    def format_for_llm(self, session_id: str) -> list[dict]:
        """格式化为 LLM 可用的消息列表"""
        history = self.get_history(session_id)
        # 保留最近 N 轮
        max_messages = self.min_turns * 2  # 每轮 user + assistant
        if len(history) > max_messages:
            history = history[-max_messages:]
        return history

    def merge_with_rag(self, session_id: str, rag_docs: list[dict]) -> str:
        """
        将 RAG 召回内容与记忆融合，生成上下文文本

        Args:
            session_id: 会话ID
            rag_docs: RAG 召回的文档列表

        Returns:
            str: 融合后的上下文
        """
        parts = []

        # 对话历史
        history = self.get_history(session_id)
        if history:
            history_text = "\n".join([
                f"{'用户' if m['role'] == 'user' else '客服'}: {m['content']}"
                for m in history[-6:]  # 最近3轮
            ])
            parts.append(f"【对话历史】\n{history_text}")

        # RAG 文档
        if rag_docs:
            docs_text = "\n---\n".join([
                d.get("content", "")[:500] for d in rag_docs[:settings.rag_recall_top_k]
            ])
            parts.append(f"【相关知识】\n{docs_text}")

        return "\n\n".join(parts)

    def _compress_if_needed(self, session_id: str):
        """在 token 超限时压缩历史"""
        history = self._sessions.get(session_id, [])
        if not history:
            return

        # 简单估算：按字符数 * 0.5 近似 token 数
        total_chars = sum(len(m.get("content", "")) for m in history)
        estimated_tokens = total_chars * 0.5

        if estimated_tokens > self.max_tokens:
            # 保留最少轮数，丢弃中间的轮次（保留头尾）
            keep_turns = self.min_turns
            keep_messages = keep_turns * 2
            if len(history) > keep_messages:
                # 保留最近的消息和最老的一轮（保持上下文连贯）
                latest = history[-(keep_messages - 2):]
                earliest = history[:2]  # 第一轮
                self._sessions[session_id] = earliest + latest
                logger.info(f"[记忆] 会话 {session_id} 历史已压缩，保留 {len(self._sessions[session_id])} 条")

    def clear(self, session_id: str):
        """清除会话记忆"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def clear_session(self, session_id: str):
        """清除会话记忆（别名）"""
        self.clear(session_id)