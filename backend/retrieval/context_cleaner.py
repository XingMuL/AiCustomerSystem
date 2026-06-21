"""
上下文清洗器（Context Cleaner）

在 RAG 检索后，用 DeepSeek 清洗检索到的上下文片段：
- 过滤与用户问题无关的语义信息
- 提取问题相关的核心内容
- 去除冗余描述、无关段落、广告/营销话术等噪声

清洗后的上下文将用于：
1. kb_qa_agent 生成回答（更精准的知识来源）
2. summary_agent 汇总输出（避免无关信息干扰）
3. 上下文相关性评估（更准确的评估基准）
"""

from typing import Optional

from loguru import logger
from openai import OpenAI

from backend.config import settings

CLEANER_SYSTEM_PROMPT = """你是一个知识库文档清洗助手。你的任务是从检索到的知识库文档中，提取与用户问题直接相关的核心内容，删除无关信息。

【清洗规则】
1. 只保留与用户问题直接相关的内容，删除不相关的段落、句子
2. 如果一段内容包含多个知识点，只保留与问题相关的部分
3. 保留具体的数据、数字、联系方式、流程步骤等关键信息
4. 去除广告、营销话术、免责声明、法律条款等与问题无关的文本
5. 去除明显的页面导航、页脚、版权信息等模板内容
6. 保持原文的表述方式，不要改写或总结
7. 如果某段内容与问题完全无关，直接标注 [无关-已删除]

【输出格式】
请按以下格式输出清洗后的内容，每个文档片段之间用 "---" 分隔：

[相关片段 1]
清洗后的内容...

---
[相关片段 2]
清洗后的内容...

如果所有内容都与问题无关，请输出：
[无相关内容]
"""


class ContextCleaner:
    """上下文清洗器 —— 使用 DeepSeek 过滤检索结果中的无关信息"""

    def __init__(self):
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
        return self._client

    def clean(
        self,
        user_query: str,
        contexts: list[str],
        max_chars_per_ctx: int = 600,
        max_total_chars: int = 4000,
    ) -> tuple[list[str], str]:
        """
        清洗检索到的上下文，过滤与问题无关的内容

        Args:
            user_query: 用户原始问题
            contexts: 检索到的上下文片段列表
            max_chars_per_ctx: 单条上下文最大字符数（超过则截断）
            max_total_chars: 所有上下文总字符数上限

        Returns:
            (cleaned_contexts, cleaned_text): 
                cleaned_contexts: 清洗后的上下文片段列表
                cleaned_text: 清洗后的合并文本
        """
        if not contexts:
            logger.debug("[ContextCleaner] 无上下文需要清洗")
            return [], ""

        # 截断过长的上下文
        truncated = []
        total = 0
        for ctx in contexts:
            ctx = str(ctx).strip()
            if not ctx:
                continue
            if len(ctx) > max_chars_per_ctx:
                ctx = ctx[:max_chars_per_ctx] + "..."
            truncated.append(ctx)
            total += len(ctx)
            if total >= max_total_chars:
                break

        if not truncated:
            return [], ""

        # 构建上下文块
        context_blocks = "\n\n---\n\n".join(
            f"[文档片段 {i+1}]\n{ctx}" for i, ctx in enumerate(truncated)
        )

        user_prompt = f"""用户问题：{user_query}

以下是从知识库中检索到的文档片段，请清洗出与问题直接相关的内容：

{context_blocks}

请输出清洗后的内容。"""

        try:
            response = self.client.chat.completions.create(
                model=settings.agent_llm_model,
                messages=[
                    {"role": "system", "content": CLEANER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # 低温度，确保稳定输出
                max_tokens=2000,
                timeout=settings.llm_timeout,
            )

            cleaned_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            # 检查是否无相关内容
            if "[无相关内容]" in cleaned_text or cleaned_text == "":
                logger.info(f"[ContextCleaner] 清洗后无相关内容 (tokens: {tokens_used})")
                return [], ""

            # 解析清洗后的片段
            cleaned_contexts = self._parse_cleaned_output(cleaned_text)

            # ★ 清洗质量检查：检查清洗后的内容是否包含有效信息
            # 不再用简单的字符数/比例阈值（因为有些问题确实只有几行答案）
            # 而是检查内容是否非空、是否包含有效文本（不是只有标点/空格）
            cleaned_len = sum(len(c) for c in cleaned_contexts)

            if cleaned_contexts and cleaned_len > 0:
                # 清洗后有内容：检查每段内容是否都包含有效文本
                # 过滤掉只有标点/空格的无意义片段
                valid_contexts = []
                for ctx in cleaned_contexts:
                    stripped = ctx.strip()
                    if stripped and any(ch.isalpha() or ch.isdigit() or '\u4e00' <= ch <= '\u9fff' for ch in stripped):
                        valid_contexts.append(ctx)

                if valid_contexts:
                    logger.info(
                        f"[ContextCleaner] 清洗完成: {len(contexts)} → {len(valid_contexts)} 片段, "
                        f"清洗后 {sum(len(c) for c in valid_contexts)} 字符"
                    )
                    return valid_contexts, "\n\n".join(valid_contexts)

            # 清洗后无有效内容，返回原始上下文
            logger.warning(
                f"[ContextCleaner] 清洗后无有效内容 (cleaned_len={cleaned_len}), "
                f"退回原始上下文"
            )
            return truncated, "\n\n".join(truncated)

        except Exception as e:
            logger.error(f"[ContextCleaner] 清洗失败: {e}")
            # 清洗失败时返回原始上下文
            return contexts, "\n\n".join(contexts)

    def _parse_cleaned_output(self, text: str) -> list[str]:
        """解析清洗后的输出，提取各个片段"""
        fragments = []

        # 按 "---" 分割
        parts = text.split("\n---\n")
        if len(parts) == 1:
            parts = text.split("---")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 去除 "[相关片段 N]" 或 "[文档片段 N]" 标题
            import re
            part = re.sub(r'^\[(相关片段|文档片段|片段)\s*\d*\]\s*', '', part.strip())

            # 跳过 [无关-已删除] 标记
            if "[无关-已删除]" in part or "[无相关内容]" in part:
                continue

            if len(part) > 10:  # 过滤过短的片段
                fragments.append(part)

        return fragments


# 全局单例
_context_cleaner: Optional[ContextCleaner] = None


def get_context_cleaner() -> ContextCleaner:
    """获取上下文清洗器单例"""
    global _context_cleaner
    if _context_cleaner is None:
        _context_cleaner = ContextCleaner()
    return _context_cleaner