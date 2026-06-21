"""
知识库问答智能体（KB-QA Agent）

基于 RAG 检索结果回答用户知识类问题。
支持降级策略：Level 3/4 跳过 RAG 检索，Level 4 走兜底话术。
"""

from loguru import logger
from openai import OpenAI

from backend.config import settings

KB_QA_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，只能基于提供的知识库内容回答问题。

【核心规则】
1. 只能基于知识库中明确出现的内容进行回答，禁止编造知识库中没有的任何信息
2. 禁止编造知识库中没有的政策、流程、时间、金额、数量、联系方式等具体数据
3. 如果知识库中有明确的客服电话、邮箱、平台名称等信息，可以直接引用这些内容
4. 禁止编造或补充知识库中没有的联系方式、操作建议
5. 简单说：知识库中有什么就答什么，知识库中没有就明确说"没有相关信息"

【冲突信息处理规则】★ 重要 ★
1. 不同文档标注了不同的"相关度"分数，分数越高表示内容越相关、越可靠
2. 如果不同文档中的信息存在冲突或不一致，**必须优先采用相关度分数最高的文档中的表述**
3. 不得将冲突信息进行融合、折中或拼接，只能引用最高分文档中的原文
4. 如果最高分文档中缺少某些细节，只能说"知识库中暂无相关信息"，不得从低分数文档中补充
5. 优先原则：**严格按照相关度从高到低排序使用，高分文档完全覆盖低分文档**

【处理规则】
1. 如果知识库中有明确答案，直接引用原文内容进行回答
2. 如果知识库内容部分相关但不足以完整回答问题，列出知识库中已有的相关内容，不可补充缺失部分
3. 如果知识库完全没有相关内容，直接告知："知识库中暂无相关信息，建议您提供更具体的问题描述"
4. 回答要简洁、准确，避免推测、假设或补充知识库外的内容
5. 禁止使用"可能"、"大概"、"也许"、"应该"等推测性词语，除非原文中有此表述"""


class KBQAAgent:
    """知识库问答智能体"""

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

    def answer(
        self,
        user_input: str,
        memory_context: str,
        rag_docs: list[dict],
        degradation_level: int = 0,
        rag_available: bool = True,
    ) -> str:
        """
        知识库问答

        Args:
            user_input: 用户输入
            memory_context: 记忆上下文（对话历史）
            rag_docs: RAG 召回的文档
            degradation_level: 降级等级
            rag_available: RAG 是否可用

        Returns:
            str: 生成的回答
        """
        # Level 3/4: RAG 不可用
        if degradation_level >= 3 or not rag_available:
            logger.warning(f"[KB QA] Level {degradation_level}，RAG 不可用，走兜底")
            return self._fallback_response(user_input)

        # === 相关度阈值过滤 ===
        # 如果所有召回文档的相关度都低于阈值，直接返回"无相关信息"，避免 LLM 编造
        RELEVANCE_THRESHOLD = 0.3  # 低于此分数的文档视为不相关

        # 构建 Prompt
        knowledge_text = ""
        high_relevance_docs = []
        if rag_docs:
            for i, doc in enumerate(rag_docs[:settings.rerank_top_k], 1):
                # RetrievalResult dataclass — 兼容属性访问和字典访问
                if hasattr(doc, 'content'):
                    content = doc.content[:800]
                    score = getattr(doc, 'score', 0)
                else:
                    content = doc.get("content", "")[:800]
                    score = doc.get("score", 0)

                if score >= RELEVANCE_THRESHOLD:
                    high_relevance_docs.append((i, content, score))

            if high_relevance_docs:
                docs_parts = []
                for i, content, score in high_relevance_docs:
                    docs_parts.append(f"[文档{i}] (相关度: {score:.2f})\n{content}")
                knowledge_text = "\n\n---\n\n".join(docs_parts)
                logger.info(
                    f"[KB QA] 高相关度文档: {len(high_relevance_docs)}/{len(rag_docs)} "
                    f"(阈值: {RELEVANCE_THRESHOLD})"
                )
            else:
                # 所有文档相关度都低于阈值，返回标准化提示
                logger.warning(
                    f"[KB QA] 所有文档相关度低于阈值({RELEVANCE_THRESHOLD})，"
                    f"返回标准化提示"
                )
                return {
                    "response": (
                        "知识库中暂无相关信息，建议您：\n"
                        "1. 使用更具体的关键词重新提问\n"
                        "2. 如需咨询订单、退货等业务问题，请直接描述您的需求\n"
                        "3. 如需人工帮助，请联系客服人员"
                    ),
                    "tokens": 0,
                }
        else:
            knowledge_text = "（无相关知识库文档）"

        # 构建消息
        messages = [{"role": "system", "content": KB_QA_SYSTEM_PROMPT}]

        if memory_context:
            messages.append({"role": "system", "content": f"对话历史：\n{memory_context}"})

        # 知识库内容
        messages.append({
            "role": "system",
            "content": f"以下是与用户问题相关的知识库内容：\n\n{knowledge_text}",
        })

        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model=settings.agent_llm_model,
                messages=messages,
                temperature=settings.agent_llm_temperature,
                max_tokens=settings.agent_llm_max_tokens,
                timeout=settings.llm_timeout,
            )
            return {
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"[KB QA] 生成失败: {e}")
            return {"response": self._fallback_response(user_input), "tokens": 0}

    def _fallback_response(self, user_input: str) -> str:
        """RAG 不可用时的兜底回答"""
        return (
            f"您好！关于「{user_input[:30]}...」这个问题，我目前暂时无法从知识库中获取相关信息。\n\n"
            "建议您：\n"
            "1. 尝试用更简洁的关键词重新提问\n"
            "2. 拨打电话客服热线获取即时帮助\n"
            "3. 稍后重试，系统正在恢复中\n\n"
            "给您带来的不便，敬请谅解！"
        )