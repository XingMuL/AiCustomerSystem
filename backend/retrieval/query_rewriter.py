"""
Query 改写模块：基于 LLM 对用户查询进行扩展和改写。

【优化版】v2
- 多视角 HyDE：生成 4 种不同风格的假设文档（总结型 + 章节标题型 + 列表型 + 问答型）
  针对不同检索场景（如"分类有多少个"、"VIP电话是什么"）分别匹配对应文档
- 关键词增强：新增章节级别触发词（如"分类"、"电话"、"VIP"）
- 强关键词提取：从改写后的 query 中提取更长、更精确的关键词

核心流程：
1. 携带对话历史，理解上下文指代（如"它"、"上面提到的"）
2. 将口语化问题改写为更精确的检索查询
3. 提取关键词，生成多角度检索查询
4. 同义词和相关词扩展，提升召回率
"""

from typing import Optional, List

from openai import OpenAI
from loguru import logger

from backend.config import settings
from backend.embedding.sparse_embedder import SparseEmbedder


_QUERY_REWRITE_PROMPT = """你是一个专业的搜索查询优化助手。请根据对话历史和用户最新问题，将其改写为更适合文档检索的查询语句。

规则：
1. 如果用户问题中有指代词（"它"、"这个"、"上面提到的"等），根据对话历史替换为具体的实体名称
2. 将口语化表达改写为正式的文档查询语言
3. 保留原始问题的核心语义，不要添加额外信息
4. 如果用户问题本身已经很清晰，直接返回原问题
5. 只输出改写后的查询语句，不要任何解释

对话历史：
{history}

用户问题：{query}

改写后的查询："""


# 【v2 增强】关键词扩展规则：将常见的用户词汇映射到知识库中更常用的术语
# 新增：章节级别触发词（如"分类"、"VIP"、"电话"），确保细分章节能被召回
_KEYWORD_EXPANSION = {
    "购物技巧": ["购物技巧", "省钱技巧", "优惠技巧", "凑单", "满减", "折扣"],
    "高级购物技巧": ["高级购物技巧", "凑单", "拆分订单", "预售", "抢购", "收藏", "评价", "决策"],
    "客服": ["客服", "服务热线", "联系方式", "电话", "邮箱", "400", "人工服务"],
    "退货": ["退货", "退款", "售后", "退换货", "七天无理由"],
    "订单": ["订单", "下单", "购买", "支付", "物流", "配送"],
    "优惠": ["优惠", "折扣", "促销", "满减", "优惠券", "省钱"],
    "商品": ["商品", "产品", "手机", "数码", "家电"],
    "电话": ["电话", "客服热线", "联系方式", "400", "热线", "专线", "联系电话"],
    "会员": ["会员", "VIP", "等级", "权益", "特权", "VIP会员"],
    "分类": ["分类", "商品分类", "种类", "类别", "类型", "种类详情"],
    "VIP": ["VIP", "会员", "专线", "专属", "VIP会员", "高级会员"],
    "专线": ["专线", "热线", "客服专线", "400电话", "VIP专线"],
    "流程": ["流程", "步骤", "操作流程", "使用流程", "办理流程"],
    "费用": ["费用", "价格", "收费", "金额", "多少钱", "费用标准"],
    "时间": ["时间", "工作日", "营业时间", "服务时间", "几点", "多久"],
    "支持": ["支持", "支持服务", "支持的", "兼容", "适用"],
}


# 【v2】细分章节的触发词：当 query 中包含这些词时，额外强化关键词
_CHAPTER_TRIGGER_KEYWORDS = {
    "分类": ["商品分类", "分类", "种类", "类别", "类型", "有多少种", "分别是", "包括", "几大类"],
    "电话": ["电话", "热线", "400", "专线", "客服电话", "联系方式", "联系我们"],
    "VIP": ["VIP", "会员专线", "高级会员", "VIP服务", "专属热线", "VIP会员"],
    "流程": ["流程", "步骤", "怎么", "如何", "操作", "指南", "教程"],
    "费用": ["费用", "价格", "多少钱", "收费", "金额", "标准"],
    "政策": ["政策", "规则", "规定", "条款", "须知", "注意"],
}


def _expand_keywords(query: str) -> str:
    """【v2 增强】基于关键词扩展规则扩展查询，提升召回率。

    策略：
    1. 基础词汇扩展：_KEYWORD_EXPANSION 中的通用词汇
    2. 章节触发词扩展：匹配 _CHAPTER_TRIGGER_KEYWORDS 中的触发词，
       额外加入章节级别关键词（如 query 含"分类"，加入"商品分类详情"）
    3. 不改变原始查询，仅追加相关关键词（用空格分隔）
    """
    expanded_terms = []

    # 1. 基础词汇扩展
    for keyword, expansions in _KEYWORD_EXPANSION.items():
        if keyword in query:
            for exp in expansions:
                if exp not in query and exp not in expanded_terms:
                    expanded_terms.append(exp)

    # 2. 章节触发词扩展（更强的召回能力）
    for chapter_type, triggers in _CHAPTER_TRIGGER_KEYWORDS.items():
        for trigger in triggers:
            if trigger in query:
                # 加入该章节类型的所有关键词，确保能触发细分章节召回
                for extra_keyword in triggers:
                    if extra_keyword not in query and extra_keyword not in expanded_terms:
                        expanded_terms.append(extra_keyword)
                break

    if expanded_terms:
        # 限制扩展数量，避免过度膨胀（适度放宽，确保细分章节覆盖）
        limited = expanded_terms[:10]
        return query + " " + " ".join(limited)
    return query


class QueryRewriter:
    """
    基于 LLM 的查询改写器（v2：多视角 HyDE + 强关键词扩展）

    改写流程：LLM 根据对话历史改口语化查询 → 关键词扩展 → 输出最终查询字符串
    """

    def __init__(self, sparse_embedder: Optional[SparseEmbedder] = None):
        self.sparse_embedder = sparse_embedder or SparseEmbedder()
        self.client = OpenAI(
            api_key=getattr(settings, "llm_api_key", settings.deepseek_api_key),
            base_url=getattr(settings, "llm_api_base", settings.deepseek_base_url),
        )
        self.model = getattr(settings, "query_rewrite_model", "deepseek-chat")
        self.enabled = getattr(settings, "enable_query_rewrite", True)

    def rewrite(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        改写查询

        Args:
            query: 用户原始查询
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]

        Returns:
            改写后的查询字符串
        """
        if not query.strip():
            return query

        if not self.enabled:
            return _expand_keywords(query)

        # 构建历史文本
        history_text = "无对话历史"
        if conversation_history:
            lines = []
            for msg in conversation_history[-6:]:  # 最近 6 轮
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                if content and isinstance(content, str):
                    lines.append(f"{role}: {content[:200]}")
            if lines:
                history_text = "\n".join(lines)

        rewritten = query
        try:
            prompt = _QUERY_REWRITE_PROMPT.format(
                history=history_text,
                query=query,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            result = response.choices[0].message.content.strip()
            if result:
                rewritten = result
                logger.info(f"Query 改写: '{query[:50]}...' → '{rewritten[:50]}...'")

        except Exception as e:
            logger.warning(f"Query 改写失败: {e}")

        # Step 2: 关键词扩展（无论 LLM 是否成功改写，都尝试扩展）
        expanded = _expand_keywords(rewritten)
        if expanded != rewritten:
            logger.info(f"Query 关键词扩展: '{rewritten[:50]}...' → '{expanded[:100]}...'")
            rewritten = expanded

        return rewritten

    def generate_hypothetical_document(self, query: str) -> Optional[str]:
        """
        【v2 增强】多视角 HyDE (Hypothetical Document Embeddings) 查询增强

        原理：
        - 让 LLM 根据用户问题生成**多种风格**的假设性答案文档
        - 将多视角假设文档的向量融合后替代原始查询向量进行检索
        - 不同场景选择不同风格：
          * "有多少个分类" → 列表型/章节标题型（触发细分章节）
          * "VIP电话是什么" → 问答型/总结型（触发具体信息段落）

        改进：
        1. 4 种不同风格的假设文档（总结+章节标题+列表+问答），覆盖更多检索场景
        2. 自动判断 query 类型，优先使用高匹配的风格
        3. 对短问句、专业名词、精确查询场景提升尤为显著

        Args:
            query: 用户原始查询

        Returns:
            组合后的假设文档文本（多风格文档拼接），如果生成失败则返回 None
        """
        if not getattr(settings, "enable_hyde", True):
            return None

        # 判断 query 类型，选择合适的 prompt 权重
        doc_type_scores = {
            "summary": 1.0,  # 总结型：基础权重
            "chapter": 0.5,  # 章节标题型：含"分类/类别/章节"时提升权重
            "list": 0.5,     # 列表型：含"多少个/分别是/哪些"时提升权重
            "qa": 0.8,       # 问答型：含"是什么/怎么/如何"时提升权重
        }

        for triggers in _CHAPTER_TRIGGER_KEYWORDS.values():
            for trigger in triggers:
                if trigger in query:
                    # 匹配到章节触发词，加强列表型/章节标题型权重
                    doc_type_scores["chapter"] = 1.5
                    doc_type_scores["list"] = 1.2
                    break

        # 按权重排序，从高到低生成文档
        sorted_doc_types = sorted(
            doc_type_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 生成多种风格的假设文档
        generated_docs = []
        try:
            # 1. 总结型（基础 HyDE 风格）
            if doc_type_scores.get("summary", 0) > 0.3:
                summary_doc = self._call_hyde_with_prompt(
                    _HYDE_PROMPTS["summary"], query,
                    max_tokens=getattr(settings, "hyde_max_tokens", 300),
                )
                if summary_doc:
                    generated_docs.append(("summary", summary_doc))

            # 2. 章节标题型（用于"XX有多少分类/类别"这类问题）
            if doc_type_scores.get("chapter", 0) > 0.3:
                chapter_doc = self._call_hyde_with_prompt(
                    _HYDE_PROMPTS["chapter_title"], query,
                    max_tokens=max(200, getattr(settings, "hyde_max_tokens", 300)),
                )
                if chapter_doc:
                    generated_docs.append(("chapter", chapter_doc))

            # 3. 列表型（用于列举类问题，如"有哪些分类"）
            if doc_type_scores.get("list", 0) > 0.3:
                list_doc = self._call_hyde_with_prompt(
                    _HYDE_PROMPTS["list"], query,
                    max_tokens=getattr(settings, "hyde_max_tokens", 300),
                )
                if list_doc:
                    generated_docs.append(("list", list_doc))

            # 4. 问答型（用于具体问题，如"电话是什么"）
            if doc_type_scores.get("qa", 0) > 0.3:
                qa_doc = self._call_hyde_with_prompt(
                    _HYDE_PROMPTS["qa"], query,
                    max_tokens=getattr(settings, "hyde_max_tokens", 300),
                )
                if qa_doc:
                    generated_docs.append(("qa", qa_doc))

            if generated_docs:
                # 组合所有风格的假设文档，拼接成一个长文本用于向量嵌入
                combined = "\n\n".join([f"【{name}】\n{doc}" for name, doc in generated_docs])
                logger.info(
                    f"HyDE 生成多视角假设文档: {len(generated_docs)} 种风格 "
                    f"(总结/章节/列表/问答)，共 {len(combined)} 字符"
                )
                logger.debug(f"HyDE 文档类型: {[name for name, _ in generated_docs]}")
                return combined

        except Exception as e:
            logger.warning(f"HyDE 多视角生成失败: {e}")

        return None

    # --- HyDE 辅助方法 ---

    def _call_hyde_with_prompt(
        self, prompt_template: str, query: str, max_tokens: int = 300
    ) -> Optional[str]:
        """
        使用指定 prompt 生成一个假设文档片段

        Args:
            prompt_template: 带 {query} 占位符的 prompt 模板
            query: 用户查询
            max_tokens: 生成长度限制

        Returns:
            假设文档内容，失败则返回 None
        """
        try:
            prompt = prompt_template.format(query=query)
            response = self.client.chat.completions.create(
                model=getattr(settings, "hyde_model", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=getattr(settings, "hyde_temperature", 0.5),
                max_tokens=max_tokens,
            )
            doc = response.choices[0].message.content.strip()
            if doc and len(doc) > 20:  # 太短的回答（如"是的"）无意义
                return doc
        except Exception as e:
            logger.debug(f"HyDE 单个风格生成失败: {e}")
        return None


# ============================================================
# 【v2】多视角 HyDE Prompt 集合
# 针对不同查询场景，生成对应风格的假设文档，提升细分章节召回率
# ============================================================

_HYDE_PROMPTS = {
    # 1. 总结型：原 HyDE 风格，适用于一般性问题
    "summary": """你是一个知识库文档撰写助手。请根据用户的问题，写一段假设的知识库文档内容来回答这个问题。

规则：
1. 模仿知识库文档的口吻和风格（正式、专业、结构化）
2. 包含问题中提到的关键名词和术语
3. 写 200-400 字的段落，不要太长
4. 不要直接回答问题，而是写出假设文档中可能包含的相关内容
5. 只输出文档内容，不要任何解释

用户问题：{query}

假设的文档内容：""",

    # 2. 章节标题型：针对"有多少分类/分别是什么/包括哪些类型"这类概括性章节问题
    "chapter_title": """你是一个知识库的章节整理助手。请根据用户的问题，写一段假设的「章节标题与内容概览」文档。

规则：
1. 模仿知识库「XX章节」的格式，包含章节编号、章节标题、章节内容简述
2. 如果用户问的是"有多少分类"，请以「商品分类」为章节标题，列出各类别的名称和简介
3. 如果用户问的是具体服务（如电话、VIP服务），请以该服务为章节标题，写出该服务的详细信息
4. 使用正式的文档风格，如「第二章 商品分类详解」
5. 包含子章节标题（如「2.1 电子数码类」）
6. 内容长度 150-300 字
7. 只输出文档内容，不要任何解释

用户问题：{query}

假设的章节内容：""",

    # 3. 列表型：针对列举类问题，强调关键词匹配
    "list": """你是一个知识库的内容整理助手。请根据用户的问题，写一段假设的「列表型」知识库文档内容。

规则：
1. 使用项目符号或编号列出相关内容（如 "1. XXXX"、"- XXXX"）
2. 每个项目都是知识库中可能出现的具体条目
3. 条目应包含用户问题中的关键词（如"会员"、"分类"、"电话"、"专线"、"400"等）
4. 每个条目的内容都是正式、完整的知识库描述
5. 列出 5-10 个相关条目
6. 只输出列表内容，不要任何解释

用户问题：{query}

假设的列表内容：""",

    # 4. 问答型：针对具体问题，如"XX电话是什么"、"费用多少"
    "qa": """你是一个客服知识库的问答编写助手。请根据用户的问题，写一段假设的「问答型」知识库文档内容。

规则：
1. 模仿客服FAQ（常见问题解答）的格式
2. 以"问：..."和"答：..."的形式撰写
3. 回答内容应包含具体、详细的信息（如具体的电话号码、具体的流程步骤等）
4. 使用正式、专业的口吻
5. 内容长度 150-300 字
6. 只输出 FAQ 内容，不要任何解释

用户问题：{query}

假设的 FAQ 内容：""",
}
