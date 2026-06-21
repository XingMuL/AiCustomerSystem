"""
混合检索引擎（v2 优化版）：稠密 + 稀疏双路检索 → Rerank 重排 → 父块拉取 → 父块摘要 + 轻量 LLM 精炼 → 上下文构建

【v2 核心改进】
1. 父块摘要：为每个父块生成简短摘要（100-200字），保留核心信息，不依赖子块命中
2. 上下文新格式：[来源 N] 章节: XX → [父块摘要: ...] → [精炼内容: ...]
3. 兄弟父块保留完整摘要 + 完整内容（不被 chunk_refinement_max_length 截断）
4. 更鲁棒的精炼 fallback 链（LLM → 摘要增强 → 关键词匹配 → 完整原文）

检索流程：
  1. Query 改写（由外部 QueryRewriter 完成）
  2. 章节粗筛（可选）
  3. 稠密 + 稀疏双路检索 → Qdrant RRF 融合 → top20 子块
  4. LLM Rerank 重排 → top5 高相关子块
  5. 根据子块 parent_id 拉取完整父块（从 SQLite ParentStore）
  6. 【v2】为每个父块生成「父块摘要」（保留章节概述，不丢失核心信息）
  7. 轻量 LLM 父块精炼：从父块中提取与 query/子块相关的句子
  8. 组装上下文（格式：章节标题 + 父块摘要 + 精炼内容）+ 章节溯源
"""

import re
import json
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

from loguru import logger

from backend.config import settings
from backend.retrieval.vector_store import VectorStore
from backend.storage.parent_store import ParentStore
from backend.embedding.embedder import Embedder


class RetrievalResult:
    """检索结果"""
    def __init__(self, content: str, chapter_title: str = "", score: float = 0.0,
                 source_type: str = "", metadata: Optional[dict] = None,
                 parent_id: str = "", child_id: str = ""):
        self.content = content
        self.chapter_title = chapter_title
        self.score = score
        self.source_type = source_type
        self.metadata = metadata or {}
        self.parent_id = parent_id
        self.child_id = child_id


class LLMChunkRefiner:
    """
    轻量 LLM 父块精炼器（增强版：基于子块语义匹配）

    从拉取到的父块中，以**被检索命中的子块内容为语义锚点**，
    提取与子块语义相关的上下文段落，过滤掉与检索意图无关的大段内容，
    显著减少 RAG 上下文噪声，提升主 LLM 回答质量。

    核心原则：
    - 语义锚点优先：以被向量/关键词匹配命中的子块内容为核心锚点
    - 只从原文中提取，不修改、不编造
    - 保留原文句子的顺序
    - LLM 失败时 fallback 到关键词匹配（简单但可靠）
    """

    # 精炼系统 prompt：明确的"基于语义锚点的句子提取"任务
    SYSTEM_PROMPT = """你是一个严谨的文档信息筛选助手。你的任务是：从提供的文档片段中，**以被检索命中的关键内容（语义锚点）为核心**，提取出与用户查询和锚点内容语义相关的句子或段落，严格保留原文措辞和顺序，不添加、不修改、不概括任何内容。

【执行规则】
1. 以【语义锚点】中的内容为核心参考——这些内容是检索系统匹配到的关键信息，你需要从文档中提取与锚点内容语义相同、相近、或紧密关联的原文片段
2. 只提取文档中**明确出现过**、且与用户查询主题相关的句子/段落
3. 相关句子的判断标准：句子内容直接回答了用户查询、或为查询主题提供了支撑信息
4. 保持原文中的句子顺序不变，不要重新组织或概括
5. 不要添加任何解释、备注、引导语、结论等
6. 如果某段内容与查询和锚点都无关，直接跳过
7. 如果整篇文档都与查询无关，返回特殊标记：[NO_RELEVANT_CONTENT]
8. 不要返回任何非文档原文的文字

【输出格式】
- 直接输出相关的原文句子，按原文顺序排列
- 句子之间用换行符分隔
- 保留原文中的标点符号"""

    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
            self._available = True
        except Exception as e:
            logger.warning(f"[父块精炼] LLM 客户端初始化失败，将使用关键词 fallback: {e}")
            self._client = None
            self._available = False

    # ---------- 公共接口 ----------

    def refine_parent_chunk(
        self,
        parent_content: str,
        query: str,
        child_hits: List[RetrievalResult],
    ) -> Tuple[str, Dict]:
        """
        对单个父块进行精炼（v2 增强：保护结构化文档的标题信息不丢失）

        核心改进：
        - 检测文档是否为结构化文档（3+个 Markdown/编号标题）
        - 结构化文档：优先保留所有标题行 + 与 query 相关的内容
        - 非结构化文档：原有逻辑（关键词匹配 + LLM 精炼）
        """
        stats = {
            "method": "skip",
            "original_length": len(parent_content),
            "refined_length": len(parent_content),
            "compression_ratio": 1.0,
            "error": None,
        }

        if len(parent_content) < settings.chunk_refinement_min_length:
            stats["method"] = "skip"
            return parent_content, stats

        # --- v2 增强：检测是否为结构化文档（有多个标题/分类）---
        lines = [l.strip() for l in parent_content.split("\n") if l.strip()]
        title_indices = []
        for i, line in enumerate(lines):
            if self._is_title_line(line):
                title_indices.append(i)

        # 如果是结构化文档（>=3个标题）：采用"标题骨架+内容填充"策略
        if len(title_indices) >= 3:
            stats["method"] = "structured_keyword"

            # Step 1: 构建 query 关键词集合
            query_keywords = set()
            for w in __import__('re').split(r"[，。！？；：、\s,.!?;:\-]+", query):
                if len(w) >= 2:
                    query_keywords.add(w)
            # 加入子块中的关键词
            for c in (child_hits or []):
                if c.content:
                    for w in __import__('re').split(r"[，。！？；：、\s,.!?;:\-]+", c.content[:200]):
                        if len(w) >= 2:
                            query_keywords.add(w)

            # Step 2: 构建精炼内容（标题优先策略）
            refined_parts = []

            # 保留文档开头的概述（第一个标题之前的内容）
            first_title_idx = min(title_indices) if title_indices else 0
            if first_title_idx > 0:
                for i in range(0, min(first_title_idx, 3)):
                    refined_parts.append(lines[i])
                refined_parts.append("")

            # 对每个标题：保留标题 + 标题下与 query/标题本身相关的1-2行内容
            for idx, title_idx in enumerate(title_indices):
                # 保留标题行
                title_line = lines[title_idx]
                refined_parts.append(title_line)

                # 确定下一个标题位置
                next_title_idx = title_indices[idx + 1] if idx + 1 < len(title_indices) else len(lines)

                # 策略：标题行本身就是信息（"2.1 手机数码类"直接回答了"分别是什么"）
                # 在标题下保留 1-2 行核心内容（优先含编号/类型/电话等关键词）
                priority_kw = ('编号', '类型', '电话', '热线', '服务', '内容', '适用', '卖点', '时间')

                # 优先：找包含重点关键词的行（最多1行）
                content_found = 0
                search_end = min(title_idx + 4, next_title_idx)
                for j in range(title_idx + 1, search_end):
                    if j < len(lines) and content_found < 1:
                        content_line = lines[j]
                        if not self._is_title_line(content_line):
                            if any(kw in content_line for kw in priority_kw):
                                refined_parts.append(content_line)
                                content_found += 1

                # 如果 query 本身是"列举类"问题（含"多少/哪些/分别/全部"），
                # 则该标题下额外保留 1 行内容（因为用户要的是完整列表）
                list_query_keywords = ('多少', '哪些', '分别', '全部', '所有', '几个', '各类', '分类', '列表')
                if any(q_kw in query for q_kw in list_query_keywords):
                    # 再保留1行内容（如果还有）
                    for j in range(title_idx + 1, search_end):
                        if j < len(lines) and content_found < 2:
                            content_line = lines[j]
                            if not self._is_title_line(content_line) and content_line not in refined_parts:
                                refined_parts.append(content_line)
                                content_found += 1

                # 如果 query 关键词命中了该标题下的内容，多保留1行
                if query_keywords:
                    for j in range(title_idx + 1, search_end):
                        if j < len(lines) and content_found < 3:
                            content_line = lines[j]
                            if not self._is_title_line(content_line) and content_line not in refined_parts:
                                if any(q_kw in content_line for q_kw in query_keywords if len(q_kw) >= 2):
                                    refined_parts.append(content_line)
                                    content_found += 1
                                    break

                # 标题组之间加空行
                refined_parts.append("")

            # 拼接并控制长度
            refined = "\n".join(refined_parts)

            # 长度控制：结构化文档放宽到 3500 字符（确保所有标题都在）
            max_len_structured = min(3500, settings.chunk_refinement_max_length)
            if len(refined) > max_len_structured:
                refined = refined[:max_len_structured]

            stats["refined_length"] = len(refined)
            stats["compression_ratio"] = len(refined) / max(len(parent_content), 1)
            stats["title_count"] = len(title_indices)
            return refined, stats

        # --- 非结构化文档：原有逻辑 ---
        # 组装子块关键词 + 子块原文（作为语义锚点）
        child_keywords = self._extract_child_keywords(child_hits)
        child_contents = [c.content for c in child_hits if c.content] if child_hits else []

        # 1. 优先使用轻量 LLM 调用
        refined = None
        if self._available and settings.enable_chunk_refinement:
            try:
                user_prompt = self._build_refinement_prompt(
                    parent_content, query, child_keywords, child_contents
                )
                refined = self._call_llm_refine(user_prompt)
                stats["method"] = "llm"
            except Exception as e:
                logger.warning(f"[父块精炼] LLM 调用失败，退化为关键词匹配: {e}")
                stats["error"] = f"llm_failed: {type(e).__name__}"

        # 2. LLM 不可用或失败：关键词 + 子块匹配 fallback
        if refined is None:
            refined = self._keyword_based_refine(
                parent_content, query, child_keywords, child_contents
            )
            if stats["method"] != "llm":
                stats["method"] = "keyword"

        # 3. 后处理：清理 LLM 输出
        refined = self._post_process(refined, parent_content)

        # 4. 长度控制
        min_len = max(200, len(parent_content) // 3)
        if len(refined) > settings.chunk_refinement_max_length:
            refined = refined[:settings.chunk_refinement_max_length]
        elif len(refined) < 150 and len(parent_content) > 200:
            supplementary = parent_content[len(refined):min_len - len(refined)] if len(refined) < min_len else ""
            if supplementary and len(supplementary) > 0:
                refined = refined + "\n\n[父块补充内容]\n" + supplementary[:min_len]

        stats["refined_length"] = len(refined)
        stats["compression_ratio"] = len(refined) / max(len(parent_content), 1)

        return refined, stats

    # ---------- 【v2 新增】父块摘要：不依赖子块命中，提取章节核心信息 ----------

    def _is_title_line(self, line: str) -> bool:
        """
        判断一行是否为标题行（v2 核心改进：确保分类信息不丢失）
        匹配规则：
        1. Markdown 标题：#、##、###、####
        2. 中文编号标题：2.1、2.2、3.1 等
        3. 方括号编号：【2.1】、【1】等
        4. 中文数字编号：一、二、三、第一、第二等
        5. 数字加顿号/点：1.、2.、3、1) 等
        """
        if not line:
            return False
        s = line.strip()

        # 1. Markdown 标题
        if s.startswith('#'):
            return True

        # 2. 中文编号标题：2.1 xxx、3.2 xxx 等（至少2位数字+点+数字的组合，或 数字. 中文）
        import re
        if re.match(r'^\d{1,2}\.\d{1,2}\s', s[:20]):  # 2.1、3.5 等
            return True
        if re.match(r'^\d{1,2}[.、\)）]\s', s[:10]):  # 1.、2)、一、
            return True

        # 3. 方括号编号：【2.1 xxx】或 【xxx】
        if s.startswith('【') and '】' in s[:30]:
            return True

        # 4. 中文数字编号
        cn_num = ('一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                  '第', '第一节', '第二节', '第一部分')
        if any(s.startswith(cn) for cn in cn_num):
            return True

        return False

    def summarize_parent_chunk(
        self,
        parent_content: str,
        chapter_title: str = "",
    ) -> Tuple[str, Dict]:
        """
        为每个父块生成简短摘要（v2 核心改进：确保分类标题信息不丢失）

        策略：
        1. 优先 LLM 生成摘要（prompt 明确要求保留所有分类标题）
        2. fallback：结构化提取
           - 识别并保留所有标题行（Markdown、编号、方括号等）
           - 每个标题下保留1-2行核心内容（如"主要商品类型"、"适用人群"）
           - 保留概述段落
           - 长度放宽到 500 字符
        3. 兜底：原文前 500 字符

        返回: (summary_text, stats)
        """
        stats = {
            "method": "keyword",
            "original_length": len(parent_content),
            "summary_length": 0,
            "error": None,
        }

        if not parent_content or len(parent_content) < 50:
            summary = parent_content if parent_content else ""
            stats["summary_length"] = len(summary)
            return summary, stats

        # --- Method 1: LLM 生成摘要（优先，prompt 明确要求保留分类标题）
        if self._available and settings.enable_chunk_refinement:
            try:
                summary_prompt = (
                    "你是一个严谨的文档摘要助手。请从以下文档片段中提取核心内容，生成结构化摘要：\n"
                    "【重要规则】：\n"
                    "1. 必须保留所有分类/章节标题（如 2.1、2.2、【2.7】、### 开头的标题），这是最重要的！\n"
                    "2. 每个分类下保留1-2行核心要点（如主要商品类型、商品编号、核心卖点等）\n"
                    "3. 保留具体信息（如数字、编号、服务内容、联系方式等）\n"
                    "4. 不要添加、不修改原文措辞，用原文表述\n"
                    "5. 只输出摘要内容，不要任何解释或标注\n"
                    "6. 格式：用简短 bullet points 或 '标题 + 要点' 形式，保持清晰\n"
                    "7. 长度控制在 150-400 字\n\n"
                    f"【章节标题】: {chapter_title}\n"
                    f"【文档内容】: {parent_content[:1500]}\n"
                    "【摘要】:"
                )

                response = self._client.chat.completions.create(
                    model=settings.chunk_refinement_model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.2,
                    max_tokens=300,
                    timeout=10,
                )
                summary = (response.choices[0].message.content or "").strip()
                if summary and len(summary) > 40:
                    stats["method"] = "llm"
                    stats["summary_length"] = len(summary)
                    return summary, stats
            except Exception as e:
                logger.debug(f"[父块摘要] LLM 摘要失败，使用关键词提取 fallback: {e}")
                stats["error"] = f"llm_failed: {type(e).__name__}"

        # --- Method 2: 结构化提取 fallback（v2 增强：识别所有标题，确保分类信息不丢失）
        try:
            lines = [l.strip() for l in parent_content.split("\n") if l.strip()]

            # Step 1: 识别所有标题行，建立 标题 → 后续内容 的映射
            title_indices = []
            for i, line in enumerate(lines):
                if self._is_title_line(line):
                    title_indices.append(i)

            # 如果没有识别到标题，退化为简单提取（前 6 行 + 含数字的行）
            if not title_indices:
                # 无明显标题结构的文档，简单提取
                numbered_lines = [
                    l for l in lines
                    if any(c.isdigit() for c in l[:30]) and 10 < len(l) < 200
                ]
                summary_parts = []
                summary_parts.extend(lines[:6])
                summary_parts.extend(numbered_lines[:4])
                summary_parts.extend(lines[-2:] if len(lines) > 8 else [])
                seen = set()
                final_parts = []
                for l in summary_parts:
                    key = l.strip()[:80]
                    if key[:50] not in seen:
                        seen.add(key[:50])
                        final_parts.append(key)
                summary = "\n".join(final_parts)[:400]
                if not summary:
                    summary = parent_content[:400]
                stats["summary_length"] = len(summary)
                return summary, stats

            # Step 2: 有标题结构的文档 → 智能提取（标题优先，确保分类信息不丢失）
            summary_parts = []

            # Step 2a: 提取文档开头的概述段落（第一个标题之前的内容）
            first_title_idx = min(title_indices) if title_indices else 0
            if first_title_idx > 0:
                for i in range(0, min(first_title_idx, 3)):
                    summary_parts.append(lines[i])
                summary_parts.append("")  # 概述与第一个标题之间加空行

            # Step 2b: 先确定所有标题行（这些是骨架，必须全部保留）
            # 计算标题行总长度，剩余空间分配给内容
            title_lines_total = sum(len(lines[idx]) for idx in title_indices)
            num_titles = len(title_indices)
            target_total_len = 600  # 结构化文档摘要上限放宽到 600 字符
            # 每个标题下可用的平均内容空间 = (总空间 - 标题总长度 - 空行) / 标题数
            remaining_for_content = target_total_len - title_lines_total - (num_titles * 2)
            avg_content_per_title = max(40, remaining_for_content // max(num_titles, 1)) if remaining_for_content > 0 else 40

            # 智能策略：标题数多时，每个标题下只保留 1 行内容；标题数少时保留 2 行
            if num_titles <= 3:
                content_lines_per_title = 2
            elif num_titles <= 6:
                content_lines_per_title = 1
            else:
                content_lines_per_title = 1  # 标题数很多，每个只保留1行核心内容

            # Step 2c: 提取每个标题 + 后续核心内容
            for title_idx in title_indices:
                # 保留标题行
                title_line = lines[title_idx]
                summary_parts.append(title_line)

                # 计算标题后的内容范围（到下一个标题前为止）
                next_title_idx = None
                for t_idx in title_indices:
                    if t_idx > title_idx:
                        next_title_idx = t_idx
                        break

                if next_title_idx is not None:
                    # 下一个标题之前的行
                    max_content_idx = min(title_idx + content_lines_per_title + 1, next_title_idx)
                else:
                    # 最后一个标题
                    max_content_idx = min(title_idx + content_lines_per_title + 1, len(lines))

                # 提取标题下的核心内容（优先保留：商品编号 / 主要商品类型 / 电话 等关键词）
                content_added = 0
                priority_keywords = ('编号', '类型', '电话', '热线', '服务', '内容', '适用', '卖点', '时间')

                # 策略 1: 优先找包含重点关键词的行
                if content_lines_per_title >= 1:
                    for j in range(title_idx + 1, max_content_idx):
                        if j < len(lines) and content_added < content_lines_per_title:
                            line = lines[j]
                            if self._is_title_line(line):
                                continue
                            # 优先：包含重点关键词
                            if any(kw in line for kw in priority_keywords):
                                summary_parts.append(line)
                                content_added += 1

                # 策略 2: 如果重点关键词行不够，按顺序补充
                if content_added < content_lines_per_title:
                    for j in range(title_idx + 1, max_content_idx):
                        if j < len(lines) and content_added < content_lines_per_title:
                            line = lines[j]
                            if self._is_title_line(line):
                                continue
                            # 行长度适中，不是纯空行
                            if 10 < len(line) < 200 and line not in summary_parts:
                                summary_parts.append(line)
                                content_added += 1

                # 标题组之间加空行分隔
                summary_parts.append("")

            # Step 3: 去重并生成摘要（保留空行用于结构）
            seen = set()
            final_parts = []
            for l in summary_parts:
                if not l:
                    final_parts.append(l)
                    continue
                key = l.strip()[:100]
                if key[:60] not in seen:
                    seen.add(key[:60])
                    final_parts.append(key)

            # 控制总长度在 target_total_len 字符内
            summary = "\n".join(final_parts)[:target_total_len]

            # Step 4: 检查并确保最后一个标题没有被截断 —— 如果被截断，
            # 说明空间分配不足，尝试只保留标题+更精简的内容
            # 找到最后一个被截断的标题位置
            last_title_in_summary = -1
            for title_idx in title_indices:
                if lines[title_idx] in summary:
                    last_title_in_summary = title_idx

            # 如果有标题不在摘要中，尝试精简策略：所有标题 + 每个标题只保留一行关键内容
            if last_title_in_summary >= 0 and last_title_in_summary < max(title_indices):
                missing_titles_exist = any(lines[ti] not in summary for ti in title_indices)
                if missing_titles_exist:
                    # 重建更精简的摘要：只保留标题 + 每个标题下 1 行（优先商品编号）
                    compact_parts = []
                    # 保留概述（前 2 行）
                    if first_title_idx > 0:
                        for i in range(0, min(first_title_idx, 2)):
                            compact_parts.append(lines[i])
                        compact_parts.append("")

                    for title_idx in title_indices:
                        compact_parts.append(lines[title_idx])  # 标题
                        # 只找一行包含"编号"或"类型"的内容
                        next_t = None
                        for t_idx in title_indices:
                            if t_idx > title_idx:
                                next_t = t_idx
                                break
                        search_end = min(title_idx + 4, next_t if next_t else len(lines))
                        found_content = False
                        for j in range(title_idx + 1, search_end):
                            if j < len(lines) and not self._is_title_line(lines[j]):
                                if '编号' in lines[j] or '类型' in lines[j] or '电话' in lines[j]:
                                    compact_parts.append(lines[j][:80])
                                    found_content = True
                                    break
                        if not found_content and title_idx + 1 < len(lines):
                            line = lines[title_idx + 1]
                            if not self._is_title_line(line) and len(line) > 5:
                                compact_parts.append(line[:80])
                        compact_parts.append("")  # 空行

                    compact_summary = "\n".join(compact_parts)[:target_total_len]
                    if compact_summary and len(compact_summary) > 60:
                        summary = compact_summary

            if not summary:
                summary = parent_content[:target_total_len]

            stats["summary_length"] = len(summary)
            return summary, stats
        except Exception as e:
            # --- Method 3: 兜底返回原文前 500 字符
            logger.warning(f"[父块摘要] 结构化提取失败，使用原文前 500 字符: {e}")
            summary = parent_content[:500]
            stats["summary_length"] = len(summary)
            return summary, stats

    # ---------- 核心方法：LLM 调用 ----------

    def _call_llm_refine(self, user_prompt: str) -> str:
        """调用轻量 LLM 提取相关句子"""
        response = self._client.chat.completions.create(
            model=settings.chunk_refinement_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.chunk_refinement_temperature,
            max_tokens=settings.chunk_refinement_max_tokens,
            timeout=settings.chunk_refinement_timeout,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()

        # LLM 报告无相关内容
        if "[NO_RELEVANT_CONTENT]" in content or not content:
            return ""

        return content

    def _build_refinement_prompt(
        self, parent_content: str, query: str,
        child_keywords: List[str], child_contents: Optional[List[str]] = None
    ) -> str:
        """构建精炼用户 prompt（增强版：包含子块原文作为语义锚点）"""
        parts = []
        parts.append("【用户查询】")
        parts.append(query)
        if child_keywords:
            parts.append("\n【检索关键词（理解检索意图）】")
            parts.append(", ".join(child_keywords[:8]))
        # 新增：将命中的子块原文作为语义锚点，让LLM更精确地定位相关内容
        if child_contents and len(child_contents) > 0:
            parts.append("\n【语义锚点：检索系统实际匹配到的子块内容（请重点关注这些内容的上下文）】")
            for i, c in enumerate(child_contents[:5]):  # 最多展示5个锚点子块
                content_display = c[:200] if len(c) > 200 else c
                parts.append(f"{i+1}. {content_display}")
        parts.append("\n【文档片段原文（父块）】")
        # 控制 LLM 输入长度（避免超长）
        max_input = 3000
        if len(parent_content) > max_input:
            parts.append(parent_content[:max_input] + "\n...(原文过长，已截断)")
        else:
            parts.append(parent_content)
        parts.append("\n【任务】请从上述文档片段中，**以语义锚点内容为核心**，仅提取出与用户查询相关的句子或段落。保持原文措辞和顺序，不添加任何额外文字。")
        return "\n".join(parts)

    # ---------- Fallback：关键词匹配 ----------

    def _keyword_based_refine(
        self, parent_content: str, query: str, child_keywords: List[str],
        child_contents: Optional[List[str]] = None
    ) -> str:
        """
        LLM 不可用时的 fallback：基于子块原文 + 关键词提取相关句子
        策略（增强版）：
        1. 句子匹配：在父块中查找与子块原文完全匹配/高度相似的句子（语义锚点匹配）
        2. 关键词匹配：对每个句子计算与 query/child_keywords 的关键词重叠度
        3. 合并两种结果，保留重叠度 > 阈值 的句子，按原顺序拼接
        """
        # --- 1. 关键词集合：来自 query + 子块 ---
        keywords = set()
        for w in re.split(r"[，。！？；：、\s,.!?;:\-]+", query):
            if len(w) >= 2:
                keywords.add(w)
        for kw in child_keywords:
            if len(kw) >= 2:
                keywords.add(kw)

        # --- 2. 拆分句子 ---
        sentences = re.split(r"(?<=[。！？.!?;；])\s*|\n{2,}", parent_content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return parent_content[:settings.chunk_refinement_max_length]

        # --- 3. 新增：子块原文匹配（语义锚点）---
        selected_by_child_match = set()
        if child_contents:
            # 对每个子块，查找它在父块中出现的句子
            for child_text in child_contents[:5]:
                if not child_text or len(child_text) < 10:
                    continue
                # 简化的子块内容（取前100字符作为匹配目标）
                child_snippet = child_text[:80]
                for i, sent in enumerate(sentences):
                    # 如果该句子是子块内容的子串，或子块是该句子的子串
                    if child_snippet and (child_snippet in sent or sent in child_snippet):
                        selected_by_child_match.add(i)
                        # 同时包含前后一句作为上下文
                        if i > 0:
                            selected_by_child_match.add(i - 1)
                        if i < len(sentences) - 1:
                            selected_by_child_match.add(i + 1)

        # --- 4. 关键词匹配 ---
        selected_by_keyword = set()
        for i, sent in enumerate(sentences):
            hit_count = sum(1 for kw in keywords if kw in sent)
            if hit_count > 0:
                selected_by_keyword.add(i)

        # --- 5. 合并两种匹配结果 ---
        all_selected = selected_by_child_match | selected_by_keyword

        # --- 6. 如果都没有命中，兜底保留前 3 句 ---
        if not all_selected:
            all_selected = set(range(min(3, len(sentences))))

        # --- 7. 按原顺序拼接选中的句子 ---
        selected_indices = sorted(all_selected)
        selected = [sentences[i] for i in selected_indices]

        result = "\n".join(selected)

        # --- 8. 如果结果太短，补充原父块的开头 ---
        if len(result) < 100 and len(parent_content) > 100:
            result = parent_content[:settings.chunk_refinement_max_length]

        return result

    def _extract_child_keywords(self, child_hits: List[RetrievalResult]) -> List[str]:
        """从子块中提取高频词，帮助 LLM 理解检索意图"""
        if not child_hits:
            return []

        all_text = " ".join([c.content for c in child_hits[:3]])
        # 简单词频统计（长度≥2 的词）
        word_counts = defaultdict(int)
        for w in re.split(r"[，。！？；：、\s,.!?;:\-]+", all_text):
            if len(w) >= 2 and len(w) <= 15:
                word_counts[w] += 1

        # 返回 top 关键词
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:10]]

    # ---------- 后处理 ----------

    def _post_process(self, refined: str, original: str) -> str:
        """清理 LLM 输出的常见问题"""
        if not refined or not refined.strip():
            # LLM 返回空：返回原父块的前 N 字符（兜底，避免上下文为空）
            return original[:settings.chunk_refinement_max_length]

        # 清理多余空行和空白
        refined = refined.strip()
        # 合并多个空行为单空行
        refined = re.sub(r"\n{3,}", "\n\n", refined)
        # 清理每行多余空白
        refined = "\n".join([line.strip() for line in refined.split("\n") if line.strip()])

        # 过滤 LLM 可能输出的解释性文字（如"以下是相关句子："）
        skip_phrases = [
            "以下是相关", "相关句子", "与查询相关", "相关内容如下",
            "根据文档", "提取结果", "相关信息", "相关片段",
        ]
        lines = refined.split("\n")
        filtered_lines = []
        for line in lines:
            if any(phrase in line and len(line) < 30 for phrase in skip_phrases):
                continue
            filtered_lines.append(line)
        refined = "\n".join(filtered_lines)

        if not refined.strip():
            return original[:settings.chunk_refinement_max_length]

        return refined


class HybridRetriever:
    """
    混合检索引擎

    检索路径：稠密 + 稀疏 → Qdrant RRF → LLM Rerank → 父块拉取 → 上下文构建
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        parent_store: Optional[ParentStore] = None,
        query_rewriter: Optional[object] = None,
        reranker: Optional[object] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.parent_store = parent_store or ParentStore()
        self.query_rewriter = query_rewriter
        self.reranker = reranker

        # 共享的 SparseEmbedder 单例（在 __init__ 时预加载，避免每次检索时加载）
        from backend.embedding.sparse_embedder import SparseEmbedder
        self.sparse_embedder = SparseEmbedder()
        vocab_loaded = self.sparse_embedder.load_vocab()
        if not vocab_loaded:
            logger.warning(
                "混合检索：稀疏词表未加载，首次检索可能使用纯稠密模式。"
                "请确保先调用 rag_pipeline.index_document() 生成词表。"
            )
        else:
            logger.info(f"混合检索：稀疏词表已加载 {self.sparse_embedder.get_stats()}")

        # 父块精炼器（轻量 LLM 句子提取）
        if settings.enable_chunk_refinement:
            try:
                self.chunk_refiner = LLMChunkRefiner()
                logger.info("混合检索：父块精炼器已初始化 (轻量 LLM 句子提取)")
            except Exception as e:
                logger.warning(f"混合检索：父块精炼器初始化失败，将跳过精炼: {e}")
                self.chunk_refiner = None
        else:
            self.chunk_refiner = None
            logger.info("混合检索：父块精炼已禁用 (enable_chunk_refinement=False)")

    # ========== 主检索入口 ==========

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        chapter_ids: Optional[list[str]] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        主检索入口：完整检索流程

        Args:
            query: 用户原始查询
            top_k: 最终返回的父块数量
            doc_id: 限定文档
            tags: 业务标签过滤
            chapter_ids: 限定章节
            conversation_history: 对话历史（用于 Query 改写）

        Returns:
            {
                "rewritten_query": str,
                "child_hits": [...],
                "parent_chunks": [...],
                "context": str,
                "sources": [...],
            }
        """
        # Step 1: Query 改写
        rewritten_query = query
        if self.query_rewriter:
            rewritten_query = self.query_rewriter.rewrite(query, conversation_history)
            logger.info(f"Query 改写: '{query}' → '{rewritten_query}'")

        # Step 1.5: HyDE 查询增强 —— 生成假设文档，用其向量替代原始查询向量
        hyde_doc = None
        if self.query_rewriter and getattr(settings, "enable_hyde", True):
            hyde_doc = self.query_rewriter.generate_hypothetical_document(rewritten_query)

        # Step 2: 稠密 + 稀疏混合检索 → top_k 子块
        candidate_children = self._hybrid_search_children(
            rewritten_query, recall_k=settings.hybrid_recall_k, doc_id=doc_id,
            tags=tags, chapter_ids=chapter_ids, hyde_doc=hyde_doc,
        )

        if not candidate_children:
            logger.warning("无检索结果")
            return self._empty_result(rewritten_query)

        logger.info(f"Step 1 混合检索: {len(candidate_children)} 个子块候选")

        # Step 3: Rerank 重排 → top5
        if self.reranker:
            reranked_children = self.reranker.rerank(
                rewritten_query,
                candidate_children,
                top_k=top_k,
            )
            logger.info(f"Step 2 Rerank: {len(candidate_children)} → {len(reranked_children)}")
        else:
            reranked_children = candidate_children[:top_k]

        if not reranked_children:
            return self._empty_result(rewritten_query)

        # Step 3.5: 【新增】子块相似度过滤：基于绝对/相对分数阈值过滤低相似度子块
        filtered_children = self._filter_low_score_children(reranked_children)
        if not filtered_children:
            logger.warning(
                f"Step 3.5 子块过滤: 所有 {len(reranked_children)} 个子块均低于阈值，"
                f"保留前 {settings.child_min_keep_count} 个 (兜底)"
            )
            filtered_children = reranked_children[:settings.child_min_keep_count]

        # 使用过滤后的子块（替换原 reranked_children）
        reranked_children = filtered_children

        # Step 4: 根据 parent_id 拉取完整父块，去重
        parent_ids = []
        for c in reranked_children:
            pid = c.metadata.get("parent_id", "")
            if pid and pid not in parent_ids:
                parent_ids.append(pid)

        parent_chunks = self.parent_store.get_by_parent_ids(parent_ids)

        # Step 4.1: 【新增】联动拉取兄弟父块：同一章节被拆分为多个父块时（如
        # "第二章 商品分类详解"包含 8 个分类，被切分为 2-3 个父块），
        # 检索命中其中一个父块的子块后，自动拉取同一章节的其他父块，
        # 确保概括性问题（如"有多少个分类？分别是什么？"）能拿到完整信息
        sibling_parents = self.parent_store.get_sibling_parents(parent_ids)
        if sibling_parents:
            parent_chunks.extend(sibling_parents)
            logger.info(
                f"Step 4.1 联动拉取兄弟父块: {len(parent_chunks) - len(sibling_parents)} → "
                f"{len(parent_chunks)} (新增 {len(sibling_parents)} 个同章节父块)"
            )

        # Step 4.2: 【新增】父块内容去重：删除内容高度重复的父块，避免冗余上下文
        # 多个不同的子块可能命中同一个父块（已通过parent_id去重），
        # 但不同父块之间也可能内容相似（如重复的章节信息），需要进一步去重
        parent_chunks, dedup_info = self._deduplicate_parent_chunks(
            parent_chunks, reranked_children
        )
        logger.info(
            f"Step 4.2 父块内容去重: {dedup_info['original_count']} → "
            f"{dedup_info['final_count']} (删除 {dedup_info['removed_count']} 个内容重复的父块)"
        )

        # Step 4.5: 【新增】轻量 LLM 父块精炼：从父块中提取与 query/子块相关的句子
        refinement_stats = self._refine_parent_chunks(
            parent_chunks, reranked_children, rewritten_query
        )
        if refinement_stats:
            total_orig = sum(s["original_length"] for s in refinement_stats)
            total_ref = sum(s["refined_length"] for s in refinement_stats)
            ratio = total_ref / max(total_orig, 1)
            methods = ", ".join(set(s["method"] for s in refinement_stats))
            logger.info(
                f"Step 4.5 父块精炼: {len(parent_chunks)} 个父块 → "
                f"{total_orig} → {total_ref} 字符 "
                f"(压缩率: {ratio:.1%}, 方法: {methods})"
            )

        # Step 5: 按子块得分排序父块
        score_map = {}
        for c in reranked_children:
            pid = c.metadata.get("parent_id", "")
            if pid not in score_map or c.score > score_map[pid]:
                score_map[pid] = c.score

        parent_chunks.sort(
            key=lambda p: score_map.get(p["parent_id"], 0),
            reverse=True,
        )

        # Step 6: 组装上下文（含章节溯源）——使用精炼后的父块内容
        context, sources = self._build_context(parent_chunks, reranked_children, rewritten_query)

        return {
            "rewritten_query": rewritten_query,
            "hyde_used": hyde_doc is not None,
            "child_hits": [
                {"content": c.content, "score": c.score, "chapter_title": c.chapter_title,
                 "parent_id": c.parent_id, "child_id": c.child_id}
                for c in reranked_children
            ],
            "parent_chunks": [
                {"parent_id": p["parent_id"], "chapter_title": p["chapter_title"],
                 "content_length": len(p["content"]),
                 "original_length": p.get("original_length", len(p["content"])),
                 "refinement_method": p.get("refinement_method", "none"),
                 "score": score_map.get(p["parent_id"], 0)}
                for p in parent_chunks
            ],
            "context": context,
            "sources": sources,
            "refinement_summary": {
                "total_original": sum(s["original_length"] for s in refinement_stats) if refinement_stats else 0,
                "total_refined": sum(s["refined_length"] for s in refinement_stats) if refinement_stats else 0,
                "methods": [s["method"] for s in refinement_stats] if refinement_stats else [],
                "enabled": settings.enable_chunk_refinement,
            } if refinement_stats else None,
        }

    # ========== Step 3.5: 子块相似度过滤 ==========

    def _filter_low_score_children(
        self,
        children: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        基于分数阈值过滤低相似度的子块

        过滤规则（按顺序应用）：
        1. 绝对阈值：分数 < child_score_absolute_threshold 的子块丢弃
        2. 相对阈值：分数 < (child_score_relative_threshold * 最高分) 的子块丢弃
        3. 兜底：至少保留 child_min_keep_count 个子块

        Args:
            children: Rerank 后的子块列表（已按分数降序排列）

        Returns:
            过滤后的子块列表
        """
        if not children:
            return []

        abs_threshold = settings.child_score_absolute_threshold
        rel_threshold = settings.child_score_relative_threshold
        min_keep = settings.child_min_keep_count

        # 获取最高分（用于相对阈值计算）
        max_score = max((c.score for c in children), default=0.0)

        # 收集所有子块的分数信息（用于日志）
        score_info = ", ".join([
            f"[{i}] score={c.score:.4f}" for i, c in enumerate(children)
        ])

        # 应用过滤规则
        filtered = []
        filtered_out = []
        for i, c in enumerate(children):
            # 规则 1：绝对阈值
            if c.score < abs_threshold:
                filtered_out.append((i, c.score, "below_abs_threshold"))
                continue
            # 规则 2：相对阈值（如果启用）
            if rel_threshold > 0 and c.score < (rel_threshold * max_score):
                filtered_out.append((i, c.score, "below_rel_threshold"))
                continue
            filtered.append(c)

        # 规则 3：兜底，确保至少保留 min_keep 个（children 本身已按分数降序排列）
        if len(filtered) < min_keep:
            filtered = children[:min_keep]
            filtered_out_info = ", ".join([
                f"(idx={i}, score={s:.4f}, reason={r})" for i, s, r in filtered_out
            ])
            logger.info(
                f"Step 3.5 子块过滤: {len(children)} → {len(filtered)} "
                f"(绝对阈值={abs_threshold}, 相对阈值={rel_threshold}, "
                f"max_score={max_score:.4f}, 全量低于阈值，兜底保留前 {min_keep} 个, "
                f"被过滤: {filtered_out_info})"
            )
            return filtered

        # 记录过滤结果（正常路径）
        if filtered_out:
            filtered_out_info = ", ".join([
                f"(idx={i}, score={s:.4f}, reason={r})" for i, s, r in filtered_out
            ])
        else:
            filtered_out_info = "none"

        logger.info(
            f"Step 3.5 子块过滤: {len(children)} → {len(filtered)} "
            f"(绝对阈值={abs_threshold}, 相对阈值={rel_threshold}, "
            f"max_score={max_score:.4f}, "
            f"被过滤: {filtered_out_info})"
        )
        logger.debug(f"  原始分数分布: {score_info}")

        return filtered

    # ========== Step 4.2: 父块内容去重（基于内容相似度） ==========

    def _deduplicate_parent_chunks(
        self,
        parent_chunks: List[Dict],
        reranked_children: List[RetrievalResult],
    ) -> Tuple[List[Dict], Dict]:
        """
        对拉取到的父块进行内容去重。

        去重策略（分两层）：
        1. parent_id 去重：已在 Step 4 完成（不同子块可能属于同一个父块）
        2. 内容相似度去重（新增）：不同父块之间可能内容高度相似
           - 计算父块之间的内容相似（子串/重叠）
           - 如果一个父块的核心内容（基于其子块的锚点）在另一个父块中已包含，删除前者
           - 优先保留关联到更多/更高分的子块的父块

        Returns:
            (去重后的父块列表, 去重统计信息字典)
        """
        original_count = len(parent_chunks)
        if original_count <= 1:
            return parent_chunks, {
                "original_count": original_count,
                "final_count": original_count,
                "removed_count": 0,
            }

        # --- 1. 构建 parent_id -> 子块列表 的映射（用于评估优先级） ---
        parent_to_children: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for c in reranked_children:
            pid = c.metadata.get("parent_id", "")
            if pid:
                parent_to_children[pid].append(c)

        # --- 2. 计算每个父块的"优先级评分"（子块最高分优先） ---
        # ★ 修复问题1：当多个父块内容冲突时，优先保留关联到更高分子块的父块
        # 评分策略：子块最高分 * 100 + 子块总分 + 子块数量
        #   - 最高分权重最大（*100）：确保高匹配度的父块优先保留
        #   - 总分 + 数量作为次要参考：同等最高分下，信息更丰富的父块优先
        def get_parent_priority(parent: Dict) -> float:
            pid = parent.get("parent_id", "")
            children = parent_to_children.get(pid, [])
            if not children:
                return 0.0
            # 最高分作为主要评分标准
            max_score = max(c.score for c in children)
            child_count = len(children)
            child_score_sum = sum(c.score for c in children)
            return max_score * 100 + child_score_sum + child_count

        # --- 3. 按优先级排序（优先级高的保留） ---
        parent_chunks_sorted = sorted(
            parent_chunks, key=lambda p: get_parent_priority(p), reverse=True
        )

        # --- 4. 内容去重：遍历父块，如果当前父块内容与已保留的父块内容高度相似，跳过 ---
        kept_chunks: List[Dict] = []
        removed_ids: List[str] = []

        # 用简单的内容特征（如取前N字符的集合，高频词集合）做相似性判断
        # 为了提高效率，使用：内容哈希（前300字符） + 子串包含判断
        def get_content_signature(content: str) -> str:
            """获取内容签名（简化的hash）：取去除空白后的前300字符"""
            normalized = re.sub(r"\s+", "", content)
            return normalized[:300] if normalized else ""

        kept_signatures: List[str] = []  # 已保留的父块的内容签名

        for parent in parent_chunks_sorted:
            pid = parent.get("parent_id", "")
            content = parent.get("content", "")

            if not content:
                removed_ids.append(pid)
                continue

            current_sig = get_content_signature(content)

            # 与所有已保留的父块做相似性判断
            is_duplicate = False
            for kept_sig in kept_signatures:
                # 判断标准：当前父块的核心内容是否已被保留的父块包含
                # 1. 完全相同（前300字符完全一致）
                if current_sig == kept_sig:
                    is_duplicate = True
                    break
                # 2. 当前父块的核心内容是已保留父块内容的子串（表明被包含）
                #    取当前父块的一个代表性片段（前150字符），检查是否在已保留的内容中
                if len(current_sig) >= 50 and current_sig[:150] in kept_sig:
                    is_duplicate = True
                    break
                # 3. 反向：已保留的父块核心内容是当前父块内容的子串
                #    （这种情况下当前父块可能更大，但因为优先级较低，仍然删除）
                if len(kept_sig) >= 50 and kept_sig[:150] in current_sig:
                    is_duplicate = True
                    break

            if is_duplicate:
                removed_ids.append(pid)
                continue

            # 保留该父块
            kept_chunks.append(parent)
            kept_signatures.append(current_sig)

        final_count = len(kept_chunks)
        removed_count = original_count - final_count

        # --- 5. 按原顺序恢复（非按优先级排序，保持检索到的顺序） ---
        # 构建保留的parent_id集合
        kept_parent_ids = {p.get("parent_id", "") for p in kept_chunks}
        # 从原列表中筛选保留的
        result = [p for p in parent_chunks if p.get("parent_id", "") in kept_parent_ids]

        return result, {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removed_count,
            "removed_ids": removed_ids,
        }

    # ========== Step 4.5: 父块精炼（轻量 LLM 句子提取） ==========

    def _refine_parent_chunks(
        self,
        parent_chunks: List[Dict],
        reranked_children: List[RetrievalResult],
        query: str,
    ) -> Optional[List[Dict]]:
        """
        【v2 增强】对所有父块进行精炼 + 生成摘要。

        核心策略：
        1. 每个父块生成「摘要」（保留章节核心信息，不依赖子块命中）
        2. 有子块命中的父块：LLM 精炼 + 摘要
        3. 兄弟父块（无子块命中）：保留全文 + 摘要，不被长度限制截断
        4. 为每个 parent_chunks 元素添加 "summary" 和 "refinement_method" 字段

        返回: 统计信息列表；如果精炼器未启用或失败返回 None
        """
        if not parent_chunks:
            return None

        # 如果 chunk_refiner 未启用，确保不会崩溃，仍然为每个父块添加摘要和内容
        # （使用原始内容作为摘要和内容）
        has_refiner = getattr(self, "chunk_refiner", None) is not None

        # 构建 parent_id → 子块列表 的映射（帮助理解检索意图）
        parent_to_children: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for c in reranked_children:
            pid = c.metadata.get("parent_id", "")
            if pid:
                parent_to_children[pid].append(c)

        stats_list = []
        for parent in parent_chunks:
            parent_id = parent.get("parent_id", "")
            original_content = parent.get("content", "")
            chapter_title = parent.get("chapter_title", "")

            # 保存原始长度（用于日志/统计）
            parent["original_length"] = len(original_content)

            # 获取该父块的子块
            relevant_children = parent_to_children.get(parent_id, [])

            try:
                # --- Step 1: 为每个父块生成摘要（v2 核心改进）
                try:
                    if has_refiner:
                        summary_text, _ = self.chunk_refiner.summarize_parent_chunk(
                            original_content, chapter_title
                        )
                        parent["summary"] = summary_text
                    else:
                        # 无精炼器：使用前 200 字符作为摘要
                        parent["summary"] = original_content[:200]
                except Exception as e:
                    logger.warning(f"[父块摘要] 生成失败 (parent_id={parent_id}): {e}")
                    parent["summary"] = original_content[:200]

                # --- Step 2: 精炼（保留相关内容）
                # 兄弟父块（无子块命中）：保留全文
                if not relevant_children:
                    refined_content = original_content
                    parent["refinement_method"] = "sibling_skip_full"
                    stat = {
                        "method": "sibling_skip_full",
                        "original_length": len(original_content),
                        "refined_length": len(refined_content),
                        "compression_ratio": 1.0,
                    }
                elif has_refiner:
                    # 有子块命中的父块：正常精炼
                    refined_content, stat = self.chunk_refiner.refine_parent_chunk(
                        parent_content=original_content,
                        query=query,
                        child_hits=relevant_children,
                    )
                else:
                    # 无精炼器：保留全文
                    refined_content = original_content
                    parent["refinement_method"] = "full_content"
                    stat = {
                        "method": "full_content",
                        "original_length": len(original_content),
                        "refined_length": len(refined_content),
                        "compression_ratio": 1.0,
                    }
            except Exception as e:
                logger.warning(f"[父块精炼] 单个父块精炼失败 (parent_id={parent_id}): {e}")
                refined_content = original_content
                stat = {
                    "method": "error",
                    "original_length": len(original_content),
                    "refined_length": len(refined_content),
                    "compression_ratio": len(refined_content) / max(len(original_content), 1),
                    "error": str(e),
                }

            parent["content"] = refined_content
            parent["refinement_method"] = stat["method"]
            stats_list.append(stat)

        return stats_list

    # ========== Step 2: 混合检索 ==========

    def _hybrid_search_children(
        self,
        query: str,
        recall_k: int = 30,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        chapter_ids: Optional[list[str]] = None,
        hyde_doc: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        稠密 + 稀疏混合检索子块（增强版）

        策略:
        1. 优先走混合检索（稠密+稀疏，Qdrant RRF 融合）
        2. 如果启用 HyDE，用假设文档生成稠密向量，BM25 仍用原始查询
        3. 稀疏向量词项匹配数为 0 时：退化为纯稠密检索，不报错
        4. 如果稠密向量检索也为空：尝试用查询关键词做 fallback 关键字匹配
        """
        # 稠密向量（语义匹配）
        # HyDE: 用假设文档生成稠密向量，在向量空间中更接近真实文档
        dense_text = hyde_doc if hyde_doc else query
        dense_vector = self.embedder.embed_query(dense_text)
        if hyde_doc:
            logger.debug(f"HyDE: 使用假设文档向量进行稠密检索 (假设文档 {len(hyde_doc)} 字符)")

        # 稀疏向量（关键词匹配）—— 始终使用原始查询（BM25 不需要 HyDE）
        sparse_indices, sparse_values = self.sparse_embedder.query_encode(query)

        if not sparse_indices:
            # 稀疏向量为空（查询关键词都不在词表中），降级为纯稠密检索
            logger.debug(
                f"稀疏向量为空（查询关键词未匹配词表），使用纯稠密检索。"
                f"查询: '{query[:50]}...'"
            )
            results = self.vector_store.search_children_dense(
                query_vector=dense_vector,
                chapter_ids=chapter_ids,
                top_k=recall_k,
                doc_id=doc_id,
                tags=tags,
            )
        else:
            logger.debug(f"混合检索：稀疏向量 {len(sparse_indices)} 个词项参与匹配")
            results = self.vector_store.search_children_hybrid(
                query_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                chapter_ids=chapter_ids,
                top_k=recall_k,
                doc_id=doc_id,
                tags=tags,
            )

        children = []
        child_ids = []
        for pt in results:
            payload = pt.payload or {}
            child_ids.append(str(pt.id))
            children.append(RetrievalResult(
                content=payload.get("content_snippet", payload.get("content", "")),
                chapter_title=payload.get("chapter_title", ""),
                score=pt.score,
                source_type="hybrid",
                metadata={
                    "child_id": pt.id,
                    "parent_id": payload.get("parent_id", ""),
                    "chapter_id": payload.get("chapter_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "doc_id": payload.get("doc_id", ""),
                },
                parent_id=payload.get("parent_id", ""),
                child_id=pt.id,
            ))

        # 从 SQLite 拉取完整子块内容（Qdrant 只存了简短预览）
        if children and self.parent_store:
            full_children = self.parent_store.get_children_by_ids(child_ids)
            if full_children:
                for c in children:
                    cid = str(c.child_id)
                    if cid in full_children:
                        c.content = full_children[cid]["content"]
                        if not c.chapter_title and full_children[cid].get("chapter_title"):
                            c.chapter_title = full_children[cid]["chapter_title"]

        return children

    # ========== Step 6: 上下文构建 ==========

    def _build_context(
        self,
        parent_chunks: list[dict],
        reranked_children: list[RetrievalResult],
        query: str,
    ) -> tuple[str, list[dict]]:
        """
        【v2 增强】构建 RAG 上下文和溯源信息

        新格式：
        [来源 N] 章节: XX
        [父块摘要]: 该章节核心要点
        [精炼内容]: 与查询相关的详细内容

        Returns:
            (context_text, sources): 上下文字符串和溯源列表
        """
        if not parent_chunks:
            return "未找到相关文档内容。", []

        # 组装章节溯源
        chapter_info = {}
        for c in reranked_children:
            ch_title = c.chapter_title
            if ch_title and ch_title not in chapter_info:
                chapter_info[ch_title] = c.metadata.get("chapter_id", "")

        sources = [
            {
                "chapter_title": title,
                "chapter_id": ch_id,
                "relevance_score": round(
                    max(
                        (c.score for c in reranked_children if c.chapter_title == title),
                        default=0,
                    ),
                    4,
                ),
            }
            for title, ch_id in chapter_info.items()
        ]

        # 构建上下文文本（v2：新增摘要 + 章节标题 + 精炼内容）
        parts = []
        parts.append("【检索到的文档内容】\n")

        for i, parent in enumerate(parent_chunks):
            title = parent.get("chapter_title", "")
            content = parent.get("content", "")
            summary = parent.get("summary", "")
            refinement_method = parent.get("refinement_method", "unknown")

            header = f"[来源 {i + 1}]"
            if title:
                header += f" 章节: {title}"
            header += f" (精炼方式: {refinement_method})"
            parts.append(header)

            # Step 1: 显示摘要（v2 核心改进
            if summary and summary.strip():
                parts.append("【章节摘要】")
                parts.append(summary.strip())

            # Step 2: 显示精炼后的内容
            parts.append("\n【相关内容】")
            # 长度控制
            max_len = 4000
            if len(content) > max_len:
                parts.append(content[:max_len] + "\n...(内容过长，已截断)")
            else:
                parts.append(content)

            parts.append("")

        # 添加回答指引
        parts.append("【回答要求】")
        parts.append(f"1. 基于以上文档内容回答用户问题：{query}")
        parts.append("2. 请优先参考【章节摘要】中的核心信息，结合【相关内容】中的详细说明")
        parts.append("3. 如果文档内容不足以回答，请诚实说明")
        parts.append("4. 引用时注明来源章节")

        return "\n".join(parts), sources

    # ========== 辅助方法 ==========

    def _empty_result(self, rewritten_query: str) -> dict:
        return {
            "rewritten_query": rewritten_query,
            "child_hits": [],
            "parent_chunks": [],
            "context": "未找到相关文档内容。",
            "sources": [],
        }

    def simple_search(self, query: str) -> str:
        """简单检索（返回纯文本上下文）"""
        result = self.search(query)
        return result.get("context", "")

    def retrieve(self, query: str, **kwargs) -> dict:
        """兼容旧 API 的检索方法"""
        return self.search(query, **kwargs)