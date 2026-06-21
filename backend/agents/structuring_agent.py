"""
文档结构化 Agent (DocumentStructuringAgent) V4

核心设计理念：
  不依赖原文档的章节结构，而是主动识别语义主题，将任何格式的文档
  统一转换为结构化、层级清晰的 Markdown。

工作模式（双模式自动切换）：

  ▌模式 A: 结构模式（Structure Mode）
     - 触发条件：文档中有"第N章"、`##`、"1. xxx"、"一、xxx"等明确标题
     - 处理方式：基于已有标题结构规范化层级、清洗噪声

  ▌模式 B: 语义模式（Semantic Mode）
     - 触发条件：文档无明确章节标记（只是连续文本）
     - 处理方式：调用 LLM 按语义主题自动划分章节，生成结构化标题

使用方式：
    from backend.agents.structuring_agent import DocumentStructuringAgent
    agent = DocumentStructuringAgent()
    structured_md = agent.structure(raw_markdown_text)
"""

import re
import time
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from loguru import logger
from openai import OpenAI

from backend.config import settings
from backend.ops.metrics_collector import get_metrics_collector


# =====================================================================
#  数据结构
# =====================================================================

@dataclass
class ChapterBoundary:
    """章节边界信息"""
    line_number: int    # 章节标题所在行号（从1开始）
    title: str          # 章节标题
    level: int = 2      # 标题层级（1=一级, 2=二级, 3=三级）


@dataclass
class StructuredChapter:
    level: int
    title: str
    content: str
    source_line_range: Tuple[int, int] = (0, 0)


# =====================================================================
#  预编译正则
# =====================================================================

# Markdown 标题（通用识别）
_MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')

# 中文"第N章"格式标题
_CHINESE_CHAPTER_RE = re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部]')

# 数字编号章节（如"1. xxx"、"2.1 xxx"）
_NUM_CHAPTER_RE = re.compile(r'^(\d+(\.\d+)*)\s+[^\d]')

# 中文编号章节（如"一、xxx"、"二、xxx"）
_CN_NUM_CHAPTER_RE = re.compile(r'^[一二三四五六七八九十]+[、，,]\s+.+')

# 商品条目（如"商品 001: MacBook Pro 16寸"）
_PRODUCT_ITEM_RE = re.compile(r'^商品\s*(\d+)\s*[:：\s]*')

# 噪声特征关键词
_NOISE_TITLE_KEYWORDS = [
    'ftaobao 电商平台服务指南',
    '文档标题',
    'document title',
]


# =====================================================================
#  LLM 语义分章 Prompt
# =====================================================================

SEMANTIC_CHAPTER_SYSTEM_PROMPT = """你是一个专业的文档语义分析助手。你的任务是：
- 阅读完整文档内容，按**语义主题**识别章节边界，并为每个章节生成简洁明了的标题。

# 核心原则：你只负责识别边界，不负责重写内容

系统会根据你输出的边界信息，在原始文档对应位置插入章节标题。
**不要修改文档内容，不要添加解释，不要补充信息。**

# 分析原则（按重要性排序）

1. **语义完整性优先**：一个完整语义单元（如一个商品的名称+编号+品牌+售价+规格+补充说明）必须完整归为同一章节，**绝不可以被切分到多个章节**
2. **主题一致性**：同一主题/话题的内容应归为同一章节
3. **主题切换**：当话题从一个领域跳到另一个领域时，开启新章节
4. **层级统一规则**：
   - level 1：文档总标题（整篇文档只有1个）
   - level 2：一级章节（如「第一章 平台概述」、「商品详细信息 - 手机数码类」）
   - level 3：二级子章节/具体商品条目（如「商品 008: 小米扫地机器人 X20 Pro」、「2.1 手机数码类」等具体条目）
   - ★ 商品信息单元**统一用 level 3**，确保被下游切分器识别为独立章节
5. **避免过度切分**：不要把每个句子或每段都切成一章，主题相近的内容应合并
6. **边界准确性**：start_char / end_char 必须是完整段落/内容块的边界，绝不要切在句子中间

# 输出格式

请严格按以下 JSON 数组格式输出，不要添加任何其他文字：

[
  {"level": 2, "title": "章节标题1", "start_char": 0, "end_char": 1200, "is_product": false},
  {"level": 3, "title": "商品 008: 小米扫地机器人 X20 Pro", "start_char": 1201, "end_char": 3500, "is_product": true},
  {"level": 2, "title": "其他章节标题", "start_char": 3501, "end_char": 5000, "is_product": false}
]

字段说明：
- level: 章节层级（1=文档标题, 2=一级章节, 3=二级子章节/具体商品条目）
- title: 章节标题（简洁明了，不超过 40 字；商品用商品名称+编号作为标题）
- start_char: 该章节在原文中的起始字符位置（从0开始）——必须是行首或段落开头
- end_char: 该章节在原文中的结束字符位置——必须是行尾或段落结尾
- is_product: 是否为商品信息单元（true/false）。商品信息单元应包含该商品的名称、价格、规格等所有相关信息
"""

SEMANTIC_CHAPTER_USER_PROMPT = """以下是需要分析的文档内容：

---- 文档开始 ---
{document_content}
---- 文档结束 ---

请识别上述文档的章节边界。特别注意：如果文档包含商品信息（如商品名称、商品编号、商品售价、详细规格等），请将属于同一个商品的所有信息合并为一个完整的商品信息单元（标记 is_product: true）。输出 JSON 数组格式的章节边界方案，不要修改任何文档内容。
"""


# 层级常量
L1_DOC_TITLE = 1
L2_CHAPTER = 2


class DocumentStructuringAgent:
    """文档结构化 Agent（V4 - 双模式：结构模式 + 语义模式）"""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._init_llm_client()
        self.metrics_collector = get_metrics_collector()
        logger.info("[DocumentStructuringAgent] 初始化完成")

    def _init_llm_client(self):
        """初始化 LLM 客户端（用于语义分章）"""
        api_key = getattr(settings, 'llm_api_key', None)
        api_base = getattr(settings, 'llm_api_base', None)
        if not api_key:
            logger.warning("[DocumentStructuringAgent] LLM API Key 未配置，语义模式将降级")
            return
        try:
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            logger.info(f"[DocumentStructuringAgent] LLM 客户端就绪: model={getattr(settings, 'llm_model', 'deepseek-chat')}")
        except Exception as e:
            logger.warning(f"[DocumentStructuringAgent] LLM 客户端初始化失败: {e}")
            self.client = None

    # =====================================================================
    #  主入口
    # =====================================================================

    def structure(self, raw_markdown: str, session_id: str = "indexing") -> str:
        """
        文档结构化主入口（接入 ops 监控）

        Args:
            raw_markdown: 待结构化的原始 Markdown
            session_id: 用于监控的会话 ID（文档索引时默认为 "indexing"）
        """
        _start = time.time()
        try:
            if not raw_markdown or not raw_markdown.strip():
                self.metrics_collector.record_request(
                    endpoint="agent.structuring",
                    duration_ms=0,
                    status="success",
                    agent="structuring_agent",
                    tokens=0,
                    session_id=session_id,
                )
                return raw_markdown

            lines = raw_markdown.split('\n')
            total_lines = len(lines)
            total_chars = len(raw_markdown)
            logger.info(f"[Structuring] 开始结构化，原始文档 {total_lines} 行, {total_chars} 字符")

            # ---- Step 0: 文档整体去重 ----
            lines = self._dedup_whole_document(lines)
            logger.info(f"[Structuring] Step0: 整体去重后 {len(lines)} 行")

            # ---- Step 1: 模式检测 ----
            mode = self._detect_mode(lines)
            logger.info(f"[Structuring] Step1: 检测为 {mode} 模式")

            # ---- Step 2: 根据模式选择处理路径 ----
            if mode == "structure":
                output = self._process_structure_mode(lines)
            else:
                output = self._process_semantic_mode(lines, session_id=session_id)

            _duration_ms = (time.time() - _start) * 1000
            logger.info(f"[Structuring] Step4: 输出完成，{len(output)} 字符，耗时 {_duration_ms:.0f}ms")

            # ---- ops 监控：记录整体结构化请求 ----
            self.metrics_collector.record_request(
                endpoint="agent.structuring",
                duration_ms=_duration_ms,
                status="success",
                agent="structuring_agent",
                tokens=0,
                session_id=session_id,
            )
            return output
        except Exception as e:
            _duration_ms = (time.time() - _start) * 1000
            logger.error(f"[Structuring] 结构化异常: {e}")
            # ---- ops 监控：记录错误 ----
            self.metrics_collector.record_request(
                endpoint="agent.structuring",
                duration_ms=_duration_ms,
                status="error",
                agent="structuring_agent",
                tokens=0,
                session_id=session_id,
            )
            return raw_markdown

    # =====================================================================
    #  Step 0: 文档整体去重
    # =====================================================================

    def _dedup_whole_document(self, lines: List[str]) -> List[str]:
        """
        检测并处理"整份文档被复制粘贴了多份"的情况。
        """
        all_chapter_positions: List[Tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            title_text = None
            md_match = _MD_HEADING_RE.match(stripped)
            if md_match:
                title_text = md_match.group(2).strip()
            elif _CHINESE_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                title_text = stripped
            elif _NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                title_text = stripped

            if title_text and _CHINESE_CHAPTER_RE.match(title_text):
                num_match = re.search(r'[一二三四五六七八九十百千\d]+', title_text)
                if num_match:
                    try:
                        chapter_num = self._chinese_to_number(num_match.group())
                        all_chapter_positions.append((idx, chapter_num, title_text))
                    except Exception:
                        pass

        if len(all_chapter_positions) < 3:
            return lines

        max_num = max(p[1] for p in all_chapter_positions)
        max_pos_idx = None
        for i, (_, num, _) in enumerate(all_chapter_positions):
            if num == max_num:
                max_pos_idx = i
                break

        if max_pos_idx is None or max_pos_idx >= len(all_chapter_positions) - 1:
            return lines

        max_pos_line_num = all_chapter_positions[max_pos_idx][0]
        for i in range(max_pos_idx + 1, len(all_chapter_positions)):
            pos, num, title = all_chapter_positions[i]
            if num <= max(max_num // 2, 3):
                logger.info(
                    f"  [整体去重] 检测到章节编号回溯: 第{max_num}章在第{max_pos_line_num + 1}行，"
                    f"第{num}章'{title[:30]}'在第{pos + 1}行重复出现"
                )
                logger.info(f"  [整体去重] 保留第 1-{pos} 行，去除后续重复内容")
                return lines[:pos]

        return lines

    def _chinese_to_number(self, chinese_num: str) -> int:
        chinese_num = chinese_num.strip()
        if not chinese_num:
            return 0
        if re.match(r'^\d+$', chinese_num):
            return int(chinese_num)

        digit_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
                     '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        unit_map = {'十': 10, '百': 100, '千': 1000}

        result = 0
        current = 0
        for ch in chinese_num:
            if ch in digit_map:
                current = digit_map[ch]
            elif ch in unit_map:
                unit = unit_map[ch]
                if current == 0:
                    result += unit
                else:
                    result += current * unit
                    current = 0
            else:
                break

        result += current
        return result if result > 0 else 100

    # =====================================================================
    #  Step 1: 模式检测
    # =====================================================================

    def _detect_mode(self, lines: List[str]) -> str:
        """
        检测文档模式：
        - "structure": 有明确的章节标记（"第N章"、`##`、数字编号等），且不包含商品信息
        - "semantic": 没有明确章节标记，或包含商品信息（由 LLM 进行语义分析）

        ★ 关键规则：当文档包含商品信息（商品、售价、规格等）时，强制使用语义模式，
           让 LLM 识别并合并同一个商品的信息单元。
        """
        structure_signals = 0
        total_non_empty = 0
        product_related_lines = 0  # 检测到商品相关关键词的行数

        # 商品相关关键词检测
        PRODUCT_KEYWORDS = ['商品', '产品', '售价', '价格', '规格', '品牌', '编号', '型号']

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            total_non_empty += 1

            # 检测商品相关内容
            for kw in PRODUCT_KEYWORDS:
                if kw in stripped:
                    product_related_lines += 1
                    break

            # 信号 1: 中文章节标记（"第N章"、"第N节"）
            if _CHINESE_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                structure_signals += 1
                continue

            # 信号 2: Markdown 标题（# ## ### ####）
            md_match = _MD_HEADING_RE.match(stripped)
            if md_match:
                heading_text = md_match.group(2).strip()
                # 排除噪声标题
                norm_text = heading_text.lower()
                if not any(noise in norm_text for noise in _NOISE_TITLE_KEYWORDS):
                    structure_signals += 1
                continue

            # 信号 3: 数字编号章节（"1. xxx"、"1.1 xxx"）
            if _NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                structure_signals += 1
                continue

            # 信号 4: 中文数字编号（"一、xxx"、"二、xxx"）
            if _CN_NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                structure_signals += 1
                continue

        # ★ 决策逻辑：
        # 1. 如果文档包含商品相关内容（>= 2 行检测到商品关键词）→ 强制语义模式
        #    理由：结构模式可能把商品信息切分到不同章节，需要 LLM 语义识别来合并
        product_ratio = product_related_lines / max(total_non_empty, 1)
        if product_related_lines >= 2 or product_ratio > 0.1:
            logger.info(
                f"  [模式检测] 检测到商品相关内容 {product_related_lines}/{total_non_empty} 行 "
                f"→ 强制使用语义模式（LLM 识别商品信息单元）"
            )
            return "semantic"

        # 2. 如果有 >= 3 个结构信号 → 结构模式
        if structure_signals >= 3:
            logger.info(f"  [模式检测] 结构信号={structure_signals}/{total_non_empty}行 → 使用结构模式")
            return "structure"

        # 3. 否则 → 语义模式
        logger.info(f"  [模式检测] 结构信号={structure_signals}/{total_non_empty}行 → 使用语义模式")
        return "semantic"

    # =====================================================================
    #  模式 A: 结构模式处理
    # =====================================================================

    def _process_structure_mode(self, lines: List[str]) -> str:
        """
        结构模式：基于已有标题结构进行规范化。
        """
        # ---- Step 1: 扫描章节边界 ----
        boundaries = self._scan_chapter_boundaries_structure(lines)
        logger.info(f"  [结构模式] 识别到 {len(boundaries)} 个章节边界")

        if not boundaries:
            logger.warning("  [结构模式] 未识别到有效章节，降级为语义模式")
            return self._process_semantic_mode(lines, session_id="indexing")

        # ---- Step 2: 构建章节块 ----
        chapters = self._build_chapters_from_boundaries(lines, boundaries)
        logger.info(f"  [结构模式] 构建 {len(chapters)} 个章节")

        # ---- Step 3: 渲染 Markdown ----
        return self._render_markdown(chapters)

    def _scan_chapter_boundaries_structure(self, lines: List[str]) -> List[ChapterBoundary]:
        """
        在结构模式下扫描章节边界。
        识别规则（优先级从高到低）：
        1. "第N章 xxx" → level 2
        2. `## xxx` → level 2
        3. `# xxx` → level 1
        4. `### xxx` → level 3
        5. "1. xxx" / "一、xxx" → level 2
        """
        boundaries: List[ChapterBoundary] = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 200:
                continue

            title_text = None
            level = 2

            md_match = _MD_HEADING_RE.match(stripped)

            # 规则 1 & 2 & 3: Markdown 标题
            if md_match:
                md_level = len(md_match.group(1))
                text = md_match.group(2).strip()

                # 过滤噪声标题
                norm_text = text.lower()
                if any(noise in norm_text for noise in _NOISE_TITLE_KEYWORDS):
                    continue

                # 根据 # 的数量决定层级
                if md_level == 1:
                    level = 1
                    title_text = text
                elif md_level == 2:
                    level = 2
                    title_text = text
                elif md_level == 3:
                    level = 3
                    title_text = text
                else:
                    # === 商品信息单元特殊处理 ===
                    # `#### 商品 NNN` 虽然是 level 4，但这是重要的语义单元
                    # 必须识别为独立章节（降级为 level 3）
                    if _PRODUCT_ITEM_RE.match(text):
                        level = 3
                        title_text = text
                    else:
                        # 其他 #### 及更深 → 不识别为章节边界（保留为章节内容）
                        continue

            # 规则 4: "第N章 xxx" 格式
            elif _CHINESE_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                level = 2
                title_text = stripped

            # 规则 5: 数字编号 "1. xxx"
            elif _NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                level = 2
                title_text = stripped

            # 规则 6: 中文数字编号 "一、xxx"
            elif _CN_NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
                level = 2
                title_text = stripped

            if title_text:
                # 规范化标题
                normalized_title = re.sub(r'\s+', ' ', title_text.strip())
                boundaries.append(ChapterBoundary(
                    line_number=idx + 1,
                    title=normalized_title,
                    level=level,
                ))

        return boundaries

    def _build_chapters_from_boundaries(
        self,
        lines: List[str],
        boundaries: List[ChapterBoundary],
    ) -> List[StructuredChapter]:
        """
        根据章节边界构建章节块。
        """
        chapters: List[StructuredChapter] = []
        seen_titles: Dict[str, int] = {}

        # 文档前言（第一个章节之前的内容）
        first_chapter_line = boundaries[0].line_number
        preamble_lines = lines[:first_chapter_line - 1]
        preamble_content = self._clean_chapter_content(preamble_lines)
        if preamble_content.strip():
            chapters.append(StructuredChapter(
                level=1,
                title="文档前言",
                content=preamble_content,
                source_line_range=(1, first_chapter_line - 1),
            ))

        # 构建每个章节的内容
        for i, boundary in enumerate(boundaries):
            start_line = boundary.line_number  # 1-based, inclusive
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1].line_number  # 1-based, exclusive
            else:
                end_line = len(lines) + 1  # 文档末尾

            # 提取该章节的内容（标题行之后，到下一章之前）
            content_lines = lines[start_line:end_line - 1]  # 0-based, 不含标题行

            # 清洗章节内容
            cleaned_content = self._clean_chapter_content(content_lines)

            # 去重
            title_key = re.sub(r'\s+', ' ', boundary.title.strip().lower())
            if title_key in seen_titles:
                logger.info(f"  [去重] 章节 '{boundary.title[:40]}' 重复，跳过")
                continue
            seen_titles[title_key] = len(chapters)

            chapters.append(StructuredChapter(
                level=boundary.level,
                title=boundary.title,
                content=cleaned_content,
                source_line_range=(start_line, end_line - 1),
            ))

        return chapters

    # =====================================================================
    #  模式 B: 语义模式处理（核心新能力）
    # =====================================================================

    def _process_semantic_mode(self, lines: List[str], session_id: str = "indexing") -> str:
        """
        语义模式（V2：整文档直接丢给 LLM，>5万token才分块）

        处理流程：
        1. 合并为连续文本
        2. 估算 token：中文约 4 字符 = 1 token，5 万 token ≈ 20 万字符
        3. < 20 万字符：一次性把完整文档丢给 LLM
        4. >= 20 万字符：按现有文档结构标记（## / 第N章 / 数字编号）智能分块，避免在语义中间切断
        5. LLM 输出章节边界 → 调用验证 Agent 做 LLM 语义评审
        6. 在章节边界行首插入 `##`/`###` 标题，★ 绝不修改、截断原始内容
        """
        logger.info("  [语义模式] 开始语义分章处理（整文档模式）...")

        # 合并为连续文本
        full_text = '\n'.join(lines)
        doc_len = len(full_text)

        # 步骤 1: 检测是否有 LLM 客户端
        if not self.client:
            logger.warning("  [语义模式] LLM 不可用，使用规则降级模式")
            return self._semantic_mode_fallback(lines)

        # 步骤 2: 估算 token，决定是否分块
        # 中文约 4 字符 = 1 token → 5 万 token ≈ 20 万字符
        TOKEN_SAFE_THRESHOLD = 200000  # 5 万 token 的安全字符数上限

        if doc_len < TOKEN_SAFE_THRESHOLD:
            # ★ 整文档模式：一次性把完整内容丢给 LLM
            logger.info(f"  [语义模式] 整文档模式: {doc_len} 字符（< 5万token阈值），一次性丢给 LLM")
            all_chapters = self._call_llm_semantic_split(full_text, base_offset=0, session_id=session_id)
            if all_chapters:
                product_count = sum(1 for c in all_chapters if c.get('is_product', False))
                logger.info(f"  [语义模式] 第一轮 LLM 识别 {len(all_chapters)} 个章节，其中 {product_count} 个商品信息单元")
        else:
            # ★ 超大文档：按现有文档结构智能分块
            logger.info(f"  [语义模式] 超大文档模式: {doc_len} 字符（>= 5万token阈值），按结构标记分块")
            all_chapters = self._split_and_call_llm_by_structure(full_text, lines, session_id=session_id)

        if not all_chapters:
            logger.warning("  [语义模式] 第一轮 LLM 无输出，返回原文档")
            return full_text

        # 步骤 3: 调用验证 Agent —— LLM 语义评审 + 修正
        # =====================================================================
        # ★ 纯 LLM 评审：把第一轮 LLM 章节边界交给第二轮 LLM 做质量评审
        #   - 评审每个章节边界是否语义合理
        #   - 对错误/不完整的章节给出修正边界
        #   - 0 规则，100% LLM 语义判断
        # =====================================================================
        try:
            from backend.agents.structuring_verifier import ChapterBoundaryVerificationAgent
            verifier = ChapterBoundaryVerificationAgent(llm_client=self.client, llm_model=getattr(settings, 'llm_model', 'deepseek-chat'))
            verified_chapters = verifier.verify_and_refine(all_chapters, full_text, session_id=session_id)
            if verified_chapters:
                all_chapters = verified_chapters
                refined_count = sum(1 for c in all_chapters if c.get('source') == 'llm-refined')
                logger.info(f"  [语义模式] LLM 评审完成: {len(all_chapters)} 章节，{refined_count} 个由 LLM 修正")
        except Exception as e:
            logger.warning(f"  [语义模式] 验证 Agent 异常: {e}，使用第一轮结果")

        if not all_chapters:
            logger.warning("  [语义模式] 无有效章节，返回原文档")
            return full_text

        # =====================================================================
        # ★ 标题插入：仅在章节边界行首插入标题，永不修改、截断原始内容
        # =====================================================================

        output_parts: List[str] = []
        last_end = 0

        # 按 start_char 排序，确保顺序正确
        all_chapters.sort(key=lambda c: c.get('start_char', 0))

        # 吸附：确保所有 start_char 都是行边界（\n 之后）
        for chap in all_chapters:
            start = int(chap.get('start_char', 0))
            # 如果不在行首，向后找最近的换行符
            if 0 < start < doc_len and full_text[start - 1] != '\n':
                # 在 start 之后找换行符（优先）
                newline_after = full_text.find('\n', start, min(start + 200, doc_len))
                if newline_after > 0:
                    chap['start_char'] = newline_after + 1
                else:
                    # 向前找
                    newline_before = full_text.rfind('\n', max(0, start - 200), start)
                    if newline_before >= 0:
                        chap['start_char'] = newline_before + 1

        inserted_count = 0
        for chap in all_chapters:
            start = int(chap.get('start_char', 0))
            level = int(chap.get('level', 2))
            title = chap.get('title', '未命名章节')
            level = max(1, min(4, level))

            if start < last_end or start >= doc_len:
                continue  # 跳过无效/重叠位置

            # --- 原始内容，一字不动 ---
            if start > last_end:
                output_parts.append(full_text[last_end:start])

            # --- 在此处插入章节标题，不截断、不删除任何原始文本 ---
            boundary_text = full_text[start:start + 150].lstrip()
            heading_prefix = '#' * level + ' '
            already_has_heading = (
                boundary_text.startswith('#' * level + ' ') or
                boundary_text.startswith('#' * (level + 1) + ' ')
            )

            if not already_has_heading:
                output_parts.append(heading_prefix + title + '\n\n')
                inserted_count += 1
                logger.debug(
                    f"    插入标题 [{level}级] {title[:40]} @ char {start}"
                )

            last_end = start

        # --- 剩余原始文本，一字不动 ---
        if last_end < doc_len:
            output_parts.append(full_text[last_end:])

        structured_text = ''.join(output_parts)

        product_count = sum(1 for c in all_chapters if c.get('is_product', False))
        logger.info(
            f"  [语义模式] 完成: {len(all_chapters)} 个章节（{product_count} 个商品单元）, "
            f"新插入 {inserted_count} 个标题, 原始 {len(full_text)} → 结构化 {len(structured_text)} 字符"
        )
        return structured_text

    def _split_and_call_llm_by_structure(self, full_text: str, lines: List[str], session_id: str = "indexing") -> List[Dict]:
        """
        超大文档分块：按现有文档结构标记（## / 第N章 / 数字编号 / ####）智能分块，
        避免在语义中间切断。每块 8~15 万字符（约 2~3.5 万 token），确保 LLM 可处理。
        """
        doc_len = len(full_text)
        # 目标块大小：约 10 万字符（~2.5 万 token），留 8k 给 prompt + 输出
        TARGET_CHUNK = 100000
        MIN_CHUNK = 60000

        # 步骤 1: 扫描结构标记的行号（作为天然分块断点）
        # 按优先级：## 二级标题 > 第N章 > 数字编号 > ####
        break_points: List[int] = []  # 存储 char position 断点
        current_char_pos = 0

        for idx, line in enumerate(lines):
            stripped = line.strip()
            is_break = False

            # 优先级 1: ## 二级标题
            if stripped.startswith('## '):
                is_break = True
            # 优先级 2: 第N章 格式
            elif re.match(r'^第[一二三四五六七八九十百千\d]+[章节篇部]', stripped) and len(stripped) < 100:
                is_break = True
            # 优先级 3: 数字编号 1. xxx
            elif re.match(r'^(\d+)\.\s+\S', stripped) and len(stripped) < 100:
                is_break = True

            if is_break:
                break_points.append(current_char_pos)

            # 累加字符位置（+1 for \n）
            current_char_pos += len(line) + 1

        # 步骤 2: 在断点之间按目标大小切分
        chunks: List[Tuple[int, int]] = []  # (start_char, end_char)

        if len(break_points) <= 1:
            # 几乎没有结构标记 → 按大小简单分块
            pos = 0
            while pos < doc_len:
                end = min(pos + TARGET_CHUNK, doc_len)
                if end < doc_len:
                    # 向后找 \n\n 双换行
                    double_newline = full_text.find('\n\n', end - 5000, end + 5000)
                    if double_newline > 0:
                        end = double_newline
                    else:
                        # 找不到双换行，找单换行
                        newline = full_text.find('\n', end - 1000, end + 1000)
                        if newline > 0:
                            end = newline
                chunks.append((pos, end))
                pos = end
        else:
            # 有结构标记 → 用其作为天然断点来分
            # 先加入文档末尾作为潜在断点
            break_points.append(doc_len)

            # 在断点间按 TARGET_CHUNK 为目标合并或拆分
            i = 0
            while i < len(break_points) - 1:
                chunk_start = break_points[i]
                # 尝试累积到达到 MIN_CHUNK
                accumulated = break_points[i + 1] - break_points[i]
                j = i + 1

                while j < len(break_points) - 1 and accumulated < MIN_CHUNK:
                    j += 1
                    accumulated = break_points[j] - break_points[i]

                # 如果累积块太大 → 在 break_points 中间再切
                while accumulated > TARGET_CHUNK * 1.5 and j > i + 1:
                    j -= 1
                    accumulated = break_points[j] - break_points[i]

                chunk_end = break_points[j]
                if chunk_end > chunk_start + 100:
                    chunks.append((chunk_start, chunk_end))
                i = j

        logger.info(f"  [语义模式] 大文档分为 {len(chunks)} 块")

        # 步骤 3: 对每个块调用 LLM
        all_chapters: List[Dict] = []
        for idx, (start_pos, end_pos) in enumerate(chunks, 1):
            chunk_text = full_text[start_pos:end_pos]
            try:
                logger.info(f"  [语义模式] 处理第 {idx}/{len(chunks)} 块 ({len(chunk_text)} 字符)...")
                chapter_plan = self._call_llm_semantic_split(chunk_text, base_offset=start_pos, session_id=session_id)
                if chapter_plan:
                    all_chapters.extend(chapter_plan)
                    product_count = sum(1 for c in chapter_plan if c.get('is_product', False))
                    logger.info(f"  [语义模式] 第{idx}块识别 {len(chapter_plan)} 章节，{product_count} 商品")
            except Exception as e:
                logger.warning(f"  [语义模式] 第{idx}块 LLM 调用失败: {e}")
                all_chapters.append({
                    'level': 2,
                    'title': f'内容段落 {idx}',
                    'start_char': start_pos,
                    'end_char': end_pos,
                    'is_product': False,
                })

        return all_chapters

    def _call_llm_semantic_split(self, text_chunk: str, base_offset: int = 0, session_id: str = "indexing") -> Optional[List[Dict]]:
        """
        调用 LLM 对一段文本进行语义分章（接入 ops 监控）。

        Args:
            text_chunk: 待分章的文本
            base_offset: 该文本块在原文中的起始位置（用于字符位置校正）
            session_id: 用于监控的会话 ID

        Returns:
            章节列表，如 [{"level": 2, "title": "xxx", "start_char": 0, "end_char": 1200}, ...]
            失败时返回 None
        """
        _start = time.time()
        try:
            model = getattr(settings, 'llm_model', 'deepseek-chat')
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SEMANTIC_CHAPTER_SYSTEM_PROMPT},
                    {"role": "user", "content": SEMANTIC_CHAPTER_USER_PROMPT.format(
                        document_content=text_chunk
                    )},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            # ---- ops 监控：记录 LLM 调用指标 ----
            _duration_ms = (time.time() - _start) * 1000
            _tokens = 0
            try:
                if getattr(response, 'usage', None):
                    _tokens = response.usage.total_tokens
            except Exception:
                pass
            self.metrics_collector.record_request(
                endpoint="agent.structuring.llm_split",
                duration_ms=_duration_ms,
                status="success",
                agent="structuring_agent",
                tokens=_tokens,
                session_id=session_id,
            )

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"    LLM 原始输出: {result_text[:200]}...")

            # 解析 JSON
            # 有时 LLM 会在 JSON 前后加其他文字，这里做容错处理
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if not json_match:
                logger.warning("    未检测到 JSON 格式输出")
                return None

            json_text = json_match.group()
            try:
                chapters = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"    JSON 解析失败: {e}")
                # 尝试简单清理
                json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
                try:
                    chapters = json.loads(json_text)
                except Exception:
                    return None

            if not isinstance(chapters, list):
                logger.warning(f"    输出不是数组格式")
                return None

            # 校正字符位置（加上 base_offset）
            for chapter in chapters:
                if 'start_char' in chapter:
                    chapter['start_char'] = int(chapter['start_char']) + base_offset
                if 'end_char' in chapter:
                    chapter['end_char'] = int(chapter['end_char']) + base_offset

            return chapters

        except Exception as e:
            _duration_ms = (time.time() - _start) * 1000
            logger.error(f"  [语义分章] LLM 调用失败: {e}")
            self.metrics_collector.record_request(
                endpoint="agent.structuring.llm_split",
                duration_ms=_duration_ms,
                status="error",
                agent="structuring_agent",
                tokens=0,
                session_id=session_id,
            )
            return None

    def _semantic_mode_fallback(self, lines: List[str]) -> str:
        """
        LLM 不可用时的语义模式降级方案。
        策略：
        1. 基于段落长度进行粗略切分
        2. 检测段落中的"主题关键词"（如"商品"、"服务"、"政策"等）
        3. 将相关段落归为同一章节
        """
        logger.info("  [语义模式] 使用规则降级方案")

        # 合并段落
        paragraphs: List[Tuple[int, str]] = []  # (start_line, text)
        current_paragraph_lines: List[str] = []
        current_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # 空行 → 段落结束
                if current_paragraph_lines:
                    para_text = '\n'.join(current_paragraph_lines).strip()
                    if para_text:
                        paragraphs.append((current_start + 1, para_text))
                    current_paragraph_lines = []
                    current_start = i
            else:
                if not current_paragraph_lines:
                    current_start = i
                current_paragraph_lines.append(line)

        # 处理最后一个段落
        if current_paragraph_lines:
            para_text = '\n'.join(current_paragraph_lines).strip()
            if para_text:
                paragraphs.append((current_start + 1, para_text))

        # 主题关键词检测
        topic_keywords = {
            '商品与产品': ['商品', '产品', '目录', '分类', '编号', '售价', '简介', '规格'],
            '服务与支持': ['客服', '服务', '售后', '支持', '电话', '邮箱', '在线'],
            '政策与流程': ['政策', '流程', '规定', '退换', '保修', '退货', '退款'],
            '会员与积分': ['会员', '积分', '等级', '特权', '权益'],
            '支付与配送': ['支付', '配送', '物流', '运费', '发货'],
            '平台简介': ['平台', '简介', '概述', '公司', '承诺', '理念'],
        }

        # 将段落按主题分组
        chapters_content: Dict[str, List[str]] = {}

        for start_line, para_text in paragraphs:
            # 检测段落主题
            matched_topic = None
            for topic, keywords in topic_keywords.items():
                for kw in keywords:
                    if kw in para_text:
                        matched_topic = topic
                        break
                if matched_topic:
                    break

            # 未检测到主题 → 归入"其他内容"
            if not matched_topic:
                matched_topic = "其他内容"

            if matched_topic not in chapters_content:
                chapters_content[matched_topic] = []
            chapters_content[matched_topic].append(para_text)

        # 渲染为 Markdown
        output_lines = []
        for topic, contents in chapters_content.items():
            output_lines.append(f"## {topic}")
            output_lines.append("")
            output_lines.append('\n\n'.join(contents))
            output_lines.append("")

        if not output_lines:
            return '\n'.join(lines)

        return '\n'.join(output_lines).strip() + '\n'

    # =====================================================================
    #  通用: 章节内容清洗
    # =====================================================================

    def _clean_chapter_content(self, content_lines: List[str]) -> str:
        """
        基础内容清洗：仅移除高频垃圾标题，不强行重组商品信息。
        商品信息的识别和重组由 LLM 语义模式负责。
        """
        if not content_lines:
            return ""

        # Step 1: 扫描所有 Markdown 标题行（用于识别高频垃圾标题）
        heading_counter: Counter = Counter()
        for line in content_lines:
            stripped = line.strip()
            md_match = _MD_HEADING_RE.match(stripped)
            if md_match:
                text = md_match.group(2).strip().lower()
                norm_text = re.sub(r'[\s_:\-：【】\[\]]+', ' ', text).strip()
                heading_counter[norm_text] += 1

        # Step 2: 识别垃圾标题
        noise_headings = set()
        for heading_text, count in heading_counter.items():
            # 关键词匹配的噪声标题
            for noise_kw in _NOISE_TITLE_KEYWORDS:
                if noise_kw in heading_text:
                    noise_headings.add(heading_text)
                    break
            # 高频出现的标题（>= 3 次）
            if count >= 3:
                noise_headings.add(heading_text)

        # Step 3: 清理内容（移除垃圾标题行，保留其他内容）
        cleaned_lines: List[str] = []
        consecutive_empty = 0

        for line in content_lines:
            stripped = line.strip()

            # 检查是否为 Markdown 标题
            md_match = _MD_HEADING_RE.match(stripped)
            if md_match:
                heading_text = md_match.group(2).strip().lower()
                norm_heading = re.sub(r'[\s_:\-：【】\[\]]+', ' ', heading_text).strip()

                # 如果是垃圾标题，跳过该行
                if norm_heading in noise_headings:
                    continue

                cleaned_lines.append(line)
                consecutive_empty = 0
                continue

            # 空行处理
            if not stripped:
                if consecutive_empty < 2:
                    cleaned_lines.append(line)
                consecutive_empty += 1
                continue

            # 普通内容行
            cleaned_lines.append(line)
            consecutive_empty = 0

        return '\n'.join(cleaned_lines).strip()

    # =====================================================================
    #  辅助方法
    # =====================================================================

    def _line_is_title(self, line: str) -> bool:
        """判断一行是否为标题行（用于语义模式下避免标题重复）"""
        stripped = line.strip()
        if not stripped:
            return False

        if _MD_HEADING_RE.match(stripped):
            return True
        if _CHINESE_CHAPTER_RE.match(stripped) and len(stripped) < 80:
            return True
        if _NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
            return True
        if _CN_NUM_CHAPTER_RE.match(stripped) and len(stripped) < 80:
            return True

        return False

    # =====================================================================
    #  渲染 Markdown
    # =====================================================================

    def _render_markdown(self, chapters: List[StructuredChapter]) -> str:
        output_lines: List[str] = []

        for chapter in chapters:
            # 确保 level 在合理范围
            level = max(1, min(6, int(chapter.level)))
            output_lines.append(f"{'#' * level} {chapter.title}")

            if chapter.content:
                output_lines.append("")
                output_lines.append(chapter.content)

            output_lines.append("")

        return '\n'.join(output_lines).strip() + '\n'
