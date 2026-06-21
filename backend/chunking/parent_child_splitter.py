"""
章节分割器：根据目录将文档按章节切分，并进一步做父子块切分。

多层多级切块策略：
  第一层：根据目录抽象每个章节
  第二层：每个章节内进行子块切分（自然段落边界 + 滑动窗口）
  第三层：对子块进行父块组织（每 N 个子块一组，用于上下文扩展）

注意：
  子块直接参与检索（向量+BM25双路），父块用于组织关联和构建上下文时扩展相邻块。
"""

from typing import Optional, Iterable
from dataclasses import dataclass, field
from uuid import uuid4
import re

from loguru import logger

from backend.chunking.toc_extractor import Chapter, TOCExtractor
from backend.preprocessing.parser import ParsedDocument, DocumentMeta


# 中文标点边界（优先在句号、分号、冒号等处分界）
_SENTENCE_BOUNDARY = set("。！？；!?;；\n")

# 方案A：结构化列表项检测模式（行首匹配）
# 1. 阿拉伯数字编号: 1. / 2、 / 3：
# 2. 中文数字编号: 一、/ 二. / 三
# 3. 方括号编号: 【1】/ [2]
# 4. 圆圈编号: ①②③
# 5. Markdown子章节标题: ### 2.1 xxx / #### 3.1.2 yyy（新增：修复检索分类信息丢失）
_LIST_ITEM_PATTERNS = [
    re.compile(r'^\s*\d+[.、:：]\s+\S'),        # 1. / 1、/ 1：
    re.compile(r'^\s*[一二三四五六七八九十百]+[、.:：]\s+\S'),  # 一、/ 二.
    re.compile(r'^\s*[【\[]\s*\d+\s*[】\]]\s*\S'),  # 【1】/ [1]
    re.compile(r'^\s*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕]\s*\S'),  # ①②③
    re.compile(r'^\s*#{3,}\s+\d+[.\d]*\s+\S'),   # ### 2.1 xxx / #### 3.1.2 yyy
]

# 结构化列表判定阈值：检测到至少多少个列表项才视为结构化文档
_STRUCTURED_MIN_ITEMS = 3

# 结构化列表：每个子块包含多少个列表项
_STRUCTURED_ITEMS_PER_CHUNK = 4


@dataclass
class ChildChunk:
    """子块：细粒度的文本片段，直接参与检索"""
    chunk_id: str
    chapter_id: str
    chapter_title: str
    content: str
    chunk_index: int       # 在该章节内的索引
    parent_id: str = ""    # 所属父块 ID（生成父块时回写）
    metadata: dict = field(default_factory=dict)
    doc_meta: Optional[DocumentMeta] = None  # 来源文档元数据


@dataclass
class ParentChunk:
    """父块：覆盖多个子块，用于上下文扩展时获取相邻块"""
    parent_id: str
    chapter_id: str
    chapter_title: str
    summary: str            # 子块群的内容摘要（前端预选用）
    child_ids: list[str]    # 关联的子块 ID 列表
    content_snippet: str = ""  # 拼接后的子块内容（预留，不用于检索）
    metadata: dict = field(default_factory=dict)
    doc_meta: Optional[DocumentMeta] = None  # 来源文档元数据


class ChapterSplitter:
    """章节分割器：第一层切分 - 按章节"""

    def __init__(self, toc_extractor: Optional[TOCExtractor] = None):
        self.toc_extractor = toc_extractor or TOCExtractor()

    def split(self, doc: ParsedDocument) -> list[Chapter]:
        """将文档按章节切分"""
        if doc.toc:
            logger.info(f"使用 PDF 内置目录切分章节，共 {len(doc.toc)} 条")
            return self.toc_extractor.extract_from_pdf_toc(doc)
        else:
            logger.info("无内置目录，使用标题模式检测章节")
            return self.toc_extractor.extract_from_text(doc.full_text, doc)


class ParentChildSplitter:
    """
    父子块分割器：第二层切分

    策略：
    1. 先将每个章节的内容切分为子块（1000 字符，200 重叠）
    2. 子块在标点边界处自然断句，不截断语义
    3. 每 N 个子块生成一个父块用于上下文扩展
    """

    def __init__(
        self,
        child_chunk_size: int = 1000,
        child_chunk_overlap: int = 200,
        child_chunks_per_parent: int = 3,
        separator: str = "\n",
    ):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.child_chunks_per_parent = child_chunks_per_parent
        self.separator = separator

    def split_chapter(
        self,
        chapter: Chapter,
        doc_meta: Optional[DocumentMeta] = None,
    ) -> tuple[list[ChildChunk], list[ParentChunk]]:
        """
        对单个章节进行父子块切分

        Returns:
            (child_chunks, parent_chunks): 子块列表和父块列表
        """
        chapter_id = str(uuid4())
        chapter.chapter_id = chapter_id  # 回写到 Chapter 对象，保持 ID 一致

        # Step 1: 生成子块
        child_chunks = self._generate_child_chunks(chapter, chapter_id, doc_meta)

        # Step 2: 生成父块（每 N 个子块合并为一个父块）
        parent_chunks = self._generate_parent_chunks(child_chunks, doc_meta)

        logger.info(
            f"章节 [{chapter.title}]: 生成 {len(child_chunks)} 个子块, "
            f"{len(parent_chunks)} 个父块"
        )
        return child_chunks, parent_chunks

    def split_chapters(
        self,
        chapters: list[Chapter],
        doc_meta: Optional[DocumentMeta] = None,
    ) -> tuple[list[ChildChunk], list[ParentChunk]]:
        """
        对所有章节进行父子块切分（方案A增强：父章节自动汇总子章节目录）

        核心改进：
            当检测到 level=N 的章节有 3+ 个 level=N+1 的子章节时，
            在父章节内容前插入子章节目录，确保检索时能拿到完整信息列表。
            例："第二章 商品分类详解" 自动汇总 2.1-2.8 8个分类列表
        """
        all_children = []

        # ===== 方案A层面1：构建父子章节关系，父章节汇总子章节目录 =====
        self._enrich_parent_chapters_with_children_summary(chapters)

        # Step 1: 按章节生成所有子块
        for chapter in chapters:
            children = self._generate_child_chunks_for_chapter(chapter, doc_meta)
            all_children.extend(children)

        # Step 2: 跨章节生成父块（滑动窗口，覆盖所有子块）
        all_parents = self._generate_parent_chunks(all_children, doc_meta)

        logger.info(
            f"全文档切分完成: {len(all_children)} 个子块, {len(all_parents)} 个父块"
        )
        return all_children, all_parents

    # ===== 方案A层面1：父章节汇总子章节目录 =====

    def _enrich_parent_chapters_with_children_summary(self, chapters: list[Chapter]):
        """
        检测父子章节关系，在父章节内容中插入子章节目录（修复：分类信息丢失）

        逻辑：
            1. 按 chapter_path 识别父子关系（A.path 是 B.path 的前缀，且 A.level < B.level）
            2. 父章节有 >= 3 个子章节时，在父章节内容前插入汇总目录
            3. 目录格式：
                【本章包含 {count} 个子章节】
                {child1_title} - {child1_first_line_content}
                {child2_title} - {child2_first_line_content}
                ...
        """
        if len(chapters) < 3:
            return

        # Step 1: 构建父子关系映射
        # parent_idx -> [child_idx1, child_idx2, ...]
        # 关键：选择层级最接近的父章节（level 最大但仍小于当前章节）
        parent_children_map: dict[int, list[int]] = {}

        for i, chapter in enumerate(chapters):
            # 寻找此章节的父章节：在所有匹配前缀的候选中选择 level 最大的（最近的父章节）
            if not chapter.chapter_path:
                continue

            best_parent_idx = -1
            best_parent_level = -1

            for j, potential_parent in enumerate(chapters):
                if j == i:
                    continue
                if potential_parent.level >= chapter.level:
                    continue
                if not potential_parent.chapter_path:
                    continue
                # 判断前缀关系："父 · 子"
                parent_path = potential_parent.chapter_path
                if chapter.chapter_path.startswith(parent_path + " · "):
                    # 选择 level 更大的（更接近子章节层级）作为真正的父章节
                    if potential_parent.level > best_parent_level:
                        best_parent_idx = j
                        best_parent_level = potential_parent.level

            if best_parent_idx >= 0:
                if best_parent_idx not in parent_children_map:
                    parent_children_map[best_parent_idx] = []
                parent_children_map[best_parent_idx].append(i)

        # Step 2: 对有 3+ 个子章节的父章节，插入子章节目录
        enriched_count = 0
        for parent_idx, children_indices in parent_children_map.items():
            if len(children_indices) < _STRUCTURED_MIN_ITEMS:
                continue

            parent_chapter = chapters[parent_idx]
            child_chapters = [chapters[idx] for idx in children_indices]

            # 构建子章节目录（标题 + 第一行核心内容）
            summary_lines = []
            summary_lines.append(f"【本章包含以下{len(child_chapters)}个子章节】")

            for child in child_chapters:
                # 提取子章节标题（从 chapter_path 中截取最后一部分，或者用 child.title）
                child_title = child.title
                # 提取子章节第一行核心内容（前150字符）
                child_first_lines = self._extract_summary_line(child.content, 150)
                if child_first_lines:
                    summary_lines.append(f"{child_title} - {child_first_lines}")
                else:
                    summary_lines.append(child_title)

            # 插入到父章节内容的开头（在原内容之前）
            summary_text = "\n".join(summary_lines)
            original_content = parent_chapter.content.strip() if parent_chapter.content else ""

            if original_content:
                parent_chapter.content = f"{summary_text}\n\n{original_content}"
            else:
                parent_chapter.content = summary_text

            enriched_count += 1
            logger.info(
                f"父章节 [{parent_chapter.title}] 汇总了 "
                f"{len(child_chapters)} 个子章节目录"
            )

        if enriched_count > 0:
            logger.info(f"方案A层面1完成：共 {enriched_count} 个父章节获得子章节目录汇总")

    def _extract_summary_line(self, content: str, max_len: int = 150) -> str:
        """从章节内容中提取第一行核心信息（用于子章节目录摘要）"""
        if not content or not content.strip():
            return ""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if not lines:
            return ""
        first_line = lines[0]
        if len(first_line) > max_len:
            first_line = first_line[:max_len] + "..."
        return first_line

    def generate_summary_chunks(
        self,
        chapters: list[Chapter],
        doc_meta: Optional[DocumentMeta] = None,
        min_children: int = 3,
    ) -> list[ChildChunk]:
        """
        为包含多子章节/多商品的父章节生成「汇总型子块」，用于概括性查询。

        核心场景：
        1. 用户查询「平台有多少个商品分类？分别是什么？」
           → 父章节有明确的子章节结构（2.1 手机数码类、2.2 电脑办公类...）
           → 方案 A：基于 chapter_path 父子关系汇总

        2. 用户查询「电脑办公类有哪些商品？」
           → 「第四章 商品详细信息 - 电脑办公类」下是 #### 商品 001...
           → TOCExtractor 跳过 L4 标题，商品信息作为正文内容
           → 方案 B：直接扫描章节内容中的「商品 NNN」模式汇总

        返回的汇总块会被加入 child_chunks 一起向量化入库。

        Args:
            chapters: 所有章节
            doc_meta: 文档元数据
            min_children: 至少有 N 个子章节/商品才生成汇总（默认 3）

        Returns:
            list[ChildChunk]: 汇总型子块列表
        """
        summary_chunks: list[ChildChunk] = []

        if len(chapters) < 3:
            return summary_chunks

        # ============================================================
        # 方案 A：基于 chapter_path 父子关系汇总（用于分类章节）
        # ============================================================
        # 构建父子关系映射（同 _enrich_parent_chapters_with_children_summary）
        parent_children_map: dict[int, list[int]] = {}

        for i, chapter in enumerate(chapters):
            if not chapter.chapter_path:
                continue

            best_parent_idx = -1
            best_parent_level = -1

            for j, potential_parent in enumerate(chapters):
                if j == i:
                    continue
                if potential_parent.level >= chapter.level:
                    continue
                if not potential_parent.chapter_path:
                    continue
                parent_path = potential_parent.chapter_path
                if chapter.chapter_path.startswith(parent_path + " · "):
                    if potential_parent.level > best_parent_level:
                        best_parent_idx = j
                        best_parent_level = potential_parent.level

            if best_parent_idx >= 0:
                if best_parent_idx not in parent_children_map:
                    parent_children_map[best_parent_idx] = []
                parent_children_map[best_parent_idx].append(i)

        # 方案 A：为每个满足条件的父章节生成汇总块
        for parent_idx, children_indices in parent_children_map.items():
            if len(children_indices) < min_children:
                continue

            parent_chapter = chapters[parent_idx]
            child_chapters = [chapters[idx] for idx in children_indices]

            parent_intro = parent_chapter.content.strip() if parent_chapter.content else ""
            if len(parent_intro) > 500:
                parent_intro = parent_intro[:500] + "..."

            child_titles = []
            for cc in child_chapters:
                if cc.chapter_path and " · " in cc.chapter_path:
                    title_clean = cc.chapter_path.split(" · ")[-1]
                else:
                    title_clean = cc.title
                child_titles.append(f"{len(child_titles) + 1}. {title_clean}")

            if parent_chapter.chapter_path:
                path_parts = parent_chapter.chapter_path.split(" · ")
                if len(path_parts) > 2:
                    effective_title = " · ".join(path_parts[-2:])
                else:
                    effective_title = parent_chapter.chapter_path
            else:
                effective_title = parent_chapter.title

            summary_content_lines = [
                f"【汇总】{effective_title}",
                "",
            ]
            if parent_intro:
                summary_content_lines.append(f"章节简介：{parent_intro}")
                summary_content_lines.append("")

            summary_content_lines.append(f"本章共包含 {len(child_titles)} 个分类/子章节：")
            summary_content_lines.extend(child_titles)
            summary_content_lines.append("")

            summary_content_lines.append("各分类核心信息：")
            for cc in child_chapters:
                first_line = self._extract_summary_line(cc.content, max_len=120)
                if first_line:
                    if cc.chapter_path and " · " in cc.chapter_path:
                        cat_title = cc.chapter_path.split(" · ")[-1]
                    else:
                        cat_title = cc.title
                    summary_content_lines.append(f"- {cat_title}：{first_line}")

            summary_content = "\n".join(summary_content_lines)

            chapter_id = parent_chapter.chapter_id or str(uuid4())
            summary_chunk = ChildChunk(
                chunk_id=str(uuid4()),
                chapter_id=chapter_id,
                chapter_title=effective_title + " · 汇总",
                content=summary_content,
                chunk_index=-1,
                metadata={
                    "is_summary": True,
                    "level": parent_chapter.level,
                    "summary_of": parent_chapter.title,
                    "child_count": len(child_titles),
                    "summary_type": "category",
                },
                doc_meta=doc_meta,
            )
            summary_chunks.append(summary_chunk)

        # ============================================================
        # 方案 B：扫描章节内容中的商品模式（用于商品详细信息章节）
        #
        # 场景：「第四章 商品详细信息 - 电脑办公类」章节内容中包含
        #       「#### 商品 001: MacBook Pro 16寸」「#### 商品 002: ...」
        #       这些 L4 标题被 TOCExtractor 作为正文保留，没有形成子章节
        #       因此直接扫描章节正文汇总商品列表
        # ============================================================

        # 商品模式：匹配 "商品 NNN" 后跟商品名称，支持多种格式
        # 简化为一个主模式，覆盖：#### 商品 001: MacBook Pro、【商品 004】MacBook Air、商品 001 MacBook Pro
        product_main_pattern = r'商品\s*(\d+)\s*[:：\s]*([^\n\r]+)'

        for chapter in chapters:
            if not chapter.content or len(chapter.content) < 100:
                continue

            # 跳过已有方案A汇总的章节（避免重复）
            if chapter.title in [c.metadata.get("summary_of", "") for c in summary_chunks]:
                continue

            content_to_scan = chapter.content

            # 用主模式匹配，收集唯一的商品（按编号去重）
            found_products: dict[str, str] = {}  # num -> name

            for match in re.finditer(product_main_pattern, content_to_scan):
                prod_num = match.group(1).strip()
                prod_name = match.group(2).strip()
                prod_name = prod_name.strip('：: -·【】')
                # 过滤掉属性行（商品编号、商品品牌等不是商品名称
                skip_keywords = ['商品编号', '商品品牌', '商品售价', '商品简介', '详细规格', '补充说明', '售后保障', '配送方式']
                if not prod_name or prod_num in found_products:
                    continue
                if any(kw in prod_name for kw in skip_keywords):
                    continue
                if len(prod_name) > 100:
                    continue
                found_products[prod_num] = prod_name

            if len(found_products) < min_children:
                continue

            # 按编号排序（001, 002, 003...）
            sorted_nums = sorted(found_products.keys())

            # 构建 effective_title
            if chapter.chapter_path:
                path_parts = chapter.chapter_path.split(" · ")
                if len(path_parts) > 2:
                    effective_title = " · ".join(path_parts[-2:])
                else:
                    effective_title = chapter.chapter_path
            else:
                effective_title = chapter.title

            # 汇总块正文：专门优化给「XX类有哪些商品」查询
            summary_lines = [
                f"【商品汇总】{effective_title}",
                "",
                f"章节：{chapter.title}",
                f"本章节共收录 {len(sorted_nums)} 款商品：",
                "",
            ]

            # 商品名称列表
            for idx, prod_num in enumerate(sorted_nums):
                prod_name = found_products[prod_num]
                summary_lines.append(f"{idx + 1}. 商品 {prod_num} - {prod_name}")

            summary_lines.append("")
            summary_lines.append("完整商品清单：")
            for prod_num in sorted_nums:
                summary_lines.append(f"- 商品 {prod_num}: {found_products[prod_num]}")

            # 分类信息（如果能从标题提取）
            category_hint = ""
            if "电脑办公类" in chapter.title or "电脑办公类" in effective_title:
                category_hint = "电脑办公类商品"
            elif "手机数码类" in chapter.title or "手机数码类" in effective_title:
                category_hint = "手机数码类商品"
            elif "家用电器类" in chapter.title or "家用电器类" in effective_title:
                category_hint = "家用电器类商品"

            if category_hint:
                summary_lines.insert(2, f"分类：{category_hint}")
                summary_lines.insert(2, f"商品类型：{category_hint}")
                summary_lines.insert(2, f"常见查询：本分类包括哪些商品？{category_hint}有哪些？")

            summary_content = "\n".join(summary_lines)

            # 生成汇总型子块
            chapter_id = chapter.chapter_id or str(uuid4())
            summary_chunk = ChildChunk(
                chunk_id=str(uuid4()),
                chapter_id=chapter_id,
                chapter_title=effective_title + " · 商品汇总",
                content=summary_content,
                chunk_index=-1,
                metadata={
                    "is_summary": True,
                    "level": chapter.level,
                    "summary_of": chapter.title,
                    "child_count": len(sorted_nums),
                    "summary_type": "products",
                    "category": category_hint or effective_title,
                },
                doc_meta=doc_meta,
            )
            summary_chunks.append(summary_chunk)

        if summary_chunks:
            type_counts: dict[str, int] = {}
            for sc in summary_chunks:
                t = sc.metadata.get("summary_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            type_desc = ", ".join([f"{t}={c}" for t, c in type_counts.items()])
            logger.info(
                f"汇总型子块生成完成：共 {len(summary_chunks)} 个 "
                f"（{type_desc}）"
            )

        return summary_chunks

    # ===== 方案A：结构化文档保护方法 =====

    def _is_list_item_line(self, line: str) -> bool:
        """检测一行是否为列表项开头（1. / 一、/ 【1】 / ① 等）"""
        line = line.strip()
        if not line:
            return False
        return any(p.match(line) for p in _LIST_ITEM_PATTERNS)

    def _detect_structured_content(self, content: str) -> Optional[dict]:
        """
        检测章节内容是否为结构化列表（方案A核心）

        返回:
            - 结构化文档: dict(type='numbered_list', items=[...], intro_text='...')
            - 非结构化文档: None

        检测逻辑:
            1. 逐行扫描，标记列表项起始行
            2. 若检测到 >= _STRUCTURED_MIN_ITEMS 个连续列表项，视为结构化
            3. 支持列表项多行（从一行开始，直到下一个列表项之前都属于当前项）
        """
        if not content or not content.strip():
            return None

        lines = content.split('\n')

        # Step 1: 标记所有列表项起始行的索引
        list_item_line_indices = []
        for i, line in enumerate(lines):
            if self._is_list_item_line(line):
                list_item_line_indices.append(i)

        # Step 2: 检测阈值
        if len(list_item_line_indices) < _STRUCTURED_MIN_ITEMS:
            return None

        # Step 3: 解析每个列表项的完整内容（支持多行）
        items = []
        for idx, item_start_line in enumerate(list_item_line_indices):
            # 下一个列表项的起始行 = 当前项的结束边界
            if idx + 1 < len(list_item_line_indices):
                next_item_line = list_item_line_indices[idx + 1]
            else:
                next_item_line = len(lines)

            # 收集当前列表项的所有行
            item_lines = []
            for line_idx in range(item_start_line, next_item_line):
                line_text = lines[line_idx].strip()
                if line_text:
                    item_lines.append(line_text)

            if item_lines:
                items.append('\n'.join(item_lines))

        if len(items) < _STRUCTURED_MIN_ITEMS:
            return None

        # Step 4: 提取引言内容（第一个列表项之前的文本）
        first_item_line = list_item_line_indices[0]
        intro_lines = []
        for line_idx in range(first_item_line):
            line_text = lines[line_idx].strip()
            if line_text:
                intro_lines.append(line_text)
        intro_text = '\n'.join(intro_lines) if intro_lines else ''

        return {
            'type': 'numbered_list',
            'item_count': len(items),
            'items': items,
            'intro_text': intro_text,
            'line_indices': list_item_line_indices,
        }

    def _split_structured_list(
        self,
        structured_info: dict,
        chapter: Chapter,
        chapter_id: str,
        doc_meta: Optional[DocumentMeta] = None,
    ) -> list[ChildChunk]:
        """
        对结构化列表进行智能切分（方案A核心：绝不切断单个列表项）

        策略:
            1. 按 _STRUCTURED_ITEMS_PER_CHUNK 个列表项为一组生成子块
            2. 每个子块前置 [章节: XXX] [结构化列表: X-Y/Z] 上下文
            3. 第一个子块额外包含引言（如果有）
            4. 单个超长列表项（>500字符）独立成一个子块
        """
        items = structured_info['items']
        intro_text = structured_info['intro_text']
        item_count = len(items)

        # 简化章节标题：只保留最后 2 个层级，避免过长路径干扰检索
        if chapter.chapter_path:
            path_parts = chapter.chapter_path.split(' · ')
            if len(path_parts) > 2:
                effective_title = ' · '.join(path_parts[-2:])
            else:
                effective_title = chapter.chapter_path
        else:
            effective_title = chapter.title
        items_per_chunk = _STRUCTURED_ITEMS_PER_CHUNK

        child_chunks = []
        chunk_index = 0
        current_batch_items = []
        current_batch_length = 0

        for item_idx, item in enumerate(items):
            item_length = len(item)

            # 特殊处理：单个超长列表项（>500字符）独立成块
            if item_length > 500 and not current_batch_items:
                # 这个超长项单独成块
                chunk_content = self._build_structured_chunk_content(
                    effective_title,
                    item_idx,         # 0-based
                    item_idx,         # 同 start=end = 单个项
                    item_count,
                    [item],
                    intro_text if item_idx == 0 else '',
                )
                child_chunks.append(ChildChunk(
                    chunk_id=str(uuid4()),
                    chapter_id=chapter_id,
                    chapter_title=effective_title,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'level': chapter.level,
                        'is_structured': True,
                        'structured_type': structured_info['type'],
                        'item_range': f"{item_idx+1}-{item_idx+1}",
                        'total_items': item_count,
                    },
                    doc_meta=doc_meta,
                ))
                chunk_index += 1
                continue

            # 常规逻辑：将当前项加入批次
            if current_batch_items and (
                len(current_batch_items) >= items_per_chunk
                or current_batch_length + item_length > self.child_chunk_size
            ):
                # 当前批次已满 → 生成子块
                chunk_content = self._build_structured_chunk_content(
                    effective_title,
                    item_idx - len(current_batch_items),
                    item_idx - 1,
                    item_count,
                    current_batch_items,
                    intro_text if (item_idx - len(current_batch_items)) == 0 else '',
                )
                child_chunks.append(ChildChunk(
                    chunk_id=str(uuid4()),
                    chapter_id=chapter_id,
                    chapter_title=effective_title,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        'level': chapter.level,
                        'is_structured': True,
                        'structured_type': structured_info['type'],
                        'item_range': f"{item_idx - len(current_batch_items) + 1}-{item_idx}",
                        'total_items': item_count,
                    },
                    doc_meta=doc_meta,
                ))
                chunk_index += 1
                current_batch_items = []
                current_batch_length = 0

            current_batch_items.append(item)
            current_batch_length += item_length

        # 处理最后一个批次
        if current_batch_items:
            start_idx = item_count - len(current_batch_items)
            chunk_content = self._build_structured_chunk_content(
                effective_title,
                start_idx,
                item_count - 1,
                item_count,
                current_batch_items,
                intro_text if start_idx == 0 else '',
            )
            child_chunks.append(ChildChunk(
                chunk_id=str(uuid4()),
                chapter_id=chapter_id,
                chapter_title=effective_title,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata={
                    'level': chapter.level,
                    'is_structured': True,
                    'structured_type': structured_info['type'],
                    'item_range': f"{start_idx + 1}-{item_count}",
                    'total_items': item_count,
                },
                doc_meta=doc_meta,
            ))
            chunk_index += 1

        logger.info(
            f"[结构化保护] 检测到 {item_count} 个列表项, "
            f"生成 {len(child_chunks)} 个子块 (每块约 {items_per_chunk} 项)"
        )
        return child_chunks

    def _build_structured_chunk_content(
        self,
        chapter_title: str,
        start_item_idx: int,
        end_item_idx: int,
        total_items: int,
        items: list[str],
        intro_text: str,
    ) -> str:
        """构建结构化子块的内容（章节上下文 + 引言 + 列表项）"""
        content_parts = []

        # 章节上下文前缀（用于向量检索时携带章节信息）
        content_parts.append(f"[章节: {chapter_title}]")
        content_parts.append(f"[结构化列表: 第{start_item_idx + 1}-{end_item_idx + 1}项 / 共{total_items}项]")

        # 引言（仅第一个子块包含）
        if intro_text:
            content_parts.append("")
            content_parts.append(intro_text)

        # 列表项
        content_parts.append("")
        content_parts.append('\n'.join(items))

        return '\n'.join(content_parts).strip()

    def _generate_child_chunks_for_chapter(
        self,
        chapter: Chapter,
        doc_meta: Optional[DocumentMeta] = None,
    ) -> list[ChildChunk]:
        """为单个章节生成子块（不生成父块）"""
        chapter_id = str(uuid4())
        chapter.chapter_id = chapter_id
        return self._generate_child_chunks(chapter, chapter_id, doc_meta)

    def _generate_child_chunks(
        self,
        chapter: Chapter,
        chapter_id: str,
        doc_meta: Optional[DocumentMeta] = None,
    ) -> list[ChildChunk]:
        """
        生成子块（方案A增强：先检测结构化，再决定切分策略）

        策略:
            1. 尝试结构化列表检测（1. / 一、/ 【1】 等格式）
            2. 如果是结构化列表（>=3项）：按列表项分组切分，绝不切断单个项
            3. 如果非结构化：原有逻辑（段落+标点+字符数）
            4. 所有子块前置章节上下文前缀
        """
        content = chapter.content
        if not content or not content.strip():
            return []

        # Step 1（方案A核心）：尝试结构化列表检测
        structured_info = self._detect_structured_content(content)
        if structured_info:
            # 结构化文档：按列表项分组切分（内部已处理章节上下文前缀）
            return self._split_structured_list(structured_info, chapter, chapter_id, doc_meta)

        # Step 2（原有逻辑）：非结构化文档 → 按段落+标点+字符数切分
        # 简化章节标题：只保留最后 2 个层级，避免过长路径干扰检索
        if chapter.chapter_path:
            path_parts = chapter.chapter_path.split(' · ')
            if len(path_parts) > 2:
                effective_title = ' · '.join(path_parts[-2:])
            else:
                effective_title = chapter.chapter_path
        else:
            effective_title = chapter.title

        paragraphs = content.split('\n\n')
        chunks = []
        chunk_index = 0
        MIN_PARA_CHUNK = 100
        carry_over = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 段落内按行 + 标点切分为句子
            sentences = []
            lines = para.split(self.separator)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                buf = ""
                for ch in line:
                    buf += ch
                    if ch in _SENTENCE_BOUNDARY:
                        sentences.append(buf.strip())
                        buf = ""
                if buf.strip():
                    sentences.append(buf.strip())

            sentences = self._merge_short_sentences(sentences)
            if not sentences:
                continue

            current_chunk = carry_over
            carry_over = ""
            for sent in sentences:
                if current_chunk and len(current_chunk) + len(sent) > self.child_chunk_size:
                    # 方案A：为非结构化子块也添加章节上下文前缀
                    prefixed_content = f"[章节: {effective_title}]\n\n{current_chunk.strip()}"
                    chunks.append(ChildChunk(
                        chunk_id=str(uuid4()),
                        chapter_id=chapter_id,
                        chapter_title=effective_title,
                        content=prefixed_content,
                        chunk_index=chunk_index,
                        metadata={"level": chapter.level, "is_structured": False},
                        doc_meta=doc_meta,
                    ))
                    chunk_index += 1
                    current_chunk = sent
                else:
                    if current_chunk:
                        current_chunk += " " + sent
                    else:
                        current_chunk = sent

            # 段落结束时：如果当前块 < MIN_PARA_CHUNK，留给下一段落合并
            if current_chunk.strip():
                if len(current_chunk) < MIN_PARA_CHUNK:
                    carry_over = current_chunk
                else:
                    prefixed_content = f"[章节: {effective_title}]\n\n{current_chunk.strip()}"
                    chunks.append(ChildChunk(
                        chunk_id=str(uuid4()),
                        chapter_id=chapter_id,
                        chapter_title=effective_title,
                        content=prefixed_content,
                        chunk_index=chunk_index,
                        metadata={"level": chapter.level, "is_structured": False},
                        doc_meta=doc_meta,
                    ))
                    chunk_index += 1

        # 最后剩余的 carry_over
        if carry_over.strip():
            prefixed_content = f"[章节: {effective_title}]\n\n{carry_over.strip()}"
            chunks.append(ChildChunk(
                chunk_id=str(uuid4()),
                chapter_id=chapter_id,
                chapter_title=effective_title,
                content=prefixed_content,
                chunk_index=chunk_index,
                metadata={"level": chapter.level, "is_structured": False},
                doc_meta=doc_meta,
            ))

        return chunks

    def _merge_short_sentences(self, sentences: list[str]) -> list[str]:
        """合并过短的句子（<20 字符），避免碎片化"""
        MIN_LEN = 20
        if not sentences:
            return sentences

        merged = []
        buffer = ""
        for s in sentences:
            if buffer and len(buffer) < MIN_LEN:
                buffer += " " + s
                if len(buffer) >= MIN_LEN:
                    merged.append(buffer)
                    buffer = ""
            elif not buffer and len(s) < MIN_LEN:
                buffer = s
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(s)

        if buffer:
            if merged and len(buffer) < MIN_LEN:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        return merged

    def _get_overlap_text(self, text: str) -> str:
        """获取文本尾部用于重叠的部分"""
        overlap_len = min(self.child_chunk_overlap, len(text))
        return text[-overlap_len:] + " " if overlap_len > 0 else ""

    def _generate_parent_chunks(
        self,
        child_chunks: list[ChildChunk],
        doc_meta: Optional[DocumentMeta] = None,
    ) -> list[ParentChunk]:
        """
        生成父块：跨所有子块的滑动窗口，每 N 个子块生成一个父块。
        父块用于上下文扩展（检索时获取相邻子块）。

        滑动窗口步长 = child_chunks_per_parent // 2（有重叠）
        这样确保每个子块至少出现在 2 个父块中，上下文更完整。
        """
        if not child_chunks:
            return []

        parents = []
        window_size = self.child_chunks_per_parent
        stride = max(1, window_size // 2)  # 滑动步长（有重叠）

        for i in range(0, len(child_chunks), stride):
            batch = child_chunks[i:i + window_size]

            if not batch:
                continue

            # 拼接子块内容（不截断）
            combined_content = '\n\n'.join(ch.content for ch in batch)
            child_ids = [ch.chunk_id for ch in batch]

            # 章节标题：取第一个子块的章节标题
            chapter_title = batch[0].chapter_title if batch else ""

            parent = ParentChunk(
                parent_id=str(uuid4()),
                chapter_id=batch[0].chapter_id if batch else "",
                chapter_title=chapter_title,
                summary=self._generate_summary(batch),
                child_ids=child_ids,
                content_snippet=combined_content,  # 不截断，保留完整上下文
                metadata={
                    "child_count": len(batch),
                    "child_indices": [ch.chunk_index for ch in batch],
                    "chapter_titles": list(set(ch.chapter_title for ch in batch)),
                    "level": batch[0].metadata.get("level", 1) if batch else 1,
                },
                doc_meta=doc_meta,
            )

            # 回写 parent_id 到每个子块
            for ch in batch:
                ch.parent_id = parent.parent_id

            parents.append(parent)

        return parents

    def _generate_summary(self, child_chunks: list[ChildChunk]) -> str:
        """
        生成父块摘要（本地提取式摘要，不依赖 LLM）。

        策略：取首尾关键句 + 中间关键词，保持语义不变。
        """
        if not child_chunks:
            return ""

        # 合并所有子块内容
        full_text = ' '.join(ch.content for ch in child_chunks)

        # 提取式摘要策略：
        # 1. 取第一段的前 100 字作为起始
        # 2. 取最后一段的后 100 字作为结尾
        # 3. 中间取关键句

        sentences = full_text.replace('\n', '。').split('。')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 3:
            return full_text[:300]

        # 取首尾
        summary_parts = [
            sentences[0][:100],
            sentences[-1][:100],
        ]

        # 中间取一句（取较长的句子，通常信息量更大）
        mid_sentence = sorted(sentences[1:-1], key=len, reverse=True)[0] if len(sentences) > 2 else ""
        if mid_sentence and len(mid_sentence) > 20:
            summary_parts.insert(1, mid_sentence[:100])

        return '。'.join(summary_parts) + '。'