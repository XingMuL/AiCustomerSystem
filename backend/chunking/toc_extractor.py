"""
目录提取器：从 PDF 内置目录或文本中检测章节结构。
支持有目录的 PDF 和基于标题模式的无目录文本。
"""

import re
from typing import Optional
from dataclasses import dataclass, field

from backend.preprocessing.parser import ParsedDocument


@dataclass
class Chapter:
    """章节结构"""
    level: int           # 层级（1=一级标题, 2=二级标题...）
    title: str           # 章节标题
    start_page: int      # 起始页
    end_page: int = 0    # 结束页
    content: str = ""    # 章节内容
    chapter_id: str = "" # 章节唯一标识（由 splitter 分配）
    chapter_path: str = ""  # 章节路径（如"第一部分 2025年工作回顾 · 党建引领质效升级"）
    children: list["Chapter"] = field(default_factory=list)  # 子章节


class TOCExtractor:
    """目录提取器 - 多层多级的第一层：抽象每个章节"""

    # 垃圾标题列表：PDF 转换过程中产生的无意义的元数据标题
    # 这些标题应该被当作正文内容处理，不应该创建独立章节
    NOISE_TITLE_PATTERNS = [
        '文档标题',
        'document title',
        '目录',
        'contents',
        'content',
        '章节标题',
        'fTaoBao 电商平台服务指南',
        'ftaoabao',
    ]

    # 中文标题匹配模式
    CHAPTER_PATTERNS = [
        # 第X章 / 第X节
        re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部]\s*.+'),
        # X. / X.X / X.X.X  编号标题
        re.compile(r'^\d+(\.\d+)*\s+.+'),
        # 一、/ 二、 中文数字编号
        re.compile(r'^[一二三四五六七八九十]+[、，,]\s*.+'),
        # Markdown 标题
        re.compile(r'^#{1,6}\s+.+'),
        # 大写英文编号
        re.compile(r'^[A-Z][\s.]\s*.+'),
    ]

    # 默认章节标题（无目录时）
    DEFAULT_CHAPTER_TITLE = "正文"

    def extract_from_pdf_toc(self, doc: ParsedDocument) -> list[Chapter]:
        """从 PDF 内置目录提取章节结构"""
        if not doc.toc:
            logger.info("PDF 没有内置目录，将使用标题模式检测")
            return self._extract_by_pattern(doc)

        chapters = []
        level_stack = []  # [(level, title)]
        GROUP_PATTERNS = [r'^第[一二三四五六七八九十百\d]+部分', r'^Part\s*\d+']
        for item in doc.toc:
            title = item["title"].strip()
            level = item["level"]
            while level_stack and level_stack[-1][0] >= level:
                level_stack.pop()

            path_parts = [t for _, t in level_stack] + [title]
            chapter_path = ' · '.join(path_parts) if len(path_parts) > 1 else title
            level_stack.append((level, title))

            chapter = Chapter(
                level=level,
                title=title,
                start_page=item["page"],
                chapter_path=chapter_path,
            )
            chapters.append(chapter)

        # 填充章节内容
        self._fill_chapter_content(chapters, doc)
        return chapters

    def extract_from_text(self, text: str, doc: Optional[ParsedDocument] = None) -> list[Chapter]:
        """从文本内容检测章节结构（无内置目录时）"""
        return self._extract_by_pattern(doc) if doc else self._extract_by_text(text)

    def _extract_by_pattern(self, doc: ParsedDocument) -> list[Chapter]:
        """通过正则匹配标题模式检测章节"""
        chapters = []
        lines = doc.full_text.split('\n')

        # 维护章节层级栈（用于构建章节路径和层级归一化）
        # level_stack = [(level, title, is_group), ...]
        # is_group=True 表示"第一部分"这种大分组标题：普通L1标题不将其弹出，仅被下一个分组替换
        level_stack = []

        current_chapter = None
        chapter_content = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                chapter_content.append(line)
                continue

            matched_level = None
            matched_title = None

            # 检测 Markdown 标题模式（# ## ### 等）
            md_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if md_match:
                matched_level = len(md_match.group(1))  # #=1, ##=2, ###=3
                matched_title_text = md_match.group(2).strip()
                # === 商品信息单元特殊处理 ===
                # 商品信息用 `#### 商品 NNN: xxx`（level 4），但这是重要的语义单元
                # 即使 level > 3，也应该识别为独立章节
                is_product_item = bool(re.match(r'^商品\s*\d+\s*[:：\s]*', matched_title_text))
                if matched_level > 3 and not is_product_item:
                    # level > 3 的非商品标题，作为章节内容保留
                    chapter_content.append(line)
                    continue
                if is_product_item:
                    matched_level = 3  # 商品信息单元统一为 level 3
                matched_title = matched_title_text
            # 检测中文标题模式
            elif re.match(r'^第[一二三四五六七八九十百千\d]+章', line):
                matched_level = 1
                matched_title = line
            elif re.match(r'^第[一二三四五六七八九十百千\d]+节', line):
                matched_level = 2
                matched_title = line
            elif re.match(r'^\d+\.\d+\.\d+\s', line):
                matched_level = 3
                matched_title = line
            elif re.match(r'^\d+\.\d+\s', line):
                matched_level = 2
                matched_title = line
            elif re.match(r'^\d+\s', line) and len(line) < 60:
                matched_level = 1
                matched_title = line
            elif re.match(r'^[一二三四五六七八九十]+[、]\s', line):
                matched_level = 1
                matched_title = line

            if matched_title:
                # === 关键修复：垃圾标题过滤 ===
                # "文档标题"、"document title" 等是 PDF 转换过程中产生的无意义元数据
                # 这些标题会破坏正常章节结构（例如把"第四章 商品详细信息"劈成两半）
                # 处理方式：直接作为正文内容添加到当前章节，不创建新章节
                title_lower = matched_title.strip().lower()
                is_noise_title = any(
                    noise in title_lower
                    for noise in [n.lower() for n in self.NOISE_TITLE_PATTERNS]
                )
                # 更严格的精确匹配检查（防止误判"商品目录"等正常标题）
                is_noise_title = is_noise_title or any(
                    matched_title.strip() == noise
                    for noise in self.NOISE_TITLE_PATTERNS
                )
                if is_noise_title:
                    chapter_content.append(line)  # 作为正文保留
                    continue

                # 保存上一个章节
                if current_chapter:
                    current_chapter.content = '\n'.join(chapter_content).strip()
                    chapters.append(current_chapter)

                # 检查标题行是否拼接了正文内容（LLM 清洗可能将正文压缩到标题同行）
                title = matched_title
                first_content = ""
                sent_end = re.search(r'[。！？]', matched_title)
                if sent_end:
                    split_pos = sent_end.start()
                    title = matched_title[:split_pos].strip()
                    first_content = matched_title[split_pos + 1:].strip()

                # 检测是否为语义分组标题（如"第一部分 2025年工作回顾"、"第二部分..."）
                GROUP_PATTERNS = [
                    r'^第[一二三四五六七八九十百\d]+部分',
                    r'^Part\s*\d+',
                ]
                is_group = any(re.search(p, title) for p in GROUP_PATTERNS)

                # 更新层级栈（3元组: (level, title, is_group)）
                # - 语义分组标题(is_group=True): 直接替换所有之前的分组，清空非分组的
                # - 普通标题: 弹出所有 level >= 当前 level 且不是分组的标题
                if is_group:
                    # 新的分组标题 → 清空整个栈（它是新的顶层语义分组）
                    level_stack = []
                else:
                    # 普通标题 → 只弹出非分组的、层级>=当前的标题
                    while level_stack:
                        top_level, top_title, top_is_group = level_stack[-1]
                        if not top_is_group and top_level >= matched_level:
                            level_stack.pop()
                        else:
                            break

                # 构建章节路径
                # 只保留分组标题 + 当前标题（跳过文档总标题这种无意义的前缀）
                group_titles = [t for _, t, is_g in level_stack if is_g]
                if group_titles:
                    chapter_path = ' · '.join(group_titles + [title])
                else:
                    path_parts = [t for _, t, _ in level_stack] + [title]
                    if len(path_parts) > 1:
                        chapter_path = ' · '.join(path_parts)
                    else:
                        chapter_path = title

                # 压入当前层级
                level_stack.append((matched_level, title, is_group))

                current_chapter = Chapter(
                    level=matched_level,
                    title=title,
                    start_page=1,
                    content="",
                    chapter_path=chapter_path,
                )
                chapter_content = [first_content] if first_content else []
            else:
                chapter_content.append(line)

        # 保存最后一个章节
        if current_chapter:
            current_chapter.content = '\n'.join(chapter_content).strip()
            chapters.append(current_chapter)

        # 如果没有检测到任何章节，整个文档作为一个章节
        if not chapters:
            chapters.append(Chapter(
                level=1,
                title=self.DEFAULT_CHAPTER_TITLE,
                start_page=1,
                content=doc.full_text.strip(),
                chapter_path=self.DEFAULT_CHAPTER_TITLE,
            ))

        return chapters

    def _extract_by_text(self, text: str) -> list[Chapter]:
        """纯文本检测（无 ParsedDocument 时），支持章节路径"""
        lines = text.split('\n')
        chapters = []
        current_chapter = None
        chapter_content = []
        level_stack = []  # [(level, title, is_group)]
        GROUP_PATTERNS = [r'^第[一二三四五六七八九十百\d]+部分', r'^Part\s*\d+']

        for line in lines:
            line = line.strip()
            if not line:
                chapter_content.append(line)
                continue

            matched_level = None
            matched_title = None

            # 检测 Markdown 标题模式
            md_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if md_match:
                matched_level = len(md_match.group(1))
                # 同样修复：#### 及更深层级的标题视为正文
                if matched_level > 3:
                    chapter_content.append(line)
                    continue
                matched_title = md_match.group(2).strip()
            elif re.match(r'^第[一二三四五六七八九十百千\d]+章', line):
                matched_level = 1
                matched_title = line
            elif re.match(r'^\d+\.\d+\s', line):
                matched_level = 2
                matched_title = line
            elif re.match(r'^[一二三四五六七八九十]+[、]\s', line):
                matched_level = 1
                matched_title = line

            if matched_title:
                # === 关键修复：垃圾标题过滤 ===
                # "文档标题"、"document title" 等是 PDF 转换过程中产生的无意义元数据
                # 处理方式：直接作为正文内容添加到当前章节，不创建新章节
                title_lower = matched_title.strip().lower()
                is_noise_title = any(
                    noise in title_lower
                    for noise in [n.lower() for n in self.NOISE_TITLE_PATTERNS]
                )
                is_noise_title = is_noise_title or any(
                    matched_title.strip() == noise
                    for noise in self.NOISE_TITLE_PATTERNS
                )
                if is_noise_title:
                    chapter_content.append(line)  # 作为正文保留
                    continue

                if current_chapter:
                    current_chapter.content = '\n'.join(chapter_content)
                    chapters.append(current_chapter)

                # 检查标题行是否拼接了正文内容
                title = matched_title
                first_content = ""
                sent_end = re.search(r'[。！？]', matched_title)
                if sent_end:
                    split_pos = sent_end.start()
                    title = matched_title[:split_pos].strip()
                    first_content = matched_title[split_pos + 1:].strip()

                is_group = any(re.search(p, title) for p in GROUP_PATTERNS)

                # 更新层级栈（同 _extract_by_pattern）
                if is_group:
                    level_stack = []
                else:
                    while level_stack:
                        top_level, top_title, top_is_group = level_stack[-1]
                        if not top_is_group and top_level >= matched_level:
                            level_stack.pop()
                        else:
                            break

                # 构建章节路径
                group_titles = [t for _, t, is_g in level_stack if is_g]
                if group_titles:
                    chapter_path = ' · '.join(group_titles + [title])
                else:
                    path_parts = [t for _, t, _ in level_stack] + [title]
                    chapter_path = ' · '.join(path_parts) if len(path_parts) > 1 else title

                level_stack.append((matched_level, title, is_group))

                current_chapter = Chapter(
                    level=matched_level, title=title, start_page=1, content="",
                    chapter_path=chapter_path,
                )
                chapter_content = [first_content] if first_content else []
            else:
                chapter_content.append(line)

        if current_chapter:
            current_chapter.content = '\n'.join(chapter_content)
            chapters.append(current_chapter)

        if not chapters:
            chapters.append(Chapter(
                level=1, title=self.DEFAULT_CHAPTER_TITLE, start_page=1, content=text.strip(),
                chapter_path=self.DEFAULT_CHAPTER_TITLE,
            ))

        return chapters

    def _fill_chapter_content(self, chapters: list[Chapter], doc: ParsedDocument):
        """根据章节的起始页填充内容"""
        for i, chapter in enumerate(chapters):
            # 计算结束页
            if i + 1 < len(chapters):
                end_page = chapters[i + 1].start_page - 1
            else:
                end_page = len(doc.pages)

            chapter.end_page = end_page

            # 提取对应页码的内容
            content_parts = []
            for page in doc.pages:
                if chapter.start_page <= page.page_num <= end_page:
                    content_parts.append(page.text)
            chapter.content = '\n'.join(content_parts)


# 全局日志
from loguru import logger