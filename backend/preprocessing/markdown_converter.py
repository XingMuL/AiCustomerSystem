"""
Markdown 格式转换器

将所有解析后的文档内容统一转换为 Markdown 格式，
作为后续 Kimi 清洗流程的中间格式。

各格式转换策略：
  - PDF / DOCX / TXT → 提取文本后按段落转 MD：
      * 标题行 → # ## ###
      * 列表 → - 
      * 普通段落 → 原样保留
      * 页码/页脚等脏数据 → 由后续 Kimi 清洗处理
  - Excel (tab 分隔) → Markdown 表格
  - 图片 → 保留 OCR 文字 + 图片元信息
  - 已有 Markdown 文件 → 原样通过
"""

import re
from pathlib import Path

from loguru import logger

from backend.preprocessing.parser import ParsedDocument, ParsedPage


class MarkdownConverter:
    """
    将 ParsedDocument 转为统一的 Markdown 格式。
    转换后的结果回写到 doc.raw_text 和各 page.text 中。
    """

    def convert(self, doc: ParsedDocument) -> ParsedDocument:
        """
        将文档内容统一转为 Markdown 格式，直接修改 ParsedDocument 对象。

        Args:
            doc: 解析后的文档

        Returns:
            同一文档对象，text 字段已转为 Markdown
        """
        ext = Path(doc.file_path).suffix.lower()

        if ext in (".md",):
            # Markdown 文件直接通过
            logger.info(f"文档已是 Markdown 格式，跳过转换: {doc.file_path}")
            return doc

        logger.info(f"转换 Markdown: {doc.file_path}  [{ext}]")

        if ext in (".pdf",):
            doc = self._convert_pdf_to_md(doc)
        elif ext in (".docx",):
            doc = self._convert_docx_to_md(doc)
        elif ext in (".xlsx", ".xls"):
            doc = self._convert_excel_to_md(doc)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            doc = self._convert_image_to_md(doc)
        elif ext in (".txt",):
            doc = self._convert_txt_to_md(doc)
        else:
            logger.warning(f"未知格式，不做转换: {ext}")

        return doc

    # =====================================================================
    #  PDF → Markdown
    # =====================================================================

    def _convert_pdf_to_md(self, doc: ParsedDocument) -> ParsedDocument:
        """PDF 文本 → Markdown"""
        md_pages = []
        for page in doc.pages:
            md_text = self._raw_text_to_md(page.text)
            md_pages.append(md_text)

        doc.raw_text = "\n\n".join(md_pages)
        for i, page in enumerate(doc.pages):
            if i < len(md_pages):
                page.text = md_pages[i]
        return doc

    # =====================================================================
    #  DOCX → Markdown
    # =====================================================================

    def _convert_docx_to_md(self, doc: ParsedDocument) -> ParsedDocument:
        """DOCX 文本 → Markdown（保留标题层级）"""
        for page in doc.pages:
            md_lines = []
            for line in page.text.split("\n"):
                line = line.strip()
                if not line:
                    md_lines.append("")
                    continue

                # 检测是否为标题行
                # DOCX 解析时已丢失样式信息，通过启发式判断
                if self._looks_like_heading(line):
                    # 字数较少 + 无标点结尾 → 可能是标题
                    md_lines.append(f"## {line}")
                else:
                    md_lines.append(line)

            page.text = "\n".join(md_lines)

        doc.raw_text = "\n\n".join(p.text for p in doc.pages)
        return doc

    # =====================================================================
    #  Excel → Markdown Table
    # =====================================================================

    def _convert_excel_to_md(self, doc: ParsedDocument) -> ParsedDocument:
        """制表符分隔的 Excel 内容 → Markdown 表格"""
        md_pages = []

        for page in doc.pages:
            md_lines = [f"# {page.metadata.get('sheet_name', '工作表')}", ""]
            rows = page.text.split("\n")

            # 跳过工作表标题行
            data_start = 0
            for i, row in enumerate(rows):
                if row.startswith("【工作表"):
                    data_start = i
                    break

            data_rows = rows[data_start + 1:] if data_start < len(rows) else rows
            if not data_rows:
                md_pages.append("\n".join(md_lines))
                continue

            # 将 tab 分隔转换为 Markdown 表格
            # 第一行作为表头
            headers = [c.strip() for c in data_rows[0].split("\t")]
            if not headers:
                md_pages.append("\n".join(md_lines))
                continue

            # 构建表头行
            md_lines.append("| " + " | ".join(h if h else " " for h in headers) + " |")
            # 分隔行
            md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
            # 数据行
            for row in data_rows[1:]:
                cols = row.split("\t")
                # 补齐列数
                while len(cols) < len(headers):
                    cols.append("")
                md_lines.append("| " + " | ".join(c.strip() for c in cols[:len(headers)]) + " |")

            md_lines.append("")
            md_pages.append("\n".join(md_lines))

        for i, page in enumerate(doc.pages):
            if i < len(md_pages):
                page.text = md_pages[i]

        doc.raw_text = "\n\n".join(p.text for p in doc.pages)
        return doc

    # =====================================================================
    #  图片 → Markdown
    # =====================================================================

    def _convert_image_to_md(self, doc: ParsedDocument) -> ParsedDocument:
        """图片内容 → Markdown（OCR 文字 + 元信息）"""
        # 图片解析阶段已经生成 Markdown 格式的描述
        # 这里维持不变，由 Kimi 视觉模型进一步处理
        return doc

    # =====================================================================
    #  纯文本 → Markdown
    # =====================================================================

    def _convert_txt_to_md(self, doc: ParsedDocument) -> ParsedDocument:
        """纯文本 → Markdown（基本格式识别）"""
        for page in doc.pages:
            page.text = self._raw_text_to_md(page.text)
        doc.raw_text = "\n\n".join(p.text for p in doc.pages)
        return doc

    # =====================================================================
    #  通用工具：原始文本 → Markdown
    # =====================================================================

    # =====================================================================
    #  层级规范（统一规划，RAG 检索的基础）
    #  L1 (#)    : 文档主标题（唯一）
    #  L2 (##)   : 主要章节 —— "第N章"、"第N部分"、顶层服务说明
    #  L3 (###)  : 章节内子分类 —— "数字.数字 xxx"、"【数字.数字 xxx】"
    #  L4 (####) : 具体条目 —— "商品 NNN"、商品属性关键词
    # =====================================================================

    # 商品属性关键词（作为 L4 小标题）
    _KEYWORD_SUBHEADINGS = {
        '商品编号', '商品品牌', '商品售价', '商品简介',
        '详细规格', '补充说明', '售后保障', '保修政策',
        '配送方式', '包装清单', '商品编号前缀', '主要商品类型',
        '适用人群', '核心卖点', '发票说明', '售后服务',
        '退换说明', '安装服务',
    }
    _KEYWORD_HEADING_RE = re.compile(
        r'^(商品编号|商品品牌|商品售价|商品简介|详细规格|补充说明|'
        r'售后保障|保修政策|配送方式|包装清单|商品编号前缀|主要商品类型|'
        r'适用人群|核心卖点|发票说明|售后服务|退换说明|安装服务)\s*[:：]\s*(.*)$'
    )

    # 方括号标题模式（统一转为标准 Markdown）
    _BRACKET_NUM_SECTION = re.compile(r'^[【\[]\s*(\d+\.\d+)\s*(.+?)\s*[】\]]\s*$')   # 【2.7 运动户外类】
    _BRACKET_ITEM_ID = re.compile(r'^[【\[]\s*([^】\]]*?\d+[^】\]]*?)\s*[】\]]\s*(.+)$')  # 【商品 004】华为
    _BRACKET_PLAIN = re.compile(r'^[【\[]\s*([^】\]]+?)\s*[】\]]\s*$')                   # 【商品编号查询说明】

    # "第N章 / 第N部分" 模式 —— 强制 L2
    _CHINESE_CHAPTER_RE = re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部]\s*(.*)$')
    _CHINESE_PART_RE = re.compile(r'^第[一二三四五六七八九十百千\d]+部分\s*(.*)$')

    # 数字编号标题模式
    _NUM_N_M_RE = re.compile(r'^(\d+)\.(\d+)\s+(.+)$')        # "2.1 手机数码类" → ###
    _NUM_N_RE = re.compile(r'^(\d+)\s*[.、]\s+(.+)$')          # "1. xxx" → ##

    # Markdown 标题（已有）
    _MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')

    # =====================================================================
    #  语义启发式模式（用于没有明确章节标记的文档）
    # =====================================================================

    # 语义关键词：根据这些词推断层级
    # 高层级关键词：出现在文档开头，概括性强
    _HIGH_LEVEL_KEYWORDS = {
        '概述', '简介', '介绍', '背景', '目的', '范围',
        '前言', '引言', '总览', '概要', '绪论',
        '什么是', '为什么', '如何', '什么',
    }

    # 中层级关键词：具体主题分类
    _MID_LEVEL_KEYWORDS = {
        '说明', '指南', '流程', '步骤', '方法', '方式',
        '特点', '优势', '功能', '类型', '分类',
        '使用', '操作', '配置', '设置', '安装',
        '注意', '要求', '规则', '标准', '规范',
        '常见问题', 'FAQ', '问题', '解决方案',
        '服务', '政策', '条款', '条件',
    }

    # 低层级关键词：具体细节
    _LOW_LEVEL_KEYWORDS = {
        '补充', '备注', '附录', '参考', '示例', '案例',
        '注意事项', '提示', '说明补充', '详情',
    }

    # 中文编号标题（"一、"、"二、"等）
    _CHINESE_NUM_PATTERN = re.compile(r'^[一二三四五六七八九十百]+、\s*(.+)$')
    _CHINESE_NUM_BRACKET = re.compile(r'^[（(][一二三四五六七八九十百\d]+[）)]\s*(.+)$')

    # 英文编号标题（"Section 1."、"Part 1"等）
    _ENGLISH_SECTION_RE = re.compile(r'^(Section|Part|Chapter|SECTION|PART|CHAPTER)\s*\d+[.、:：]*\s*(.*)$', re.IGNORECASE)

    # 语义标题启发式：短文本且包含主题词汇
    # 如："产品介绍"、"使用说明"、"技术参数"、"售后服务"
    _SEMANTIC_TITLE_RE = re.compile(r'^[\u4e00-\u9fa5A-Za-z]{2,15}(介绍|说明|指南|服务|政策|功能|特点|参数|概述|要求|流程|步骤|方法)$')

    # 问答式标题
    _QA_PATTERN = re.compile(r'^(Q[.、:：]|问题[：:])\s*(.+)$')
    _ANS_PATTERN = re.compile(r'^(A[.、:：]|回答[：:]|解答[：:])\s*(.+)$')

    # 英文标题模式（全部大写的短词）
    _ENGLISH_TITLE_RE = re.compile(r'^[A-Z][A-Z\s]{3,30}$')

    # 页眉页脚 / 页码模式（待 LLM 深度清洗）
    PAGE_NUMBER_PATTERN = re.compile(r'^\s*-?\s*\d{1,4}\s*-?\s*$')
    HEADER_FOOTER_PATTERNS = [
        re.compile(r'^第\s*\d+\s*页.*$'),
        re.compile(r'^Page\s+\d+.*$', re.IGNORECASE),
        re.compile(r'^©.*$'),
        re.compile(r'^Confidential.*$', re.IGNORECASE),
    ]

    def _raw_text_to_md(self, text: str) -> str:
        """
        将原始文本转为统一层级规范的 Markdown。

        统一层级规则（核心）：
        L1 (#)   : 文档主标题（整个文档一个，由后续处理自动判定）
        L2 (##)  : 第N章 / 第N部分 / 顶层服务说明
        L3 (###) : N.M 子分类（如 "2.1 手机数码类"、"【2.7 运动户外类】"）
        L4 (####): 具体商品条目（"商品 001: xxx"）、商品属性关键词（核心卖点等）

        处理顺序：先识别最明显的标题模式，再处理正文
        """
        lines = text.split("\n")
        md_lines = []
        prev_blank = False

        for line in lines:
            stripped = line.strip()

            # 跳过空行（仅保留一个）
            if not stripped:
                if not prev_blank:
                    md_lines.append("")
                    prev_blank = True
                continue
            prev_blank = False

            # 跳过页码和页眉页脚
            if self.PAGE_NUMBER_PATTERN.match(stripped):
                continue
            if any(pat.match(stripped) for pat in self.HEADER_FOOTER_PATTERNS):
                continue

            # ============== 已有 Markdown 标题：保留但验证层级 ==============
            md_match = self._MD_HEADING_RE.match(stripped)
            if md_match:
                # 保留现有标题，但不做降级处理（留给 kimi_cleaner 统一后处理）
                md_lines.append(stripped)
                continue

            # ============== L2：第N章 / 第N部分 ==============
            # "第二章 商品分类详解" → ## 第二章 商品分类详解
            chap_match = self._CHINESE_CHAPTER_RE.match(stripped)
            if chap_match and len(stripped) < 80:
                rest = chap_match.group(1).strip()
                chapter_prefix = stripped[:len(stripped) - len(rest)].strip() if rest else stripped
                title = rest if rest else stripped
                md_lines.append(f"## {chapter_prefix} {rest}" if rest else f"## {stripped}")
                continue
            part_match = self._CHINESE_PART_RE.match(stripped)
            if part_match and len(stripped) < 80:
                md_lines.append(f"## {stripped}")
                continue

            # ============== L3：N.M 子分类标题 ==============
            num_m_match = self._NUM_N_M_RE.match(stripped)
            if num_m_match and len(stripped) < 80:
                md_lines.append(f"### {stripped}")
                continue

            # ============== 方括号标题智能提取 ==============
            # 【2.7 运动户外类】 → ### 2.7 运动户外类（L3）
            b_match_1 = self._BRACKET_NUM_SECTION.match(stripped)
            if b_match_1 and len(stripped) < 80:
                num_part, title_part = b_match_1.group(1), b_match_1.group(2).strip()
                md_lines.append(f"### {num_part} {title_part}")
                continue

            # 【商品 004】华为 Mate 60 Pro → #### 商品 004: 华为 Mate 60 Pro（L4）
            b_match_2 = self._BRACKET_ITEM_ID.match(stripped)
            if b_match_2 and len(stripped) < 150:
                bracket_content = ' '.join(b_match_2.group(1).strip().split())
                after_content = b_match_2.group(2).strip()
                title = f"{bracket_content}: {after_content}" if after_content else bracket_content
                md_lines.append(f"#### {title}")
                continue

            # 【商品编号查询说明】 → ### 商品编号查询说明（L3）
            b_match_3 = self._BRACKET_PLAIN.match(stripped)
            if b_match_3 and len(stripped) < 60 and '。' not in stripped and '！' not in stripped:
                md_lines.append(f"### {b_match_3.group(1).strip()}")
                continue

            # ============== L4：商品属性关键词 ==============
            # "核心卖点：品牌授权、正品保障" → #### 核心卖点 + 正文
            kw_match = self._KEYWORD_HEADING_RE.match(stripped)
            if kw_match:
                keyword = kw_match.group(1)
                content = kw_match.group(2).strip()
                md_lines.append(f"#### {keyword}")
                if content:
                    md_lines.append(content)
                continue

            # ============== N. xxx 格式处理（保守策略）==============
            # 原始文本中 "数字. xxx" 可能是章节标题，也可能是列表项
            # 只有当它：1) 没有冒号（冒号后跟内容通常是正文）
            #          2) 简短（<50字符）
            #          3) 不是 "N.M" 格式（已在上方处理）
            # 才转换为 ## 标题
            num_n_match = self._NUM_N_RE.match(stripped)
            if num_n_match and len(stripped) < 50 and '：' not in stripped and ':' not in stripped:
                # 排除 "N.M" 格式（已经在 L3 规则中处理）
                if not re.match(r'^\d+\.\d+\s+', stripped):
                    md_lines.append(f"## {stripped}")
                    continue

            # =====================================================================
            #  语义启发式标题识别（没有明确章节标记的文档）
            # =====================================================================

            # --- 规则 S1：中文编号标题（"一、xxx"、"（一）xxx"）---
            cn_num_match = self._CHINESE_NUM_PATTERN.match(stripped)
            if cn_num_match and len(stripped) < 60:
                md_lines.append(f"### {cn_num_match.group(1)}")
                continue
            cn_br_match = self._CHINESE_NUM_BRACKET.match(stripped)
            if cn_br_match and len(stripped) < 60:
                md_lines.append(f"#### {cn_br_match.group(1)}")
                continue

            # --- 规则 S2：英文编号标题（"Section 1 xxx"）---
            eng_sec_match = self._ENGLISH_SECTION_RE.match(stripped)
            if eng_sec_match and len(stripped) < 80:
                rest = eng_sec_match.group(2).strip()
                md_lines.append(f"## {rest}" if rest else f"## {stripped}")
                continue

            # --- 规则 S3：问答式标题（"Q: xxx"、"问题：xxx"）---
            qa_match = self._QA_PATTERN.match(stripped)
            if qa_match and len(stripped) < 80:
                md_lines.append(f"### {qa_match.group(2).strip()}")
                continue

            # --- 规则 S4：语义标题启发式 ---
            # 短文本（2-20字符），没有句号结尾，且包含特定关键词 → 可能是标题
            # 需要同时满足：前后有空行，或者没有逗号/冒号分隔的正文
            is_short = 2 <= len(stripped) <= 20
            no_punct_end = not stripped.endswith(("。", "，", "；", "：", ":", ",", ";", "）", ")", "！", "？", "!", "?"))
            has_colon_content = re.match(r'^.+[：:]\s*.+', stripped)  # 有 "关键词：内容" 格式的通常是正文

            # 检查是否像语义标题
            looks_like_semantic_title = False
            inferred_level = 2  # 默认 L2

            # 检查是否匹配语义标题模式
            if is_short and no_punct_end and not has_colon_content:
                # 匹配 "xxx介绍"、"xxx说明" 等
                if self._SEMANTIC_TITLE_RE.match(stripped):
                    looks_like_semantic_title = True
                    # 重新划分：只有纯粹的 "概述" "简介" 是 L2，其他具体主题都是 L3
                    if stripped in {'概述', '简介', '前言', '背景', '总览', '概要'}:
                        inferred_level = 2
                    # 包含具体内容的标题（如"产品介绍"、"使用说明"、"购买流程"）→ L3
                    elif any(kw in stripped for kw in {'介绍', '说明', '指南', '流程', '步骤', '服务', '政策', '功能', '特点', '要求', '规范', '类型', '分类', '使用', '操作', '配置'}):
                        inferred_level = 3
                    elif any(kw in stripped for kw in {'补充', '备注', '附录', '参考', '示例', '注意事项', '详情', '参数', '规格'}):
                        inferred_level = 4

                # 匹配常见的中文主题词（如"常见问题"、"注意事项"等）
                elif stripped in {'常见问题', 'FAQ', '注意事项', '产品介绍', '使用说明', '售后服务', '技术参数'} or \
                     any(stripped.startswith(kw) for kw in {'常见问题', '注意事项', 'FAQ', '产品介绍', '使用说明', '售后服务', '技术参数'}):
                    looks_like_semantic_title = True
                    # 重新划分：只有纯粹的 "概述" "简介" 是 L2，其他具体主题都是 L3
                    if stripped in {'概述', '简介', '前言', '背景', '总览', '概要'}:
                        inferred_level = 2
                    # 包含具体内容的标题（如"产品介绍"、"使用说明"）→ L3
                    elif '介绍' in stripped or '说明' in stripped or '服务' in stripped or '流程' in stripped or '问题' in stripped or stripped == 'FAQ' or '功能' in stripped or '特点' in stripped:
                        inferred_level = 3
                    elif '补充' in stripped or '注意事项' in stripped or '备注' in stripped:
                        inferred_level = 4

                # 匹配全部大写英文标题
                elif self._ENGLISH_TITLE_RE.match(stripped) and len(stripped.split()) <= 5:
                    looks_like_semantic_title = True
                    inferred_level = 3

                # 匹配以 "什么是"、"如何" 开头的问答式标题
                elif stripped.startswith(("什么是", "为什么", "如何", "什么")) and len(stripped) < 20:
                    looks_like_semantic_title = True
                    inferred_level = 3

            if looks_like_semantic_title:
                prefix = "#" * inferred_level
                md_lines.append(f"{prefix} {stripped}")
                continue

            # ============== 列表识别 ==============
            if re.match(r'^[\-•\*]\s+', stripped):
                md_lines.append(stripped)
                continue

            # 有序列表（较长的排除被当标题）
            if re.match(r'^\d+[\.、）\)]\s+\S', stripped) and len(stripped) > 40:
                md_lines.append(stripped)
                continue

            # ============== 普通段落 ==============
            md_lines.append(stripped)

        return "\n".join(md_lines)

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        """启发式判断是否为标题"""
        text = text.strip()
        if len(text) > 50:
            return False
        if text.endswith(("。", "，", "；", ".", ",", ";", "）", ")")):
            return False

        heading_keywords = [
            "第", "一", "二", "三", "四", "五",
            "概述", "介绍", "前言", "总结", "结论",
            "定义", "说明", "注意", "提示", "要求",
            "功能", "特性", "安装", "配置", "使用",
            "概览", "背景", "目标", "范围", "方案",
        ]
        for kw in heading_keywords:
            if text.startswith(kw):
                return True
        return False