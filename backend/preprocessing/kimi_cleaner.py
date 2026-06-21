"""
DeepSeek 文档清洗器

使用 DeepSeek 模型对 Markdown 格式文档进行智能清洗：
  - 去除页眉页脚、页码等脏数据
  - 修正 OCR 识别错误
  - 规范化 Markdown 格式（标题层级、表格对齐等）
  - 图片内容提取
  - 去除重复和无意义内容

优化说明：
  - 多页 / 多片并发调用 DeepSeek API（ThreadPoolExecutor）
  - 并发度默认 3，避免触发 rate limit
  - 长文本分片也并行处理，总耗时 ≈ max(单页耗时) 而非 sum(所有页)
"""

from pathlib import Path
import time
import re

from openai import OpenAI
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.config import settings
from backend.preprocessing.parser import ParsedDocument

# =====================================================================
# 预编译正则（避免每次调用重新编译）
# =====================================================================
_MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)')
_CHINESE_SECTION_RE = re.compile(r'^第[一二三四五六七八九十百\d]+部分')
_PURE_PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$')
_HEADER_FOOTER_RE = re.compile(r'^(第\s*\d+\s*页|Page\s+\d+|-\s*\d+\s*-)$', re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r'\s{2,}')

# LLM 清洗并发度：默认 3（避免 DeepSeek rate limit）
# 如果你的 API key 配额更高，可以在 settings 中设置 cleaning_concurrency
_LLM_CONCURRENCY = getattr(settings, 'cleaning_concurrency', 3)


# =====================================================================
#  DeepSeek 清洗 Prompt 模板
# =====================================================================

CLEANING_SYSTEM_PROMPT = """你是一个专业的 Markdown 文档清洗助手。你的任务是清洗文档、去除脏数据，并**统一所有章节标题为标准 Markdown 格式**，但不改变原有内容语义。

# 严格规则（必须遵守）

## 1. 统一层级规范（核心，必须严格执行）

所有标题**必须**符合以下 4 级层级：

| 层级 | Markdown | 匹配规则 | 示例 |
|------|----------|---------|------|
| L1 | `#` | 文档总标题（唯一） | `# fTaoBao 电商平台服务指南` |
| L2 | `##` | "第N章"、"第N部分"、顶层服务说明、一级数字编号章节 | `## 第二章 商品分类详解`、`## 第三章 商品详细信息 - 手机数码类` |
| L3 | `###` | "N.M xxx" 子分类、方括号编号章节、中文编号小节 | `### 2.1 手机数码类`、`### 2.7 运动户外类` |
| L4 | `####` | "商品 NNN" 具体条目、商品属性关键词 | `#### 商品 001: iPhone 15 Pro Max`、`#### 核心卖点` |

**强制规则（重中之重）：**
- 所有格式的"第N章"、"第N部分" → `##`（无论原格式是什么）
  - 例：`### 第三章 商品详细信息` → `## 第三章 商品详细信息`
  - 例：`第二章 商品分类详解` → `## 第二章 商品分类详解`
- 所有格式的"N.M xxx"（N和M是数字）→ `###`（无论原格式是什么）
  - 例：`## 2.7 运动户外类` → `### 2.7 运动户外类`
  - 例：`【2.7 运动户外类】` → `### 2.7 运动户外类`
  - 例：`2.1 手机数码类` → `### 2.1 手机数码类`
- 所有格式的"商品 NNN xxx" → `####`
  - 例：`## 商品 001 MacBook Pro 16寸` → `#### 商品 001: MacBook Pro 16寸`
  - 例：`【商品 004】华为 Mate 60 Pro` → `#### 商品 004: 华为 Mate 60 Pro`
- 所有商品属性关键词（见下方列表）→ `####`
  - 例：`## 售后保障：正品行货...` → `#### 售后保障` + 正文

## 2. 非标准标题格式转换

将所有非标准格式标题转换为标准 Markdown 标题：

- `【2.7 运动户外类】` → `### 2.7 运动户外类`
- `【商品 001】iPhone 15 Pro Max` → `#### 商品 001: iPhone 15 Pro Max`
- `【商品编号查询说明】` → `### 商品编号查询说明`
- `一、产品概述` → `### 产品概述`
- `（一）安装步骤` → `#### 安装步骤`
- `1. xxx` → `## xxx`（一级数字编号章节）
- `1.1 xxx` → `### xxx`（二级数字编号）
- `1.1.1 xxx` → `#### xxx`（三级数字编号）
- `第一部分 XXX` → `## 第一部分 XXX`
- `第二章 XXX` → `## 第二章 XXX`
- 已有标准 Markdown 标题（`#` / `##` / `###` / `####`）：**如果层级正确则保留，否则按上述规则修正**

## 3. 商品属性关键词（#### 级别）

以下关键词**必须**作为 `####` 小标题（无论原格式是什么）：

商品编号、商品品牌、商品售价、商品简介、详细规格、补充说明、售后保障、
保修政策、配送方式、包装清单、商品编号前缀、主要商品类型、
适用人群、核心卖点、发票说明、售后服务、退换说明、安装服务

**规则：**
- 如果出现为"核心卖点：品牌授权、正品保障" → 拆分为 `#### 核心卖点` + 正文段落
- 如果被误判为 `##` 或 `###` → 强制降级为 `####`
- 如果关键词后跟冒号（`：`或`:`）+ 内容 → 关键词独立为 `####` 行，内容作为正文

## 4. 语义标题识别（重要！无明确章节标记的文档）

**对于没有"第N章"、"1.1"等明确编号的文档，你必须通过语义分析来识别标题：**

### 4.1 如何判断某行是标题（而非正文）

如果一段文字满足以下条件之一，它可能是标题：
- 是一个短句（2-20字），且后面跟着详细描述
- 包含"概述"、"介绍"、"说明"、"指南"、"流程"、"步骤"、"功能"、"特点"、"服务"、"政策"、"要求"、"规范"、"注意"、"问题"、"方案"、"参考"、"示例"、"补充"、"备注"等主题词
- 是一个独立的主题名，如"产品介绍"、"使用说明"、"售后服务"、"技术参数"
- 是英文大写的短语，如"INTRODUCTION"、"INSTALLATION"
- 以"什么是"、"如何"、"为什么"开头的问答式标题
- 以"Q:"、"问题："开头的提问

**但是：** 如果这行文字包含冒号（":"或"："）并且后面跟有具体内容，则很可能是正文的一部分而不是独立标题（例如"核心卖点：品牌授权"应该拆为"#### 核心卖点" + 正文）

### 4.2 如何根据语义推断层级

根据标题的概括性和位置推断层级：

- **L2 (##) 高层级标题**：概括性强，覆盖较大范围的主题
  - 出现在文档开头或文档的主要分段处
  - 包含"概述"、"简介"、"介绍"、"背景"、"总览"、"前言"等词
  - 是一个大主题的开端，如"电商平台服务"、"产品介绍"、"使用指南"

- **L3 (###) 中层级标题**：具体的主题分类
  - 包含"说明"、"流程"、"步骤"、"功能"、"特点"、"服务"、"政策"、"要求"、"规范"等词
  - 是某个大主题下的具体子主题
  - 如"购买流程"、"售后服务"、"常见问题"、"使用方法"

- **L4 (####) 低层级标题**：具体细节和属性
  - 包含"补充"、"备注"、"示例"、"案例"、"详情"、"参数"、"规格"等词
  - 是某个具体主题下的细节描述
  - 如"核心卖点"、"售后保障"、"详细规格"

### 4.3 识别主题切换

当你发现以下情况时，说明可能是一个新标题/新章节的开始：
- 一行简短的主题词后，跟着几段详细的说明文字
- 话题从一个主题明显切换到另一个主题（例如从"介绍产品"切换到"购买流程"）
- 问答格式的内容：每个"Q:"、"问题："都是新标题的开始

### 4.4 避免误判

不要将以下内容识别为标题：
- 正常的段落文本（即使包含主题词，例如"我们的产品特点是..."是正文）
- 列表项（例如"- 核心功能：xxx"保持为列表项，除非它确实是独立的小标题）
- 带有冒号的句子（例如"注意：请仔细阅读以下内容"是正文提示，不是标题）

## 5. 内容保留与合并
- 所有正文段落、表格、列表必须完整保留，不得删除任何实质性信息。
- 不要对内容进行改写、摘要、合并或精简（但可以**将标题+内容的同行拆分为两行**）。
- 如果原文段落被页码/页眉切断，将它们合并为一个完整段落。
- **同一商品的信息（详细规格、保修政策、配送方式等）必须保持在同一个 `#### 商品 NNN` 子树内**。

## 6. 去除脏数据
- 删除页眉（如每页重复出现的公司名、文档名等）。
- 删除页脚（版权声明、联系方式等）。
- 删除页码（单独成行的数字，如 "42"、"- 42 -"、"第 42 页"）。
- 删除无意义的分隔线（如 "------"、"======"）。

## 7. 修正格式
- 修正乱码字符、错误的标点符号（如全角/半角混用）。
- 列表使用标准 Markdown 格式（"- "、"1. "）。
- 表格使用标准 Markdown 表格格式。
- 段落之间保留一个空行作为分隔。
- 标题行与正文之间必须有一个空行。

## 8. 禁止行为
- ❌ 不要添加任何原文没有的标题或内容。
- ❌ 不要将不同语义的内容合并到同一章节（如"手机数码类"和"电脑办公类"必须是两个独立章节）。
- ❌ 不要将属于同一主题的内容分割到不同章节（如同一条商品的详细信息、保修、配送应该在同一商品章节下）。
- ❌ **不要将"第N章"格式标题写成 `###`，必须是 `##`！**
- ❌ **不要将"N.M"格式标题写成 `##`，必须是 `###`！**
- ❌ **不要将"商品 NNN"格式标题写成 `##`，必须是 `####`！**
- ❌ **不要将商品属性关键词写成 `##` 或 `###`，必须是 `####`！**
- ❌ 不要添加任何解释性文字、注释、清洗日志。
- ❌ 不要在输出前后添加任何额外的标记。
- ❌ 不要过度识别为标题！只有确实是标题的行才转为 Markdown 标题，普通正文保持原样。

# 输出要求
只输出清洗后的 Markdown 内容，不添加任何其他文字。"""


# 图片专用清洗 Prompt
IMAGE_CLEANING_PROMPT = """请提取这张图片中的所有文字信息，并保留图片的结构。

要求：
1. 完整提取所有可见文字
2. 保留表格结构（用 Markdown 表格表示）
3. 保留列表结构
4. 忽略水印和无关装饰文字
5. 按阅读顺序输出
6. 只输出提取的文字内容，不要解释
"""


class DocCleaner:
    """
    DeepSeek 文档清洗器

    使用 DeepSeek API 对 Markdown 内容进行清洗。
    DeepSeek API 兼容 OpenAI 格式，通过 base_url 切换。
    """

    # 每次清洗的最大字符数（避免超出 token 限制）
    MAX_CHUNK_SIZE = 6000

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 DeepSeek API 客户端"""
        api_key = settings.llm_api_key
        if not api_key:
            logger.warning("DeepSeek API Key 未配置，清洗将降级为本地清洗")
            return

        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.llm_api_base,
        )
        logger.info(f"DeepSeek 清洗客户端初始化: model={settings.cleaning_model}")

    # =====================================================================
    #  对外接口
    # =====================================================================

    def clean(self, doc: ParsedDocument, source_type: str = "document") -> ParsedDocument:
        """
        清洗文档内容。

        Args:
            doc: 待清洗的文档
            source_type: 来源类型 "document"(文本) / "image"(图片) / "excel"(表格)

        Returns:
            清洗后的文档对象（text 字段已更新）
        """
        ext = Path(doc.file_path).suffix.lower()

        if not settings.enable_llm_cleaning:
            logger.info(f"DeepSeek 清洗已禁用，跳过: {doc.file_path}")
            return doc

        if source_type == "image" or ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            return self._clean_image(doc)

        if source_type == "excel" or ext in (".xlsx", ".xls"):
            return self._clean_excel(doc)

        return self._clean_document(doc)

    # =====================================================================
    #  文本/常规文档清洗
    # =====================================================================

    def _clean_document(self, doc: ParsedDocument) -> ParsedDocument:
        """清洗常规 Markdown 文档（多页并行版）"""
        logger.info(f"LLM 清洗文档: {doc.title}  [{len(doc.raw_text)} 字符, {len(doc.pages)} 页]")

        if doc.raw_text.strip().startswith("# 图片:"):
            return self._clean_image(doc)

        # ---------- 多页并行清洗 ----------
        # 每一页互相独立，可以并发调用 DeepSeek API
        # 对每页，如果长度超过 MAX_CHUNK_SIZE 还会在内部再分片并发
        non_empty_indices = []
        non_empty_texts = []
        for i, page in enumerate(doc.pages):
            if page.text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(page.text)

        total_original = sum(len(t) for t in non_empty_texts)
        cleaned_results: dict[int, str] = {}

        if non_empty_texts:
            t0 = time.time()
            # 控制并发度：避免 DeepSeek rate limit
            concurrency = min(_LLM_CONCURRENCY, len(non_empty_texts))

            def _clean_page(idx: int, text: str) -> tuple[int, str]:
                return idx, self._clean_text_chunk(text)

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(_clean_page, idx, text)
                    for idx, text in zip(non_empty_indices, non_empty_texts)
                ]
                for fut in as_completed(futures):
                    idx, cleaned = fut.result()
                    cleaned_results[idx] = cleaned

            dt = time.time() - t0
            logger.info(
                f"LLM 清洗 {len(non_empty_texts)} 页, 并发度={concurrency}, 耗时 {dt:.1f}s "
                f"({len(non_empty_texts) / max(dt, 0.01):.1f} 页/s)"
            )

        # ---------- 回填 pages ----------
        total_cleaned = 0
        cleaned_pages = []
        for i, page in enumerate(doc.pages):
            if i in cleaned_results:
                cleaned = cleaned_results[i]
                total_cleaned += len(cleaned)
                cleaned_pages.append(cleaned)
            else:
                cleaned_pages.append(page.text)

        # 更新 page.text
        for i, page in enumerate(doc.pages):
            if i < len(cleaned_pages):
                page.text = cleaned_pages[i]

        doc.raw_text = "\n\n".join(cleaned_pages)

        # ---------- 清洗后处理：智能修复标题层级 ----------
        # LLM 逐页清洗时可能破坏跨页的层级关系，这里进行全局修复
        doc.raw_text = self._post_cleaning_fix_headings(doc.raw_text)

        # 同步更新 page.text（简化处理：将修复后的全文按比例分配给各页）
        self._sync_pages_from_raw(doc)

        logger.info(
            f"LLM 清洗完成: {total_original} → {total_cleaned} 字符 "
            f"({(total_cleaned / max(total_original, 1)) * 100:.1f}%)"
        )
        return doc

    def _post_cleaning_fix_headings(self, text: str) -> str:
        """
        统一标题层级 —— 智能修复（最终保障）

        核心处理逻辑（只修改有明确模式的标题）：
        1. "第N章"、"第N部分" → 强制 L2 (##)
           - 例：`### 第三章 商品详细信息` → `## 第三章 商品详细信息`
        2. "N.M xxx"（如 "2.1 手机数码类"、"2.7 运动户外类"）→ 强制 L3 (###)
           - 例：`## 2.7 运动户外类` → `### 2.7 运动户外类`
        3. "商品 NNN xxx"（商品条目）→ 强制 L4 (####)
           - 例：`## 商品 001 MacBook Pro 16寸` → `#### 商品 001: MacBook Pro 16寸`
        4. 方括号标题 `【...】` → 转为对应层级标准 Markdown
        5. 商品属性关键词（核心卖点/售后保障等）→ 强制 L4 (####)
           - 例：`## 售后保障：正品行货...` → `#### 售后保障` + 正文
        6. 仅有一个 # 主标题（必要时自动生成）

        注意："N. xxx" 格式（如 "1. 正品保障"）层级依赖上下文，不做强制修改，
             只在它被错误地识别为 "第N章"、"N.M"、"商品 NNN" 或关键词时才修正。
        """
        md_heading_re = re.compile(r'^(#+)\s+(.+)$')
        chinese_chapter_re = re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部]\b')
        num_n_m_re = re.compile(r'^(\d+)\.(\d+)\s+(.+)$')
        product_item_re = re.compile(r'^(商品\s*\d+[^:：]*?)[:：\s]*\s*(.+)$')
        bracket_num_section = re.compile(r'^[【\[]\s*(\d+\.\d+)\s*(.+?)\s*[】\]]\s*$')
        bracket_item = re.compile(r'^[【\[]\s*([^】\]]*?\d+[^】\]]*?)\s*[】\]]\s*(.+)$')
        bracket_plain = re.compile(r'^[【\[]\s*([^】\]]+?)\s*[】\]]\s*$')

        keyword_subheadings = {
            '商品编号', '商品品牌', '商品售价', '商品简介',
            '详细规格', '补充说明', '售后保障', '保修政策',
            '配送方式', '包装清单', '商品编号前缀', '主要商品类型',
            '适用人群', '核心卖点', '发票说明', '售后服务',
            '退换说明', '安装服务',
        }

        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue

            md_match = md_heading_re.match(stripped)

            if md_match:
                level = len(md_match.group(1))
                title = md_match.group(2).strip()

                # --- 检查 1：标题是 "第N章" / "第N部分" 模式？---
                # 例：`### 第三章 商品详细信息` → `## 第三章 商品详细信息`
                if chinese_chapter_re.match(title):
                    if level != 2:
                        fixed_lines.append(f'## {title}')
                    else:
                        fixed_lines.append(line)
                    continue

                # --- 检查 2：标题是 "N.M xxx" 模式？（注意要优先于关键词检查）---
                # 例：`## 2.7 运动户外类` → `### 2.7 运动户外类`
                nm_match = num_n_m_re.match(title)
                if nm_match:
                    if level != 3:
                        fixed_lines.append(f'### {title}')
                    else:
                        fixed_lines.append(line)
                    continue

                # --- 检查 3：标题是 "商品 NNN xxx" 商品条目？---
                # 例：`## 商品 001 MacBook Pro 16寸` → `#### 商品 001: MacBook Pro 16寸`
                prod_match = product_item_re.match(title)
                if prod_match:
                    fixed_lines.append(f'#### {prod_match.group(1).strip()}: {prod_match.group(2).strip()}')
                    continue

                # --- 检查 4：标题是商品属性关键词？---
                # 例：`## 售后保障：正品行货...` → `#### 售后保障` + 正文
                is_keyword = False
                for kw in keyword_subheadings:
                    if title == kw or title.startswith(kw + ':') or title.startswith(kw + '：'):
                        is_keyword = True
                        rest = title[len(kw):].lstrip('：: :：,，').strip()
                        fixed_lines.append(f'#### {kw}')
                        if rest:
                            fixed_lines.append(rest)
                        break
                if is_keyword:
                    continue

                # =====================================================================
                #  检查 5：基于标题内容的语义层级推断（无明确编号的文档）
                # =====================================================================
                # 规则：根据标题中的关键词推断其层级
                #   - L2 (##)：纯粹的概括性主题，如"概述"、"简介"、"背景"
                #   - L3 (###)：具体的主题/章节，如"产品介绍"、"使用说明"、"购买流程"、"售后服务"
                #   - L4 (####)：具体的细节/属性，如"补充说明"、"注意事项"、"参数规格"
                # 高层级关键词（L2）：纯粹的概括
                h2_keywords = {'概述', '简介', '背景', '前言', '总览', '概要', '引言', '绪论'}
                # 中层级关键词（L3）：具体主题、章节
                h3_keywords = {'介绍', '说明', '指南', '流程', '步骤', '方法', '功能', '特点', '服务', '政策',
                               '要求', '规范', '注意', '问题', '方案', '类型', '分类', '使用', '操作',
                               '配置', '设置', '安装', '常见问题', 'FAQ', '规则', '标准', '方式', '售后服务'}
                # 低层级关键词（L4）：具体细节、属性
                h4_keywords = {'补充', '备注', '附录', '参考', '示例', '案例', '注意事项', '详情', '参数', '规格'}

                inferred_level = None

                # 检查是否包含高/中/低层级关键词（优先检查更精确的匹配）
                # 1. 检查是否以关键词开头或等于关键词
                if title in h2_keywords or any(title.startswith(kw) for kw in h2_keywords):
                    inferred_level = 2
                elif title in h3_keywords or any(title.startswith(kw) for kw in h3_keywords):
                    inferred_level = 3
                elif title in h4_keywords or any(title.startswith(kw) for kw in h4_keywords):
                    inferred_level = 4

                # 2. 检查是否包含关键词（如"产品介绍"包含"介绍" → L3）
                if inferred_level is None:
                    for kw in h4_keywords:
                        if kw in title and len(title) < 20:
                            inferred_level = 4
                            break
                if inferred_level is None:
                    for kw in h3_keywords:
                        if kw in title and len(title) < 20:
                            inferred_level = 3
                            break
                if inferred_level is None:
                    for kw in h2_keywords:
                        if kw in title and len(title) < 20:
                            inferred_level = 2
                            break

                # 3. 检查问答式标题（"什么是xxx"、"如何xxx"）→ L3
                if inferred_level is None:
                    if title.startswith(("什么是", "为什么", "如何", "什么")) and len(title) < 20:
                        inferred_level = 3

                # 如果推断出层级且与当前层级不同，则调整
                if inferred_level is not None and level != inferred_level:
                    prefix = "#" * inferred_level
                    fixed_lines.append(f"{prefix} {title}")
                    continue

                # --- 其他标题：保持原样（不做强制修改）---
                fixed_lines.append(line)
                continue

            # ============ 非 Markdown 行：检查非标准标题格式 ============
            b1 = bracket_num_section.match(stripped)
            if b1 and len(stripped) < 80:
                fixed_lines.append(f"### {b1.group(1)} {b1.group(2).strip()}")
                continue

            b2 = bracket_item.match(stripped)
            if b2 and len(stripped) < 150:
                bracket_content = ' '.join(b2.group(1).strip().split())
                after_content = b2.group(2).strip()
                title = f"{bracket_content}: {after_content}" if after_content else bracket_content
                fixed_lines.append(f"#### {title}")
                continue

            b3 = bracket_plain.match(stripped)
            if b3 and len(stripped) < 60 and '。' not in stripped:
                fixed_lines.append(f"### {b3.group(1).strip()}")
                continue

            is_kw_line = False
            for kw in keyword_subheadings:
                if stripped.startswith(kw + '：') or stripped.startswith(kw + ':'):
                    rest = stripped[len(kw) + 1:].strip()
                    fixed_lines.append(f'#### {kw}')
                    if rest:
                        fixed_lines.append(rest)
                    is_kw_line = True
                    break
            if is_kw_line:
                continue

            fixed_lines.append(line)

        # ============ 最终保障：确保至少有一个 # 主标题 ============
        has_h1 = False
        for line in fixed_lines:
            m = md_heading_re.match(line.strip())
            if m and len(m.group(1)) == 1:
                has_h1 = True
                break

        if not has_h1:
            for i, line in enumerate(fixed_lines):
                m = md_heading_re.match(line.strip())
                if m:
                    fixed_lines[i] = f"# {m.group(2).strip()}"
                    break

        return '\n'.join(fixed_lines)

    def _sync_pages_from_raw(self, doc: ParsedDocument):
        """根据修复后的 raw_text 同步更新 pages.text"""
        if not doc.pages:
            return

        total_len = len(doc.raw_text)
        page_count = len(doc.pages)

        if page_count == 1:
            doc.pages[0].text = doc.raw_text
            return

        # 按字符位置大致分配
        chars_per_page = total_len // page_count
        pos = 0
        for i, page in enumerate(doc.pages):
            if i == page_count - 1:
                page.text = doc.raw_text[pos:]
            else:
                # 避免在段落中间切断
                end = min(pos + chars_per_page, total_len)
                # 尝试在换行处截断
                newline_pos = doc.raw_text.find('\n\n', end - 200, end + 200)
                if newline_pos == -1:
                    newline_pos = end
                page.text = doc.raw_text[pos:newline_pos]
                pos = newline_pos

    # =====================================================================
    #  Excel 表格清洗
    # =====================================================================

    def _clean_excel(self, doc: ParsedDocument) -> ParsedDocument:
        """清洗 Excel 转换后的 Markdown 表格（多页并行版）"""
        logger.info(f"LLM 清洗表格: {doc.title}  [{len(doc.pages)} 页]")

        non_empty_indices = []
        non_empty_texts = []
        for i, page in enumerate(doc.pages):
            if page.text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(page.text)

        cleaned_results: dict[int, str] = {}
        if non_empty_texts:
            t0 = time.time()
            concurrency = min(_LLM_CONCURRENCY, len(non_empty_texts))

            def _clean_page(idx: int, text: str) -> tuple[int, str]:
                return idx, self._clean_text_chunk(
                    text,
                    extra_instruction="这是表格数据，请将表格格式规范化，去除空白行和无效数据。",
                )

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(_clean_page, idx, text)
                    for idx, text in zip(non_empty_indices, non_empty_texts)
                ]
                for fut in as_completed(futures):
                    idx, cleaned = fut.result()
                    cleaned_results[idx] = cleaned

            dt = time.time() - t0
            logger.info(
                f"LLM 表格清洗 {len(non_empty_texts)} 页, 并发度={concurrency}, 耗时 {dt:.1f}s"
            )

        cleaned_pages = []
        for i, page in enumerate(doc.pages):
            cleaned_pages.append(cleaned_results.get(i, page.text))

        for i, page in enumerate(doc.pages):
            if i < len(cleaned_pages):
                page.text = cleaned_pages[i]

        doc.raw_text = "\n\n".join(cleaned_pages)
        return doc

    # =====================================================================
    #  图片清洗
    # =====================================================================

    def _clean_image(self, doc: ParsedDocument) -> ParsedDocument:
        """
        使用 DeepSeek 视觉模型提取图片内容。
        将图片文件直接发送给 DeepSeek Vision API 进行 OCR/内容提取。
        """
        logger.info(f"DeepSeek Vision 处理图片: {doc.title}")

        if not self.client:
            logger.warning("DeepSeek API 不可用，保留 OCR 结果")
            return doc

        try:
            # 读取图片为 base64
            import base64
            with open(doc.file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = Path(doc.file_path).suffix.lower().lstrip(".")
            mime_types = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
                "gif": "image/gif",
            }
            mime_type = mime_types.get(ext, "image/png")

            response = self.client.chat.completions.create(
                model=settings.cleaning_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": IMAGE_CLEANING_PROMPT,
                            },
                        ],
                    }
                ],
                temperature=settings.cleaning_temperature,
                max_tokens=settings.cleaning_max_tokens,
            )

            extracted_text = response.choices[0].message.content.strip()

            if extracted_text:
                # 替换文档内容为 DeepSeek 提取的结果
                md_content = (
                    f"# {doc.title}\n\n"
                    f"> 来源图片: {Path(doc.file_path).name}\n\n"
                    f"{extracted_text}"
                )

                doc.raw_text = md_content
                for page in doc.pages:
                    page.text = md_content

                logger.info(f"DeepSeek Vision 提取完成: {len(extracted_text)} 字符")

        except Exception as e:
            logger.error(f"DeepSeek Vision 处理失败: {e}")
            logger.info("降级使用 OCR 结果")

        return doc

    # =====================================================================
    #  核心：分段清洗
    # =====================================================================

    def _clean_text_chunk(self, text: str, extra_instruction: str = "") -> str:
        """
        清洗一段文本。

        - 短文本（< MAX_CHUNK_SIZE）：直接发送给 DeepSeek
        - 长文本：分片发送，最后合并
        """
        if not self.client:
            return self._local_clean(text)

        if len(text) <= self.MAX_CHUNK_SIZE:
            return self._call_llm_clean(text, extra_instruction)

        # 长文本分片
        return self._clean_long_text(text)

    def _call_llm_clean(self, text: str, extra_instruction: str = "") -> str:
        """调用 DeepSeek API 清洗文本"""
        try:
            user_instruction = CLEANING_SYSTEM_PROMPT
            if extra_instruction:
                user_instruction += f"\n\n额外要求：{extra_instruction}"

            response = self.client.chat.completions.create(
                model=settings.cleaning_model,
                messages=[
                    {"role": "system", "content": user_instruction},
                    {"role": "user", "content": f"请清洗以下文档内容：\n\n{text}"},
                ],
                temperature=settings.cleaning_temperature,
                max_tokens=settings.cleaning_max_tokens,
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            logger.error(f"DeepSeek 清洗调用失败: {e}")
            logger.info("降级为本地清洗")
            return self._local_clean(text)

    def _clean_long_text(self, text: str) -> str:
        """长文本分片清洗（多片并发版）"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.MAX_CHUNK_SIZE and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        if len(chunks) <= 1:
            return self._call_llm_clean(chunks[0]) if chunks else ""

        logger.info(f"长文本分 {len(chunks)} 片并行清洗，每片 ≤ {self.MAX_CHUNK_SIZE} 字符")

        # 并发调用：控制并发度为 _LLM_CONCURRENCY
        concurrency = min(_LLM_CONCURRENCY, len(chunks))
        cleaned_chunks: list[str] = [""] * len(chunks)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self._call_llm_clean, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                cleaned_chunks[i] = fut.result()

        dt = time.time() - t0
        logger.info(
            f"长文本 {len(chunks)} 片清洗完成, 并发度={concurrency}, 耗时 {dt:.1f}s"
        )

        return "\n\n".join(cleaned_chunks)

    # =====================================================================
    #  本地降级清洗（DeepSeek 不可用时）
    # =====================================================================

    def _local_clean(self, text: str) -> str:
        """本地规则清洗（降级方案，优化版：预编译正则）"""
        lines = text.split("\n")
        cleaned = []
        prev_blank = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if not prev_blank:
                    cleaned.append("")
                    prev_blank = True
                continue
            prev_blank = False

            # 跳过纯页码
            if _PURE_PAGE_NUM_RE.match(stripped):
                continue

            # 跳过明显页眉页脚
            if _HEADER_FOOTER_RE.match(stripped):
                continue

            cleaned_line = _MULTI_SPACE_RE.sub(' ', stripped)
            cleaned.append(cleaned_line)

        return "\n".join(cleaned)