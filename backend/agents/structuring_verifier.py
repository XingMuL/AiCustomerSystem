"""
LLM 语义分割验证 Agent —— 纯 LLM 评审版
===============================================

核心职责：
1. 接收第一轮 LLM 输出的章节边界，用 LLM 做第二遍"语义质量评审"
2. 识别出质量不完整/错误章节，由 LLM 给出修正后的边界
3. 合并修正结果，确保最终章节边界是 LLM 语义认可

设计原则：
- 不做任何规则硬边界（不扫描 #### 商品 NNN:）
- 不做任何规则打分（不用关键词匹配/边界位置检查等）
- 100% 由 LLM 做语义评判和修正
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from loguru import logger

from backend.ops.metrics_collector import get_metrics_collector


# ============================================================
# LLM 评审 Prompt
# ============================================================
VERIFIER_SYSTEM_PROMPT = """你是一位文档结构评审专家。

你的职责：
1. 阅读完整文档
2. 阅读别人已经识别出的章节边界
3. 从语义上评审每个边界是否合理
4. 对错误或不完整的章节给出修正建议

# 评审标准（按重要性排序）

1. **语义完整性**：一个完整语义单元必须完整，不要被切分到多个章节
2. **主题一致性**：同一主题的内容应归为同一章节
3. **边界准确性**：章节边界应落在完整段落/内容块的自然断点处，不在句子中间
4. **层级合理**：大主题用 level 2，子主题/具体条目用 level 3

# 典型错误（需要你修正）：
- 把同一商品（名称/规格/售价被切分到多个章节
- 把一个完整段落中间切开
- 章节边界切在句子中间
- 同一个文档段落漏掉内容

请严格以 JSON 格式输出评审结果，不要添加任何其他文字。
"""

VERIFIER_USER_PROMPT = """以下是完整文档内容：
---- 文档开始 ---
{document_content}
---- 文档结束 ---

以下是第一轮 LLM 识别出的章节边界：
---- 章节列表 ---
{chapters_json}
---- 章节列表结束 ---

请从语义上评审每个章节，识别出所有需要修正的章节，并给出修正后的正确边界。

请严格按以下 JSON 格式输出，不要添加任何其他文字：
{{
  "reviewed_chapters": [
    {{
      "original_index": 0,
      "keep": true,
      "title": "章节标题",
      "level": 2,
      "start_char": 0,
      "end_char": 1000
    }},
    {{
      "original_index": 1,
      "keep": false,
      "reason": "商品信息被拆分",
      "corrected_title": "商品 004: 华为 Mate 60 Pro",
      "level": 3,
      "start_char": 500,
      "end_char": 3200
    }}
  ]
}}

字段说明：
- original_index: 对应上面"章节列表"中的序号（0开始）
- keep: true=接受原边界，false=需要修正
- reason: keep=false 时必须给出修正理由
- title/corrected_title: 章节标题
- level: 1=文档标题，2=一级章节，3=二级子章节
- start_char / end_char: 相对完整文档的字符位置索引

重要提示：
- start_char 必须是行首或段落开头，不要切在句子中间
- end_char 必须是行尾或段落结尾
- 如果原章节完全正确则 keep=true
- 如果原章节需要修正则 keep=false 并给出修正边界
- 如果发现原章节完全漏掉了某些内容（如商品编号、规格等），请在修正边界中包含这些内容
- 输出的 start_char/end_char 必须对应完整文档，而非片段
"""


# ============================================================
# 核心 Agent
# ============================================================
class ChapterBoundaryVerificationAgent:
    """
    纯 LLM 语义评审 Agent：100% LLM 评审，0 规则"""

    def __init__(
        self,
        llm_client: Optional[object] = None,
        llm_model: str = "auto",
    ):
        self.client = llm_client
        self.model = llm_model
        self.metrics_collector = get_metrics_collector()

    # ============================================================
    # 核心入口
    # ============================================================
    def verify_and_refine(
        self,
        llm_chapters: List[Dict],
        full_text: str,
        session_id: str = "indexing",
    ) -> List[Dict]:
        """
        主入口：用 LLM 评审章节边界质量，修正后返回最终章节列表（接入 ops 监控）

        Args:
            llm_chapters: 第一轮 LLM 识别出的章节列表
            full_text: 完整文档内容
            session_id: 用于监控的会话 ID
        Returns:
            修正后的章节列表（按 start_char 排序）
        """
        _start = time.time()
        try:
            if not llm_chapters or not self.client:
                logger.info("  [验证Agent] 无 LLM 客户端或无章节，直接返回原章节")
                self.metrics_collector.record_request(
                    endpoint="agent.verifier",
                    duration_ms=0,
                    status="success",
                    agent="verification_agent",
                    tokens=0,
                    session_id=session_id,
                )
                return llm_chapters

            logger.info(f"  [验证Agent] 开始 LLM 语义评审: {len(llm_chapters)} 个章节")

            # Step 1: 准备第一轮 LLM 章节的 JSON 描述
            chapters_desc = []
            for i, chap in enumerate(llm_chapters):
                start = int(chap.get('start_char', 0))
                end = int(chap.get('end_char', 0))
                # 截取章节内容做描述
                content_preview = full_text[start:min(end, start + 200)]
                chapters_desc.append({
                    'index': i,
                    'title': chap.get('title', ''),
                    'level': chap.get('level', 2),
                    'start_char': start,
                    'end_char': end,
                    'is_product': chap.get('is_product', False),
                    'content_preview': content_preview[:200],
                })

            # Step 2: 调用 LLM 做评审
            try:
                reviewed = self._call_llm_verify(chapters_desc, full_text, session_id=session_id)
            except Exception as e:
                logger.warning(f"  [验证Agent] LLM 评审调用失败: {e}，返回原章节")
                _duration_ms = (time.time() - _start) * 1000
                self.metrics_collector.record_request(
                    endpoint="agent.verifier",
                    duration_ms=_duration_ms,
                    status="error",
                    agent="verification_agent",
                    tokens=0,
                    session_id=session_id,
                )
                return llm_chapters

            # Step 3: 解析 LLM 评审结果
            final_chapters = self._parse_llm_review(reviewed, llm_chapters, full_text)

            # Step 4: 排序 & 边界修正（确保边界在行首）
            final_chapters = self._normalize(final_chapters, full_text)

            _duration_ms = (time.time() - _start) * 1000
            logger.info(f"  [验证Agent] 评审完成: {len(final_chapters)} 个章节, 耗时 {_duration_ms:.0f}ms")

            # ---- ops 监控：记录整体评审请求 ----
            self.metrics_collector.record_request(
                endpoint="agent.verifier",
                duration_ms=_duration_ms,
                status="success",
                agent="verification_agent",
                tokens=0,
                session_id=session_id,
            )
            return final_chapters
        except Exception as e:
            _duration_ms = (time.time() - _start) * 1000
            logger.error(f"  [验证Agent] 评审异常: {e}")
            self.metrics_collector.record_request(
                endpoint="agent.verifier",
                duration_ms=_duration_ms,
                status="error",
                agent="verification_agent",
                tokens=0,
                session_id=session_id,
            )
            return llm_chapters

    # ============================================================
    # LLM 调用
    # ============================================================
    def _call_llm_verify(
        self,
        chapters_desc: List[Dict],
        full_text: str,
        session_id: str = "indexing",
    ) -> Optional[Dict]:
        """调用 LLM 做章节评审（接入 ops 监控）"""

        chapters_json = json.dumps(chapters_desc, ensure_ascii=False, indent=2)

        # 文档内容截断（避免超 token）
        doc_for_verify = full_text
        # 如果文档过长（> 5 中文字符，LLM 可能出错，做长度限制
        if len(full_text) > 150000:
            # 取前 80000 + 后 70000 字符，保留完整文档结构
            doc_for_verify = full_text[:80000] + "\n...[中间部分省略]...\n" + full_text[-70000:]

        prompt = VERIFIER_USER_PROMPT.format(
            document_content=doc_for_verify,
            chapters_json=chapters_json,
        )

        _start = time.time()
        try:
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
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
                endpoint="agent.verifier.llm_review",
                duration_ms=_duration_ms,
                status="success",
                agent="verification_agent",
                tokens=_tokens,
                session_id=session_id,
            )

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"    LLM 评审输出: {result_text[:300]}...")

            # 解析 JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                logger.warning("    评审未检测到 JSON 格式")
                return None
            json_text = json_match.group()
            return json.loads(json_text)

        except json.JSONDecodeError as e:
            _duration_ms = (time.time() - _start) * 1000
            logger.warning(f"    评审 JSON 解析失败: {e}")
            self.metrics_collector.record_request(
                endpoint="agent.verifier.llm_review",
                duration_ms=_duration_ms,
                status="error",
                agent="verification_agent",
                tokens=0,
                session_id=session_id,
            )
            return None
        except Exception as e:
            _duration_ms = (time.time() - _start) * 1000
            logger.warning(f"    评审 LLM 调用异常: {e}")
            self.metrics_collector.record_request(
                endpoint="agent.verifier.llm_review",
                duration_ms=_duration_ms,
                status="error",
                agent="verification_agent",
                tokens=0,
                session_id=session_id,
            )
            return None

    # ============================================================
    # 解析 LLM 评审结果
    # ============================================================
    def _parse_llm_review(
        self,
        reviewed: Optional[Dict],
        llm_chapters: List[Dict],
        full_text: str,
    ) -> List[Dict]:
        """解析 LLM 评审结果，合并为最终章节列表"""

        if not reviewed or 'reviewed_chapters' not in reviewed or not reviewed['reviewed_chapters']:
            logger.warning("    评审结果为空，返回原章节")
            return llm_chapters

        final_chapters: List[Dict] = []
        doc_len = len(full_text)

        for item in reviewed['reviewed_chapters']:
            keep = item.get('keep', True)
            orig_idx = item.get('original_index', -1)

            if keep:
                # 保留原章节
                if 0 <= orig_idx < len(llm_chapters):
                    chap = llm_chapters[orig_idx]
                    final_chapters.append({
                        'title': chap.get('title', ''),
                        'level': int(chap.get('level', 2)),
                        'start_char': int(chap.get('start_char', 0)),
                        'end_char': int(chap.get('end_char', 0)),
                        'is_product': chap.get('is_product', False),
                        'source': 'original',
                    })
                    logger.debug(f"    ✓ 保留: {chap.get('title', '')[:40]}")
            else:
                # 使用 LLM 修正后的边界
                title = item.get('corrected_title') or item.get('title', '')
                start = int(item.get('start_char', 0))
                end = int(item.get('end_char', 0))
                level = int(item.get('level', 2))
                reason = item.get('reason', '')

                # 边界 sanity check
                if 0 <= start < end <= doc_len:
                    final_chapters.append({
                        'title': title,
                        'level': level,
                        'start_char': start,
                        'end_char': end,
                        'is_product': False,
                        'source': 'llm-refined',
                    })
                    logger.debug(f"    ✗ 修正: {title[:40]} [{start}:{end}] ({reason[:60]})")
                else:
                    # LLM 给的边界越界，保留原章节
                    if 0 <= orig_idx < len(llm_chapters):
                        chap = llm_chapters[orig_idx]
                        final_chapters.append({
                            'title': chap.get('title', ''),
                            'level': int(chap.get('level', 2)),
                            'start_char': int(chap.get('start_char', 0)),
                            'end_char': int(chap.get('end_char', 0)),
                            'is_product': chap.get('is_product', False),
                            'source': 'original',
                        })
                        logger.debug(f"    ⚠ 修正边界越界，保留原章节: {chap.get('title', '')[:40]}")

        # 如果评审结果比原章节少（LLM 可能合并了章节），但原章节没评审到的内容做补全检查
        # 这里简单处理：按 start_char 排序后的结果即为最终结果
        # 但要确保不丢失重要内容（如商品编号、规格等）

        return final_chapters

    # ============================================================
    # 最终规范化（确保边界在行首）
    # ============================================================
    def _normalize(self, chapters: List[Dict], full_text: str) -> List[Dict]:
        """规范化：按 start_char 排序，确保边界在行首"""
        if not chapters:
            return chapters

        doc_len = len(full_text)

        # 按 start_char 排序
        chapters.sort(key=lambda c: c.get('start_char', 0))

        # 确保边界在行首
        for chap in chapters:
            start = int(chap.get('start_char', 0))
            end = int(chap.get('end_char', 0))

            # 修正 start 到行首
            if 0 < start < doc_len and full_text[start - 1] != '\n':
                # 向后找最近的 \n
                newline_after = full_text.find('\n', start, min(start + 150, doc_len))
                if newline_after > 0:
                    chap['start_char'] = newline_after + 1
                else:
                    # 向前找
                    newline_before = full_text.rfind('\n', max(0, start - 150), start)
                    if newline_before >= 0 and isinstance(newline_before, int):
                        chap['start_char'] = newline_before + 1

            # 修正 end 到行尾
            if 0 < end < doc_len and full_text[end] != '\n':
                newline_after = full_text.find('\n', end, min(end + 150, doc_len))
                if newline_after > 0:
                    chap['end_char'] = newline_after
                else:
                    # 向前找
                    newline_before = full_text.rfind('\n', max(0, end - 150), end)
                    if newline_before >= 0 and isinstance(newline_before, int):
                        chap['end_char'] = newline_before

            # 确保 end > start
            if chap['end_char'] <= chap['start_char']:
                chap['end_char'] = min(doc_len, chap['start_char'] + 100)

        return chapters
