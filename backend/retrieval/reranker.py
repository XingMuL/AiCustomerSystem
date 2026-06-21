"""
LLM Rerank 重排序模块：对候选子块进行精细相关性排序。

使用 LLM 对每个候选子块与查询的相关性进行打分，筛选最相关的 top_k 个子块。

Rerank 策略：
  1. 批量提交候选子块给 LLM 评分
  2. 按相关性得分降序排序
  3. 取 top_k 返回
"""

import json
from typing import Optional

from openai import OpenAI
from loguru import logger

from backend.config import settings


_RERANK_PROMPT = """你是一个文档相关性评估助手。请评估以下文档片段与用户查询的相关性。

用户查询：{query}

文档片段：
{chunks}

请以JSON格式返回每个文档片段的相关性评分（0-10分，10分表示完全相关，0分表示完全不相关）。
只输出JSON，格式如下：
[{{"score": 分数, "index": 片段编号}}, ...]

相关性评分标准：
- 9-10: 直接回答查询，包含明确的关键信息
- 7-8: 高度相关，包含了查询所需的大部分信息
- 5-6: 部分相关，含有关键词但信息不完整
- 3-4: 间接相关，话题接近但未直接回答
- 1-2: 弱相关，仅有少量关键词匹配
- 0: 完全不相关

只输出JSON数组，不要任何其他内容。"""


class LLMReranker:
    """
    基于 LLM 的重排序器

    对候选子块进行批量相关行打分，筛选最相关的 top_k 个。
    """

    def __init__(self):
        # 优先使用 llm_api_key/llm_api_base（项目统一配置），fallback 到 deepseek_api_key
        self.client = OpenAI(
            api_key=getattr(settings, "llm_api_key", settings.deepseek_api_key),
            base_url=getattr(settings, "llm_api_base", settings.deepseek_base_url),
        )
        self.model = getattr(settings, "rerank_model", "deepseek-chat")
        self.batch_size = 15  # 每批最多评估 15 个片段
        self.enabled = getattr(settings, "enable_rerank", True)

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ) -> list:
        """
        重排序候选子块

        Args:
            query: 查询字符串
            candidates: RetrievalResult 列表
            top_k: 返回数量

        Returns:
            按相关性降序排列的 RetrievalResult 列表
        """
        if not self.enabled or not candidates:
            return candidates[:top_k]

        if len(candidates) <= top_k:
            return candidates

        try:
            scores = self._score_candidates(query, candidates)
            # 按得分降序
            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            # 取 top_k
            result = []
            for cand, score in scored[:top_k]:
                cand.score = score
                result.append(cand)
            return result

        except Exception as e:
            logger.warning(f"Rerank 失败，降级为原始排序: {e}")
            return candidates[:top_k]

    def _score_candidates(self, query: str, candidates: list) -> list[float]:
        """批量打分"""
        scores = [0.0] * len(candidates)

        for batch_start in range(0, len(candidates), self.batch_size):
            batch = candidates[batch_start:batch_start + self.batch_size]
            batch_scores = self._score_batch(query, batch)

            for i, s in enumerate(batch_scores):
                idx = batch_start + i
                if idx < len(scores):
                    scores[idx] = s

        return scores

    def _score_batch(
        self,
        query: str,
        batch: list,
    ) -> list[float]:
        """对一批候选子块打分"""
        # 构建片段文本
        chunks_text = ""
        for i, cand in enumerate(batch):
            content = cand.content[:500].replace("\n", " ")  # 截断子块内容
            chunks_text += f"[{i}] {content}\n\n"

        prompt = _RERANK_PROMPT.format(query=query, chunks=chunks_text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )

        output = response.choices[0].message.content.strip()

        # 提取 JSON 数组
        try:
            # 尝试直接解析
            ratings = json.loads(output)
        except json.JSONDecodeError:
            # 尝试从回复中提取 JSON 数组
            import re
            match = re.search(r'\[.*\]', output, re.DOTALL)
            if match:
                try:
                    ratings = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"无法解析 Rerank 结果: {output[:200]}")
                    return [0.0] * len(batch)
            else:
                logger.warning(f"Rerank 结果中无 JSON: {output[:200]}")
                return [0.0] * len(batch)

        # 映射分数到原始位置
        index_to_score = {}
        for item in ratings:
            idx = item.get("index", -1)
            score = item.get("score", 0)
            if isinstance(score, (int, float)):
                index_to_score[idx] = float(score) / 10.0  # 归一化到 0-1

        return [index_to_score.get(i, 0.0) for i in range(len(batch))]