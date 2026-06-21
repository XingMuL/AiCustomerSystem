"""
反思评判模块

对 Agent 输出进行质量评判，支持自动重试修正。
"""

from loguru import logger
from openai import OpenAI

from backend.config import settings
from backend.models.state import ReflectionResult

# 反思评判 Prompt
REFLECTION_SYSTEM_PROMPT = """你是一个智能客服质量评审专家。请对以下客服回复进行质量评判。

评判维度（每个维度 0-1 分）：
1. 准确性：回答是否正确，与知识库/事实是否一致
2. 完整性：是否涵盖了用户的全部问题
3. 流畅性：语言是否自然流畅，语气是否亲和
4. 安全性：是否避免了敏感/风险内容、不当承诺

请返回 JSON 格式：
{
    "score": 0.85,
    "is_acceptable": true,
    "issues": ["问题描述1", "问题描述2"],
    "retry_instruction": "修正时应注意...",
    "dimensions": {
        "accuracy": 0.9,
        "completeness": 0.8,
        "fluency": 0.9,
        "safety": 0.8
    }
}"""


class ReflectionJudge:
    """反思评判器：对 Agent 输出进行质量评判"""

    def __init__(self):
        self._client: OpenAI = None
        self._enabled = settings.enable_reflection

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
        return self._client

    def evaluate(self, agent_name: str, user_input: str, response: str) -> ReflectionResult:
        """
        评判 Agent 回复质量

        Args:
            agent_name: Agent 名称
            user_input: 用户原始输入
            response: Agent 生成的回复

        Returns:
            ReflectionResult: 反思评判结果
        """
        if not self._enabled:
            return ReflectionResult(score=1.0, is_acceptable=True)

        try:
            prompt = (
                f"Agent [{agent_name}] 对用户问题的回复：\n\n"
                f"用户问题：{user_input}\n\n"
                f"Agent 回复：{response}\n\n"
                "请评审判定。"
            )

            completion = self.client.chat.completions.create(
                model=settings.reflection_model,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=settings.llm_timeout,
            )

            result_text = completion.choices[0].message.content
            # 提取 JSON
            import json
            import re

            # 尝试提取 JSON 块
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            else:
                # 尝试找到 { } 包裹的 JSON
                json_match = re.search(r'\{[\s\S]*"score"[\s\S]*\}', result_text)
                if json_match:
                    result_text = json_match.group(0)

            data = json.loads(result_text.strip())

            return ReflectionResult(
                score=float(data.get("score", 0.5)),
                is_acceptable=data.get("is_acceptable", True),
                issues=data.get("issues", []),
                retry_instruction=data.get("retry_instruction", ""),
                dimensions=data.get("dimensions", {}),
            )

        except Exception as e:
            logger.error(f"[反思] 评判失败: {e}")
            return ReflectionResult(score=0.5, is_acceptable=True, issues=[str(e)])

    def should_retry(self, result: ReflectionResult, retry_count: int) -> bool:
        """
        判断是否需要重试

        Args:
            result: 反思评判结果
            retry_count: 当前已重试次数

        Returns:
            bool: 是否需要重试
        """
        if not self._enabled:
            return False
        if retry_count >= settings.reflection_max_retries:
            return False
        if result.score >= settings.reflection_min_score:
            return False
        return True