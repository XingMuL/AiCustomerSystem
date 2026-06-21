"""
路由智能体（Router Agent）

意图分类：将用户输入路由到对应的子智能体。
支持降级策略：Level 4 跳过、Level 2 跳过已熔断 Agent。
"""

import json
from loguru import logger
from openai import OpenAI

from backend.config import settings
from backend.models.state import Intent

# 意图分类 Prompt
ROUTER_SYSTEM_PROMPT = """你是一个智能客服系统的意图分类器。请将用户输入分类到以下意图之一：

【意图定义】
1. kb_qa - 知识库问答：用户询问产品、政策、流程等知识类问题
2. ticket - 工单处理：用户要投诉、退换货、报修、查订单、查进度等需要系统操作的请求
3. chitchat - 闲聊：打招呼、感谢、测试等非业务对话
4. unknown - 无法判断

【重要规则 - sub_intents】
- 当主意图为 "ticket"（退货、投诉、报修、查订单）时，sub_intents 必须为空数组 []，不要包含 "kb_qa"
- 当主意图为 "kb_qa" 时，sub_intents 可以包含 "ticket"（如果问题可能需要订单查询）
- 当主意图为 "chitchat" 时，sub_intents 必须为空数组 []

请返回 JSON 格式：
{
    "intent": "kb_qa",
    "confidence": 0.9,
    "reason": "简要说明分类理由",
    "sub_intents": ["kb_qa"]
}"""


class RouterAgent:
    """路由智能体：意图分类与分发"""

    def __init__(self):
        self._client: OpenAI = None
        self._fallback_used = False

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
        return self._client

    def classify(self, user_input: str, degradation_level: int = 0) -> dict:
        """
        分类用户意图

        Args:
            user_input: 用户输入
            degradation_level: 当前降级等级

        Returns:
            dict: {"intent": str, "confidence": float, "target_agents": list[str]}
        """
        # Level 4 全局故障：所有请求直接走兜底
        if degradation_level >= 4:
            logger.warning("[Router] Level 4 全局故障，所有请求走兜底")
            return {
                "intent": Intent.CHITCHAT.value,
                "confidence": 1.0,
                "target_agents": ["summary_agent"],
                "reason": "系统降级，直接兜底",
            }

        try:
            response = self.client.chat.completions.create(
                model=settings.router_model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                temperature=settings.router_temperature,
                max_tokens=512,
                timeout=settings.llm_timeout,
            )

            result_text = response.choices[0].message.content

            # 解析 JSON 结果
            json_match = __import__("re").search(
                r'\{[\s\S]*"intent"[\s\S]*\}', result_text
            )
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(result_text)

            intent = data.get("intent", Intent.UNKNOWN.value)
            confidence = float(data.get("confidence", 0.5))
            sub_intents = data.get("sub_intents", [intent])
            reason = data.get("reason", "")

            # 将意图映射到 Agent
            target_agents = self._intent_to_agents(intent, sub_intents)

            logger.info(f"[Router] 意图: {intent}, 置信度: {confidence:.2f}, 目标: {target_agents}")
            return {
                "intent": intent,
                "confidence": confidence,
                "target_agents": target_agents,
                "reason": reason,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"[Router] 分类失败: {e}")
            return {
                "intent": Intent.UNKNOWN.value,
                "confidence": 0.0,
                "target_agents": ["chitchat_agent", "summary_agent"],
                "reason": f"分类出错: {e}",
            }

    def _intent_to_agents(self, intent: str, sub_intents: list[str]) -> list[str]:
        """将意图映射到对应的 Agent 列表"""
        agent_map = {
            Intent.KB_QA.value: ["kb_qa_agent"],
            Intent.TICKET.value: ["ticket_agent"],
            Intent.CHITCHAT.value: ["chitchat_agent"],
        }

        # 始终包含 summary_agent
        agents = list(set(agent_map.get(intent, ["chitchat_agent"]) + ["summary_agent"]))

        # 如果有次要意图，也加入对应的 Agent（用于多 Agent 协作）
        for si in sub_intents:
            for a in agent_map.get(si, []):
                if a not in agents and a != "summary_agent":
                    agents.append(a)

        return agents