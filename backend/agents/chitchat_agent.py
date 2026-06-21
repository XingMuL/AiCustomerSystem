"""
闲聊智能体（Chitchat Agent）

处理用户的非业务对话（问候、感谢、测试等）。
支持降级策略：Level 4 返回固定兜底话术。
"""

from loguru import logger
from openai import OpenAI

from backend.config import settings

CHITCHAT_SYSTEM_PROMPT = """你是一个友好、热情的智能客服机器人。请用亲切自然的语气与用户对话。

要求：
1. 语气友好、亲和力强
2. 避免过于机械的回复
3. 可以适当表达同理心
4. 引导用户说出具体需求
5. 回复简洁，不超过 200 字"""

# Level 4 兜底话术
FALLBACK_CHITCHAT = (
    "感谢您的联系！系统当前遇到一些技术问题，暂时无法提供完整服务。\n\n"
    "如有紧急需求，请拨打客服热线获取即时帮助。给您带来的不便，敬请谅解！"
)


class ChitchatAgent:
    """闲聊智能体"""

    def __init__(self):
        self._client: OpenAI = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
        return self._client

    def chat(
        self,
        user_input: str,
        memory_context: str,
        degradation_level: int = 0,
        llm_available: bool = True,
    ) -> str:
        """
        闲聊对话

        Args:
            user_input: 用户输入
            memory_context: 记忆上下文
            degradation_level: 降级等级
            llm_available: LLM 是否可用

        Returns:
            str: 回复内容
        """
        # Level 4: 全局故障
        if degradation_level >= 4 or not llm_available:
            logger.warning(f"[Chitchat] Level 4 全局故障，返回兜底话术")
            return FALLBACK_CHITCHAT

        messages = [{"role": "system", "content": CHITCHAT_SYSTEM_PROMPT}]

        if memory_context:
            messages.append({"role": "system", "content": f"对话历史：\n{memory_context}"})

        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model=settings.agent_llm_model,
                messages=messages,
                temperature=settings.agent_llm_temperature,
                max_tokens=1024,
                timeout=settings.llm_timeout,
            )
            return {
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"[Chitchat] 生成失败: {e}")
            return {"response": FALLBACK_CHITCHAT, "tokens": 0}