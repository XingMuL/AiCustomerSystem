"""
汇总智能体（Summary Agent）

将多个子智能体的结果汇总为最终回复。
支持降级策略：根据降级等级调整话术，L4 全局故障使用固定兜底话术。
"""

from loguru import logger
from openai import OpenAI

from backend.config import settings

SUMMARY_SYSTEM_PROMPT = """你是一个智能客服系统的汇总助手。请将多个客服智能体的结果整合为一段连贯、自然的回复。

【重要规则 - 禁止产生幻觉（请严格遵守）】
1. 绝对禁止编造、推断或补充任何未在输入数据中明确出现的信息
2. 只能基于各智能体实际返回的内容进行整理，不得发挥想象
3. 如果某个智能体没有返回某个信息，不要猜测或填补该信息
4. 禁止编造订单信息、价格、商品名称、物流信息、客服电话、政策条款等
5. 如果输入数据中缺少某项信息，直接省略该部分，不要用"可能"、"大概"、"应该"等词推断
6. 禁止添加任何未明确提供的联系方式、网址、时间、金额等具体数据
7. 禁止将某个智能体的结果解读为另一个智能体的结果
8. 如果某条信息的来源不明确，不要在回复中包含该信息

【处理规则】
1. 去掉重复内容
2. 按逻辑顺序排列（先问候/共情 → 知识回答 → 工单指引/退货信息 → 结束语）
3. 语气亲切、专业，使用中文回复
4. 如有多个 Agent 结果，自然地融合在一起
5. 回复要简洁，不超过 500 字

【退货信息处理规则】
如果 ticket_agent 返回了退货相关信息（如退货预览、退货结果）：
- 只能使用其中明确给出的数据（订单号、退款金额、退货原因、预计时间等）
- 禁止添加任何退货政策、流程说明，除非 ticket_agent 明确返回了这些内容
- 如果退货信息不完整，只展示已有的部分，不要填补缺失信息

【检查清单 - 在输出前确认】
- [ ] 我是否只使用了输入中明确出现的信息？
- [ ] 我是否编造了任何具体数据（金额、时间、联系方式、订单号等）？
- [ ] 我是否对不确定的信息进行了推断或猜测？
- [ ] 我是否添加了任何未在输入中出现的政策、流程、建议？

如果以上任何一项为"是"，请删除相应内容。"""

# L4 兜底话术模板（按情境分类）
FALLBACK_SCRIPTS = {
    "default": "抱歉，系统暂时繁忙，请您稍后再试。如有紧急帮助，请拨打客服热线。",
    "kb_qa": (
        "您好！非常抱歉，知识库服务暂时不可用。\n\n"
        "对于您的问题，建议您：\n"
        "1. 拨打客服热线获取即时解答\n"
        "2. 访问我们的官网帮助中心\n"
        "3. 稍后重新咨询，系统正在恢复中\n\n"
        "给您带来的不便，敬请谅解！"
    ),
    "ticket": (
        "您好！非常抱歉，工单系统暂时不可用。\n\n"
        "请您通过以下方式联系我们：\n"
        "1. 拨打客服热线人工创建工单\n"
        "2. 发送邮件至 support@example.com\n"
        "3. 稍后重新提交，系统正在恢复中\n\n"
        "我们会在系统恢复后第一时间处理您的请求！"
    ),
    "chitchat": (
        "感谢您的关注！由于系统维护升级，部分功能暂时受限。\n\n"
        "如需帮助，请拨打客服热线。给您带来的不便，敬请谅解！"
    ),
    "partial": (
        "您好！以下是部分处理结果（其他功能暂时不可用）：\n\n"
        "{partial_results}\n\n"
        "如需完整服务，请稍后重试或拨打客服热线。"
    ),
}


class SummaryAgent:
    """汇总智能体"""

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

    def summarize(
        self,
        user_input: str,
        agent_results: dict,
        memory_context: str,
        degradation_level: int = 0,
        llm_available: bool = True,
        intent: str = "",
        cleaned_context: str = "",
    ) -> str:
        """
        汇总多个 Agent 的结果

        Args:
            user_input: 用户输入
            agent_results: 各 Agent 的结果 {"agent_name": {"response": "..."}}
            memory_context: 记忆上下文
            degradation_level: 降级等级
            llm_available: LLM 是否可用
            intent: 用户意图
            cleaned_context: 清洗后的知识库上下文（用于增强回答准确性）

        Returns:
            str: 最终汇总回复
        """
        # L4 全局故障：直接返回兜底话术
        if degradation_level >= 4 or not llm_available:
            logger.warning("[Summary] L4 全局故障，使用兜底话术")
            return {"response": self._get_fallback(intent), "tokens": 0}

        # L2 局部故障：部分 Agent 结果缺失
        if degradation_level >= 2:
            logger.warning(f"[Summary] Level {degradation_level}，合并部分结果")

        # === 收集 Agent 回复，带优先级判断 ===
        # 策略：
        # 1. ticket_agent 的结果具有最高优先级（业务操作结果）
        # 2. 如果 ticket_agent 返回了有效内容，直接使用其回复
        # 3. kb_qa_agent 的"知识库无相关信息"类回复视为无效回复，不参与汇总
        # 4. 只有当 ticket_agent 无结果时，才考虑 kb_qa_agent 的回复

        ticket_response = None
        kb_qa_response = None
        other_responses = []

        for agent_name, result in agent_results.items():
            if not result or not result.get("response"):
                continue

            response_text = result["response"].strip()

            # 过滤低质量 KB-QA 回复
            low_quality_patterns = [
                "知识库中暂无相关信息",
                "暂时无法提供该问题的准确信息",
                "知识库不可用",
                "建议您拨打客服",
                "无法从知识库中获取相关信息",
                "稍后重试",
            ]

            # ★ 智能过滤：仅在回复"主要是"拒绝/兜底信息时才过滤
            # 如果回复中有实质性内容（如部分回答了问题），即使包含兜底话术也不过滤
            matched_pattern = next((p for p in low_quality_patterns if p in response_text), None)
            if matched_pattern:
                # 回复很短（< 80 字符），或回复以兜底话术开头 → 纯拒绝回复，过滤
                if len(response_text) < 80 or response_text.startswith(matched_pattern):
                    is_low_quality = True
                else:
                    # 回复较长且不以兜底开头 → 有实质性内容，保留
                    is_low_quality = False
            else:
                is_low_quality = False

            if agent_name == "ticket_agent":
                ticket_response = response_text
                logger.info(f"[Summary] 检测到 ticket_agent 回复（优先级最高），长度: {len(response_text)}")
            elif agent_name == "kb_qa_agent":
                if is_low_quality:
                    logger.warning(f"[Summary] kb_qa_agent 回复为低质量信息，已过滤: {response_text[:50]}")
                    continue
                kb_qa_response = response_text
                logger.info(f"[Summary] 检测到 kb_qa_agent 回复，长度: {len(response_text)}")
            else:
                other_responses.append(response_text)

        # === 快速路径：ticket_agent 有有效回复时直接返回 ===
        if ticket_response:
            logger.info("[Summary] 使用 ticket_agent 结果（优先级最高），跳过汇总")
            return {"response": ticket_response, "tokens": 0}

        # 构建最终需要汇总的回复部分
        response_parts = []
        if kb_qa_response:
            response_parts.append(f"【知识库回答】\n{kb_qa_response}")
        for resp in other_responses:
            response_parts.append(f"【对话回复】\n{resp}")

        if not response_parts:
            return {"response": self._get_fallback(intent), "tokens": 0}

        # 只有一个 Agent 的结果，直接返回
        if len(response_parts) == 1:
            text = response_parts[0].split("\n", 1)[1] if "\n" in response_parts[0] else response_parts[0]
            return {"response": text, "tokens": 0}

        combined = "\n\n---\n\n".join(response_parts)

        # 多个 Agent 结果，用 LLM 汇总
        try:
            messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]

            if memory_context:
                messages.append({"role": "system", "content": f"对话历史：\n{memory_context}"})

            # 如果提供了清洗后的知识库上下文，作为参考依据追加
            context_section = ""
            if cleaned_context:
                context_section = f"【清洗后的知识库上下文（仅包含与问题相关的内容）】\n{cleaned_context[:2000]}\n\n"

            messages.append({
                "role": "user",
                "content": (
                    f"用户问题：{user_input}\n\n"
                    f"{context_section}"
                    f"各智能体回复：\n{combined}\n\n"
                    "请汇总为一段最终回复。重点关注清洗后的知识库上下文中的关键信息。"
                ),
            })

            response = self.client.chat.completions.create(
                model=settings.agent_llm_model,
                messages=messages,
                temperature=settings.agent_llm_temperature,
                max_tokens=settings.agent_llm_max_tokens,
                timeout=settings.llm_timeout,
            )
            return {
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"[Summary] 汇总失败: {e}")
            # 汇总失败时，使用简单的字符串拼接
            return {
                "response": "\n\n".join(p.split("\n", 1)[1] if "\n" in p else p for p in response_parts),
                "tokens": 0,
            }

    def _get_fallback(self, intent: str) -> str:
        """根据意图获取兜底话术"""
        if intent in FALLBACK_SCRIPTS:
            return FALLBACK_SCRIPTS[intent]
        return FALLBACK_SCRIPTS["default"]