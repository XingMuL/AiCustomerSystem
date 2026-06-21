"""
状态图编排引擎

构建智能客服多智能体协作工作流：
1. 输入清洗 → 2. 降级状态注入 → 3. 记忆融合 → 4. 路由分发 →
5. 子 Agent 并行执行 → 6. 反思评判 → 7. 结果汇总 → 8. 记忆更新

集成 Gossip 协议 + 向量时钟，支持流式输出。
"""

import time
from typing import Generator, Optional
from loguru import logger

from backend.config import settings
from backend.models.state import AgentState, Intent
from backend.preprocessor.input_cleaner import InputCleaner
from backend.agents.router_agent import RouterAgent
from backend.agents.kb_qa_agent import KBQAAgent
from backend.agents.ticket_agent import TicketAgent
from backend.agents.chitchat_agent import ChitchatAgent
from backend.agents.summary_agent import SummaryAgent
from backend.agents.memory_merge_agent import MemoryMergeAgent
from backend.agents.structuring_agent import DocumentStructuringAgent
from backend.agents.structuring_verifier import ChapterBoundaryVerificationAgent
from backend.memory.memory_manager import MemoryManager
from backend.memory.gossip_coordinator import get_gossip_coordinator
from backend.reflection.reflection import ReflectionJudge
from backend.degradation.degradation_manager import get_degradation_manager
from backend.ops.metrics_collector import get_metrics_collector
from backend.retrieval.context_cleaner import get_context_cleaner


class StateGraph:
    """多智能体状态图编排引擎（支持流式输出 + Gossip 协调）"""

    def __init__(self):
        # 初始化各组件
        self.input_cleaner = InputCleaner()
        self.router = RouterAgent()
        self.kb_qa_agent = KBQAAgent()
        self.ticket_agent = TicketAgent()
        self.chitchat_agent = ChitchatAgent()
        self.summary_agent = SummaryAgent()
        self.memory_merge_agent = MemoryMergeAgent()
        self.structuring_agent = DocumentStructuringAgent()
        self.verification_agent = ChapterBoundaryVerificationAgent()
        self.memory_manager = MemoryManager()
        self.reflection_judge = ReflectionJudge()
        self.degradation_manager = get_degradation_manager()
        self.gossip_coordinator = get_gossip_coordinator()
        self.metrics_collector = get_metrics_collector()

        # 注册 Agent 到 Gossip 协调器
        self._agent_map = {
            "kb_qa_agent": self.kb_qa_agent,
            "ticket_agent": self.ticket_agent,
            "chitchat_agent": self.chitchat_agent,
            "summary_agent": self.summary_agent,
            "memory_merge_agent": self.memory_merge_agent,
            "router_agent": self.router,
            "structuring_agent": self.structuring_agent,
            "verification_agent": self.verification_agent,
        }
        self._agent_roles = {
            "kb_qa_agent": "知识库问答",
            "ticket_agent": "工单处理",
            "chitchat_agent": "闲聊对话",
            "summary_agent": "结果汇总",
            "memory_merge_agent": "记忆融合",
            "router_agent": "路由分发",
            "structuring_agent": "文档结构化",
            "verification_agent": "章节边界验证",
        }
        for name, inst in self._agent_map.items():
            self.gossip_coordinator.register_agent(name, inst, self._agent_roles.get(name, ""))

        # 启动 Gossip 同步
        self.gossip_coordinator.start_sync_loop(interval=30)

        # 会话活跃时间追踪
        self._session_activity: dict[str, float] = {}
        # 会话元数据追踪
        self._session_metadata: dict[str, dict] = {}
        # 会话 Token 消耗追踪
        self._session_tokens: dict[str, int] = {}
        # 会话消息计数
        self._session_msg_count: dict[str, int] = {}

    def process(self, raw_input: str, session_id: str = "default", user_id: int = None) -> str:
        """处理用户输入（非流式）"""
        result = ""
        for chunk in self.process_stream(raw_input, session_id, user_id):
            result = chunk
        return result

    def process_stream(self, raw_input: str, session_id: str = "default", user_id: int = None) -> Generator[str, None, None]:
        """
        流式处理用户输入

        Yields:
            每个处理步骤的中间文本或最终回复
        """
        self._session_activity[session_id] = time.time()
        _pipeline_start = time.time()
        _pipeline_status = "success"
        _primary_agent = "router_agent"

        # ========== Step 0: 初始化状态 ==========
        state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "node_id": settings.node_id,
            "raw_input": raw_input,
            "messages": [],
            "agent_results": {},
            "rag_docs": [],
            "cleaned_rag_docs": [],
            "cleaned_context": "",
            "current_turn": len(self.memory_manager.get_history(session_id)) // 2,
            "final_response": "",
            "vector_clock": self.gossip_coordinator.get_global_clock(),
            "ticket": None,
            "metadata": {},
            "degradation_level": 0,
            "rag_available": True,
            "llm_available": True,
            "circuit_breakers": {},
            "reflection_results": {},
        }

        # ========== Step 1: 输入清洗 ==========
        yield "🔍 正在分析输入..."
        state = self._clean_input(state)
        if state["cleaned_input"] == "":
            yield "您好，请问有什么可以帮您的？"
            return

        # ========== Step 2: 降级状态注入 ==========
        state = self._inject_degradation(state)

        # ========== Step 3: 记忆融合 ==========
        state = self._merge_memory(state)

        # ========== Step 4: 路由分发 ==========
        yield "🧠 正在理解意图..."
        state = self._route(state)

        intent_labels = {
            "kb_qa": "知识库问答",
            "ticket": "工单处理",
            "chitchat": "日常闲聊",
            "unknown": "通用回复",
        }
        intent_label = intent_labels.get(state.get("intent", ""), "通用回复")
        _primary_agent = state["target_agents"][0] if state.get("target_agents") else "router_agent"
        yield f"📌 识别意图: {intent_label}"

        # ========== Step 5: 子 Agent 执行 ==========
        state = self._execute_agents(state)

        # ========== Step 6: 反思评判 ==========
        state = self._reflect(state)

        # ========== Step 7: 结果汇总 ==========
        yield "📝 正在汇总结果..."
        state = self._summarize(state)

        # ========== Step 8: 记忆更新 ==========
        self._update_memory(state)

        # 流式输出最终回复
        final_response = state["final_response"]
        if final_response:
            # 逐字输出打字机效果
            for i, char in enumerate(final_response):
                yield final_response[:i + 1]
        else:
            yield "抱歉，我暂时无法处理您的请求，请稍后再试。"

        # 更新会话活跃时间
        self._session_activity[session_id] = time.time()

        # 记录会话元数据
        total_tokens = sum(
            r.get("tokens", 0) 
            for r in state.get("agent_results", {}).values() 
            if isinstance(r, dict)
        )
        self._session_tokens[session_id] = self._session_tokens.get(session_id, 0) + total_tokens
        self._session_msg_count[session_id] = self._session_msg_count.get(session_id, 0) + 1
        self._session_metadata[session_id] = {
            "last_intent": state.get("intent", "unknown"),
            "last_agent": ", ".join(state.get("target_agents", [])),
            "degradation_level": state.get("degradation_level", 0),
            "total_tokens": self._session_tokens[session_id],
            "msg_count": self._session_msg_count[session_id],
        }

        # ========== Step 9: 记录性能指标（供监控页使用） ==========
        _pipeline_duration = (time.time() - _pipeline_start) * 1000

        # 记录整体请求指标（全部使用英文内部名称）
        if not state.get("final_response") or "抱歉，系统正在处理您的请求" in state.get("final_response", ""):
            _pipeline_status = "error"
        self.metrics_collector.record_request(
            endpoint="chat",
            duration_ms=_pipeline_duration,
            status=_pipeline_status,
            agent=_primary_agent,
            tokens=total_tokens,
            session_id=session_id,
        )

        # 记录每个子 Agent 的独立指标（用于 Agent 性能明细表，全部使用英文内部名称）
        for agent_name, agent_result in state.get("agent_results", {}).items():
            if isinstance(agent_result, dict):
                agent_status = "error" if agent_result.get("status") == "error" else "success"
                self.metrics_collector.record_request(
                    endpoint=f"agent.{agent_name}",
                    duration_ms=agent_result.get("duration_ms", 100),
                    status=agent_status,
                    agent=agent_name,
                    tokens=agent_result.get("tokens", 0),
                    session_id=session_id,
                )

        # 记录路由 Agent
        route_state = state.get("metadata", {})
        if route_state.get("route_duration"):
            self.metrics_collector.record_request(
                endpoint="agent.router",
                duration_ms=route_state["route_duration"],
                status="success",
                agent="router_agent",
                tokens=route_state.get("route_tokens", 0),
                session_id=session_id,
            )

    # ------------- 各步骤实现 -------------

    def _clean_input(self, state: AgentState) -> AgentState:
        """Step 1: 输入清洗"""
        result = self.input_cleaner.clean(
            state["raw_input"], state["session_id"]
        )
        state["cleaned_input"] = result["cleaned"]
        state["input_tokens"] = result["tokens"]
        logger.info(f"[Pipeline] 输入清洗完成, tokens={result['tokens']}")

        if result.get("is_spam"):
            state["final_response"] = (
                "检测到重复消息。如需帮助，请描述您的具体问题，我会尽力协助您！"
            )
        return state

    def _inject_degradation(self, state: AgentState) -> AgentState:
        """Step 2: 降级状态注入 + Gossip 同步"""
        state = self.degradation_manager.apply_snapshot_to_state(state)
        level = state["degradation_level"]

        # 通过 Gossip 传播降级状态
        self.gossip_coordinator.prepare_agent_gossip("degradation", {
            "level": level,
            "llm_available": state["llm_available"],
            "rag_available": state["rag_available"],
        })

        logger.info(
            f"[Pipeline] 降级状态: Level {level}, "
            f"LLM={state['llm_available']}, RAG={state['rag_available']}"
        )
        return state

    def _merge_memory(self, state: AgentState) -> AgentState:
        """Step 3: 记忆融合"""
        memory_context = self.memory_manager.merge_with_rag(
            state["session_id"], state["rag_docs"]
        )
        state["memory_context"] = memory_context

        # 更新向量时钟
        self.gossip_coordinator.tick_agent_clock("memory_merge")
        return state

    def _route(self, state: AgentState) -> AgentState:
        """Step 4: 路由分发"""
        if state["final_response"]:
            return state

        route_start = time.time()
        route_result = self.router.classify(
            state["cleaned_input"],
            degradation_level=state["degradation_level"],
        )
        route_duration = (time.time() - route_start) * 1000

        state["intent"] = route_result["intent"]
        state["intent_confidence"] = route_result["confidence"]
        state["target_agents"] = route_result["target_agents"]

        # 将路由指标写入 state.metadata，供后续 record_request 使用
        state["metadata"]["route_duration"] = route_duration
        state["metadata"]["route_tokens"] = route_result.get("tokens", 0)

        self.degradation_manager.record_agent_success("router_agent")
        self.gossip_coordinator.finalize_agent_call(
            "router_agent",
            token_usage=route_result.get("tokens", 0),
            duration_ms=route_duration,
        )
        return state

    def _execute_agents(self, state: AgentState) -> AgentState:
        """Step 5: 子 Agent 执行（降级感知 + Gossip 协调 + 向量时钟冲突检测）"""
        if state["final_response"]:
            return state

        degradation_level = state["degradation_level"]
        target_agents = state["target_agents"]
        failed_agents = self.degradation_manager.get_failed_agents()

        # RAG 检索：如果目标包含 kb_qa_agent，先检索知识库
        if "kb_qa_agent" in target_agents and state["cleaned_input"]:
            try:
                from backend.pipeline.rag_pipeline import RAGPipeline
                rag_pipeline = RAGPipeline()
                rag_result = rag_pipeline.retrieve(state["cleaned_input"])
                state["rag_docs"] = rag_result.get("child_hits", []) if isinstance(rag_result, dict) else []
                context = rag_result.get("context", "") if isinstance(rag_result, dict) else ""
                logger.info(
                    f"[Pipeline] RAG检索完成: {len(state['rag_docs'])} 子块, "
                    f"{len(context)} 字符上下文"
                )
                # 调试：打印传给 kb_qa_agent 的文档摘要
                for i, doc in enumerate(state["rag_docs"]):
                    content_preview = doc.get("content", "")[:100] if isinstance(doc, dict) else ""
                    logger.info(
                        f"  RAG文档[{i}]: chapter={doc.get('chapter_title', 'N/A') if isinstance(doc, dict) else 'N/A'}, "
                        f"score={doc.get('score', 0) if isinstance(doc, dict) else 0:.4f}, "
                        f"content_preview={content_preview}..."
                    )
            except Exception as e:
                logger.error(f"[Pipeline] RAG检索失败: {e}")
                state["rag_docs"] = []

        # 上下文清洗：用 DeepSeek 过滤检索结果中与问题无关的语义信息
        # 清洗后的上下文将用于 kb_qa_agent 回答和 context_relevance 评估
        state["cleaned_rag_docs"] = state["rag_docs"]
        state["cleaned_context"] = ""
        if state["rag_docs"] and state["cleaned_input"]:
            try:
                # ★ 收集所有上下文：父块完整上下文 + 子块内容
                raw_contexts = []

                # 1. 父块上下文（完整章节内容，包含价格等关键信息）
                if context:
                    raw_contexts.append(str(context))

                # 2. 子块内容（精炼的检索命中片段）
                for doc in state["rag_docs"]:
                    ctx = doc.get("content", "") if isinstance(doc, dict) else getattr(doc, "content", "")
                    if ctx:
                        raw_contexts.append(str(ctx))

                if raw_contexts:
                    cleaner = get_context_cleaner()
                    cleaned_contexts, cleaned_text = cleaner.clean(
                        user_query=state["cleaned_input"],
                        contexts=raw_contexts,
                    )
                    if cleaned_text:
                        # ★ 清洗后的上下文（父块+子块按问题语义清洗）作为 kb_qa_agent 的知识来源
                        # 创建单个文档，包含所有清洗后的相关内容
                        cleaned_docs = [{
                            "content": cleaned_text,
                            "chapter_title": "检索上下文（按问题清洗）",
                            "score": 1.0,
                            "cleaned": True,
                        }]
                        state["cleaned_rag_docs"] = cleaned_docs
                        state["cleaned_context"] = cleaned_text
                        logger.info(
                            f"[Pipeline] 上下文清洗完成: {len(raw_contexts)} 片段（含父块） → "
                            f"清洗后 {len(cleaned_text)} 字符"
                        )
                    else:
                        state["cleaned_rag_docs"] = state["rag_docs"]
                        state["cleaned_context"] = ""
            except Exception as e:
                logger.warning(f"[Pipeline] 上下文清洗失败，使用原始上下文: {e}")
                state["cleaned_rag_docs"] = state["rag_docs"]
                state["cleaned_context"] = ""

        # Gossip 协调：为并发 Agent 调用准备上下文（含向量时钟）
        call_contexts = self.gossip_coordinator.coordinate_agent_calls(target_agents)

        for agent_name in target_agents:
            # L2: 跳过已熔断的 Agent
            if agent_name in failed_agents:
                logger.warning(f"[Pipeline] 跳过已熔断 Agent: {agent_name}")
                state["agent_results"][agent_name] = {
                    "response": f"({agent_name} 暂时不可用)",
                    "status": "circuit_open",
                }
                continue

            # 检查熔断器
            if not self.degradation_manager.allow_agent(agent_name):
                logger.warning(f"[Pipeline] 熔断器拒绝 Agent: {agent_name}")
                state["agent_results"][agent_name] = {
                    "response": f"({agent_name} 暂时不可用)",
                    "status": "circuit_open",
                }
                continue

            try:
                agent_start = time.time()
                result = self._run_agent(agent_name, state, degradation_level)
                agent_duration = (time.time() - agent_start) * 1000

                # 将执行时间写入 result，供后续性能指标记录
                if isinstance(result, dict):
                    result["duration_ms"] = agent_duration
                state["agent_results"][agent_name] = result
                self.degradation_manager.record_agent_success(agent_name)

                # === 权限错误快速路径：ticket_agent 检测到权限错误时，
                #     直接设置 final_response，跳过后续所有 agent 和 summary
                if agent_name == "ticket_agent" and result.get("permission_denied"):
                    logger.warning(
                        f"[Pipeline] ticket_agent 检测到权限错误，直接返回用户提示: "
                        f"{result.get('response', '')[:80]}"
                    )
                    state["final_response"] = result["response"]
                    state["agent_results"]["_permission_denied"] = {
                        "permission_denied": True,
                        "source": agent_name,
                    }
                    return state

                # Gossip 完成回调
                self.gossip_coordinator.finalize_agent_call(
                    agent_name,
                    token_usage=result.get("tokens", 0),
                    duration_ms=agent_duration,
                )
                self.gossip_coordinator.heartbeat_agent(agent_name)
            except Exception as e:
                logger.error(f"[Pipeline] Agent {agent_name} 执行失败: {e}")
                state["agent_results"][agent_name] = {
                    "response": f"({agent_name} 执行出错)",
                    "status": "error",
                    "error": str(e),
                }
                self.degradation_manager.record_agent_failure(agent_name)

        # 向量时钟冲突检测与解决
        if len(target_agents) > 1:
            resolved = self.gossip_coordinator.resolve_agent_results(
                state["agent_results"], call_contexts
            )
            if len(resolved) != len(state["agent_results"]):
                logger.info(
                    f"[Pipeline] 向量时钟冲突解决: 将 {len(state['agent_results'])} 个结果合并为 {len(resolved)} 个"
                )
                state["agent_results"] = resolved

        return state

    def _run_agent(self, agent_name: str, state: AgentState, degradation_level: int) -> dict:
        """执行单个 Agent，返回统一格式 {"response": str, "tokens": int, "status": str}"""
        user_input = state["cleaned_input"]
        memory_context = state["memory_context"]

        if agent_name == "kb_qa_agent":
            result = self.kb_qa_agent.answer(
                user_input=user_input,
                memory_context=memory_context,
                rag_docs=state.get("cleaned_rag_docs", state["rag_docs"]),
                degradation_level=degradation_level,
                rag_available=state["rag_available"],
            )
            # kb_qa_agent now returns {"response": str, "tokens": int}
            return {"response": result["response"], "tokens": result.get("tokens", 0), "status": "ok"}

        elif agent_name == "ticket_agent":
            result = self.ticket_agent.process(
                user_input=user_input,
                memory_context=memory_context,
                user_id=state.get("user_id"),
                degradation_level=degradation_level,
                llm_available=state["llm_available"],
            )
            if result.get("ticket"):
                state["ticket"] = result["ticket"]
            agent_result = {
                "response": result["response"],
                "tokens": result.get("tokens", 0),
                "status": "ok",
            }
            if result.get("permission_denied"):
                agent_result["permission_denied"] = True
                logger.warning(
                    f"[Pipeline] ticket_agent 返回权限拒绝信号，将透传到 state_graph"
                )
            return agent_result

        elif agent_name == "chitchat_agent":
            result = self.chitchat_agent.chat(
                user_input=user_input,
                memory_context=memory_context,
                degradation_level=degradation_level,
                llm_available=state["llm_available"],
            )
            return {"response": result["response"], "tokens": result.get("tokens", 0), "status": "ok"}

        elif agent_name == "summary_agent":
            result = self.summary_agent.summarize(
                user_input=user_input,
                agent_results=state["agent_results"],
                memory_context=memory_context,
                degradation_level=degradation_level,
                llm_available=state["llm_available"],
                intent=state.get("intent", ""),
                cleaned_context=state.get("cleaned_context", ""),
            )
            return {"response": result["response"], "tokens": result.get("tokens", 0), "status": "ok"}

        elif agent_name == "structuring_agent":
            result = self.structuring_agent.structure(
                raw_markdown=state.get("cleaned_input", ""),
                session_id=state.get("session_id", "indexing"),
            )
            return {"response": str(result), "tokens": 0, "status": "ok"}

        elif agent_name == "verification_agent":
            chapters = state.get("rag_docs", [])
            result = self.verification_agent.verify_and_refine(
                llm_chapters=chapters,
                full_text=state.get("cleaned_input", ""),
                session_id=state.get("session_id", "indexing"),
            )
            return {"response": str(result), "tokens": 0, "status": "ok"}

        else:
            return {"response": f"Unknown agent: {agent_name}", "tokens": 0, "status": "unknown"}

    def _reflect(self, state: AgentState) -> AgentState:
        """Step 6: 反思评判"""
        if state["final_response"]:
            return state

        if not settings.enable_reflection:
            return state

        for agent_name, result in state["agent_results"].items():
            if agent_name == "summary_agent":
                continue

            response = result.get("response", "")
            if not response:
                continue

            judge_result = self.reflection_judge.evaluate(
                agent_name,
                state["cleaned_input"],
                response,
            )
            state["reflection_results"][agent_name] = judge_result

            retry_count = 0
            while self.reflection_judge.should_retry(judge_result, retry_count):
                logger.warning(
                    f"[反思] Agent {agent_name} 质量不足 "
                    f"(score={judge_result.score:.2f})，重试 {retry_count + 1}"
                )
                try:
                    new_result = self._run_agent(
                        agent_name, state, state["degradation_level"]
                    )
                    state["agent_results"][agent_name] = new_result

                    judge_result = self.reflection_judge.evaluate(
                        agent_name,
                        state["cleaned_input"],
                        new_result.get("response", ""),
                    )
                    state["reflection_results"][agent_name] = judge_result

                    # Gossip 记录反思重试
                    self.gossip_coordinator.tick_agent_clock(f"{agent_name}:retry")
                except Exception as e:
                    logger.error(f"[反思] 重试失败: {e}")
                    break

                retry_count += 1

        return state

    def _summarize(self, state: AgentState) -> AgentState:
        """Step 7: 结果汇总"""
        if state["final_response"]:
            return state

        try:
            summary_start = time.time()
            result = self.summary_agent.summarize(
                user_input=state["cleaned_input"],
                agent_results=state["agent_results"],
                memory_context=state["memory_context"],
                degradation_level=state["degradation_level"],
                llm_available=state["llm_available"],
                intent=state.get("intent", ""),
                cleaned_context=state.get("cleaned_context", ""),
            )
            summary_duration = (time.time() - summary_start) * 1000
            state["final_response"] = result.get("response", "抱歉，系统正在处理您的请求，请稍后再试。")
            self.gossip_coordinator.finalize_agent_call(
                "summary_agent",
                token_usage=result.get("tokens", 0),
                duration_ms=summary_duration,
            )

        except Exception as e:
            logger.error(f"[Pipeline] 汇总失败: {e}")
            state["final_response"] = "抱歉，系统正在处理您的请求，请稍后再试。"

        return state

    def _update_memory(self, state: AgentState):
        """Step 8: 记忆更新"""
        self.memory_manager.add_turn(
            state["session_id"],
            state["raw_input"],
            state["final_response"],
        )

        # 更新 Gossip 向量时钟
        self.gossip_coordinator.tick_agent_clock("memory_update")
        self._session_activity[state["session_id"]] = time.time()

    # ============ 会话管理 ============

    def get_active_sessions(self) -> list[dict]:
        """获取所有活跃会话（含元数据）"""
        now = time.time()
        sessions = []
        for session_id, last_activity in self._session_activity.items():
            idle_seconds = now - last_activity
            meta = self._session_metadata.get(session_id, {})
            sessions.append({
                "id": session_id,
                "user": session_id[:8] if len(session_id) > 8 else session_id,
                "last_active": time.strftime("%H:%M:%S", time.localtime(last_activity)),
                "idle_time": self._format_idle_time(idle_seconds),
                "agent": meta.get("last_agent", "多智能体"),
                "intent": meta.get("last_intent", "unknown"),
                "message_count": meta.get("msg_count", 0),
                "total_tokens": meta.get("total_tokens", 0),
                "degradation_level": meta.get("degradation_level", 0),
                "is_dead": idle_seconds > 300,  # 5分钟无活动视为僵死
            })
        sessions.sort(key=lambda s: s["last_active"], reverse=True)
        return sessions

    def kill_session(self, session_id: str) -> bool:
        """终止会话"""
        if session_id in self._session_activity:
            del self._session_activity[session_id]
            self._session_metadata.pop(session_id, None)
            self._session_tokens.pop(session_id, None)
            self._session_msg_count.pop(session_id, None)
            self.memory_manager.clear_session(session_id)
            return True
        return False

    def _format_idle_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds // 60)}分钟"
        else:
            return f"{int(seconds // 3600)}小时"


# 全局单例
_state_graph: Optional[StateGraph] = None


def get_state_graph() -> StateGraph:
    global _state_graph
    if _state_graph is None:
        _state_graph = StateGraph()
    return _state_graph