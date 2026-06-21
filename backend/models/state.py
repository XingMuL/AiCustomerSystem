"""
多智能体系统状态定义

定义 AgentState、意图分类、降级等级、反思结果等核心数据结构。
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import TypedDict, Optional, Any


# ================================================================
#  意图分类
# ================================================================

class Intent(str, Enum):
    """用户意图分类"""
    KB_QA = "kb_qa"           # 知识库问答
    TICKET = "ticket"         # 工单处理
    CHITCHAT = "chitchat"     # 闲聊
    UNKNOWN = "unknown"       # 未知意图


# ================================================================
#  降级相关枚举
# ================================================================

class DegradationLevel(int, Enum):
    """降级等级"""
    L0_NORMAL = 0          # 正常（无降级）
    L1_LIGHT = 1           # 轻度降级（性能不佳、响应慢），限流
    L2_PARTIAL = 2         # 局部故障（单个/部分子Agent异常），熔断
    L3_MEDIUM = 3          # 中度降级（RAG/知识库不可用），停用RAG
    L4_GLOBAL = 4          # 全局故障（LLM集群、核心依赖全挂），纯兜底


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 正常（请求通过）
    OPEN = "open"           # 熔断（拒绝请求）
    HALF_OPEN = "half_open" # 半开（探测恢复）


class AgentStatus(str, Enum):
    """Agent 健康状态"""
    HEALTHY = "healthy"         # 健康
    DEGRADED = "degraded"       # 降级（性能不佳）
    CIRCUIT_OPEN = "circuit_open"  # 已熔断
    UNKNOWN = "unknown"         # 未知


# ================================================================
#  反思评判结果
# ================================================================

@dataclass
class ReflectionResult:
    """反思评判结果"""
    score: float                          # 质量评分 0-1
    is_acceptable: bool                   # 是否可接受
    issues: list[str] = field(default_factory=list)   # 问题列表
    retry_instruction: str = ""           # 重试修正指令
    dimensions: dict = field(default_factory=dict)    # 各维度评分详情


# ================================================================
#  AgentState
# ================================================================

class AgentState(TypedDict, total=False):
    """Agent 全局状态"""
    # 基础信息
    session_id: str
    user_id: Optional[int]          # 关联的用户 ID（用于查询订单等）
    node_id: str
    raw_input: str
    cleaned_input: str
    input_tokens: int
    messages: list[dict]

    # 路由与意图
    intent: str
    intent_confidence: float
    target_agents: list[str]

    # 子 Agent 结果
    agent_results: dict[str, dict]

    # 记忆上下文
    memory_context: str
    rag_docs: list[dict]
    cleaned_rag_docs: list[dict]   # 清洗后的文档（过滤无关语义信息）
    cleaned_context: str           # 清洗后的上下文合并文本
    current_turn: int

    # 最终回复
    final_response: str

    # 向量时钟（Gossip 同步）
    vector_clock: dict

    # 工单
    ticket: Optional[dict]

    # 元数据
    metadata: dict

    # ===== 降级相关状态 =====
    degradation_level: int        # 当前降级等级
    rag_available: bool           # RAG 是否可用
    llm_available: bool           # LLM 是否可用
    circuit_breakers: dict[str, str]  # 各 Agent 熔断器状态

    # ===== 反思评判 =====
    reflection_results: dict[str, ReflectionResult]  # 各 Agent 反思结果