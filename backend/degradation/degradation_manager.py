"""
降级管理器

实现 5 级降级策略、熔断器、令牌桶限流、系统健康监控。
"""

import time
import json
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from backend.config import settings
from backend.models.state import (
    DegradationLevel, CircuitState, AgentStatus
)


# ================================================================
#  熔断器
# ================================================================

class CircuitBreaker:
    """熔断器：三状态（CLOSED / OPEN / HALF_OPEN）管理"""

    def __init__(self, name: str):
        self.name = name
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0
        self._last_fail_time: float = 0.0
        self.threshold = settings.circuit_breaker_fail_threshold
        self.timeout = settings.circuit_breaker_timeout
        self.half_open_max = settings.circuit_breaker_half_open_max

    @property
    def state(self) -> str:
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(f"[熔断器:{self.name}] OPEN → HALF_OPEN")
        return self._state.value

    def allow_request(self) -> bool:
        """判断是否允许请求通过"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(f"[熔断器:{self.name}] 超时，进入 HALF_OPEN")
                return True
            return False
        if self._state == CircuitState.HALF_OPEN:
            return self._success_count < self.half_open_max
        return True

    def record_success(self):
        """记录成功"""
        self._fail_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max:
                self._state = CircuitState.CLOSED
                self._success_count = 0
                logger.info(f"[熔断器:{self.name}] HALF_OPEN → CLOSED（恢复）")

    def record_failure(self):
        """记录失败"""
        self._fail_count += 1
        self._last_fail_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            logger.warning(f"[熔断器:{self.name}] HALF_OPEN → OPEN（探测失败）")
        elif self._state == CircuitState.CLOSED and self._fail_count >= self.threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            logger.warning(f"[熔断器:{self.name}] CLOSED → OPEN（连续{self._fail_count}次失败）")

    def reset(self):
        """手动重置熔断器"""
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._success_count = 0


# ================================================================
#  令牌桶限流
# ================================================================

class RateLimiter:
    """令牌桶限流器"""

    def __init__(self, rate: float = None, burst: int = None):
        self.rate = rate or settings.rate_limit_rps
        self.burst = burst or settings.rate_limit_burst
        self._tokens = float(self.burst)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """判断是否允许请求"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


# ================================================================
#  DegradationManager
# ================================================================

class DegradationManager:
    """降级管理器：统一管理 5 级降级策略"""

    def __init__(self):
        self._llm_available = True
        self._rag_available = True
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._rate_limiter = RateLimiter()
        self._agent_status: dict[str, AgentStatus] = {}
        self._lock = threading.Lock()
        self._state_path = Path(settings.degradation_state_path)

        # 为每个 Agent 创建熔断器
        for agent_name in ["kb_qa_agent", "ticket_agent", "chitchat_agent", "router_agent", "summary_agent"]:
            self._circuit_breakers[agent_name] = CircuitBreaker(agent_name)
            self._agent_status[agent_name] = AgentStatus.HEALTHY

        # 加载持久化状态
        self._load_state()

    # ----- 健康状态管理 -----

    def set_llm_available(self, available: bool):
        self._llm_available = available
        if not available:
            logger.warning("[降级] LLM 不可用，进入 L4 全局故障")
        self._save_state()

    def set_rag_available(self, available: bool):
        self._rag_available = available
        if not available:
            logger.warning("[降级] RAG 不可用，进入 L3 中度降级")
        self._save_state()

    def get_agent_circuit_breaker(self, agent_name: str) -> Optional[CircuitBreaker]:
        """获取 Agent 的熔断器"""
        return self._circuit_breakers.get(agent_name)

    def allow_agent(self, agent_name: str) -> bool:
        """判断是否允许 Agent 执行（熔断 + 状态检查）"""
        cb = self._circuit_breakers.get(agent_name)
        if cb and not cb.allow_request():
            self._agent_status[agent_name] = AgentStatus.CIRCUIT_OPEN
            logger.warning(f"[降级] Agent {agent_name} 已熔断，拒绝请求")
            return False
        return True

    def record_agent_success(self, agent_name: str):
        """记录 Agent 执行成功"""
        cb = self._circuit_breakers.get(agent_name)
        if cb:
            cb.record_success()
            self._agent_status[agent_name] = AgentStatus.HEALTHY

    def record_agent_failure(self, agent_name: str):
        """记录 Agent 执行失败"""
        cb = self._circuit_breakers.get(agent_name)
        if cb:
            cb.record_failure()
            self._agent_status[agent_name] = AgentStatus.DEGRADED

    # ----- 限流 -----

    def allow_request(self) -> bool:
        """限流检查"""
        return self._rate_limiter.allow()

    # ----- 降级等级判定 -----

    def get_current_level(self) -> int:
        """计算当前全局降级等级"""
        if not self._llm_available:
            return DegradationLevel.L4_GLOBAL
        if not self._rag_available:
            return DegradationLevel.L3_MEDIUM

        # 检查是否有 Agent 熔断
        failed_agents = [
            name for name, status in self._agent_status.items()
            if status == AgentStatus.CIRCUIT_OPEN
        ]
        if failed_agents:
            logger.warning(f"[降级] 检测到熔断 Agent: {failed_agents}，进入 L2")
            return DegradationLevel.L2_PARTIAL

        degraded_agents = [
            name for name, status in self._agent_status.items()
            if status == AgentStatus.DEGRADED
        ]
        if degraded_agents:
            logger.warning(f"[降级] 检测到性能降级 Agent: {degraded_agents}，进入 L1")
            return DegradationLevel.L1_LIGHT

        return DegradationLevel.L0_NORMAL

    def get_failed_agents(self) -> list[str]:
        """获取所有已熔断的 Agent 名称"""
        return [
            name for name, status in self._agent_status.items()
            if status == AgentStatus.CIRCUIT_OPEN
        ]

    # ----- 状态快照 -----

    def apply_snapshot_to_state(self, state: dict) -> dict:
        """将当前降级快照注入到 AgentState"""
        level = self.get_current_level()
        state["degradation_level"] = level
        state["rag_available"] = self._rag_available
        state["llm_available"] = self._llm_available
        state["circuit_breakers"] = {
            name: cb.state for name, cb in self._circuit_breakers.items()
        }
        return state

    def get_status_report(self) -> dict:
        """获取降级状态报告"""
        return {
            "degradation_level": self.get_current_level(),
            "llm_available": self._llm_available,
            "rag_available": self._rag_available,
            "agent_status": {k: v.value for k, v in self._agent_status.items()},
            "circuit_breakers": {k: cb.state for k, cb in self._circuit_breakers.items()},
        }

    # ----- 重置与持久化 -----

    def reset_all(self):
        """重置所有降级状态"""
        self._llm_available = True
        self._rag_available = True
        for cb in self._circuit_breakers.values():
            cb.reset()
        for name in self._agent_status:
            self._agent_status[name] = AgentStatus.HEALTHY
        self._save_state()
        logger.info("[降级] 所有状态已重置")

    def _save_state(self):
        """持久化降级状态"""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "llm_available": self._llm_available,
                "rag_available": self._rag_available,
                "agent_status": {k: v.value for k, v in self._agent_status.items()},
                "timestamp": time.time(),
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[降级] 状态持久化失败: {e}")

    def _load_state(self):
        """加载持久化的降级状态"""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._llm_available = data.get("llm_available", True)
            self._rag_available = data.get("rag_available", True)
            for name, status in data.get("agent_status", {}).items():
                if name in self._agent_status:
                    self._agent_status[name] = AgentStatus(status)
            logger.info("[降级] 已加载持久化状态")
        except Exception as e:
            logger.error(f"[降级] 状态加载失败: {e}")


# 全局单例
_degradation_manager: Optional[DegradationManager] = None


def get_degradation_manager() -> DegradationManager:
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = DegradationManager()
    return _degradation_manager