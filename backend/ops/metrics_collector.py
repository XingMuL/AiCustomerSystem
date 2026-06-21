"""
运维指标采集器

提供：
  - 请求记录（含 Token 消耗追踪）
  - QPS / 延迟 / 错误率实时计算
  - 会话级别 Token 使用统计
  - Agent 调用链路追踪
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestRecord:
    """单次请求记录"""
    timestamp: float
    endpoint: str
    duration_ms: float
    status: str
    agent: str = ""
    tokens: int = 0
    session_id: str = ""


class Metrics:
    """系统实时指标快照"""

    def __init__(self):
        self.qps: float = 0.0
        self.avg_latency_ms: float = 0.0
        self.error_rate: float = 0.0
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self.active_sessions: int = 0


class MetricsCollector:
    """指标采集器（单例）"""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 请求记录（滑动窗口，保留最近 1000 条）
        self._requests: deque[RequestRecord] = deque(maxlen=1000)
        self._requests_lock = threading.Lock()

        # Token 消耗累计（按 Agent 维度）
        self._token_usage: dict[str, int] = {}
        self._token_lock = threading.Lock()

        # 会话记录
        self._session_requests: dict[str, list[RequestRecord]] = {}

        # 指标缓存
        self._cached_metrics: Optional[Metrics] = None
        self._last_calc_time: float = 0
        self._calc_interval: float = 5.0  # 每 5 秒更新一次

    def record_request(
        self,
        endpoint: str,
        duration_ms: float,
        status: str,
        agent: str = "",
        tokens: int = 0,
        session_id: str = "",
    ):
        """
        记录一次请求

        Args:
            endpoint: 请求端点
            duration_ms: 耗时（毫秒）
            status: 状态 (success / error)
            agent: 处理 Agent 名称
            tokens: Token 消耗数（调用 LLM 时传入）
            session_id: 会话 ID
        """
        record = RequestRecord(
            timestamp=time.time(),
            endpoint=endpoint,
            duration_ms=duration_ms,
            status=status,
            agent=agent,
            tokens=tokens,
            session_id=session_id,
        )

        with self._requests_lock:
            self._requests.append(record)

        # 累计 Token 消耗
        if tokens > 0:
            with self._token_lock:
                self._token_usage[agent] = self._token_usage.get(agent, 0) + tokens

        # 记录会话请求
        if session_id:
            if session_id not in self._session_requests:
                self._session_requests[session_id] = []
            self._session_requests[session_id].append(record)

    def get_token_usage(self, agent: str = "") -> int:
        """获取 Token 累计消耗"""
        with self._token_lock:
            if agent:
                return self._token_usage.get(agent, 0)
            return sum(self._token_usage.values())

    def get_agent_tokens(self) -> dict[str, int]:
        """获取各 Agent Token 消耗"""
        with self._token_lock:
            return dict(self._token_usage)

    def get_recent_requests(self, count: int = 50) -> list[RequestRecord]:
        """获取最近的请求记录"""
        with self._requests_lock:
            items = list(self._requests)[-count:]
            return items

    def get_session_tokens(self, session_id: str) -> int:
        """获取会话 Token 消耗"""
        records = self._session_requests.get(session_id, [])
        return sum(r.tokens for r in records)

    def get_session_request_count(self, session_id: str) -> int:
        """获取会话请求数"""
        return len(self._session_requests.get(session_id, []))

    def get_metrics(self) -> Metrics:
        """获取当前指标快照"""
        now = time.time()
        # 使用缓存避免频繁计算
        if self._cached_metrics and (now - self._last_calc_time) < self._calc_interval:
            return self._cached_metrics

        with self._requests_lock:
            all_requests = list(self._requests)

        if not all_requests:
            return Metrics()

        m = Metrics()
        now_time = time.time()
        window_size = 60.0  # 60 秒滑动窗口

        # 统计总请求数
        m.total_requests = len(all_requests)

        # 统计活跃会话
        active_sessions = set()
        window_requests = []
        for r in all_requests:
            if r.session_id:
                active_sessions.add(r.session_id)
            if now_time - r.timestamp <= window_size:
                window_requests.append(r)
        m.active_sessions = len(active_sessions)

        # QPS: 窗口内请求数 / 窗口大小
        if len(window_requests) >= 2:
            time_span = max(
                window_requests[-1].timestamp - window_requests[0].timestamp, 0.001
            )
            m.qps = round(len(window_requests) / time_span, 2)
        else:
            m.qps = 0

        # 平均延迟
        latencies = [r.duration_ms for r in window_requests if r.duration_ms > 0]
        m.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

        # 错误率
        errors = sum(1 for r in all_requests if r.status == "error")
        m.error_rate = round(errors / len(all_requests), 4) if all_requests else 0.0

        # Token 总和
        m.total_tokens = self.get_token_usage()

        self._cached_metrics = m
        self._last_calc_time = now
        return m


# 全局单例
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector