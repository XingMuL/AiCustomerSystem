"""
多智能体 Gossip 协调器

将 Gossip 协议和向量时钟集成到多智能体协作流程中：
- 每个 Agent 调用递增向量时钟，确保因果一致性
- Agent 之间通过 Gossip 传播状态，实现最终一致性同步
- 支持并发 Agent 执行时的冲突检测与合并
- 降级/熔断状态通过 Gossip 在节点间传播
"""

import time
import threading
import json
from typing import Optional
from loguru import logger

from backend.memory.gossip import GossipManager, VectorClock
from backend.config import settings


class GossipAgentCoordinator:
    """
    多智能体 Gossip 协调器（单例）

    职责：
    1. 管理全局向量时钟，追踪所有 Agent 操作顺序
    2. 通过 Gossip 协议在节点间同步 Agent 状态
    3. 管理 Agent 实例生命周期（重启、健康检查）
    4. 提供并发 Agent 调用的因果一致性保证
    """

    _instance = None
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

        # Gossip 管理器
        self.gossip = GossipManager()
        self.node_id = settings.node_id

        # Agent 实例注册表
        self._agent_instances: dict[str, dict] = {}
        self._agent_restart_counts: dict[str, int] = {}
        self._agent_restart_logs: dict[str, list[str]] = {}
        self._agent_lock = threading.Lock()

        # 全局操作向量时钟
        self._global_clock = VectorClock()

        # 节点间同步的 Agent 状态快照
        self._agent_state_snapshot: dict = {}

        # 定时 Gossip 同步线程
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"[GossipCoordinator] 初始化完成, node_id={self.node_id}, "
                    f"peers={self.gossip.peers}")

    # ============ Agent 生命周期管理 ============

    def register_agent(self, name: str, instance: object, role: str = ""):
        """注册 Agent 实例"""
        with self._agent_lock:
            self._agent_instances[name] = {
                "instance": instance,
                "role": role,
                "status": "healthy",
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
            }
            if name not in self._agent_restart_counts:
                self._agent_restart_counts[name] = 0
            if name not in self._agent_restart_logs:
                self._agent_restart_logs[name] = []
            logger.info(f"[GossipCoordinator] 注册 Agent: {name} ({role})")

    def restart_agent(self, name: str) -> bool:
        """重启 Agent 实例"""
        with self._agent_lock:
            if name not in self._agent_instances:
                logger.warning(f"[GossipCoordinator] Agent 不存在: {name}")
                return False

            agent_info = self._agent_instances[name]
            old_instance = agent_info["instance"]
            agent_class = type(old_instance)

            try:
                # 重新创建实例
                new_instance = agent_class()
                agent_info["instance"] = new_instance
                agent_info["status"] = "restarting"
                agent_info["last_heartbeat"] = time.time()

                self._agent_restart_counts[name] = self._agent_restart_counts.get(name, 0) + 1
                restart_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self._agent_restart_logs[name].append(
                    f"{restart_time} - 重启成功 (第{self._agent_restart_counts[name]}次)"
                )
                # 保留最近 10 条记录
                if len(self._agent_restart_logs[name]) > 10:
                    self._agent_restart_logs[name] = self._agent_restart_logs[name][-10:]

                # 短暂延迟后标记为健康
                def _set_healthy():
                    time.sleep(1)
                    with self._agent_lock:
                        if name in self._agent_instances:
                            self._agent_instances[name]["status"] = "healthy"
                threading.Thread(target=_set_healthy, daemon=True).start()

                logger.info(
                    f"[GossipCoordinator] Agent 重启成功: {name} "
                    f"(第{self._agent_restart_counts[name]}次)"
                )
                return True
            except Exception as e:
                logger.error(f"[GossipCoordinator] Agent 重启失败: {name}, error={e}")
                agent_info["status"] = "error"
                return False

    def get_restarted_agent_instance(self, name: str):
        """获取重启后的 Agent 实例（供 StateGraph 更新引用）"""
        with self._agent_lock:
            if name in self._agent_instances:
                return self._agent_instances[name]["instance"]
        return None

    def get_agent_restart_info(self, name: str) -> dict:
        """获取 Agent 重启信息"""
        return {
            "count": self._agent_restart_counts.get(name, 0),
            "logs": self._agent_restart_logs.get(name, []),
        }

    def heartbeat_agent(self, name: str):
        """Agent 心跳"""
        with self._agent_lock:
            if name in self._agent_instances:
                self._agent_instances[name]["last_heartbeat"] = time.time()

    # ============ 向量时钟集成 ============

    def tick_agent_clock(self, agent_name: str) -> dict:
        """
        递增指定 Agent 的向量时钟

        每次 Agent 调用前调用，确保操作的因果顺序。
        """
        clock_key = f"{self.node_id}:{agent_name}"
        self._global_clock.tick(clock_key)
        self.gossip.vector_clock.tick(clock_key)
        return dict(self._global_clock)

    def get_agent_clock(self, agent_name: str) -> dict:
        """获取 Agent 的向量时钟快照"""
        return dict(self._global_clock)

    def get_global_clock(self) -> dict:
        """获取全局向量时钟快照"""
        return dict(self._global_clock)

    def merge_clock(self, remote_clock: dict):
        """合并远程向量时钟"""
        self._global_clock.merge(VectorClock(remote_clock))

    # ============ Agent 状态 Gossip 同步 ============

    def prepare_agent_gossip(self, agent_name: str, state: dict) -> dict:
        """准备 Agent 状态的 Gossip 传播负载"""
        self.tick_agent_clock(agent_name)
        payload = self.gossip.prepare_gossip_payload(state)
        # 附加 Agent 元数据
        payload["agent_name"] = agent_name
        payload["agent_clock"] = self.get_agent_clock(agent_name)
        return payload

    def receive_agent_gossip(self, payload: dict):
        """接收并处理 Agent Gossip 消息"""
        agent_name = payload.get("agent_name", "unknown")
        remote_clock = payload.get("agent_clock", {})
        remote_node = payload.get("node_id", "unknown")

        # 合并向量时钟
        self.merge_clock(remote_clock)

        # 同步远程状态
        self.gossip.sync_state(
            remote_state=payload.get("state", {}),
            remote_clock=payload.get("vector_clock", {}),
            remote_node=remote_node,
        )

        self._agent_state_snapshot[agent_name] = {
            "state": payload.get("state", {}),
            "node": remote_node,
            "clock": remote_clock,
            "timestamp": payload.get("timestamp", time.time()),
        }
        logger.debug(f"[GossipCoordinator] 接收 Agent Gossip: {agent_name} from {remote_node}")

    def get_agent_state_snapshot(self, agent_name: str) -> Optional[dict]:
        """获取 Agent 状态快照"""
        return self._agent_state_snapshot.get(agent_name)

    # ============ 并发协调 ============

    def coordinate_agent_calls(self, agent_names: list[str]) -> dict:
        """
        协调多个 Agent 并发调用

        在调用前递增向量时钟，确保因果一致性。
        返回每个 Agent 的调用上下文。
        """
        contexts = {}
        with self._agent_lock:
            for name in agent_names:
                clock_before = self.tick_agent_clock(name)
                contexts[name] = {
                    "clock_before": clock_before,
                    "timestamp": time.time(),
                }
        return contexts

    def finalize_agent_call(self, agent_name: str, token_usage: int = 0, duration_ms: float = 0):
        """完成 Agent 调用并记录指标"""
        clock_after = self.tick_agent_clock(agent_name)

        # 记录 Token 消耗（延迟导入避免循环依赖）
        if token_usage > 0:
            from backend.ops.metrics_collector import get_metrics_collector
            collector = get_metrics_collector()
            collector.record_request(
                endpoint=f"/agent/{agent_name}",
                duration_ms=duration_ms,
                status="success",
                agent=agent_name,
                tokens=token_usage,
            )

        return clock_after

    # ============ 向量时钟冲突检测 ============

    def detect_conflict(self, clock_a: dict, clock_b: dict) -> str:
        """
        使用向量时钟检测两个事件间的因果关系

        Returns:
            "concurrent" - 并发事件，存在冲突
            "a_after_b" - clock_a 在 clock_b 之后
            "b_after_a" - clock_b 在 clock_a 之后
            "equal" - 同一事件
        """
        all_keys = set(clock_a.keys()) | set(clock_b.keys())
        a_greater = False
        b_greater = False

        for key in all_keys:
            a_val = clock_a.get(key, 0)
            b_val = clock_b.get(key, 0)
            if a_val > b_val:
                a_greater = True
            elif b_val > a_val:
                b_greater = True

        if a_greater and b_greater:
            return "concurrent"
        elif a_greater:
            return "a_after_b"
        elif b_greater:
            return "b_after_a"
        else:
            return "equal"

    def resolve_agent_results(
        self, results: dict[str, dict], call_contexts: dict
    ) -> dict[str, dict]:
        """
        使用向量时钟解决并发 Agent 结果冲突

        当多个 Agent 并发执行时，比较它们的向量时钟来确定最终状态。
        采用 last-writer-wins (LWW) 策略：保留最新时钟对应的结果。
        """
        if len(results) <= 1:
            return results

        # 按向量时钟排序，最新的排前面
        sorted_agents = sorted(
            results.items(),
            key=lambda item: sum(
                call_contexts.get(item[0], {}).get("clock_before", {}).values()
            ),
            reverse=True,
        )

        resolved = {}
        for name, result in sorted_agents:
            has_conflict = False
            for existing_name, existing_result in resolved.items():
                relation = self.detect_conflict(
                    call_contexts.get(name, {}).get("clock_before", {}),
                    call_contexts.get(existing_name, {}).get("clock_before", {}),
                )
                if relation == "concurrent":
                    has_conflict = True
                    logger.warning(
                        f"[GossipCoordinator] 并发冲突检测: {name} vs {existing_name}, "
                        f"采用最新结果"
                    )
                    # LWW: 删除旧结果，保留新的
                    if sum(call_contexts.get(name, {}).get("clock_before", {}).values()) > \
                       sum(call_contexts.get(existing_name, {}).get("clock_before", {}).values()):
                        resolved.pop(existing_name, None)
                        resolved[name] = result
                    break

            if not has_conflict:
                resolved[name] = result

        return resolved

    # ============ HTTP Gossip 传播 ============

    def gossip_to_peers(self):
        """
        主动向对等节点传播状态

        从已知 peers 中随机选择 fanout 个，发送 HTTP POST
        """
        import urllib.request
        import random

        if not self.gossip.peers:
            return

        # 随机选择 fanout 个对等节点
        targets = random.sample(
            self.gossip.peers,
            min(self.gossip.fanout, len(self.gossip.peers)),
        )

        # 准备负载
        agent_states = {}
        with self._agent_lock:
            for name, info in self._agent_instances.items():
                agent_states[name] = {
                    "status": info["status"],
                    "heartbeat": info["last_heartbeat"],
                    "restarts": self._agent_restart_counts.get(name, 0),
                    "clock": self.get_agent_clock(name),
                }

        payload = self.gossip.prepare_gossip_payload({
            "agent_states": agent_states,
            "global_clock": self.get_global_clock(),
            "node_id": self.node_id,
        })

        payload_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for target in targets:
            try:
                url = f"http://{target}/gossip/receive"
                req = urllib.request.Request(
                    url,
                    data=payload_json,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.debug(f"[GossipCoordinator] 成功传播到 {target}")
                    else:
                        logger.warning(f"[GossipCoordinator] 传播到 {target} 返回 {resp.status}")
            except Exception as e:
                logger.warning(f"[GossipCoordinator] 无法连接到 {target}: {e}")

    def receive_gossip_message(self, payload: dict) -> dict:
        """
        接收来自其他节点的 Gossip 消息

        1. 验证向量时钟，只接受更新的数据
        2. 更新本地 Agent 状态
        3. 合并全局向量时钟
        """
        remote_node = payload.get("node_id", "unknown")
        remote_clock = payload.get("vector_clock", {})
        remote_state = payload.get("state", {})

        # 检查向量时钟 - 是否需要接受更新
        local_clock = self.gossip.get_vector_clock()
        relation = self.detect_conflict(remote_clock, local_clock)

        accepted = False
        if relation in ("b_after_a", "concurrent"):
            # 远程时钟不落后于本地，接受更新
            self.gossip.sync_state(remote_state, remote_clock, remote_node)
            self.gossip.vector_clock.merge(VectorClock(remote_clock))
            self.merge_clock(remote_clock)
            accepted = True

            # 同步远程 Agent 状态
            remote_agent_states = remote_state.get("agent_states", {})
            with self._agent_lock:
                for agent_name, agent_info in remote_agent_states.items():
                    if agent_name in self._agent_instances:
                        # 只在远程状态是 unhealthy 时才更新（健康信息传播）
                        if agent_info.get("status") == "unhealthy":
                            logger.warning(
                                f"[GossipCoordinator] 从 {remote_node} 收到 Agent 异常: {agent_name}"
                            )
                            self._agent_instances[agent_name]["status"] = "unhealthy"

            logger.info(
                f"[GossipCoordinator] 接受来自 {remote_node} 的状态更新, "
                f"relation={relation}"
            )
        else:
            logger.debug(
                f"[GossipCoordinator] 忽略来自 {remote_node} 的过时状态, "
                f"relation={relation}"
            )

        return {
            "accepted": accepted,
            "relation": relation,
            "local_clock": self.get_global_clock(),
        }

    # ============ 定时同步 ============

    def start_sync_loop(self, interval: float = 30.0):
        """启动定时 Gossip 同步"""
        if self._running:
            return
        self._running = True

        def _sync_loop():
            while self._running:
                try:
                    logger.debug(f"[GossipCoordinator] 定时同步, clock={self.gossip.get_vector_clock()}")
                    # 实际向对等节点传播状态
                    self.gossip_to_peers()
                except Exception as e:
                    logger.error(f"[GossipCoordinator] 同步异常: {e}")

                time.sleep(interval)

        self._sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info(f"[GossipCoordinator] 定时同步已启动, interval={interval}s")

    def stop_sync_loop(self):
        """停止定时同步"""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("[GossipCoordinator] 定时同步已停止")

    # ============ 健康检查 ============

    def check_agent_health(self, agent_name: str) -> str:
        """检查 Agent 健康状态"""
        with self._agent_lock:
            if agent_name not in self._agent_instances:
                return "unknown"
            info = self._agent_instances[agent_name]
            # 心跳超时判定
            if time.time() - info["last_heartbeat"] > 120:
                info["status"] = "unhealthy"
                return "unhealthy"
            return info["status"]

    def get_all_agent_health(self) -> dict:
        """获取所有 Agent 健康状态"""
        return {
            name: self.check_agent_health(name)
            for name in self._agent_instances
        }

    def get_status_report(self) -> dict:
        """获取完整状态报告"""
        return {
            "node_id": self.node_id,
            "vector_clock": self.get_global_clock(),
            "gossip_peers": self.gossip.peers,
            "agents": {
                name: {
                    "status": info["status"],
                    "role": info["role"],
                    "restarts": self._agent_restart_counts.get(name, 0),
                    "restart_logs": self._agent_restart_logs.get(name, []),
                    "clock": self.get_agent_clock(name),
                }
                for name, info in self._agent_instances.items()
            },
            "agent_snapshots": self._agent_state_snapshot,
        }


# 全局单例
_coordinator: Optional[GossipAgentCoordinator] = None


def get_gossip_coordinator() -> GossipAgentCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = GossipAgentCoordinator()
    return _coordinator