"""
Gossip 去中心化同步

实现向量时钟、状态传播、最终一致性同步。
"""

import time
from loguru import logger

from backend.config import settings


class VectorClock(dict):
    """向量时钟"""

    def tick(self, node_id: str):
        """递增当前节点的时钟"""
        self[node_id] = self.get(node_id, 0) + 1

    def merge(self, other: "VectorClock"):
        """合并另一个向量时钟（取最大值）"""
        for node_id, count in other.items():
            if node_id not in self or count > self[node_id]:
                self[node_id] = count


class GossipManager:
    """Gossip 同步管理器"""

    def __init__(self):
        self.node_id = settings.node_id
        self.interval = settings.gossip_interval
        self.fanout = settings.gossip_fanout
        self.peers: list[str] = []
        self._last_sync: dict[str, float] = {}
        self.vector_clock = VectorClock()
        self.vector_clock.tick(self.node_id)

        # 解析对等节点
        if settings.gossip_peers:
            self.peers = [p.strip() for p in settings.gossip_peers.split(",") if p.strip()]
            logger.info(f"[Gossip] 对等节点: {self.peers}")

    def get_vector_clock(self) -> dict:
        """获取当前向量时钟快照"""
        return dict(self.vector_clock)

    def sync_state(self, remote_state: dict, remote_clock: dict, remote_node: str):
        """
        同步远程状态

        使用向量时钟判断因果顺序，实现最终一致性。
        """
        local_clock = dict(self.vector_clock)

        # 判断是否比本地更新
        if self._is_after(remote_clock, local_clock):
            logger.debug(f"[Gossip] 接受来自 {remote_node} 的状态更新")
            self.vector_clock.merge(VectorClock(remote_clock))
            self.vector_clock.tick(self.node_id)

        self._last_sync[remote_node] = time.time()

    def prepare_gossip_payload(self, state: dict) -> dict:
        """准备 Gossip 传播负载"""
        self.vector_clock.tick(self.node_id)
        return {
            "state": state,
            "vector_clock": dict(self.vector_clock),
            "node_id": self.node_id,
            "timestamp": time.time(),
        }

    def _is_after(self, clock_a: dict, clock_b: dict) -> bool:
        """判断 clock_a 是否严格在 clock_b 之后（happens-after）"""
        all_nodes = set(clock_a.keys()) | set(clock_b.keys())
        has_greater = False
        for node in all_nodes:
            a_val = clock_a.get(node, 0)
            b_val = clock_b.get(node, 0)
            if a_val < b_val:
                return False  # A 在某些维度上落后
            if a_val > b_val:
                has_greater = True
        return has_greater  # 至少一个维度 A 更大，其余相等