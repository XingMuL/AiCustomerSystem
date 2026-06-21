<template>
  <div>
    <div class="page-header">
      <h1>Gossip 拓扑 & 调用链路</h1>
      <p>去中心化 Gossip 协议节点拓扑、向量时钟与 Agent 状态同步 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <!-- 节点信息 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">当前节点</div>
        <div class="stat-value" style="font-size:16px;font-family:monospace">{{ topoData.node_id || '--' }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">对等节点</div>
        <div class="stat-value" style="color:#51cf66">{{ (topoData.gossip_peers || []).length }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">向量时钟项</div>
        <div class="stat-value">{{ Object.keys(topoData.vector_clock || {}).length }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">同步 Agent 数</div>
        <div class="stat-value">{{ Object.keys(topoData.agents || {}).length }}</div>
      </div>
    </div>

    <!-- Gossip 拓扑图 -->
    <div class="chart-panel">
      <h3>Gossip 节点拓扑</h3>
      <div ref="topoChart" class="chart-box" style="height:350px"></div>
    </div>

    <!-- Agent 节点状态 -->
    <div class="chart-panel">
      <h3>Agent 节点状态</h3>
      <table class="data-table">
        <thead>
          <tr><th>Agent 名称</th><th>角色</th><th>状态</th><th>重启次数</th><th>最后心跳</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="(info, name) in (topoData.agents || {})" :key="name">
            <td><strong>{{ name }}</strong></td>
            <td><span class="badge badge-info">{{ info.role || name }}</span></td>
            <td>
              <span class="badge" :class="info.status === 'healthy' ? 'badge-success' : 'badge-danger'">
                {{ info.status || 'unknown' }}
              </span>
            </td>
            <td>{{ info.restarts || 0 }}</td>
            <td>{{ info.last_heartbeat || '--' }}</td>
            <td>
              <button class="btn btn-sm" @click="viewAgentDetail(name, info)">详情</button>
            </td>
          </tr>
          <tr v-if="!topoData.agents || Object.keys(topoData.agents).length === 0">
            <td colspan="6" style="text-align:center;color:#667788;padding:20px">暂无 Agent 注册信息</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 向量时钟详情 -->
    <div class="chart-panel" style="margin-top:16px">
      <h3>向量时钟详情</h3>
      <table class="data-table">
        <thead>
          <tr><th>节点/Agent</th><th>时钟计数</th></tr>
        </thead>
        <tbody>
          <tr v-for="(count, key) in (topoData.vector_clock || {})" :key="key">
            <td style="font-family:monospace;font-size:12px">{{ key }}</td>
            <td>{{ count }}</td>
          </tr>
          <tr v-if="!topoData.vector_clock || Object.keys(topoData.vector_clock).length === 0">
            <td colspan="2" style="text-align:center;color:#667788;padding:20px">暂无向量时钟数据</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Agent 详情弹窗 -->
    <div class="modal-overlay" v-if="selectedDetail" @click.self="selectedDetail = null">
      <div class="modal-box">
        <h3>{{ selectedDetail.name }} - Gossip 详情</h3>
        <div class="grid-2">
          <div><strong>角色:</strong> {{ selectedDetail.info.role || '-' }}</div>
          <div><strong>状态:</strong> {{ selectedDetail.info.status || '-' }}</div>
          <div><strong>重启次数:</strong> {{ selectedDetail.info.restarts || 0 }}</div>
        </div>
        <div v-if="selectedDetail.info.clock && Object.keys(selectedDetail.info.clock).length > 0" style="margin-top:12px">
          <strong style="font-size:13px">Agent 向量时钟:</strong>
          <div v-for="(count, key) in selectedDetail.info.clock" :key="key" style="font-size:12px;color:#667788;margin-top:2px">
            {{ key }}: {{ count }}
          </div>
        </div>
        <div v-if="selectedDetail.info.restart_logs?.length" style="margin-top:12px">
          <strong style="font-size:13px">重启日志:</strong>
          <div v-for="(log, i) in selectedDetail.info.restart_logs" :key="i" style="font-size:12px;color:#667788;margin-top:4px">{{ log }}</div>
        </div>
        <div style="margin-top:12px;font-size:12px;color:#667788">
          快照时间: {{ selectedDetail.info.last_heartbeat ? new Date(selectedDetail.info.last_heartbeat * 1000).toLocaleString() : '--' }}
        </div>
        <div class="modal-actions"><button class="btn" @click="selectedDetail = null">关闭</button></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import echarts from '../utils/echarts.js'
import { api } from '../utils/api.js'

const loading = ref(false)
const topoData = ref({ node_id: '', vector_clock: {}, agents: {}, gossip_peers: [], agent_snapshots: {} })
const selectedDetail = ref(null)
const topoChart = ref(null)
let chart = null

const viewAgentDetail = (name, info) => {
  selectedDetail.value = { name, info }
}

const loadData = async () => {
  loading.value = true
  try {
    const data = await api.getTopology()
    topoData.value = data
    updateChart()
  } catch (e) {
    console.warn('Topology API 不可用', e)
  } finally {
    loading.value = false
  }
}

const makeTopoChart = () => {
  const agents = topoData.value.agents || {}
  const nodeId = topoData.value.node_id || 'unknown'

  // 构建节点：中心节点 + Agent 节点 + 对等节点
  const nodes = [
    { name: nodeId, symbolSize: 50, category: 0, itemStyle: { color: '#51cf66' } },
  ]

  // Agent 子节点
  let i = 1
  for (const [name, info] of Object.entries(agents)) {
    nodes.push({
      name,
      symbolSize: 30 + (info.restarts || 0) * 5,
      category: 1,
      itemStyle: { color: info.status === 'healthy' ? '#4dabf7' : '#ff6b6b' },
    })
    i++
  }

  // 对等节点
  for (const peer of (topoData.value.gossip_peers || [])) {
    nodes.push({
      name: peer,
      symbolSize: 35,
      category: 2,
      itemStyle: { color: '#ffd43b' },
    })
  }

  // 构建连线（中心节点到所有其他节点）
  const links = nodes.slice(1).map(n => ({
    source: nodeId,
    target: n.name,
  }))

  return {
    tooltip: {},
    series: [{
      type: 'graph', layout: 'force', roam: true, draggable: true,
      force: { repulsion: 300, edgeLength: [100, 250] },
      label: { show: true, color: '#c8d6e5', fontSize: 11 },
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 6,
      data: nodes,
      links,
      lineStyle: { color: '#4dabf7', opacity: 0.3, curveness: 0.2 },
      categories: [
        { name: '主节点' },
        { name: 'Agent' },
        { name: '对等节点' },
      ],
    }]
  }
}

const updateChart = () => {
  if (chart) {
    chart.setOption(makeTopoChart())
  }
}

let resizeHandler = null
onMounted(() => {
  chart = echarts.init(topoChart.value, 'dark')
  chart.setOption(makeTopoChart())
  resizeHandler = () => chart.resize()
  window.addEventListener('resize', resizeHandler)
  loadData()
})
onUnmounted(() => {
  if (chart) chart.dispose()
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
})
</script>