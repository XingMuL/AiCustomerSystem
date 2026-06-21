<template>
  <div>
    <div class="page-header">
      <h1>总览大盘</h1>
      <p>系统实时运行状态概览（Gossip + 降级 + 多智能体） <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <!-- 核心指标卡片 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">当前 QPS</div>
        <div class="stat-value">{{ overview.qps || '--' }}</div>
        <div class="stat-change" style="color:#51cf66">实时</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">平均延迟</div>
        <div class="stat-value">{{ overview.avg_latency_ms || '--' }}ms</div>
        <div class="stat-change" style="color:#51cf66">实时</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">错误率</div>
        <div class="stat-value" :style="{color: (overview.error_rate || 0) > 0.05 ? '#ff6b6b' : '#51cf66'}">
          {{ ((overview.error_rate || 0) * 100).toFixed(2) }}%
        </div>
        <div class="stat-change">实时</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">总请求数</div>
        <div class="stat-value">{{ overview.total_requests || 0 }}</div>
        <div class="stat-change">累计</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">活跃会话</div>
        <div class="stat-value" style="color:#51cf66">{{ overview.active_sessions || 0 }}</div>
        <div class="stat-change">僵死: {{ overview.dead_sessions || 0 }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Token 消耗</div>
        <div class="stat-value">{{ (overview.total_tokens || 0).toLocaleString() }}</div>
        <div class="stat-change">累计</div>
      </div>
    </div>

    <!-- 性能 + Token 分布 -->
    <div class="grid-2" style="margin-top:16px">
      <div class="chart-panel">
        <h3>系统延迟分布</h3>
        <div ref="latencyGauge" class="chart-box"></div>
      </div>
      <div class="chart-panel">
        <h3>Agent 健康状态</h3>
        <div ref="agentHealth" class="chart-box"></div>
      </div>
    </div>

    <!-- 降级状态 -->
    <div class="chart-panel">
      <h3>系统降级状态</h3>
      <div class="degradation-bar">
        <div v-for="lvl in degradationLevels" :key="lvl.level" class="degradation-level" :class="{ active: lvl.active }">
          <div class="lvl-indicator">{{ lvl.label }}</div>
          <div class="lvl-desc">{{ lvl.desc }}</div>
        </div>
      </div>
      <div v-if="degradationInfo.level !== undefined" style="margin-top:8px;font-size:12px;color:#667788;">
        LLM 可用: {{ degradationInfo.llm_available ? '是' : '否' }} |
        RAG 可用: {{ degradationInfo.rag_available ? '是' : '否' }} |
        熔断器: {{ Object.keys(degradationInfo.circuit_breakers || {}).length }} 个
      </div>
    </div>

    <!-- 向量时钟 + 节点信息 -->
    <div class="grid-2" style="margin-top:16px">
      <div class="chart-panel">
        <h3>Gossip 向量时钟</h3>
        <div ref="clockChart" class="chart-box"></div>
      </div>
      <div class="chart-panel">
        <h3>节点信息</h3>
        <div style="padding:16px;font-size:13px;color:#8899aa">
          <div style="margin-bottom:8px"><strong>节点 ID:</strong> {{ nodeId }}</div>
          <div style="margin-bottom:8px"><strong>对等节点:</strong> {{ peerCount }} 个</div>
          <div style="margin-bottom:8px"><strong>Agent 总数:</strong> {{ agentCount }}</div>
          <div style="margin-bottom:8px"><strong>向量时钟项:</strong> {{ clockItemCount }} 项</div>
          <div v-for="(count, key) in vectorClockSample" :key="key" style="margin-bottom:4px;font-family:monospace;font-size:11px;padding-left:8px;border-left:2px solid #4dabf7">
            {{ key }}: {{ count }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import echarts from '../utils/echarts.js'
import { api } from '../utils/api.js'

const loading = ref(false)
const overview = ref({ qps: 0, avg_latency_ms: 0, error_rate: 0, total_requests: 0, active_sessions: 0, dead_sessions: 0, total_tokens: 0 })
const degradationInfo = ref({ level: 0, llm_available: true, rag_available: true, circuit_breakers: {} })
const nodeId = ref('--')
const peerCount = ref(0)
const agentCount = ref(0)
const clockItemCount = ref(0)
const vectorClockSample = ref({})
const latencyGauge = ref(null)
const agentHealth = ref(null)
const clockChart = ref(null)
let charts = []

const degradationLevels = ref([
  { level: 0, label: 'L0', desc: '正常运行', active: false },
  { level: 1, label: 'L1', desc: '轻度降级', active: false },
  { level: 2, label: 'L2', desc: '局部故障', active: false },
  { level: 3, label: 'L3', desc: '中度降级', active: false },
  { level: 4, label: 'L4', desc: '全局故障', active: false },
])

const loadData = async () => {
  loading.value = true
  try {
    const [dashRes, perfRes] = await Promise.all([
      api.getDashboard(),
      api.getPerformance(),
    ])

    const dash = dashRes.overview || {}
    overview.value = {
      qps: dash.qps || 0,
      avg_latency_ms: dash.avg_latency_ms || 0,
      error_rate: dash.error_rate || 0,
      total_requests: dash.total_requests || 0,
      active_sessions: dash.active_sessions || 0,
      dead_sessions: dash.dead_sessions || 0,
      total_tokens: dash.total_tokens || 0,
    }

    const deg = dashRes.degradation || {}
    degradationInfo.value = {
      level: deg.degradation_level || 0,
      llm_available: deg.llm_available !== false,
      rag_available: deg.rag_available !== false,
      circuit_breakers: deg.circuit_breakers || {},
    }

    const agents = dashRes.agents || {}
    agentCount.value = Object.keys(agents).length
    nodeId.value = dashRes.node_id || '--'
    peerCount.value = (dashRes.peers || []).length
    clockItemCount.value = Object.keys(dashRes.vector_clock || {}).length
    // 取前 5 项向量时钟
    const clock = dashRes.vector_clock || {}
    vectorClockSample.value = Object.fromEntries(Object.entries(clock).slice(0, 5))

    // 更新降级状态
    const level = degradationInfo.value.level
    degradationLevels.value = degradationLevels.value.map((l, i) => ({ ...l, active: i === level }))

    // 延迟分布数据（来自性能 API）
    updateCharts(perfRes)
  } catch (e) {
    console.warn('Dashboard API 不可用', e)
  } finally {
    loading.value = false
  }
}

const makeLatencyGauge = (perfData) => {
  const dist = perfData?.latency_distribution || {}
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 10, bottom: 30 },
    xAxis: { type: 'category', data: ['P50', 'P90', 'P99', 'Avg', 'Max'], axisLabel: { color: '#667788' } },
    yAxis: { type: 'value', axisLabel: { color: '#667788', formatter: '{value}ms' }, splitLine: { lineStyle: { color: '#1a2744' } } },
    series: [{
      type: 'bar',
      data: [dist.p50 || 0, dist.p90 || 0, dist.p99 || 0, dist.avg || 0, dist.max || 0],
      itemStyle: {
        color: (params) => {
          const colors = ['#51cf66', '#4dabf7', '#ffd43b', '#8899aa', '#ff6b6b']
          return colors[params.dataIndex] || '#4dabf7'
        },
        borderRadius: [4, 4, 0, 0],
      },
      label: { show: true, position: 'top', color: '#8899aa', fontSize: 11, formatter: '{c}ms' },
    }]
  }
}

const makeAgentHealthChart = (perfData) => {
  const statusDist = perfData?.request_status_distribution || { success: 0, error: 0 }
  return {
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie', radius: ['50%', '70%'], center: ['50%', '50%'],
      label: { color: '#8899aa', fontSize: 12, formatter: '{b}: {c}' },
      data: [
        { value: statusDist.success || 0, name: '成功', itemStyle: { color: '#51cf66' } },
        { value: statusDist.error || 0, name: '失败', itemStyle: { color: '#ff6b6b' } },
      ],
    }]
  }
}

const makeClockDistribution = () => {
  const clock = vectorClockSample.value
  const names = Object.keys(clock)
  const values = names.map(n => clock[n] || 0)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 100, right: 20, top: 10, bottom: 20 },
    xAxis: { type: 'value', axisLabel: { color: '#667788' }, splitLine: { lineStyle: { color: '#1a2744' } } },
    yAxis: { type: 'category', data: names, axisLabel: { color: '#8899aa', fontSize: 10 } },
    series: [{
      type: 'bar', data: values,
      itemStyle: { color: '#4dabf7', borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: '#8899aa', fontSize: 11 },
    }]
  }
}

const updateCharts = (perfData) => {
  if (charts[0]) charts[0].setOption(makeLatencyGauge(perfData))
  if (charts[1]) charts[1].setOption(makeAgentHealthChart(perfData))
  if (charts[2]) charts[2].setOption(makeClockDistribution())
}

let resizeHandler = null
onMounted(() => {
  charts = [
    echarts.init(latencyGauge.value, 'dark'),
    echarts.init(agentHealth.value, 'dark'),
    echarts.init(clockChart.value, 'dark'),
  ]
  charts[0].setOption(makeLatencyGauge({}))
  charts[1].setOption(makeAgentHealthChart({}))
  charts[2].setOption(makeClockDistribution())
  resizeHandler = () => charts.forEach(c => c.resize())
  window.addEventListener('resize', resizeHandler)
  loadData()
})
onUnmounted(() => {
  charts.forEach(c => c.dispose())
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
})
</script>