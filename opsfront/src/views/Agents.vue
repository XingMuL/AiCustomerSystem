<template>
  <div>
    <div class="page-header">
      <h1>Agent 实例列表</h1>
      <p>Agent 状态、向量时钟、Token 消耗与重启管理 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Agent 总数</div>
        <div class="stat-value">{{ agents.length }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">在线实例</div>
        <div class="stat-value" style="color:#51cf66">{{ onlineCount }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">节点 ID</div>
        <div class="stat-value" style="font-size:13px;font-family:monospace">{{ agentData.node_id || '--' }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">对等节点数</div>
        <div class="stat-value">{{ (agentData.peers || []).length }}</div>
      </div>
    </div>

    <!-- Agent 实例详情表 -->
    <div class="chart-panel">
      <h3>Agent 实例详情</h3>
      <table class="data-table">
        <thead>
          <tr><th>Agent 名称</th><th>角色</th><th>状态</th><th>重启次数</th><th>向量时钟项</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="agent in agents" :key="agent.name">
            <td><strong>{{ agent.name }}</strong></td>
            <td><span class="badge badge-info">{{ agent.role }}</span></td>
            <td>
              <span class="badge" :class="agent.status === 'healthy' ? 'badge-success' : 'badge-danger'">
                {{ agent.status }}
              </span>
            </td>
            <td>{{ agent.restart_count }}</td>
            <td style="font-family:monospace;font-size:10px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
              {{ Object.keys(agent.vector_clock || {}).length }} 项
            </td>
            <td>
              <button class="btn btn-sm" @click="viewDetail(agent)">详情</button>
              <button class="btn btn-sm btn-danger" style="margin-left:4px" @click="restartAgent(agent.name)">重启</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Token 消耗分布（按 Agent） -->
    <div class="grid-2" style="margin-top:16px">
      <div class="chart-panel">
        <h3>Token 消耗分布（按 Agent）</h3>
        <div ref="tokenPie" class="chart-box"></div>
      </div>
      <div class="chart-panel">
        <h3>向量时钟状态</h3>
        <div ref="clockBar" class="chart-box"></div>
      </div>
    </div>

    <!-- 重启日志 -->
    <div class="chart-panel" style="margin-top:16px">
      <h3>最近重启记录</h3>
      <table class="data-table">
        <thead>
          <tr><th>Agent</th><th>时间</th><th>日志</th></tr>
        </thead>
        <tbody>
          <tr v-for="log in restartLogs" :key="log.id">
            <td>{{ log.agent }}</td>
            <td>{{ log.time }}</td>
            <td>{{ log.message }}</td>
          </tr>
          <tr v-if="restartLogs.length === 0">
            <td colspan="3" style="text-align:center;color:#667788;padding:20px">暂无重启记录</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Agent 详情弹窗 -->
    <div class="modal-overlay" v-if="selectedAgent" @click.self="selectedAgent = null">
      <div class="modal-box">
        <h3>{{ selectedAgent.name }} - 详情</h3>
        <div class="grid-2">
          <div><strong>角色:</strong> {{ selectedAgent.role }}</div>
          <div><strong>状态:</strong> {{ selectedAgent.status }}</div>
          <div><strong>重启次数:</strong> {{ selectedAgent.restart_count }}</div>
          <div><strong>向量时钟项:</strong> {{ Object.keys(selectedAgent.vector_clock || {}).length }}</div>
        </div>
        <div v-if="selectedAgent.restart_logs?.length" style="margin-top:12px">
          <strong style="font-size:13px">重启日志:</strong>
          <div v-for="(log, i) in selectedAgent.restart_logs" :key="i" style="font-size:12px;color:#667788;margin-top:4px">{{ log }}</div>
        </div>
        <div v-if="Object.keys(selectedAgent.vector_clock || {}).length > 0" style="margin-top:12px">
          <strong style="font-size:13px">向量时钟:</strong>
          <div v-for="(count, key) in selectedAgent.vector_clock" :key="key" style="font-size:12px;color:#667788;margin-top:2px">
            {{ key }}: {{ count }}
          </div>
        </div>
        <div class="modal-actions">
          <button class="btn" @click="selectedAgent = null">关闭</button>
          <button class="btn btn-danger" style="margin-left:8px" @click="restartAgent(selectedAgent.name); selectedAgent = null">重启</button>
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
const agentData = ref({ node_id: '', global_clock: {}, peers: [] })
const agents = ref([])
const selectedAgent = ref(null)
const tokenData = ref({ by_agent: [] })
const restartLogs = ref([])
const tokenPie = ref(null)
const clockBar = ref(null)
let charts = []

const onlineCount = computed(() => agents.value.filter(a => a.status === 'healthy').length)

const viewDetail = (agent) => { selectedAgent.value = agent }

const restartAgent = async (name) => {
  try {
    await api.restartAgent(name)
    loadData()
  } catch (e) {
    console.error('重启失败:', e)
  }
}

const loadData = async () => {
  loading.value = true
  try {
    const [agentRes, tokenRes] = await Promise.all([
      api.getAgents(),
      api.getTokens(),
    ])

    agentData.value = {
      node_id: agentRes.node_id,
      global_clock: agentRes.global_clock || {},
      peers: agentRes.peers || [],
    }
    agents.value = agentRes.agents || []
    tokenData.value = tokenRes

    // 收集重启日志
    const logs = []
    for (const agent of agents.value) {
      for (const log of (agent.restart_logs || [])) {
        logs.push({
          id: `${agent.name}_${logs.length}`,
          agent: agent.name,
          time: log.split(' - ')[0] || '',
          message: log,
        })
      }
    }
    restartLogs.value = logs.sort((a, b) => b.time.localeCompare(a.time))

    updateCharts()
  } catch (e) {
    console.warn('Agents API 不可用', e)
  } finally {
    loading.value = false
  }
}

const makeTokenPie = () => {
  const agents_token = tokenData.value.by_agent || []
  return {
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie', radius: '65%', center: ['50%', '50%'],
      label: { color: '#8899aa', fontSize: 11, formatter: '{b}: {c}' },
      data: agents_token.map(a => ({
        value: a.total_tokens || 0,
        name: a.agent || 'unknown',
      })),
    }]
  }
}

const makeClockBar = () => {
  const clock = agentData.value.global_clock || {}
  const names = Object.keys(clock).slice(0, 10)
  const values = names.map(n => clock[n] || 0)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 100, right: 20, top: 10, bottom: 20 },
    xAxis: { type: 'value', axisLabel: { color: '#667788' }, splitLine: { lineStyle: { color: '#1a2744' } } },
    yAxis: { type: 'category', data: names, axisLabel: { color: '#8899aa', fontSize: 10 } },
    series: [{
      type: 'bar', data: values,
      itemStyle: { color: '#51cf66', borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: '#8899aa', fontSize: 11 },
    }]
  }
}

const updateCharts = () => {
  if (charts[0]) charts[0].setOption(makeTokenPie())
  if (charts[1]) charts[1].setOption(makeClockBar())
}

let resizeHandler = null
onMounted(() => {
  const c1 = echarts.init(tokenPie.value, 'dark'); c1.setOption(makeTokenPie())
  const c2 = echarts.init(clockBar.value, 'dark'); c2.setOption(makeClockBar())
  charts = [c1, c2]
  resizeHandler = () => charts.forEach(c => c.resize())
  window.addEventListener('resize', resizeHandler)
  loadData()
})
onUnmounted(() => {
  charts.forEach(c => c.dispose())
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
})
</script>