<template>
  <div>
    <div class="page-header">
      <h1>会话 & 上下文监控</h1>
      <p>实时后台会话状态、Token 消耗与僵死会话管理 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">活跃会话</div>
        <div class="stat-value" style="color:#51cf66">{{ sessionData.active || 0 }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">僵死会话</div>
        <div class="stat-value" style="color:#ff6b6b">{{ sessionData.dead || 0 }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">总会话数</div>
        <div class="stat-value">{{ sessionData.total || 0 }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">今日 Token 总量</div>
        <div class="stat-value">{{ tokenData.totalTokens }}</div>
      </div>
    </div>

    <!-- Token 消耗按 Agent 分布 -->
    <div class="grid-2">
      <div class="chart-panel">
        <h3>Token 消耗分布（按 Agent）</h3>
        <div ref="tokenPie" class="chart-box"></div>
      </div>
      <div class="chart-panel">
        <h3>Token 消耗趋势（按分钟）</h3>
        <div ref="tokenDistChart" class="chart-box"></div>
      </div>
    </div>

    <!-- 活跃会话列表 -->
    <div class="chart-panel">
      <h3>
        会话列表
        <span style="font-weight:normal;font-size:12px;color:#667788;margin-left:12px">
          <button class="btn btn-sm" :class="{ 'btn-primary': !filterStatus }" @click="filterStatus = ''; loadData()">全部</button>
          <button class="btn btn-sm" :class="{ 'btn-primary': filterStatus === 'active' }" @click="filterStatus = 'active'; loadData()" style="margin-left:4px">活跃</button>
          <button class="btn btn-sm" :class="{ 'btn-primary': filterStatus === 'dead' }" @click="filterStatus = 'dead'; loadData()" style="margin-left:4px">僵死</button>
        </span>
      </h3>
      <table class="data-table">
        <thead>
          <tr><th>会话 ID</th><th>用户</th><th>最后活动</th><th>闲置时间</th><th>Agent</th><th>消息数</th><th>状态</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="s in sessions" :key="s.id">
            <td style="font-family:monospace;font-size:11px">{{ s.id }}</td>
            <td>{{ s.user }}</td>
            <td>{{ s.last_active }}</td>
            <td>{{ s.idle_time }}</td>
            <td>{{ s.agent }}</td>
            <td>{{ s.message_count }}</td>
            <td><span class="badge" :class="s.is_dead ? 'badge-danger' : 'badge-success'">{{ s.is_dead ? '僵死' : '活跃' }}</span></td>
            <td><button class="btn btn-sm btn-danger" @click="killSession(s.id)">终止</button></td>
          </tr>
          <tr v-if="sessions.length === 0">
            <td colspan="8" style="text-align:center;color:#667788;padding:20px">暂无会话</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import echarts from '../utils/echarts.js'
import { api } from '../utils/api.js'

const loading = ref(false)
const filterStatus = ref('')
const sessionData = ref({ total: 0, active: 0, dead: 0 })
const sessions = ref([])
const tokenData = ref({ totalTokens: '--', by_agent: [] })
const tokenDistData = ref({ distribution: [] })
const tokenPie = ref(null)
const tokenDistChart = ref(null)
let charts = []

const loadData = async () => {
  loading.value = true
  try {
    const [sessionRes, tokenRes, tokenDistRes] = await Promise.all([
      api.getSessions(filterStatus.value === 'dead' ? 'dead' : filterStatus.value === 'active' ? 'active' : ''),
      api.getTokens(),
      api.getTokenDistribution(),
    ])

    sessionData.value = sessionRes
    sessions.value = sessionRes.sessions || []
    tokenData.value = {
      totalTokens: (tokenRes.total_tokens || 0).toLocaleString(),
      by_agent: tokenRes.by_agent || [],
    }
    tokenDistData.value = tokenDistRes
    updateCharts()
  } catch (e) {
    console.warn('Sessions API 不可用', e)
  } finally {
    loading.value = false
  }
}

const killSession = async (id) => {
  try {
    await api.killSession(id)
    loadData()
  } catch (e) { console.error(e) }
}

const makeTokenPie = () => {
  const agents = tokenData.value.by_agent || []
  return {
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie', radius: '65%', center: ['50%', '50%'],
      label: { color: '#8899aa', fontSize: 11, formatter: '{b}: {c}' },
      data: agents.map(a => ({
        value: a.total_tokens || 0,
        name: a.agent || 'unknown',
      })),
    }]
  }
}

const makeTokenDistChart = () => {
  const dist = tokenDistData.value.distribution || []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 10, bottom: 30 },
    xAxis: { type: 'category', data: dist.map(d => d.time), axisLabel: { color: '#667788', fontSize: 10 } },
    yAxis: { type: 'value', axisLabel: { color: '#667788' }, splitLine: { lineStyle: { color: '#1a2744' } } },
    series: [{
      type: 'bar', data: dist.map(d => d.tokens),
      itemStyle: { color: '#4dabf7', borderRadius: [4, 4, 0, 0] }
    }]
  }
}

const updateCharts = () => {
  if (charts[0]) charts[0].setOption(makeTokenPie())
  if (charts[1]) charts[1].setOption(makeTokenDistChart())
}

let resizeHandler = null
onMounted(() => {
  const c1 = echarts.init(tokenPie.value, 'dark'); c1.setOption(makeTokenPie())
  const c2 = echarts.init(tokenDistChart.value, 'dark'); c2.setOption(makeTokenDistChart())
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