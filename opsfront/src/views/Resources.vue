<template>
  <div>
    <div class="page-header">
      <h1>资源与日志</h1>
      <p>服务器资源、容器状态、实时错误日志与慢请求 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">CPU 使用率</div>
        <div class="stat-value">{{ resData.cpu_percent || '--' }}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">内存使用</div>
        <div class="stat-value">{{ resData.memory_used_gb || '--' }} / {{ resData.memory_total_gb || '--' }} GB</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">磁盘使用</div>
        <div class="stat-value">{{ resData.disk_used_gb || '--' }} / {{ resData.disk_total_gb || '--' }} GB</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">运行时间</div>
        <div class="stat-value" id="uptime">--</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="chart-panel">
        <h3>CPU & 内存趋势</h3>
        <div ref="resourceChart" class="chart-box"></div>
      </div>
      <div class="chart-panel">
        <h3>容器状态</h3>
        <div ref="containerChart" class="chart-box"></div>
      </div>
    </div>

    <div class="chart-panel">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3>实时错误日志</h3>
        <div>
          <select v-model="logLevel" @change="loadData" class="filter-select" style="margin-right:8px">
            <option value="">全部</option>
            <option value="ERROR">ERROR</option>
            <option value="WARNING">WARNING</option>
            <option value="INFO">INFO</option>
          </select>
          <button class="btn btn-sm" @click="clearLogs">清空日志</button>
        </div>
      </div>
      <div style="max-height:300px;overflow-y:auto;background:#0a0e27;border-radius:6px;padding:12px;font-family:monospace;font-size:12px">
        <div v-for="log in logs" :key="log.id" style="padding:2px 0;color: #8899aa">
          <span style="color:#667788">{{ log.time }}</span>
          <span :style="{color: log.level==='ERROR'?'#ff6b6b':log.level==='WARNING'?'#ffd43b':'#4dabf7', margin:'0 8px'}">[{{ log.level }}]</span>
          {{ log.message }}
        </div>
        <div v-if="logs.length === 0" style="text-align:center;color:#667788;padding:20px">暂无日志</div>
      </div>
    </div>

    <div class="chart-panel" style="margin-top:16px">
      <h3>慢请求 TOP10</h3>
      <table class="data-table">
        <thead>
          <tr><th>#</th><th>接口</th><th>耗时</th><th>Agent</th><th>时间</th></tr>
        </thead>
        <tbody>
          <tr v-for="(sr, idx) in resData.slow_requests || []" :key="idx">
            <td>{{ idx + 1 }}</td>
            <td><code>{{ sr.endpoint }}</code></td>
            <td><span :style="{color: parseInt(sr.duration)>5000?'#ff6b6b':'#ffd43b'}">{{ sr.duration }}</span></td>
            <td>{{ sr.agent || '-' }}</td>
            <td>{{ sr.time }}</td>
          </tr>
          <tr v-if="(resData.slow_requests || []).length === 0">
            <td colspan="5" style="text-align:center;color:#667788;padding:20px">暂无慢请求记录</td>
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
const logLevel = ref('')
const resData = ref({ cpu_percent: '--', memory_used_gb: '--', memory_total_gb: '--', disk_used_gb: '--', disk_total_gb: '--', containers: [], logs: [], slow_requests: [] })
const logs = ref([])
const resourceChart = ref(null)
const containerChart = ref(null)
let charts = []

const loadData = async () => {
  loading.value = true
  try {
    const [res, logData] = await Promise.all([
      api.getResources(),
      api.getLogs(logLevel.value),
    ])
    resData.value = res
    logs.value = logData.logs || res.logs || []
    updateCharts()
  } catch (e) {
    console.warn('Resources API 不可用', e)
  } finally {
    loading.value = false
  }
}

const clearLogs = async () => {
  try { await api.clearLogs(); loadData() } catch (e) {}
}

const makeResourceChart = () => ({
  tooltip: { trigger: 'axis' },
  grid: { left: 50, right: 20, top: 10, bottom: 30 },
  xAxis: { type: 'category', data: Array.from({length:12},(_,i)=>i*5+'s'), axisLabel: { color: '#667788', fontSize: 10 } },
  yAxis: { type: 'value', max: 100, axisLabel: { color: '#667788' }, splitLine: { lineStyle: { color: '#1a2744' } } },
  series: [
    { name: 'CPU', type: 'line', data: [45,48,52,49,62,58,55,53,51,48,50,52], smooth: true, lineStyle: { color: '#4dabf7', width: 2 }, areaStyle: { color: 'rgba(77,171,247,0.1)' }, symbol: 'none' },
    { name: '内存', type: 'line', data: [42,43,45,44,46,47,48,47,46,45,44,43], smooth: true, lineStyle: { color: '#51cf66', width: 2 }, areaStyle: { color: 'rgba(81,207,102,0.1)' }, symbol: 'none' },
  ]
})

const makeContainerChart = () => ({
  tooltip: { trigger: 'item' },
  series: [{
    type: 'pie', radius: '65%',
    label: { color: '#8899aa', fontSize: 11 },
    data: [
      { value: 3, name: '运行中', itemStyle: { color: '#51cf66' } },
      { value: 0, name: '停止', itemStyle: { color: '#ff6b6b' } },
    ]
  }]
})

const updateCharts = () => {
  charts[0]?.setOption(makeResourceChart())
  charts[1]?.setOption(makeContainerChart())
}

let resizeHandler = null
onMounted(() => {
  const c1 = echarts.init(resourceChart.value, 'dark'); c1.setOption(makeResourceChart())
  const c2 = echarts.init(containerChart.value, 'dark'); c2.setOption(makeContainerChart())
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