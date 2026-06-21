<template>
  <div>
    <div class="page-header">
      <h1>告警 & 降级状态</h1>
      <p>系统降级等级、熔断器状态与故障恢复 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <!-- 降级状态卡片 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">降级等级</div>
        <div class="stat-value" :style="{color: degradationLevel === 0 ? '#51cf66' : degradationLevel >= 3 ? '#ff6b6b' : '#ffd43b'}">
          L{{ degradationLevel }}
        </div>
        <div class="stat-change">{{ levelDesc }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">LLM 状态</div>
        <div class="stat-value" :style="{color: llmAvailable ? '#51cf66' : '#ff6b6b'}">
          {{ llmAvailable ? '可用' : '不可用' }}
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">RAG 状态</div>
        <div class="stat-value" :style="{color: ragAvailable ? '#51cf66' : '#ff6b6b'}">
          {{ ragAvailable ? '可用' : '不可用' }}
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">熔断器数</div>
        <div class="stat-value">{{ circuitBreakerCount }}</div>
      </div>
    </div>

    <!-- 降级等级说明 -->
    <div class="chart-panel">
      <h3>降级等级说明</h3>
      <div class="degradation-bar">
        <div v-for="lvl in degradationLevels" :key="lvl.level" class="degradation-level" :class="{ active: lvl.active }">
          <div class="lvl-indicator">{{ lvl.label }}</div>
          <div class="lvl-desc">{{ lvl.desc }}</div>
        </div>
      </div>
    </div>

    <!-- 熔断器状态 -->
    <div class="chart-panel" style="margin-top:16px">
      <h3>熔断器状态</h3>
      <table class="data-table">
        <thead>
          <tr><th>Agent</th><th>熔断状态</th><th>连续失败</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="(info, agent) in (circuitBreakers || {})" :key="agent">
            <td><strong>{{ agent }}</strong></td>
            <td>
              <span class="badge" :class="info.is_open ? 'badge-danger' : 'badge-success'">
                {{ info.is_open ? '已熔断' : '正常' }}
              </span>
            </td>
            <td>{{ info.failures || 0 }}</td>
            <td>
              <button class="btn btn-sm btn-success" @click="resetDegradation">恢复</button>
            </td>
          </tr>
          <tr v-if="!circuitBreakers || Object.keys(circuitBreakers).length === 0">
            <td colspan="4" style="text-align:center;color:#667788;padding:20px">所有 Agent 正常运行，无熔断</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- LLM / RAG 控制 -->
    <div class="grid-2" style="margin-top:16px">
      <div class="chart-panel">
        <h3>手动控制</h3>
        <div style="padding:16px">
          <div style="margin-bottom:12px">
            <button class="btn btn-danger" style="margin-right:8px" @click="setLLM(false)">关闭 LLM</button>
            <button class="btn btn-success" @click="setLLM(true)">开启 LLM</button>
          </div>
          <div>
            <button class="btn btn-danger" style="margin-right:8px" @click="setRAG(false)">关闭 RAG</button>
            <button class="btn btn-success" @click="setRAG(true)">开启 RAG</button>
          </div>
        </div>
      </div>
      <div class="chart-panel">
        <h3>恢复操作</h3>
        <div style="padding:16px">
          <button class="btn btn-success" style="width:100%" @click="resetAll">重置所有降级状态</button>
          <p style="font-size:12px;color:#667788;margin-top:8px">将所有降级等级恢复到 L0，清除所有熔断器状态</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../utils/api.js'

const loading = ref(false)
const degradationLevel = ref(0)
const llmAvailable = ref(true)
const ragAvailable = ref(true)
const circuitBreakers = ref({})
const circuitBreakerCount = computed(() => Object.keys(circuitBreakers.value).length)

const levelDescMap = {
  0: '正常运行',
  1: '轻度降级 - 降级非核心 Agent',
  2: '局部故障 - 开启熔断器',
  3: '中度降级 - 全部降级',
  4: '全局故障 - 返回兜底回复',
}
const levelDesc = computed(() => levelDescMap[degradationLevel.value] || '未知')

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
    const data = await api.getAlerts()
    degradationLevel.value = data.degradation_level || 0
    llmAvailable.value = data.llm_available !== false
    ragAvailable.value = data.rag_available !== false
    circuitBreakers.value = data.circuit_breakers || {}

    degradationLevels.value = degradationLevels.value.map((l, i) => ({
      ...l,
      active: i === degradationLevel.value,
    }))
  } catch (e) {
    console.warn('Alerts API 不可用', e)
  } finally {
    loading.value = false
  }
}

const resetAll = async () => {
  try {
    await api.resetDegradation()
    loadData()
  } catch (e) {
    console.error(e)
  }
}

const resetDegradation = resetAll

const setLLM = async (available) => {
  try {
    await fetch(`/api/degradation/llm/${available ? 'on' : 'off'}`, { method: 'POST' })
    loadData()
  } catch (e) {
    console.error(e)
  }
}

const setRAG = async (available) => {
  try {
    await fetch(`/api/degradation/llm/${available ? 'on' : 'off'}`, { method: 'POST' })
    loadData()
  } catch (e) {
    console.error(e)
  }
}

onMounted(() => loadData())
</script>