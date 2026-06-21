<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <span class="logo">🤖</span>
        <h1>智能客服系统</h1>
      </div>
      <div class="header-right">
        <span class="status-badge" :class="statusClass">
          {{ statusText }}
        </span>
      </div>
    </header>
    <main class="app-main">
      <ChatWindow />
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import ChatWindow from './components/ChatWindow.vue'

const degradationLevel = ref(0)

const statusClass = computed(() => {
  if (degradationLevel.value === 0) return 'status-healthy'
  if (degradationLevel.value <= 2) return 'status-degraded'
  return 'status-critical'
})

const statusText = computed(() => {
  const labels = {
    0: '系统正常',
    1: '轻度降级',
    2: '局部故障',
    3: '中度降级',
    4: '全局故障'
  }
  return labels[degradationLevel.value] || '未知'
})

let statusTimer = null

const checkStatus = async () => {
  try {
    const res = await fetch('/api/degradation/status')
    const data = await res.json()
    degradationLevel.value = data.degradation_level
  } catch (e) {
    // ignore
  }
}

onMounted(() => {
  checkStatus()
  statusTimer = setInterval(checkStatus, 10000)
})

onUnmounted(() => {
  if (statusTimer) clearInterval(statusTimer)
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #f0f2f5;
  height: 100vh;
  overflow: hidden;
}

.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 900px;
  margin: 0 auto;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  font-size: 28px;
}

.header-left h1 {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 1px;
}

.status-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  background: rgba(255, 255, 255, 0.2);
}

.status-healthy {
  background: rgba(82, 196, 26, 0.3);
}

.status-degraded {
  background: rgba(250, 173, 20, 0.3);
}

.status-critical {
  background: rgba(255, 77, 79, 0.3);
}

.app-main {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>