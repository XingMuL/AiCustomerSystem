<template>
  <div class="ops-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>运维平台</h2>
      </div>
      <nav class="sidebar-nav">
        <router-link v-for="item in menuItems" :key="item.path" :to="item.path" class="nav-item">
          <span class="nav-icon">{{ item.icon }}</span>
          <span class="nav-label">{{ item.label }}</span>
        </router-link>
      </nav>
      <div class="sidebar-footer">
        <span class="status-dot" :class="backendOnline ? 'online' : 'offline'"></span>
        {{ backendOnline ? '后端在线' : '后端离线' }}
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const backendOnline = ref(false)
const menuItems = [
  { path: '/', label: '总览大盘', icon: '📊' },
  { path: '/agents', label: 'Agent 实例', icon: '🤖' },
  { path: '/topology', label: '调用链路', icon: '🔗' },
  { path: '/sessions', label: '会话监控', icon: '💬' },
  { path: '/resources', label: '资源与日志', icon: '🖥️' },
  { path: '/alerts', label: '告警中心', icon: '🔔' },
  { path: '/benchmark', label: '性能与评估', icon: '🎯' },
  { path: '/knowledge', label: '知识库管理', icon: '📚' },
]

const checkBackend = async () => {
  try {
    const res = await fetch('/api/degradation/status')
    backendOnline.value = res.ok
  } catch { backendOnline.value = false }
}

onMounted(() => { checkBackend(); setInterval(checkBackend, 15000) })
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #0f1923; color: #c8d6e5; }

.ops-layout { display: flex; height: 100vh; }

/* Sidebar */
.sidebar { width: 200px; background: #0a0e27; display: flex; flex-direction: column; flex-shrink: 0; }
.sidebar-header { padding: 20px; border-bottom: 1px solid #1a2744; }
.sidebar-header h2 { font-size: 18px; color: #fff; }
.sidebar-nav { flex: 1; padding: 12px 0; overflow-y: auto; }
.nav-item { display: flex; align-items: center; gap: 10px; padding: 12px 20px; color: #8899aa; text-decoration: none; font-size: 14px; transition: all 0.2s; border-left: 3px solid transparent; }
.nav-item:hover { background: #1a2744; color: #c8d6e5; }
.nav-item.router-link-active { background: #1a2744; color: #4dabf7; border-left-color: #4dabf7; }
.nav-icon { font-size: 16px; width: 20px; text-align: center; }
.sidebar-footer { padding: 12px 20px; border-top: 1px solid #1a2744; font-size: 12px; color: #667788; display: flex; align-items: center; gap: 6px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.online { background: #51cf66; }
.status-dot.offline { background: #ff6b6b; }

/* Main */
.main-content { flex: 1; overflow-y: auto; padding: 24px; }

/* Page header */
.page-header { margin-bottom: 24px; }
.page-header h1 { font-size: 22px; color: #fff; margin-bottom: 4px; }
.page-header p { font-size: 13px; color: #667788; }

/* Cards */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
.stat-card { background: #1a2744; border-radius: 8px; padding: 20px; border: 1px solid #243356; }
.stat-card .stat-label { font-size: 12px; color: #667788; margin-bottom: 8px; }
.stat-card .stat-value { font-size: 28px; font-weight: 700; color: #fff; }
.stat-card .stat-change { font-size: 12px; margin-top: 4px; }
.stat-card .stat-change.up { color: #51cf66; }
.stat-card .stat-change.down { color: #ff6b6b; }

/* Charts */
.chart-panel { background: #1a2744; border-radius: 8px; border: 1px solid #243356; padding: 20px; margin-bottom: 16px; }
.chart-panel h3 { font-size: 14px; color: #c8d6e5; margin-bottom: 16px; }
.chart-box { width: 100%; height: 300px; }

/* Tables */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { text-align: left; padding: 10px 12px; background: #0a0e27; color: #667788; font-weight: 500; border-bottom: 1px solid #243356; }
.data-table td { padding: 10px 12px; border-bottom: 1px solid #1a2744; color: #c8d6e5; }
.data-table tr:hover td { background: #1a2744; }

/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.badge-success { background: rgba(81, 207, 102, 0.15); color: #51cf66; }
.badge-warning { background: rgba(255, 212, 59, 0.15); color: #ffd43b; }
.badge-danger { background: rgba(255, 107, 107, 0.15); color: #ff6b6b; }
.badge-info { background: rgba(77, 171, 247, 0.15); color: #4dabf7; }

/* Buttons */
.btn { padding: 8px 16px; border: 1px solid #243356; border-radius: 6px; background: transparent; color: #c8d6e5; font-size: 13px; cursor: pointer; transition: all 0.2s; }
.btn:hover { background: #243356; }
.btn-primary { background: #4dabf7; border-color: #4dabf7; color: #fff; }
.btn-primary:hover { background: #3b9ce0; }
.btn-danger { border-color: #ff6b6b; color: #ff6b6b; }
.btn-danger:hover { background: rgba(255, 107, 107, 0.15); }
.btn-sm { padding: 4px 10px; font-size: 12px; }

/* Filters */
.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-bar select, .filter-bar input { padding: 6px 12px; background: #0a0e27; border: 1px solid #243356; border-radius: 4px; color: #c8d6e5; font-size: 13px; }

/* Grid layouts */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid-2-1 { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }

/* Empty state */
.empty-state { text-align: center; padding: 60px 20px; color: #667788; }
.empty-state .empty-icon { font-size: 48px; margin-bottom: 12px; }

/* Pagination */
.pagination { display: flex; gap: 8px; justify-content: center; margin-top: 16px; }
.pagination button { padding: 6px 12px; border: 1px solid #243356; background: transparent; color: #c8d6e5; border-radius: 4px; cursor: pointer; }
.pagination button.active { background: #4dabf7; border-color: #4dabf7; }

/* Modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal-box { background: #1a2744; border: 1px solid #243356; border-radius: 12px; padding: 24px; min-width: 400px; max-width: 600px; max-height: 80vh; overflow-y: auto; }
.modal-box h3 { margin-bottom: 16px; color: #fff; }
.modal-box .form-group { margin-bottom: 12px; }
.modal-box .form-group label { display: block; font-size: 12px; color: #8899aa; margin-bottom: 4px; }
.modal-box .form-group input, .modal-box .form-group textarea, .modal-box .form-group select { width: 100%; padding: 8px 12px; background: #0a0e27; border: 1px solid #243356; border-radius: 4px; color: #c8d6e5; font-size: 13px; }
.modal-box .form-group textarea { min-height: 80px; resize: vertical; }
.modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
</style>