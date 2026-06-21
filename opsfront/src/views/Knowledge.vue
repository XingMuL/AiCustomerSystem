<template>
  <div>
    <div class="page-header">
      <h1>知识库文档更新</h1>
      <p>管理 Qdrant 向量知识库中的文档 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <div style="display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap">
      <input v-model="searchQuery" @keyup.enter="loadDocs" placeholder="搜索文档..." class="search-input" style="flex:1;min-width:200px">
      <select v-model="fileTypeFilter" @change="loadDocs" class="filter-select">
        <option value="">全部类型</option>
        <option value="pdf">PDF</option>
        <option value="md">Markdown</option>
        <option value="txt">Text</option>
        <option value="docx">DOCX</option>
        <option value="csv">CSV</option>
      </select>
      <input ref="fileInput" type="file" @change="handleFileUpload" style="display:none" accept=".pdf,.txt,.md,.docx,.csv" multiple>
      <button class="btn btn-success" @click="$refs.fileInput.click()">上传文档</button>
      <button v-if="selectedIds.length > 0" class="btn btn-danger" @click="batchDelete">批量删除 ({{ selectedIds.length }})</button>
    </div>

    <!-- 整体上传进度条 -->
    <div v-if="activeUploads.length > 0" class="upload-progress-panel">
      <h3 style="margin:0 0 12px 0">📤 正在上传 ({{ activeUploads.length }} 个文档)</h3>
      <div v-for="up in activeUploads" :key="up.doc_id" class="upload-progress-item">
        <!-- 文件名 + 主进度 + 百分比 -->
        <div class="upload-progress-header">
          <span class="upload-progress-name">{{ up.file_name }}</span>
          <span class="upload-progress-stage" :class="'stage-' + up.stage">{{ getStageLabel(up.stage) }}</span>
          <span class="upload-progress-percent" :class="{ percent_done: up.stage === 'done' || up.stage === 'error' }">{{ up.progress }}%</span>
        </div>
        <!-- 动画进度条 -->
        <div class="progress-bar-container" :class="{'progress-bar-error-wrap': up.stage === 'error', 'progress-bar-done-wrap': up.stage === 'done'}">
          <div
            class="progress-bar-fill"
            :class="{'progress-bar-error': up.stage === 'error', 'progress-bar-done': up.stage === 'done'}"
            :style="{width: up.progress + '%'}"
          ></div>
          <!-- 进度条内的流光动画（活跃时有呼吸效果 -->
          <div class="progress-bar-shimmer" v-if="up.stage !== 'done' && up.stage !== 'error'"></div>
        </div>
        <!-- 分阶段小圆点指示 -->
        <div class="stage-indicator">
          <div
            v-for="(stage, idx) in uploadStageList" :key="stage.key"
            class="stage-dot-wrapper"
            :class="{
              'stage-dot-active': stage.key === up.stage,
              'stage-dot-done': isStageDone(up.stage, stage.key, idx),
              'stage-dot-error': up.stage === 'error',
            }"
            :title="stage.label"
          >
            <span class="stage-dot">{{ idx + 1 }}</span>
            <span class="stage-dot-label">{{ stage.label }}</span>
          </div>
        </div>
        <!-- 详细说明（小字） -->
        <div class="upload-progress-message" v-if="up.message && up.stage !== 'done' && up.stage !== 'error'">
          <span class="msg-prefix">正在执行:</span> {{ up.message }}
        </div>
        <div class="upload-progress-message" v-if="up.stage === 'done'" style="color:#6ee7b7">
          ✅ 上传完成
        </div>
        <div class="upload-progress-message" v-if="up.stage === 'error'" style="color:#fca5a5">
          ❌ {{ up.message || '上传失败' }}
        </div>
        <!-- 耗时信息 -->
        <div class="upload-progress-elapsed">
          <span>已耗时 {{ up.elapsed_seconds.toFixed(1) }}s</span>
          <span v-if="up.stage !== 'done' && up.stage !== 'error'" style="margin-left: 12px">阶段: {{ getStageLabel(up.stage) }}</span>
        </div>
      </div>
    </div>

    <div class="chart-panel">
      <h3>文档列表 (共 {{ docs.length }} 份)</h3>
      <table class="data-table">
        <thead>
          <tr>
            <th><input type="checkbox" @change="toggleSelectAll" :checked="allSelected"></th>
            <th>文档名称</th>
            <th>类型</th>
            <th>大小</th>
            <th>向量数</th>
            <th>标签</th>
            <th>上传时间</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="doc in docs" :key="doc.id" :class="{ selected: doc.selected }">
            <td><input type="checkbox" v-model="doc.selected" @change="updateSelectedIds"></td>
            <td><strong>{{ doc.name }}</strong></td>
            <td><span class="badge badge-info">{{ doc.type }}</span></td>
            <td>{{ doc.size }}</td>
            <td>{{ doc.vectors }}</td>
            <td><span v-for="t in doc.tags" :key="t" class="badge" style="background:#243356;margin:1px">{{ t }}</span></td>
            <td style="font-size:12px">{{ doc.uploadTime }}</td>
            <td><span class="badge" :class="doc.status === '已索引' ? 'badge-success' : 'badge-warning'">{{ doc.status }}</span></td>
            <td>
              <button class="btn btn-sm btn-danger" @click="deleteDoc(doc.id)">删除</button>
            </td>
          </tr>
          <tr v-if="docs.length === 0 && !loading">
            <td colspan="8" style="text-align:center;color:#667788;padding:30px">暂无文档，请上传文档到知识库</td>
          </tr>
        </tbody>
      </table>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { api } from '../utils/api.js'

// 上传阶段定义：后端 pipeline 的 5 个阶段
const uploadStageList = [
  { key: 'initializing', label: '初始化' },
  { key: 'parsing', label: '解析文档' },
  { key: 'cleaning', label: '清洗文档' },
  { key: 'chunking', label: '切分文档' },
  { key: 'vectorizing', label: '向量化' },
  { key: 'storing', label: '存储向量' },
  { key: 'done', label: '完成' },
]

function getStageLabel(stageKey) {
  const found = uploadStageList.find(s => s.key === stageKey)
  return found ? found.label : (stageKey || '处理中')
}

// 判断某个阶段是否已完成（根据当前阶段的索引）
function isStageDone(currentStage, checkStage, checkIdx) {
  const currentIdx = uploadStageList.findIndex(s => s.key === currentStage)
  if (currentIdx < 0) return false
  if (currentStage === 'done') return checkStage !== 'done'  // 全部完成时，只有最后一个是 done 状态
  return checkIdx < currentIdx
}

const loading = ref(false)
const uploading = ref(false)
const searchQuery = ref('')
const fileTypeFilter = ref('')
const docs = ref([])

// 上传进度追踪
const activeUploads = ref([])          // 当前正在上传的文档进度列表
const _sseConnections = ref({})        // doc_id -> EventSource 映射
const _progressPollTimer = ref(null)   // 轮询刷新上传中列表的 timer
const _localTickTimer = ref(null)      // 本地计时器，每秒更新 elapsed_seconds
const _docIdAlias = ref({})            // temp_doc_id -> real_doc_id 别名映射（用于进度记录迁移）

const selectedIds = computed(() => docs.value.filter(d => d.selected).map(d => d.id))
const allSelected = computed(() => docs.value.length > 0 && docs.value.every(d => d.selected))

const toggleSelectAll = () => {
  const val = !allSelected.value
  docs.value.forEach(d => d.selected = val)
}
const updateSelectedIds = () => {}

// ---------- 进度条相关 ----------

function _removeUploadById(doc_id) {
  activeUploads.value = activeUploads.value.filter(u => u.doc_id !== doc_id)
  // 直接关闭与 doc_id 完全匹配的 SSE 连接
  if (_sseConnections.value[doc_id]) {
    try { _sseConnections.value[doc_id].close() } catch (e) {}
    delete _sseConnections.value[doc_id]
  }
  // 通过别名映射查找并关闭相关连接（tempDocId/realDocId 互查）
  for (const [tempId, realId] of Object.entries(_docIdAlias.value)) {
    if (realId === doc_id || tempId === doc_id) {
      if (_sseConnections.value[tempId]) {
        try { _sseConnections.value[tempId].close() } catch (e) {}
        delete _sseConnections.value[tempId]
      }
      if (_sseConnections.value[realId]) {
        try { _sseConnections.value[realId].close() } catch (e) {}
        delete _sseConnections.value[realId]
      }
      // 清理已处理的别名映射
      delete _docIdAlias.value[tempId]
    }
  }
}

function _addOrUpdateUpload(uploadData) {
  // 处理 ID 别名：如果 doc_id 已被映射到新 ID，先查找旧 ID 的记录
  const effectiveDocId = uploadData.doc_id
  let idx = activeUploads.value.findIndex(u => u.doc_id === effectiveDocId)

  // 如果直接找不到，尝试通过别名映射查找
  if (idx < 0) {
    for (const [tempId, realId] of Object.entries(_docIdAlias.value)) {
      if (realId === effectiveDocId) {
        const existingIdx = activeUploads.value.findIndex(u => u.doc_id === tempId)
        if (existingIdx >= 0) {
          // 将原有的 temp_id 记录更新 doc_id 为 real_id
          const existing = activeUploads.value[existingIdx]
          activeUploads.value[existingIdx] = {
            ...existing,
            ...uploadData,
            doc_id: effectiveDocId,
            _start_time: existing._start_time || Date.now() / 1000,
          }
          idx = existingIdx
          break
        }
      }
    }
  }

  if (idx >= 0) {
    const existing = activeUploads.value[idx]
    activeUploads.value[idx] = {
      ...existing,
      ...uploadData,
      _start_time: existing._start_time || Date.now() / 1000,
    }
  } else {
    activeUploads.value.push({
      ...uploadData,
      _start_time: Date.now() / 1000,
    })
  }
  // 完成或失败的任务 10 秒后从列表中移除
  if (uploadData.stage === 'done' || uploadData.stage === 'error') {
    setTimeout(() => _removeUploadById(uploadData.doc_id), 10000)
  }
}

function _startProgressTracking(doc_id, file_name) {
  // 先用一条占位记录，保证 UI 上立即显示
  _addOrUpdateUpload({
    doc_id,
    file_name,
    stage: 'initializing',
    stage_name: '初始化',
    progress: 0,
    message: '文件上传中...',
    elapsed_seconds: 0,
  })

  // 开启 SSE 流式订阅
  try {
    const es = new EventSource(`/api/ops/knowledge/documents/progress/stream/${doc_id}`)
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        // 若后端返回真实 doc_id，则替换占位
        const realDocId = data.doc_id || doc_id
        _addOrUpdateUpload({ ...data, doc_id: realDocId })
      } catch (err) {
        console.warn('进度事件解析失败:', err)
      }
    }
    es.onerror = (e) => {
      console.warn('SSE 连接中断，将通过轮询补充:', e)
      es.close()
    }
    _sseConnections.value[doc_id] = es
  } catch (e) {
    console.warn('浏览器不支持 SSE，改用轮询:', e)
  }
}

function _stopAllProgressTracking() {
  for (const doc_id in _sseConnections.value) {
    try { _sseConnections.value[doc_id].close() } catch (e) {}
  }
  _sseConnections.value = {}
  if (_progressPollTimer.value) {
    clearInterval(_progressPollTimer.value)
    _progressPollTimer.value = null
  }
  if (_localTickTimer.value) {
    clearInterval(_localTickTimer.value)
    _localTickTimer.value = null
  }
}

// ---------- 上传 ----------

const handleFileUpload = async (e) => {
  const files = e.target.files
  if (!files || files.length === 0) return

  uploading.value = true
  // 循环逐个上传（避免并发过高）
  for (const file of files) {
    // 为每个文件预先创建一个占位 doc_id（与后端共享同一个 progress_id）
    const tempDocId = `upload_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    _startProgressTracking(tempDocId, file.name)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('tags', '')
      formData.append('rebuild', 'false')
      formData.append('progress_id', tempDocId)

      const resp = await fetch('/api/ops/knowledge/documents/upload', { method: 'POST', body: formData })
      // fetch 完成后，立即关闭对应 tempDocId 的 SSE 连接（避免记录已迁移到 real_doc_id）
      if (_sseConnections.value[tempDocId]) {
        try { _sseConnections.value[tempDocId].close() } catch (e) {}
        delete _sseConnections.value[tempDocId]
      }
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: '上传失败' }))
        _addOrUpdateUpload({
          doc_id: tempDocId,
          file_name: file.name,
          stage: 'error',
          stage_name: '上传失败',
          progress: 0,
          message: err.detail || '未知错误',
        })
        continue
      }
      const data = await resp.json()
      // 上传完成后，用真实 doc_id 更新占位，并注册 ID 别名
      const realDocId = data.doc_id || tempDocId
      if (realDocId !== tempDocId) {
        _docIdAlias.value[tempDocId] = realDocId
      }
      _addOrUpdateUpload({
        doc_id: realDocId,
        file_name: file.name,
        stage: 'done',
        stage_name: '已完成',
        progress: 100,
        message: `上传完成，共 ${data.child_chunks || 0} 个子块`,
      })
    } catch (err) {
      console.error('上传失败:', file.name, err)
      _addOrUpdateUpload({
        doc_id: tempDocId,
        file_name: file.name,
        stage: 'error',
        stage_name: '上传失败',
        progress: 0,
        message: err.message || '网络错误',
      })
    }
  }

  uploading.value = false
  e.target.value = ''
  // 上传结束后刷新文档列表
  loadDocs()
}

// ---------- 其他操作 ----------

const loadDocs = async () => {
  loading.value = true
  try {
    const data = await api.getKnowledgeDocs(searchQuery.value, fileTypeFilter.value)
    docs.value = (data.documents || []).map(d => ({ ...d, selected: !!docs.value.find(o => o.id === d.id)?.selected }))
  } catch (e) {
    console.warn('Knowledge API 不可用', e)
  } finally {
    loading.value = false
  }
}

const deleteDoc = async (id) => {
  if (!confirm('确认删除此文档？')) return
  try {
    await api.deleteDoc(id)
    loadDocs()
  } catch (e) { console.error(e) }
}

const batchDelete = async () => {
  if (!confirm(`确认删除选中的 ${selectedIds.value.length} 份文档？`)) return
  try {
    await api.batchDeleteDocs(selectedIds.value)
    loadDocs()
  } catch (e) { console.error(e) }
}

onMounted(() => {
  loadDocs()
  // 周期刷新当前上传中任务（作为 SSE 的补充）
  _progressPollTimer.value = setInterval(async () => {
    try {
      const resp = await fetch('/api/ops/knowledge/documents/progress')
      if (resp.ok) {
        const data = await resp.json()
        if (data.active_uploads && data.active_uploads.length > 0) {
          for (const up of data.active_uploads) {
            _addOrUpdateUpload(up)
          }
        }
      }
    } catch (e) { /* 静默 */ }
  }, 2000)
  // 本地计时器：每秒更新 activeUploads 的 elapsed_seconds
  _localTickTimer.value = setInterval(() => {
    const now = Date.now() / 1000
    for (const up of activeUploads.value) {
      if (up.stage === 'done' || up.stage === 'error') continue
      if (up._start_time) {
        up.elapsed_seconds = Math.round((now - up._start_time) * 10) / 10
      }
    }
  }, 1000)
})

onUnmounted(() => {
  _stopAllProgressTracking()
})
</script>

<style scoped>
.upload-progress-panel {
  margin-bottom: 16px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(77,171,247,0.10) 0%, rgba(52,120,180,0.06) 100%);
  border: 1px solid rgba(77,171,247,0.3);
  border-radius: 10px;
  color: #a8c9e8;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.upload-progress-item {
  padding: 14px 0;
  border-bottom: 1px dashed rgba(77,171,247,0.12);
}
.upload-progress-item:last-child { border-bottom: none; padding-bottom: 4px; }
.upload-progress-item:first-child { padding-top: 0; }
.upload-progress-header {
  display:flex; justify-content: space-between; align-items:center; gap: 12px;
  margin-bottom: 10px; font-size: 13px;
}
.upload-progress-name { font-weight: 600; color: #e8f1f8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 40%; }
.upload-progress-stage {
  color: #4dabf7;
  font-size: 12px;
  padding: 3px 10px;
  background: rgba(77,171,247,0.18);
  border-radius: 12px;
  border: 1px solid rgba(77,171,247,0.3);
  font-weight: 500;
}
.upload-progress-stage.stage-done { color: #51cf66; background: rgba(81,207,102,0.15); border-color: rgba(81,207,102,0.3); }
.upload-progress-stage.stage-error { color: #ff8787; background: rgba(255,107,107,0.15); border-color: rgba(255,107,107,0.3); }
.upload-progress-stage.stage-vectorizing { color: #fab005; background: rgba(250,176,5,0.15); border-color: rgba(250,176,5,0.3); }
.upload-progress-stage.stage-storing { color: #cc5de8; background: rgba(204,93,232,0.15); border-color: rgba(204,93,232,0.3); }
.upload-progress-percent { font-weight: 700; color: #4dabf7; min-width: 50px; text-align: right; font-size: 14px; }
.upload-progress-percent.percent_done { color: #51cf66; }

/* 进度条容器 - 更粗更显眼 */
.progress-bar-container {
  width: 100%;
  height: 14px;
  background: #0f1530;
  border-radius: 7px;
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}
.progress-bar-done-wrap { background: rgba(81,207,102,0.1); }
.progress-bar-error-wrap { background: rgba(255,107,107,0.1); }

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #4dabf7 0%, #228be6 100%);
  transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 7px;
  position: relative;
  box-shadow: 0 0 10px rgba(77,171,247,0.5);
}
.progress-bar-done {
  background: linear-gradient(90deg, #51cf66 0%, #37b24d 100%);
  box-shadow: 0 0 12px rgba(81,207,102,0.5);
}
.progress-bar-error {
  background: linear-gradient(90deg, #ff6b6b 0%, #c92a2a 100%);
  box-shadow: 0 0 12px rgba(255,107,107,0.5);
}

/* 进度条流光动画 */
.progress-bar-shimmer {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255,255,255,0.3) 50%,
    transparent 100%
  );
  animation: shimmer 2s infinite;
  pointer-events: none;
}
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* 分阶段小圆点指示 */
.stage-indicator {
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  margin-bottom: 8px;
  padding: 0 2px;
}
.stage-dot-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  position: relative;
}
.stage-dot-wrapper:not(:last-child)::after {
  content: '';
  position: absolute;
  top: 10px;
  left: 60%;
  right: -40%;
  height: 2px;
  background: #2d3757;
  z-index: 0;
}
.stage-dot-wrapper.stage-dot-done:not(:last-child)::after {
  background: #51cf66;
}
.stage-dot-wrapper.stage-dot-active:not(:last-child)::after {
  background: linear-gradient(90deg, #51cf66 0%, #2d3757 100%);
}
.stage-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #2d3757;
  color: #5a6a8a;
  font-size: 11px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid #3d4a6e;
  position: relative;
  z-index: 1;
  transition: all 0.3s ease;
}
.stage-dot-done .stage-dot {
  background: #51cf66;
  color: #fff;
  border-color: #51cf66;
  box-shadow: 0 0 8px rgba(81,207,102,0.5);
}
.stage-dot-active .stage-dot {
  background: #4dabf7;
  color: #fff;
  border-color: #4dabf7;
  box-shadow: 0 0 12px rgba(77,171,247,0.8);
  animation: pulse 1.5s infinite;
}
.stage-dot-error .stage-dot {
  background: #ff6b6b;
  color: #fff;
  border-color: #ff6b6b;
}
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.15); }
}
.stage-dot-label {
  margin-top: 6px;
  font-size: 10px;
  color: #5a6a8a;
  white-space: nowrap;
  text-align: center;
}
.stage-dot-done .stage-dot-label { color: #8bdca2; }
.stage-dot-active .stage-dot-label { color: #74c0fc; font-weight: 500; }
.stage-dot-error .stage-dot-label { color: #ffa8a8; }

.upload-progress-message {
  margin-top: 10px;
  font-size: 12px;
  color: #8899aa;
  line-height: 1.6;
}
.msg-prefix {
  color: #4dabf7;
  font-weight: 500;
  margin-right: 4px;
}
.upload-progress-elapsed {
  margin-top: 6px;
  font-size: 11px;
  color: #556677;
}
</style>
