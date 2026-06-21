const API_BASE = '/api'

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  // 总览大盘
  getDashboard: () => fetchJSON(`${API_BASE}/ops/dashboard`),

  // Agent 实例列表
  getAgents: () => fetchJSON(`${API_BASE}/ops/agents`),
  restartAgent: (name) => fetchJSON(`${API_BASE}/ops/agents/${name}/restart`, { method: 'POST' }),

  // 调用链路（Gossip 拓扑）
  getTopology: () => fetchJSON(`${API_BASE}/gossip/status`),

  // 会话监控（真实后台数据）
  getSessions: (filter = '') => {
    const params = filter ? `?filter_status=${filter}` : ''
    return fetchJSON(`${API_BASE}/ops/sessions${params}`)
  },
  killSession: (id) => fetchJSON(`${API_BASE}/ops/sessions/${id}/kill`, { method: 'POST' }),

  // Token 消耗
  getTokens: (period = '') => {
    const params = period ? `?period=${period}` : ''
    return fetchJSON(`${API_BASE}/ops/tokens${params}`)
  },
  getTokenDistribution: () => fetchJSON(`${API_BASE}/ops/tokens/distribution`),

  // 系统性能
  getPerformance: () => fetchJSON(`${API_BASE}/ops/performance`),

  // 告警中心（降级状态）
  getAlerts: () => fetchJSON(`${API_BASE}/degradation/status`),
  resetDegradation: () => fetchJSON(`${API_BASE}/degradation/reset`, { method: 'POST' }),

  // 性能测试（真实测试数据）
  getBenchmark: () => fetchJSON(`${API_BASE}/ops/benchmark`),
  runBenchmark: () => fetchJSON(`${API_BASE}/ops/benchmark/run`),

  // RAG 评估（异步模式：先提交任务 → 轮询状态 → 返回结果）
  getEvalTestCases: () => fetchJSON(`${API_BASE}/ops/evaluation/test-cases`),
  getEvaluationStatus: (taskId) => fetchJSON(`${API_BASE}/ops/evaluation/status/${taskId}`),
  runEvaluation: (caseIds = null) => new Promise((resolve, reject) => {
    const POLL_INTERVAL = 3000  // 每 3 秒轮询一次
    const MAX_WAIT = 1800000    // 最长等待 30 分钟（与后端 eval_max_wait_seconds 对齐）
    const startTs = Date.now()

    // 1. 先提交任务
    fetchJSON(`${API_BASE}/ops/evaluation/run`, {
      method: 'POST',
      body: caseIds ? JSON.stringify({ case_ids: caseIds }) : '{}',
    }).then((resp) => {
      const taskId = resp.task_id
      if (!taskId) {
        // 退化：如果后端是老版本直接返回了结果（status === completed）
        if (resp.status === 'completed' && resp.summary) return resolve(resp)
        return reject(new Error('后端未返回 task_id'))
      }

      // 2. 轮询状态
      function poll() {
        if (Date.now() - startTs > MAX_WAIT) {
          return reject(new Error('评估超时'))
        }
        fetchJSON(`${API_BASE}/ops/evaluation/status/${taskId}`)
          .then((statusResp) => {
            if (statusResp.status === 'completed') {
              // 评估完成：返回 result（即 EvaluationSummary.to_dict()）
              // 格式：{ total_cases, successful_cases, avg_xxx, per_category, cases: [...] }
              return resolve(statusResp.result)
            } else if (statusResp.status === 'failed') {
              return reject(new Error(statusResp.error || '评估失败'))
            } else if (statusResp.status === 'not_found') {
              return reject(new Error('任务不存在或已过期'))
            } else {
              // pending / running：继续轮询
              setTimeout(poll, POLL_INTERVAL)
            }
          })
          .catch((err) => {
            // 网络错误等：稍等再试几次
            setTimeout(poll, POLL_INTERVAL * 2)
          })
      }
      poll()
    }).catch(reject)
  }),
  runCustomEvaluation: (payload) => fetchJSON(`${API_BASE}/ops/evaluation/custom`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  // 资源/系统状态
  getResources: () => fetchJSON(`${API_BASE}/ops/resources`),

  // 知识库管理
  getKnowledgeDocs: (search = '', fileType = '') => {
    const params = new URLSearchParams()
    if (search) params.set('search', search)
    if (fileType) params.set('file_type', fileType)
    return fetchJSON(`${API_BASE}/ops/knowledge/documents?${params}`)
  },
  uploadDoc: async (file, tags = '', rebuild = false) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('tags', tags)
    formData.append('rebuild', rebuild)
    const res = await fetch(`${API_BASE}/ops/knowledge/documents/upload`, { method: 'POST', body: formData })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: '上传失败' }))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    return res.json()
  },
  deleteDoc: (id) => fetchJSON(`${API_BASE}/ops/knowledge/documents/${id}`, { method: 'DELETE' }),
  batchDeleteDocs: (ids) => fetchJSON(`${API_BASE}/ops/knowledge/documents/batch-delete`, { method: 'POST', body: JSON.stringify({ ids }) }),
  reindexDoc: (id) => fetchJSON(`${API_BASE}/ops/knowledge/documents/${id}/reindex`, { method: 'POST' }),
  getLogs: (level = '') => fetchJSON(`${API_BASE}/ops/resources/logs?level=${level}`),
  clearLogs: () => fetchJSON(`${API_BASE}/ops/resources/logs`, { method: 'DELETE' }),
  ackAlert: (agent) => fetchJSON(`${API_BASE}/degradation/reset`, { method: 'POST' }),
  resolveAlert: (agent) => fetchJSON(`${API_BASE}/degradation/reset`, { method: 'POST' }),
}