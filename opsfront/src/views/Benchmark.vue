<template>
  <div>
    <div class="page-header">
      <h1>🎯 性能与评估</h1>
      <p>系统性能测试 + RAG 质量评估 <span v-if="loading" style="color:#4dabf7">(加载中...)</span></p>
    </div>

    <!-- Tab 切换 -->
    <div class="tab-bar">
      <button class="tab-btn" :class="{active: activeTab === 'performance'}" @click="activeTab = 'performance'">
        ⚡ 系统性能
      </button>
      <button class="tab-btn" :class="{active: activeTab === 'evaluation'}" @click="activeTab = 'evaluation'">
        🧪 RAG 评估
      </button>
      <button class="tab-btn" :class="{active: activeTab === 'custom'}" @click="activeTab = 'custom'">
        ✍️ 自定义评估
      </button>
    </div>

    <!-- ============ Tab 1: 系统性能（原 Benchmark 内容） ============ -->
    <div v-if="activeTab === 'performance'">
      <div style="display:flex;gap:12px;margin-bottom:16px">
        <button class="btn btn-success" @click="runBenchmark" :disabled="running">运行在线测试</button>
        <span v-if="running" style="color:#4dabf7;line-height:36px">测试运行中...</span>
        <span v-if="resultTimestamp" style="color:#8899aa;line-height:36px;margin-left:auto">最后测试: {{ resultTimestamp }}</span>
      </div>

      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-label">系统 QPS</div>
          <div class="stat-value" style="color:#51cf66">{{ benchmarkData.system_throughput?.qps || '--' }}</div>
          <div class="stat-change">实时</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">平均延迟</div>
          <div class="stat-value">{{ benchmarkData.system_throughput?.avg_latency_ms || '--' }}ms</div>
          <div class="stat-change">实时</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">错误率</div>
          <div class="stat-value" style="color:#ff6b6b">{{ ((benchmarkData.system_throughput?.error_rate || 0) * 100).toFixed(2) }}%</div>
          <div class="stat-change">实时</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">总请求数</div>
          <div class="stat-value">{{ benchmarkData.system_throughput?.total_requests || 0 }}</div>
          <div class="stat-change">累计</div>
        </div>
      </div>

      <div class="grid-2" style="margin-top:16px">
        <div class="chart-panel">
          <h3>RAG 质量雷达</h3>
          <div ref="ragRadar" class="chart-box"></div>
        </div>
        <div class="chart-panel">
          <h3>Agent 响应延迟</h3>
          <div ref="agentLatencyBar" class="chart-box"></div>
        </div>
      </div>

      <div class="chart-panel" style="margin-top:16px">
        <h3>Agent 性能明细</h3>
        <table class="data-table">
          <thead>
            <tr><th>Agent</th><th>平均延迟</th><th>P50</th><th>P99</th><th>最大延迟</th><th>请求数</th><th>状态</th></tr>
          </thead>
          <tbody>
            <tr v-for="(stats, name) in benchmarkData.agent_performance || {}" :key="name">
              <td><strong>{{ name }}</strong></td>
              <td>{{ stats.avg_ms }}ms</td>
              <td>{{ stats.p50_ms }}ms</td>
              <td>{{ stats.p99_ms }}ms</td>
              <td>{{ stats.max_ms }}ms</td>
              <td>{{ stats.requests }}</td>
              <td>
                <span class="badge" :class="(benchmarkData.agent_status || {})[name] === 'healthy' ? 'badge-success' : 'badge-warning'">
                  {{ (benchmarkData.agent_status || {})[name] || 'unknown' }}
                </span>
              </td>
            </tr>
            <tr v-if="!benchmarkData.agent_performance || Object.keys(benchmarkData.agent_performance).length === 0">
              <td colspan="7" style="text-align:center;color:#667788;padding:20px">暂无 Agent 性能数据</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="chart-panel" style="margin-top:16px" v-if="testResults.length > 0">
        <h3>最近测试结果</h3>
        <table class="data-table">
          <thead>
            <tr><th>测试项目</th><th>平均耗时</th><th>最小</th><th>最大</th><th>样本数</th><th>吞吐量</th></tr>
          </thead>
          <tbody>
            <tr v-for="t in testResults" :key="t.name">
              <td><strong>{{ t.name }}</strong></td>
              <td>{{ t.avg_ms || '--' }}ms</td>
              <td>{{ t.min_ms || '--' }}ms</td>
              <td>{{ t.max_ms || '--' }}ms</td>
              <td>{{ t.samples || t.concurrent_requests || '--' }}</td>
              <td>{{ t.throughput_qps ? t.throughput_qps + ' QPS' : '--' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ============ Tab 2: RAG 评估（内置测试集） ============ -->
    <div v-if="activeTab === 'evaluation'">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div style="display:flex;gap:12px;align-items:center">
          <select v-model="selectedCategory" @change="filterCases">
            <option value="">全部类别</option>
            <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
          </select>
          <button class="btn" @click="selectAllCases">全选</button>
          <button class="btn" @click="deselectAllCases">全不选</button>
          <button class="btn btn-primary" :disabled="evaluating || selectedCaseIds.length === 0" @click="runEvaluation">
            {{ evaluating ? '评估中...' : `开始评估 (${selectedCaseIds.length})` }}
          </button>
        </div>
        <div style="color:#8899aa;font-size:12px">
          共 {{ testCases.length }} 个内置测试用例
        </div>
      </div>

      <!-- 测试集展示 -->
      <div class="chart-panel" v-if="testCases.length > 0">
        <h3>内置测试集</h3>
        <div class="case-list">
          <div v-for="c in filteredCases" :key="c.id" class="case-item">
            <label class="case-checkbox">
              <input type="checkbox" v-model="selectedCaseIds" :value="c.id" />
              <span class="case-id-badge">{{ c.id }}</span>
              <span class="badge badge-info" style="margin-left:6px">{{ c.category }}</span>
              <span class="badge" style="margin-left:4px"
                :class="{'badge-success': c.difficulty === 'easy', 'badge-warning': c.difficulty === 'medium', 'badge-danger': c.difficulty === 'hard'}">
                {{ c.difficulty }}
              </span>
            </label>
            <div class="case-question">{{ c.question }}</div>
            <div class="case-meta" v-if="c.expected_answer"><strong>期望回答：</strong>{{ c.expected_answer }}</div>
            <div class="case-meta">
              <strong>关键词：</strong>
              <span class="kw-tag" v-for="kw in c.expected_keywords" :key="kw">✓ {{ kw }}</span>
              <span class="kw-tag kw-forbidden" v-for="kw in c.forbidden_keywords" :key="kw">✗ {{ kw }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 评估结果 -->
      <div v-if="lastEvalResult" class="result-section">
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">总体得分</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_overall)">{{ (lastEvalResult.summary.avg_overall * 100).toFixed(1) }}%</div>
            <div class="stat-change">{{ scoreText(lastEvalResult.summary.avg_overall) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">忠诚度 Faithfulness</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_faithfulness)">{{ (lastEvalResult.summary.avg_faithfulness * 100).toFixed(1) }}%</div>
            <div class="stat-change">回答是否基于上下文</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">检索精度 Context Precision</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_context_precision)">{{ (lastEvalResult.summary.avg_context_precision * 100).toFixed(1) }}%</div>
            <div class="stat-change">检索上下文是否相关</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">上下文召回 Context Recall</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_context_recall)">{{ (lastEvalResult.summary.avg_context_recall * 100).toFixed(1) }}%</div>
            <div class="stat-change">参考答案是否被覆盖</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">上下文相关 Context Relevancy</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_context_relevancy)">{{ (lastEvalResult.summary.avg_context_relevancy * 100).toFixed(1) }}%</div>
            <div class="stat-change">上下文对问题的支撑</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">答案相关 Answer Relevancy</div>
            <div class="stat-value" :class="scoreColor(lastEvalResult.summary.avg_answer_relevancy)">{{ (lastEvalResult.summary.avg_answer_relevancy * 100).toFixed(1) }}%</div>
            <div class="stat-change">回答是否紧扣问题</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">用例通过</div>
            <div class="stat-value">{{ lastEvalResult.summary.successful_cases }} / {{ lastEvalResult.summary.total_cases }}</div>
            <div class="stat-change">平均延迟 {{ lastEvalResult.summary.avg_latency_ms?.toFixed(0) || 0 }}ms</div>
          </div>
        </div>

        <div class="chart-panel" v-if="Object.keys(lastEvalResult.summary.per_category || {}).length > 0">
          <h3>按类别统计</h3>
          <table class="data-table">
            <thead>
              <tr><th>类别</th><th>用例数</th><th>忠诚度</th><th>检索精度</th><th>上下文召回</th><th>上下文相关</th><th>答案相关</th><th>总体</th></tr>
            </thead>
            <tbody>
              <tr v-for="(data, cat) in lastEvalResult.summary.per_category" :key="cat">
                <td><strong>{{ cat }}</strong></td>
                <td>{{ data.cases }}</td>
                <td :class="scoreColor(data.avg_faithfulness)">{{ (data.avg_faithfulness * 100).toFixed(1) }}%</td>
                <td :class="scoreColor(data.avg_context_precision)">{{ (data.avg_context_precision * 100).toFixed(1) }}%</td>
                <td :class="scoreColor(data.avg_context_recall)">{{ (data.avg_context_recall * 100).toFixed(1) }}%</td>
                <td :class="scoreColor(data.avg_context_relevancy)">{{ (data.avg_context_relevancy * 100).toFixed(1) }}%</td>
                <td :class="scoreColor(data.avg_answer_relevancy)">{{ (data.avg_answer_relevancy * 100).toFixed(1) }}%</td>
                <td :class="scoreColor(data.avg_overall)" style="font-weight:600">{{ (data.avg_overall * 100).toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="chart-panel">
          <h3>分用例详情</h3>
          <div class="case-detail-list">
            <div v-for="c in lastEvalResult.cases" :key="c.case_id" class="case-detail-item">
              <div class="case-detail-header" @click="toggleCase(c.case_id)">
                <div class="case-detail-header-left">
                  <span class="expand-icon">{{ expandedCases.includes(c.case_id) ? '▼' : '▶' }}</span>
                  <strong>{{ c.case_id }}</strong>
                  <span class="badge badge-info">{{ overallCaseScore(c).toFixed(1) }}%</span>
                  <span v-if="!c.success" class="badge badge-danger">失败</span>
                </div>
                <div class="case-detail-header-right">
                  <span class="badge badge-success" style="margin-right:4px">忠 {{ (c.metrics.faithfulness?.score * 100).toFixed(0) }}%</span>
                  <span class="badge badge-info" style="margin-right:4px">精 {{ (c.metrics.context_precision?.score * 100).toFixed(0) }}%</span>
                  <span class="badge badge-info" style="margin-right:4px">召 {{ (c.metrics.contextualrecall?.score * 100).toFixed(0) }}%</span>
                  <span class="badge badge-info" style="margin-right:4px">相 {{ (c.metrics.context_relevancy?.score * 100).toFixed(0) }}%</span>
                  <span class="badge badge-success">答 {{ (c.metrics.answer_relevancy?.score * 100).toFixed(0) }}%</span>
                </div>
              </div>
              <div v-if="expandedCases.includes(c.case_id)" class="case-detail-body">
                <div class="case-detail-row">
                  <strong>问题：</strong>{{ c.question }}
                </div>
                <div class="case-detail-row">
                  <strong>回答：</strong>
                  <pre class="answer-box">{{ c.generated_answer }}</pre>
                </div>
                <div v-for="(m, name) in c.metrics" :key="name" class="case-metric">
                  <div class="case-metric-header">
                    <span class="metric-name">{{ metricDisplayName(name) }}</span>
                    <span class="metric-score" :class="scoreColor(m.score)">{{ (m.score * 100).toFixed(1) }}%</span>
                  </div>
                  <div class="metric-reason">{{ m.reason }}</div>
                  <div v-if="m.evidence && m.evidence.length > 0">
                    <ul>
                      <li v-for="(e, i) in m.evidence" :key="i">{{ e }}</li>
                    </ul>
                  </div>
                </div>
                <div v-if="c.error" style="color:#ff6b6b">错误：{{ c.error }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-else-if="testCases.length === 0" class="empty-state">
        <div class="empty-icon">🎯</div>
        <p>未能加载内置测试集，请检查后端服务</p>
      </div>
      <div v-else class="empty-state">
        <div class="empty-icon">🧪</div>
        <p>选择测试用例并点击 "开始评估"，以获得 RAG 系统的五大核心指标评估（忠诚度 / 检索精度 / 上下文召回 / 上下文相关 / 答案相关）</p>
      </div>
    </div>

    <!-- ============ Tab 3: 自定义评估 ============ -->
    <div v-if="activeTab === 'custom'">
      <div class="chart-panel">
        <h3>自定义单次评估</h3>
        <p style="font-size:12px;color:#8899aa;margin-bottom:16px">
          输入问题、检索上下文与模型回答，即时评估单次 RAG 的五大指标（忠诚度 / 检索精度 / 上下文召回 / 上下文相关 / 答案相关）
        </p>
        <div class="form-group">
          <label>用户问题</label>
          <textarea v-model="customForm.question" placeholder="例如：我的订单什么时候能送到？" rows="2"></textarea>
        </div>
        <div class="form-group">
          <label>检索上下文（每行一个上下文，或用空行分隔）</label>
          <textarea v-model="customForm.contextsText" placeholder="上下文 1&#10;&#10;上下文 2" rows="6"></textarea>
        </div>
        <div class="form-group">
          <label>模型回答</label>
          <textarea v-model="customForm.answer" placeholder="模型生成的回答文本" rows="4"></textarea>
        </div>
        <div class="grid-2">
          <div class="form-group">
            <label>期望关键词（逗号分隔）</label>
            <input v-model="customForm.expectedKeywordsText" placeholder="订单,客服,物流" />
          </div>
          <div class="form-group">
            <label>禁用关键词（逗号分隔）</label>
            <input v-model="customForm.forbiddenKeywordsText" placeholder="南极分店,张三地址" />
          </div>
        </div>
        <div class="form-group">
          <label>参考回答（可选，用于更精细的生成质量评分）</label>
          <textarea v-model="customForm.reference" rows="2"></textarea>
        </div>
        <button class="btn btn-primary" :disabled="evaluatingCustom || !customForm.question || !customForm.answer" @click="runCustomEval">
          {{ evaluatingCustom ? '评估中...' : '立即评估' }}
        </button>
      </div>

      <div v-if="customResult" class="result-section">
        <div class="stat-grid">
          <div class="stat-card" v-for="(m, name) in customResult.metrics" :key="name">
            <div class="stat-label">{{ metricDisplayName(name) }}</div>
            <div class="stat-value" :class="scoreColor(m.score)">{{ (m.score * 100).toFixed(1) }}%</div>
            <div class="stat-change">{{ m.reason }}</div>
          </div>
        </div>

        <div class="chart-panel">
          <h3>详细证据</h3>
          <div v-for="(m, name) in customResult.metrics" :key="name" class="case-metric" style="margin-bottom:16px">
            <div class="case-metric-header">
              <span class="metric-name">{{ metricDisplayName(name) }}</span>
              <span class="metric-score" :class="scoreColor(m.score)">{{ (m.score * 100).toFixed(1) }}%</span>
            </div>
            <div v-if="m.evidence && m.evidence.length > 0">
              <ul>
                <li v-for="(e, i) in m.evidence" :key="i">{{ e }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import echarts from '../utils/echarts.js'
import { api } from '../utils/api.js'

// ========== 基础状态 ==========
const activeTab = ref('performance')
const loading = ref(false)
const running = ref(false)
const evaluating = ref(false)
const evaluatingCustom = ref(false)
const resultTimestamp = ref('')
const benchmarkData = ref({
  system_throughput: {},
  agent_performance: {},
  agent_status: {},
  rag_evaluation: {},
  node_id: '',
  global_clock: {},
})
const testResults = ref([])
const ragRadar = ref(null)
const agentLatencyBar = ref(null)
const charts = ref([])

// ========== RAG 评估状态 ==========
const testCases = ref([])
const categories = ref([])
const selectedCategory = ref('')
const selectedCaseIds = ref([])
const lastEvalResult = ref(null)
const expandedCases = ref([])

const filteredCases = computed(() => {
  if (!selectedCategory.value) return testCases.value
  return testCases.value.filter(c => c.category === selectedCategory.value)
})

// ========== 自定义评估 ==========
const customResult = ref(null)
const customForm = ref({
  question: '',
  contextsText: '',
  answer: '',
  expectedKeywordsText: '',
  forbiddenKeywordsText: '',
  reference: '',
})

// ========== 工具函数 ==========
function scoreColor(score) {
  const s = Number(score || 0)
  if (s >= 0.8) return 'up'
  if (s >= 0.6) return ''
  return 'down'
}

function scoreText(score) {
  const s = Number(score || 0)
  if (s >= 0.85) return '优秀'
  if (s >= 0.7) return '良好'
  if (s >= 0.5) return '一般'
  if (s > 0) return '较差'
  return '无数据'
}

function overallCaseScore(c) {
  const vals = Object.values(c.metrics || {}).map(m => Number(m.score || 0))
  if (!vals.length) return 0
  return (vals.reduce((a, b) => a + b, 0) / vals.length) * 100
}

function metricDisplayName(name) {
  const map = {
    faithfulness: '忠诚度 Faithfulness',
    context_precision: '检索精度 Context Precision',
    contextualrecall: '上下文召回 Context Recall',
    context_relevancy: '上下文相关 Context Relevancy',
    answer_relevancy: '答案相关 Answer Relevancy',
  }
  return map[name] || name
}

function toggleCase(id) {
  const idx = expandedCases.value.indexOf(id)
  if (idx >= 0) expandedCases.value.splice(idx, 1)
  else expandedCases.value.push(id)
}

function filterCases() {
  selectedCaseIds.value = filteredCases.value.map(c => c.id)
}

function selectAllCases() {
  selectedCaseIds.value = filteredCases.value.map(c => c.id)
}

function deselectAllCases() {
  selectedCaseIds.value = []
}

// ========== 图表 ==========
function makeRagRadar() {
  // 统一从 benchmark 数据取五指标（后台所有请求记录的统计平均值）
  // 不使用 DeepEval 评估结果作为雷达图的默认数据源（问题4）
  const rag = (benchmarkData.value && benchmarkData.value.rag_evaluation) || {}

  // 五指标取值（benchmark 的 rag_evaluation 字段名与 DeepEval 对齐）
  const faith = parseFloat(rag.avg_faithfulness) || 0
  const cp = parseFloat(rag.avg_context_precision) || 0
  const cr = parseFloat(rag.avg_context_recall) || 0
  const crel = parseFloat(rag.avg_context_relevancy) || 0
  const arel = parseFloat(rag.avg_answer_relevancy) || 0

  // 有真实统计数据（>0）的标志
  const hasRealData = (faith + cp + cr + crel + arel) > 0

  // 标题 / 标签
  const titleText = hasRealData ? 'RAG 质量雷达' : '暂无请求数据'
  const seriesLabel = hasRealData ? '实时统计（平均）' : '等待请求'

  return {
    title: {
      text: titleText,
      left: 'center',
      top: 5,
      textStyle: { color: '#8899aa', fontSize: 12 },
    },
    legend: {
      data: [seriesLabel],
      textStyle: { color: '#8899aa' },
      top: 28,
    },
    radar: {
      indicator: [
        { name: '忠诚度', max: 1 },
        { name: '检索精度', max: 1 },
        { name: '上下文召回', max: 1 },
        { name: '上下文相关', max: 1 },
        { name: '答案相关', max: 1 },
      ],
      axisName: { color: '#8899aa' },
      // 明确设置 center 和 radius，避免被顶部的 legend/title 遮挡
      center: ['50%', '57%'],
      radius: '60%',
      shape: 'polygon',
    },
    series: [{
      type: 'radar',
      symbol: 'circle',
      symbolSize: 5,
      data: [{
        value: [faith, cp, cr, crel, arel],
        name: seriesLabel,
        areaStyle: {
          color: hasRealData ? 'rgba(77,171,247,0.25)' : 'rgba(102,119,136,0.1)',
        },
        lineStyle: {
          color: hasRealData ? '#4dabf7' : '#8899aa',
          width: 2,
        },
        itemStyle: {
          color: hasRealData ? '#4dabf7' : '#8899aa',
        },
      }]
    }]
  }
}

function makeAgentLatencyBar() {
  const perf = (benchmarkData.value && benchmarkData.value.agent_performance) || {}
  const agent_status = (benchmarkData.value && benchmarkData.value.agent_status) || {}
  const names = Object.keys(perf)

  if (names.length === 0) {
    const fallbackNames = ['知识库问答', '工单处理', '闲聊对话', '路由分发', '结果汇总']
    return {
      title: {
        text: '等待请求数据...',
        left: 'center',
        top: 10,
        textStyle: { color: '#667788', fontSize: 12 }
      },
      tooltip: { trigger: 'axis' },
      legend: { data: ['平均', 'P50', 'P99'], textStyle: { color: '#8899aa' } },
      grid: { left: 100, right: 20, top: 40, bottom: 20 },
      xAxis: {
        type: 'value',
        axisLabel: { color: '#667788', formatter: '{value}ms' },
        splitLine: { lineStyle: { color: '#1a2744' } }
      },
      yAxis: {
        type: 'category',
        data: fallbackNames,
        axisLabel: { color: '#667788', fontSize: 11 }
      },
      series: [
        { name: '平均', type: 'bar', data: fallbackNames.map(() => 0), itemStyle: { color: '#2a3f5f', borderRadius: [0, 2, 2, 0] } },
        { name: 'P50', type: 'bar', data: fallbackNames.map(() => 0), itemStyle: { color: '#2a3f5f', borderRadius: [0, 2, 2, 0] } },
        { name: 'P99', type: 'bar', data: fallbackNames.map(() => 0), itemStyle: { color: '#2a3f5f', borderRadius: [0, 2, 2, 0] } },
      ]
    }
  }

  const hasRealData = names.some(n => ((perf[n] || {}).requests || 0) > 0)

  const avgData = names.map(n => {
    const p = perf[n] || {}
    if ((p.requests || 0) > 0) return p.avg_ms || 0
    return 0
  })
  const p50Data = names.map(n => {
    const p = perf[n] || {}
    if ((p.requests || 0) > 0) return p.p50_ms || 0
    return 0
  })
  const p99Data = names.map(n => {
    const p = perf[n] || {}
    if ((p.requests || 0) > 0) return p.p99_ms || 0
    return 0
  })

  const avgColor = hasRealData ? '#4dabf7' : '#2a3f5f'
  const p50Color = hasRealData ? '#51cf66' : '#2a3f5f'
  const p99Color = hasRealData ? '#ff6b6b' : '#2a3f5f'

  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const idx = params[0].dataIndex
        const name = names[idx]
        const p = perf[name] || {}
        const reqCount = p.requests || 0
        if (reqCount === 0) {
          return `${name}<br/>状态: ${agent_status[name] || 'healthy'}<br/>暂无请求数据`
        }
        let out = `${name}<br/>请求数: ${reqCount}<br/>`
        params.forEach(param => {
          out += `${param.marker} ${param.seriesName}: ${param.value}ms<br/>`
        })
        return out
      }
    },
    legend: { data: ['平均', 'P50', 'P99'], textStyle: { color: '#8899aa' } },
    grid: { left: 100, right: 20, top: 30, bottom: 20 },
    xAxis: { type: 'value', axisLabel: { color: '#667788', formatter: '{value}ms' }, splitLine: { lineStyle: { color: '#1a2744' } } },
    yAxis: { type: 'category', data: names, axisLabel: { color: '#8899aa', fontSize: 11 } },
    series: [
      { name: '平均', type: 'bar', data: avgData, itemStyle: { color: avgColor, borderRadius: [0, 2, 2, 0] } },
      { name: 'P50', type: 'bar', data: p50Data, itemStyle: { color: p50Color, borderRadius: [0, 2, 2, 0] } },
      { name: 'P99', type: 'bar', data: p99Data, itemStyle: { color: p99Color, borderRadius: [0, 2, 2, 0] } },
    ]
  }
}

function safeSetOption(chart, optionBuilder, notMerge = true) {
  if (!chart || typeof chart.setOption !== 'function') return
  try {
    const opt = optionBuilder()
    if (opt && Array.isArray(opt.series) && opt.series.length > 0) {
      chart.setOption(opt, notMerge)
    } else {
      console.warn('Chart option has no valid series')
    }
  } catch (e) {
    console.error('Chart render error:', e)
  }
}

function initCharts() {
  if (!ragRadar.value || !agentLatencyBar.value) return false
  try {
    charts.value.forEach(c => c && c.dispose && c.dispose())
    charts.value = [
      echarts.init(ragRadar.value),
      echarts.init(agentLatencyBar.value),
    ]
    return true
  } catch (e) {
    console.error('Failed to init charts:', e)
    return false
  }
}

function updateCharts() {
  if (ragRadar.value && agentLatencyBar.value) {
    const needReinit = !charts.value.length ||
      charts.value.some(c => !c) ||
      (charts.value[0] && charts.value[0].getDom && charts.value[0].getDom() !== ragRadar.value)

    if (needReinit) {
      if (!initCharts()) return
    }

    safeSetOption(charts.value[0], makeRagRadar, true)
    safeSetOption(charts.value[1], makeAgentLatencyBar, true)
    charts.value.forEach(c => c && c.resize && c.resize())
  }
}

// Tab 切换监听：当切换回性能 tab 时，DOM 已重建，需要重新初始化图表
watch(activeTab, (newTab) => {
  if (newTab === 'performance') {
    nextTick(() => {
      updateCharts()
    })
  }
})

// ========== 数据加载 ==========
async function loadData() {
  loading.value = true
  try {
    const data = await api.getBenchmark()
    if (data && typeof data === 'object') {
      benchmarkData.value = Object.assign({}, benchmarkData.value, data)
      updateCharts()
    }
  } catch (e) {
    console.warn('Benchmark API 不可用', e)
  } finally {
    loading.value = false
  }
}

async function runBenchmark() {
  running.value = true
  try {
    const data = await api.runBenchmark()
    testResults.value = data.tests || []
    resultTimestamp.value = data.timestamp || ''
    await loadData()
  } catch (e) {
    console.error('测试运行失败', e)
  } finally {
    running.value = false
  }
}

async function loadTestCases() {
  try {
    const data = await api.getEvalTestCases()
    if (data && data.cases) {
      testCases.value = data.cases || []
      categories.value = data.categories || []
      selectedCaseIds.value = testCases.value.map(c => c.id).filter(Boolean)
    } else {
      console.warn('评估测试集 API 返回数据格式异常')
    }
  } catch (e) {
    console.warn('评估测试集 API 不可用', e)
  }
}

async function runEvaluation() {
  if (!selectedCaseIds.value.length) return
  evaluating.value = true
  try {
    // api.runEvaluation 已内置异步轮询：提交任务 → 轮询 status → 返回 result
    // result 是 EvaluationSummary.to_dict()，格式:
    // { total_cases, successful_cases, avg_faithfulness, ..., per_category, cases: [...] }
    const data = await api.runEvaluation(selectedCaseIds.value)

    // 适配前端渲染期望的结构：{ summary: {...}, cases: [...] }
    // 注意：data 本身就是完整的 EvaluationSummary.to_dict()，里面已经有 cases
    const casesList = data.cases || []
    lastEvalResult.value = {
      summary: data,           // 供 stat-grid / summary 用：lastEvalResult.summary.avg_overall
      cases: casesList,        // 供分用例详情用：lastEvalResult.cases
    }
    expandedCases.value = casesList.slice(0, 3).map(c => c.case_id)

    // 评估完成后从后台刷新性能数据
    await loadData()
  } catch (e) {
    console.error('评估失败', e)
    alert('评估失败: ' + (e.message || '请查看后端日志'))
  } finally {
    evaluating.value = false
  }
}

async function runCustomEval() {
  if (!customForm.value.question || !customForm.value.answer) return
  evaluatingCustom.value = true
  customResult.value = null
  try {
    const contexts = customForm.value.contextsText.split(/\n\s*\n|\n{2,}/).map(s => s.trim()).filter(Boolean)
    const body = {
      question: customForm.value.question,
      contexts,
      answer: customForm.value.answer,
      reference: customForm.value.reference || undefined,
      expected_keywords: customForm.value.expectedKeywordsText.split(/[,，]/).map(s => s.trim()).filter(Boolean),
      forbidden_keywords: customForm.value.forbiddenKeywordsText.split(/[,，]/).map(s => s.trim()).filter(Boolean),
    }
    const data = await api.runCustomEvaluation(body)
    if (data.result) customResult.value = data.result
  } catch (e) {
    console.error('自定义评估失败', e)
    alert('评估失败，请查看后端日志')
  } finally {
    evaluatingCustom.value = false
  }
}

// ========== 生命周期 ==========
let resizeHandler = null
let autoRefreshTimer = null
onMounted(async () => {
  if (ragRadar.value && agentLatencyBar.value) {
    if (initCharts()) {
      safeSetOption(charts.value[0], makeRagRadar, true)
      safeSetOption(charts.value[1], makeAgentLatencyBar, true)
    }
    resizeHandler = () => charts.value.forEach(c => c && c.resize && c.resize())
    window.addEventListener('resize', resizeHandler)
  }
  await loadData()
  await loadTestCases()

  // 自动刷新性能数据（每 10 秒）
  autoRefreshTimer = setInterval(() => {
    if (activeTab.value === 'performance') {
      loadData()
    }
  }, 10000)
})

onUnmounted(() => {
  charts.value.forEach(c => c && c.dispose && c.dispose())
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
  if (autoRefreshTimer) clearInterval(autoRefreshTimer)
})
</script>

<style scoped>
.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid #1a2744;
}
.tab-btn {
  padding: 10px 20px;
  background: transparent;
  border: none;
  color: #8899aa;
  font-size: 13px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}
.tab-btn.active {
  color: #4dabf7;
  border-bottom-color: #4dabf7;
}
.tab-btn:hover { color: #c8d6e5; }

.case-list { display: flex; flex-direction: column; gap: 12px; }
.case-item { background: #0a0e27; padding: 14px; border-radius: 8px; border: 1px solid #243356; }
.case-checkbox { display: flex; align-items: center; margin-bottom: 8px; cursor: pointer; font-size: 13px; }
.case-checkbox input { margin-right: 8px; transform: scale(1.2); accent-color: #4dabf7; }
.case-id-badge { font-family: monospace; font-size: 12px; color: #4dabf7; }
.case-question { font-size: 14px; color: #c8d6e5; margin: 8px 0; padding-left: 24px; }
.case-meta { font-size: 12px; color: #8899aa; padding-left: 24px; line-height: 1.8; }
.kw-tag { display: inline-block; padding: 1px 8px; margin: 2px 4px 0 0; background: rgba(81,207,102,0.12); color: #51cf66; border-radius: 3px; font-size: 11px; }
.kw-forbidden { background: rgba(255,107,107,0.12); color: #ff6b6b; }

.result-section { margin-top: 20px; }
.case-detail-list { display: flex; flex-direction: column; gap: 8px; }
.case-detail-item { background: #0a0e27; border: 1px solid #243356; border-radius: 8px; overflow: hidden; }
.case-detail-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; transition: background 0.15s; }
.case-detail-header:hover { background: #1a2744; }
.case-detail-header-left { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.expand-icon { font-size: 10px; color: #667788; width: 12px; }
.case-detail-body { padding: 16px; border-top: 1px solid #243356; background: #0a0e27; }
.case-detail-row { margin-bottom: 12px; font-size: 13px; line-height: 1.7; }
.case-detail-row strong { color: #8899aa; display: inline-block; min-width: 64px; }
.answer-box { background: #0a0e27; border: 1px solid #243356; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; font-size: 13px; font-family: inherit; margin: 6px 0; }

.case-metric { background: #1a2744; padding: 10px 12px; border-radius: 6px; margin-bottom: 8px; }
.case-metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.metric-name { font-size: 12px; color: #8899aa; }
.metric-score { font-size: 13px; font-weight: 600; }
.metric-reason { font-size: 12px; color: #667788; margin-bottom: 4px; }
.case-metric ul { margin: 6px 0 0; padding-left: 18px; font-size: 12px; color: #c8d6e5; }
.case-metric li { margin-bottom: 4px; }

.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 12px; color: #8899aa; margin-bottom: 4px; }
.form-group input, .form-group textarea {
  width: 100%; padding: 8px 12px; background: #0a0e27; border: 1px solid #243356;
  border-radius: 4px; color: #c8d6e5; font-size: 13px; font-family: inherit;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
.stat-card { background: #1a2744; border-radius: 8px; padding: 20px; border: 1px solid #243356; }
.stat-card .stat-label { font-size: 12px; color: #667788; margin-bottom: 8px; }
.stat-card .stat-value { font-size: 28px; font-weight: 700; color: #fff; }
.stat-card .stat-change { font-size: 12px; margin-top: 4px; }
.stat-card .stat-change.up { color: #51cf66; }
.stat-card .stat-change.down { color: #ff6b6b; }

.chart-panel { background: #1a2744; border-radius: 8px; border: 1px solid #243356; padding: 20px; margin-bottom: 16px; }
.chart-panel h3 { font-size: 14px; color: #c8d6e5; margin-bottom: 16px; }
.chart-box { width: 100%; height: 300px; }

.btn { padding: 8px 16px; border: 1px solid #243356; border-radius: 6px; background: transparent; color: #c8d6e5; font-size: 13px; cursor: pointer; transition: all 0.2s; }
.btn:hover { background: #243356; }
.btn-primary { background: #4dabf7; border-color: #4dabf7; color: #fff; }
.btn-success { background: #51cf66; border-color: #51cf66; color: #fff; }

.filter-bar select, .filter-bar input { padding: 6px 12px; background: #0a0e27; border: 1px solid #243356; border-radius: 4px; color: #c8d6e5; font-size: 13px; }

.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.badge-success { background: rgba(81, 207, 102, 0.15); color: #51cf66; }
.badge-warning { background: rgba(255, 212, 59, 0.15); color: #ffd43b; }
.badge-danger { background: rgba(255, 107, 107, 0.15); color: #ff6b6b; }
.badge-info { background: rgba(77, 171, 247, 0.15); color: #4dabf7; }

.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { text-align: left; padding: 10px 12px; background: #0a0e27; color: #667788; font-weight: 500; border-bottom: 1px solid #243356; }
.data-table td { padding: 10px 12px; border-bottom: 1px solid #1a2744; color: #c8d6e5; }

.empty-state { text-align: center; padding: 60px 20px; color: #667788; }
.empty-state .empty-icon { font-size: 48px; margin-bottom: 12px; }
</style>
