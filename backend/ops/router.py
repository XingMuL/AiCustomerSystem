"""
运维监控 API 路由

提供：
  - 会话实时监控（真实数据）
  - Agent 实例管理（重启、状态查看）
  - Token 消耗统计（真实累计）
  - 系统性能分布
  - 在线性能测试案例获取实际数据
"""

import time
import os
import tempfile
import psutil
import json
import asyncio
import gc
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.graph.state_graph import get_state_graph
from backend.memory.gossip_coordinator import get_gossip_coordinator
from backend.ops.metrics_collector import get_metrics_collector
from backend.degradation.degradation_manager import get_degradation_manager
from backend.ops.progress_store import get_progress_store

router = APIRouter(prefix="/ops", tags=["运维监控"])

# ================================================================
#  RAG 评估任务管理（异步后台执行 + 轮询）
# ================================================================
# 任务状态存储：{task_id: {"status": "pending|running|completed|failed",
#                           "result": {...},
#                           "error": "...",
#                           "created_at": timestamp,
#                           "updated_at": timestamp,
#                           "progress": {"current": N, "total": N}}}
_eval_task_store: dict = {}
_eval_task_lock = None  # 延迟初始化 asyncio.Lock


def _get_eval_lock():
    """延迟初始化锁，避免在非事件循环线程中创建 asyncio.Lock"""
    global _eval_task_lock
    if _eval_task_lock is None:
        try:
            _eval_task_lock = asyncio.Lock()
        except RuntimeError:
            # 非 async 上下文时降级为简单的 dict（单线程写一般没问题）
            _eval_task_lock = None
    return _eval_task_lock


import uuid
import threading
import math as _math


def _safe_float_json(v):
    """递归确保值 JSON 可序列化：NaN/Inf → 0.0，非基本类型 → str"""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        fv = float(v)
        if _math.isnan(fv) or _math.isinf(fv):
            return 0.0
        return fv
    if isinstance(v, dict):
        return {str(k): _safe_float_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_float_json(x) for x in v]
    try:
        # 尝试转 float
        fv = float(v)
        if _math.isnan(fv) or _math.isinf(fv):
            return 0.0
        return fv
    except (ValueError, TypeError):
        return str(v)


def _clean_eval_result_for_json(result):
    """把 EvaluationSummary 转为完全 JSON 安全的 dict"""
    if result is None:
        return None
    try:
        # 优先使用 EvaluationSummary.to_dict()
        if hasattr(result, "to_dict") and callable(result.to_dict):
            raw = result.to_dict()
        elif isinstance(result, dict):
            raw = result
        else:
            raw = {"_raw_str": str(result)}
        # 递归清理
        return _safe_float_json(raw)
    except Exception as e:
        logger.error(f"[RAG评估] 结果 JSON 清理失败: {e}", exc_info=True)
        return {"_error": f"序列化失败: {type(e).__name__}: {e}"}


def _run_eval_background_sync(task_id: str, test_cases: list, settings_obj):
    """后台线程同步执行评估（不阻塞 HTTP 请求）"""
    try:
        from backend.evaluation.rag_runner import run_evaluation

        total_cases = len(test_cases)
        logger.info(f"[RAG评估][{task_id}] 后台任务开始，{total_cases} 个用例")

        # 更新状态为 running
        store = _eval_task_store
        store[task_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "created_at": store.get(task_id, {}).get("created_at", time.time()),
            "updated_at": time.time(),
            "progress": {"current": 0, "total": total_cases},
        }

        def _progress_cb(current, total, case_id):
            store[task_id]["progress"] = {"current": current, "total": total}
            store[task_id]["updated_at"] = time.time()
            if current % 5 == 0 or current == total:
                logger.info(f"[RAG评估][{task_id}] 进度 {current}/{total}")

        # 执行评估（用配置参数）
        summary = run_evaluation(
            test_cases,
            eval_concurrency=getattr(settings_obj, "eval_concurrency", 3),
            batch_size=getattr(settings_obj, "eval_batch_size", 5),
            progress_callback=_progress_cb,
        )

        logger.info(
            f"[RAG评估][{task_id}] 完成: overall={summary.avg_overall:.3f}, "
            f"success={summary.successful_cases}/{summary.total_cases}"
        )

        # 转为 JSON 安全的结果
        cleaned_result = _clean_eval_result_for_json(summary)

        store[task_id] = {
            "status": "completed",
            "result": cleaned_result,
            "error": None,
            "created_at": store[task_id].get("created_at", time.time()),
            "updated_at": time.time(),
            "progress": {"current": total_cases, "total": total_cases},
        }

        # 清理大对象引用
        del summary
        del cleaned_result
        gc.collect()

    except Exception as e:
        logger.exception(f"[RAG评估][{task_id}] 后台任务异常: {e}")
        store = _eval_task_store
        created = store.get(task_id, {}).get("created_at", time.time())
        store[task_id] = {
            "status": "failed",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "created_at": created,
            "updated_at": time.time(),
            "progress": {"current": 0, "total": len(test_cases) if test_cases else 0},
        }
        gc.collect()


# ================================================================
#  会话监控（真实数据）
# ================================================================

@router.get("/sessions")
def get_sessions(
    filter_status: Optional[str] = Query(None, description="过滤状态: active | dead"),
):
    """
    获取活跃会话列表（真实后台数据）

    返回每个会话的：
    - 会话 ID
    - 用户标识
    - 最后活跃时间
    - 空闲时长
    - 处理 Agent
    - 消息数量
    - 是否僵死会话
    """
    state_graph = get_state_graph()
    sessions = state_graph.get_active_sessions()

    if filter_status == "dead":
        sessions = [s for s in sessions if s.get("is_dead", False)]
    elif filter_status == "active":
        sessions = [s for s in sessions if not s.get("is_dead", False)]

    total = len(sessions)
    dead_count = sum(1 for s in sessions if s.get("is_dead", False))

    return {
        "sessions": sessions,
        "total": total,
        "active": total - dead_count,
        "dead": dead_count,
    }


@router.post("/sessions/{session_id}/kill")
def kill_session(session_id: str):
    """终止指定会话"""
    state_graph = get_state_graph()
    success = state_graph.kill_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "killed", "session_id": session_id}


@router.get("/sessions/{session_id}/detail")
def get_session_detail(session_id: str):
    """获取单个会话的详细信息"""
    state_graph = get_state_graph()
    collector = get_metrics_collector()

    # 查找会话
    sessions = state_graph.get_active_sessions()
    session_info = None
    for s in sessions:
        if s["id"] == session_id:
            session_info = s
            break

    if not session_info:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 获取会话的 Token 消耗
    session_tokens = collector.get_session_tokens(session_id)
    request_count = collector.get_session_request_count(session_id)

    return {
        "session": session_info,
        "token_usage": session_tokens,
        "request_count": request_count,
        "avg_tokens_per_request": round(session_tokens / max(request_count, 1), 1),
    }


# ================================================================
#  Agent 实例管理
# ================================================================

@router.get("/agents")
def get_agent_instances():
    """获取所有 Agent 实例状态"""
    coordinator = get_gossip_coordinator()
    report = coordinator.get_status_report()

    agents = []
    for name, info in report.get("agents", {}).items():
        agents.append({
            "name": name,
            "role": info.get("role", name),
            "status": info.get("status", "unknown"),
            "restart_count": info.get("restarts", 0),
            "restart_logs": info.get("restart_logs", []),
            "vector_clock": info.get("clock", {}),
        })

    return {
        "agents": agents,
        "node_id": report["node_id"],
        "global_clock": report["vector_clock"],
        "peers": report["gossip_peers"],
    }


@router.post("/agents/{agent_name}/restart")
def restart_agent_instance(agent_name: str):
    """重启指定 Agent 实例"""
    coordinator = get_gossip_coordinator()
    success = coordinator.restart_agent(agent_name)
    if not success:
        raise HTTPException(status_code=500, detail=f"重启失败: {agent_name}")
    return {
        "status": "restarted",
        "agent": agent_name,
        "info": coordinator.get_agent_restart_info(agent_name),
    }


# ================================================================
#  Token 消耗统计
# ================================================================

@router.get("/tokens")
def get_token_usage(
    period: Optional[str] = Query(None, description="统计周期: 1h | 6h | 24h | all"),
):
    """
    获取真实 Token 消耗统计

    返回：
    - 各 Agent 的 Token 消耗汇总
    - 按时间段分布
    - 最近请求的 Token 明细
    """
    collector = get_metrics_collector()

    # 获取最近的请求记录（带 Token 信息）
    recent_records = collector.get_recent_requests(50)

    # 汇总 Token 消耗
    agent_token_summary = {}
    total_tokens = 0

    for record in recent_records:
        if record.tokens > 0:
            agent = record.agent or record.endpoint or "unknown"
            if agent not in agent_token_summary:
                agent_token_summary[agent] = {
                    "total_tokens": 0,
                    "request_count": 0,
                    "avg_tokens": 0,
                }
            agent_token_summary[agent]["total_tokens"] += record.tokens
            agent_token_summary[agent]["request_count"] += 1
            total_tokens += record.tokens

    # 计算平均值
    for agent_info in agent_token_summary.values():
        if agent_info["request_count"] > 0:
            agent_info["avg_tokens"] = round(
                agent_info["total_tokens"] / agent_info["request_count"]
            )

    # 最终汇总格式
    token_usage = []
    for agent, info in agent_token_summary.items():
        token_usage.append({
            "agent": agent,
            "total_tokens": info["total_tokens"],
            "request_count": info["request_count"],
            "avg_tokens": info["avg_tokens"],
            "percentage": round(info["total_tokens"] / total_tokens * 100, 1) if total_tokens > 0 else 0,
        })

    # 按 total_tokens 降序排列
    token_usage.sort(key=lambda x: x["total_tokens"], reverse=True)

    return {
        "total_tokens": total_tokens,
        "by_agent": token_usage,
        "per_minute": collector.get_metrics().qps * 60,  # 估算每分钟
        "today_estimated": collector.get_metrics().qps * 3600 * 24,  # 估算一天
        "recent_requests": [
            {
                "timestamp": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
                "endpoint": r.endpoint,
                "tokens": r.tokens,
                "duration_ms": r.duration_ms,
            }
            for r in recent_records
            if r.tokens > 0
        ][:20],
    }


@router.get("/tokens/distribution")
def get_token_distribution():
    """获取 Token 消耗分布"""
    collector = get_metrics_collector()
    recent_records = collector.get_recent_requests(200)

    # 按时间窗口聚合（每分钟）
    windows = {}
    for record in recent_records:
        if record.tokens > 0:
            minute_key = time.strftime("%H:%M", time.localtime(record.timestamp))
            if minute_key not in windows:
                windows[minute_key] = 0
            windows[minute_key] += record.tokens

    return {
        "distribution": [
            {"time": k, "tokens": v}
            for k, v in sorted(windows.items())
        ],
    }


# ================================================================
#  系统性能分布
# ================================================================

@router.get("/performance")
def get_performance():
    """获取系统性能分布"""
    collector = get_metrics_collector()
    m = collector.get_metrics()
    degradation = get_degradation_manager()

    recent_records = collector.get_recent_requests(100)

    # 延迟分布
    latencies = [r.duration_ms for r in recent_records if r.duration_ms > 0]
    latency_distribution = {
        "p50": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p90": sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0,
        "p99": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "avg": sum(latencies) / len(latencies) if latencies else 0,
        "max": max(latencies) if latencies else 0,
        "min": min(latencies) if latencies else 0,
    }

    # 请求状态分布
    status_count = {"success": 0, "error": 0}
    for r in recent_records:
        status_count[r.status] = status_count.get(r.status, 0) + 1

    return {
        "qps": m.qps,
        "avg_latency_ms": m.avg_latency_ms,
        "error_rate": m.error_rate,
        "latency_distribution": latency_distribution,
        "request_status_distribution": status_count,
        "total_requests": m.total_requests,
        "degradation_level": degradation.get_status_report()["degradation_level"],
    }


# ================================================================
#  性能测试（真实数据）
# ================================================================

@router.get("/benchmark")
def get_benchmark():
    """
    获取性能测试数据（真实运行数据）

    基于系统实际运行指标计算：
    - RAG 召回评估（基于最近查询的命中率）
    - Agent 响应质量（基于反思评分）
    - 系统吞吐量
    - 各 Agent 延迟明细

    **时间窗口：仅统计近 1 小时内的数据**
    """
    collector = get_metrics_collector()
    m = collector.get_metrics()

    # --- 【修改】时间窗口过滤：仅保留近 1 小时的请求记录 ---
    one_hour_ago = time.time() - 3600  # 1 小时 = 3600 秒
    all_recent = collector.get_recent_requests(1000)  # 放宽获取上限
    recent_records = [
        r for r in all_recent if getattr(r, "timestamp", 0) >= one_hour_ago
    ]
    logger.info(
        f"[Benchmark] 近1小时请求记录: {len(recent_records)}/{len(all_recent)} "
        f"(总共有 {len(all_recent)} 条)"
    )

    coordinator = get_gossip_coordinator()
    report = coordinator.get_status_report()

    # Agent 延迟统计（优先使用实际请求数据）
    agent_latency = {}
    for r in recent_records:
        if r.agent:
            if r.agent not in agent_latency:
                agent_latency[r.agent] = []
            agent_latency[r.agent].append(r.duration_ms)

    # 如果没有实际请求数据，使用 gossip_coordinator 中注册的 Agent 生成占位
    # 确保前端始终有 Agent 列表可展示（避免"空白"）
    agent_stats = {}
    for agent_name, lat_list in agent_latency.items():
        if lat_list:
            sorted_lat = sorted(lat_list)
            agent_stats[agent_name] = {
                "avg_ms": round(sum(lat_list) / len(lat_list), 1),
                "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 1),
                "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 1),
                "max_ms": round(max(lat_list), 1),
                "requests": len(lat_list),
            }

    # 补充 gossip_coordinator 中已知但还没有请求记录的 Agent
    if len(agent_stats) == 0:
        # 完全没有请求数据时，从 StateGraph 的注册 Agent 中获取名称
        # 作为 fallback，使用 coordinator 的 agents
        known_agents = report.get("agents", {})
        if known_agents:
            for agent_name in known_agents.keys():
                agent_stats[agent_name] = {
                    "avg_ms": 0.0,
                    "p50_ms": 0.0,
                    "p99_ms": 0.0,
                    "max_ms": 0.0,
                    "requests": 0,
                }

    # --- 【修改】agent_status 使用英文内部名，与 agent_performance 保持一致 ---
    agent_status = {}
    known_agents_from_report = report.get("agents", {})

    # 收集所有已知的 Agent（来自请求记录的 + coordinator 注册的）
    known_labels = set(agent_stats.keys())
    known_names = set(known_agents_from_report.keys())

    # 构建完整的 Agent 列表（全部使用英文内部名称）
    all_labels = known_names | known_labels

    for agent_name in sorted(all_labels):
        status = "healthy"
        info = known_agents_from_report.get(agent_name)
        if isinstance(info, dict):
            s = info.get("status", "healthy")
        elif info:
            s = info
        else:
            s = "healthy"
        if s and s != "unknown" and s != "none":
            status = s
        agent_status[agent_name] = status

    # --- 【修改】补充 agent_stats 中的 Agent，使其与 agent_status 对齐 ---
    # 防止 agent_stats 为空但 agent_status 有值
    for agent_name in agent_status:
        if agent_name not in agent_stats:
            agent_stats[agent_name] = {
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
                "requests": 0,
            }

    # RAG 召回评估
    rag_eval = _evaluate_rag_performance(recent_records)

    return {
        "agent_performance": agent_stats,
        "rag_evaluation": rag_eval,
        "system_throughput": {
            "qps": m.qps,
            "avg_latency_ms": m.avg_latency_ms,
            "total_requests": m.total_requests,
            "error_rate": m.error_rate,
        },
        "agent_status": agent_status,
        "node_id": report["node_id"],
        "global_clock": report["vector_clock"],
    }


@router.get("/benchmark/run")
def run_benchmark():
    """
    运行在线性能测试，获取实际数据

    测试项目：
    1. 知识库检索延迟测试
    2. Agent 响应时间测试
    3. 并发吞吐量测试
    """
    import concurrent.futures
    from backend.graph.state_graph import get_state_graph
    from backend.pipeline.rag_pipeline import RAGPipeline

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": [],
    }

    test_queries = [
        "你好",
        "请问如何退款",
        "我的订单出现问题",
        "帮我查一下物流",
        "有什么优惠活动",
        "今天天气不错",
        "我需要帮助",
    ]

    # ======== 测试 1: 知识库检索延迟 ========
    logger.info("[Benchmark] 测试知识库检索延迟...")
    pipeline = RAGPipeline()
    rag_latencies = []

    for query in test_queries[:3]:
        t0 = time.time()
        try:
            result = pipeline.retrieve(query=query)
            elapsed = (time.time() - t0) * 1000
            rag_latencies.append(elapsed)
        except Exception as e:
            logger.warning(f"[Benchmark] RAG 检索失败: {e}")
            rag_latencies.append(0)

    if rag_latencies:
        valid = [l for l in rag_latencies if l > 0]
        results["tests"].append({
            "name": "知识库检索延迟",
            "avg_ms": round(sum(valid) / len(valid), 1) if valid else 0,
            "min_ms": round(min(valid), 1) if valid else 0,
            "max_ms": round(max(valid), 1) if valid else 0,
            "samples": len(valid),
        })

    # ======== 测试 2: Agent 响应时间 ========
    logger.info("[Benchmark] 测试 Agent 响应时间...")
    state_graph = get_state_graph()
    collector = get_metrics_collector()

    agent_latencies = {}
    total_tokens_before = collector.get_token_usage()
    for query in test_queries:
        t0 = time.time()
        try:
            response = state_graph.process(query, session_id=f"bench_{int(time.time())}")
            elapsed = (time.time() - t0) * 1000
            agent_latencies[query] = {"response": response[:50], "latency_ms": round(elapsed, 1)}
        except Exception as e:
            agent_latencies[query] = {"response": f"error: {e}", "latency_ms": 0}
    total_tokens_after = collector.get_token_usage()
    tokens_consumed = total_tokens_after - total_tokens_before

    results["tests"].append({
        "name": "Agent 响应时间",
        "details": agent_latencies,
        "avg_ms": round(
            sum(v["latency_ms"] for v in agent_latencies.values()) / len(agent_latencies), 1
        ),
        "total_tokens_consumed": tokens_consumed,
        "avg_tokens_per_query": round(tokens_consumed / max(len(test_queries), 1)),
    })

    # ======== 测试 3: 并发吞吐量 ========
    logger.info("[Benchmark] 测试并发吞吐量...")
    concurrent_results = []

    def concurrent_task(query):
        t0 = time.time()
        try:
            state_graph.process(query, session_id=f"conc_{int(time.time())}")
            return (time.time() - t0) * 1000
        except Exception:
            return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(concurrent_task, q) for q in test_queries[:3]]
        for f in concurrent.futures.as_completed(futures):
            concurrent_results.append(f.result())

    valid_conc = [l for l in concurrent_results if l > 0]
    results["tests"].append({
        "name": "并发吞吐量",
        "concurrent_requests": len(futures),
        "total_time_ms": round(max(valid_conc), 1) if valid_conc else 0,
        "avg_time_ms": round(sum(valid_conc) / len(valid_conc), 1) if valid_conc else 0,
        "throughput_qps": round(len(valid_conc) / (max(valid_conc) / 1000), 2) if valid_conc else 0,
    })

    return results


def _evaluate_rag_performance(recent_records) -> dict:
    """
    基于真实请求记录统计 RAG 性能（**不跑任何测试集、不调用 RAG Pipeline**）。

    指标逻辑：
      - 召回率 / 准确率 / F1：从最近 N 条请求记录的成功率 + 平均延迟推导
      - 延迟：直接使用 MetricsCollector 聚合的窗口数据
      - 测试集相关字段：为空字符串（前端做安全判断，不会崩溃）

    与旧版相比，这里完全避免了对 RAG Pipeline 的调用，
    因为：
      1) 每次打开性能页面都跑 5 条测试查询 → 会让服务器内存持续增长
      2) 真实请求记录已经反映了生产环境的检索质量，更有参考价值
    """
    try:
        collector = get_metrics_collector()
        m = collector.get_metrics()
    except Exception:
        collector = None
        m = None

    # ---- 1. 基础窗口聚合 ----
    if not recent_records:
        recent_records = []

    # 统计窗口：最近 200 条请求（若还没 200 条就用现有全部）
    window_records = recent_records[-200:] if recent_records else []
    if not window_records:
        # 尚无请求记录，返回占位但合法的结构
        return {
            "recall_rate": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "avg_score": 0.0,
            # 新增：五核心指标（实时追踪 proxy，字段名与 DeepEval 评估结果一致）
            "avg_faithfulness": 0.0,
            "avg_context_precision": 0.0,
            "avg_context_recall": 0.0,
            "avg_context_relevancy": 0.0,
            "avg_answer_relevancy": 0.0,
            "avg_overall": 0.0,
            "total_queries": 0,
            "rag_hits": 0,
            "total_docs_retrieved": 0,
            "high_quality_docs": 0,
            "avg_docs_per_query": 0.0,
            "latency": {
                "avg_ms": 0.0,
                "p50_ms": 0,
                "p90_ms": 0,
                "p99_ms": 0,
            },
            "avg_response_time_ms": 0.0,
            "evaluation_method": "实时请求追踪（无测试集调用）",
            "test_queries": [],
            "data_source": "尚无请求记录",
        }

    # ---- 2. 延迟分布（毫秒）----
    latencies = [
        max(0.0, r.duration_ms)
        for r in window_records
        if r.duration_ms is not None and r.duration_ms > 0
    ]
    if latencies:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        def _pct(p):
            idx = min(n - 1, int(n * p))
            return round(sorted_lat[idx], 1)

        latency_info = {
            "avg_ms": round(sum(sorted_lat) / n, 1),
            "p50_ms": _pct(0.50),
            "p90_ms": _pct(0.90),
            "p99_ms": _pct(0.99),
        }
    else:
        latency_info = {"avg_ms": 0.0, "p50_ms": 0, "p90_ms": 0, "p99_ms": 0}

    # ---- 3. 成功率 / 稳定性 ----
    success_count = sum(1 for r in window_records if r.status == "success")
    total_count = len(window_records)
    recall_rate = round(success_count / max(total_count, 1), 4)

    # 低延迟请求占比（< 3000ms 视为正常），作为准确率 proxy
    fast_count = sum(1 for l in latencies if l < 3000)
    precision = round(fast_count / max(len(latencies), 1), 4) if latencies else 0.0

    # F1
    if recall_rate + precision > 0.001:
        f1_score = round(2 * recall_rate * precision / (recall_rate + precision), 4)
    else:
        f1_score = 0.0

    # 平均综合分（结合成功率、延迟、稳定性）
    if m:
        err = float(getattr(m, "error_rate", 0.0) or 0.0)
        avg_lat = float(getattr(m, "avg_latency_ms", 0.0) or 0.0)
        # 归一化：错误率越低、延迟越低，分数越高
        avg_score = round(max(0.0, min(1.0, (1.0 - err) * max(0.0, 1.0 - avg_lat / 10000.0))), 4)
    else:
        avg_score = round(recall_rate, 4)

    # ---- 3.5 新增：五大核心指标（DeepEval 五指标，基于实时请求追踪的 proxy 估算）
    # 注：这些是"实时请求追踪得到的近似值"，不是真正的 DeepEval 评估结果
    # 但用于让 RAG 质量雷达统一展示五大指标名称
    faithfulness = avg_score                                # 忠诚度：回答基于上下文的可信程度（proxy:综合质量）
    context_precision = precision                           # 检索精度：检索上下文是否相关（proxy:低延迟请求占比）
    context_recall = recall_rate                            # 上下文召回：参考答案是否被覆盖（proxy:请求成功率）
    context_relevancy = f1_score                            # 上下文相关：上下文对问题的支撑（proxy:F1 综合分数）
    answer_relevancy = avg_score                            # 答案相关：回答是否紧扣问题（proxy:综合质量）
    # 计算五指标综合分（与 DeepEval 评估页一致，便于对比）
    five_metric_overall = round(
        (faithfulness + context_precision + context_recall + context_relevancy + answer_relevancy) / 5, 4
    )

    # ---- 4. 聚合统计量（仅用于展示规模）----
    avg_docs_per_query = round(
        sum(1 for r in window_records if r.tokens > 0) / max(total_count, 1),
        2,
    )
    total_tokens = sum(r.tokens for r in window_records if r.tokens)

    return {
        "recall_rate": recall_rate,
        "precision": precision,
        "f1_score": f1_score,
        "avg_score": avg_score,
        # 新增：五核心指标（字段名与 DeepEval 评估结果一致，便于前端统一渲染）
        "avg_faithfulness": faithfulness,
        "avg_context_precision": context_precision,
        "avg_context_recall": context_recall,
        "avg_context_relevancy": context_relevancy,
        "avg_answer_relevancy": answer_relevancy,
        "avg_overall": five_metric_overall,
        "total_queries": total_count,
        "rag_hits": success_count,
        "total_docs_retrieved": total_tokens,  # token 数作为参考
        "high_quality_docs": success_count,
        "avg_docs_per_query": avg_docs_per_query,
        "latency": latency_info,
        "avg_response_time_ms": float(getattr(m, "avg_latency_ms", 0.0) or 0.0) if m else latency_info["avg_ms"],
        "evaluation_method": "实时请求追踪（无测试集调用）",
        "test_queries": [],  # 不再跑测试集
        "data_source": f"最近 {total_count} 条真实请求",
    }


# ================================================================
#  综合仪表盘
# ================================================================

@router.get("/dashboard")
def get_dashboard():
    """综合仪表盘数据"""
    collector = get_metrics_collector()
    m = collector.get_metrics()
    coordinator = get_gossip_coordinator()
    state_graph = get_state_graph()
    degradation = get_degradation_manager()

    sessions = state_graph.get_active_sessions()
    agent_report = coordinator.get_status_report()

    # Token 总和
    recent_records = collector.get_recent_requests(100)
    total_tokens = sum(r.tokens for r in recent_records)

    return {
        "overview": {
            "qps": m.qps,
            "avg_latency_ms": m.avg_latency_ms,
            "error_rate": m.error_rate,
            "total_requests": m.total_requests,
            "total_tokens": total_tokens,
            "active_sessions": len([s for s in sessions if not s.get("is_dead")]),
            "dead_sessions": len([s for s in sessions if s.get("is_dead")]),
        },
        "degradation": degradation.get_status_report(),
        "agents": agent_report.get("agents", {}),
        "vector_clock": agent_report["vector_clock"],
        "node_id": agent_report["node_id"],
        "peers": agent_report["gossip_peers"],
    }


# ================================================================
#  知识库管理（通过 pipeline 操作真实知识库）
# ================================================================


@router.get("/knowledge/documents")
def get_knowledge_docs(
    search: Optional[str] = Query(None),
    file_type: Optional[str] = Query(None),
):
    """获取知识库文档列表（从 Qdrant payload 读取 type/size/vectors/upload_time/status）"""
    from backend.pipeline.rag_pipeline import RAGPipeline
    try:
        pipeline = RAGPipeline()
        docs = pipeline.list_documents()

        # 过滤
        if search:
            docs = [d for d in docs if search.lower() in d.get("file_name", "").lower()]
        if file_type:
            docs = [d for d in docs if (d.get("file_type", "") or "").lower() == file_type.lower()]

        # 格式化前端需要的字段
        formatted = []
        for d in docs:
            # 文件大小：转换为人类可读的格式
            raw_size = d.get("file_size", 0) or 0
            try:
                if raw_size >= 1024 * 1024:
                    size_str = f"{int(raw_size / (1024 * 1024))}MB"
                elif raw_size >= 1024:
                    size_str = f"{int(raw_size / 1024)}KB"
                else:
                    size_str = f"{int(raw_size)}B" if raw_size else ""
            except Exception:
                size_str = str(raw_size)

            # 上传时间：优先 created_at，其次 updated_at
            upload_time = d.get("created_at", "") or d.get("updated_at", "")

            # 状态：统一为中文
            raw_status = d.get("status", "active") or ""
            if raw_status == "active":
                status_display = "已索引"
            elif raw_status == "indexing":
                status_display = "索引中"
            elif raw_status == "archived":
                status_display = "已归档"
            elif raw_status == "error":
                status_display = "错误"
            else:
                status_display = raw_status

            formatted.append({
                "id": d.get("doc_id", ""),
                "name": d.get("file_name", ""),
                "type": d.get("file_type", "") or "",
                "size": size_str,
                "vectors": d.get("vector_count", 0) or 0,
                "tags": d.get("tags", []),
                "uploadTime": upload_time,
                "status": status_display,
            })
        return {"documents": formatted, "total": len(formatted)}
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return {"documents": [], "total": 0}


@router.post("/knowledge/documents/upload")
def upload_knowledge_doc(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    rebuild: str = Form(default="false"),
    progress_id: str = Form(default=""),
):
    """上传文档到知识库（带进度追踪）

    progress_id: 可选，由前端生成的临时进度 ID，
    用于前端 SSE 订阅与后端进度记录共享同一个 ID。
    """
    from backend.pipeline.rag_pipeline import RAGPipeline

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = os.path.splitext(file.filename)[1].lower().lstrip(".")
    supported = {"pdf", "docx", "xlsx", "txt", "md", "csv", "png", "jpg", "jpeg"}
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{suffix}，支持: {', '.join(sorted(supported))}",
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    rebuild_bool = rebuild.lower() == "true"

    # 1. 读取文件内容到临时文件
    # UploadFile.read() 是异步方法，在 def 同步路由中从底层 file 对象读取
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        file.file.seek(0)
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name
        file_size = len(content)

    # 2. 创建进度对象 - 优先使用前端传入的 progress_id，否则生成一个
    progress_store = get_progress_store()
    temp_doc_id = progress_id.strip() if progress_id.strip() else f"upload_{int(time.time() * 1000)}"
    p = progress_store.create(temp_doc_id, file.filename or f"upload.{suffix}")
    p.set_stage("parsing", "解析文档", 0, "准备文件...")
    final_doc_id = temp_doc_id

    def _progress_cb(progress_obj: dict):
        """pipeline 阶段进度回调 -> 写入全局 store。

        支持两种模式:
        1. 阶段切换: progress_obj 含 "stage" 字段且与当前不同 -> 调用 set_stage
        2. 细粒度更新: 仅更新阶段内进度 -> 调用 update_stage_detail 并同步 progress/message
        """
        new_stage = progress_obj.get("stage", p.stage)
        new_stage_name = progress_obj.get("stage_name", p.stage_name)
        new_progress = progress_obj.get("progress", p.progress)
        new_message = progress_obj.get("message", p.message)

        if new_stage != p.stage or new_stage_name != p.stage_name:
            # 阶段切换：更新 stage_name + 起始进度
            p.set_stage(new_stage, new_stage_name, new_progress, new_message)
            p.progress = new_progress  # 确保总进度被更新（set_stage 默认将 details[stage_name]=0）
            p.message = new_message
        else:
            # 阶段内细粒度更新
            p.update_stage_detail(new_stage_name, new_progress, new_message)
            p.progress = new_progress
            p.message = new_message

    try:
        # 3. 执行索引
        pipeline = RAGPipeline()
        result = pipeline.index_document(
            file_path=tmp_path,
            operator="admin",
            tags=tag_list,
            rebuild=rebuild_bool,
            display_name=file.filename,
            progress_callback=_progress_cb,
        )

        final_doc_id = result["doc_id"]
        # 将进度记录从 temp_doc_id 迁移到真实 doc_id
        progress_store._progresses[final_doc_id] = p
        p.doc_id = final_doc_id
        progress_store._progresses.pop(temp_doc_id, None)

        p.mark_done(f"上传完成，共 {result.get('child_chunks', 0)} 个子块")
        return {
            "status": "success",
            "doc_id": result["doc_id"],
            "file_name": file.filename,
            "chapters": result.get("chapters", 0),
            "parent_chunks": result.get("parent_chunks", 0),
            "child_chunks": result.get("child_chunks", 0),
        }
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        p.mark_error(str(e))
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/knowledge/documents/progress/{doc_id}")
def get_upload_progress(doc_id: str):
    """查询某个文档的当前上传进度（一次性）"""
    p = get_progress_store().get(doc_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"未找到进度记录: {doc_id}")
    return p.to_dict()


@router.get("/knowledge/documents/progress/stream/{doc_id}")
async def stream_upload_progress(doc_id: str, request: Request):
    """
    SSE 流式推送某个文档的上传进度

    前端使用方式:
        const es = new EventSource('/api/ops/knowledge/documents/progress/stream/<doc_id>');
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    progress_store = get_progress_store()

    async def _event_gen():
        last_progress = -1
        last_stage = ""
        heartbeat = 0

        while True:
            # 检查客户端是否断开
            if await request.is_disconnected():
                break

            p = progress_store.get(doc_id)
            if not p:
                # 进度记录尚未创建（或已过期），告知前端等待
                yield f"data: {json.dumps({'stage': 'waiting', 'stage_name': '等待上传开始...', 'progress': 0})}\n\n"
            else:
                cur = p.progress
                stage = p.stage

                # 仅当进度或阶段变化时发送（减少流量），或每 30 次心跳发送一次
                if cur != last_progress or stage != last_stage or heartbeat >= 30:
                    yield f"data: {json.dumps(p.to_dict())}\n\n"
                    last_progress = cur
                    last_stage = stage
                    heartbeat = 0

                # 结束条件
                if p.stage in ("done", "error"):
                    # 再推送一次结束状态，然后断开
                    yield f"data: {json.dumps(p.to_dict())}\n\n"
                    break

            heartbeat += 1
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/knowledge/documents/progress")
def list_all_upload_progresses():
    """列出所有正在进行的上传任务的进度"""
    progress_store = get_progress_store()
    all_items = progress_store.get_all()

    active = []
    for doc_id, p in all_items.items():
        if p.stage in ("done", "error"):
            continue
        active.append({
            "doc_id": doc_id,
            "file_name": p.file_name,
            "stage": p.stage,
            "stage_name": p.stage_name,
            "progress": p.progress,
            "message": p.message,
            "elapsed_seconds": round(time.time() - p.start_time, 1),
        })
    return {"active_uploads": active, "total": len(active)}


@router.delete("/knowledge/documents/{doc_id}")
def delete_knowledge_doc(doc_id: str):
    """删除知识库文档（轻量级，不初始化 Embedder/SpareEmbedder）"""
    from backend.retrieval.vector_store import VectorStore
    from backend.storage.parent_store import ParentStore
    try:
        vs = VectorStore()
        ps = ParentStore()
        vector_deleted = vs.delete_by_doc_id(doc_id)
        parent_deleted = ps.delete_by_doc_id(doc_id)
        child_deleted = ps.delete_children_by_doc_id(doc_id)
        total = vector_deleted + parent_deleted + child_deleted
        if total == 0:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"status": "deleted", "doc_id": doc_id, "points_removed": total}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/documents/batch-delete")
def batch_delete_knowledge_docs(request: dict):
    """批量删除知识库文档（轻量级）"""
    from backend.retrieval.vector_store import VectorStore
    from backend.storage.parent_store import ParentStore
    ids = request.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="请提供要删除的文档 ID 列表")

    vs = VectorStore()
    ps = ParentStore()
    deleted = 0
    errors = []
    for doc_id in ids:
        try:
            v = vs.delete_by_doc_id(doc_id)
            p = ps.delete_by_doc_id(doc_id)
            c = ps.delete_children_by_doc_id(doc_id)
            deleted += v + p + c
        except Exception as e:
            errors.append({"id": doc_id, "error": str(e)})

    return {"status": "completed", "deleted_count": deleted, "errors": errors}


@router.post("/knowledge/documents/{doc_id}/reindex")
def reindex_knowledge_doc(doc_id: str):
    """重建文档索引"""
    from backend.pipeline.rag_pipeline import RAGPipeline
    try:
        pipeline = RAGPipeline()
        # 新架构不再需要单独重建 BM25 索引（稀疏向量存储在 Qdrant 中）
        res = {"status": "success", "doc_id": doc_id, "message": "索引无需重建（稀疏向量已内置于 Qdrant Collection）"}
        return res
    except Exception as e:
        logger.warning(f"重建索引失败: {e}")
        return {"status": "warning", "doc_id": doc_id, "message": f"重建失败: {str(e)}"}


# ================================================================
#  资源监控 & 日志
# ================================================================


@router.get("/resources")
def get_resources():
    """获取服务器资源信息"""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
    except Exception:
        cpu = 0

    try:
        mem = psutil.virtual_memory()
        mem_used_gb = round(mem.used / (1024**3), 1)
        mem_total_gb = round(mem.total / (1024**3), 1)
        mem_percent = mem.percent
    except Exception:
        mem_used_gb = 0
        mem_total_gb = 0
        mem_percent = 0

    try:
        disk = psutil.disk_usage("/")
        disk_used_gb = round(disk.used / (1024**3), 1)
        disk_total_gb = round(disk.total / (1024**3), 1)
        disk_percent = disk.percent
    except Exception:
        disk_used_gb = 0
        disk_total_gb = 0
        disk_percent = 0

    # 获取进程运行时间
    try:
        proc = psutil.Process()
        uptime_seconds = time.time() - proc.create_time()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m"
    except Exception:
        uptime_str = "--"

    # 慢请求 TOP10
    collector = get_metrics_collector()
    all_reqs = collector.get_recent_requests(200)
    slow_requests = sorted(
        [r for r in all_reqs if r.duration_ms > 1000],
        key=lambda r: r.duration_ms,
        reverse=True,
    )[:10]
    slow_list = []
    for sr in slow_requests:
        slow_list.append({
            "endpoint": sr.endpoint,
            "duration": f"{sr.duration_ms:.0f}ms",
            "agent": sr.agent or "-",
            "time": time.strftime("%H:%M:%S", time.localtime(sr.timestamp)),
        })

    return {
        "cpu_percent": cpu,
        "memory_used_gb": mem_used_gb,
        "memory_total_gb": mem_total_gb,
        "memory_percent": mem_percent,
        "disk_used_gb": disk_used_gb,
        "disk_total_gb": disk_total_gb,
        "disk_percent": disk_percent,
        "uptime": uptime_str,
        "slow_requests": slow_list,
    }


@router.get("/resources/logs")
def get_logs(level: Optional[str] = Query(None)):
    """获取最近的系统日志（从请求记录中提取错误和警告）"""
    collector = get_metrics_collector()
    all_reqs = collector.get_recent_requests(200)

    logs = []
    for r in all_reqs:
        log_level = "INFO"
        if r.status == "error":
            log_level = "ERROR"
        elif r.duration_ms > 5000:
            log_level = "WARNING"

        # 根据筛选条件过滤
        if level and log_level != level:
            continue

        logs.append({
            "id": str(int(r.timestamp * 1000)),
            "time": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
            "level": log_level,
            "message": f"{r.endpoint} - {r.status} ({r.duration_ms:.0f}ms)"
                       + (f" [Agent: {r.agent}]" if r.agent else "")
                       + (f" [Tokens: {r.tokens}]" if r.tokens > 0 else ""),
        })

    return {"logs": logs}


@router.delete("/resources/logs")
def clear_logs():
    """清空日志（请求记录）"""
    collector = get_metrics_collector()
    with collector._requests_lock:
        collector._requests.clear()
    return {"status": "cleared", "message": "日志已清空"}


# =====================================================================
#  RAG 性能评估（异步后台执行 + 轮询模式，避免长时间 HTTP 请求超时）
# =====================================================================

@router.get("/evaluation/test-cases")
def list_evaluation_test_cases():
    """获取评估测试用例列表"""
    try:
        from backend.evaluation.rag_test_cases import get_default_test_cases, list_categories
        cases = get_default_test_cases()
        return {
            "categories": list_categories(),
            "total": len(cases),
            "cases": [
                {
                    "id": c["id"],
                    "category": c.get("category", ""),
                    "question": c["question"],
                    "expected_answer": c.get("expected_answer", ""),
                    "expected_keywords": c.get("expected_keywords", []),
                    "forbidden_keywords": c.get("forbidden_keywords", []),
                    "difficulty": c.get("difficulty", "medium"),
                }
                for c in cases
            ],
        }
    except Exception as e:
        logger.error(f"加载评估测试集失败: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/evaluation/run")
def run_rag_evaluation(payload: dict = {}):
    """
    提交 RAG 评估任务（后台异步执行，前端通过 /status/{task_id} 轮询）

    请求体 (可选):
    {
        "case_ids": ["refund_policy_001", "logistics_query_001"],
        "categories": ["退款政策"],
    }
    """
    try:
        from backend.evaluation.rag_test_cases import get_default_test_cases
        from backend.config import settings as _settings

        all_cases = get_default_test_cases()
        case_ids = set(payload.get("case_ids") or [])
        categories = set(payload.get("categories") or [])

        selected = []
        for c in all_cases:
            if case_ids and c.get("id") not in case_ids:
                continue
            if categories and c.get("category") not in categories:
                continue
            selected.append(c)

        if not selected:
            raise HTTPException(status_code=400, detail="未选中任何测试用例")

        # 生成任务 ID 并记录初始状态
        task_id = str(uuid.uuid4())
        _eval_task_store[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": {"current": 0, "total": len(selected)},
        }

        logger.info(f"[RAG评估] 提交后台任务 task_id={task_id}, {len(selected)} 个用例")

        # 启动后台线程执行评估（不阻塞当前 HTTP 请求）
        t = threading.Thread(
            target=_run_eval_background_sync,
            args=(task_id, selected, _settings),
            daemon=True,
            name=f"rag-eval-{task_id[:8]}",
        )
        t.start()

        return {
            "status": "pending",
            "task_id": task_id,
            "message": f"评估任务已提交，共 {len(selected)} 个用例，请通过 /evaluation/status/{task_id} 轮询结果",
            "total_cases": len(selected),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[RAG评估] 提交任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/evaluation/status/{task_id}")
def get_evaluation_status(task_id: str):
    """
    查询评估任务状态与结果

    可能的状态:
      - pending:   任务已提交，等待开始
      - running:   评估正在执行（含 progress 进度信息）
      - completed: 评估完成（含完整的 summary 和 cases 数据）
      - failed:    评估失败（含 error 信息）
      - not_found: 任务 ID 不存在或已过期
    """
    task = _eval_task_store.get(task_id)

    if task is None:
        return {"status": "not_found", "task_id": task_id, "error": "任务不存在或已过期"}

    status = task.get("status", "unknown")
    result = task.get("result")
    error = task.get("error")
    progress = task.get("progress", {"current": 0, "total": 0})
    created_at = task.get("created_at", 0)
    updated_at = task.get("updated_at", 0)

    # 安全校验：确保 result（即便 None 以外）始终是 JSON 可序列化的 dict
    if status == "completed" and result is not None:
        # 用 EvaluationSummary.to_dict() 格式返回。注意：result 已经在后台线程中
        # 经 _clean_eval_result_for_json 处理过，已是 JSON 安全的 dict
        return {
            "status": "completed",
            "task_id": task_id,
            "result": result,  # 对应 EvaluationSummary.to_dict()
            "created_at": created_at,
            "updated_at": updated_at,
            "progress": progress,
        }
    elif status == "failed":
        return {
            "status": "failed",
            "task_id": task_id,
            "error": error or "未知错误",
            "created_at": created_at,
            "updated_at": updated_at,
            "progress": progress,
        }
    else:
        return {
            "status": status,
            "task_id": task_id,
            "progress": progress,
            "created_at": created_at,
            "updated_at": updated_at,
        }


@router.post("/evaluation/custom")
def run_custom_evaluation(payload: dict):
    """
    对自定义问答进行单次评估（无需运行整个数据集，直接提供上下文 + 回答）

    请求体:
    {
        "question": "问题",
        "contexts": ["上下文 1", "上下文 2"],
        "answer": "模型回答",
        "reference": "（可选）期望回答",
        "expected_keywords": ["关键词"],
        "forbidden_keywords": ["禁用词"]
    }
    """
    try:
        from backend.evaluation.rag_evaluator import (
            RetrievalResult,
            evaluate_retrieval_result,
        )

        question = (payload.get("question") or "").strip()
        contexts = list(payload.get("contexts") or [])
        answer = (payload.get("answer") or "").strip()
        if not question or not answer:
            raise HTTPException(status_code=400, detail="请提供 question 和 answer")

        rr = RetrievalResult(
            case_id="custom",
            case_category=payload.get("category") or "自定义",
            question=question,
            retrieved_contexts=contexts,
            generated_answer=answer,
            reference_answer=payload.get("reference"),
            expected_keywords=list(payload.get("expected_keywords") or []),
            forbidden_keywords=list(payload.get("forbidden_keywords") or []),
        )
        case_result = evaluate_retrieval_result(rr, use_deepeval=True)
        # 通过 _clean_eval_result_for_json 确保 JSON 安全
        safe_result = _clean_eval_result_for_json(case_result)
        return {
            "status": "ok",
            "result": safe_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自定义评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")