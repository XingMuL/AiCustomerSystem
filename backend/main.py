"""
智能客服系统入口 - FastAPI 服务

提供：
  - 多智能体对话 API（支持流式 SSE）
  - RAG 文档索引与检索 API
  - 降级状态管理 + Agent 重启 API
  - 系统健康监控
"""

import os
import sys
import tempfile
import json
import asyncio
import atexit
from contextlib import asynccontextmanager
from typing import Optional

# ===== Windows asyncio 事件循环修复 (必须在任何 asyncio 操作之前) =====
# 1. ProactorEventLoop 关闭后不允许 call_soon 操作 → 改用 SelectorEventLoop
# 2. httpx/openai/qdrant_client 内部的 async 客户端在进程退出的 __del__ 阶段
#    会尝试关闭资源，但此时 event loop 已关闭 → monkey-patch 核心方法捕获 RuntimeError
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # ---- monkey-patch: 修复 event loop 关闭后的资源清理异常 ----
    try:
        import _selector_transport_patch  # noqa: F401
    except Exception:
        pass

    # Patch asyncio 的 call_soon: event loop 关闭后静默返回，不抛 RuntimeError
    try:
        _orig_call_soon = asyncio.BaseEventLoop.call_soon

        def _safe_call_soon(self, callback, *args, **kwargs):
            try:
                if self.is_closed():
                    return asyncio.Future(loop=self) if False else None
                return _orig_call_soon(self, callback, *args, **kwargs)
            except RuntimeError:
                return None

        asyncio.BaseEventLoop.call_soon = _safe_call_soon
    except Exception:
        pass

    # Patch httpx AsyncClient.aclose: 捕获 event loop 关闭后的 RuntimeError
    try:
        import httpx as _httpx
        _orig_async_aclose = _httpx.AsyncClient.aclose

        async def _safe_async_aclose(self, *args, **kwargs):
            try:
                return await _orig_async_aclose(self, *args, **kwargs)
            except (RuntimeError, Exception):
                return None

        _httpx.AsyncClient.aclose = _safe_async_aclose

        _orig_sync_close = _httpx.Client.close

        def _safe_sync_close(self, *args, **kwargs):
            try:
                return _orig_sync_close(self, *args, **kwargs)
            except (RuntimeError, Exception):
                return None

        _httpx.Client.close = _safe_sync_close
    except Exception:
        pass

    # Patch httpcore async connection pool cleanup
    try:
        import httpcore._async.connection_pool as _httpcore_pool
        _orig_pool_aclose = _httpcore_pool.AsyncConnectionPool.aclose

        async def _safe_pool_aclose(self, *args, **kwargs):
            try:
                return await _orig_pool_aclose(self, *args, **kwargs)
            except (RuntimeError, Exception):
                return None

        _httpcore_pool.AsyncConnectionPool.aclose = _safe_pool_aclose
    except Exception:
        pass

    # Patch httpcore async HTTP11 protocol cleanup
    try:
        import httpcore._async.http11 as _httpcore_http11
        _orig_http11_aclose = _httpcore_http11.AsyncHTTP11Connection.aclose

        async def _safe_http11_aclose(self, *args, **kwargs):
            try:
                return await _orig_http11_aclose(self, *args, **kwargs)
            except (RuntimeError, Exception):
                return None

        _httpcore_http11.AsyncHTTP11Connection.aclose = _safe_http11_aclose
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from backend.config import settings
from backend.graph.state_graph import get_state_graph
from backend.degradation.degradation_manager import get_degradation_manager
from backend.pipeline.rag_pipeline import RAGPipeline, IndexConflictError
from backend.ops.router import router as ops_router
from backend.ops.metrics_collector import get_metrics_collector
from backend.memory.gossip_coordinator import get_gossip_coordinator

# ===== 应用生命周期：使用 lifespan 确保资源在 event loop 关闭前被正确释放 =====
_resources_to_close = []


def _register_resource(obj):
    """注册需要在退出时同步关闭的资源对象（必须有 close() 方法）"""
    _resources_to_close.append(obj)
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：
    - startup: 初始化全局资源
    - shutdown: 在 event loop 关闭前主动释放所有资源，避免 "Event loop is closed"
    """
    logger.info("[lifespan] 启动中: 初始化全局资源...")

    # 初始化
    global pipeline, state_graph, degradation_manager, metrics_collector, gossip_coordinator
    pipeline = _register_resource(RAGPipeline())
    state_graph = get_state_graph()
    degradation_manager = get_degradation_manager()
    metrics_collector = get_metrics_collector()
    gossip_coordinator = get_gossip_coordinator()

    yield

    # shutdown 阶段：在 event loop 还活着时主动释放所有资源
    logger.info("[lifespan] 关闭中: 释放全局资源...")
    for resource in reversed(_resources_to_close):
        try:
            if hasattr(resource, "close"):
                resource.close()
        except Exception as e:
            logger.warning(f"[lifespan] 关闭资源失败: {e}")
    logger.info("[lifespan] 所有资源已释放")


def _atexit_cleanup():
    """进程退出最后的兜底：同步关闭任何还活着的资源"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for resource in reversed(_resources_to_close):
            try:
                if hasattr(resource, "close"):
                    resource.close()
            except Exception:
                pass


atexit.register(_atexit_cleanup)

# 初始化
app = FastAPI(
    title="智能客服系统",
    description=(
        "多智能体协作 + RAG 知识库 + 5级降级策略 + Gossip 去中心化同步\n\n"
        "## 多智能体\n"
        "- 路由智能体 (Router): 意图分类\n"
        "- 知识库问答智能体 (KB-QA): 基于 RAG 的问答\n"
        "- 工单处理智能体 (Ticket): 工单创建与管理\n"
        "- 闲聊智能体 (Chitchat): 日常对话\n"
        "- 汇总智能体 (Summary): 多结果融合\n\n"
        "## RAG 特性\n"
        "- 支持 PDF/DOCX/XLSX/图片等多格式\n"
        "- 父子块切分 + 混合检索\n"
        "- 元数据版本管理\n\n"
        "## 降级策略\n"
        "- L0 正常 → L1 轻度 → L2 局部 → L3 中度 → L4 全局\n"
        "- 熔断器 + 令牌桶限流 + 自动恢复\n\n"
        "## Gossip 同步\n"
        "- 向量时钟因果一致性\n"
        "- 去中心化状态传播\n"
        "- 最终一致性保证"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS 支持（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 运维指标采集中间件
@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time as _time
    start = _time.time()
    response = await call_next(request)
    duration_ms = (_time.time() - start) * 1000
    status = "success" if response.status_code < 400 else "error"
    agent = request.url.path.split("/")[-1] if "/chat" in request.url.path else ""
    # 修复：/chat/stream 应记录为 chat 而非 stream
    if agent == "stream":
        agent = "chat"
    metrics_collector.record_request(
        endpoint=request.url.path,
        duration_ms=duration_ms,
        status=status,
        agent=agent,
    )
    return response

# 注册运维 API 路由
app.include_router(ops_router)


# ================================================================
#  请求/响应模型
# ================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    degradation_level: int = 0
    intent: str = ""
    confidence: float = 0.0


class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    tags: Optional[list[str]] = None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    child_hits: list[dict]
    parent_chunks: list[dict]
    context: str
    sources: list[dict]


class IndexResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    file_name: Optional[str] = None
    version: Optional[int] = None
    message: Optional[str] = None
    chapters: Optional[int] = None
    parent_chunks: Optional[int] = None
    child_chunks: Optional[int] = None
    vector_stats: Optional[dict] = None


# ================================================================
#  对话 API（多智能体入口 - 支持流式）
# ================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多智能体对话入口（非流式）

    完整工作流：
    1. 输入清洗（刷屏检测、截断）
    2. 降级状态注入（5级判定）
    3. 路由分发（意图分类 → KB-QA / Ticket / Chitchat）
    4. 子 Agent 并行执行（降级感知）
    5. 反思评判（质量检查 + 自动重试）
    6. 结果汇总
    7. 记忆更新

    - **message**: 用户输入
    - **session_id**: 会话 ID（用于多轮记忆）
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        response = state_graph.process(
            raw_input=request.message.strip(),
            session_id=request.session_id,
            user_id=request.user_id,
        )

        status = degradation_manager.get_status_report()

        return ChatResponse(
            response=response,
            session_id=request.session_id,
            degradation_level=status["degradation_level"],
        )

    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统内部错误: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口（SSE - Server-Sent Events）

    实时传输处理进度和最终回复，前端可逐字渲染打字机效果。

    事件格式：
    - data: {"type": "progress", "content": "..."}  处理进度
    - data: {"type": "chunk", "content": "..."}      逐字输出
    - data: {"type": "done", "content": "..."}       完成
    - data: {"type": "error", "content": "..."}      错误
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    async def event_stream():
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'content': '开始处理...'}, ensure_ascii=False)}\n\n"

            last_content = ""
            for chunk in state_graph.process_stream(
                request.message.strip(),
                request.session_id,
                request.user_id,
            ):
                # 进度信息（以 emoji 开头）
                if any(chunk.startswith(prefix) for prefix in ["🔍", "🧠", "📌", "📝"]):
                    yield f"data: {json.dumps({'type': 'progress', 'content': chunk}, ensure_ascii=False)}\n\n"
                    continue

                # 逐字打字机输出（带延迟模拟真实打字效果）
                if chunk.startswith(last_content) and len(chunk) > len(last_content):
                    new_chars = chunk[len(last_content):]
                    for char in new_chars:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': char}, ensure_ascii=False)}\n\n"
                        # 打字机延迟：中文约30ms/字，英文/标点约15ms
                        delay = 0.03 if ord(char) > 127 else 0.015
                        await asyncio.sleep(delay)
                elif chunk != last_content:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

                last_content = chunk

            # 发送完成事件
            status = degradation_manager.get_status_report()
            yield f"data: {json.dumps({'type': 'done', 'content': last_content, 'degradation_level': status['degradation_level']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ================================================================
#  Agent 管理 API（重启 + Gossip 状态）
# ================================================================

@app.post("/agents/{agent_name}/restart")
def restart_agent(agent_name: str):
    """重启指定 Agent 实例"""
    if agent_name not in ["router_agent", "kb_qa_agent", "ticket_agent", "chitchat_agent", "summary_agent", "memory_merge_agent"]:
        raise HTTPException(status_code=400, detail=f"未知 Agent: {agent_name}")

    success = gossip_coordinator.restart_agent(agent_name)
    if not success:
        raise HTTPException(status_code=500, detail=f"重启失败: {agent_name}")

    # 将重启后的 Agent 实例同步回 StateGraph
    new_instance = gossip_coordinator.get_restarted_agent_instance(agent_name)
    if new_instance and agent_name in state_graph._agent_map:
        state_graph._agent_map[agent_name] = new_instance
        # 更新 StateGraph 上的直接引用
        if agent_name == "router_agent":
            state_graph.router = new_instance
        elif agent_name == "kb_qa_agent":
            state_graph.kb_qa_agent = new_instance
        elif agent_name == "ticket_agent":
            state_graph.ticket_agent = new_instance
        elif agent_name == "chitchat_agent":
            state_graph.chitchat_agent = new_instance
        elif agent_name == "summary_agent":
            state_graph.summary_agent = new_instance
        elif agent_name == "memory_merge_agent":
            state_graph.memory_merge_agent = new_instance

    return {
        "status": "restarted",
        "agent": agent_name,
        "info": gossip_coordinator.get_agent_restart_info(agent_name),
    }


@app.get("/agents/{agent_name}/info")
def get_agent_info(agent_name: str):
    """获取 Agent 详细信息"""
    return {
        "agent": agent_name,
        "restart_info": gossip_coordinator.get_agent_restart_info(agent_name),
        "clock": gossip_coordinator.get_agent_clock(agent_name),
        "health": gossip_coordinator.check_agent_health(agent_name),
    }


@app.get("/gossip/status")
def get_gossip_status():
    """获取 Gossip 协调器状态"""
    return gossip_coordinator.get_status_report()


# ================================================================
#  Gossip 去中心化同步 API
# ================================================================

class GossipPayload(BaseModel):
    state: dict = {}
    vector_clock: dict = {}
    node_id: str = ""
    timestamp: float = 0
    agent_name: str = ""
    agent_clock: dict = {}


@app.post("/gossip/receive")
def receive_gossip(payload: GossipPayload):
    """
    接收来自其他节点的 Gossip 消息

    用于节点间去中心化状态同步：
    - 向量时钟冲突检测
    - Agent 健康状况传播
    - 降级状态同步
    """
    result = gossip_coordinator.receive_gossip_message(payload.model_dump())
    return {
        "status": "accepted" if result["accepted"] else "rejected",
        "relation": result["relation"],
        "local_clock": result["local_clock"],
    }


# ================================================================
#  降级管理 API
# ================================================================

@app.get("/degradation/status")
def get_degradation_status():
    """获取当前降级状态"""
    return degradation_manager.get_status_report()


@app.post("/degradation/reset")
def reset_degradation():
    """重置所有降级状态（恢复服务）"""
    degradation_manager.reset_all()
    return {"status": "reset", "message": "所有降级状态已重置"}


@app.post("/degradation/llm/off")
def set_llm_off():
    """手动将 LLM 标记为不可用（测试用）"""
    degradation_manager.set_llm_available(False)
    return {"status": "llm_off", "message": "LLM 已标记为不可用"}


@app.post("/degradation/llm/on")
def set_llm_on():
    """手动将 LLM 标记为可用"""
    degradation_manager.set_llm_available(True)
    return {"status": "llm_on", "message": "LLM 已恢复可用"}


# ================================================================
#  RAG 文档索引与检索 API
# ================================================================

@app.post("/index", response_model=IndexResponse)
async def index_document(
    file: UploadFile = File(...),
    operator: str = Form(default="admin"),
    tags: str = Form(default=""),
    rebuild: bool = Form(default=False),
):
    """
    上传文档并索引到向量知识库

    支持的格式：PDF、DOCX、XLSX、TXT、MD、CSV、图片
    """
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.index_document(
            file_path=tmp_path,
            operator=operator,
            tags=tag_list,
            rebuild=rebuild,
        )
        return IndexResponse(
            status="success",
            doc_id=result["doc_id"],
            file_name=file.filename,
            version=result["version"],
            chapters=result["chapters"],
            parent_chunks=result["parent_chunks"],
            child_chunks=result["child_chunks"],
            vector_stats=result.get("vector_stats"),
        )
    except IndexConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query_knowledge(request: QueryRequest):
    """检索知识库"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")

    try:
        result = pipeline.retrieve(
            query=request.query,
            doc_id=request.doc_id,
            tags=request.tags,
        )

        return QueryResponse(
            query=request.query,
            rewritten_query=result.get("rewritten_query", ""),
            child_hits=result.get("child_hits", []),
            parent_chunks=result.get("parent_chunks", []),
            context=result.get("context", ""),
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.get("/documents")
def list_documents():
    """列出所有已索引文档"""
    try:
        docs = pipeline.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除指定文档"""
    try:
        count = pipeline.delete_document(doc_id)
        if count == 0:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"status": "deleted", "doc_id": doc_id, "points_removed": count}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
#  系统健康检查
# ================================================================

@app.get("/health")
def health_check():
    """系统健康检查"""
    return {
        "status": "running",
        "version": "3.0.0",
        "degradation": degradation_manager.get_status_report(),
        "gossip": gossip_coordinator.get_status_report(),
    }