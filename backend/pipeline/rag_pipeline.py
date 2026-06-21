"""
RAG 主流水线：编排文档处理、切块、向量化、检索的完整流程。

索引流程:
  1. 解析文档 → 提取元数据 + 原始文本
  2. 转换为 Markdown（统一中间格式）
  3. DeepSeek 模型清洗（去除脏数据、修正格式）
  4. 冲突检测 & 版本管理
  5. 章节切分 → 6. 父子块切分
  7. 父块原文 → SQLite ParentStore
  8. 稠密向量 + 稀疏向量（BM25 TF-IDF）→ Qdrant 子块 Collection
  9. 章节向量 → Qdrant 章节 Collection

检索流程:
  1. Query 改写（LLM，携带对话历史）
  2. 章节粗筛 → 稠密+稀疏混合检索 → Qdrant RRF → top20 子块
  3. LLM Rerank 重排 → top5 高相关子块
  4. 根据子块 parent_id 从 SQLite 拉取完整父块
  5. 组装上下文 + 章节溯源
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from backend.config import settings
from backend.preprocessing.parser import DocumentParser, DocumentMeta
from backend.preprocessing.cleaner import DocumentCleaner
from backend.preprocessing.markdown_converter import MarkdownConverter
from backend.preprocessing.kimi_cleaner import DocCleaner
from backend.chunking.toc_extractor import Chapter, TOCExtractor
from backend.chunking.parent_child_splitter import (
    ChapterSplitter,
    ParentChildSplitter,
    ChildChunk,
    ParentChunk,
)
from backend.embedding.embedder import (
    Embedder, _init_embedding_worker, _embed_batch_worker, _calc_safe_worker_count,
)
from backend.embedding.sparse_embedder import SparseEmbedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.query_rewriter import QueryRewriter
from backend.retrieval.reranker import LLMReranker
from backend.storage.parent_store import ParentStore
from backend.agents.structuring_agent import DocumentStructuringAgent


class IndexConflictError(Exception):
    """索引冲突异常"""
    pass


class RAGPipeline:
    """
    RAG 主流水线

    索引流程: 解析 → 清洗 → 分块 → 父块存SQLite → 子块向量化(稠密+稀疏) → Qdrant
    检索流程: Query改写 → 混合检索 → Rerank → 父块拉取 → 上下文组装
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.cleaner = DocumentCleaner()
        self.md_converter = MarkdownConverter()
        self.doc_cleaner = DocCleaner()
        self.chapter_splitter = ChapterSplitter()
        self.parent_child_splitter = ParentChildSplitter(
            child_chunk_size=settings.child_chunk_size,
            child_chunk_overlap=settings.child_chunk_overlap,
            child_chunks_per_parent=settings.child_chunks_per_parent,
        )
        self.embedder = Embedder()
        # 共享单例 SparseEmbedder：先加载已持久化的词表（如果存在），
        # 后续 index_document 时增量添加新词
        self.sparse_embedder = SparseEmbedder()
        self.sparse_embedder.load_vocab()
        logger.info(f"SparseEmbedder 初始化: {self.sparse_embedder.get_stats()}")
        self.vector_store = VectorStore()
        self.parent_store = ParentStore()
        self.structuring_agent = DocumentStructuringAgent()

        # 检索组件（延迟初始化）
        self._retriever: Optional[HybridRetriever] = None
        self._query_rewriter: Optional[QueryRewriter] = None
        self._reranker: Optional[LLMReranker] = None

    def close(self):
        """关闭所有资源连接（Qdrant、Embedder 等）"""
        try:
            self.vector_store.close()
        except Exception:
            pass
        try:
            self.embedder.close()
        except Exception:
            pass

    # =====================================================================
    #  索引流程
    # =====================================================================

    def index_document(
        self,
        file_path: str,
        operator: str = None,
        tags: list[str] = None,
        rebuild: bool = False,
        display_name: str = None,
        progress_callback=None,
    ) -> dict:
        """
        索引单个文档

        Args:
            progress_callback: 进度回调函数，签名为 func(progress_obj)
                              progress_obj 为 dict，包含 stage, stage_name, progress, message 等

        Returns:
            索引统计与元数据信息
        """
        logger.info(f"========== 开始索引文档: {file_path} ==========")

        operator = operator or settings.default_operator
        tags = tags or []

        # 各阶段权重（百分比，总和 100）
        stage_weights = {
            "parsing": 10,
            "cleaning": 25,
            "chunking": 15,
            "vectorizing": 50,  # 向量化 + 存储合并，含章节向量
        }
        stage_base = {}     # 每个阶段开始时的基准进度
        _cumulative = 0
        for st, w in stage_weights.items():
            stage_base[st] = _cumulative
            _cumulative += w

        _current_stage = "initializing"

        def _set_stage(stage: str, stage_name: str, message: str = ""):
            nonlocal _current_stage
            _current_stage = stage
            if progress_callback:
                progress_callback({
                    "stage": stage,
                    "stage_name": stage_name,
                    "progress": stage_base.get(stage, 0),
                    "message": message,
                })

        def _update_stage_detail(sub_progress_pct: int, message: str = ""):
            """当前阶段内细粒度更新，sub_progress_pct 为当前阶段内的 0-100"""
            nonlocal _current_stage
            if progress_callback:
                base = stage_base.get(_current_stage, 0)
                w = stage_weights.get(_current_stage, 0)
                overall = base + int(w * min(max(sub_progress_pct, 0), 100) / 100)
                progress_callback({
                    "stage": _current_stage,
                    "message": message,
                    "progress": overall,
                })

        # ---------- Stage: parsing (10%) ----------
        _set_stage("parsing", "解析文档", "正在读取文件...")
        if rebuild:
            logger.info("rebuild=True，将忽略冲突策略直接覆盖")
            doc = self._parse_document(file_path, operator, tags, display_name)
        else:
            doc = self._parse_document(file_path, operator, tags, display_name)
        _update_stage_detail(100, f"解析完成: {doc.title}")

        # ---------- Stage: cleaning (25%) ----------
        _set_stage("cleaning", "清洗文档", "转换为 Markdown...")
        doc = self._convert_to_markdown(doc)
        _update_stage_detail(30, "Markdown 转换完成，开始 LLM 清洗...")
        doc = self._clean_with_llm(doc, file_path)
        _update_stage_detail(80, "LLM 清洗完成，本地补充清洗...")

        if rebuild:
            doc.doc_meta.version = 1
            self.vector_store.delete_by_doc_id(doc.doc_meta.doc_id)
            self.parent_store.delete_by_doc_id(doc.doc_meta.doc_id)
            self.parent_store.delete_children_by_doc_id(doc.doc_meta.doc_id)
        else:
            # Step 4: 冲突检测 & 版本管理
            resolution = self.vector_store.resolve_conflict(
                doc.doc_meta.doc_id, doc.doc_meta
            )
            action = resolution["action"]

            if action == "reject":
                raise IndexConflictError(resolution["message"])
            elif action == "skip":
                return {
                    "status": "skipped",
                    "message": resolution["message"],
                    "doc_id": doc.doc_meta.doc_id,
                    "version": resolution["version"],
                    "file_name": doc.doc_meta.file_name,
                }
            elif action in ("overwrite", "keep_both"):
                doc.doc_meta.version = resolution["version"]
                logger.info(f"冲突解决: {resolution['message']}")
            else:
                logger.info("新文档，创建索引")

        # 本地补充清洗
        self.cleaner.clean_pages(doc.pages)
        doc.raw_text = "\n".join(p.text for p in doc.pages)
        _update_stage_detail(100, f"本地清洗完成, 总字符数: {len(doc.raw_text)}")
        logger.info(f"本地清洗完成, 总字符数: {len(doc.raw_text)}")

        # 保存清洗后的 Markdown 到本地文件，方便调试
        self._save_markdown(doc)

        # ---------- Step 3.5: LLM 语义结构化（将商品信息单元识别为 ### 独立章节） ----------
        if settings.enable_llm_structuring:
            _update_stage_detail(100, "LLM 语义结构化...")
            doc = self._restructure_with_llm(doc)
            _update_stage_detail(100, f"LLM 结构化完成, 总字符数: {len(doc.raw_text)}")
            # 保存结构化后的 Markdown 到本地文件，方便调试
            self._save_markdown(doc, suffix="_structured")

        # ---------- Stage: chunking (15%) ----------
        _set_stage("chunking", "切分文档", "检测章节结构...")
        chapters = self.chapter_splitter.split(doc)
        _update_stage_detail(40, f"章节切分完成: {len(chapters)} 个章节")
        child_chunks, parent_chunks = self.parent_child_splitter.split_chapters(
            chapters, doc_meta=doc.doc_meta
        )
        _update_stage_detail(100, f"父子块切分完成: {len(child_chunks)} 子块, {len(parent_chunks)} 父块")
        logger.info(f"章节切分完成, 共 {len(chapters)} 个章节")
        logger.info(f"父子块切分完成: {len(child_chunks)} 子块, {len(parent_chunks)} 父块")

        # === 新增：生成汇总型子块（修复概括性查询如 "有多少个商品分类" 检索不到的问题）===
        summary_chunks = self.parent_child_splitter.generate_summary_chunks(
            chapters, doc_meta=doc.doc_meta, min_children=3
        )
        if summary_chunks:
            logger.info(f"汇总型子块生成: {len(summary_chunks)} 个")
            # 汇总块放在最前面，被检索时优先返回
            child_chunks = summary_chunks + child_chunks
            logger.info(f"子块总数（含汇总块）: {len(child_chunks)}")

        # 父块原文存入 SQLite（作为 chunking 阶段的一部分）
        self.parent_store.store_parents(
            parent_chunks,
            doc_id=doc.doc_meta.doc_id,
            file_name=doc.doc_meta.file_name,
        )
        logger.info(f"父块原文入 SQLite 完成: {len(parent_chunks)} 个")

        # ---------- Stage: vectorizing + storing (40%) ----------
        # 入库前：将 status 设为 active，确保写入 Qdrant 的数据可被检索
        doc.doc_meta.status = "active"

        # 生产者-消费者流水线：读文本 → 向量化 → 批量入库，全程流式并行
        _set_stage("vectorizing", "向量化", "准备文本数据...")
        if settings.child_vectorize_with_chapter_title:
            child_texts = [
                (c.chapter_title + "\n" + c.content) if c.chapter_title else c.content
                for c in child_chunks
            ]
        else:
            child_texts = [c.content for c in child_chunks]
        _update_stage_detail(2, f"准备 {len(child_texts)} 条文本，开始向量化流水线...")

        import time
        import queue
        import threading
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime, timezone

        t_pipe_start = time.time()
        total_texts = len(child_texts)

        # ===== 1. 稀疏向量：fit + 编码（主线程，轻量操作） =====
        _update_stage_detail(5, "稀疏向量 fit...")
        self.sparse_embedder.fit(child_texts)
        self.sparse_embedder.save_vocab()
        _update_stage_detail(8, "稀疏向量编码...")
        sparse_embeddings = self.sparse_embedder.encode_batch(child_texts)
        _update_stage_detail(10, f"稀疏编码完成: {len(sparse_embeddings)} 条")

        # 子块原文 SQLite 存储（与流水线并发）
        _update_stage_detail(11, "子块原文入 SQLite...")
        self.parent_store.store_children(
            child_chunks,
            doc_id=doc.doc_meta.doc_id,
            file_name=doc.doc_meta.file_name,
        )

        # ===== 2. 根据配置 + 检测，决定用 GPU/CPU 及多进程/多线程 =====
        # 关键：优先读取 settings.use_gpu，其次自动检测
        device = self.embedder.device  # "cpu" / "cuda:0" / "cuda:1" ...
        is_gpu = device.startswith("cuda")
        # CPU 用多进程规避 GIL，GPU 用多线程（避免多个进程同时用 GPU 导致 OOM）
        use_multiprocessing = not is_gpu

        # 安全计算：根据可用内存限制 worker 数量（每个 worker 约需 2.5GB RAM）
        if use_multiprocessing:
            safe_workers = _calc_safe_worker_count(
                self.embedder.model_name, settings.embedding_workers
            )
            effective_workers = safe_workers
        else:
            effective_workers = settings.embedding_workers

        logger.info(
            f"[流水线] 模式: {'多进程(CPU)' if use_multiprocessing else '多线程(GPU)'}, "
            f"device={device}, "
            f"workers={effective_workers}, "
            f"embedding_batch={settings.embedding_batch_size}, "
            f"upsert_batch={settings.vector_upsert_batch_size}"
        )

        # ===== 3. 构建队列 =====
        text_queue = queue.Queue(maxsize=settings.text_queue_maxsize)
        vector_queue = queue.Queue(maxsize=settings.vector_queue_maxsize)
        stop_event = threading.Event()
        error_event = threading.Event()
        error_info: list = [None]  # [exception]

        # 进度追踪
        progress_lock = threading.Lock()
        progress_state = {
            "text_produced": 0,
            "text_consumed": 0,
            "vectors_produced": 0,
            "vectors_written": 0,
        }

        def _update_pipe_progress():
            """根据流水线进度更新阶段进度（10%-98% 区间）"""
            with progress_lock:
                produced = progress_state["text_produced"]
                consumed = min(progress_state["text_consumed"], total_texts)
                written = min(progress_state["vectors_written"], total_texts)

                # 加权平均：文本生产 10%，向量生成 60%，写入 30%
                # 上限 90，留给 post-pipeline（章节向量化等）10 个点
                pct = int(10 + 80 * (
                    0.10 * (produced / max(total_texts, 1)) +
                    0.60 * (consumed / max(total_texts, 1)) +
                    0.30 * (written / max(total_texts, 1))
                ))
                _update_stage_detail(
                    pct,
                    f"流水线: 文本{produced}/{total_texts} | "
                    f"向量{consumed}/{total_texts} | "
                    f"写入{written}/{total_texts}"
                )

        # ===== 4. 生产者线程：投喂文本批次到 text_queue =====
        def _producer():
            """IO 线程：读取文本分块，投喂到队列"""
            try:
                batch_size = settings.embedding_batch_size
                for batch_idx, start in enumerate(range(0, total_texts, batch_size)):
                    if error_event.is_set():
                        break
                    end = min(start + batch_size, total_texts)
                    batch = child_texts[start:end]
                    text_queue.put((batch_idx, start, batch), timeout=30)
                    with progress_lock:
                        progress_state["text_produced"] = end
                    _update_pipe_progress()
                logger.info(f"[生产者] 文本投喂完成: {total_texts} 条")
            except Exception as e:
                logger.error(f"[生产者] 异常: {e}")
                error_info[0] = e
                error_event.set()
            finally:
                # 发送终止信号
                for _ in range(settings.embedding_workers + 2):
                    try:
                        text_queue.put(None, timeout=5)
                    except queue.Full:
                        pass

        # ===== 5. 消费者（Embedding）：多进程/多线程 =====
        if use_multiprocessing:
            # 多进程模式：使用 ProcessPoolExecutor，每个子进程加载独立模型副本
            # executor 内部管理跨进程队列，比手动 queue.Queue 更可靠
            # 关键：在 worker 初始化失败时回退到单进程模式，避免整个流水线崩溃
            import concurrent.futures
            ctx = mp.get_context("spawn")
            _mp_executor = None
            _mp_success = False

            try:
                _mp_executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=effective_workers,
                    mp_context=ctx,
                    initializer=_init_embedding_worker,
                    initargs=(settings.embedding_model, device),
                )

                # 批量提交所有任务
                _futures: dict = {}  # {future: (batch_idx, start, len)}
                for batch_idx, start in enumerate(range(0, total_texts, settings.embedding_batch_size)):
                    end = min(start + settings.embedding_batch_size, total_texts)
                    batch = child_texts[start:end]
                    fut = _mp_executor.submit(_embed_batch_worker, batch)
                    _futures[fut] = (batch_idx, start, len(batch))
                    with progress_lock:
                        progress_state["text_produced"] = end
                    _update_pipe_progress()

                # 收集结果，按完成顺序放入 vector_queue
                for fut in concurrent.futures.as_completed(_futures):
                    try:
                        vecs = fut.result()
                        batch_idx, start, batch_len = _futures[fut]
                        vector_queue.put((batch_idx, start, vecs), timeout=60)
                        with progress_lock:
                            progress_state["text_consumed"] = start + batch_len
                            progress_state["vectors_produced"] = (
                                progress_state["vectors_produced"] + batch_len
                            )
                        _update_pipe_progress()
                    except Exception as e:
                        logger.error(f"[Embedding Worker] 单批次异常: {e}")
                        # 某个批次失败不中断整个流水线，标记为软错误继续处理
                        # 但如果是 BrokenProcessPool，则直接 fallback
                        if "BrokenProcessPool" in type(e).__name__ or "terminated abruptly" in str(e):
                            error_info[0] = e
                            error_event.set()
                            break
                        # 其他 worker 级错误：记录但继续（缺失的向量会用零向量填充）
                        continue

                _mp_success = True

            except Exception as pool_err:
                logger.warning(
                    f"[流水线] 多进程模式失败: {type(pool_err).__name__}: {pool_err}. "
                    f"回退到单进程模式（慢但可靠）"
                )

            finally:
                if _mp_executor:
                    try:
                        _mp_executor.shutdown(wait=False)
                    except Exception:
                        pass

            # 如果多进程模式失败（部分/全部任务未完成），回退到单进程
            if not _mp_success or error_event.is_set():
                logger.warning(
                    f"[流水线] 多进程嵌入失败，切换到单进程模式重新处理 "
                    f"({total_texts} 条文本)"
                )
                # 重置进度
                error_event.clear()
                error_info[0] = None
                with progress_lock:
                    progress_state["text_consumed"] = 0
                    progress_state["vectors_produced"] = 0

                # 单进程：直接使用主线程 embedder
                _fallback_embedder = self.embedder
                for batch_idx, start in enumerate(range(0, total_texts, settings.embedding_batch_size)):
                    end = min(start + settings.embedding_batch_size, total_texts)
                    batch = child_texts[start:end]
                    try:
                        vecs = _fallback_embedder.embed_batch(batch, batch_size=len(batch))
                        vector_queue.put((batch_idx, start, vecs), timeout=60)
                        with progress_lock:
                            progress_state["text_consumed"] = end
                            progress_state["vectors_produced"] += len(batch)
                        _update_pipe_progress()
                    except Exception as fallback_err:
                        logger.error(f"[流水线] 单进程回退也失败: {fallback_err}")
                        # 生成零向量占位，避免完全失败
                        dim = settings.vector_dim
                        placeholder = [[0.0] * dim for _ in range(len(batch))]
                        vector_queue.put((batch_idx, start, placeholder), timeout=60)

            # 发送终止信号给 writer
            for _ in range(2):
                try:
                    vector_queue.put(None, timeout=5)
                except queue.Full:
                    pass
        else:
            # GPU 多线程模式：共享模型，线程并发（GPU 操作释放 GIL）
            _embedder = self.embedder

            def _embedding_worker():
                """线程：从 text_queue 取文本，批量向量化后放入 vector_queue"""
                try:
                    while not error_event.is_set():
                        try:
                            item = text_queue.get(timeout=2)
                        except queue.Empty:
                            continue
                        if item is None:
                            break
                        batch_idx, start, texts = item
                        if not texts:
                            continue
                        vecs = _embedder.embed_batch(texts, batch_size=len(texts))
                        vector_queue.put((batch_idx, start, vecs), timeout=60)
                        with progress_lock:
                            progress_state["text_consumed"] = start + len(texts)
                            progress_state["vectors_produced"] = (
                                progress_state["vectors_produced"] + len(vecs)
                            )
                        _update_pipe_progress()
                except Exception as e:
                    logger.error(f"[Embedding Worker] 异常: {e}")
                    error_info[0] = e
                    error_event.set()

            # 启动线程
            threads = []
            for _ in range(effective_workers):
                t = threading.Thread(target=_embedding_worker, daemon=True)
                t.start()
                threads.append(t)

        # ===== 6. 写入线程：从 vector_queue 取向量，攒够一批后批量 upsert =====
        _write_buffer = []  # [(batch_idx, start, vecs), ...]
        _write_buffer_vectors = 0  # 缓冲中累计向量条数

        def _flush_buffer(final: bool = False) -> int:
            """将写缓冲中积累的向量批量 upsert 到 Qdrant，返回写入条数"""
            nonlocal _write_buffer, _write_buffer_vectors
            if not _write_buffer:
                return 0
            # 按 batch_idx 排序，从 child_chunks 和 sparse_embeddings 取对应数据
            _write_buffer.sort(key=lambda x: x[0])
            write_chunks = []
            write_dense = []
            write_sparse = []
            for bi, st, vecs in _write_buffer:
                end = st + len(vecs)
                write_chunks.extend(child_chunks[st:end])
                write_dense.extend(vecs)
                write_sparse.extend(sparse_embeddings[st:end])

            n = len(write_chunks)
            if n > 0:
                self.vector_store.store_children_batch(
                    write_chunks, write_dense, write_sparse,
                    doc_meta=doc.doc_meta, wait=final,
                )
                with progress_lock:
                    progress_state["vectors_written"] += n
                _update_pipe_progress()

            _write_buffer = []
            _write_buffer_vectors = 0
            return n

        def _writer():
            """写入线程：攒批后批量写入 Qdrant"""
            nonlocal _write_buffer_vectors
            try:
                pending = {}          # {batch_idx: (start, vecs)}
                next_to_write = 0     # 下一个要写入的 batch_idx
                local_written = 0     # 本地追踪已写入条数

                while not error_event.is_set():
                    # 检查是否所有向量都已写入
                    if local_written >= total_texts:
                        break

                    try:
                        item = vector_queue.get(timeout=5)
                    except queue.Empty:
                        # 超时：可能是所有数据已投喂完毕，检查是否有积压
                        if _write_buffer:
                            local_written += _flush_buffer()
                        # 再次检查是否全部完成
                        if local_written >= total_texts:
                            break
                        # 也有可能生产者/embedding 还没完成，重置超时次数
                        continue

                    if item is None:
                        # 收到终止信号：flush 剩余缓冲后退出
                        if _write_buffer:
                            local_written += _flush_buffer(final=True)
                        continue

                    batch_idx, start, vecs = item
                    pending[batch_idx] = (start, vecs)

                    # 按顺序攒批写入（保证顺序，避免乱序）
                    while next_to_write in pending:
                        _start, _vecs = pending.pop(next_to_write)
                        _write_buffer.append((next_to_write, _start, _vecs))
                        _write_buffer_vectors += len(_vecs)
                        next_to_write += 1

                        # 攒够 upsert_batch_size 条向量就写入
                        if _write_buffer_vectors >= settings.vector_upsert_batch_size:
                            local_written += _flush_buffer()

                # 写入剩余缓冲
                local_written += _flush_buffer(final=True)

                logger.info(f"[写入线程] 完成: {local_written} 条向量已入库")
            except Exception as e:
                logger.error(f"[写入线程] 异常: {e}")
                error_info[0] = e
                error_event.set()

        # ===== 7. 启动流水线 =====
        if use_multiprocessing:
            # 多进程模式：所有批次已提交到 ProcessPoolExecutor，结果已收集到 vector_queue
            # 只需启动 writer 线程消费 vector_queue
            _update_stage_detail(75, "向量化已提交，等待写入...")
            writer_thread = threading.Thread(target=_writer, daemon=True)
            writer_thread.start()
            writer_thread.join(timeout=600)
            if error_event.is_set():
                err = error_info[0]
                raise RuntimeError(f"向量化流水线失败: {err}")
            _mp_executor.shutdown(wait=True)
        else:
            # GPU 多线程模式：启动 producer + writer
            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()
            writer_thread = threading.Thread(target=_writer, daemon=True)
            writer_thread.start()
            producer_thread.join()
            _update_stage_detail(75, "文本投喂完成，等待向量化...")
            writer_thread.join(timeout=600)
            if error_event.is_set():
                err = error_info[0]
                raise RuntimeError(f"向量化流水线失败: {err}")
            for t in threads:
                t.join(timeout=60)

        # 最终 flush 缓冲（确保落盘）
        _flush_buffer(final=True)

        # 最后一批确保落盘
        _update_stage_detail(92, "向量入库完成，等待落盘...")

        t_pipe_end = time.time()
        logger.info(
            f"[流水线] 完成: {total_texts} 条, "
            f"耗时 {t_pipe_end - t_pipe_start:.1f}s, "
            f"平均 {total_texts / max(t_pipe_end - t_pipe_start, 0.01):.1f} 条/s"
        )

        # 更新 vector_count 和 updated_at（status 已在向量化前设为 active）
        doc.doc_meta.vector_count = len(child_chunks)
        doc.doc_meta.updated_at = datetime.now(timezone.utc).isoformat()

        # 章节向量化（数量通常较小，直接批量处理）
        _update_stage_detail(95, "章节向量化...")
        chapter_texts = [
            (ch.chapter_path if ch.chapter_path else ch.title) + "\n" + (ch.content[:1500] if ch.content else "")
            for ch in chapters
        ]
        if chapter_texts:
            chapter_embeddings = self.embedder.embed_batch(chapter_texts, batch_size=50)
            _update_stage_detail(98, f"章节向量入库: {len(chapter_texts)} 节...")
            self.vector_store.store_chapters(chapters, chapter_embeddings, doc_meta=doc.doc_meta)
            logger.info(f"章节向量入库完成: {len(chapter_texts)} 节")

        _update_stage_detail(100, f"Qdrant 入库完成 [v{doc.doc_meta.version}]")
        logger.info(f"Qdrant 入库完成 [v{doc.doc_meta.version}]")

        # 元数据（vector_count、status、updated_at）已在入库前写入 doc_meta，
        # 随 _meta_to_payload 一起存入 Qdrant，无需额外 update_doc_meta 调用
        logger.info(
            f"文档 [{doc.doc_meta.file_name}] 索引完成: "
            f"child_chunks={len(child_chunks)}, vector_count={doc.doc_meta.vector_count}"
        )

        stats = self.get_stats()
        logger.info(f"========== 索引完成 ==========")
        return {
            "status": "success",
            "doc_id": doc.doc_meta.doc_id,
            "file_name": doc.doc_meta.file_name,
            "version": doc.doc_meta.version,
            "file_hash": doc.doc_meta.file_hash[:16],
            "operator": operator,
            "tags": tags,
            "chapters": len(chapters),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "sparse_vocab_size": self.sparse_embedder._next_id,
            "vector_stats": stats,
        }

    # =====================================================================
    #  子步骤：解析 → 转换 → 清洗
    # =====================================================================

    def _parse_document(self, file_path: str, operator: str, tags: list[str] = None, display_name: str = None):
        """Step 1: 解析文档 + 提取元数据"""
        doc = self.parser.parse(file_path, operator=operator)
        doc.doc_meta.tags = tags or []
        if display_name:
            doc.doc_meta.file_name = display_name
        ext = Path(file_path).suffix.lower()
        logger.info(
            f"文档解析完成: {doc.title}  [{ext}] "
            f"doc_id={doc.doc_meta.doc_id}, 字符数={len(doc.raw_text)}"
        )
        return doc

    def _convert_to_markdown(self, doc):
        """Step 2: 统一转换为 Markdown 格式"""
        if not settings.convert_to_markdown_first:
            logger.info("Markdown 转换已禁用")
            return doc
        doc = self.md_converter.convert(doc)
        logger.info(f"Markdown 转换完成, 字符数: {len(doc.raw_text)}")
        return doc

    def _clean_with_llm(self, doc, file_path: str):
        """Step 3: DeepSeek 模型深度清洗"""
        if not settings.enable_llm_cleaning:
            logger.info("DeepSeek 清洗已禁用")
            return doc

        ext = Path(file_path).suffix.lower()
        if ext in (".xlsx", ".xls"):
            source_type = "excel"
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            source_type = "image"
        else:
            source_type = "document"

        doc = self.doc_cleaner.clean(doc, source_type=source_type)
        logger.info(f"DeepSeek 清洗完成, 字符数: {len(doc.raw_text)}")

        # 清洗后修复标题层级：LLM 逐页清洗可能把子标题提升为 #
        doc = self._normalize_headings(doc)

        # 清洗后清除 PDF 内置目录，改用文本模式检测章节
        # PDF 内置目录往往有误（条目少、标题与正文粘连），而清洗后的 Markdown 已有完整标题层级
        doc.toc = None
        logger.info("已清除 PDF 内置目录，后续将使用 Markdown 标题模式检测章节")

        return doc

    def _normalize_headings(self, doc):
        """修复被 LLM 清洗打乱的 Markdown 标题层级"""
        import re
        lines = doc.raw_text.split('\n')
        fixed_lines = []

        # 编号型标题识别正则（如 "## 1. xxx"、"## 2. xxx"、"## 11.1 xxx" 等）
        numbered_heading_pattern = re.compile(r'^#{2,}\s+[\d一二三四五六七八九十]+[.、\s]')
        # 编号型标题识别正则（不含编号前缀，如 "## 凑单与拆分技巧" 但前面的同级章节已出现编号）
        pure_number_pattern = re.compile(r'^#{2,}\s+\d+(\.\d+)*[.、\s]')

        first_h1_found = False
        for line in lines:
            stripped = line.strip()
            md_match = re.match(r'^(#{1,6})\s+(.+)', stripped)

            if md_match:
                level = len(md_match.group(1))
                title = md_match.group(2)

                if level == 1:
                    if not first_h1_found:
                        # 第一个 # 保留为主标题
                        first_h1_found = True
                        fixed_lines.append(line)
                    else:
                        # 后续的 # 降级为 ###（子章节）
                        fixed_lines.append('### ' + title)
                elif level == 2:
                    # ## 标题需要进一步判断：是否为编号型内容标题（如 "## 1. xxx"）
                    if numbered_heading_pattern.match(stripped) or pure_number_pattern.match(stripped):
                        # 编号型二级标题 → 降级为三级标题（内容型子章节）
                        fixed_lines.append('### ' + title)
                    else:
                        # 非编号型二级标题（如 "## 高级购物技巧"）→ 保留为二级章节标题
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        # 更新 raw_text 和 pages
        doc.raw_text = '\n'.join(fixed_lines)
        if doc.pages:
            doc.pages[0].text = '\n'.join(fixed_lines)
        logger.info("标题层级修复完成: 编号型 ## 标题降级为 ###, 非编号型 ## 保留为章节标题")
        return doc

    def _restructure_with_llm(self, doc):
        """
        Step 3.5: LLM 语义结构化

        调用 DocumentStructuringAgent 对文档进行语义分析，将商品信息单元
        识别为 `###` 独立章节，确保后续 TOCExtractor 和 parent_child_splitter
        不会将商品信息（名称、价格、规格）切分到不同子块中。
        """
        logger.info(f"[LLM 结构化] 开始处理文档: {doc.doc_meta.file_name}")
        try:
            raw_markdown = doc.raw_text
            structured_markdown = self.structuring_agent.structure(raw_markdown)

            doc.raw_text = structured_markdown
            if doc.pages:
                doc.pages[0].text = structured_markdown

            # 清除 PDF 内置目录，后续使用 Markdown 标题模式检测章节
            doc.toc = None

            logger.info(
                f"[LLM 结构化] 完成: {len(raw_markdown)} → {len(structured_markdown)} 字符"
            )
            return doc
        except Exception as e:
            logger.warning(f"[LLM 结构化] 失败: {e}，降级为原始文档")
            return doc

    def _save_markdown(self, doc, suffix: str = ""):
        """保存 Markdown 到本地文件"""
        import os
        output_dir = Path(__file__).parent.parent / "markdown"
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(doc.doc_meta.file_name).stem
        safe_name = base_name.replace("/", "_").replace("\\", "_")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{safe_name}{suffix}_{timestamp}.md"

        output_path.write_text(doc.raw_text, encoding="utf-8")
        logger.info(f"清洗后 Markdown 已保存: {output_path}")

    def index_directory(
        self,
        dir_path: str,
        operator: str = None,
        tags: list[str] = None,
    ) -> list[dict]:
        """索引目录中的所有支持文档"""
        path = Path(dir_path)
        files = []
        for ext in settings.supported_extensions:
            files.extend(path.glob(f"**/*{ext}"))

        logger.info(f"发现 {len(files)} 个文档待索引")
        results = []

        for i, file_path in enumerate(files, 1):
            logger.info(f"\n处理 [{i}/{len(files)}]: {file_path}")
            try:
                result = self.index_document(str(file_path), operator=operator, tags=tags)
                results.append(result)
            except IndexConflictError as e:
                logger.warning(f"索引冲突，跳过: {file_path} - {e}")
                results.append({"status": "conflict", "file": str(file_path), "error": str(e)})
            except Exception as e:
                logger.error(f"索引失败: {file_path} - {e}")
                results.append({"status": "error", "file": str(file_path), "error": str(e)})

        return results

    # =====================================================================
    #  检索流程
    # =====================================================================

    def _get_retriever(self) -> HybridRetriever:
        """获取或创建检索器（延迟初始化，共享 sparse_embedder）"""
        if self._retriever is None:
            # 确保 sparse_embedder 已加载最新词表
            self.sparse_embedder.load_vocab()
            self._query_rewriter = QueryRewriter(sparse_embedder=self.sparse_embedder)
            self._reranker = LLMReranker()
            self._retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                parent_store=self.parent_store,
                query_rewriter=self._query_rewriter,
                reranker=self._reranker,
            )
        return self._retriever

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        执行检索

        Args:
            query: 用户查询
            doc_id: 限定文档
            tags: 业务标签过滤
            conversation_history: 对话历史（用于 Query 改写）

        Returns:
            检索结果字典
        """
        retriever = self._get_retriever()
        result = retriever.search(
            query=query,
            top_k=settings.rerank_top_k,
            doc_id=doc_id,
            tags=tags,
            conversation_history=conversation_history,
        )

        # 调试日志
        logger.info(f"========== RAG 检索结果 (query={query}) ==========")
        if result.get("rewritten_query"):
            logger.info(f"  → 改写后查询: {result['rewritten_query']}")
        if result.get("hyde_used"):
            logger.info(f"  → HyDE 查询增强: 已启用")
        logger.info(f"  → 子块候选: {len(result.get('child_hits', []))}")
        for c in result.get("child_hits", []):
            logger.info(f"    [{c['score']:.4f}] [{c['chapter_title']}] {c['content'][:100]}...")
        logger.info(f"  → 父块来源: {len(result.get('parent_chunks', []))}")
        for p in result.get("parent_chunks", []):
            logger.info(f"    [{p.get('score', 0):.4f}] {p['chapter_title']} ({p['content_length']} 字符)")
        logger.info(f"  → 上下文长度: {len(result.get('context', ''))} 字符")
        logger.info(f"==============================================")

        return result

    # =====================================================================
    #  文档管理
    # =====================================================================

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档"""
        return self.vector_store.list_documents()

    def get_doc_versions(self, doc_id: str) -> list[dict]:
        """获取某文档的版本历史"""
        return self.vector_store.get_doc_versions(doc_id)

    def delete_document(self, doc_id: str) -> int:
        """删除指定文档的所有数据（向量 + 父块原文 + 子块原文）"""
        vector_deleted = self.vector_store.delete_by_doc_id(doc_id)
        parent_deleted = self.parent_store.delete_by_doc_id(doc_id)
        child_deleted = self.parent_store.delete_children_by_doc_id(doc_id)
        logger.info(f"删除文档 {doc_id}: 向量 {vector_deleted} 条, 父块 {parent_deleted} 条, 子块 {child_deleted} 条")
        return vector_deleted + parent_deleted + child_deleted

    def archive_document(self, doc_id: str):
        """归档文档（软删除）"""
        self.vector_store.archive_doc(doc_id)

    # =====================================================================
    #  管理方法
    # =====================================================================

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        stats = self.vector_store.get_stats()
        try:
            with self.parent_store._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()
                stats["parent_store"] = row[0] if row else 0
        except Exception:
            stats["parent_store"] = 0
        return stats

    def clear(self):
        """清空所有数据（向量 + 父块原文）"""
        self.vector_store.clear_all()
        try:
            with self.parent_store._get_conn() as conn:
                conn.execute("DELETE FROM parent_chunks")
                conn.commit()
        except Exception:
            pass
        logger.info("已清空所有数据")