"""
向量数据库操作模块：基于 Qdrant 存储和管理向量数据。

存储两个层级：
  1. 章节向量（chapter）：粗粒度定位 + 元数据过滤
  2. 子块向量（child）：稠密 + 稀疏双向量检索

父块原文存储在 backend.storage.parent_store.ParentStore（SQLite）。

所有层级均携带文档元数据（版本、时间、操作者等），
支持基于 doc_id 的版本管理与冲突解决。
"""

from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
    HasIdCondition,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Prefetch,
    Fusion,
    FusionQuery,
    PayloadSchemaType,
    PointsSelector,
    PointIdsList,
    FilterSelector,
)
from loguru import logger

from backend.config import settings
from backend.chunking.toc_extractor import Chapter
from backend.chunking.parent_child_splitter import ChildChunk
from backend.preprocessing.parser import DocumentMeta


class VectorStore:
    """Qdrant 向量存储管理器（含元数据与版本管理）"""

    # 两个 Collection 名称
    COLLECTION_CHAPTER = f"{settings.qdrant_collection}_chapters"
    COLLECTION_CHILD = f"{settings.qdrant_collection}_children"

    # 向量名称
    VECTOR_DENSE = "dense"      # 稠密向量
    VECTOR_SPARSE = "sparse"    # 稀疏向量

    def __init__(self):
        # 为大量向量写入提供足够的超时时间
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=300)
        # 初始化状态：记录哪些 collection 可用（用于容错搜索）
        self._collections_ready = {
            self.COLLECTION_CHAPTER: False,
            self.COLLECTION_CHILD: False,
        }
        # 运行模式: 'full' = 有 children collection, 'chapter_only' = 仅章节, 'broken' = 完全不可用
        self._mode = "chapter_only"
        try:
            self._ensure_collections()
            if self._collections_ready.get(self.COLLECTION_CHILD, False):
                self._mode = "full"
            elif self._collections_ready.get(self.COLLECTION_CHAPTER, False):
                self._mode = "chapter_only"
            else:
                self._mode = "broken"
            logger.info(f"VectorStore 运行模式: {self._mode} (章节={self._collections_ready.get(self.COLLECTION_CHAPTER)}, 子块={self._collections_ready.get(self.COLLECTION_CHILD)})")
        except Exception as e:
            self._mode = "broken"
            logger.error(f"Qdrant collection 初始化失败，服务将以降级模式运行: {e}")
            logger.info("提示: 1) 检查 Qdrant 服务是否运行 2) 重启 Qdrant 进程 3) 在 Qdrant 控制台手动创建 collection")

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    # =====================================================================
    #  Collection 管理（带超时保护 + 轮询检测 + 容错）
    # =====================================================================

    def _ensure_collections(self):
        """确保所需的 Collection 存在且向量配置正确（带超时保护，永不挂死服务启动）"""
        # 1. 章节 Collection — 仅稠密向量（无名向量）
        self._ensure_collection(
            self.COLLECTION_CHAPTER,
            require_named_vectors=False,
        )

        # 2. 子块 Collection — 稠密 + 稀疏双向量（命名向量）
        self._ensure_collection(
            self.COLLECTION_CHILD,
            require_named_vectors=True,
            require_sparse=True,
        )

        # 3. 为两个 Collection 创建 doc_id payload 索引（加速按 doc_id 的过滤与删除）
        self._ensure_payload_index(self.COLLECTION_CHAPTER, "doc_id", PayloadSchemaType.KEYWORD)
        self._ensure_payload_index(self.COLLECTION_CHILD, "doc_id", PayloadSchemaType.KEYWORD)

    def _ensure_collection(self, coll_name: str, require_named_vectors: bool = False, require_sparse: bool = False):
        """确保 Collection 存在且格式正确。失败时不抛异常。"""
        # 检查 collection 是否存在
        exists = False
        try:
            exists = self.client.collection_exists(coll_name)
        except Exception as e:
            logger.warning(f"Collection [{coll_name}] 存在性检查失败: {e}")
            # 无法连接 Qdrant，不标记为 ready（稍后搜索会重试）
            self._collections_ready[coll_name] = False
            return

        if not exists:
            self._create_collection(coll_name, require_named_vectors, require_sparse)
            try:
                self._collections_ready[coll_name] = self.client.collection_exists(coll_name)
            except Exception:
                self._collections_ready[coll_name] = False
            return

        # Collection 存在 —— 检查配置兼容性
        try:
            info = self.client.get_collection(coll_name)
        except Exception as e:
            logger.warning(f"Collection [{coll_name}] 读取配置失败: {e}")
            # 存在但读不到配置 → 可能 Qdrant 异常但 collection 在
            self._collections_ready[coll_name] = True
            return

        params = info.config.params
        needs_recreate = False

        if require_named_vectors:
            if not isinstance(params.vectors, dict):
                needs_recreate = True
            elif self.VECTOR_DENSE not in params.vectors:
                needs_recreate = True
            # 有命名 dense 向量但无 sparse → 接受降级（不重建）
        else:
            if isinstance(params.vectors, dict):
                needs_recreate = True

        if needs_recreate:
            logger.warning(f"Collection [{coll_name}] 格式不兼容，将重建")
            try:
                self.client.delete_collection(coll_name)
            except Exception as e:
                logger.warning(f"Collection [{coll_name}] 删除失败: {e}")
                self._collections_ready[coll_name] = True
                return
            self._create_collection(coll_name, require_named_vectors, require_sparse)
            try:
                self._collections_ready[coll_name] = self.client.collection_exists(coll_name)
            except Exception:
                self._collections_ready[coll_name] = False
        else:
            logger.debug(f"Collection [{coll_name}] 格式兼容，跳过")
            self._collections_ready[coll_name] = True

    def _create_collection(self, coll_name: str, require_named_vectors: bool, require_sparse: bool):
        """创建 Collection。带超时保护和轮询检测，防止 Qdrant 挂死服务启动。

        Windows 下 Qdrant 1.18.x 在创建 collection 时可能长时间不响应。
        策略（总耗时 ≈ 5+5+5+5+5 = 25s 最坏情况）：
          1. 快速路径 1: 直接 HTTP PUT (5s 超时) + 轮询 (5s)
          2. 快速路径 2: 仅 dense 向量的降级创建 (若原计划含 sparse)
          3. 慢速路径: 子线程 qdrant_client (5s 超时)
          4. 最后轮询: 5s
          5. 均失败: 记录告警，服务继续运行（降级模式）
        """
        import threading

        desc = "dense + sparse" if require_named_vectors else "dense"
        logger.info(f"创建 Collection: {coll_name} ({desc})")

        if require_named_vectors:
            create_kwargs = dict(
                collection_name=coll_name,
                vectors_config={
                    self.VECTOR_DENSE: VectorParams(
                        size=settings.vector_dim,
                        distance=Distance.COSINE,
                    ),
                },
            )
            if require_sparse:
                create_kwargs["sparse_vectors_config"] = {
                    self.VECTOR_SPARSE: SparseVectorParams(),
                }
        else:
            create_kwargs = dict(
                collection_name=coll_name,
                vectors_config=VectorParams(
                    size=settings.vector_dim,
                    distance=Distance.COSINE,
                ),
            )

        # 快速路径 1: requests HTTP PUT
        if self._try_create_via_http(coll_name, require_named_vectors, require_sparse):
            if self._poll_collection_ready(coll_name, poll_total=5, poll_interval=1):
                logger.info(f"Collection [{coll_name}] 已就绪 (HTTP PUT)")
                return

        # 快速路径 2: 仅 dense 向量的降级 collection（若原计划包含 sparse）
        if require_sparse:
            logger.info(f"尝试降级创建: {coll_name} (仅 dense)")
            if self._try_create_via_http(coll_name, require_named_vectors, False):
                if self._poll_collection_ready(coll_name, poll_total=5, poll_interval=1):
                    logger.info(f"Collection [{coll_name}] 已就绪 (仅 dense)")
                    return

        # 慢速路径: qdrant_client Python SDK
        exc_container = []

        def _worker():
            try:
                self.client.create_collection(**create_kwargs)
            except Exception as e:
                exc_container.append(e)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(5)

        if not worker.is_alive() and not exc_container:
            if self._poll_collection_ready(coll_name, poll_total=5, poll_interval=1):
                logger.info(f"Collection [{coll_name}] 已就绪 (qdrant_client)")
                return

        # 最后一次轮询
        if self._poll_collection_ready(coll_name, poll_total=5, poll_interval=1):
            logger.info(f"Collection [{coll_name}] 已就绪 (后台创建)")
            return

        # 所有路径均失败
        logger.error(
            f"Collection [{coll_name}] 创建失败（所有路径超时）。"
            f"Qdrant 可能存在资源/锁问题。"
            f"建议: 1) 重启 Qdrant 进程 "
            f"2) 检查 Qdrant 日志（磁盘/内存/网络） "
            f"3) 访问 http://{settings.qdrant_host}:{settings.qdrant_port} 确认服务状态 "
            f"4) 当前服务以 [chapter_only] 降级模式运行"
        )

    def _try_create_via_http(self, coll_name: str, require_named_vectors: bool, require_sparse: bool) -> bool:
        """使用 requests 直接发送 PUT 请求创建 collection，绕开 qdrant_client 的额外开销。"""
        try:
            import requests
        except ImportError:
            return False

        host = getattr(settings, "qdrant_host", "localhost")
        port = getattr(settings, "qdrant_port", 6333)
        url = f"http://{host}:{port}/collections/{coll_name}"

        if require_named_vectors:
            payload = {
                "vectors": {
                    self.VECTOR_DENSE: {
                        "size": settings.vector_dim,
                        "distance": "Cosine",
                    },
                },
            }
            if require_sparse:
                payload["sparse_vectors"] = {self.VECTOR_SPARSE: {}}
        else:
            payload = {
                "vectors": {"size": settings.vector_dim, "distance": "Cosine"},
            }

        try:
            # 短超时: Qdrant 若健康应在 5s 内返回
            r = requests.put(url, json=payload, timeout=5)
            logger.debug(f"HTTP PUT {url}: {r.status_code}")
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"HTTP PUT {url} 失败 ({type(e).__name__}: {e})")
            return False

    def _poll_collection_ready(self, coll_name: str, poll_total: int = 60, poll_interval: int = 3) -> bool:
        """轮询检测 collection 是否已经存在且状态为 green。"""
        import time
        t0 = time.time()
        while time.time() - t0 < poll_total:
            try:
                if self.client.collection_exists(coll_name):
                    info = self.client.get_collection(coll_name)
                    if hasattr(info, "status") and str(info.status) != "green":
                        # 还在 yellow/初始化，继续等
                        pass
                    else:
                        return True
            except Exception:
                pass
            time.sleep(poll_interval)
        # 最后再检查一次
        try:
            return self.client.collection_exists(coll_name)
        except Exception:
            return False

    def _ensure_payload_index(self, coll_name: str, field_name: str, field_type: PayloadSchemaType):
        """确保 Collection 上存在指定 payload 字段的索引，否则创建"""
        try:
            info = self.client.get_collection(collection_name=coll_name)
            indexed_fields = set()
            if info.payload_schema:
                for name, schema_info in info.payload_schema.items():
                    indexed_fields.add(name)

            if field_name not in indexed_fields:
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=True,
                    timeout=60,
                )
                logger.info(f"已在 {coll_name} 上创建 payload 索引: {field_name}")
        except Exception as e:
            logger.warning(f"为 {coll_name} 创建索引失败（{field_name}）: {e}")

    # =====================================================================
    #  元数据序列化
    # =====================================================================

    @staticmethod
    def _meta_to_payload(meta: Optional[DocumentMeta]) -> dict:
        """将 DocumentMeta 序列化为 Qdrant payload 字段"""
        if meta is None:
            return {}
        return {
            "doc_id": meta.doc_id,
            "file_name": meta.file_name,
            "file_path": meta.file_path,
            "file_hash": meta.file_hash,
            "file_size": meta.file_size,
            "file_type": meta.file_type or "",
            "doc_version": meta.version,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
            "operator": meta.operator,
            "source": meta.source,
            "doc_status": meta.status,
            "tags": meta.tags,
            "vector_count": meta.vector_count or 0,
        }

    @staticmethod
    def _payload_to_meta(payload: dict) -> Optional[DocumentMeta]:
        """从 Qdrant payload 反序列化为 DocumentMeta"""
        if not payload or "doc_id" not in payload:
            return None
        return DocumentMeta(
            doc_id=payload.get("doc_id", ""),
            file_name=payload.get("file_name", ""),
            file_path=payload.get("file_path", ""),
            file_hash=payload.get("file_hash", ""),
            file_size=payload.get("file_size", 0),
            file_type=payload.get("file_type", ""),
            version=payload.get("doc_version", 1),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            operator=payload.get("operator", "unknown"),
            source=payload.get("source", "unknown"),
            status=payload.get("doc_status", "active"),
            tags=payload.get("tags", []),
            vector_count=payload.get("vector_count", 0),
        )

    @staticmethod
    def _merge_meta_with_system_fields(meta_payload: dict) -> dict:
        """将元数据字段加上系统级索引时间戳"""
        result = {}
        for k, v in meta_payload.items():
            result[k] = v
        result["_indexed_at"] = datetime.now(timezone.utc).isoformat()
        return result

    # =====================================================================
    #  章节存储
    # =====================================================================

    def store_chapters(
        self,
        chapters: list[Chapter],
        embeddings: list[list[float]],
        doc_meta: Optional[DocumentMeta] = None,
    ):
        """存储章节向量（异步批量写入，大批次，显著加速入库）"""
        if not chapters or not embeddings:
            return

        import time
        t0 = time.time()
        meta_payload = self._meta_to_payload(doc_meta)

        # 批量构建 points
        all_points = []
        for i, chapter in enumerate(chapters):
            chapter_id = getattr(chapter, 'chapter_id', None) or str(uuid4())
            if not getattr(chapter, 'chapter_id', None):
                chapter.chapter_id = chapter_id

            chapter_path = getattr(chapter, 'chapter_path', '') or chapter.title
            payload = {
                "type": "chapter",
                "chapter_id": chapter_id,
                "title": chapter.title,
                "chapter_path": chapter_path,
                "level": chapter.level,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "content_snippet": (chapter.content or "")[:1500],
                **self._merge_meta_with_system_fields(meta_payload),
            }
            all_points.append(PointStruct(id=chapter_id, vector=embeddings[i], payload=payload))

        # 分批写入（加大批大小，使用 wait=False 异步，最后一批 wait=True 确保落盘）
        batch_size = 100
        total = len(all_points)
        total_batches = (total + batch_size - 1) // batch_size
        failed = 0
        for batch_idx, start in enumerate(range(0, total, batch_size), 1):
            batch = all_points[start:start + batch_size]
            is_last = (batch_idx == total_batches)
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_CHAPTER,
                    points=batch,
                    wait=is_last,
                    timeout=600,
                )
            except Exception as e:
                msg = str(e).lower()
                # payload 超限/超时 → 自动减半递归
                if ("larger than allowed" in msg or "payload" in msg or
                        "timed out" in msg or "timeout" in msg or
                        "400" in msg or "bad request" in msg or
                        "deadline" in msg or "connect" in msg) and len(batch) > 10:
                    logger.warning(f"章节向量 batch {batch_idx}/{total_batches} 失败({len(batch)}条)，自动减半: {e}")
                    mid = len(batch) // 2
                    try:
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHAPTER,
                            points=batch[:mid],
                            wait=is_last,
                            timeout=600,
                        )
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHAPTER,
                            points=batch[mid:],
                            wait=is_last,
                            timeout=600,
                        )
                    except Exception as e2:
                        logger.warning(f"减半仍失败，逐条写入: {e2}")
                        for p in batch:
                            try:
                                self.client.upsert(
                                    collection_name=self.COLLECTION_CHAPTER,
                                    points=[p],
                                    wait=is_last,
                                    timeout=600,
                                )
                            except Exception as e3:
                                logger.warning(f"单条 upsert 也失败，跳过: {e3}")
                else:
                    logger.warning(f"章节向量 batch {batch_idx}/{total_batches} 写入失败(重试): {e}")
                    try:
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHAPTER,
                            points=batch,
                            wait=True,
                            timeout=600,
                        )
                    except Exception as e2:
                        logger.warning(f"重试仍失败，降级减半: {e2}")
                        mid = len(batch) // 2
                        try:
                            self.client.upsert(
                                collection_name=self.COLLECTION_CHAPTER,
                                points=batch[:mid],
                                wait=is_last,
                                timeout=600,
                            )
                            self.client.upsert(
                                collection_name=self.COLLECTION_CHAPTER,
                                points=batch[mid:],
                                wait=is_last,
                                timeout=600,
                            )
                        except Exception as e3:
                            logger.warning(f"减半仍失败，逐条: {e3}")
                            for p in batch:
                                try:
                                    self.client.upsert(
                                        collection_name=self.COLLECTION_CHAPTER,
                                        points=[p],
                                        wait=is_last,
                                        timeout=600,
                                    )
                                except Exception as e4:
                                    logger.warning(f"单条 upsert 也失败，跳过: {e4}")

        dt = time.time() - t0
        logger.info(
            f"存储 {total} 个章节向量完成 [doc={meta_payload.get('file_name', '?')}], "
            f"耗时 {dt:.1f}s, 失败 {failed} 批"
        )

    # =====================================================================
    #  子块存储（稠密 + 稀疏双向量）
    # =====================================================================

    def store_children(
        self,
        child_chunks: list[ChildChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[tuple[list[int], list[float]]],
        doc_meta: Optional[DocumentMeta] = None,
    ):
        """存储子块向量（稠密 + 稀疏，大批次异步写入，动态分块避免超限）"""
        if not child_chunks or not dense_embeddings:
            return

        import time
        t0 = time.time()
        meta = doc_meta or (child_chunks[0].doc_meta if child_chunks else None)
        meta_payload = self._meta_to_payload(meta)

        # 批量构建 points
        all_points = []
        for i, child in enumerate(child_chunks):
            chunk_meta = self._meta_to_payload(child.doc_meta) if child.doc_meta else meta_payload

            # child payload 简化：只保留检索所需的关键字段
            payload = {
                "type": "child",
                "doc_id": chunk_meta.get("doc_id", ""),
                "file_name": chunk_meta.get("file_name", ""),
                "chapter_id": child.chapter_id,
                "chapter_title": child.chapter_title,
                "chunk_index": child.chunk_index,
                "content_snippet": (child.content or "")[:200],
                "parent_id": child.parent_id,
                "tags": chunk_meta.get("tags", []),
                "operator": chunk_meta.get("operator", ""),
                "source": chunk_meta.get("source", ""),
                "doc_status": chunk_meta.get("doc_status", "active"),
                "_indexed_at": datetime.now(timezone.utc).isoformat(),
            }

            # 构建双向量
            sparse = sparse_embeddings[i] if i < len(sparse_embeddings) else ([], [])
            vectors = {
                self.VECTOR_DENSE: dense_embeddings[i],
                self.VECTOR_SPARSE: SparseVector(indices=sparse[0], values=sparse[1]),
            }

            all_points.append(PointStruct(id=child.chunk_id, vector=vectors, payload=payload))

        # 分批写入（保守批大小，避免 32MB 限制）
        batch_size = 100
        total = len(all_points)
        total_batches = (total + batch_size - 1) // batch_size
        failed = 0

        for batch_idx, start in enumerate(range(0, total, batch_size), 1):
            batch = all_points[start:start + batch_size]
            is_last = (batch_idx == total_batches)
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_CHILD,
                    points=batch,
                    wait=is_last,
                    timeout=600,
                )
            except Exception as e:
                msg = str(e).lower()
                # payload 超限/超时 → 自动减半递归
                if ("larger than allowed" in msg or "payload" in msg or
                        "timed out" in msg or "timeout" in msg or
                        "400" in msg or "bad request" in msg or
                        "deadline" in msg or "connect" in msg) and len(batch) > 10:
                    logger.warning(f"子块向量 batch {batch_idx}/{total_batches} 失败({len(batch)}条)，自动减半: {e}")
                    mid = len(batch) // 2
                    try:
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHILD,
                            points=batch[:mid],
                            wait=is_last,
                            timeout=600,
                        )
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHILD,
                            points=batch[mid:],
                            wait=is_last,
                            timeout=600,
                        )
                    except Exception as e2:
                        logger.warning(f"减半仍失败，改为逐条写入: {e2}")
                        for p in batch:
                            try:
                                self.client.upsert(
                                    collection_name=self.COLLECTION_CHILD,
                                    points=[p],
                                    wait=is_last,
                                    timeout=600,
                                )
                            except Exception as e3:
                                logger.warning(f"单条 upsert 也失败，跳过: {e3}")
                else:
                    logger.warning(f"子块向量 batch {batch_idx}/{total_batches} 写入失败(重试): {e}")
                    try:
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHILD,
                            points=batch,
                            wait=True,
                            timeout=600,
                        )
                    except Exception as e2:
                        logger.warning(f"重试仍失败，降级减半: {e2}")
                        mid = len(batch) // 2
                        try:
                            self.client.upsert(
                                collection_name=self.COLLECTION_CHILD,
                                points=batch[:mid],
                                wait=is_last,
                                timeout=600,
                            )
                            self.client.upsert(
                                collection_name=self.COLLECTION_CHILD,
                                points=batch[mid:],
                                wait=is_last,
                                timeout=600,
                            )
                        except Exception as e3:
                            logger.warning(f"减半仍失败，逐条: {e3}")
                            for p in batch:
                                try:
                                    self.client.upsert(
                                        collection_name=self.COLLECTION_CHILD,
                                        points=[p],
                                        wait=is_last,
                                        timeout=600,
                                    )
                                except Exception as e4:
                                    logger.warning(f"单条 upsert 也失败，跳过: {e4}")

        dt = time.time() - t0
        logger.info(
            f"存储 {total} 个子块向量 (dense+sparse) 完成 "
            f"[doc={meta_payload.get('file_name', '?')}], 耗时 {dt:.1f}s, 失败 {failed} 批, "
            f"平均 {total/max(dt, 0.01):.1f} 条/s"
        )

    def store_children_batch(
        self,
        child_chunks: list[ChildChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[tuple[list[int], list[float]]],
        doc_meta: Optional[DocumentMeta] = None,
        wait: bool = True,
    ):
        """子块向量写入（内部按大小动态分块，避免 payload 超限/索引构建超时）。"""
        if not child_chunks:
            return

        meta = doc_meta or (child_chunks[0].doc_meta if child_chunks else None)
        meta_payload = self._meta_to_payload(meta)

        # 预先构建所有 points
        points = []
        for i, child in enumerate(child_chunks):
            chunk_meta = self._meta_to_payload(child.doc_meta) if child.doc_meta else meta_payload
            payload = {
                "type": "child",
                "doc_id": chunk_meta.get("doc_id", ""),
                "file_name": chunk_meta.get("file_name", ""),
                "chapter_id": child.chapter_id,
                "chapter_title": child.chapter_title,
                "chunk_index": child.chunk_index,
                "content_snippet": (child.content or "")[:200],
                "parent_id": child.parent_id,
                "tags": chunk_meta.get("tags", []),
                "operator": chunk_meta.get("operator", ""),
                "source": chunk_meta.get("source", ""),
                "doc_status": chunk_meta.get("doc_status", "active"),
                "_indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            sparse = sparse_embeddings[i] if i < len(sparse_embeddings) else ([], [])
            vectors = {
                self.VECTOR_DENSE: dense_embeddings[i],
                self.VECTOR_SPARSE: SparseVector(indices=sparse[0], values=sparse[1]),
            }
            points.append(PointStruct(id=child.chunk_id, vector=vectors, payload=payload))

        # 实际 upsert：失败（超限/超时）自动降级（减半 → 1/4 → 逐条）
        def _do_upsert(sub_points: list, sub_wait: bool, depth: int = 0):
            """递归 upsert：超限/超时自动减半，最多 3 级降级后逐条写入。"""
            if depth > 3 or len(sub_points) <= 5:
                # 降级到底：逐条写入，给更长 timeout
                for p in sub_points:
                    try:
                        self.client.upsert(
                            collection_name=self.COLLECTION_CHILD,
                            points=[p],
                            wait=sub_wait,
                            timeout=600,
                        )
                    except Exception as e3:
                        logger.warning(f"单条向量 upsert 也失败，跳过: {e3}")
                return

            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_CHILD,
                    points=sub_points,
                    wait=sub_wait,
                    timeout=600,
                )
                return
            except Exception as e:
                msg = str(e).lower()
                # 超限/超时 → 自动减半重试
                if ("larger than allowed" in msg or "payload" in msg or
                        "timed out" in msg or "timeout" in msg or
                        "400" in msg or "bad request" in msg or
                        "deadline" in msg or "connect" in msg) and len(sub_points) > 10:
                    logger.warning(
                        f"子块向量批({len(sub_points)}条) upsert 失败(第{depth+1}级)，自动减半: {e}"
                    )
                    mid = len(sub_points) // 2
                    _do_upsert(sub_points[:mid], sub_wait, depth + 1)
                    _do_upsert(sub_points[mid:], sub_wait, depth + 1)
                    return

                # 其他错误：重试一次完整批
                logger.warning(f"子块向量批({len(sub_points)}条) upsert 失败(重试): {e}")
                try:
                    self.client.upsert(
                        collection_name=self.COLLECTION_CHILD,
                        points=sub_points,
                        wait=True,
                        timeout=600,
                    )
                except Exception as e2:
                    logger.warning(f"重试仍失败，降级减半: {e2}")
                    mid = len(sub_points) // 2
                    _do_upsert(sub_points[:mid], sub_wait, depth + 1)
                    _do_upsert(sub_points[mid:], sub_wait, depth + 1)

        # 保守分批：默认 safe_chunk = 100（同时避免 payload 超限 + 索引构建超时）
        # 100 条 × 20KB/条 ≈ 2MB << 32MB，且索引构建时间短
        safe_chunk = 100
        if len(points) <= safe_chunk:
            _do_upsert(points, wait)
        else:
            logger.info(f"子块向量 {len(points)} 条，按 {safe_chunk}/批 分批写入")
            for start in range(0, len(points), safe_chunk):
                end = min(start + safe_chunk, len(points))
                sub_points = points[start:end]
                is_last = end == len(points)
                _do_upsert(sub_points, wait and is_last)

    # =====================================================================
    #  版本管理与冲突解决
    # =====================================================================

    def check_doc_exists(self, doc_id: str) -> Optional[dict]:
        """检查文档是否已存在，返回最新版本的元数据"""
        scroll_result, _ = self.client.scroll(
            collection_name=self.COLLECTION_CHAPTER,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=True,
        )
        if not scroll_result:
            return None

        payload = scroll_result[0].payload or {}
        return {
            "doc_id": doc_id,
            "file_name": payload.get("file_name", ""),
            "version": payload.get("doc_version", 1),
            "file_hash": payload.get("file_hash", ""),
            "status": payload.get("doc_status", "active"),
            "operator": payload.get("operator", ""),
            "updated_at": payload.get("updated_at", ""),
        }

    def get_doc_versions(self, doc_id: str) -> list[dict]:
        """获取某文档的历史版本列表"""
        scroll_result, _ = self.client.scroll(
            collection_name=self.COLLECTION_CHAPTER,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=100,
            with_payload=True,
        )
        versions = {}
        for pt in scroll_result:
            v = (pt.payload or {}).get("doc_version", 0)
            if v not in versions:
                versions[v] = {
                    "version": v,
                    "operator": pt.payload.get("operator", ""),
                    "updated_at": pt.payload.get("updated_at", ""),
                    "file_hash": pt.payload.get("file_hash", ""),
                    "points_count": 1,
                }
            else:
                versions[v]["points_count"] += 1
        return sorted(versions.values(), key=lambda x: x["version"], reverse=True)

    def resolve_conflict(
        self,
        doc_id: str,
        new_meta: DocumentMeta,
    ) -> dict:
        """冲突检测与解决"""
        existing = self.check_doc_exists(doc_id)
        if existing is None:
            return {"action": "create", "version": 1, "message": "新文档，将创建索引"}

        strategy = settings.conflict_strategy

        if settings.skip_unchanged and existing["file_hash"] == new_meta.file_hash:
            logger.info(f"文档 [{doc_id}] 内容未变化，跳过索引")
            return {"action": "skip", "version": existing["version"], "message": "内容未变化，已跳过"}

        if strategy == "reject":
            return {
                "action": "reject",
                "version": existing["version"],
                "message": (
                    f"文档已存在 (v{existing['version']}, 操作者: {existing['operator']})。"
                    "当前策略为 reject，请先删除旧版本或修改冲突策略。"
                ),
            }
        elif strategy == "overwrite":
            self._delete_by_doc_id(doc_id)
            new_version = existing["version"] + 1
            return {
                "action": "overwrite",
                "version": new_version,
                "message": f"覆盖 v{existing['version']} -> v{new_version}",
            }
        elif strategy == "keep_both":
            new_version = existing["version"] + 1
            return {
                "action": "keep_both",
                "version": new_version,
                "message": f"保留旧版本 v{existing['version']}，新建 v{new_version}",
            }
        else:
            logger.warning(f"未知冲突策略: {strategy}，默认覆盖")
            self._delete_by_doc_id(doc_id)
            new_version = existing["version"] + 1
            return {
                "action": "overwrite",
                "version": new_version,
                "message": f"未知策略，默认覆盖 v{existing['version']} -> v{new_version}",
            }

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有 Collection 中的相关数据"""
        return self._delete_by_doc_id(doc_id)

    def _delete_by_doc_id(self, doc_id: str) -> int:
        """
        内部实现：按 doc_id 直接删除（走 Qdrant Filter 删除，跳过全集合扫描）。
        关键优化：
          1. 使用 count API（官方轻量计数）替代 scroll 统计
          2. 直接使用 delete + FilterSelector 一次性删除
          3. wait=False 避免长阻塞 + 增加超时时间
          4. 回退方案使用标准 PointsSelector 格式
        """
        from qdrant_client.http.models import PointsSelector
        total = 0
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        for coll_name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            # 1. 使用官方 count API 快速统计（< 50ms，比 scroll 快 10-100 倍）
            count = 0
            try:
                count = self.client.count(
                    collection_name=coll_name,
                    count_filter=doc_filter,
                    timeout=30,
                ).count
            except Exception:
                count = 0

            if count == 0:
                continue

            # 2. 直接按 Filter 删除 —— Qdrant 原生支持，不需要先扫描点 ID
            try:
                # wait=False: 不阻塞等待索引重建完成，立即返回
                self.client.delete(
                    collection_name=coll_name,
                    points_selector=FilterSelector(filter=doc_filter),
                    wait=False,
                    timeout=30,
                )
                total += count
            except Exception as e:
                logger.warning(f"{coll_name} 按 Filter 删除失败（{e}），回退到按点 ID 分批删除")
                # 回退方案：先分页扫描点 ID，再分批删除
                point_ids = []
                offset = None
                try:
                    while True:
                        r, next_offset = self.client.scroll(
                            collection_name=coll_name,
                            scroll_filter=doc_filter,
                            limit=500,
                            with_payload=False,
                            with_vectors=False,
                            timeout=30,
                            offset=offset,
                        )
                        for pt in r:
                            point_ids.append(pt.id)
                        if next_offset is None or len(r) == 0:
                            break
                        offset = next_offset
                except Exception as e2:
                    logger.error(f"{coll_name} scroll 扫描失败: {e2}")

                if point_ids:
                    # 分批删除，每批 500 条
                    for batch_start in range(0, len(point_ids), 500):
                        batch_ids = point_ids[batch_start:batch_start + 500]
                        try:
                            self.client.delete(
                                collection_name=coll_name,
                                points_selector=PointIdsList(points=batch_ids),
                                wait=False,
                                timeout=30,
                            )
                        except Exception as e3:
                            logger.warning(f"{coll_name} 分批删除 {len(batch_ids)} 条失败: {e3}")
                    total += len(point_ids)

        logger.info(f"已删除 doc_id={doc_id} 的所有数据（共 {total} 条）")
        return total

    def archive_doc(self, doc_id: str):
        """将文档标记为 archived 状态（软删除，优化：分页 scroll + 批量 set_payload）"""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        for coll_name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            # 分页 scroll 获取所有点 ID（不读 payload）
            point_ids = []
            offset = None
            while True:
                r, next_offset = self.client.scroll(
                    collection_name=coll_name,
                    scroll_filter=doc_filter,
                    limit=500,
                    with_payload=False,
                    with_vectors=False,
                    timeout=30,
                    offset=offset,
                )
                for pt in r:
                    point_ids.append(pt.id)
                if not next_offset or len(r) == 0:
                    break
                offset = next_offset

            if not point_ids:
                continue

            # 批量 set_payload，每批 500 条（一次 API 调用处理 500 点）
            for batch_start in range(0, len(point_ids), 500):
                batch = point_ids[batch_start:batch_start + 500]
                try:
                    self.client.set_payload(
                        collection_name=coll_name,
                        payload={"doc_status": "archived"},
                        points=batch,
                        wait=False,
                        timeout=30,
                    )
                except Exception as e:
                    logger.warning(f"{coll_name} set_payload 失败: {e}")
        logger.info(f"已将 doc_id={doc_id} 标记为 archived")

    def update_doc_meta(self, doc_id: str, updates: dict) -> bool:
        """更新指定文档的元数据（scroll 查找点 ID，再批量 set_payload）

        Args:
            doc_id: 文档 ID
            updates: 要更新的字段，如 {"vector_count": 120, "doc_status": "active",
                     "updated_at": "2024-01-01..."}

        Returns:
            True 表示更新成功
        """
        if not updates:
            return False
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        total_updated = 0
        for coll_name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            try:
                # 用 scroll+filter 找到该 doc_id 的所有点 ID
                point_ids = []
                offset = None
                while True:
                    scroll_result, next_offset = self.client.scroll(
                        collection_name=coll_name,
                        scroll_filter=doc_filter,
                        limit=1000,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    for pt in scroll_result:
                        point_ids.append(pt.id)
                    if not next_offset:
                        break
                    offset = next_offset

                if not point_ids:
                    logger.debug(f"集合 {coll_name} 中未找到 doc_id={doc_id} 的点，跳过更新")
                    continue

                # 分批 set_payload（每批最多 500 个点）
                batch_size = 500
                for i in range(0, len(point_ids), batch_size):
                    batch = point_ids[i:i + batch_size]
                    self.client.set_payload(
                        collection_name=coll_name,
                        payload=updates,
                        points=batch,
                    )
                total_updated += len(point_ids)
                logger.info(f"集合 {coll_name} 已更新 {len(point_ids)} 个点的元数据")
            except Exception as e:
                logger.warning(f"集合 {coll_name} 更新失败: {e}")

        return total_updated > 0

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档（从 Qdrant payload 读取完整字段）"""
        scroll_result, _ = self.client.scroll(
            collection_name=self.COLLECTION_CHAPTER,
            limit=10000,
            with_payload=True,
        )

        docs = {}
        for pt in scroll_result:
            payload = pt.payload or {}
            doc_id = payload.get("doc_id", "")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": payload.get("file_name", ""),
                    "file_type": payload.get("file_type", ""),
                    "file_size": payload.get("file_size", 0),
                    "vector_count": payload.get("vector_count", 0),
                    "version": payload.get("doc_version", 1),
                    "status": payload.get("doc_status", "active"),
                    "operator": payload.get("operator", ""),
                    "created_at": payload.get("created_at", ""),
                    "updated_at": payload.get("updated_at", ""),
                    "tags": payload.get("tags", []),
                }

        return sorted(docs.values(), key=lambda x: x["updated_at"], reverse=True)

    # =====================================================================
    #  检索操作
    # =====================================================================

    def search_chapters(
        self,
        query_vector: list[float],
        top_k: int = 3,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: str = "active",
    ) -> list[ScoredPoint]:
        """检索最相关的章节，支持 doc_id 和 tags 过滤（collection 不存在时返回空）"""
        if not self._collections_ready.get(self.COLLECTION_CHAPTER, False):
            try:
                if not self.client.collection_exists(self.COLLECTION_CHAPTER):
                    logger.warning(f"Collection [{self.COLLECTION_CHAPTER}] 不存在，章节搜索跳过")
                    return []
            except Exception:
                return []
        # doc_status 处理：兼容旧数据（无字段视为 active）
        must_conds = []
        must_not_conds = []
        if status == "active":
            must_not_conds.append(FieldCondition(key="doc_status", match=MatchValue(value="archived")))
        else:
            must_conds.append(FieldCondition(key="doc_status", match=MatchValue(value=status)))
        if doc_id:
            must_conds.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if tags:
            for tag in tags:
                must_conds.append(FieldCondition(key="tags", match=MatchValue(value=tag)))

        query_filter = Filter(must=must_conds, must_not=must_not_conds)
        try:
            return self.client.query_points(
                collection_name=self.COLLECTION_CHAPTER,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            ).points
        except Exception as e:
            logger.warning(f"章节搜索失败: {e}")
            return []

    def search_children_hybrid(
        self,
        query_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        chapter_ids: Optional[list[str]] = None,
        top_k: int = 20,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: str = "active",
    ) -> list[ScoredPoint]:
        """混合检索子块：稠密 + 稀疏双路检索，RRF 融合（collection 不存在时返回空）"""
        if not self._collections_ready.get(self.COLLECTION_CHILD, False):
            try:
                if not self.client.collection_exists(self.COLLECTION_CHILD):
                    logger.warning(f"Collection [{self.COLLECTION_CHILD}] 不存在，子块混合搜索跳过")
                    return []
            except Exception:
                return []
        # doc_status 处理：兼容旧数据（无字段视为 active）
        must_conds = []
        must_not_conds = []
        if status == "active":
            must_not_conds.append(FieldCondition(key="doc_status", match=MatchValue(value="archived")))
        else:
            must_conds.append(FieldCondition(key="doc_status", match=MatchValue(value=status)))

        if chapter_ids:
            must_conds.append(
                Filter(
                    should=[
                        FieldCondition(key="chapter_id", match=MatchValue(value=ch_id))
                        for ch_id in chapter_ids
                    ]
                )
            )
        if doc_id:
            must_conds.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if tags:
            for tag in tags:
                must_conds.append(FieldCondition(key="tags", match=MatchValue(value=tag)))

        query_filter = Filter(must=must_conds, must_not=must_not_conds)

        try:
            results = self.client.query_points(
                collection_name=self.COLLECTION_CHILD,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using=self.VECTOR_DENSE,
                        filter=query_filter,
                        limit=top_k * 2,
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using=self.VECTOR_SPARSE,
                        filter=query_filter,
                        limit=top_k * 2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points
            return results
        except Exception as e:
            logger.warning(f"子块混合搜索失败: {e}，回退到纯稠密搜索")
            # 回退：仅用稠密向量搜索（绕过可能不工作的 sparse 向量）
            try:
                return self.client.query_points(
                    collection_name=self.COLLECTION_CHILD,
                    query=query_vector,
                    using=self.VECTOR_DENSE,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                ).points
            except Exception as e2:
                logger.warning(f"子块纯稠密搜索也失败: {e2}")
                return []

    def search_children_dense(
        self,
        query_vector: list[float],
        chapter_ids: Optional[list[str]] = None,
        top_k: int = 10,
        doc_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: str = "active",
    ) -> list[ScoredPoint]:
        """仅稠密检索子块（collection 不存在时返回空）"""
        if not self._collections_ready.get(self.COLLECTION_CHILD, False):
            try:
                if not self.client.collection_exists(self.COLLECTION_CHILD):
                    logger.warning(f"Collection [{self.COLLECTION_CHILD}] 不存在，子块稠密搜索跳过")
                    return []
            except Exception:
                return []
        # doc_status 处理：兼容旧数据（无字段视为 active）
        must_conds = []
        must_not_conds = []
        if status == "active":
            must_not_conds.append(FieldCondition(key="doc_status", match=MatchValue(value="archived")))
        else:
            must_conds.append(FieldCondition(key="doc_status", match=MatchValue(value=status)))
        if chapter_ids:
            must_conds.append(
                Filter(
                    should=[
                        FieldCondition(key="chapter_id", match=MatchValue(value=ch_id))
                        for ch_id in chapter_ids
                    ]
                )
            )
        if doc_id:
            must_conds.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if tags:
            for tag in tags:
                must_conds.append(FieldCondition(key="tags", match=MatchValue(value=tag)))

        query_filter = Filter(must=must_conds, must_not=must_not_conds)
        try:
            return self.client.query_points(
                collection_name=self.COLLECTION_CHILD,
                query=query_vector,
                using=self.VECTOR_DENSE,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning(f"子块稠密搜索失败: {e}")
            return []

    def get_children_by_ids(self, child_ids: list[str]) -> list[dict]:
        """根据子块 ID 获取子块内容"""
        if not child_ids:
            return []
        results = self.client.retrieve(
            collection_name=self.COLLECTION_CHILD,
            ids=child_ids,
            with_payload=True,
        )
        return [{"id": r.id, **r.payload} for r in results if r.payload]

    def get_children_by_parent_id(self, parent_id: str) -> list[dict]:
        """根据 parent_id 获取所有子块"""
        scroll_result, _ = self.client.scroll(
            collection_name=self.COLLECTION_CHILD,
            scroll_filter=Filter(
                must=[FieldCondition(key="parent_id", match=MatchValue(value=parent_id))]
            ),
            limit=500,
            with_payload=True,
        )
        return [{"id": pt.id, **pt.payload} for pt in scroll_result if pt.payload]

    # =====================================================================
    #  管理操作
    # =====================================================================

    def clear_all(self):
        """清空所有 Collection"""
        for name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
        logger.info("已清空所有 Collection")
        self._ensure_collections()

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        stats = {}
        for name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            if self.client.collection_exists(name):
                info = self.client.get_collection(name)
                stats[name] = {
                    "points_count": info.points_count,
                    "vectors_count": info.points_count,
                }
        return stats

    def get_stats_by_doc(self, doc_id: str) -> dict:
        """按文档获取存储统计"""
        result = {}
        for coll_name in [self.COLLECTION_CHAPTER, self.COLLECTION_CHILD]:
            count_result = self.client.count(
                collection_name=coll_name,
                count_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            result[coll_name] = count_result.count
        return result