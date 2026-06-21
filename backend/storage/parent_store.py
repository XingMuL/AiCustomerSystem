"""
父块原文存储（SQLite）—— 解耦父块原文与向量库。

设计：
  父块携带完整原文（2k-4k token），不适合存入 Qdrant payload（有大小限制），
  因此单独存入 SQLite，检索时通过 parent_id 拉取完整原文。
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from backend.chunking.parent_child_splitter import ParentChunk


class ParentStore:
    """SQLite 父块存储"""

    DB_DIR = Path(__file__).parent.parent.parent / "data"
    DB_PATH = DB_DIR / "parent_chunks.db"

    def __init__(self):
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    chapter_id TEXT NOT NULL,
                    chapter_title TEXT NOT NULL DEFAULT '',
                    doc_id TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    child_ids TEXT NOT NULL DEFAULT '[]',
                    child_indices TEXT NOT NULL DEFAULT '[]',
                    child_count INTEGER NOT NULL DEFAULT 0,
                    level INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_doc_id 
                ON parent_chunks(doc_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_chapter_id 
                ON parent_chunks(chapter_id)
            """)
            # 子块内容表：存储完整子块内容，Qdrant 中只存预览
            conn.execute("""
                CREATE TABLE IF NOT EXISTS child_chunks (
                    child_id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    chapter_title TEXT NOT NULL DEFAULT '',
                    doc_id TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_child_parent_id 
                ON child_chunks(parent_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_child_doc_id 
                ON child_chunks(doc_id)
            """)
            conn.commit()
        logger.info(f"父块+子块存储初始化完成: {self.DB_PATH}")

    def store_parents(self, parent_chunks: list[ParentChunk], doc_id: str, file_name: str):
        """批量存储父块"""
        if not parent_chunks:
            return

        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            rows = []
            for p in parent_chunks:
                rows.append((
                    p.parent_id,
                    p.chapter_id,
                    p.chapter_title,
                    doc_id,
                    file_name,
                    p.content_snippet,  # 完整拼接内容
                    json.dumps(p.child_ids, ensure_ascii=False),
                    json.dumps(p.metadata.get("child_indices", []), ensure_ascii=False),
                    p.metadata.get("child_count", len(p.child_ids)),
                    p.metadata.get("level", 1),
                    now,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO parent_chunks 
                (parent_id, chapter_id, chapter_title, doc_id, file_name, 
                 content, child_ids, child_indices, child_count, level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        logger.info(f"存储 {len(parent_chunks)} 个父块完成 [doc={file_name}]")

    def get_by_parent_ids(self, parent_ids: list[str]) -> list[dict]:
        """根据 parent_id 列表获取父块"""
        if not parent_ids:
            return []

        placeholders = ','.join('?' for _ in parent_ids)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM parent_chunks WHERE parent_id IN ({placeholders})",
                parent_ids,
            ).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_sibling_parents(self, parent_ids: list[str]) -> list[dict]:
        """
        获取已拉取父块的「兄弟父块」：同一章节下被拆分到不同父块中的其他部分

        场景：父块子块拆分时，一个章节可能被切分为多个父块（如"第二章 商品分类详解"
        包含 8 个子分类，被切分为 2-3 个父块）。当检索命中其中一个父块的子块时，
        需要联动拉取同一章节的其他父块，确保概括性问题（如"有多少个分类"）能拿到完整信息。

        策略：根据 chapter_title 前缀匹配（去掉最后一个 " · " 后缀后的前缀）
        """
        if not parent_ids:
            return []

        # 先获取已拉取的父块，提取 chapter_title 前缀
        direct_parents = self.get_by_parent_ids(parent_ids)
        if not direct_parents:
            return []

        prefixes = set()
        for p in direct_parents:
            title = p.get("chapter_title", "")
            # 提取章节前缀：取前两级（文档 + 章节），跳过只有一级的标题
            # 例如 "fTaoBao ... · 第二章 商品分类详解 · 2.1 手机数码类"
            # → "fTaoBao ... · 第二章 商品分类详解"
            # 对于 "fTaoBao ... · 商品 003" 这种只有一级分隔的，跳过
            parts = title.split(" · ")
            if len(parts) >= 3:
                prefix = " · ".join(parts[:2])  # 只取前两级
                if prefix:
                    prefixes.add(prefix)

        if not prefixes:
            return []

        # 查询所有以同一前缀开头的父块（排除已拉取的）
        direct_ids = set(parent_ids)
        with self._get_conn() as conn:
            all_rows = conn.execute("SELECT * FROM parent_chunks").fetchall()

        siblings = []
        for row in all_rows:
            row_dict = self._row_to_dict(row)
            if row_dict["parent_id"] in direct_ids:
                continue
            title = row_dict.get("chapter_title", "")
            for prefix in prefixes:
                if title.startswith(prefix):
                    siblings.append(row_dict)
                    break

        return siblings

    def get_by_child_ids(self, child_ids: list[str]) -> list[dict]:
        """根据子块 ID 反查所属父块"""
        if not child_ids:
            return []

        with self._get_conn() as conn:
            all_rows = conn.execute("SELECT * FROM parent_chunks").fetchall()

        results = []
        seen = set()
        for row in all_rows:
            row_dict = self._row_to_dict(row)
            stored_child_ids = json.loads(row_dict["child_ids"])
            for cid in child_ids:
                if cid in stored_child_ids and row_dict["parent_id"] not in seen:
                    seen.add(row_dict["parent_id"])
                    results.append(row_dict)
                    break

        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除某文档的所有父块"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()
            count = cursor.rowcount
            if count:
                logger.info(f"已删除 doc_id={doc_id} 的 {count} 个父块")
            return count

    def get_doc_parents(self, doc_id: str) -> list[dict]:
        """获取某文档的所有父块（按章节排序）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM parent_chunks WHERE doc_id = ? ORDER BY chapter_id, created_at",
                (doc_id,),
            ).fetchall()

        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "parent_id": row["parent_id"],
            "chapter_id": row["chapter_id"],
            "chapter_title": row["chapter_title"],
            "doc_id": row["doc_id"],
            "file_name": row["file_name"],
            "content": row["content"],
            "child_ids": json.loads(row["child_ids"]),
            "child_indices": json.loads(row["child_indices"]),
            "child_count": row["child_count"],
            "level": row["level"],
            "created_at": row["created_at"],
        }

    # ============================================================
    #  子块内容存储（完整内容存在 SQLite，Qdrant 只存向量+预览）
    # ============================================================

    def store_children(self, child_chunks: list, doc_id: str, file_name: str):
        """批量存储子块完整内容"""
        if not child_chunks:
            return

        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            rows = []
            for c in child_chunks:
                meta = c.metadata
                rows.append((
                    c.chunk_id,
                    c.parent_id or meta.get("parent_id", ""),
                    c.chapter_id or meta.get("chapter_id", ""),
                    c.chapter_title or meta.get("chapter_title", ""),
                    doc_id,
                    file_name,
                    c.content,
                    c.chunk_index,
                    now,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO child_chunks 
                (child_id, parent_id, chapter_id, chapter_title, doc_id, file_name, 
                 content, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        logger.info(f"存储 {len(child_chunks)} 个子块内容完成 [doc={file_name}]")

    def get_children_by_ids(self, child_ids: list[str]) -> dict[str, dict]:
        """根据子块 ID 批量获取完整内容（返回 {child_id: content_dict}"""
        if not child_ids:
            return {}

        placeholders = ','.join('?' for _ in child_ids)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM child_chunks WHERE child_id IN ({placeholders})",
                child_ids,
            ).fetchall()

        results = {}
        for row in rows:
            results[row["child_id"]] = {
                "child_id": row["child_id"],
                "parent_id": row["parent_id"],
                "chapter_id": row["chapter_id"],
                "chapter_title": row["chapter_title"],
                "doc_id": row["doc_id"],
                "file_name": row["file_name"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "created_at": row["created_at"],
            }
        return results

    def delete_children_by_doc_id(self, doc_id: str) -> int:
        """删除某文档的所有子块内容"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM child_chunks WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()
            count = cursor.rowcount
            if count:
                logger.info(f"已删除 doc_id={doc_id} 的 {count} 个子块内容")
            return count