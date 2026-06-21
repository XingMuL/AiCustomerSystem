"""
上传进度全局存储

用于在文档索引过程中，实时追踪各阶段进度，并通过 SSE 推送给前端。
"""

import threading
import time
from typing import Dict, Optional


class UploadProgress:
    """单个文档的上传进度对象"""

    def __init__(self, doc_id: str, file_name: str):
        self.doc_id = doc_id
        self.file_name = file_name
        self.stage = "initializing"           # initializing | parsing | cleaning | chunking | vectorizing | storing | done | error
        self.stage_name = "初始化"
        self.progress = 0                     # 0-100 总体进度
        self.message = ""
        self.error = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.details: Dict[str, int] = {}     # 各阶段进度 { stage_name: progress }

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "details": self.details,
        }

    def set_stage(self, stage: str, stage_name: str, base_progress: int, message: str = ""):
        self.stage = stage
        self.stage_name = stage_name
        self.progress = base_progress
        self.message = message
        self.details[stage_name] = 0

    def update_stage_progress(self, stage_name: str, progress: int, message: str = ""):
        self.details[stage_name] = progress
        self.message = message

    def update_stage_detail(self, stage_name: str, progress: int, message: str = ""):
        """子阶段细粒度更新（别名），用于 pipeline 内部通过 progress_callback 调用的细粒度进度"""
        self.details[stage_name] = progress
        self.message = message

    def mark_done(self, message: str = "完成"):
        self.stage = "done"
        self.stage_name = "已完成"
        self.progress = 100
        self.message = message
        self.end_time = time.time()

    def mark_error(self, error_msg: str):
        self.stage = "error"
        self.stage_name = "失败"
        self.message = error_msg
        self.error = error_msg
        self.end_time = time.time()


class UploadProgressStore:
    """全局上传进度存储（线程安全）"""

    def __init__(self):
        self._progresses: Dict[str, UploadProgress] = {}
        self._lock = threading.Lock()
        self._gc_ttl = 3600  # 完成/失败后保留 1 小时

    def create(self, doc_id: str, file_name: str) -> UploadProgress:
        with self._lock:
            p = UploadProgress(doc_id, file_name)
            self._progresses[doc_id] = p
            return p

    def get(self, doc_id: str) -> Optional[UploadProgress]:
        with self._lock:
            return self._progresses.get(doc_id)

    def get_all(self) -> Dict[str, UploadProgress]:
        with self._lock:
            return dict(self._progresses)

    def cleanup(self):
        """清理过期记录"""
        now = time.time()
        with self._lock:
            expired = []
            for doc_id, p in self._progresses.items():
                if p.end_time and (now - p.end_time) > self._gc_ttl:
                    expired.append(doc_id)
            for doc_id in expired:
                del self._progresses[doc_id]


# 全局单例
_upload_progress_store = UploadProgressStore()


def get_progress_store() -> UploadProgressStore:
    return _upload_progress_store
