"""
向量化嵌入模块：将文本块转为向量，存入向量数据库。

使用的模型：BAAI/bge-large-zh-v1.5（中文优化）
"""

from typing import Optional
import numpy as np

# ========== PyTorch 2.6 兼容性补丁 + 文件完整性校验 ==========
# 问题 1: PyTorch 2.6 将 torch.load 的 weights_only 默认改为 True
#   sentence-transformers 2.2.2 保存的模型文件包含非张量对象
#   导致 UnpicklingError: Unsupported operand 118
# 问题 2: 文件上传时可能被损坏（文本模式 vs 二进制模式）
#   导致 invalid load key, 'v' 等 pickle 解析失败
#
# 修复策略：
#   1. 无条件 patch torch.load → weights_only=False
#   2. patch transformers.modeling_utils.load_state_dict → 也强制 False
#   3. 在模型加载前做 ZIP 格式校验，失败时给出诊断提示
#   4. 提供多策略回退（直接 torch.load → safetensors → HuggingFace 名称）
try:
    import torch
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    # patch transformers 的内部引用（如果已经导入）
    try:
        import transformers.modeling_utils as _tf_mu
        if hasattr(_tf_mu, "torch") and hasattr(_tf_mu.torch, "load"):
            _tf_mu.torch.load = _patched_torch_load
    except Exception:
        pass

    from loguru import logger as _patch_logger
    _patch_logger.info("[PyTorch 兼容] torch.load 已补丁：weights_only 强制为 False")
except Exception as _patch_err:
    from loguru import logger as _patch_logger2
    _patch_logger2.warning(f"[PyTorch 兼容] torch.load 补丁失败: {_patch_err}")
# ========== 兼容性补丁结束 ==========

# ========== 模型文件完整性校验 ==========
def _validate_model_file(filepath: str) -> tuple:
    """校验 PyTorch 模型文件的 ZIP 格式完整性。
    返回 (ok, message)"""
    import os
    import zipfile

    if not os.path.exists(filepath):
        return False, f"文件不存在: {filepath}"

    size = os.path.getsize(filepath)
    if size < 1024:
        return False, f"文件过小 ({size} bytes)，疑似损坏"

    # 检查 ZIP magic
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
    except Exception as e:
        return False, f"无法读取文件头: {e}"

    if magic[:2] != b"PK":
        hex_header = " ".join(f"{b:02x}" for b in magic[:8])
        return False, (
            f"不是有效的 PyTorch 模型文件。首字节: {hex_header}。"
            f"可能原因：1) 上传时使用了文本模式而非二进制模式；"
            f"2) 文件传输中断导致不完整；3) 文件系统错误。"
            f"建议：重新以二进制模式上传 pytorch_model.bin，"
            f"或使用 HuggingFace Hub 自动下载 (model_name='BAAI/bge-large-zh-v1.5')"
        )

    # 尝试打开 ZIP
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            names = zf.namelist()
            # 检查必需的内部文件
            has_data_pkl = any("data.pkl" in n for n in names)
            has_data_dir = any(n.startswith("data/") for n in names)
            if not (has_data_pkl or has_data_dir):
                return False, (
                    f"ZIP 容器有效但缺少 PyTorch 序列化数据。"
                    f"ZIP 内容: {names[:10]}。这不是有效的 .bin 模型文件。"
                )
            # 快速 CRC 校验
            bad = zf.testzip()
            if bad is not None:
                return False, (
                    f"ZIP CRC 校验失败（文件损坏）。首个损坏条目: {bad}。"
                    f"建议：重新以二进制模式上传 pytorch_model.bin"
                )
    except zipfile.BadZipFile as e:
        return False, f"ZIP 文件损坏 (BadZipFile: {e})。建议重新上传。"
    except Exception as e:
        return False, f"ZIP 解析异常: {e}"

    return True, f"文件完整性校验通过 ({size/1024/1024:.1f} MB)"


def _validate_model_dir(model_path: str) -> tuple:
    """校验整个模型目录。返回 (ok, detailed_message)"""
    import os

    bin_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_file):
        ok, msg = _validate_model_file(bin_file)
        return ok, f"[pytorch_model.bin] {msg}"

    safetensors_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_file):
        return True, f"找到 safetensors 格式模型 ({safetensors_file})"

    shard_files = [
        f for f in os.listdir(model_path) if f.startswith("pytorch_model-") and f.endswith(".bin")
    ] if os.path.exists(model_path) else []
    if shard_files:
        return True, f"找到分片模型文件 ({len(shard_files)} 个分片)"

    return False, (
        f"模型目录中未找到 pytorch_model.bin 或 model.safetensors。"
        f"目录内容: {os.listdir(model_path)[:20] if os.path.exists(model_path) else '目录不存在'}"
    )
# ========== 完整性校验结束 ==========

from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.config import settings


class Embedder:
    """文本向量化器"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model: Optional[SentenceTransformer] = None
        self._is_bge = "bge" in self.model_name.lower()
        self._device: Optional[str] = None

    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型（带文件完整性校验 + 多策略回退）"""
        if self._model is None:
            device = self._resolve_device()
            logger.info(f"加载嵌入模型: {self.model_name}, device={device}")

            # 策略 1: 如果是本地路径，先做文件完整性校验
            import os
            load_targets = []
            if os.path.isdir(self.model_name):
                ok, msg = _validate_model_dir(self.model_name)
                if ok:
                    logger.info(f"[模型校验] {msg}")
                    load_targets.append(self.model_name)
                else:
                    logger.warning(f"[模型校验] {msg}")
                    # 本地文件损坏，尝试用 HuggingFace 名称回退
                    hf_name = self._infer_hf_name(self.model_name)
                    if hf_name:
                        logger.info(f"[模型回退] 尝试从 HuggingFace 加载: {hf_name}")
                        load_targets.append(hf_name)
            else:
                # 非本地路径（如 HuggingFace 名称），直接尝试
                load_targets.append(self.model_name)

            # 策略 2: 依次尝试加载，记录每次失败原因
            last_err = None
            for target in load_targets:
                try:
                    self._model = SentenceTransformer(target, device=device)
                    logger.info(f"[模型加载] 成功从 {target} 加载")
                    break
                except Exception as e:
                    last_err = e
                    logger.error(
                        f"[模型加载] 从 {target} 加载失败: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue

            if self._model is None:
                raise RuntimeError(
                    f"嵌入模型加载失败（所有策略均失败）。"
                    f"最后一次错误: {type(last_err).__name__}: {last_err}. "
                    f"如为本地文件损坏，建议：1) 以二进制模式重新上传 pytorch_model.bin；"
                    f"2) 或修改配置使用 HuggingFace 名称（如 'BAAI/bge-large-zh-v1.5'）"
                )

        return self._model

    @staticmethod
    def _infer_hf_name(local_path: str) -> Optional[str]:
        """从本地路径推断可能的 HuggingFace 模型名称"""
        import os
        basename = os.path.basename(local_path.rstrip("/\\"))
        known_map = {
            "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
            "bge-large-zh": "BAAI/bge-large-zh",
            "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
        }
        for key, hf_name in known_map.items():
            if key in basename.lower():
                return hf_name
        return None

    @property
    def device(self) -> str:
        """返回当前使用的设备（如 "cpu" / "cuda:0"）"""
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> str:
        """
        根据配置 + 实际 GPU 检测，返回最终使用的设备。
        优先级：
            1. use_gpu = True   → 强制使用 GPU（cuda:<gpu_device_id>），不可用则回退 CPU
            2. use_gpu = False  → 强制使用 CPU
            3. use_gpu = None   → 自动检测：有 GPU 用 GPU，否则 CPU
        """
        use_gpu_cfg = getattr(settings, "use_gpu", None)
        gpu_id = getattr(settings, "gpu_device_id", 0)

        # 实际 GPU 检测（仅当可能用到 GPU 时才检测，避免不必要的 torch import）
        gpu_available = False
        if use_gpu_cfg in (True, None):
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    num_gpus = torch.cuda.device_count()
                    logger.info(f"检测到 {num_gpus} 个 CUDA 设备")
            except ImportError:
                gpu_available = False

        # 按配置决策
        if use_gpu_cfg is True:
            if gpu_available:
                device = f"cuda:{gpu_id}"
                logger.info(f"[GPU 加速] 已启用：device={device}（use_gpu=True 强制模式）")
                self._device = device
                return device
            else:
                logger.warning(f"[GPU 加速] 配置 use_gpu=True 但未检测到可用 GPU，回退到 CPU")
                self._device = "cpu"
                return "cpu"

        elif use_gpu_cfg is False:
            logger.info(f"[GPU 加速] 已禁用（use_gpu=False），使用 CPU")
            self._device = "cpu"
            return "cpu"

        else:
            # None → 自动检测模式
            if gpu_available:
                device = f"cuda:{gpu_id}"
                logger.info(f"[GPU 加速] 自动检测：已启用 GPU device={device}")
                self._device = device
                return device
            else:
                logger.info(f"[GPU 加速] 自动检测：未检测到 GPU，使用 CPU")
                self._device = "cpu"
                return "cpu"

    @staticmethod
    def _has_gpu() -> bool:
        """兼容旧接口：根据配置 + 检测结果返回布尔值"""
        try:
            from backend.config import settings as _s
            use_gpu_cfg = getattr(_s, "use_gpu", None)
        except Exception:
            use_gpu_cfg = None

        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        if use_gpu_cfg is True:
            return gpu_available
        if use_gpu_cfg is False:
            return False
        return gpu_available

    def close(self):
        """释放模型资源"""
        if self._model is not None:
            try:
                del self._model
                self._model = None
            except Exception:
                pass

    def embed_text(self, text: str) -> list[float]:
        """对单条文本进行向量化"""
        if not text or not text.strip():
            return [0.0] * settings.vector_dim

        # BGE 模型需要添加查询指令前缀
        if self._is_bge:
            text = f"为这个句子生成表示以用于检索相关文章：{text}"

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """批量文本向量化（优化版：更大批 + 异步进度条）"""
        if not texts:
            return []

        if self._is_bge:
            texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]

        logger.info(f"批量向量化 {len(texts)} 条文本 (batch_size={batch_size})...")

        # GPU 加速：优先使用大批次，同时关闭 show_progress_bar (True 则为了观察进度）
        # sentence-transformers 在 batch_size 越大吞吐越高，但注意显存
        import time
        t0 = time.time()

        # 动态调整批大小：根据文本长度较短时加大批次，小文本时使用更大批次
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        dt = time.time() - t0
        logger.info(f"批量向量化完成: {len(texts)} 条, 耗时 {dt:.1f}s, "
                     f"平均 {len(texts)/max(dt, 0.01):.1f} 条/s")
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """对查询文本进行向量化（使用查询专用前缀）"""
        if not query or not query.strip():
            return [0.0] * settings.vector_dim

        if self._is_bge:
            query = f"为这个句子生成表示以用于检索相关文章：{query}"

        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


# =====================================================================
#  多进程 Embedding Worker（模块级函数，供 multiprocessing 使用）
# =====================================================================

_worker_model: Optional[SentenceTransformer] = None
_worker_is_bge: bool = False


def _resolve_device_from_config() -> str:
    """模块级：从配置解析最终 device（供多进程 worker 使用）"""
    try:
        from backend.config import settings as _s
        use_gpu_cfg = getattr(_s, "use_gpu", None)
        gpu_id = getattr(_s, "gpu_device_id", 0)
    except Exception:
        use_gpu_cfg = None
        gpu_id = 0

    gpu_available = False
    if use_gpu_cfg in (True, None):
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

    if use_gpu_cfg is True:
        return f"cuda:{gpu_id}" if gpu_available else "cpu"
    elif use_gpu_cfg is False:
        return "cpu"
    else:
        return f"cuda:{gpu_id}" if gpu_available else "cpu"


def _init_embedding_worker(model_name: str, device: Optional[str] = None):
    """多进程 worker 初始化：每个子进程加载自己的模型副本
    设计原则：任何异常都必须被捕获，不得让进程崩溃导致进程池连锁失败
    """
    global _worker_model, _worker_is_bge
    import os

    try:
        # 子进程中限制 OpenMP/BLAS 线程数，避免多进程争抢 CPU
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")

        # 未指定 device 时，自动从配置解析（但 worker 模式下强制 CPU，避免多进程 GPU 冲突）
        if device is None:
            device = "cpu"
        elif device.startswith("cuda"):
            logger.warning(f"[Worker] 检测到多进程 GPU 模式，强制降级为 CPU 以避免多进程 GPU 冲突")
            device = "cpu"

        # 同主进程：先做文件完整性校验 + 多策略回退
        load_targets = []
        if os.path.isdir(model_name):
            ok, msg = _validate_model_dir(model_name)
            if ok:
                logger.info(f"[Worker] 模型校验: {msg}")
                load_targets.append(model_name)
            else:
                logger.warning(f"[Worker] 模型校验: {msg}")
                hf_name = Embedder._infer_hf_name(model_name)
                if hf_name:
                    logger.info(f"[Worker] 回退到 HuggingFace: {hf_name}")
                    load_targets.append(hf_name)
        if not load_targets:
            load_targets.append(model_name)

        for target in load_targets:
            try:
                _worker_model = SentenceTransformer(target, device=device)
                _worker_is_bge = "bge" in target.lower()
                logger.info(f"[Worker] 子进程模型加载完成: {target}, device={device}")
                return  # 正常退出
            except Exception as e:
                logger.error(f"[Worker] {target} 加载失败: {type(e).__name__}: {e}")
                continue

        # 所有策略都失败 → 不抛异常，设置标志位
        logger.error(
            "[Worker] 嵌入模型加载全部失败。"
            "可能原因: 1) pytorch_model.bin 是 LFS 指针文件（未安装 git-lfs）; "
            "2) 服务器内存不足; 3) 模型文件损坏"
        )
        _worker_model = "__FAILED__"  # 特殊标记，表示初始化失败（但进程存活）
        _worker_is_bge = False

    except Exception as top_level_err:
        # 最外层防御：任何异常都不能让进程崩溃
        logger.exception(f"[Worker] 顶层异常: {top_level_err}")
        _worker_model = "__FAILED__"
        _worker_is_bge = False


def _embed_batch_worker(texts: list) -> list:
    """多进程 worker：接收一批文本，返回向量列表
    关键：检测 _worker_model 是否有效，返回清晰错误，不得崩溃
    """
    global _worker_model, _worker_is_bge

    if _worker_model is None:
        raise RuntimeError("Embedding worker 未初始化")
    if _worker_model == "__FAILED__":
        raise RuntimeError(
            "Worker 模型加载失败。请检查: "
            "1) pytorch_model.bin 是否为 LFS 指针文件; "
            "2) 是否有足够内存 (每个 worker 约需 3GB); "
            "3) 使用 python diagnose_model_file.py 诊断"
        )

    try:
        if _worker_is_bge:
            texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]

        embeddings = _worker_model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"[Worker] 向量化失败: {type(e).__name__}: {e}")
        raise


def _calc_safe_worker_count(model_path: str, max_workers: int) -> int:
    """根据可用内存估算安全的 worker 数量
    bge-large-zh-v1.5 每个 worker 约需 2.5GB 内存
    """
    import os
    try:
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024 ** 3)  # GB
        except Exception:
            # psutil 不可用时，按保守估算（只开 2 个 worker）
            logger.warning("[Worker] psutil 不可用，默认使用 2 个 worker")
            return min(max_workers, 2)

        # 保留 4GB 给操作系统和其他组件
        usable = max(0, available - 4.0)
        per_worker = 2.5  # GB
        safe_count = max(1, int(usable / per_worker))

        final = min(safe_count, max_workers)
        logger.info(
            f"[Worker] 内存估算: 可用 {available:.1f}GB, "
            f"安全 worker={safe_count}, 配置={max_workers}, "
            f"最终={final}"
        )
        return final
    except Exception as e:
        logger.warning(f"[Worker] worker 估算异常: {e}, 使用默认 {min(max_workers, 2)}")
        return min(max_workers, 2)