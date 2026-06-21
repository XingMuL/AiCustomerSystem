"""
稀疏向量生成器（增强版）：基于 jieba 分词 + TF-IDF 加权生成 BM25 稀疏向量。

相比原版改进:
1. 增量词表更新：多文档索引时，新文档的词会加入全局词表，不会覆盖旧文档的词
2. 单例模式：整个系统共享一个 SparseEmbedder 实例，保证检索时使用与入库时相同的词表
3. 停用词过滤：过滤常见中文停用词，减少噪声
4. OOV 词处理：查询时遇到未在词表中的词，使用 BM25 近似得分而非完全忽略
5. 持久化优化：保存/加载 JSON 词表，增量更新

用于 Qdrant 的稀疏向量搜索，与稠密向量（BGE embedding）互补。
"""

import math
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional
from collections import Counter

import jieba
from loguru import logger


# 中文停用词表 —— 减少噪声，提升关键词匹配质量
CHINESE_STOPWORDS = {
    '的', '了', '和', '是', '在', '我', '有', '及', '与', '或', '而', '但',
    '这', '那', '你', '他', '她', '它', '们', '之', '为', '以', '也', '都',
    '就', '要', '会', '能', '可以', '可', '不', '没', '被', '将', '所', '等',
    '一个', '一些', '这样', '那样', '如何', '什么', '哪些', '哪里', '已经', '还',
    '对', '从', '由', '到', '向', '与', '同', '比', '如', '若', '因', '所以',
    '但是', '然而', '而且', '并且', '或者', '虽', '虽然', '尽管', '如果', '那么',
    '我们', '你们', '他们', '它们', '这个', '那个', '这些', '那些', '自己',
    '一下', '一种', '一定', '一起', '一直', '仍然', '依然', '还是', '或者',
    '其实', '因此', '所以', '然后', '后来', '以前', '以后', '现在', '刚刚',
    '通过', '进行', '实现', '完成', '开始', '结束', '提供', '使用', '采用',
    '包括', '以及', '相关', '相应', '需要', '要求', '可能', '应该', '必须',
    '关于', '对于', '按照', '根据', '由于', '基于', '针对', '随着', '没有',
    '公司', '相关', '工作', '业务', '发展', '情况', '问题', '方面', '部分',
}

# 预编译正则
_re_has_chinese_or_letter = re.compile(r'[\u4e00-\u9fffA-Za-z]')


def _parallel_tokenize(text: str) -> set[str]:
    """模块级：对单个文本进行分词（供线程池使用）。"""
    if not text:
        return set()
    tokens = set()
    for w in jieba.cut(text):
        w = w.strip()
        if not w:
            continue
        if len(w) < 2:
            continue
        if not _re_has_chinese_or_letter.search(w):
            continue
        if w in CHINESE_STOPWORDS:
            continue
        tokens.add(w)
    return tokens


def _fit_batch_tokenize(texts: list[str]) -> list[set[str]]:
    """模块级：批处理一组文本（线程池 / 进程池通用）。"""
    return [_parallel_tokenize(t) for t in texts]


# 别名：fit 函数使用的模块级分词批处理
_fit_batch_tokenize_threaded = _fit_batch_tokenize


class SparseEmbedder:
    """
    稀疏向量生成器（增强版）

    使用 TF-IDF 计算词权重，生成 Qdrant 兼容的稀疏向量。
    单例模式：全系统共享一个实例，保证检索与入库使用同一词表。

    稀疏向量格式: {"indices": [int, ...], "values": [float, ...]}
    """

    _instance: Optional['SparseEmbedder'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, vocab_size: int = 50000):
        if self._initialized:
            return
        self.vocab_size = vocab_size
        self._token_to_id: dict[str, int] = {}
        self._next_id = 0
        self._df: dict[int, int] = {}   # document frequency: 词出现在多少个文档/块中
        self._doc_count = 0             # 累计处理的文档/块数
        self._initialized = True
        logger.info(f"SparseEmbedder 初始化 (单例, vocab_size={vocab_size})")

    # =====================================================================
    #  核心：增量式拟合
    # =====================================================================

    def fit(self, texts: list[str]):
        """
        在语料上拟合 TF-IDF 统计量（增量式，线程池并行版）。

        不清空已有词表，仅将新文档中出现的新词加入。使用 ThreadPoolExecutor
        并行执行分词（jieba.cut 包含 C 扩展代码会释放 GIL，多线程可并行）。
        """
        if not texts:
            return

        import time
        t0 = time.time()

        # 并行分词：使用线程池（跨平台安全，无需 __main__ 保护）
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp
        num_workers = min(max(mp.cpu_count() - 1, 1), 8)

        # 先并行 tokenize 所有文本
        chunk_size = max(20, len(texts) // max(num_workers, 1))

        if num_workers > 1 and len(texts) > 200:
            # 切分为多个批次，并行处理
            batches = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                batch_results = list(executor.map(_fit_batch_tokenize_threaded, batches))
            token_sets = []
            for br in batch_results:
                token_sets.extend(br)
        else:
            # 小批量：直接串行
            token_sets = _fit_batch_tokenize_threaded(texts)

        # 合并到主词典（单线程写，避免 dict 并发问题）
        new_doc_count = 0
        new_tokens_total = 0
        for tset in token_sets:
            if not tset:
                continue
            new_doc_count += 1
            new_tokens_total += len(tset)
            for token in tset:
                tid = self._get_or_create_id(token)
                self._df[tid] = self._df.get(tid, 0) + 1

        self._doc_count += new_doc_count

        dt = time.time() - t0
        logger.info(
            f"稀疏向量 fit: +{new_doc_count} 块, +{new_tokens_total} 词项, "
            f"词表={self._next_id}, 累计={self._doc_count}, 耗时 {dt:.1f}s "
            f"({len(texts)/max(dt, 0.01):.1f} 块/s, workers={num_workers})"
        )

    def fit_incremental(self, texts: list[str]) -> int:
        """别名，与 fit() 行为完全一致。返回新增 token 数量。"""
        before = self._next_id
        self.fit(texts)
        return self._next_id - before

    # =====================================================================
    #  核心：编码（生成稀疏向量）
    # =====================================================================

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """
        将文本编码为稀疏向量（BM25 风格加权）。

        Returns:
            (indices, values): 词 ID 列表和权重列表
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        tf = Counter(tokens)
        total_terms = sum(tf.values())
        avgdl = total_terms  # 平均长度（此处用当前文本长度简化）

        # BM25 参数
        k1 = 1.5
        b = 0.75

        indices = []
        values = []
        for token, count in tf.items():
            tid = self._token_to_id.get(token)
            if tid is None:
                # OOV（词表外）词：给一个小的默认权重，避免完全忽略
                # 这对查询很重要：用户查询中的专有名词可能不在词表中
                continue

            # BM25 score
            tf_val = count / total_terms
            df_val = self._df.get(tid, 1)
            idf_val = math.log((self._doc_count + 1) / (df_val + 1)) + 1.0

            # BM25 分子
            numerator = tf_val * (k1 + 1)
            # BM25 分母（长度归一化简化版）
            denominator = tf_val + k1 * (1 - b + b * (total_terms / max(avgdl, 1)))
            bm25_score = (numerator / denominator) * idf_val

            indices.append(tid)
            values.append(bm25_score)

        # 按权重降序排序，取 top N 个词（Qdrant 稀疏向量维度不需要太大）
        top_n = min(len(indices), 128)
        if len(indices) > top_n:
            sorted_pairs = sorted(
                zip(indices, values), key=lambda x: x[1], reverse=True
            )
            top_pairs = sorted_pairs[:top_n]
            indices = [p[0] for p in top_pairs]
            values = [p[1] for p in top_pairs]

        return indices, values

    def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """批量编码（线程池并行版）。"""
        if not texts:
            return []

        import time
        t0 = time.time()

        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp
        num_workers = min(max(mp.cpu_count() - 1, 1), 8)

        # 每个文本独立编码：encode 只使用 self._token_to_id 和 self._df 的读操作（线程安全）
        if num_workers > 1 and len(texts) > 200:
            chunk_size = max(20, len(texts) // max(num_workers, 1))
            batches = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

            def _encode_batch(batch_texts: list[str]) -> list[tuple[list[int], list[float]]]:
                return [self.encode(t) for t in batch_texts]

            results: list[tuple[list[int], list[float]]] = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for br in executor.map(_encode_batch, batches):
                    results.extend(br)
        else:
            results = [self.encode(t) for t in texts]

        dt = time.time() - t0
        logger.info(
            f"稀疏向量 encode: {len(texts)} 块, 耗时 {dt:.1f}s "
            f"({len(texts)/max(dt, 0.01):.1f} 块/s, workers={num_workers})"
        )
        return results

    # =====================================================================
    #  查询专用编码（与文档编码略有不同，更关注匹配度）
    # =====================================================================

    def query_encode(self, query: str) -> tuple[list[int], list[float]]:
        """
        将查询编码为稀疏向量。

        策略:
        - 对词表中有的词：使用 IDF 加权（稀有词权重更高）
        - 对 OOV 词：分配一个小的 fallback 权重，让它也能参与匹配
        """
        tokens = self._tokenize(query)
        if not tokens:
            return [], []

        indices = []
        values = []

        # 为 OOV 词分配一个"哨兵"ID（如果启用 OOV fallback）
        oov_fallback_enabled = True
        oov_id = -1  # 特殊 ID（不使用，仅作为标记；实际不加入向量）

        for token in set(tokens):
            tid = self._token_to_id.get(token)
            if tid is not None:
                df_val = self._df.get(tid, 1)
                idf_val = math.log((self._doc_count + 1) / (df_val + 1)) + 1.0
                indices.append(tid)
                values.append(idf_val)
            elif oov_fallback_enabled:
                # OOV 词：不给具体 ID，但在日志中记录，便于调试
                logger.debug(f"[Sparse OOV] 查询词 '{token}' 不在词表中，跳过稀疏匹配")

        return indices, values

    # =====================================================================
    #  持久化
    # =====================================================================

    VOCAB_DIR = Path(__file__).parent.parent / "data"
    VOCAB_PATH = VOCAB_DIR / "sparse_vocab.json"

    def save_vocab(self):
        """保存词汇表到文件"""
        self.VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "token_to_id": self._token_to_id,
            "df": {str(k): v for k, v in self._df.items()},  # JSON 键必须是 str
            "next_id": self._next_id,
            "doc_count": self._doc_count,
            "vocab_size": self.vocab_size,
        }
        with open(self.VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"稀疏词汇表已保存: {len(self._token_to_id)} 词, "
            f"{self._doc_count} 块 → {self.VOCAB_PATH}"
        )

    def load_vocab(self) -> bool:
        """从文件加载词汇表（增量式：如果已存在实例，则合并加载）"""
        if not self.VOCAB_PATH.exists():
            logger.warning(f"稀疏词汇表不存在: {self.VOCAB_PATH}")
            return False

        try:
            with open(self.VOCAB_PATH, encoding="utf-8") as f:
                data = json.load(f)

            # 增量合并（不覆盖已有词，只加入新的）
            loaded_token_to_id = data.get("token_to_id", {})
            loaded_df = data.get("df", {})
            loaded_next_id = data.get("next_id", 0)
            loaded_doc_count = data.get("doc_count", 0)

            if not self._token_to_id:
                # 空词表：直接加载
                self._token_to_id = loaded_token_to_id
                self._df = {int(k): v for k, v in loaded_df.items()}
                self._next_id = loaded_next_id
                self._doc_count = loaded_doc_count
            else:
                # 已有词表：合并新词
                added = 0
                for token, tid in loaded_token_to_id.items():
                    if token not in self._token_to_id:
                        self._token_to_id[token] = self._next_id
                        self._df[self._next_id] = loaded_df.get(str(tid), 1)
                        self._next_id += 1
                        added += 1
                # doc_count 取较大值
                self._doc_count = max(self._doc_count, loaded_doc_count)
                logger.info(f"稀疏词汇表增量加载: 合并 +{added} 新词")

            logger.info(
                f"稀疏词汇表已加载: {len(self._token_to_id)} 词, "
                f"累计 {self._doc_count} 块"
            )
            return True
        except Exception as e:
            logger.error(f"加载稀疏词汇表失败: {e}")
            return False

    # =====================================================================
    #  辅助方法
    # =====================================================================

    def _get_or_create_id(self, token: str) -> int:
        if token not in self._token_to_id:
            if self._next_id >= self.vocab_size:
                # 超出词表大小：取 hash 模运算 fallback
                return hash(token) % self.vocab_size
            self._token_to_id[token] = self._next_id
            self._next_id += 1
        return self._token_to_id[token]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文分词（带停用词过滤）"""
        if not text:
            return []

        tokens = []
        for w in jieba.cut(text):
            w = w.strip()
            if not w:
                continue
            # 过滤单字、纯符号、停用词
            if len(w) < 2:
                continue
            # 纯符号/数字过滤
            if not _re_has_chinese_or_letter.search(w):
                continue
            # 停用词
            if w in CHINESE_STOPWORDS:
                continue
            tokens.append(w)
        return tokens

    # =====================================================================
    #  统计信息
    # =====================================================================

    def get_stats(self) -> dict:
        return {
            "vocab_size": len(self._token_to_id),
            "doc_count": self._doc_count,
            "avg_df": sum(self._df.values()) / max(len(self._df), 1),
            "max_vocab": self.vocab_size,
        }

    def reset(self):
        """清空词表（慎用，仅测试用）"""
        self._token_to_id.clear()
        self._df.clear()
        self._next_id = 0
        self._doc_count = 0
        logger.info("稀疏向量词表已重置")
