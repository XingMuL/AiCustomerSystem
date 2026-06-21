# 智能客服 RAG 系统配置

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # ========== 模型配置 ==========
    # 向量化模型
    embedding_model: str = "./bge-large-zh-v1.5"

    # ========== GPU 加速配置 ==========
    # 是否启用 GPU 加速向量化（True=强制启用GPU，False=强制禁用GPU，None=自动检测）
    # 自动检测模式：有 GPU 且可用则用 GPU，否则回退 CPU
    use_gpu: Optional[bool] = None
    # 使用的 GPU 设备 ID（单 GPU 填 0；多 GPU 可选填 0/1/2...；use_gpu=False 时忽略）
    gpu_device_id: int = 0

    # LLM 模型（DeepSeek，统一用于所有智能体及文档清洗）
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_api_key: str = "sk-d05b323874a8497b91360a8a68ae42b7"
    llm_model: str = "deepseek-chat"

    # ========== 文档清洗配置（使用 DeepSeek） ==========
    # 是否启用 LLM 文档清洗
    enable_llm_cleaning: bool = True
    # 清洗时使用的模型（可不同于对话模型，deepseek-chat 通用能力强）
    cleaning_model: str = "deepseek-chat"
    cleaning_temperature: float = 0.1
    cleaning_max_tokens: int = 4096
    # 是否在解析阶段就转为 Markdown 再清洗
    convert_to_markdown_first: bool = True

    # ========== 文档结构化配置（LLM 语义分章） ==========
    # 是否启用 LLM 语义结构化（将商品信息单元识别为 ### 独立章节，避免碎片化）
    enable_llm_structuring: bool = True

    # ========== 文档处理配置 ==========
    # 支持的文件类型（新增 Excel 和图片）
    supported_extensions: list = [
        ".pdf", ".docx", ".txt", ".md",
        ".xlsx", ".xls",           # Excel
        ".png", ".jpg", ".jpeg",   # 图片
        ".bmp", ".tiff", ".webp",
    ]

    # ========== 分块配置（优化为 2-4k token 父块 + 300-800 token 子块） ==========
    # 子块大小（字符数）—— 300-800 token ≈ 600-1600 中文字符
    child_chunk_size: int = 800
    child_chunk_overlap: int = 150

    # 父块对应子块数量（一个父块覆盖多少个子块）
    # 父块 = child_chunks_per_parent * child_chunk_size ≈ 6*800=4800 字符 ≈ 2400 token
    child_chunks_per_parent: int = 6

    # 子块向量化是否拼接章节标题前缀（增强上下文关联）
    child_vectorize_with_chapter_title: bool = True

    # ========== 向量数据库配置 ==========
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "customer_service_kb"

    # 向量维度（取决于 embedding_model，bge-large-zh-v1.5 = 1024）
    vector_dim: int = 1024

    # ========== 入库流水线性能配置 ==========
    # Embedding 批大小（每次喂给模型的文本条数，GPU 可设 128-256，CPU 设 32-64）
    embedding_batch_size: int = 64
    # 向量 upsert 批大小（攒够此数量后批量写入 Qdrant）
    # 注意：Qdrant 默认限制请求体 < 32MB，且索引构建需时间
    # 1024 维 dense vector ≈ 10-15KB/条 + sparse + payload ≈ 20KB/条
    # 保守：100 条 ≈ 2MB，既避免 payload 超限，也避免索引构建超时
    vector_upsert_batch_size: int = 100
    # 文本队列最大容量（生产者-消费者架构中的缓冲）
    text_queue_maxsize: int = 20000
    # 向量队列最大容量
    vector_queue_maxsize: int = 10000
    # IO 密集型工作线程数（读文件、清洗文本）
    io_workers: int = 4
    # Embedding 工作进程数（CPU 模式下用多进程规避 GIL；GPU 模式下自动降级为线程）
    embedding_workers: int = 2

    # ========== 检索配置 ==========
    # 混合检索召回数量（稠密 + 稀疏各取 N 个 → RRF 融合）
    hybrid_recall_k: int = 30

    # Rerank 后保留数量（粗召回 30 → Rerank 打分 → 保留 8 条）
    rerank_top_k: int = 8

    # 章节检索返回数量
    top_k_chapters: int = 5

    # 是否启用 Query 改写
    enable_query_rewrite: bool = True

    # 是否启用 HyDE（Hypothetical Document Embeddings）查询增强
    # 生成假设文档，用假设文档的向量替代原始查询向量进行检索，
    # 对短问句、专业名词、精确查询场景提升显著
    enable_hyde: bool = True
    # HyDE 使用的 LLM 模型（轻量即可，仅生成假设文档）
    hyde_model: str = "deepseek-chat"
    # HyDE 生成假设文档的 temperature（适当提高以增加多样性）
    hyde_temperature: float = 0.5
    # HyDE 生成的假设文档最大 token 数
    hyde_max_tokens: int = 300

    # 是否启用 Rerank 重排
    enable_rerank: bool = True

    # ========== 父块精炼（轻量 LLM 句子提取） ==========
    # 是否启用：从父块中提取与 query/子块相关的句子，过滤无关内容
    enable_chunk_refinement: bool = True
    # 父块精炼使用的模型（复用 DeepSeek API，轻量调用）
    chunk_refinement_model: str = "deepseek-chat"
    # 精炼调用的 temperature（越低越保守，只保留高相关句子）
    chunk_refinement_temperature: float = 0.1
    # 精炼 LLM 单次调用的最大输出 token（只输出句子，不需要长篇）
    chunk_refinement_max_tokens: int = 800
    # 精炼调用超时（秒）
    chunk_refinement_timeout: int = 15
    # 父块字符数低于此阈值跳过精炼（内容已经很短）
    chunk_refinement_min_length: int = 500
    # 单个父块精炼后最大字符数（超过则截断）
    chunk_refinement_max_length: int = 1200

    # ========== 子块相似度过滤 ==========
    # 子块最低绝对相似度阈值（0-1）：低于此分数的子块直接丢弃
    # 建议值：0.25-0.35，过低会保留噪声，过高会丢失边缘相关内容
    child_score_absolute_threshold: float = 0.35
    # 子块相对相似度阈值（0-1）：相对于最高得分的比例
    # 例如 0.3 表示低于最高分 30% 的子块被过滤掉
    # 设置为 0 则关闭相对阈值过滤
    child_score_relative_threshold: float = 0.0
    # 过滤后最少保留的子块数（确保即使所有分数都低，也不会导致上下文完全为空）
    child_min_keep_count: int = 1

    # DeepSeek API（用于 Query 改写、Rerank）
    deepseek_api_key: str = "sk-d05b323874a8497b91360a8a68ae42b7"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # ========== 文档元数据与版本管理配置 ==========
    # 默认操作者
    default_operator: str = "system"
    # 冲突处理策略: "reject" | "overwrite" | "keep_both" | "ask"
    conflict_strategy: str = "overwrite"
    # 是否在文档内容未变化时跳过索引
    skip_unchanged: bool = True

    # ========== 服务配置 ==========
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ========== 多智能体协作配置 ==========
    # LLM 模型（多智能体共用）
    agent_llm_model: str = "deepseek-chat"
    agent_llm_temperature: float = 0.3
    agent_llm_max_tokens: int = 4096       # 单次 LLM 调用的 token 上限

    # 意图分类使用更快的模型
    router_model: str = "deepseek-chat"
    router_temperature: float = 0.0

    # ========== 输入预处理配置 ==========
    # SentencePiece Unigram 模型路径 (若为空则使用内置规则)
    sp_model_path: str = "./data/sp_model.model"
    sp_vocab_size: int = 8000
    # 单次喂给 LLM 的最大 token 数
    input_max_tokens: int = 4096
    # 超长单句截断阈值（字符数）
    max_sentence_length: int = 500
    # 刷屏重复检测阈值（连续相同消息数）
    spam_threshold: int = 3

    # ========== 记忆系统配置 ==========
    # Redis 配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    # 会话过期时间（秒），默认 24 小时
    session_ttl: int = 86400
    # 滑动窗口最大 token 数
    max_memory_tokens: int = 8000
    # 滑动窗口保留最近轮次下限
    min_history_turns: int = 5
    # RAG 召回补漏条数
    rag_recall_top_k: int = 5

    # ========== Gossip 去中心化同步配置 ==========
    # 本节点 ID
    node_id: str = "node-1"
    # Gossip 同步间隔（秒）
    gossip_interval: int = 30
    # 每次 Gossip 同步的随机对等节点数
    gossip_fanout: int = 3
    # 已知对等节点列表（逗号分隔）
    gossip_peers: str = ""

    # ========== 数据库持久化配置 ==========
    db_path: str = "data/memory_backup.db"

    # ========== 降级策略配置 ==========
    # 熔断器：连续失败次数阈值
    circuit_breaker_fail_threshold: int = 3

    # ========== fTaoBao 电商系统集成配置 ==========
    # 票务 Agent 通过此接口查询用户订单信息
    ftaobao_api_base_url: str = "http://127.0.0.1:8001"
    ftaobao_internal_api_key: str = "ftaobao-internal-api-key-2024"
    # 熔断器：半开状态超时（秒），超时后尝试恢复
    circuit_breaker_timeout: int = 30
    # 熔断器：半开状态最多允许的探测请求数
    circuit_breaker_half_open_max: int = 2
    # 限流：每秒最大请求数（Level 1 轻度降级时生效）
    rate_limit_rps: float = 10.0
    # 限流：突发容量
    rate_limit_burst: int = 20
    # LLM 调用超时（秒），超时视为失败
    llm_timeout: int = 30
    # RAG 健康检查间隔（秒）
    rag_health_check_interval: int = 60
    # 降级状态持久化路径
    degradation_state_path: str = "data/degradation_state.json"

    # ========== 反思评判配置 ==========
    # 是否启用反思评判
    enable_reflection: bool = True
    # 反思最大重试次数
    reflection_max_retries: int = 2
    # 反思质量最低分（0-1），低于此分触发重试
    reflection_min_score: float = 0.6
    # 反思评判使用的模型（可复用 agent 模型）
    reflection_model: str = "deepseek-chat"

    # ========== RAG 评估配置 ==========
    # 评估并发数（同时评估的用例数，避免 LLM 限流，建议 2-5）
    eval_concurrency: int = 3
    # 评估每批处理大小（每批处理完触发一次 GC，控制内存峰值）
    eval_batch_size: int = 5
    # 评估结果缓存过期时间（秒），0 表示不缓存
    eval_result_ttl: int = 1800
    # 评估任务状态轮询最长等待时间（秒），超过视为失败
    eval_max_wait_seconds: int = 1800

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()