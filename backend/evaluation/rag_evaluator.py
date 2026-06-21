"""
RAG 评估核心模块（基于 DeepEval 框架）

核心思路：
  1. 使用 DeepEval 原生 5 大指标（LLM-as-a-Judge 模式）：
     - FaithfulnessMetric: 回答是否忠实于检索到的上下文
     - ContextualPrecisionMetric: 检索到的上下文是否都是相关的
     - ContextualRecallMetric: 参考回答中的信息是否都被覆盖
     - AnswerRelevancyMetric: 回答是否紧扣问题
     - ContextualRelevancyMetric: 上下文是否相关
  2. 同时保留轻量规则模式作为 fallback（当网络/API 失败时

所有指标归一化到 [0, 1]，1 为最优。
"""

import re
import math
import time
import json
import sys
import threading
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict

# Windows: 修复 ProactorEventLoop 关闭后 httpx async client 清理异常
# 必须在任何 asyncio 使用前设置，否则 RuntimeError: Event loop is closed
if sys.platform == "win32":
    import asyncio as _asyncio
    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass  # 可能已被 main.py 或其他模块设置

    # Patch asyncio.call_soon: event loop 关闭后静默，不抛 RuntimeError
    try:
        _orig_call_soon = _asyncio.BaseEventLoop.call_soon

        def _safe_call_soon(self, callback, *args, **kwargs):
            try:
                if self.is_closed():
                    return None
                return _orig_call_soon(self, callback, *args, **kwargs)
            except RuntimeError:
                return None

        _asyncio.BaseEventLoop.call_soon = _safe_call_soon
    except Exception:
        pass

    # Patch httpx client cleanup
    try:
        import httpx as _httpx
        if not getattr(_httpx.AsyncClient, "_patched", False):
            _orig_async_aclose = _httpx.AsyncClient.aclose

            async def _safe_async_aclose(self, *args, **kwargs):
                try:
                    return await _orig_async_aclose(self, *args, **kwargs)
                except (RuntimeError, Exception):
                    return None

            _httpx.AsyncClient.aclose = _safe_async_aclose
            _httpx.AsyncClient._patched = True
    except Exception:
        pass

from loguru import logger

# 模块级锁：保护 _run_deepeval_metrics 中 os.environ 的并发读写
_env_lock = threading.Lock()

# 模块加载时注入 DeepEval 所需的 OPENAI_API_KEY 环境变量
# DeepEval 内部创建 GPTModel 时会检查该变量，必须在 deepeval 包导入前设置
import os as _os
try:
    from backend.config import settings
    _os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)
    _os.environ.setdefault("OPENAI_BASE_URL", getattr(settings, "llm_api_base", "https://api.deepseek.com/v1"))
except Exception:
    pass  # 模块级静默失败，运行时还会在 _run_deepeval_metrics 中再次设置


# =====================================================================
#  数据结构
# =====================================================================

@dataclass
class LLMEvaluatorConfig:
    """LLM 评审配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class RetrievalResult:
    """单次检索的结果"""
    question: str
    retrieved_contexts: List[str]          # 检索到的上下文片段（原始，用于召回率计算）
    generated_answer: str                  # Agent 生成的最终回答
    reference_answer: Optional[str] = None # 期望回答（可选）
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    case_id: Optional[str] = None
    case_category: Optional[str] = None
    cleaned_contexts: List[str] = field(default_factory=list)  # 清洗后的上下文（用于精度/相关性计算）


@dataclass
class MetricScore:
    """单个评估指标的得分 + 证据"""
    name: str
    score: float                          # 0~1
    reason: str                           # 打分理由（简短描述）
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseEvalResult:
    """单个测试用例的完整评估结果"""
    case_id: str
    question: str
    generated_answer: str
    metrics: Dict[str, MetricScore] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    num_contexts: int = 0

    @property
    def overall_score(self) -> float:
        if not self.metrics:
            return 0.0
        vals = [m.score for m in self.metrics.values()]
        # NaN/Inf 防护
        valid = []
        for v in vals:
            try:
                fv = float(v)
                if not math.isnan(fv) and not math.isinf(fv):
                    valid.append(fv)
            except (ValueError, TypeError):
                pass
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "overall_score": self.overall_score,
            "success": self.success,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "num_contexts": self.num_contexts,
        }


@dataclass
class EvaluationSummary:
    """整批测试的汇总结果"""
    total_cases: int = 0
    successful_cases: int = 0
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_relevancy: float = 0.0
    avg_overall: float = 0.0
    avg_latency_ms: float = 0.0
    per_category: Dict[str, Dict] = field(default_factory=dict)
    cases: List[CaseEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_context_precision": round(self.avg_context_precision, 4),
            "avg_context_recall": round(self.avg_context_recall, 4),
            "avg_answer_relevancy": round(self.avg_answer_relevancy, 4),
            "avg_context_relevancy": round(self.avg_context_relevancy, 4),
            "avg_overall": round(self.avg_overall, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "per_category": self.per_category,
            "cases": [c.to_dict() for c in self.cases],
        }


# =====================================================================
#  工具函数（文本预处理
# =====================================================================

_PUNCT_RE = re.compile(r"[\s\u3000\u2000-\u206f,，.。!！?？;；:：\"'「」『』()（）\[\]【】<>《》/\\\-—_=+*&^%$#@!~`]+")


def _normalize(text: str) -> str:
    """标准化文本：去标点、统一小写、合并空白"""
    if not text:
        return ""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_key_phrases(text: str, min_len: int = 2) -> List[str]:
    """从文本中提取关键短语（数字、邮箱、中文短语）"""
    if not text:
        return []
    norm = _normalize(text)
    phrases = set()
    for m in re.findall(r"\d+(?:\.\d+)?%?", norm):
        if len(m) >= 1:
            phrases.add(m)
    for m in re.findall(r"\d{3,4}[-\s]\d{3,4}[-\s]\d{3,4}", norm):
        phrases.add(m)
    chars = [c for c in norm if '\u4e00' <= c <= '\u9fff' or c.isdigit() or c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    for n in (2, 3, 4):
        for i in range(len(chars) - n + 1):
            phrase = "".join(chars[i:i + n])
            if phrase and len(phrase) >= min_len:
                phrases.add(phrase)
    return list(phrases)


# =====================================================================
#  DeepEval 评估（主模式）
# =====================================================================

def _build_deepeval_case(result: RetrievalResult):
    """把 RetrievalResult -> LLMTestCase"""
    return _build_deepeval_case_with_context(result, result.retrieved_contexts)


def _build_deepeval_case_with_context(result: RetrievalResult, contexts: list[str]):
    """用指定的上下文列表构建 LLMTestCase"""
    try:
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=result.question,
            actual_output=result.generated_answer or "(未回答)",
            expected_output=result.reference_answer or "",
            retrieval_context=[c for c in contexts if c],
        )
    except Exception:
        return None


def evaluate_retrieval_result(
    result: RetrievalResult,
    use_deepeval: bool = True,
    llm_config: Optional[LLMEvaluatorConfig] = None,
) -> CaseEvalResult:
    """评估单个检索+生成结果"""
    t0 = time.time()
    case = CaseEvalResult(
        case_id=result.case_id or f"case_{int(time.time()*1000)}",
        question=result.question,
        generated_answer=result.generated_answer,
        num_contexts=len(result.retrieved_contexts),
    )

    try:
        if use_deepeval:
            _run_deepeval_metrics(case, result, llm_config)
        else:
            _run_rule_based_metrics(case, result)
    except Exception as e:
        case.success = False
        case.error = f"{type(e).__name__}: {e}"

    case.latency_ms = (time.time() - t0) * 1000
    return case


def _run_deepeval_metrics(
    case: CaseEvalResult,
    result: RetrievalResult,
    llm_config: Optional[LLMEvaluatorConfig] = None,
):
    """使用 DeepEval 原生指标进行评估"""
    import os

    # 线程安全：加锁保护 os.environ 的并发读写，避免多线程评估时的竞态条件
    with _env_lock:
        original_key = os.environ.get("OPENAI_API_KEY")
        original_base = os.environ.get("OPENAI_BASE_URL")

    try:
        if llm_config is None:
            from backend.config import settings
            llm_config = LLMEvaluatorConfig(
                api_key=settings.llm_api_key,
                base_url=getattr(settings, "llm_base_url", "https://api.deepseek.com/v1"),
                model=getattr(settings, "agent_llm_model", "deepseek-chat"),
            )
    except Exception:
        llm_config = None

    if llm_config:
        with _env_lock:
            os.environ["OPENAI_API_KEY"] = llm_config.api_key
            os.environ["OPENAI_BASE_URL"] = llm_config.base_url

    eval_model = llm_config.model if llm_config else "deepseek-chat"

    try:
        from deepeval.metrics import (
            FaithfulnessMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
        )
    except Exception as e:
        case.metrics["deepeval_not_available"] = MetricScore(
            "deepeval_not_available", 0.0,
            f"deepeval 框架不可用: {e}", []
        )
        _run_rule_based_metrics(case, result)
        return

    deepeval_case = _build_deepeval_case(result)

    # 如果存在清洗后的上下文，为召回率指标单独构建一个使用原始上下文的 case
    # 原因：contextualrecall 衡量检索系统对参考回答的完整覆盖能力，应使用原始上下文
    #       context_precision / context_relevancy 衡量上下文质量，应使用清洗后上下文
    deepeval_case_recall = None
    if result.cleaned_contexts:
        # 召回率专用 case：使用原始上下文（未经清洗）
        deepeval_case_recall = _build_deepeval_case_with_context(
            result, result.retrieved_contexts
        )
        # 主 case 使用清洗后上下文（精度/相关性指标）
        deepeval_case = _build_deepeval_case_with_context(
            result, result.cleaned_contexts
        )
        logger.debug(
            f"[评估] 双上下文模式: 原始 {len(result.retrieved_contexts)} 片段(召回), "
            f"清洗 {len(result.cleaned_contexts)} 片段(精度)"
        )

    if deepeval_case is None:
        _run_rule_based_metrics(case, result)
        return

    metrics_to_run = [
        ("faithfulness", FaithfulnessMetric(
            threshold=0.5, model=eval_model, verbose_mode=False)),
        ("context_precision", ContextualPrecisionMetric(
            threshold=0.5, model=eval_model, verbose_mode=False)),
        ("contextualrecall", ContextualRecallMetric(
            threshold=0.5, model=eval_model, verbose_mode=False)),
        ("answer_relevancy", AnswerRelevancyMetric(
            threshold=0.5, model=eval_model, verbose_mode=False)),
        ("context_relevancy", ContextualRelevancyMetric(
            threshold=0.5, model=eval_model, verbose_mode=False)),
    ]

    # 并发执行 5 个 DeepEval 指标
    def _evaluate_one(metric_item):
        metric_name, metric = metric_item
        # 召回率使用原始上下文，其他指标使用清洗后上下文
        if metric_name == "contextualrecall" and deepeval_case_recall is not None:
            target_case = deepeval_case_recall
        else:
            target_case = deepeval_case
        try:
            metric.measure(target_case)
            reason_str = str(metric.reason) if metric.reason else ""
            evidence = []
            if reason_str:
                evidence = [reason_str[:300]]
            score = float(metric.score) if metric.score is not None else 0.0
            # NaN / Inf 防护：DeepEval 偶尔返回非数值，导致 JSON 序列化失败
            if math.isnan(score) or math.isinf(score):
                score = 0.0
            return (metric_name, MetricScore(
                metric_name, round(max(0.0, min(1.0, score)), 4),
                reason_str[:200], evidence))
        except Exception as metric_err:
            return (metric_name, MetricScore(
                metric_name, 0.0, f"评估失败: {type(metric_err).__name__}", []))

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_evaluate_one, item) for item in metrics_to_run]
            for fut in concurrent.futures.as_completed(futures):
                metric_name, metric_score = fut.result()
                case.metrics[metric_name] = metric_score
    except Exception:
        for metric_name, metric in metrics_to_run:
            if metric_name in case.metrics:
                continue
            try:
                metric.measure(deepeval_case)
                reason_str = str(metric.reason) if metric.reason else ""
                evidence = []
                if reason_str:
                    evidence = [reason_str[:300]]
                score = float(metric.score) if metric.score is not None else 0.0
                if math.isnan(score) or math.isinf(score):
                    score = 0.0
                case.metrics[metric_name] = MetricScore(
                    metric_name, round(max(0.0, min(1.0, score)), 4),
                    reason_str[:200], evidence)
            except Exception as metric_err:
                case.metrics[metric_name] = MetricScore(
                    metric_name, 0.0, f"评估失败: {type(metric_err).__name__}", [])

    # 至少跑一些轻量指标作为参考
    _add_rule_based_fallback(case, result)

    # 恢复环境变量（线程安全）
    with _env_lock:
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            os.environ.pop("OPENAI_API_KEY", "")
        if original_base is not None:
            os.environ["OPENAI_BASE_URL"] = original_base
        else:
            os.environ.pop("OPENAI_BASE_URL", "")

    if not any(k in case.metrics for k in ["faithfulness", "context_precision", "contextualrecall", "answer_relevancy"]):
        _run_rule_based_metrics(case, result)


def _run_rule_based_metrics(case: CaseEvalResult, result: RetrievalResult):
    """轻量规则 fallback（当 deepeval 失败时）"""
    contexts_joined = _normalize(" ".join(result.retrieved_contexts))
    answer = _normalize(result.generated_answer)
    q_phrases = _extract_key_phrases(result.question, min_len=2)
    ctx_phrases = _extract_key_phrases(contexts_joined, min_len=2)

    # 问题关键词覆盖（faithfulness：回答内容是否在上下文中

    # faithfulness: 回答中的关键信息是否有支撑
    ans_phrases = _extract_key_phrases(result.generated_answer, min_len=2)
    if ans_phrases and contexts_joined:
        supported = [p for p in ans_phrases if _normalize(p) in contexts_joined]
        faith_score = round(len(supported) / len(ans_phrases), 4) if ans_phrases else 0.5
    else:
        faith_score = 0.5
    case.metrics["faithfulness"] = MetricScore("faithfulness", faith_score, "基于文本匹配", [])

    # context_precision: 检索到的上下文是否覆盖问题关键词
    if q_phrases and contexts_joined:
        q_hit = [p for p in q_phrases if _normalize(p) in contexts_joined]
        cp_score = round(len(q_hit) / len(q_phrases), 4) if q_phrases else 0.5
    else:
        cp_score = 0.5
    case.metrics["context_precision"] = MetricScore("context_precision", cp_score, "上下文问题关键词覆盖", [])

    # contextualrecall: 参考回答信息是否被覆盖
    if result.reference_answer:
        ref_phrases = _extract_key_phrases(result.reference_answer, min_len=2)
        ref_hit = [p for p in ref_phrases if _normalize(p) in answer]
        cr_score = round(len(ref_hit) / len(ref_phrases), 4) if ref_phrases else 0.5
    else:
        cr_score = 0.5
    case.metrics["contextualrecall"] = MetricScore("contextualrecall", cr_score, "参考回答关键词在回答中", [])

    # answer_relevancy: 回答是否切题
    if q_phrases and answer:
        q_in_ans = [p for p in q_phrases if _normalize(p) in answer]
        ar_score = round(len(q_in_ans) / len(q_phrases), 4) if q_phrases else 0.5
    else:
        ar_score = 0.5
    case.metrics["answer_relevancy"] = MetricScore("answer_relevancy", ar_score, "问题关键词在回答中覆盖", [])

    # context_relevancy: 上下文是否相关
    if q_phrases and contexts_joined:
        cr2_hit = [p for p in q_phrases if _normalize(p) in contexts_joined]
        cr2_score = round(len(cr2_hit) / len(q_phrases), 4) if q_phrases else 0.5
    else:
        cr2_score = 0.5
    case.metrics["context_relevancy"] = MetricScore("context_relevancy", cr2_score, "上下文覆盖问题关键词", [])


def _add_rule_based_fallback(case: CaseEvalResult, result: RetrievalResult):
    """为 case 增加一个轻量指标作为参考（但不影响主评估结果）"""
    try:
        ans_phrases = _extract_key_phrases(result.generated_answer, min_len=2)
        contexts_joined = _normalize(" ".join(result.retrieved_contexts))
        if ans_phrases and contexts_joined:
            supported = [p for p in ans_phrases if _normalize(p) in contexts_joined]
            faith = round(len(supported) / len(ans_phrases), 4) if ans_phrases else 0.5
            case.metrics["faithfulness"] = case.metrics.get("faithfulness", MetricScore("faithfulness", faith, "轻量规则检测", []))
    except Exception:
        pass


def evaluate_batch(results: List[RetrievalResult], **kwargs) -> EvaluationSummary:
    """批量评估并返回汇总"""
    summary = EvaluationSummary()
    summary.total_cases = len(results)

    category_scores: Dict[str, List[CaseEvalResult]] = {}

    for r in results:
        case_result = evaluate_retrieval_result(r, **kwargs)
        summary.cases.append(case_result)
        if case_result.success:
            summary.successful_cases += 1

        cat = r.case_category or "未分类"
        category_scores.setdefault(cat, []).append(case_result)

    if summary.cases:
        summary.avg_faithfulness = _avg_metric(summary.cases, "faithfulness")
        summary.avg_context_precision = _avg_metric(summary.cases, "context_precision")
        summary.avg_contextualrecall = _avg_metric(summary.cases, "contextualrecall")
        summary.avg_answer_relevancy = _avg_metric(summary.cases, "answer_relevancy")
        summary.avg_context_relevancy = _avg_metric(summary.cases, "context_relevancy")
        summary.avg_overall = sum(c.overall_score for c in summary.cases) / len(summary.cases)
        summary.avg_latency_ms = sum(c.latency_ms for c in summary.cases) / len(summary.cases)

    for cat, cases_in_cat in category_scores.items():
        valid = [c for c in cases_in_cat if c.success]
        if valid:
            summary.per_category[cat] = {
                "cases": len(valid),
                "avg_faithfulness": round(_avg_metric(valid, "faithfulness"), 4),
                "avg_context_precision": round(_avg_metric(valid, "context_precision"), 4),
                "avg_contextualrecall": round(_avg_metric(valid, "contextualrecall"), 4),
                "avg_answer_relevancy": round(_avg_metric(valid, "answer_relevancy"), 4),
                "avg_context_relevancy": round(_avg_metric(valid, "context_relevancy"), 4),
                "avg_overall": round(sum(c.overall_score for c in valid) / len(valid), 4),
            }
        else:
            summary.per_category[cat] = {"cases": 0, "error": "所有用例执行失败"}

    return summary


def _avg_metric(cases: List[CaseEvalResult], metric_name: str) -> float:
    vals = [c.metrics[metric_name].score for c in cases if metric_name in c.metrics]
    return sum(vals) / len(vals) if vals else 0.0


# =====================================================================
#  LLM 评审模式（可选增强）
# =====================================================================

def evaluate_with_llm(
    result: RetrievalResult,
    config: LLMEvaluatorConfig,
) -> Optional[CaseEvalResult]:
    """
    LLM 评审模式（可选增强）
    使用 DeepSeek Chat 让大模型作为评审员，对 RetrievalResult 进行综合打分
    返回 CaseEvalResult（如果 LLM 调用失败返回 None）
    """
    try:
        from openai import OpenAI  # lazy loading
    except Exception:
        return None

    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        context_block = "\n---\n".join(
            f"[上下文 {i+1}]\n{c}" for i, c in enumerate(result.retrieved_contexts)
        ) or "(未检索到上下文)"

        prompt = f"""你是一名严谨的 RAG 系统评测员。请对以下问答进行评估。

## 用户问题
{result.question}

## 检索到的上下文
{context_block}

## 系统回答
{result.generated_answer}

## 期望关键词（回答中应尽量包含）
{', '.join(result.expected_keywords) if result.expected_keywords else '（无）'}

## 禁用关键词（回答中不得出现，否则判定为幻觉）
{', '.join(result.forbidden_keywords) if result.forbidden_keywords else '（无）'}

## 参考回答（仅作质量参考，不需完全一致）
{result.reference_answer or '（无）'}

请以严格 JSON 格式返回，包含以下字段:
- faithfulness (0~1): 回答中所有关键信息是否都有上下文支撑
- hallucination_score (0~1): 1 表示无幻觉，0 表示严重幻觉
- context_precision (0~1): 检索到的上下文与问题的相关程度
- answer_relevancy (0~1): 回答是否精准、完整、通顺地回答了问题
- evidence (list[str]): 2~5 条简短的证据片段

仅输出 JSON，不要额外文本。"""

        t0 = time.time()
        response = client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)

        case = CaseEvalResult(
            case_id=result.case_id or f"llm_case_{int(time.time()*1000)}",
            question=result.question,
            generated_answer=result.generated_answer,
            num_contexts=len(result.retrieved_contexts),
            latency_ms=(time.time() - t0) * 1000,
        )
        for name, score_field in [
            ("faithfulness", "faithfulness"),
            ("hallucination", "hallucination_score"),
            ("context_precision", "context_precision"),
            ("answer_relevancy", "answer_relevancy"),
        ]:
            score = float(data.get(score_field, 0.0))
            case.metrics[name] = MetricScore(
                name=f"llm_{name}",
                score=round(max(0.0, min(1.0, score)), 4),
                reason="LLM 评审员评分",
                evidence=list(data.get("evidence", []))[:5],
            )
        return case
    except Exception as e:  # noqa
        return None
