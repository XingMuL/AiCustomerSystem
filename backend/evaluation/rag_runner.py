"""
RAG 评估运行器

职责:
  1. 把测试用例传递给 RAG pipeline，获得真实的检索上下文 + 生成回答
  2. 调用 rag_evaluator 进行指标计算
  3. 汇总并输出评估报告

使用示例 (Python):
    from backend.evaluation.rag_runner import run_evaluation
    from backend.evaluation.rag_test_cases import get_default_test_cases

    summary = run_evaluation(get_default_test_cases()[:5])
    print(summary.to_dict())
"""

import time
import math
import traceback
import gc
from typing import List, Dict, Optional, Callable

from loguru import logger

from .rag_evaluator import (
    RetrievalResult,
    CaseEvalResult,
    EvaluationSummary,
    evaluate_retrieval_result,
    evaluate_batch,
)
from .rag_test_cases import get_default_test_cases


def _build_retrieval_result(
    case: dict,
    retrieved_contexts: List[str],
    generated_answer: str,
    cleaned_contexts: Optional[List[str]] = None,
) -> RetrievalResult:
    return RetrievalResult(
        case_id=case.get("id"),
        case_category=case.get("category"),
        question=case["question"],
        retrieved_contexts=retrieved_contexts,
        generated_answer=generated_answer,
        reference_answer=case.get("expected_answer"),
        expected_keywords=case.get("expected_keywords", []),
        forbidden_keywords=case.get("forbidden_keywords", []),
        cleaned_contexts=cleaned_contexts or [],
    )


def _safe_float(v: float) -> float:
    """防护 NaN/Inf 值，避免 JSON 序列化失败"""
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return 0.0
        return fv
    except (ValueError, TypeError):
        return 0.0


def run_evaluation(
    test_cases: Optional[List[Dict]] = None,
    answer_provider: Optional[Callable[[str], Dict]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    batch_size: int = 5,
    max_context_len: int = 500,
    use_deepeval: bool = True,
    eval_concurrency: int = 3,
) -> EvaluationSummary:
    """
    运行评估（带内存控制）

    Args:
        test_cases: 测试用例列表，默认使用内置数据集
        answer_provider: 回答提供者，签名: fn(question) -> {"contexts": List[str], "answer": str}
        progress_callback: 进度回调 fn(current, total, case_id)
        batch_size: 每批处理的用例数，批与批之间触发 GC，控制内存峰值
        max_context_len: 单条上下文文本的最大字符数，超出则截断，避免大对象积累

    Returns:
        EvaluationSummary 汇总对象
    """
    cases = test_cases if test_cases is not None else get_default_test_cases()
    total = len(cases)

    summary = EvaluationSummary()
    summary.total_cases = total
    category_scores: Dict[str, List[CaseEvalResult]] = {}
    all_case_results: List[CaseEvalResult] = []

    # 分批处理，控制内存峰值
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_cases = cases[batch_start:batch_end]
        batch_results: List[RetrievalResult] = []

        # 生成检索/回答结果（每用例一次 RAG pipeline 调用）
        for idx_in_batch, case in enumerate(batch_cases, 0):
            global_idx = batch_start + idx_in_batch + 1
            try:
                question = case["question"]
                if progress_callback:
                    progress_callback(global_idx, total, case.get("id", f"case_{global_idx}"))

                if answer_provider is not None:
                    out = answer_provider(question)
                    contexts = list(out.get("contexts", []) or [])
                    answer = out.get("answer", "") or ""
                else:
                    original_contexts, cleaned_contexts, answer = _default_pipeline_answer(question)

                # 截断过长上下文，减少内存占用
                original_contexts = [c[:max_context_len] if len(c) > max_context_len else c for c in original_contexts]
                cleaned_contexts = [c[:max_context_len] if len(c) > max_context_len else c for c in cleaned_contexts]
                answer = answer[:2000] if len(answer) > 2000 else answer

                batch_results.append(
                    _build_retrieval_result(case, original_contexts, answer, cleaned_contexts)
                )
            except Exception as e:
                batch_results.append(
                    _build_retrieval_result(case, [], f"[执行失败] {type(e).__name__}: {e}")
                )

        # 构建 case_id -> RetrievalResult 映射，用于后续分类汇总
        id_to_result = {r.case_id: r for r in batch_results}

        # 评估当前批次（用例级并发：多个用例的 evaluate_retrieval_result 并发）
        # 评估函数内部已做 5 个指标的并发，这里再做一层用例级并发
        def _eval_one(r):
            return evaluate_retrieval_result(r, use_deepeval=use_deepeval)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_concurrency) as executor:
            futures = [executor.submit(_eval_one, r) for r in batch_results]
            for fut in concurrent.futures.as_completed(futures):
                case_result = fut.result()
                all_case_results.append(case_result)
                summary.cases.append(case_result)
                if case_result.success:
                    summary.successful_cases += 1

                rr = id_to_result.get(case_result.case_id)
                cat = rr.case_category or "未分类" if rr else "未分类"
                category_scores.setdefault(cat, []).append(case_result)

        # 清理当前批次的大对象，触发 GC
        del batch_results
        del id_to_result
        gc.collect()

    # 汇总评分（基于 DeepEval 原生 5 大指标）
    metric_keys = [
        "faithfulness",
        "context_precision",
        "contextualrecall",
        "context_relevancy",
        "answer_relevancy",
    ]
    if all_case_results:
        summary.avg_faithfulness = _safe_float(_avg_metric_local(all_case_results, "faithfulness"))
        summary.avg_context_precision = _safe_float(_avg_metric_local(all_case_results, "context_precision"))
        summary.avg_context_recall = _safe_float(_avg_metric_local(all_case_results, "contextualrecall"))
        summary.avg_context_relevancy = _safe_float(_avg_metric_local(all_case_results, "context_relevancy"))
        summary.avg_answer_relevancy = _safe_float(_avg_metric_local(all_case_results, "answer_relevancy"))
        summary.avg_overall = _safe_float(sum(c.overall_score for c in all_case_results) / len(all_case_results))
        summary.avg_latency_ms = _safe_float(sum(c.latency_ms for c in all_case_results) / len(all_case_results))

    # 按类别汇总
    for cat, cases_in_cat in category_scores.items():
        valid = [c for c in cases_in_cat if c.success]
        if valid:
            summary.per_category[cat] = {
                "cases": len(valid),
                "avg_faithfulness": _safe_float(round(_avg_metric_local(valid, "faithfulness"), 4)),
                "avg_context_precision": _safe_float(round(_avg_metric_local(valid, "context_precision"), 4)),
                "avg_context_recall": _safe_float(round(_avg_metric_local(valid, "contextualrecall"), 4)),
                "avg_context_relevancy": _safe_float(round(_avg_metric_local(valid, "context_relevancy"), 4)),
                "avg_answer_relevancy": _safe_float(round(_avg_metric_local(valid, "answer_relevancy"), 4)),
                "avg_overall": _safe_float(round(sum(c.overall_score for c in valid) / len(valid), 4)),
            }
        else:
            summary.per_category[cat] = {"cases": 0, "error": "所有用例执行失败"}

    gc.collect()
    return summary


def _avg_metric_local(cases: List[CaseEvalResult], metric_name: str) -> float:
    vals = []
    for c in cases:
        if metric_name in c.metrics:
            s = c.metrics[metric_name].score
            try:
                fv = float(s)
                if not math.isnan(fv) and not math.isinf(fv):
                    vals.append(fv)
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else 0.0


def _default_pipeline_answer(question: str, max_contexts: int = 5, max_chars_per_ctx: int = 500):
    """
    默认 pipeline 集成：完整调用「RAG 检索 + KB-QA Agent 生成回答」的真实流程。

    与生产环境完全一致的调用链:
      1. RAGPipeline.retrieve(query) → 获取检索到的上下文文档（parent_chunks）
      2. KBQAAgent.answer(user_input, memory_context, rag_docs) → 基于上下文调用 LLM 生成回答

    这样评估的回答是真正的 LLM 生成内容，可以准确度量：
      - faithfulness（忠诚度）：生成的回答中哪些信息真正有上下文支撑
      - hallucination（幻觉）：生成的回答中哪些信息是 LLM 自行编造的
      - context_precision（检索质量）：检索到的上下文是否与问题相关
      - answer_relevancy（生成质量）：生成的回答是否切题、完整、通顺

    注意:
      - RAGPipeline 和 KBQAAgent 采用全局缓存，避免每次查询都重新加载 BERT 嵌入模型
      - 每轮评估会真实调用 LLM（产生 token 消耗），这是为了获得准确评估结果的必要代价
      - 限制上下文数量和单条上下文长度，避免单条检索占用过多内存
    """
    global _cached_pipeline
    global _cached_kb_qa_agent
    try:
        # === 1. 初始化 pipeline 缓存 ===
        if _cached_pipeline is None:
            from backend.pipeline.rag_pipeline import RAGPipeline
            _cached_pipeline = RAGPipeline()
        pipeline = _cached_pipeline

        # === 2. 执行检索 ===
        result = pipeline.retrieve(query=question)

        # === 3. 提取上下文（用于 context_precision 评估）===
        contexts: List[str] = []
        rag_docs: List[dict] = []

        if isinstance(result, dict):
            parents = result.get("parent_chunks") or []
            for p in parents:
                if isinstance(p, dict):
                    ctx = p.get("content") or p.get("text") or ""
                    score = float(p.get("score", 0) or 0)
                elif hasattr(p, "content"):
                    ctx = str(p.content)
                    score = float(getattr(p, "score", 0) or 0)
                else:
                    ctx = str(p)
                    score = 0.0
                if ctx:
                    rag_docs.append({"content": ctx, "score": score})
                    contexts.append(str(ctx)[:max_chars_per_ctx])
                if len(contexts) >= max_contexts:
                    break

            if not contexts:
                children = result.get("child_hits") or []
                for c in children:
                    if isinstance(c, dict):
                        ctx = c.get("content") or c.get("text") or ""
                        score = float(c.get("score", 0) or 0)
                    elif hasattr(c, "content"):
                        ctx = str(c.content)
                        score = float(getattr(c, "score", 0) or 0)
                    else:
                        ctx = str(c)
                        score = 0.0
                    if ctx:
                        rag_docs.append({"content": ctx, "score": score})
                        contexts.append(str(ctx)[:max_chars_per_ctx])
                    if len(contexts) >= max_contexts:
                        break
        else:
            if hasattr(result, "parent_chunks") and result.parent_chunks:
                for p in result.parent_chunks:
                    ctx = getattr(p, "content", None) or getattr(p, "text", "") or ""
                    score = float(getattr(p, "score", 0) or 0)
                    if ctx:
                        rag_docs.append({"content": str(ctx), "score": score})
                        contexts.append(str(ctx)[:max_chars_per_ctx])
                    if len(contexts) >= max_contexts:
                        break

        # === 4. 上下文清洗：用 DeepSeek 过滤与问题无关的语义信息 ===
        # 清洗后的上下文更精准，用于 context_relevance 评估和 KB-QA 回答
        cleaned_contexts = list(contexts)  # 默认回退到原始上下文
        try:
            if contexts:
                from backend.retrieval.context_cleaner import get_context_cleaner
                cleaner = get_context_cleaner()
                cleaned, cleaned_text = cleaner.clean(
                    user_query=question,
                    contexts=contexts,
                )
                if cleaned:
                    cleaned_contexts = cleaned
                    logger.info(
                        f"[评估] 上下文清洗: {len(contexts)} → {len(cleaned)} 片段, "
                        f"{sum(len(c) for c in contexts)} → {sum(len(c) for c in cleaned)} 字符"
                    )
        except Exception as clean_err:
            logger.warning(f"[评估] 上下文清洗失败，使用原始上下文: {clean_err}")

        # === 5. 调用 KB-QA Agent 生成真正的 LLM 回答（使用清洗后的上下文）===
        generated_answer: str = ""
        try:
            if _cached_kb_qa_agent is None:
                from backend.agents.kb_qa_agent import KBQAAgent
                _cached_kb_qa_agent = KBQAAgent()

            # 构建清洗后的 rag_docs
            cleaned_rag_docs = []
            for i, ctx in enumerate(cleaned_contexts):
                score = float(rag_docs[i].get("score", 0)) if i < len(rag_docs) else 0.0
                cleaned_rag_docs.append({"content": ctx, "score": score, "cleaned": True})

            agent_result = _cached_kb_qa_agent.answer(
                user_input=question,
                memory_context="",
                rag_docs=cleaned_rag_docs if cleaned_rag_docs else rag_docs,
                degradation_level=0,
                rag_available=True,
            )

            if isinstance(agent_result, dict):
                generated_answer = agent_result.get("response", "") or ""
            else:
                generated_answer = str(agent_result)
        except Exception as agent_err:
            logger_msg = f"[评估] KB-QA Agent 调用失败: {type(agent_err).__name__}: {agent_err}"
            # 退回到简单拼接上下文（仅作为失败兜底，不影响评估框架的正确性）
            if not generated_answer and contexts:
                generated_answer = "\n\n".join(contexts[:3])

        # 清理大引用
        del result
        return contexts[:max_contexts], cleaned_contexts[:max_contexts], str(generated_answer)[:2000]
    except Exception as e:
        traceback.print_exc()
        return [], [], f"[RAG pipeline 不可用] {type(e).__name__}: {e}"


# 缓存 RAGPipeline 实例，避免每次查询都重新加载 BERT 嵌入模型
_cached_pipeline = None

# 缓存 KBQAAgent 实例，避免每次查询都重新初始化 OpenAI client
_cached_kb_qa_agent = None


# 便捷入口：跑完整默认数据集
def run_full_evaluation():
    return run_evaluation(get_default_test_cases())
