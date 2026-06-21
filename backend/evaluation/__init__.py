"""RAG 评估模块

用法:
    from backend.evaluation import run_evaluation, get_default_test_cases, EvaluationSummary

    summary = run_evaluation()
    print(summary.to_dict())
"""

from .rag_evaluator import (
    RetrievalResult,
    CaseEvalResult,
    EvaluationSummary,
    MetricScore,
    evaluate_retrieval_result,
    evaluate_batch,
    evaluate_with_llm,
    LLMEvaluatorConfig,
)
from .rag_test_cases import (
    get_default_test_cases,
    get_test_cases_by_category,
    list_categories,
)
from .rag_runner import run_evaluation, run_full_evaluation

__all__ = [
    "RetrievalResult",
    "CaseEvalResult",
    "EvaluationSummary",
    "MetricScore",
    "evaluate_retrieval_result",
    "evaluate_batch",
    "evaluate_with_llm",
    "LLMEvaluatorConfig",
    "get_default_test_cases",
    "get_test_cases_by_category",
    "list_categories",
    "run_evaluation",
    "run_full_evaluation",
]
