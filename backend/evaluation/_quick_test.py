"""快速验证 RAG 测试集和评估框架能否正常工作。

直接运行： python -m backend.evaluation._quick_test
"""
import sys
from pprint import pprint

try:
    from backend.evaluation.rag_test_cases import get_default_test_cases, list_categories, get_test_cases_by_category
except Exception as e:
    print(f"[FAIL] 导入测试集失败: {e}")
    sys.exit(1)

try:
    from backend.evaluation.rag_evaluator import (
        evaluate_faithfulness,
        evaluate_hallucination,
        evaluate_context_precision,
        evaluate_answer_relevancy,
        evaluate_retrieval_result,
    )
except Exception as e:
    print(f"[FAIL] 导入评估器失败: {e}")
    sys.exit(1)


# 1. 测试用例加载
print("\n=== 1. 测试用例加载 ===")
cases = get_default_test_cases()
print(f"共加载 {len(cases)} 个测试用例")
cats = list_categories()
print(f"类别: {cats}")
for c in cases[:2]:
    print(f"  - {c['id']} [{c['category']}] 难度={c.get('difficulty')}")

# 2. 评估器 - 简单测试数据
print("\n=== 2. 评估器基本运行 ===")
test_result = {
    "case_id": "demo_01",
    "question": "如何申请退货退款？",
    "retrieved_contexts": [
        "用户可以在订单详情页点击申请退货按钮，填写退货原因后提交。",
        "客服会在 24 小时内审核退货申请，审核通过后用户需寄回商品。",
        "商品寄回并验收后，退款会在 3-5 个工作日内原路退回。",
    ],
    "generated_answer": "您可以在订单详情页点击申请退货按钮，填写退货原因后提交，客服会在24小时内审核。",
    "expected_keywords": ["申请", "退货", "客服", "审核"],
    "forbidden_keywords": ["免费赠送", "全额退款无需寄回"],
    "expected_answer": "在订单详情页申请退货，等待客服审核通过后寄回商品，退款在3-5个工作日内原路退回。",
}

from backend.evaluation.rag_evaluator import RetrievalResult
rr = RetrievalResult(
    case_id=test_result["case_id"],
    case_category="demo",
    question=test_result["question"],
    retrieved_contexts=test_result["retrieved_contexts"],
    generated_answer=test_result["generated_answer"],
    reference_answer=test_result["expected_answer"],
    expected_keywords=test_result["expected_keywords"],
    forbidden_keywords=test_result["forbidden_keywords"],
)

r = evaluate_retrieval_result(rr)
print(f"  faithfulness = {r.metrics['faithfulness'].score:.2f} - {r.metrics['faithfulness'].reason}")
print(f"  hallucination = {r.metrics['hallucination'].score:.2f} - {r.metrics['hallucination'].reason}")
print(f"  context_precision = {r.metrics['context_precision'].score:.2f} - {r.metrics['context_precision'].reason}")
print(f"  answer_relevancy = {r.metrics['answer_relevancy'].score:.2f} - {r.metrics['answer_relevancy'].reason}")
print(f"  success = {r.success}")

# 3. 评估 runner（不调用真实 RAG pipeline，使用模拟 answer provider）
print("\n=== 3. Runner 批量评估 (模拟) ===")
try:
    from backend.evaluation.rag_runner import run_evaluation
except Exception as e:
    print(f"[FAIL] 导入 runner 失败: {e}")
    sys.exit(1)


def mock_provider(question: str) -> dict:
    """模拟的回答提供者：返回固定上下文和简略回答。"""
    return {
        "contexts": [
            "客服提供详细的退货流程和退款说明。",
            "用户需要在订单详情页发起操作，等待审核。",
        ],
        "answer": "您可以在订单详情页申请退货，客服会在24小时内审核，审核通过后寄回商品，退款在3-5个工作日内原路退回。",
    }


# 只选 3 个快速测试
selected = cases[:3]
summary = run_evaluation(test_cases=selected, answer_provider=mock_provider)
print(f"  总用例: {summary.total_cases}")
print(f"  成功: {summary.successful_cases}")
print(f"  平均总体得分: {summary.avg_overall:.2f}")
print(f"  平均延迟: {summary.avg_latency_ms:.0f}ms")
for cat, data in (summary.per_category or {}).items():
    print(f"    [{cat}] {data.get('cases', 0)} 用例, 总体 {data.get('avg_overall', 0):.2f}")

print("\n✅  测试集和评估框架运行正常！")
