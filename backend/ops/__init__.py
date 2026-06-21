from backend.ops.metrics_collector import get_metrics_collector
# router 在 main.py 中直接通过 from backend.ops.router import router 导入
# 避免循环导入: __init__.py → router.py → state_graph.py → __init__.py