"""
测试套件

测试结构：
- tests/unit/          单元测试（快速、隔离）
- tests/integration/   集成测试（端到端流程）
- tests/regression/    回归测试（数据快照对比）
- tests/conftest.py    共享fixtures

运行测试：
    pytest                          # 运行所有测试
    pytest -m unit                  # 只运行单元测试
    pytest -m integration           # 只运行集成测试
    pytest -m regression            # 只运行回归测试
    pytest --cov=wyckoff_ai         # 运行测试并生成覆盖率报告
    pytest -k "test_features"       # 运行名称匹配的测试
"""

