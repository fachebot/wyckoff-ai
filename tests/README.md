# 测试文档

## 目录结构

```
tests/
├── __init__.py           # 测试套件说明
├── conftest.py           # 共享 fixtures 和配置
├── data/                 # 测试数据
├── snapshots/            # 回归测试快照
├── unit/                 # 单元测试
│   ├── test_features.py  # 特征计算测试
│   ├── test_rules.py     # 规则引擎测试
│   ├── test_state_machine.py  # 状态机测试
│   └── test_backtest.py  # 回测系统测试
├── integration/          # 集成测试
│   └── test_pipeline.py  # 完整流程测试
└── regression/           # 回归测试
    └── test_snapshots.py # 快照对比测试
```

## 运行测试

### 安装测试依赖

```bash
poetry install --with dev
```

### 运行所有测试

```bash
pytest
```

### 运行特定类型的测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 只运行回归测试
pytest -m regression

# 跳过慢速测试
pytest -m "not slow"

# 运行包含慢速测试
pytest --slow
```

### 运行特定测试

```bash
# 按文件名
pytest tests/unit/test_features.py

# 按测试名
pytest -k "test_atr"

# 按类名
pytest -k "TestComputeFeatures"
```

### 生成覆盖率报告

```bash
# 运行测试并生成覆盖率
pytest --cov=wyckoff_ai

# 生成 HTML 报告
pytest --cov=wyckoff_ai --cov-report=html

# 查看报告
open htmlcov/index.html
```

### 更新回归测试快照

```bash
pytest --update-snapshots -m regression
```

## 测试标记

| 标记 | 说明 |
|------|------|
| `@pytest.mark.unit` | 单元测试，快速、隔离 |
| `@pytest.mark.integration` | 集成测试，可能需要外部服务 |
| `@pytest.mark.regression` | 回归测试，快照对比 |
| `@pytest.mark.slow` | 慢速测试，默认跳过 |

## 编写测试指南

### 单元测试

```python
@pytest.mark.unit
class TestMyFunction:
    def test_basic_functionality(self):
        """基本功能应该正常"""
        result = my_function(input)
        assert result == expected
    
    def test_edge_case(self):
        """边界情况应该处理正确"""
        result = my_function(edge_input)
        assert result is not None
```

### 集成测试

```python
@pytest.mark.integration
class TestPipeline:
    def test_full_flow(self, sample_data):
        """完整流程应该正常"""
        # 步骤1
        result1 = step1(sample_data)
        # 步骤2
        result2 = step2(result1)
        # 验证
        assert result2.is_valid()
```

### 回归测试

```python
@pytest.mark.regression
class TestSnapshots:
    def test_output_snapshot(self, snapshot_manager, update_snapshots):
        """输出应该与快照匹配"""
        result = compute_something()
        
        match, msg = snapshot_manager.compare_snapshot(
            "my_snapshot",
            result,
            update=update_snapshots,
        )
        
        assert match, f"输出变化: {msg}"
```

## Fixtures

### 可用的 fixtures

| Fixture | 说明 |
|---------|------|
| `sample_ohlcv_df` | 模拟 K 线数据（100根） |
| `sample_ohlcv_with_features` | 带特征的 K 线数据 |
| `trending_up_df` | 上涨趋势数据 |
| `trending_down_df` | 下跌趋势数据 |
| `ranging_df` | 横盘震荡数据 |
| `sample_events` | 示例威科夫事件 |
| `climax_pattern_df` | 卖出高潮模式数据 |
| `snapshot_manager` | 快照管理器 |
| `update_snapshots` | 是否更新快照标志 |

### 使用示例

```python
def test_with_fixture(sample_ohlcv_df):
    df = compute_features(sample_ohlcv_df)
    assert "atr_14" in df.columns
```

## CI 配置

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Install dependencies
        run: poetry install --with dev
      
      - name: Run tests
        run: poetry run pytest --cov=wyckoff_ai
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 最佳实践

1. **测试命名**：使用 `test_` 前缀，描述清晰
2. **测试隔离**：每个测试独立，不依赖其他测试
3. **使用 fixtures**：避免在测试中重复创建数据
4. **添加文档字符串**：说明测试目的
5. **覆盖边界情况**：空数据、极端值、错误输入
6. **保持测试快速**：单元测试应该秒级完成

