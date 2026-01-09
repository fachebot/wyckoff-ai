# 威科夫 K 线分析工具

一个“像小型研究员一样”的威科夫（Wyckoff）分析工具：从 Binance 拉取 K 线数据 → 计算特征 → 规则识别结构/事件 → 生成 **中文报告（Markdown）+ 结构化结论（JSON）** →（可选）接入 LLM 让解释更像分析师。

## 功能

### 核心功能

- **数据**：Binance Spot REST 拉取 OHLCV（UTC），自动标记缺口
- **特征**：ATR、趋势斜率、成交量相对强弱（z-score）、pivot 高低点、Donchian 区间宽度、简易横盘区间识别
- **威科夫**：基于规则识别阶段/事件（SC/AR/ST/SOS/LPS/UT/UTAD/SOW/LPSY…），每个事件带 **证据** 与 **置信度**
- **输出**：`analysis.json` + `report.md`（含关键价位与剧本）
- **LangChain**：负责"编排"和（可选）LLM 生成更强的中文解释；不配 Key 也能跑（降级为模板报告）

### 回测系统

- **事件后验评估**：计算每种事件的 MFE/MAE、胜率、收益率统计
- **交易模拟引擎**：支持止损止盈、仓位管理、手续费滑点
- **性能指标**：胜率、盈亏比、Sharpe/Sortino/Calmar 比率、最大回撤
- **Walk-Forward 测试**：滚动窗口验证策略稳健性
- **按事件/方向统计**：分析不同事件类型的表现差异

### 可视化系统

- **交互式 K 线图表**：基于 Plotly 的专业级 K 线图
- **事件标注可视化**：威科夫事件在图表上的标注（带颜色编码和置信度）
- **关键价位标注**：支撑/阻力位、交易区间高亮
- **状态机转换图**：威科夫阶段状态机可视化
- **阶段进度图**：当前阶段进度条展示
- **回测图表**：资金曲线、回撤分析、交易分布图
- **HTML 交互报告**：带图表的完整回测报告

### Web 服务 ✨ NEW

- **REST API**：完整的 RESTful API 接口（FastAPI）
- **交互式前端**：现代化 Web 界面
- **实时 K 线图**：基于 TradingView Lightweight Charts
- **事件标注**：图表上实时显示威科夫事件
- **关键价位**：自动绘制支撑/阻力线
- **状态显示**：实时显示市场状态和交易建议
- **API 文档**：自动生成的 Swagger/ReDoc 文档

## 环境要求

- Python 3.10+

## 安装

```bash
poetry install
```

如果你还没装 Poetry：

```bash
pip install -U poetry
```

（可选）配置文件：

1. 复制 `wyckoff.example.toml` 为 `wyckoff.toml`（或 `~/.wyckoff/config.toml`）
2. 根据需要修改配置参数

（可选）启用 LLM 解释：

1. 复制 `env.example` 为 `.env`（或直接在系统环境变量里设置）
2. 填写 `OPENAI_API_KEY`（或你后续改成别的 provider 也可以）

## 配置系统

支持多种配置来源（优先级从高到低）：

1. **CLI 参数** - 命令行直接指定
2. **环境变量** - 前缀 `WYCKOFF_`，嵌套用双下划线（如 `WYCKOFF_ANALYSIS__DEFAULT_SYMBOL`）
3. **配置文件** - TOML 格式
4. **默认值** - 内置默认配置

### 配置文件查找顺序

1. `--config` 参数指定的路径
2. 当前目录 `wyckoff.toml`
3. 用户目录 `~/.wyckoff/config.toml`
4. 环境变量 `WYCKOFF_CONFIG_FILE` 指定的路径

### 查看当前配置

```bash
poetry run wyckoff-ai --show-config
```

### 配置文件示例

参见 `wyckoff.example.toml`

## 运行示例

### 基础分析

```bash
# 使用默认配置（BTC/USDT, 1h）
poetry run wyckoff-ai analyze

# 指定交易对和参数
poetry run wyckoff-ai analyze -s ETH/USDT -t 4h -l 500

# 完整参数
poetry run wyckoff-ai analyze --symbol BTC/USDT --timeframe 1h --limit 720 --out out
```

输出文件：

- `out/analysis.json` - 结构化分析结果
- `out/report.md` - 中文 Markdown 报告

### 多时间框架分析

```bash
# 使用默认交易对
poetry run wyckoff-ai mtf --preset swing

# 指定交易对
poetry run wyckoff-ai mtf -s BTC/USDT --preset intraday
```

### 策略回测

```bash
# 使用默认配置
poetry run wyckoff-ai backtest -s BTC/USDT

# 自定义参数（会覆盖配置文件）
poetry run wyckoff-ai backtest -s BTC/USDT --capital 50000 --stop-atr 1.5
```

回测参数（均可在配置文件中设置默认值）：

- `--capital` - 初始资金（默认 100000）
- `--position-size` - 单笔仓位百分比（默认 10%）
- `--stop-atr` - 止损距离 ATR 倍数（默认 2.0）
- `--target-atr` - 止盈距离 ATR 倍数（默认 3.0）
- `--min-confidence` - 最小置信度（默认 0.6）
- `--events` - 只交易特定事件，如 `SOS,SPRING,LPS`

输出文件：

- `out/backtest_report.md` - 回测报告
- `out/backtest_result.json` - 回测结果 JSON

### 事件后验评估

```bash
# 使用默认交易对
poetry run wyckoff-ai eval-events

# 指定参数
poetry run wyckoff-ai eval-events -s BTC/USDT -t 1h -l 1000
```

评估每种威科夫事件发生后的价格表现（MFE/MAE、胜率、收益率等）。

输出文件：

- `out/event_eval_report.md` - 事件评估报告
- `out/event_eval_result.json` - 评估结果 JSON

### 生成可视化图表

```bash
# 生成 HTML 交互式图表（默认）
poetry run wyckoff-ai chart -s BTC/USDT -t 1h -l 200

# 生成 PNG 静态图
poetry run wyckoff-ai chart -s BTC/USDT --format png

# 自定义参数
poetry run wyckoff-ai chart -s ETH/USDT -t 4h -l 300 --width 1920 --height 1080 --min-confidence 0.7
```

### 启动 Web 服务 ✨ NEW

```bash
# 启动 Web 服务（默认端口 8000）
poetry run wyckoff-ai serve

# 指定端口
poetry run wyckoff-ai serve --port 3000

# 开发模式（自动重载）
poetry run wyckoff-ai serve --reload
```

启动后访问：

- **前端界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**REST API 端点**：

- `GET /api/symbols` - 获取支持的交易对
- `GET /api/timeframes` - 获取支持的时间周期
- `GET /api/ohlcv/{symbol}` - 获取 K 线数据
- `GET /api/analyze/{symbol}` - 完整威科夫分析
- `GET /api/quick/{symbol}` - 快速分析

图表参数：

- `--format` - 输出格式：`html`（默认）、`png`、`svg`、`pdf`
- `--width` - 图片宽度（默认 1600）
- `--height` - 图片高度（默认 900）
- `--min-confidence` - 最小事件置信度
- `--no-volume` - 不显示成交量
- `--no-ema` - 不显示 EMA 均线

输出文件：

- `out/kline_chart.html` - K 线图表（带事件标注）
- `out/state_diagram.html` - 状态机转换图
- `out/phase_progress.html` - 阶段进度图
- `out/state_timeline.html` - 状态转换时间线

### 使用配置文件

```bash
# 使用指定配置文件
poetry run wyckoff-ai -c my-config.toml analyze

# 详细输出模式
poetry run wyckoff-ai -v analyze -s BTC/USDT

# 日志输出到文件
poetry run wyckoff-ai --log-file logs/analysis.log analyze -s BTC/USDT
```

## 兼容：继续使用 pip/requirements.txt（不推荐）

如果你必须使用 `pip`，仍可：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m wyckoff_ai.cli analyze --exchange binance --symbol BTC/USDT --timeframe 1h --limit 720 --out out
```

## 下一步增强建议

- **实时监控**：WebSocket 实时数据、事件推送通知
- **参数优化**：基于回测结果的自动参数调优
- **更稳的区间识别**：变点检测 / HMM / K-means state clustering
- **更强事件**：UTAD/派发结构、复合结构、分型级别嵌套
