/**
 * 威科夫分析系统 - 前端应用
 */

// ============== 配置 ==============
const API_BASE = '/api';

// 事件颜色配置
const EVENT_COLORS = {
    // 看涨事件 - 绿色系
    SC: '#00C853', AR: '#69F0AE', ST: '#00E676', SOS: '#1B5E20',
    LPS: '#4CAF50', SPRING: '#76FF03', JAC: '#00BFA5', BUEC: '#64FFDA', TEST: '#A5D6A7',
    // 看跌事件 - 红色系
    BC: '#FF1744', SOW: '#D50000', LPSY: '#F44336', UT: '#FF5252', UTAD: '#FF8A80',
    // 中性事件
    PSY: '#2196F3', TR: '#9E9E9E',
};

// 事件图标
const EVENT_ICONS = {
    bullish: '▲',
    bearish: '▼',
    neutral: '◆',
};

// ============== 全局状态 ==============
let klineChart = null;
let volumeChart = null;
let klineSeries = null;
let volumeSeries = null;
let markers = [];
let currentData = null;

// ============== 初始化 ==============
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    bindEvents();
    
    // 默认加载 BTC/USDT
    loadAnalysis('BTC-USDT', '1h', 200);
});

/**
 * 初始化图表
 */
function initCharts() {
    const container = document.getElementById('chart-container');
    const klineContainer = document.getElementById('kline-chart');
    const volumeContainer = document.getElementById('volume-chart');
    
    // 计算尺寸
    const width = container.clientWidth;
    const klineHeight = Math.max(350, container.clientHeight * 0.7);
    const volumeHeight = Math.max(150, container.clientHeight * 0.3);
    
    // 图表通用配置
    const chartOptions = {
        width: width,
        layout: {
            background: { type: 'solid', color: '#161b22' },
            textColor: '#8b949e',
        },
        grid: {
            vertLines: { color: '#21262d' },
            horzLines: { color: '#21262d' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: '#58a6ff',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
            },
            horzLine: {
                color: '#58a6ff',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
            },
        },
        timeScale: {
            borderColor: '#30363d',
            timeVisible: true,
            secondsVisible: false,
        },
        rightPriceScale: {
            borderColor: '#30363d',
        },
    };
    
    // 创建 K 线图表
    klineChart = LightweightCharts.createChart(klineContainer, {
        ...chartOptions,
        height: klineHeight,
    });
    
    // K 线系列
    klineSeries = klineChart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });
    
    // 创建成交量图表
    volumeChart = LightweightCharts.createChart(volumeContainer, {
        ...chartOptions,
        height: volumeHeight,
    });
    
    // 成交量系列
    volumeSeries = volumeChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
    });
    
    volumeChart.priceScale('').applyOptions({
        scaleMargins: {
            top: 0.1,
            bottom: 0,
        },
    });
    
    // 同步时间轴
    klineChart.timeScale().subscribeVisibleTimeRangeChange(() => {
        const range = klineChart.timeScale().getVisibleRange();
        if (range) {
            volumeChart.timeScale().setVisibleRange(range);
        }
    });
    
    volumeChart.timeScale().subscribeVisibleTimeRangeChange(() => {
        const range = volumeChart.timeScale().getVisibleRange();
        if (range) {
            klineChart.timeScale().setVisibleRange(range);
        }
    });
    
    // 响应窗口大小变化
    window.addEventListener('resize', () => {
        const newWidth = container.clientWidth;
        klineChart.applyOptions({ width: newWidth });
        volumeChart.applyOptions({ width: newWidth });
    });
}

/**
 * 绑定事件
 */
function bindEvents() {
    // 分析按钮
    document.getElementById('analyze-btn').addEventListener('click', () => {
        const symbol = document.getElementById('symbol-select').value;
        const timeframe = document.getElementById('timeframe-select').value;
        const limit = parseInt(document.getElementById('limit-input').value) || 200;
        loadAnalysis(symbol, timeframe, limit);
    });
    
    // 刷新按钮
    document.getElementById('refresh-btn').addEventListener('click', () => {
        if (currentData) {
            const symbol = document.getElementById('symbol-select').value;
            const timeframe = document.getElementById('timeframe-select').value;
            const limit = parseInt(document.getElementById('limit-input').value) || 200;
            loadAnalysis(symbol, timeframe, limit);
        }
    });
    
    // 回车键触发分析
    document.getElementById('limit-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('analyze-btn').click();
        }
    });
}

/**
 * 加载分析数据
 */
async function loadAnalysis(symbol, timeframe, limit) {
    showLoading(true);
    
    try {
        const response = await fetch(
            `${API_BASE}/analyze/${symbol}?timeframe=${timeframe}&limit=${limit}`
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '分析失败');
        }
        
        const data = await response.json();
        currentData = data;
        
        // 更新图表
        updateCharts(data);
        
        // 更新状态栏
        updateStatusBar(data);
        
        // 更新事件列表
        updateEventsList(data.events);
        
        // 更新预测
        updatePredictions(data.predictions);
        
        // 更新操作建议
        updateActionPanel(data);
        
    } catch (error) {
        console.error('分析失败:', error);
        alert(`分析失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * 更新图表数据
 */
function updateCharts(data) {
    // 转换 K 线数据
    const klineData = data.ohlcv.map(bar => ({
        time: new Date(bar.timestamp).getTime() / 1000,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
    }));
    
    // 转换成交量数据
    const volumeData = data.ohlcv.map(bar => ({
        time: new Date(bar.timestamp).getTime() / 1000,
        value: bar.volume,
        color: bar.close >= bar.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
    }));
    
    // 设置数据
    klineSeries.setData(klineData);
    volumeSeries.setData(volumeData);
    
    // 添加事件标记
    addEventMarkers(data.events);
    
    // 添加支撑/阻力线
    addPriceLines(data.levels, data.range);
    
    // 自适应显示
    klineChart.timeScale().fitContent();
    volumeChart.timeScale().fitContent();
}

/**
 * 添加事件标记
 */
function addEventMarkers(events) {
    // 清除旧标记
    markers = [];
    
    events.forEach(event => {
        const time = new Date(event.timestamp).getTime() / 1000;
        const color = EVENT_COLORS[event.type] || '#FFFFFF';
        const position = event.direction === 'bearish' ? 'aboveBar' : 'belowBar';
        const shape = event.direction === 'bearish' ? 'arrowDown' : 'arrowUp';
        
        markers.push({
            time: time,
            position: position,
            color: color,
            shape: shape,
            text: `${event.type} (${(event.confidence * 100).toFixed(0)}%)`,
        });
    });
    
    klineSeries.setMarkers(markers);
}

/**
 * 添加价格线
 */
function addPriceLines(levels, range) {
    // 移除旧的价格线（通过重新创建系列来实现）
    // 注意：lightweight-charts 没有直接删除价格线的方法
    
    // 添加支撑线
    if (levels && levels.support) {
        levels.support.slice(0, 3).forEach((price, i) => {
            klineSeries.createPriceLine({
                price: price,
                color: '#4CAF50',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `S${i + 1}`,
            });
        });
    }
    
    // 添加阻力线
    if (levels && levels.resistance) {
        levels.resistance.slice(0, 3).forEach((price, i) => {
            klineSeries.createPriceLine({
                price: price,
                color: '#F44336',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `R${i + 1}`,
            });
        });
    }
    
    // 添加区间线
    if (range && range.low && range.high) {
        klineSeries.createPriceLine({
            price: range.low,
            color: '#FFD700',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: '区间低',
        });
        klineSeries.createPriceLine({
            price: range.high,
            color: '#FFD700',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: '区间高',
        });
    }
}

/**
 * 更新状态栏
 */
function updateStatusBar(data) {
    // 市场结构
    const structureEl = document.getElementById('market-structure');
    structureEl.textContent = translateStructure(data.market_structure);
    
    // 方向偏向
    const biasEl = document.getElementById('bias-value');
    if (data.state) {
        biasEl.textContent = translateBias(data.state.bias);
        biasEl.className = `status-value ${data.state.bias}`;
    } else {
        biasEl.textContent = '--';
        biasEl.className = 'status-value';
    }
    
    // 当前阶段
    const stateEl = document.getElementById('state-value');
    if (data.state) {
        stateEl.textContent = translateState(data.state.current_state);
    } else {
        stateEl.textContent = '--';
    }
    
    // 事件数量
    document.getElementById('events-count').textContent = data.events_count;
}

/**
 * 更新事件列表
 */
function updateEventsList(events) {
    const container = document.getElementById('events-list');
    
    if (!events || events.length === 0) {
        container.innerHTML = '<p class="empty-hint">未检测到威科夫事件</p>';
        return;
    }
    
    // 按时间倒序排列，显示最近的事件
    const sortedEvents = [...events].reverse();
    
    container.innerHTML = sortedEvents.map(event => `
        <div class="event-item ${event.direction}">
            <div class="event-info">
                <span class="event-type">${EVENT_ICONS[event.direction]} ${event.type}</span>
                <span class="event-price">价格: ${formatPrice(event.price)}</span>
            </div>
            <span class="event-confidence">${(event.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

/**
 * 更新预测面板
 */
function updatePredictions(predictions) {
    const container = document.getElementById('predictions-list');
    
    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p class="empty-hint">暂无交易预测</p>';
        return;
    }
    
    container.innerHTML = predictions.map(pred => `
        <div class="prediction-item">
            <div class="prediction-header">
                <span class="prediction-bias ${pred.bias}">
                    ${pred.bias === 'bullish' ? '📈 看涨' : pred.bias === 'bearish' ? '📉 看跌' : '➖ 中性'}
                </span>
                <span class="prediction-probability">${(pred.probability * 100).toFixed(0)}%</span>
            </div>
            <p class="prediction-desc">${pred.description || '暂无描述'}</p>
            ${pred.entry_price ? `
                <div class="prediction-levels">
                    <small>入场: ${formatPrice(pred.entry_price)}</small>
                    ${pred.stop_loss ? `<small>止损: ${formatPrice(pred.stop_loss)}</small>` : ''}
                </div>
            ` : ''}
        </div>
    `).join('');
}

/**
 * 更新操作建议面板
 */
function updateActionPanel(data) {
    const container = document.getElementById('action-content');
    
    if (!data.state) {
        container.innerHTML = '<p class="empty-hint">等待分析结果</p>';
        return;
    }
    
    const state = data.state;
    const levels = data.levels;
    
    let html = `
        <div class="action-text">
            <strong>当前状态：</strong>${state.state_description}<br>
            <strong>阶段进度：</strong>${(state.phase_progress * 100).toFixed(0)}%<br>
            <strong>建议操作：</strong>${state.action_suggestion || '观望等待'}
        </div>
    `;
    
    // 添加关键价位
    if (levels && (levels.support.length > 0 || levels.resistance.length > 0)) {
        html += '<div class="action-levels">';
        
        if (levels.support.length > 0) {
            html += `
                <div class="level-item">
                    <span class="level-label">支撑位</span>
                    <span class="level-value support">${formatPrice(levels.support[0])}</span>
                </div>
            `;
        }
        
        if (levels.resistance.length > 0) {
            html += `
                <div class="level-item">
                    <span class="level-label">阻力位</span>
                    <span class="level-value resistance">${formatPrice(levels.resistance[0])}</span>
                </div>
            `;
        }
        
        html += '</div>';
    }
    
    // 添加预期事件
    if (state.next_expected_events && state.next_expected_events.length > 0) {
        html += `
            <div style="margin-top: 12px; font-size: 12px; color: var(--text-secondary);">
                <strong>预期事件：</strong>${state.next_expected_events.join(', ')}
            </div>
        `;
    }
    
    container.innerHTML = html;
}

/**
 * 显示/隐藏加载状态
 */
function showLoading(show) {
    const loading = document.getElementById('chart-loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

// ============== 辅助函数 ==============

function formatPrice(price) {
    if (price >= 1000) {
        return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else if (price >= 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(6);
    }
}

function translateStructure(structure) {
    const map = {
        'range': '横盘整理',
        'markup': '上涨趋势',
        'markdown': '下跌趋势',
        'accumulation': '吸筹',
        'distribution': '派发',
        'unknown': '未知',
    };
    return map[structure] || structure;
}

function translateBias(bias) {
    const map = {
        'bullish': '看涨 📈',
        'bearish': '看跌 📉',
        'neutral': '中性 ➖',
    };
    return map[bias] || bias;
}

function translateState(state) {
    const map = {
        'unknown': '未知',
        'accumulation_phase_a': '吸筹 A',
        'accumulation_phase_b': '吸筹 B',
        'accumulation_phase_c': '吸筹 C',
        'accumulation_phase_d': '吸筹 D',
        'accumulation_phase_e': '吸筹 E',
        'distribution_phase_a': '派发 A',
        'distribution_phase_b': '派发 B',
        'distribution_phase_c': '派发 C',
        'distribution_phase_d': '派发 D',
        'markup': '上涨',
        'markdown': '下跌',
        'range': '横盘',
    };
    return map[state] || state;
}

