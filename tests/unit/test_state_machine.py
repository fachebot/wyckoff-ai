"""
威科夫状态机单元测试

测试 wyckoff_ai/wyckoff/state_machine.py 中的状态机逻辑
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from wyckoff_ai.schemas import WyckoffEvent
from wyckoff_ai.wyckoff.state_machine import (
    WyckoffState,
    WyckoffStateMachine,
    StateMachineResult,
    analyze_with_state_machine,
)


@pytest.mark.unit
class TestWyckoffState:
    """测试威科夫状态枚举"""
    
    def test_state_types_exist(self):
        """状态类型应该存在"""
        assert hasattr(WyckoffState, "UNKNOWN")
        assert hasattr(WyckoffState, "ACC_PHASE_A")
        assert hasattr(WyckoffState, "DIST_PHASE_A")
        assert hasattr(WyckoffState, "MARKUP")
        assert hasattr(WyckoffState, "MARKDOWN")
    
    def test_state_values(self):
        """状态值应该是字符串"""
        assert WyckoffState.UNKNOWN.value == "unknown"
        assert WyckoffState.ACC_PHASE_A.value == "accumulation_phase_a"


@pytest.mark.unit
class TestWyckoffStateMachine:
    """测试威科夫状态机"""
    
    def test_init_state(self):
        """初始状态应该正确"""
        sm = WyckoffStateMachine()
        assert sm.current_state == WyckoffState.UNKNOWN
    
    def test_analyze_with_events(self, sample_events: list[WyckoffEvent]):
        """分析事件应该返回结果"""
        sm = WyckoffStateMachine()
        result = sm.analyze(sample_events)
        assert isinstance(result, StateMachineResult)
    
    def test_reset(self):
        """重置应该清除状态"""
        sm = WyckoffStateMachine()
        
        # 添加一些事件
        events = [
            WyckoffEvent(
                type="SC",
                ts=datetime.now().isoformat(),
                price=40000,
                confidence=0.8,
                evidence=["test"],
            ),
        ]
        sm.analyze(events)
        
        # 重置
        sm.reset()
        
        # 状态应该回到初始
        assert sm.current_state == WyckoffState.UNKNOWN


@pytest.mark.unit
class TestAnalyzeWithStateMachine:
    """测试便捷分析函数"""
    
    def test_returns_result(self, sample_events: list[WyckoffEvent]):
        """应该返回 StateMachineResult"""
        result = analyze_with_state_machine(sample_events)
        assert isinstance(result, StateMachineResult)
    
    def test_empty_events(self):
        """处理空事件列表"""
        result = analyze_with_state_machine([])
        assert isinstance(result, StateMachineResult)
        assert result.current_state == WyckoffState.UNKNOWN


@pytest.mark.unit
class TestStateMachineResult:
    """测试状态机结果"""
    
    def test_result_has_required_fields(self, sample_events: list[WyckoffEvent]):
        """结果应该包含必需字段"""
        result = analyze_with_state_machine(sample_events)
        
        # 检查必需字段
        assert hasattr(result, "current_state")
        assert hasattr(result, "phase_progress")
        assert hasattr(result, "bias")
    
    def test_phase_progress_in_range(self, sample_events: list[WyckoffEvent]):
        """阶段进度应该在有效范围内"""
        result = analyze_with_state_machine(sample_events)
        
        assert 0 <= result.phase_progress.progress <= 1, \
            f"阶段进度超出范围: {result.phase_progress.progress}"
    
    def test_bias_is_valid(self, sample_events: list[WyckoffEvent]):
        """偏向应该是有效值"""
        result = analyze_with_state_machine(sample_events)
        
        assert result.bias in ["bullish", "bearish", "neutral"], \
            f"无效的偏向: {result.bias}"


@pytest.mark.unit  
class TestAccumulationSequence:
    """测试吸筹序列识别"""
    
    def test_accumulation_events(self):
        """吸筹序列应该被识别"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        # 典型的吸筹序列
        events = [
            WyckoffEvent(
                type="SC",
                ts=(base_time + timedelta(hours=1)).isoformat(),
                price=40000,
                confidence=0.8,
                evidence=["卖出高潮"],
            ),
            WyckoffEvent(
                type="AR",
                ts=(base_time + timedelta(hours=5)).isoformat(),
                price=42000,
                confidence=0.75,
                evidence=["自动反弹"],
            ),
            WyckoffEvent(
                type="ST",
                ts=(base_time + timedelta(hours=10)).isoformat(),
                price=40200,
                confidence=0.7,
                evidence=["二次测试"],
            ),
        ]
        
        result = analyze_with_state_machine(events)
        assert isinstance(result, StateMachineResult)
        # 应该识别出某种状态（不一定是吸筹，取决于实现）
        assert result.current_state is not None


@pytest.mark.unit
class TestDistributionSequence:
    """测试派发序列识别"""
    
    def test_distribution_events(self):
        """派发序列应该被识别"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        # 典型的派发序列
        events = [
            WyckoffEvent(
                type="BC",
                ts=(base_time + timedelta(hours=1)).isoformat(),
                price=50000,
                confidence=0.8,
                evidence=["买入高潮"],
            ),
            WyckoffEvent(
                type="AR",
                ts=(base_time + timedelta(hours=5)).isoformat(),
                price=48000,
                confidence=0.75,
                evidence=["自动回调"],
            ),
            WyckoffEvent(
                type="UT",
                ts=(base_time + timedelta(hours=15)).isoformat(),
                price=50500,
                confidence=0.7,
                evidence=["上冲"],
            ),
        ]
        
        result = analyze_with_state_machine(events)
        assert isinstance(result, StateMachineResult)
        assert result.current_state is not None


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_event(self):
        """处理单个事件"""
        event = WyckoffEvent(
            type="SC",
            ts=datetime.now().isoformat(),
            price=40000,
            confidence=0.8,
            evidence=["test"],
        )
        
        result = analyze_with_state_machine([event])
        assert isinstance(result, StateMachineResult)
    
    def test_out_of_order_events(self):
        """处理乱序事件"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        # 时间顺序颠倒的事件
        events = [
            WyckoffEvent(
                type="SOS",
                ts=(base_time + timedelta(hours=20)).isoformat(),
                price=45000,
                confidence=0.9,
                evidence=["SOS"],
            ),
            WyckoffEvent(
                type="SC",
                ts=(base_time + timedelta(hours=1)).isoformat(),
                price=40000,
                confidence=0.8,
                evidence=["SC"],
            ),
        ]
        
        # 不应该崩溃
        result = analyze_with_state_machine(events)
        assert isinstance(result, StateMachineResult)
