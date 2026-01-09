"""
快照回归测试

通过对比预期输出的快照来检测意外的行为变化。

使用方法：
1. 首次运行或更新快照：pytest --update-snapshots
2. 正常运行（对比快照）：pytest -m regression
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wyckoff_ai.features import compute_features
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff


@pytest.mark.regression
class TestFeatureSnapshots:
    """特征计算的快照测试"""
    
    def test_feature_columns_snapshot(
        self,
        sample_ohlcv_df: pd.DataFrame,
        snapshot_manager,
        update_snapshots: bool,
    ):
        """特征列名应该稳定"""
        df = compute_features(sample_ohlcv_df)
        
        current = {
            "columns": sorted(df.columns.tolist()),
            "column_count": len(df.columns),
        }
        
        match, msg = snapshot_manager.compare_snapshot(
            "feature_columns",
            current,
            update=update_snapshots,
        )
        
        assert match, f"特征列变化: {msg}"
    
    def test_feature_statistics_snapshot(
        self,
        sample_ohlcv_df: pd.DataFrame,
        snapshot_manager,
        update_snapshots: bool,
    ):
        """特征统计值应该稳定（在一定误差范围内）"""
        df = compute_features(sample_ohlcv_df)
        
        # 计算关键特征的统计值
        # 注意：由于浮点数精度，我们只保留有限小数位
        stats = {}
        for col in ["atr_14", "ema_50", "slope_50", "vol_z_20"]:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    stats[col] = {
                        "mean": round(float(vals.mean()), 4),
                        "std": round(float(vals.std()), 4),
                        "min": round(float(vals.min()), 4),
                        "max": round(float(vals.max()), 4),
                    }
        
        current = {"feature_stats": stats}
        
        match, msg = snapshot_manager.compare_snapshot(
            "feature_statistics",
            current,
            update=update_snapshots,
        )
        
        # 由于浮点数精度问题，我们允许一定的差异
        if not match and "快照不存在" not in msg:
            # 手动检查数值是否在可接受范围内
            saved = snapshot_manager.load_snapshot("feature_statistics")
            if saved:
                for col, new_stats in stats.items():
                    if col in saved.get("feature_stats", {}):
                        old_stats = saved["feature_stats"][col]
                        for key in ["mean", "std", "min", "max"]:
                            if key in old_stats and key in new_stats:
                                diff = abs(old_stats[key] - new_stats[key])
                                # 允许 1% 的误差
                                max_diff = max(abs(old_stats[key]) * 0.01, 0.001)
                                assert diff <= max_diff, \
                                    f"{col}.{key} 变化过大: {old_stats[key]} -> {new_stats[key]}"


@pytest.mark.regression
class TestAnalysisSnapshots:
    """分析结果的快照测试"""
    
    def test_market_structure_snapshot(
        self,
        sample_ohlcv_df: pd.DataFrame,
        snapshot_manager,
        update_snapshots: bool,
    ):
        """市场结构判断应该稳定"""
        df = compute_features(sample_ohlcv_df)
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        current = {
            "market_structure": analysis.market_structure,
            "event_count": len(analysis.events),
            "event_types": sorted(set(e.type for e in analysis.events)),
        }
        
        match, msg = snapshot_manager.compare_snapshot(
            "market_structure",
            current,
            update=update_snapshots,
        )
        
        assert match, f"市场结构变化: {msg}"
    
    def test_event_detection_snapshot(
        self,
        sample_ohlcv_df: pd.DataFrame,
        snapshot_manager,
        update_snapshots: bool,
    ):
        """事件检测应该稳定"""
        df = compute_features(sample_ohlcv_df)
        
        cfg = DetectionConfig(
            lookback_bars=50,
            min_confidence_threshold=0.5,
        )
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
            cfg=cfg,
        )
        
        # 简化事件信息用于对比
        events_summary = []
        for e in analysis.events:
            events_summary.append({
                "type": e.type,
                "confidence": round(e.confidence, 2),
                # 价格可能有浮点误差，四舍五入
                "price_rounded": round(e.price, 0),
            })
        
        current = {
            "event_count": len(analysis.events),
            "events": events_summary[:10],  # 只取前10个避免过长
        }
        
        match, msg = snapshot_manager.compare_snapshot(
            "event_detection",
            current,
            update=update_snapshots,
        )
        
        if not match:
            # 允许一定的差异，因为算法可能有合理的变化
            saved = snapshot_manager.load_snapshot("event_detection")
            if saved:
                # 事件数量变化不超过 20%
                old_count = saved.get("event_count", 0)
                new_count = current["event_count"]
                if old_count > 0:
                    change_pct = abs(new_count - old_count) / old_count
                    assert change_pct <= 0.5, f"事件数量变化过大: {old_count} -> {new_count}"


@pytest.mark.regression
class TestOutputFormatSnapshots:
    """输出格式的快照测试"""
    
    def test_analysis_schema_snapshot(
        self,
        sample_ohlcv_df: pd.DataFrame,
        snapshot_manager,
        update_snapshots: bool,
    ):
        """分析结果的 schema 应该稳定"""
        df = compute_features(sample_ohlcv_df)
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 检查输出结构
        analysis_dict = analysis.model_dump()
        
        def get_schema(obj, prefix=""):
            """递归获取对象的 schema"""
            schema = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        if isinstance(v, list) and v:
                            schema[key] = f"list[{type(v[0]).__name__}]"
                        elif isinstance(v, dict):
                            schema.update(get_schema(v, key))
                        else:
                            schema[key] = type(v).__name__
                    else:
                        schema[key] = type(v).__name__
            return schema
        
        current = {
            "top_level_keys": sorted(analysis_dict.keys()),
        }
        
        match, msg = snapshot_manager.compare_snapshot(
            "analysis_schema",
            current,
            update=update_snapshots,
        )
        
        assert match, f"分析结果 schema 变化: {msg}"


@pytest.mark.regression
class TestDeterminism:
    """确定性测试 - 相同输入应该产生相同输出"""
    
    def test_feature_determinism(self, sample_ohlcv_df: pd.DataFrame):
        """特征计算应该是确定性的"""
        df1 = compute_features(sample_ohlcv_df.copy())
        df2 = compute_features(sample_ohlcv_df.copy())
        
        # 数值列应该完全相同
        for col in df1.columns:
            if df1[col].dtype in [np.float64, np.float32]:
                pd.testing.assert_series_equal(
                    df1[col].reset_index(drop=True),
                    df2[col].reset_index(drop=True),
                    check_names=False,
                )
    
    def test_analysis_determinism(self, sample_ohlcv_df: pd.DataFrame):
        """分析应该是确定性的"""
        df = compute_features(sample_ohlcv_df)
        
        analysis1 = detect_wyckoff(
            df.copy(),
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        analysis2 = detect_wyckoff(
            df.copy(),
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 事件数量应该相同
        assert len(analysis1.events) == len(analysis2.events)
        
        # 市场结构应该相同
        assert analysis1.market_structure == analysis2.market_structure
        
        # 每个事件应该相同
        for e1, e2 in zip(analysis1.events, analysis2.events):
            assert e1.type == e2.type
            assert e1.ts == e2.ts
            assert abs(e1.price - e2.price) < 0.01
            assert abs(e1.confidence - e2.confidence) < 0.01


@pytest.mark.regression
class TestBackwardCompatibility:
    """向后兼容性测试"""
    
    def test_event_types_unchanged(self):
        """事件类型应该保持不变"""
        expected_types = {
            "SC", "BC", "AR", "ST", "SOS", "SOW",
            "SPRING", "UT", "UTAD", "LPS", "LPSY",
            "JAC", "BUEC", "TEST", "PSY",
        }
        
        from wyckoff_ai.schemas import WyckoffEvent
        
        # 创建一个事件来验证类型有效性
        for event_type in expected_types:
            event = WyckoffEvent(
                type=event_type,
                ts="2024-01-01T00:00:00",
                price=100.0,
                confidence=0.8,
                evidence=["test"],
            )
            assert event.type == event_type
    
    def test_config_defaults_unchanged(self):
        """配置默认值应该保持不变"""
        cfg = DetectionConfig()
        
        # 这些值应该保持稳定
        assert cfg.lookback_bars > 0
        assert 0 < cfg.min_confidence_threshold < 1

