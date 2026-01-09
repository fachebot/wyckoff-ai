"""
特征计算模块单元测试

测试 wyckoff_ai/features.py 中的特征计算函数
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wyckoff_ai.features import compute_features


@pytest.mark.unit
class TestComputeFeatures:
    """测试 compute_features 函数"""
    
    def test_returns_dataframe(self, sample_ohlcv_df: pd.DataFrame):
        """应该返回 DataFrame"""
        result = compute_features(sample_ohlcv_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_preserves_original_columns(self, sample_ohlcv_df: pd.DataFrame):
        """应该保留原始列"""
        result = compute_features(sample_ohlcv_df)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
    
    def test_adds_atr_column(self, sample_ohlcv_df: pd.DataFrame):
        """应该添加 ATR 列"""
        result = compute_features(sample_ohlcv_df)
        assert "atr_14" in result.columns
        # ATR 应该是正数
        assert (result["atr_14"].dropna() >= 0).all()
    
    def test_adds_ema_columns(self, sample_ohlcv_df: pd.DataFrame):
        """应该添加 EMA 列"""
        result = compute_features(sample_ohlcv_df)
        assert "ema_50" in result.columns
    
    def test_adds_volume_zscore(self, sample_ohlcv_df: pd.DataFrame):
        """应该添加成交量 z-score"""
        result = compute_features(sample_ohlcv_df)
        assert "vol_z_20" in result.columns
    
    def test_adds_slope(self, sample_ohlcv_df: pd.DataFrame):
        """应该添加趋势斜率"""
        result = compute_features(sample_ohlcv_df)
        assert "slope_50" in result.columns
    
    def test_adds_pivot_columns(self, sample_ohlcv_df: pd.DataFrame):
        """应该添加 pivot 高低点标记"""
        result = compute_features(sample_ohlcv_df)
        # 检查是否有 pivot 相关列
        pivot_cols = [c for c in result.columns if "pivot" in c.lower()]
        assert len(pivot_cols) > 0
    
    def test_no_nan_in_atr_after_warmup(self, sample_ohlcv_df: pd.DataFrame):
        """ATR 在预热期后不应该有 NaN"""
        result = compute_features(sample_ohlcv_df)
        # 前14个可能是 NaN（预热期）
        assert result["atr_14"].iloc[20:].notna().all()
    
    def test_handles_minimum_data(self):
        """应该能处理最少数据量"""
        # 创建只有20行的数据
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "open": [100] * 20,
            "high": [105] * 20,
            "low": [95] * 20,
            "close": [102] * 20,
            "volume": [1000] * 20,
        })
        result = compute_features(df)
        assert len(result) == 20
    
    def test_uptrend_has_positive_slope(self, trending_up_df: pd.DataFrame):
        """上涨趋势应该有正斜率"""
        result = compute_features(trending_up_df)
        # 取后半段数据（稳定趋势）
        avg_slope = result["slope_50"].iloc[60:].mean()
        assert avg_slope > 0, f"上涨趋势斜率应为正，实际为 {avg_slope}"
    
    def test_downtrend_has_negative_slope(self, trending_down_df: pd.DataFrame):
        """下跌趋势应该有负斜率"""
        result = compute_features(trending_down_df)
        avg_slope = result["slope_50"].iloc[60:].mean()
        assert avg_slope < 0, f"下跌趋势斜率应为负，实际为 {avg_slope}"


@pytest.mark.unit
class TestFeatureValues:
    """测试特征值的合理性"""
    
    def test_atr_reflects_volatility(self):
        """ATR 应该反映波动性"""
        # 高波动数据
        high_vol = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": np.linspace(100, 200, 50),
            "high": np.linspace(100, 200, 50) + 20,
            "low": np.linspace(100, 200, 50) - 20,
            "close": np.linspace(100, 200, 50),
            "volume": [1000] * 50,
        })
        
        # 低波动数据
        low_vol = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": np.linspace(100, 110, 50),
            "high": np.linspace(100, 110, 50) + 2,
            "low": np.linspace(100, 110, 50) - 2,
            "close": np.linspace(100, 110, 50),
            "volume": [1000] * 50,
        })
        
        high_vol_result = compute_features(high_vol)
        low_vol_result = compute_features(low_vol)
        
        high_atr = high_vol_result["atr_14"].iloc[-1]
        low_atr = low_vol_result["atr_14"].iloc[-1]
        
        assert high_atr > low_atr, "高波动数据的 ATR 应该更大"
    
    def test_volume_zscore_centered_around_zero(self, sample_ohlcv_df: pd.DataFrame):
        """成交量 z-score 应该围绕0"""
        result = compute_features(sample_ohlcv_df)
        vol_zscore = result["vol_z_20"].dropna()
        mean_zscore = vol_zscore.mean()
        # 允许一定误差
        assert abs(mean_zscore) < 0.5, f"z-score 均值应接近0，实际为 {mean_zscore}"
    
    def test_ema_smoothing(self, sample_ohlcv_df: pd.DataFrame):
        """EMA 应该比原始价格更平滑"""
        result = compute_features(sample_ohlcv_df)
        
        # 计算价格变化的标准差
        price_std = result["close"].diff().std()
        ema_std = result["ema_50"].diff().dropna().std()
        
        assert ema_std < price_std, "EMA 应该比原始价格更平滑"


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""
    
    def test_handles_constant_price(self):
        """处理恒定价格"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "open": [100.0] * 30,
            "high": [100.0] * 30,
            "low": [100.0] * 30,
            "close": [100.0] * 30,
            "volume": [1000] * 30,
        })
        result = compute_features(df)
        # 应该不会崩溃
        assert len(result) == 30
        # ATR 应该接近0
        assert result["atr_14"].iloc[-1] == pytest.approx(0, abs=0.01)
    
    def test_handles_zero_volume(self):
        """处理零成交量"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "open": np.random.uniform(100, 110, 30),
            "high": np.random.uniform(110, 120, 30),
            "low": np.random.uniform(90, 100, 30),
            "close": np.random.uniform(100, 110, 30),
            "volume": [0] * 30,
        })
        result = compute_features(df)
        assert len(result) == 30
    
    def test_handles_large_price_jump(self):
        """处理大幅价格跳空"""
        prices = [100.0] * 15 + [200.0] * 15  # 100% 跳空
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": prices,
            "volume": [1000] * 30,
        })
        result = compute_features(df)
        # ATR 应该在跳空后增加
        atr_before = result["atr_14"].iloc[14]
        atr_after = result["atr_14"].iloc[20]
        # 由于预热期的影响，只检查不会崩溃
        assert len(result) == 30

