"""
tests/test_cwh_detector.py - CWH 検出エンジンのユニットテスト

pytest で実行: pytest tests/test_cwh_detector.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from cwh_screener import CWHCandidate, detect_cwh_candidates, _rolling_mean
from cwh_pattern_analyzer import PatternType, classify_pattern, analyze_patterns


# ============================
# フィクスチャ
# ============================

@pytest.fixture()
def base_cfg() -> dict:
    """テスト用デフォルト設定。"""
    return {
        "prior_trend_min": 0.25,
        "prior_trend_bars": 45,
        "cup_depth_min": 0.15,
        "cup_depth_max": 0.35,
        "cup_bars_min": 25,
        "handle_depth_min": 0.05,
        "handle_depth_max": 0.15,
        "handle_bars_min": 4,
        "handle_bars_max": 25,
        "handle_volume_ratio": 0.90,
        "rim_symmetry_max": 0.15,
        "ma_period": 50,
        "pivot_tolerance": 0.10,
        "breakout_volume_ratio": 1.5,
        "breakout_volume_period": 20,
        "cup_search_range": 120,
    }


@pytest.fixture()
def pattern_cfg() -> dict:
    """パターン分類用設定。"""
    return {
        "c_nh_handle_bars_max": 3,
        "c_sh_handle_depth_max": 0.10,
    }


def _make_cup_df(
    n_prior: int = 50,
    n_cup: int = 60,
    n_handle: int = 10,
    cup_depth: float = 0.25,
    handle_depth: float = 0.10,
    prior_gain: float = 0.40,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    """テスト用に CWH 形状を持つ OHLCV DataFrame を生成する。"""
    total = n_prior + n_cup + n_handle + 5
    dates = pd.date_range("2020-01-01", periods=total, freq="B")

    close = np.zeros(total)
    volume = np.full(total, base_volume)

    # 事前上昇（prior_gain 分上昇）
    start_price = 1000.0
    prior_end = start_price * (1 + prior_gain)
    close[:n_prior] = np.linspace(start_price, prior_end, n_prior)

    # カップ（左リムから底へ下落、その後右リムへ回復）
    left_rim = prior_end
    cup_bottom = left_rim * (1 - cup_depth)
    half_cup = n_cup // 2
    close[n_prior : n_prior + half_cup] = np.linspace(left_rim, cup_bottom, half_cup)
    close[n_prior + half_cup : n_prior + n_cup] = np.linspace(cup_bottom, left_rim, n_cup - half_cup)

    # ハンドル（右リムから下落、ハンドル底を形成）
    right_rim = left_rim
    handle_bottom = right_rim * (1 - handle_depth)
    half_handle = n_handle // 2
    cup_end = n_prior + n_cup
    close[cup_end : cup_end + half_handle] = np.linspace(right_rim, handle_bottom, half_handle)
    close[cup_end + half_handle : cup_end + n_handle] = np.linspace(handle_bottom, right_rim, n_handle - half_handle)

    # ピボット後
    close[cup_end + n_handle :] = right_rim * 1.05

    # ハンドル期の出来高を縮小（カップ期の 70%）
    volume[cup_end : cup_end + n_handle] = base_volume * 0.70

    # ピボットバーの出来高を急増（2倍）
    pv_bar = cup_end + n_handle
    if pv_bar < total:
        volume[pv_bar] = base_volume * 2.5

    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ============================
# テスト
# ============================

class TestRollingMean:
    def test_basic(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_mean(s, 3)
        # 最初の 2 要素は NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_window_larger_than_series(self) -> None:
        s = pd.Series([1.0, 2.0])
        result = _rolling_mean(s, 5)
        assert result.isna().all()


class TestCWHCandidateFlags:
    def test_all_pass_property(self) -> None:
        cand = CWHCandidate(
            ticker="TEST",
            idx_pl=0, idx_bc=1, idx_pr=2, idx_bh=3, idx_pv=4,
            r_prior=0.30, d_cup=0.25, n_cup=30, d_handle=0.10,
            n_handle=8, v_handle_avg=800, v_cup_avg=1000,
            s_rim=0.05, ma50_bh=900, close_bh=950,
            e_pivot=0.05, v_pv=2000, v20_pv=1000,
            cond_flags={
                "cond1_prior_trend": True,
                "cond2_cup_depth": True,
                "cond3_cup_bars": True,
                "cond4_handle_depth": True,
                "cond5_handle_bars": True,
                "cond6_volume": True,
                "cond7_symmetry": True,
                "cond8_ma50": True,
                "cond9_pivot": True,
                "cond10_breakout_vol": True,
            },
        )
        assert cand.all_pass is True

    def test_partial_fail(self) -> None:
        cand = CWHCandidate(
            ticker="TEST",
            idx_pl=0, idx_bc=1, idx_pr=2, idx_bh=3, idx_pv=4,
            r_prior=0.10,  # 条件1 失敗
            d_cup=0.25, n_cup=30, d_handle=0.10,
            n_handle=8, v_handle_avg=800, v_cup_avg=1000,
            s_rim=0.05, ma50_bh=900, close_bh=950,
            e_pivot=0.05, v_pv=2000, v20_pv=1000,
            cond_flags={"cond1_prior_trend": False, "cond2_cup_depth": True},
        )
        assert cand.all_pass is False


class TestDetectCWHCandidates:
    def test_returns_list(self, base_cfg: dict) -> None:
        df = _make_cup_df()
        result = detect_cwh_candidates(df, "TEST", base_cfg)
        assert isinstance(result, list)

    def test_short_df_returns_empty(self, base_cfg: dict) -> None:
        df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [101.0] * 10,
                "Low": [99.0] * 10,
                "Close": [100.0] * 10,
                "Volume": [1000.0] * 10,
            }
        )
        result = detect_cwh_candidates(df, "TEST", base_cfg)
        assert result == []

    def test_empty_df_returns_empty(self, base_cfg: dict) -> None:
        df = pd.DataFrame()
        result = detect_cwh_candidates(df, "TEST", base_cfg)
        assert result == []


class TestPatternClassification:
    def _make_candidate(self, n_handle: int = 10, d_handle: float = 0.10) -> CWHCandidate:
        return CWHCandidate(
            ticker="TEST",
            idx_pl=0, idx_bc=30, idx_pr=60, idx_bh=70, idx_pv=71,
            r_prior=0.30, d_cup=0.25, n_cup=60, d_handle=d_handle,
            n_handle=n_handle, v_handle_avg=800, v_cup_avg=1000,
            s_rim=0.05, ma50_bh=900, close_bh=950,
            e_pivot=0.05, v_pv=2000, v20_pv=1000,
            cond_flags={f"cond{i}": True for i in range(1, 11)},
        )

    def test_c_nh_classification(self, pattern_cfg: dict) -> None:
        """N_handle < 4 は C-NH と判定される。"""
        cand = self._make_candidate(n_handle=2)
        result = classify_pattern(cand, pattern_cfg)
        assert result.pattern_type == PatternType.C_NH

    def test_c_sh_classification(self, pattern_cfg: dict) -> None:
        """D_handle < 10% は C-SH と判定される。"""
        cand = self._make_candidate(n_handle=8, d_handle=0.07)
        result = classify_pattern(cand, pattern_cfg)
        assert result.pattern_type == PatternType.C_SH

    def test_cwh_classification(self, pattern_cfg: dict) -> None:
        """N_handle >= 4 かつ D_handle >= 10% は CWH と判定される。"""
        cand = self._make_candidate(n_handle=10, d_handle=0.12)
        result = classify_pattern(cand, pattern_cfg)
        assert result.pattern_type == PatternType.CWH

    def test_analyze_patterns_filters_non_pass(self, pattern_cfg: dict) -> None:
        """all_pass=False の候補は analyze_patterns に含まれない。"""
        cand_pass = self._make_candidate(n_handle=10, d_handle=0.12)
        cand_fail = CWHCandidate(
            ticker="FAIL",
            idx_pl=0, idx_bc=1, idx_pr=2, idx_bh=3, idx_pv=4,
            r_prior=0.10, d_cup=0.25, n_cup=30, d_handle=0.10,
            n_handle=8, v_handle_avg=800, v_cup_avg=1000,
            s_rim=0.05, ma50_bh=900, close_bh=950,
            e_pivot=0.05, v_pv=2000, v20_pv=1000,
            cond_flags={"cond1_prior_trend": False},
        )
        results = analyze_patterns([cand_pass, cand_fail], pattern_cfg)
        assert len(results) == 1
        assert results[0].ticker == "TEST"
