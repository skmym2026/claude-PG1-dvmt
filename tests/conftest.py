"""
tests/conftest.py - pytest 共有フィクスチャ

全テストファイルから参照できる共有フィクスチャを定義する。
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def screener_cfg() -> dict:
    """CWH スクリーナー設定（テスト用デフォルト値）。"""
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
        "local_high_window": 5,
    }


@pytest.fixture()
def pattern_cfg() -> dict:
    """パターン分類設定（テスト用デフォルト値）。"""
    return {
        "c_nh_handle_bars_max": 3,
        "c_sh_handle_depth_max": 0.10,
    }


@pytest.fixture()
def backtest_cfg() -> dict:
    """バックテスト・出口ルール設定（テスト用デフォルト値）。"""
    return {
        "entry_price_margin": 0.03,
        "entry_volume_ratio": 1.5,
        "entry_volume_period": 25,
        "stop_loss_ratio": 0.92,
        "take_profit_ratio": 1.20,
        "trailing_stop_enabled": True,
        "trailing_stop_ratio": 0.92,
        "trailing_stop_min_gain": 0.05,
        "max_hold_bars": 60,
        "peak_detection_volume_ratio": 2.0,
        "peak_detection_wick_ratio": 1.0,
        "cutoff_25days_gain_min": 0.10,
        "cutoff_75days_enabled": True,
    }


@pytest.fixture()
def sbi_cfg() -> dict:
    """SBI 証券連携設定（テスト用デフォルト値）。"""
    return {
        "backtest": {
            "stop_loss_ratio": 0.92,
            "take_profit_ratio": 1.20,
            "trailing_stop_min_gain": 0.05,
        },
        "sbi_integration": {
            "trail_width_percent": 3.0,
        },
    }


def make_cup_df(
    n_prior: int = 50,
    n_cup: int = 60,
    n_handle: int = 10,
    cup_depth: float = 0.25,
    handle_depth: float = 0.10,
    prior_gain: float = 0.40,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    """テスト用に CWH 形状を持つ OHLCV DataFrame を生成する。

    パターン構造:
      [prior: 事前上昇] → [cup: カップ形成] → [handle: ハンドル] → [breakout]
    """
    total = n_prior + n_cup + n_handle + 10
    dates = pd.date_range("2020-01-01", periods=total, freq="B")

    close = np.zeros(total)
    volume = np.full(total, base_volume)

    # 事前上昇
    start_price = 1000.0
    prior_end = start_price * (1 + prior_gain)
    close[:n_prior] = np.linspace(start_price, prior_end, n_prior)

    # カップ（U 字型）
    left_rim = prior_end
    cup_bottom = left_rim * (1 - cup_depth)
    half_cup = n_cup // 2
    close[n_prior : n_prior + half_cup] = np.linspace(left_rim, cup_bottom, half_cup)
    close[n_prior + half_cup : n_prior + n_cup] = np.linspace(cup_bottom, left_rim, n_cup - half_cup)

    # ハンドル
    right_rim = left_rim
    handle_bottom = right_rim * (1 - handle_depth)
    half_handle = n_handle // 2
    cup_end = n_prior + n_cup
    close[cup_end : cup_end + half_handle] = np.linspace(right_rim, handle_bottom, half_handle)
    close[cup_end + half_handle : cup_end + n_handle] = np.linspace(handle_bottom, right_rim, n_handle - half_handle)

    # ブレイクアウト後
    close[cup_end + n_handle :] = right_rim * 1.05

    # ハンドル期の出来高縮小（70%）
    volume[cup_end : cup_end + n_handle] = base_volume * 0.70

    # ピボットバーの出来高急増（2.5 倍）
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


@pytest.fixture()
def cup_df() -> pd.DataFrame:
    """標準的な CWH 形状の OHLCV DataFrame フィクスチャ。

    Note: テスト内でパラメータを変えたい場合は make_cup_df() を直接呼ぶこと。
    """
    return make_cup_df(
        n_prior=50,
        n_cup=60,
        n_handle=10,
        cup_depth=0.25,
        handle_depth=0.10,
    )
