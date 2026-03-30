"""
tests/test_peak_detection.py
ピーク検知テスト：上ヒゲ・大商い、窓開け陰線
"""

import pandas as pd
import pytest
from cwh_peak_detection import detect_upper_wick_large_volume, detect_gap_up_bearish, detect_peak


def _make_candle(open_, high, low, close, volume, prev_close=None):
    """1〜2本足のDataFrameを生成する（前日データを含む場合はprev_closeを指定）。"""
    rows = []
    if prev_close is not None:
        rows.append({
            "Open": prev_close * 0.99, "High": prev_close * 1.01,
            "Low": prev_close * 0.98, "Close": prev_close, "Volume": 500_000,
        })
    rows.append({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
    rng = pd.date_range("2025-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=rng)


# ---- 上ヒゲ・大商い ----

def test_upper_wick_large_volume_triggered():
    """上ヒゲ > 実体 かつ 出来高 >= 2 * V25 → ピーク判定 True"""
    # 始値100、終値101（小さな陽線）、高値150（大きな上ヒゲ）
    df = _make_candle(open_=100, high=150, low=99, close=101, volume=1_100_000)
    v25 = 500_000  # 2 * v25 = 1,000,000 → volume 1,100,000 で条件成立
    triggered, reason = detect_upper_wick_large_volume(df, v25)
    assert triggered is True
    assert "Upper wick" in reason


def test_upper_wick_small_volume_not_triggered():
    """出来高が少ない場合はピーク判定 False"""
    df = _make_candle(open_=100, high=150, low=99, close=101, volume=800_000)
    v25 = 500_000  # 2 * v25 = 1,000,000 → volume 800,000 で条件不成立
    triggered, _ = detect_upper_wick_large_volume(df, v25)
    assert triggered is False


def test_no_upper_wick_large_volume_not_triggered():
    """上ヒゲが実体以下の場合はピーク判定 False"""
    # 始値100、終値110（実体=10）、高値112（上ヒゲ=2 < 実体10）
    df = _make_candle(open_=100, high=112, low=99, close=110, volume=1_200_000)
    v25 = 500_000
    triggered, _ = detect_upper_wick_large_volume(df, v25)
    assert triggered is False


# ---- 窓開け陰線 ----

def test_gap_up_bearish_triggered():
    """窓開け（+3%以上）かつ陰線 → ピーク判定 True"""
    prev_close = 1000.0
    open_ = prev_close * 1.05  # +5% 窓開け
    close = open_ * 0.98       # 陰線（終値 < 始値）
    df = _make_candle(open_=open_, high=open_ * 1.01, low=close * 0.99,
                      close=close, volume=600_000, prev_close=prev_close)
    triggered, reason = detect_gap_up_bearish(df)
    assert triggered is True
    assert "Gap up" in reason


def test_gap_up_bullish_not_triggered():
    """窓開け後に陽線（終値 >= 始値）→ ピーク判定 False"""
    prev_close = 1000.0
    open_ = prev_close * 1.05
    close = open_ * 1.02  # 陽線
    df = _make_candle(open_=open_, high=close * 1.01, low=open_ * 0.99,
                      close=close, volume=600_000, prev_close=prev_close)
    triggered, _ = detect_gap_up_bearish(df)
    assert triggered is False


def test_small_gap_bearish_not_triggered():
    """窓開けが3%未満の場合は陰線でもピーク判定 False"""
    prev_close = 1000.0
    open_ = prev_close * 1.01  # +1%（小さな窓）
    close = open_ * 0.99       # 陰線
    df = _make_candle(open_=open_, high=open_ * 1.005, low=close * 0.99,
                      close=close, volume=600_000, prev_close=prev_close)
    triggered, _ = detect_gap_up_bearish(df)
    assert triggered is False


# ---- detect_peak 統合 ----

def test_detect_peak_returns_first_match():
    """上ヒゲ条件が成立すれば detect_peak が True を返す。"""
    df = _make_candle(open_=100, high=150, low=99, close=101, volume=1_100_000)
    v25 = 500_000
    triggered, reason = detect_peak(df, v25)
    assert triggered is True


def test_detect_peak_no_signal():
    """シグナルなしの場合は False を返す。"""
    df = _make_candle(open_=100, high=105, low=98, close=104, volume=600_000,
                      prev_close=100)
    v25 = 500_000
    triggered, _ = detect_peak(df, v25)
    assert triggered is False
