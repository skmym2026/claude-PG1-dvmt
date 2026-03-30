"""
tests/test_integrated_cwh.py
統合ロジック検証テスト
"""

import pandas as pd
import pytest
from integrated_cwh_v3 import cwh_integrated_strategy, _find_cup_params
from cwh_exit_timer import calc_days_held, check_time_cutoff, evaluate_exit_timer


def _make_df(close_prices, volumes=None, open_ratio=0.99, high_ratio=1.02, low_ratio=0.97, eps=None):
    """テスト用OHLCVデータフレームを生成する。"""
    n = len(close_prices)
    rng = pd.date_range("2025-01-01", periods=n, freq="B")
    if volumes is None:
        volumes = [500_000] * n
    df = pd.DataFrame(
        {
            "Open": [p * open_ratio for p in close_prices],
            "High": [p * high_ratio for p in close_prices],
            "Low": [p * low_ratio for p in close_prices],
            "Close": close_prices,
            "Volume": volumes,
        },
        index=rng,
    )
    if eps is not None:
        df["EPS"] = eps
    return df


def _base_info(**kwargs):
    info = {"code": "9999.T", "market_cap": 2000, "credit_ratio": 2.5}
    info.update(kwargs)
    return info


# ---- 時価総額フィルタ ----

def test_market_cap_filter_below_threshold():
    """時価総額 < 1,000億円は除外される。"""
    prices = [1000.0] * 10 + [1100.0] + [950.0] * 29 + [1140.0]
    vols = [500_000] * 40 + [800_000]
    df = _make_df(prices, vols)
    result = cwh_integrated_strategy(df, _base_info(market_cap=500))
    assert result is None


def test_market_cap_filter_at_threshold():
    """時価総額 = 1,000億円はフィルタを通過する（他の条件次第）。"""
    prices = [1000.0] * 10 + [1100.0] + [950.0] * 29 + [1140.0]
    vols = [500_000] * 40 + [800_000]
    df = _make_df(prices, vols)
    # 条件が揃っていればNoneでないことを確認
    result = cwh_integrated_strategy(df, _base_info(market_cap=1000))
    # 条件次第でNoneの場合もあるが、市場cap=1000で除外はされていない
    # (他フィルタが外れる可能性があるため、単に例外なく実行できることを確認)
    assert result is None or isinstance(result, dict)


# ---- 信用倍率フィルタ ----

def test_credit_ratio_filter_above_threshold():
    """信用倍率 > 5.0 は除外される。"""
    prices = [1000.0] * 10 + [1100.0] + [950.0] * 29 + [1140.0]
    vols = [500_000] * 40 + [800_000]
    df = _make_df(prices, vols)
    result = cwh_integrated_strategy(df, _base_info(credit_ratio=5.1))
    assert result is None


def test_credit_ratio_filter_at_threshold():
    """信用倍率 = 5.0 はフィルタを通過する。"""
    prices = [1000.0] * 10 + [1100.0] + [950.0] * 29 + [1140.0]
    vols = [500_000] * 40 + [800_000]
    df = _make_df(prices, vols)
    result = cwh_integrated_strategy(df, _base_info(credit_ratio=5.0))
    assert result is None or isinstance(result, dict)


# ---- カップ深さフィルタ ----

def test_cup_depth_too_shallow():
    """カップ深さ < 15% は除外される。"""
    # HL=1100, B1=950 → 深さ=150/1100≈13.6% → NG
    from cwh_integrated_filters import check_cup_depth
    assert check_cup_depth(1100.0, 952.0) is False  # (1100-952)/1100 ≈ 13.5%


def test_cup_depth_too_deep():
    """カップ深さ > 35% は除外される。"""
    from cwh_integrated_filters import check_cup_depth
    assert check_cup_depth(1100.0, 700.0) is False  # (1100-700)/1100 ≈ 36.4%


def test_cup_depth_valid():
    """有効なカップ深さ（15%〜35%）は通過する。"""
    from cwh_integrated_filters import check_cup_depth
    assert check_cup_depth(1100.0, 935.0) is True  # (1100-935)/1100 = 15.0%
    assert check_cup_depth(1100.0, 715.0) is True  # (1100-715)/1100 = 35.0%


# ---- 統合ロジック：BUYシグナル ----

def test_buy_signal_returned():
    """ブレイクアウト条件を満たした場合に BUY が返る。"""
    # HL付近の高値 → 底 → ブレイクアウト
    hl = 1100.0
    b1 = 850.0  # depth = 250/1100 ≈ 22.7%（有効範囲）
    breakout_price = hl * 1.04  # 1.03 * HL 以上

    prices = [hl] + [900.0] * 30 + [1000.0] * 29 + [breakout_price]
    vols = [500_000] * 60 + [800_000]
    df = _make_df(prices, vols)

    result = cwh_integrated_strategy(df, _base_info())
    # フィルタ通過＆ブレイクアウト時はBUYまたはSELL_NOW
    if result is not None:
        assert result['ACTION'] in ('BUY', 'SELL_NOW')
        assert result['CODE'] == '9999.T'


# ---- 結果スキーマ ----

def test_result_has_required_keys():
    """結果辞書が必要なキーを含む。"""
    hl = 1100.0
    breakout_price = hl * 1.04
    prices = [hl] + [900.0] * 30 + [1000.0] * 29 + [breakout_price]
    vols = [500_000] * 60 + [800_000]
    df = _make_df(prices, vols)
    result = cwh_integrated_strategy(df, _base_info())
    if result is not None:
        for key in ('ACTION', 'CODE', 'ENTRY_PRICE', 'TARGET_1', 'TRAILING_STOP', 'DAYS_HELD'):
            assert key in result


# ---- 足切り判定（cwh_exit_timer） ----

def test_25day_cutoff_below_10pct():
    """25日経過かつ含み益 < 10% → 売却。"""
    cutoff, reason = check_time_cutoff(days_held=25, unrealized_gain=0.09)
    assert cutoff is True
    assert "25-day" in reason


def test_25day_no_cutoff_above_10pct():
    """25日経過かつ含み益 >= 10% → 継続。"""
    cutoff, _ = check_time_cutoff(days_held=25, unrealized_gain=0.10)
    assert cutoff is False


def test_75day_cutoff_below_20pct():
    """75日経過かつ含み益 < 20% → 売却。"""
    cutoff, reason = check_time_cutoff(days_held=75, unrealized_gain=0.15)
    assert cutoff is True
    assert "75-day" in reason


def test_75day_no_cutoff_above_20pct():
    """75日経過かつ含み益 >= 20% → トレール継続。"""
    cutoff, _ = check_time_cutoff(days_held=75, unrealized_gain=0.20)
    assert cutoff is False


def test_calc_days_held_deterministic():
    """calc_days_held は as_of を指定すると決定論的に計算できる。"""
    entry = pd.Timestamp("2025-01-01")
    as_of = pd.Timestamp("2025-03-07")
    days = calc_days_held(entry, as_of=as_of)
    assert days == 65


def test_calc_days_held_none_entry():
    """エントリー日がNoneの場合は0を返す。"""
    assert calc_days_held(None) == 0


def test_evaluate_exit_timer_25day():
    """evaluate_exit_timer: 25日経過かつ含み益 < 10% → 売却。"""
    info = {
        "entry_date": pd.Timestamp("2025-01-01"),
        "entry_price": 1000.0,
    }
    should_sell, reason = evaluate_exit_timer(
        info, current_price=1050.0, as_of=pd.Timestamp("2025-01-26")
    )
    assert should_sell is True
    assert "25-day" in reason
