"""
tests/test_split_tp.py

分割TP機能付き出口判定関数 check_exit_with_split_tp のテスト

テスト設計の前提:
  buy_price = 1000
  SL  = 920  (-8%)
  TP1 = 1150  (+15%、厳密超過 c > 1150 で発動)
  TP2 = 1280  (+28%、c >= 1280 で発動)

優先順位: SL > TP2 > TP1（TP1超過） > TrailingStop
"""

import pytest
from cwh_cnh_v30 import check_exit_with_split_tp


def test_split_tp_tp1_partial_exit():
    """
    TP1 (+15%) 超過で50%売却をテスト。

    prices が TP1 ライン（1150）を厳密に超過（>）し、
    かつ TP2 ライン（1280）には到達しない価格列を使用。
    → TP1 で即時リターン、残ポジション50%。
    """
    buy_price = 1000
    # 1200 > tp1(1150) かつ 1200 < tp2(1280) → TP1 発動
    prices = [1000, 1050, 1150, 1200, 1250]
    lows   = [1000, 1050, 1150, 1200, 1250]
    cfg = {}

    exit_flag, reason, price, idx, qty = check_exit_with_split_tp(
        prices, lows, buy_price, cfg, 100
    )

    assert exit_flag is True
    assert reason == "TP1(+15%)"
    assert qty == 50  # 残50%


def test_split_tp_tp2_full_exit():
    """
    TP2 (+28%) で全売却をテスト。

    TP1 ライン（1150）をちょうど踏む価格（1150 == tp1、厳密超過にならない）が
    先に来た後、TP2 ライン（1280）到達で全売却。
    → TP2 でリターン、残ポジション0%。
    """
    buy_price = 1000
    # 1150 は c > tp1 の厳密超過に該当しないため TP1 スキップ
    # 1280 >= tp2 → TP2 発動
    prices = [1000, 1150, 1280, 1300, 1400]
    lows   = [1000, 1150, 1280, 1300, 1400]
    cfg = {}

    exit_flag, reason, price, idx, qty = check_exit_with_split_tp(
        prices, lows, buy_price, cfg, 100
    )

    assert exit_flag is True
    assert reason == "TP2(+28%)"
    assert qty == 0  # 全売却


def test_sl_priority_over_tp():
    """
    SL (-8%) が TP2 よりも優先されることをテスト。

    安値が SL ラインを下回った時点で即時全売却。
    その後の価格（TP2 レベル）は評価されない。
    """
    buy_price = 1000
    prices = [1000, 900, 1280]
    lows   = [1000, 900, 1280]
    cfg = {}

    exit_flag, reason, price, idx, qty = check_exit_with_split_tp(
        prices, lows, buy_price, cfg, 100
    )

    assert exit_flag is True
    assert reason == "SL(-8%)"
    assert qty == 0  # 全売却


def test_no_exit_within_hold_period():
    """
    保有期間内に出口条件を満たさない場合、exit_flag=False を返すことをテスト。
    """
    buy_price = 1000
    prices = [1000, 1010, 1020, 1030, 1040]
    lows   = [990,  1005, 1015, 1025, 1035]
    cfg = {}

    exit_flag, reason, price, idx, qty = check_exit_with_split_tp(
        prices, lows, buy_price, cfg, 100
    )

    assert exit_flag is False
    assert reason is None
    assert qty == 100  # ポジション変化なし


def test_trailing_stop_after_gain():
    """
    利益+5%以上の後、トレーリングストップが発動することをテスト。

    max_p = 1100 (buy_price * 1.10 > 1.05)
    ts_p  = 1100 * 0.92 = 1012
    安値が ts_p 以下になった場合に TrailingStop 発動。
    """
    buy_price = 1000
    # 1100 まで上昇後、安値 1010 (ts_p=1012 以下) でトレーリングストップ
    prices = [1000, 1050, 1100, 1080, 1090]
    lows   = [1000, 1040, 1090, 1010, 1080]
    cfg = {}

    exit_flag, reason, price, idx, qty = check_exit_with_split_tp(
        prices, lows, buy_price, cfg, 100
    )

    assert exit_flag is True
    assert reason == "TrailingStop"
    assert qty == 0
