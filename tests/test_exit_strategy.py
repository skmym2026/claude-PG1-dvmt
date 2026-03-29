"""
tests/test_exit_strategy.py - 出口戦略のユニットテスト

エントリー条件・ストップロス・トレーリングストップ・テイクプロフィット・
ピーク検知・足切りの各ロジックをテストする。

pytest で実行: pytest tests/test_exit_strategy.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from cwh_trade_rules import (
    ExitReason,
    TradeRecord,
    apply_exit_rules,
    compute_entry_signal,
)


# ============================
# フィクスチャ
# ============================

@pytest.fixture()
def exit_cfg() -> dict:
    """出口ルール用テスト設定。"""
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


def _make_flat_df(n: int = 120, price: float = 1000.0, volume: float = 1000.0) -> pd.DataFrame:
    """横ばいの OHLCV DataFrame を生成する。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price * 1.005] * n,
            "Low": [price * 0.995] * n,
            "Close": [price] * n,
            "Volume": [volume] * n,
        },
        index=dates,
    )


def _make_drop_df(n: int = 120, entry_price: float = 1000.0, drop_at: int = 5) -> pd.DataFrame:
    """指定バーで SL を割り込む DataFrame を生成する。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.full(n, entry_price)
    # drop_at バー目でストップロス価格を割り込む Low
    lows = np.full(n, entry_price * 0.995)
    lows[drop_at] = entry_price * 0.85  # SL を割り込む

    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.005,
            "Low": lows,
            "Close": prices,
            "Volume": np.full(n, 1000.0),
        },
        index=dates,
    )


def _make_rise_df(n: int = 120, entry_price: float = 1000.0, rise_at: int = 5) -> pd.DataFrame:
    """指定バーで TP を超える DataFrame を生成する。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.full(n, entry_price)
    highs = np.full(n, entry_price * 1.005)
    highs[rise_at] = entry_price * 1.25  # TP を超える

    return pd.DataFrame(
        {
            "Open": prices,
            "High": highs,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.full(n, 1000.0),
        },
        index=dates,
    )


# ============================
# テスト：TradeRecord
# ============================

class TestTradeRecord:
    def test_pnl_pct_win(self) -> None:
        trade = TradeRecord(ticker="TEST", entry_date=pd.Timestamp("2024-01-01"), entry_price=1000.0)
        trade.exit_price = 1200.0
        trade.exit_date = pd.Timestamp("2024-03-01")
        trade.exit_reason = ExitReason.TAKE_PROFIT
        assert trade.pnl_pct == pytest.approx(20.0)

    def test_pnl_pct_loss(self) -> None:
        trade = TradeRecord(ticker="TEST", entry_date=pd.Timestamp("2024-01-01"), entry_price=1000.0)
        trade.exit_price = 920.0
        trade.exit_date = pd.Timestamp("2024-02-01")
        trade.exit_reason = ExitReason.STOP_LOSS
        assert trade.pnl_pct == pytest.approx(-8.0)

    def test_pnl_pct_none_if_not_closed(self) -> None:
        trade = TradeRecord(ticker="TEST", entry_date=pd.Timestamp("2024-01-01"), entry_price=1000.0)
        assert trade.pnl_pct is None

    def test_is_closed(self) -> None:
        trade = TradeRecord(ticker="TEST", entry_date=pd.Timestamp("2024-01-01"), entry_price=1000.0)
        assert trade.is_closed is False
        trade.exit_date = pd.Timestamp("2024-02-01")
        trade.exit_price = 1000.0
        assert trade.is_closed is True

    def test_pmax_initialized_to_entry(self) -> None:
        trade = TradeRecord(ticker="TEST", entry_date=pd.Timestamp("2024-01-01"), entry_price=1500.0)
        assert trade.pmax == pytest.approx(1500.0)


# ============================
# テスト：apply_exit_rules
# ============================

class TestApplyExitRules:
    def test_stop_loss_triggered(self, exit_cfg: dict) -> None:
        df = _make_drop_df(drop_at=5)
        entry_bar = 0
        entry_price = 1000.0
        _, exit_price, reason = apply_exit_rules(df, entry_bar, entry_price, exit_cfg)
        assert reason == ExitReason.STOP_LOSS
        assert exit_price == pytest.approx(entry_price * exit_cfg["stop_loss_ratio"])

    def test_take_profit_triggered(self, exit_cfg: dict) -> None:
        df = _make_rise_df(rise_at=10)
        entry_bar = 0
        entry_price = 1000.0
        _, exit_price, reason = apply_exit_rules(df, entry_bar, entry_price, exit_cfg)
        assert reason == ExitReason.TAKE_PROFIT
        assert exit_price == pytest.approx(entry_price * exit_cfg["take_profit_ratio"])

    def test_end_of_period(self, exit_cfg: dict) -> None:
        """横ばいで SL/TP/TS/ピーク検知なし => 保有期限切れまたは足切りで出口。"""
        df = _make_flat_df(n=120, price=1010.0)  # +1% 程度、足切り条件未達ではないが
        entry_bar = 0
        entry_price = 1000.0
        exit_bar, _, reason = apply_exit_rules(df, entry_bar, entry_price, exit_cfg)
        # 25日足切り（+10%未満）か EoP になるはず
        assert reason in {ExitReason.CUTOFF_25D, ExitReason.CUTOFF_75D, ExitReason.END_OF_PERIOD}

    def test_sl_before_tp(self, exit_cfg: dict) -> None:
        """SL が先に発動する場合は TP より SL が優先される。"""
        n = 120
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = np.full(n, 1000.0)
        highs = np.full(n, 1010.0)
        lows = np.full(n, 990.0)
        lows[3] = 850.0   # SL を割り込む（TP より先）
        highs[20] = 1300.0  # TP はバー 20（SL より後）
        df = pd.DataFrame(
            {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": np.full(n, 1000.0)},
            index=dates,
        )
        _, _, reason = apply_exit_rules(df, 0, 1000.0, exit_cfg)
        assert reason == ExitReason.STOP_LOSS


# ============================
# テスト：compute_entry_signal
# ============================

class TestComputeEntrySignal:
    def _make_entry_cfg(self) -> dict:
        return {
            "entry_price_margin": 0.03,
            "entry_volume_ratio": 1.5,
            "entry_volume_period": 25,
        }

    def test_no_signal_when_price_too_low(self) -> None:
        """価格がエントリー閾値を下回る場合はシグナルなし。"""
        cfg = self._make_entry_cfg()
        pivot_price = 1000.0
        # 価格は 1000、閾値は 1030 → シグナルなし
        n = 50
        df = _make_flat_df(n=n, price=1000.0, volume=1000.0)
        result = compute_entry_signal(df, n - 5, pivot_price, cfg)
        assert result is None

    def test_signal_when_conditions_met(self) -> None:
        """価格と出来高の条件が揃うとシグナルが返る。"""
        cfg = self._make_entry_cfg()
        pivot_price = 900.0    # 閾値 = 927
        n = 50
        prices = np.full(n, 950.0)   # 950 > 927
        volumes = np.full(n, 1000.0)
        volumes[-5:] = 2000.0        # 出来高急増
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )
        # pivot_bar = n - 6 でシグナルを探す
        pivot_bar = n - 6
        result = compute_entry_signal(df, pivot_bar, pivot_price, cfg)
        assert result is not None
        assert result >= pivot_bar
