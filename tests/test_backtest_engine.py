"""
tests/test_backtest_engine.py - バックテストエンジンのユニットテスト

pytest で実行: pytest tests/test_backtest_engine.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from cwh_performance_metrics import (
    calc_cumulative_return,
    calc_max_drawdown,
    calc_profit_factor,
    calc_win_rate,
    build_equity_curve,
    compute_metrics,
)
from cwh_sbi_order_generator import SBIOrderSet, generate_order


# ============================
# フィクスチャ
# ============================

@pytest.fixture()
def sample_cfg() -> dict:
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


# ============================
# テスト：calc_win_rate
# ============================

class TestCalcWinRate:
    def test_all_wins(self) -> None:
        assert calc_win_rate([10.0, 20.0, 5.0]) == pytest.approx(100.0)

    def test_all_losses(self) -> None:
        assert calc_win_rate([-5.0, -10.0]) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        # 3 勝 1 敗 = 75%
        assert calc_win_rate([10.0, -5.0, 20.0, 15.0]) == pytest.approx(75.0)

    def test_empty(self) -> None:
        assert calc_win_rate([]) == pytest.approx(0.0)

    def test_zero_pnl_is_not_win(self) -> None:
        assert calc_win_rate([0.0, 10.0]) == pytest.approx(50.0)


# ============================
# テスト：calc_cumulative_return
# ============================

class TestCalcCumulativeReturn:
    def test_single_trade(self) -> None:
        # +20% のみ
        assert calc_cumulative_return([20.0]) == pytest.approx(20.0)

    def test_two_trades_compounding(self) -> None:
        # +10% → +10% = (1.1 * 1.1 - 1) * 100 = 21%
        assert calc_cumulative_return([10.0, 10.0]) == pytest.approx(21.0)

    def test_gain_then_loss(self) -> None:
        # +25% → -8% = (1.25 * 0.92 - 1) * 100 = 15%
        assert calc_cumulative_return([25.0, -8.0]) == pytest.approx(15.0, rel=1e-3)

    def test_empty(self) -> None:
        assert calc_cumulative_return([]) == pytest.approx(0.0)


# ============================
# テスト：calc_max_drawdown
# ============================

class TestCalcMaxDrawdown:
    def test_no_drawdown(self) -> None:
        # 一直線に上昇
        eq = [1.0, 1.1, 1.2, 1.3]
        assert calc_max_drawdown(eq) == pytest.approx(0.0)

    def test_simple_drawdown(self) -> None:
        # 1.0 → 2.0（ピーク）→ 1.0 = 50% ドローダウン
        eq = [1.0, 2.0, 1.0]
        assert calc_max_drawdown(eq) == pytest.approx(50.0)

    def test_multiple_drawdowns(self) -> None:
        eq = [1.0, 1.5, 1.0, 2.0, 1.2]
        # ピーク 2.0 → 1.2 = 40% が最大
        assert calc_max_drawdown(eq) == pytest.approx(40.0)

    def test_empty(self) -> None:
        assert calc_max_drawdown([]) == pytest.approx(0.0)


# ============================
# テスト：calc_profit_factor
# ============================

class TestCalcProfitFactor:
    def test_no_loss(self) -> None:
        assert calc_profit_factor([10.0, 20.0]) == float("inf")

    def test_equal_profit_loss(self) -> None:
        assert calc_profit_factor([10.0, -10.0]) == pytest.approx(1.0)

    def test_standard(self) -> None:
        # 利益 30、損失 10 => PF = 3.0
        assert calc_profit_factor([10.0, 20.0, -10.0]) == pytest.approx(3.0)

    def test_all_losses(self) -> None:
        assert calc_profit_factor([-5.0, -10.0]) == pytest.approx(0.0)

    def test_empty(self) -> None:
        assert calc_profit_factor([]) == float("inf")


# ============================
# テスト：build_equity_curve
# ============================

class TestBuildEquityCurve:
    def test_initial_value(self) -> None:
        eq = build_equity_curve([], initial=1.0)
        assert eq == [1.0]

    def test_growth(self) -> None:
        eq = build_equity_curve([100.0], initial=1.0)  # +100%
        assert eq[-1] == pytest.approx(2.0)

    def test_length(self) -> None:
        pnl = [10.0, -5.0, 20.0]
        eq = build_equity_curve(pnl)
        assert len(eq) == len(pnl) + 1


# ============================
# テスト：compute_metrics
# ============================

class TestComputeMetrics:
    def test_basic(self) -> None:
        pnl = [20.0, -8.0, 17.7, 19.1]
        m = compute_metrics(pnl)
        assert m.n_trades == 4
        assert m.win_rate == pytest.approx(75.0)
        assert m.cum_return > 0

    def test_empty(self) -> None:
        m = compute_metrics([])
        assert m.n_trades == 0
        assert m.win_rate == pytest.approx(0.0)
        assert m.cum_return == pytest.approx(0.0)


# ============================
# テスト：SBI 注文生成
# ============================

class TestGenerateOrder:
    def test_stop_loss(self, sample_cfg: dict) -> None:
        order = generate_order("5253.T", 1000.0, sample_cfg)
        assert order.stop_loss_price == pytest.approx(920.0)

    def test_take_profit(self, sample_cfg: dict) -> None:
        order = generate_order("5253.T", 1000.0, sample_cfg)
        assert order.take_profit_price == pytest.approx(1200.0)

    def test_trail_trigger(self, sample_cfg: dict) -> None:
        order = generate_order("5253.T", 1000.0, sample_cfg)
        assert order.trail_trigger_price == pytest.approx(1050.0)

    def test_trail_width(self, sample_cfg: dict) -> None:
        order = generate_order("5253.T", 1000.0, sample_cfg)
        assert order.trail_width_pct == pytest.approx(3.0)

    def test_to_dict_keys(self, sample_cfg: dict) -> None:
        order = generate_order("5253.T", 1000.0, sample_cfg)
        d = order.to_dict()
        expected_keys = {
            "ticker", "entry_price", "stop_loss_price",
            "take_profit_price", "trail_width_pct", "trail_trigger_price",
        }
        assert set(d.keys()) == expected_keys
