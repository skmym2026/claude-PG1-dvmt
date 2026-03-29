"""
cwh_performance_metrics.py - パフォーマンス指標計算モジュール

損益率・勝率・累積リターン（複利）・最大ドローダウン・プロフィットファクターを計算する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PerformanceMetrics:
    """バックテスト結果のパフォーマンス指標。"""

    n_trades: int            # 総トレード数
    win_rate: float          # 勝率 [%]
    cum_return: float        # 累積リターン（複利）[%]
    max_drawdown: float      # 最大ドローダウン [%]
    profit_factor: float     # プロフィットファクター
    avg_win: float           # 平均利益 [%]
    avg_loss: float          # 平均損失 [%]

    def to_dict(self) -> dict[str, float]:
        return {
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "cum_return": self.cum_return,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


def calc_win_rate(pnl_list: Sequence[float]) -> float:
    """勝率を計算する。

    WinRate = #{i: ri > 0} / N * 100 [%]

    Args:
        pnl_list: 各トレードの損益率リスト [%]

    Returns:
        勝率 [%]
    """
    if not pnl_list:
        return 0.0
    wins = sum(1 for r in pnl_list if r > 0)
    return wins / len(pnl_list) * 100.0


def calc_cumulative_return(pnl_list: Sequence[float]) -> float:
    """複利累積リターンを計算する。

    Rcum = (prod(1 + ri/100) - 1) * 100 [%]

    Args:
        pnl_list: 各トレードの損益率リスト [%]

    Returns:
        累積リターン [%]
    """
    if not pnl_list:
        return 0.0
    product = math.prod(1.0 + r / 100.0 for r in pnl_list)
    return (product - 1.0) * 100.0


def calc_max_drawdown(equity_curve: Sequence[float]) -> float:
    """最大ドローダウンを計算する。

    MDD = max_{j <= k} (Ej - Ek) / Ej * 100 [%]

    Args:
        equity_curve: エクイティカーブ（累積資産の推移）

    Returns:
        最大ドローダウン [%]
    """
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve, dtype=float)
    peak = arr[0]
    mdd = 0.0
    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def calc_profit_factor(pnl_list: Sequence[float]) -> float:
    """プロフィットファクターを計算する。

    PF = sum(ri > 0) / |sum(ri <= 0)|

    Args:
        pnl_list: 各トレードの損益率リスト [%]

    Returns:
        プロフィットファクター（損失ゼロの場合は inf）
    """
    gross_profit = sum(r for r in pnl_list if r > 0)
    gross_loss = abs(sum(r for r in pnl_list if r <= 0))
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def build_equity_curve(pnl_list: Sequence[float], initial: float = 1.0) -> list[float]:
    """エクイティカーブを生成する。

    Args:
        pnl_list: 各トレードの損益率リスト [%]
        initial: 初期資産（デフォルト 1.0）

    Returns:
        エクイティカーブのリスト
    """
    equity = [initial]
    current = initial
    for r in pnl_list:
        current *= 1.0 + r / 100.0
        equity.append(current)
    return equity


def compute_metrics(pnl_list: Sequence[float]) -> PerformanceMetrics:
    """全パフォーマンス指標を一括計算する。

    Args:
        pnl_list: 各トレードの損益率リスト [%]

    Returns:
        PerformanceMetrics
    """
    wins = [r for r in pnl_list if r > 0]
    losses = [r for r in pnl_list if r <= 0]
    equity = build_equity_curve(pnl_list)

    return PerformanceMetrics(
        n_trades=len(pnl_list),
        win_rate=calc_win_rate(pnl_list),
        cum_return=calc_cumulative_return(pnl_list),
        max_drawdown=calc_max_drawdown(equity),
        profit_factor=calc_profit_factor(pnl_list),
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
    )
