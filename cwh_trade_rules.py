"""
cwh_trade_rules.py - トレードルール実装

エントリー・ストップロス・トレーリングストップ・テイクプロフィット・
ピーク検知・足切りロジックを実装する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """出口事由（優先順位順）。"""

    STOP_LOSS = "SL"           # ストップロス
    TRAILING_STOP = "TS"       # トレーリングストップ
    TAKE_PROFIT = "TP"         # テイクプロフィット
    PEAK_WICK = "PEAK_WICK"    # 大商い上ヒゲ
    PEAK_GAP = "PEAK_GAP"      # 窓開け陰線
    CUTOFF_25D = "CUT25"       # 25日足切り
    CUTOFF_75D = "CUT75"       # 75日足切り
    END_OF_PERIOD = "EoP"      # 保有期限（60営業日）


@dataclass
class TradeRecord:
    """1 トレードの記録。"""

    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    # 経路中の最高値
    pmax: float = field(init=False)

    def __post_init__(self) -> None:
        self.pmax = self.entry_price

    @property
    def pnl_pct(self) -> Optional[float]:
        """損益率 [%]。ri = (Pexit - Pentry) / Pentry * 100"""
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100.0

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None


def compute_entry_signal(
    df: pd.DataFrame,
    pivot_bar: int,
    pivot_price: float,
    cfg: dict,
) -> Optional[int]:
    """エントリーシグナルを判定し、エントリーバーのインデックスを返す。

    エントリー条件:
      Pt >= 1.03 * H_L（ハンドル高値）かつ Vt >= 1.5 * V_25

    Args:
        df: OHLCV DataFrame
        pivot_bar: ピボットバーのインデックス（PV）
        pivot_price: ハンドル高値（右リム高値）
        cfg: cwh_config.yaml の backtest セクション

    Returns:
        エントリーバーのインデックス。シグナルなしは None。
    """
    entry_threshold = pivot_price * (1 + cfg["entry_price_margin"])
    vol_period: int = cfg["entry_volume_period"]
    vol_ratio: float = cfg["entry_volume_ratio"]

    for i in range(pivot_bar, min(pivot_bar + 10, len(df))):
        vol_avg = df["Volume"].iloc[max(0, i - vol_period):i].mean()
        if df["High"].iloc[i] >= entry_threshold and df["Volume"].iloc[i] >= vol_ratio * vol_avg:
            return i
    return None


def apply_exit_rules(
    df: pd.DataFrame,
    entry_bar: int,
    entry_price: float,
    cfg: dict,
) -> tuple[int, float, ExitReason]:
    """エントリー後の出口バーと出口価格を決定する。

    出口優先順位：
      1. ストップロス（SL）
      2. トレーリングストップ（TS）
      3. テイクプロフィット（TP）
      4. ピーク検知（大商い上ヒゲ・窓開け陰線）
      5. 25日足切り
      6. 75日足切り
      7. 保有期限（60営業日）

    Args:
        df: OHLCV DataFrame
        entry_bar: エントリーバーのインデックス
        entry_price: エントリー価格
        cfg: cwh_config.yaml の backtest セクション

    Returns:
        (exit_bar, exit_price, exit_reason)
    """
    psl: float = entry_price * cfg["stop_loss_ratio"]              # PSL = Pentry * 0.92
    ptp: float = entry_price * cfg["take_profit_ratio"]            # PTP = Pentry * 1.20
    trailing_ratio: float = cfg["trailing_stop_ratio"]             # 0.92
    ts_min_gain: float = cfg["trailing_stop_min_gain"]             # +5% で発動
    max_hold: int = cfg["max_hold_bars"]                           # 60 営業日
    peak_vol_ratio: float = cfg["peak_detection_volume_ratio"]     # 2.0
    cutoff_25d_gain: float = cfg["cutoff_25days_gain_min"]         # 0.10

    pmax: float = entry_price
    vol_period_peak: int = 25

    for i in range(entry_bar + 1, min(entry_bar + max(max_hold, 75) + 1, len(df))):
        bar = i - entry_bar  # 経過営業日数
        low = df["Low"].iloc[i]
        high = df["High"].iloc[i]
        close = df["Close"].iloc[i]
        open_ = df["Open"].iloc[i]

        # 最高値を更新
        if close > pmax:
            pmax = close

        # 1. ストップロス
        if low <= psl:
            return i, psl, ExitReason.STOP_LOSS

        # 2. トレーリングストップ（最小利益 +5% 発動）
        if cfg.get("trailing_stop_enabled", True):
            pts: float = pmax * trailing_ratio
            if pmax >= entry_price * (1 + ts_min_gain) and low <= pts:
                return i, pts, ExitReason.TRAILING_STOP

        # 3. テイクプロフィット
        if high >= ptp:
            return i, ptp, ExitReason.TAKE_PROFIT

        # 4a. ピーク検知：大商い上ヒゲ
        vol_avg_peak = df["Volume"].iloc[max(0, i - vol_period_peak):i].mean()
        body = abs(close - open_)
        upper_wick = high - max(close, open_)
        if (df["Volume"].iloc[i] >= peak_vol_ratio * vol_avg_peak
                and upper_wick > body):
            return i, close, ExitReason.PEAK_WICK

        # 4b. ピーク検知：窓開け陰線（始値を終値が下回る）
        if i > entry_bar + 1 and close < open_ and close < df["Close"].iloc[i - 1]:
            return i, close, ExitReason.PEAK_GAP

        # 5. 25日足切り
        if bar == 25:
            gain_25 = (close - entry_price) / entry_price
            if gain_25 < cutoff_25d_gain:
                return i, close, ExitReason.CUTOFF_25D

        # 6. 75日足切り
        if cfg.get("cutoff_75days_enabled", True) and bar >= 75:
            return i, close, ExitReason.CUTOFF_75D

        # 7. 保有期限（60 営業日）
        if bar >= max_hold:
            return i, close, ExitReason.END_OF_PERIOD

    # データ末尾
    last = min(entry_bar + max_hold, len(df) - 1)
    return last, df["Close"].iloc[last], ExitReason.END_OF_PERIOD
