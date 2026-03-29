"""
cwh_sbi_order_generator.py - SBI証券注文生成モジュール

エントリー価格からストップロス・テイクプロフィット・トレーリングストップの
具体的注文数値を生成し、SBI証券の「OCO注文」「トレール注文」形式で出力する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SBIOrderSet:
    """SBI証券への注文セット。"""

    ticker: str
    entry_price: float
    stop_loss_price: float     # OCO ストップ価格
    take_profit_price: float   # OCO リミット価格
    trail_width_pct: float     # トレール幅 [%]
    trail_trigger_price: float # トレール注文発動価格（+5% 利益時）

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "entry_price": round(self.entry_price, 0),
            "stop_loss_price": round(self.stop_loss_price, 0),
            "take_profit_price": round(self.take_profit_price, 0),
            "trail_width_pct": round(self.trail_width_pct, 1),
            "trail_trigger_price": round(self.trail_trigger_price, 0),
        }

    def format_text(self) -> str:
        """人間が読みやすいテキスト形式に変換する。"""
        return (
            f"[{self.ticker}]\n"
            f"  エントリー価格      : {self.entry_price:,.0f} 円\n"
            f"  ---\n"
            f"  【OCO注文】\n"
            f"    ストップ（-8%）   : {self.stop_loss_price:,.0f} 円\n"
            f"    リミット（+20%）  : {self.take_profit_price:,.0f} 円\n"
            f"  ---\n"
            f"  【トレール注文】\n"
            f"    発動価格（+5%）   : {self.trail_trigger_price:,.0f} 円\n"
            f"    トレール幅        : {self.trail_width_pct:.1f}%\n"
        )


def generate_order(
    ticker: str,
    entry_price: float,
    cfg: dict,
) -> SBIOrderSet:
    """エントリー価格から SBI 注文セットを生成する。

    Args:
        ticker: ティッカーシンボル
        entry_price: エントリー価格（円）
        cfg: cwh_config.yaml の全設定（辞書）

    Returns:
        SBIOrderSet
    """
    backtest_cfg = cfg.get("backtest", {})
    sbi_cfg = cfg.get("sbi_integration", {})

    stop_loss_ratio: float = backtest_cfg.get("stop_loss_ratio", 0.92)
    take_profit_ratio: float = backtest_cfg.get("take_profit_ratio", 1.20)
    ts_min_gain: float = backtest_cfg.get("trailing_stop_min_gain", 0.05)
    trail_width_pct: float = sbi_cfg.get("trail_width_percent", 3.0)

    return SBIOrderSet(
        ticker=ticker,
        entry_price=entry_price,
        stop_loss_price=entry_price * stop_loss_ratio,      # PSL = Pentry * 0.92
        take_profit_price=entry_price * take_profit_ratio,  # PTP = Pentry * 1.20
        trail_width_pct=trail_width_pct,
        trail_trigger_price=entry_price * (1 + ts_min_gain),  # +5% で発動
    )


def generate_orders_bulk(
    entries: list[tuple[str, float]],
    cfg: dict,
) -> list[SBIOrderSet]:
    """複数銘柄の注文セットを一括生成する。

    Args:
        entries: (ticker, entry_price) のリスト
        cfg: cwh_config.yaml の全設定

    Returns:
        SBIOrderSet のリスト
    """
    return [generate_order(ticker, price, cfg) for ticker, price in entries]


def print_orders(orders: list[SBIOrderSet]) -> None:
    """注文セットを標準出力に表示する。"""
    print("=" * 50)
    print("SBI証券 注文指示書")
    print("=" * 50)
    for order in orders:
        print(order.format_text())
