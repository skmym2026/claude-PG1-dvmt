"""
cwh_sbi_order_output.py
SBI証券注文フォーマット出力モジュール
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def format_sbi_order(strategy_result: Optional[Dict]) -> Optional[Dict]:
    """
    統合CWH戦略の結果をSBI証券向け注文フォーマットに変換する。

    Returns:
        SBI注文辞書、またはNone（入力がNoneの場合）
    """
    if strategy_result is None:
        return None

    action = strategy_result.get('ACTION')

    if action == "SELL_NOW":
        return {
            "ORDER_TYPE": "SELL_AT_MARKET",
            "TICKER": strategy_result['CODE'],
            "REASON": strategy_result.get('REASON', 'Peak detected'),
        }

    if action == "BUY":
        entry_price: float = strategy_result['ENTRY_PRICE']
        time_limit: str = (datetime.now() + timedelta(days=25)).strftime("%Y-%m-%d")

        return {
            "ORDER_TYPE": "OCO",
            "TICKER": strategy_result['CODE'],
            "SIDE": "BUY",
            "ENTRY_PRICE": entry_price,
            "PROFIT_TARGET": strategy_result['TARGET_1'],
            "PROFIT_TARGET_TYPE": "LIMIT",
            "STOP_LOSS": strategy_result['TRAILING_STOP'],
            "STOP_LOSS_TYPE": "STOP",
            "TIME_LIMIT": time_limit,
            "TRAIL_WIDTH_PERCENT": 3.0,
            "NOTES": (
                f"MC={strategy_result.get('MARKET_CAP', 'N/A')}B, "
                f"CR={strategy_result.get('CREDIT_RATIO', 0):.1f}"
            ),
        }

    return None


def save_orders_to_csv(orders: List[Dict], output_path: str = "output/sbi_orders.csv") -> None:
    """
    注文リストをCSVファイルに書き出す（日次出力）。

    Args:
        orders: format_sbi_order() の結果リスト
        output_path: 出力先CSVパス
    """
    valid_orders = [o for o in orders if o is not None]
    if not valid_orders:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list({key for order in valid_orders for key in order.keys()})
    fieldnames.sort()

    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for order in valid_orders:
            writer.writerow(order)
