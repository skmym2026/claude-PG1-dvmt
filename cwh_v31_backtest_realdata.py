"""
cwh_v31_backtest_realdata.py

v3.1 実データバックテスト実行スクリプト
- yfinance から実データを自動取得
- CWH / C-NH のバックテスト実行
- 修正パラメータ (max_hold=120, split_tp, rim_sym=0.15) を適用
- 結果を output/v31_backtest_results.csv に出力

実行方法:
    python cwh_v31_backtest_realdata.py
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from cwh_cnh_v30 import CWHCNHBacktest

# テスト対象銘柄
TEST_STOCKS: dict = {
    "cwh": ["5253.T", "6228.T", "5586.T", "5242.T", "5885.T", "4894.T"],
    "cnh": ["5803.T", "7011.T", "6586.T"],
}

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "v31_backtest_results.csv")


def run() -> None:
    """バックテストを実行し、結果を CSV に出力する。"""
    print("=" * 60)
    print(f"CWH/C-NH 統合バックテスト v3.1 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"max_hold_bars : 120 営業日")
    print(f"split_tp      : TP1 +15% (50%売却) / TP2 +28% (残全売却)")
    print(f"rim_sym C-NH  : 0.15（v3.1 緩和）")
    print(f"データソース  : yfinance（実データ）")
    print("-" * 60)

    backtest_engine = CWHCNHBacktest(
        config_file="cwh_config.yaml",
        start_date="2020-01-01",
        end_date="2026-03-29",
        use_real_data=True,
    )

    results = backtest_engine.run_all(TEST_STOCKS)

    # 結果サマリー表示
    print("\n=== バックテスト結果サマリー ===")
    print(f"総トレード数        : {results['total_trades']}")
    print(f"勝率                : {results['win_rate']:.1%}")
    print(f"平均リターン        : {results['avg_return']:.2%}")
    print(f"プロフィットファクタ: {results['profit_factor']:.2f}")
    print(f"最大ドローダウン    : {results['max_drawdown']:.1%}")

    # CSV 出力
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades = results.get("trades", [])
    if trades:
        fieldnames = list(trades[0].keys())
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        print(f"\n結果を {OUTPUT_FILE} に出力しました。")
    else:
        print("\nトレードが検出されませんでした。")


if __name__ == "__main__":
    run()
