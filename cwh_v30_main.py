"""
cwh_v30_main.py - CWH 統合型バックテストシステム v3.0 メインエントリーポイント

このスクリプトを実行することで、全バックテスト・スクリーニング・
レポート出力が一括実行される。
"""

from __future__ import annotations

import logging
import os
import sys

import yaml

from cwh_backtest_engine import export_trades_csv, run_backtest_all
from cwh_chart_generator import generate_all_charts
from cwh_performance_metrics import compute_metrics
from cwh_realtime_monitor import run_daily_monitor
from cwh_sbi_order_generator import generate_orders_bulk, print_orders

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cwh_v30.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "cwh_config.yaml") -> dict:
    """設定ファイルを読み込む。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def print_summary(results: list) -> None:
    """バックテスト結果のサマリーを表示する。"""
    print("\n" + "=" * 60)
    print("CWH バックテスト v3.0 - 結果サマリー")
    print("=" * 60)

    all_pnl: list[float] = []
    for result in results:
        pnl_list = [t.pnl_pct for t in result.trades if t.pnl_pct is not None]
        all_pnl.extend(pnl_list)
        m = result.metrics
        print(
            f"\n[{result.ticker}] "
            f"トレード数={m.n_trades} "
            f"勝率={m.win_rate:.1f}% "
            f"累積={m.cum_return:.1f}% "
            f"MDD={m.max_drawdown:.1f}% "
            f"PF={m.profit_factor:.2f}"
        )
        for pat in result.pattern_results:
            print(f"  パターン: {pat.pattern_type.value} ({pat.reason})")

    if all_pnl:
        total_metrics = compute_metrics(all_pnl)
        print("\n" + "-" * 60)
        print("【全銘柄統合】")
        print(f"  総トレード数  : {total_metrics.n_trades}")
        print(f"  勝率          : {total_metrics.win_rate:.1f}%")
        print(f"  累積リターン  : {total_metrics.cum_return:.1f}%")
        print(f"  最大DD        : {total_metrics.max_drawdown:.1f}%")
        print(f"  PF            : {total_metrics.profit_factor:.2f}")
        print(f"  平均利益      : {total_metrics.avg_win:.1f}%")
        print(f"  平均損失      : {total_metrics.avg_loss:.1f}%")
    print("=" * 60)


def main() -> None:
    """メイン処理。"""
    cfg = load_config()
    system_cfg = cfg.get("system", {})
    logger.info("CWH バックテストシステム v%s 開始", system_cfg.get("version", "3.0"))

    # チャートディレクトリを作成
    os.makedirs("charts", exist_ok=True)

    data_cfg = cfg.get("data", {})
    start: str = data_cfg.get("backtest_start", "2020-01-01")
    end: str = data_cfg.get("backtest_end", "2026-03-29")
    stocks_cfg = cfg.get("stocks", {})

    # メインバックテスト（一次銘柄）
    primary_tickers: list[str] = stocks_cfg.get("primary", [])
    logger.info("一次銘柄バックテスト: %s", primary_tickers)
    primary_results = run_backtest_all(primary_tickers, start, end, cfg)

    # 検証銘柄バックテスト
    validation_tickers: list[str] = stocks_cfg.get("validation", [])
    logger.info("検証銘柄バックテスト: %s", validation_tickers)
    validation_results = run_backtest_all(validation_tickers, start, end, cfg)

    all_results = primary_results + validation_results

    # 結果サマリー表示
    print_summary(all_results)

    # CSV 出力
    export_trades_csv(all_results, "backtest_results.csv")

    # チャート生成
    logger.info("チャート生成開始")
    generated_charts = generate_all_charts(all_results, output_dir="charts")
    if generated_charts:
        print(f"\n📊 生成チャート ({len(generated_charts)} 件):")
        for p in generated_charts:
            print(f"  {p}")

    # SBI 注文生成（最新トレードのエントリー価格を使用）
    entry_list: list[tuple[str, float]] = []
    for result in all_results:
        if result.trades:
            last_trade = result.trades[-1]
            entry_list.append((result.ticker, last_trade.entry_price))

    if entry_list:
        orders = generate_orders_bulk(entry_list, cfg)
        print_orders(orders)

    # リアルタイム監視（デイリー実行）
    logger.info("デイリー監視銘柄チェック開始")
    run_daily_monitor(cfg)

    logger.info("CWH バックテストシステム v%s 完了", system_cfg.get("version", "3.0"))


if __name__ == "__main__":
    main()
