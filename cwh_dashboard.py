"""
cwh_dashboard.py - Streamlit ダッシュボード

バックテスト結果のインタラクティブ可視化ダッシュボード。
起動: streamlit run cwh_dashboard.py
"""

from __future__ import annotations

import os

import pandas as pd
import yaml

# Streamlit はオプション依存のため遅延インポート
try:
    import streamlit as st
    import plotly.graph_objects as go
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


def load_config(config_path: str = "cwh_config.yaml") -> dict:
    """設定ファイルを読み込む。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(csv_path: str = "backtest_results.csv") -> pd.DataFrame:
    """バックテスト結果 CSV を読み込む。"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path, parse_dates=["entry_date", "exit_date"])


def build_equity_curve_df(df: pd.DataFrame) -> pd.DataFrame:
    """トレード一覧からエクイティカーブを生成する。"""
    if df.empty or "pnl_pct" not in df.columns:
        return pd.DataFrame()
    df_sorted = df.sort_values("entry_date").copy()
    equity = [1.0]
    for pnl in df_sorted["pnl_pct"].fillna(0):
        equity.append(equity[-1] * (1 + pnl / 100.0))
    dates = [df_sorted["entry_date"].iloc[0]] + list(df_sorted["exit_date"])
    return pd.DataFrame({"date": dates, "equity": equity})


def main() -> None:
    """ダッシュボードのメイン関数。"""
    if not _STREAMLIT_AVAILABLE:
        print("streamlit がインストールされていません。pip install streamlit を実行してください。")
        return

    st.set_page_config(page_title="CWH バックテスト v3.0", layout="wide")
    st.title("CWH 統合型バックテストシステム v3.0")

    cfg = load_config()
    results_df = load_results()

    # サイドバー
    st.sidebar.header("設定")
    st.sidebar.write(f"バージョン: {cfg.get('system', {}).get('version', 'N/A')}")

    if results_df.empty:
        st.warning("バックテスト結果が見つかりません。cwh_v30_main.py を先に実行してください。")
        return

    # 概要指標
    col1, col2, col3, col4 = st.columns(4)
    n_trades = len(results_df)
    win_rate = (results_df["pnl_pct"] > 0).mean() * 100 if n_trades > 0 else 0
    avg_pnl = results_df["pnl_pct"].mean() if n_trades > 0 else 0

    col1.metric("トレード数", n_trades)
    col2.metric("勝率", f"{win_rate:.1f}%")
    col3.metric("平均損益率", f"{avg_pnl:.1f}%")
    col4.metric("銘柄数", results_df["ticker"].nunique())

    # エクイティカーブ
    st.subheader("エクイティカーブ")
    eq_df = build_equity_curve_df(results_df)
    if not eq_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines", name="Equity"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Equity (initial=1.0)")
        st.plotly_chart(fig, use_container_width=True)

    # トレード一覧
    st.subheader("トレード一覧")
    st.dataframe(results_df.sort_values("entry_date", ascending=False))

    # 銘柄別パフォーマンス
    st.subheader("銘柄別パフォーマンス")
    if "ticker" in results_df.columns:
        def _win_rate(x: pd.Series) -> float:
            return (x > 0).mean() * 100

        summary = (
            results_df.groupby("ticker")["pnl_pct"]
            .agg(count="count", mean="mean", win_rate=_win_rate)
            .rename(columns={"count": "トレード数", "mean": "平均損益率[%]", "win_rate": "勝率[%]"})
        )
        st.dataframe(summary)


if __name__ == "__main__":
    main()
