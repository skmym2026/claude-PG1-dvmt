"""
cwh_chart_generator.py - チャート生成モジュール

バックテスト結果・エクイティカーブ・CWH パターンチャートを
plotly で生成し charts/ ディレクトリに HTML として保存する。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("plotly が未インストールのためチャート生成をスキップします。")


def _check_plotly() -> bool:
    if not _PLOTLY_AVAILABLE:
        logger.error("plotly をインストールしてください: pip install plotly")
        return False
    return True


def plot_equity_curve(
    pnl_list: list[float],
    ticker: str = "ALL",
    output_dir: str = "charts",
) -> Optional[str]:
    """エクイティカーブチャートを生成して保存する。

    Args:
        pnl_list: 各トレードの損益率リスト [%]
        ticker: 銘柄名（ファイル名に使用）
        output_dir: 出力ディレクトリ

    Returns:
        保存した HTML ファイルのパス。plotly 未インストール時は None。
    """
    if not _check_plotly():
        return None

    equity = [1.0]
    for r in pnl_list:
        equity.append(equity[-1] * (1 + r / 100.0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(equity))),
            y=equity,
            mode="lines+markers",
            name="Equity",
            line={"color": "royalblue", "width": 2},
            marker={"size": 5},
        )
    )
    # ドローダウン領域を塗りつぶし
    peak_vals = []
    peak = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        peak_vals.append(peak)

    fig.add_trace(
        go.Scatter(
            x=list(range(len(equity))),
            y=peak_vals,
            mode="lines",
            name="Peak",
            line={"color": "lightgray", "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.08)",
        )
    )

    final_return = (equity[-1] - 1.0) * 100.0
    fig.update_layout(
        title=f"エクイティカーブ [{ticker}]  累積リターン: {final_return:.1f}%",
        xaxis_title="トレード数",
        yaxis_title="資産（初期=1.0）",
        legend={"orientation": "h"},
        template="plotly_white",
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"equity_{ticker}.html")
    fig.write_html(path)
    logger.info("エクイティカーブ保存: %s", path)
    return path


def plot_cwh_pattern(
    df: pd.DataFrame,
    idx_pl: int,
    idx_bc: int,
    idx_pr: int,
    idx_bh: int,
    idx_pv: int,
    ticker: str,
    pattern_type: str = "CWH",
    output_dir: str = "charts",
) -> Optional[str]:
    """CWH パターンをローソク足チャートで可視化して保存する。

    Args:
        df: OHLCV DataFrame
        idx_pl: 左リム（PL）のバーインデックス
        idx_bc: カップ底（BC）のバーインデックス
        idx_pr: 右リム（PR）のバーインデックス
        idx_bh: ハンドル底（BH）のバーインデックス
        idx_pv: ピボット（PV）のバーインデックス
        ticker: 銘柄コード
        pattern_type: パターン種別（"CWH" / "C-SH" / "C-NH"）
        output_dir: 出力ディレクトリ

    Returns:
        保存した HTML ファイルのパス。plotly 未インストール時は None。
    """
    if not _check_plotly():
        return None

    # チャート表示範囲（パターン前後 20 バーを追加）
    display_start = max(0, idx_pl - 20)
    display_end = min(len(df), idx_pv + 20)
    df_view = df.iloc[display_start:display_end]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("価格", "出来高"),
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df_view.index,
            open=df_view["Open"],
            high=df_view["High"],
            low=df_view["Low"],
            close=df_view["Close"],
            name="OHLC",
            increasing_line_color="red",
            decreasing_line_color="blue",
        ),
        row=1,
        col=1,
    )

    # 出来高バー
    fig.add_trace(
        go.Bar(
            x=df_view.index,
            y=df_view["Volume"],
            name="Volume",
            marker_color="rgba(100,100,200,0.4)",
        ),
        row=2,
        col=1,
    )

    # CWH キーポイントをマーカーで表示
    key_points = {
        "PL": (idx_pl, df["High"].iloc[idx_pl], "triangle-up", "green", "上"),
        "BC": (idx_bc, df["Low"].iloc[idx_bc], "triangle-down", "red", "下"),
        "PR": (idx_pr, df["High"].iloc[idx_pr], "triangle-up", "green", "上"),
        "BH": (idx_bh, df["Low"].iloc[idx_bh], "triangle-down", "orange", "下"),
        "PV": (idx_pv, df["High"].iloc[idx_pv], "star", "gold", "上"),
    }

    for label, (idx, price, symbol, color, _) in key_points.items():
        if display_start <= idx < display_end:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[idx]],
                    y=[price],
                    mode="markers+text",
                    marker={"symbol": symbol, "size": 14, "color": color},
                    text=[label],
                    textposition="top center",
                    name=label,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        title=f"{ticker} - {pattern_type} パターン検出",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    path = os.path.join(output_dir, f"pattern_{safe_ticker}_{idx_pv}.html")
    fig.write_html(path)
    logger.info("パターンチャート保存: %s", path)
    return path


def plot_trade_distribution(
    pnl_list: list[float],
    ticker: str = "ALL",
    output_dir: str = "charts",
) -> Optional[str]:
    """トレードの損益率分布ヒストグラムを生成して保存する。

    Args:
        pnl_list: 各トレードの損益率リスト [%]
        ticker: 銘柄名
        output_dir: 出力ディレクトリ

    Returns:
        保存した HTML ファイルのパス。plotly 未インストール時は None。
    """
    if not _check_plotly():
        return None
    if not pnl_list:
        logger.warning("pnl_list が空のためトレード分布チャートをスキップします。")
        return None

    colors = ["green" if r > 0 else "red" for r in pnl_list]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pnl_list,
            nbinsx=20,
            marker_color="royalblue",
            opacity=0.75,
            name="損益率分布",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    wins = sum(1 for r in pnl_list if r > 0)
    win_rate = wins / len(pnl_list) * 100

    fig.update_layout(
        title=f"損益率分布 [{ticker}]  勝率: {win_rate:.1f}%  件数: {len(pnl_list)}",
        xaxis_title="損益率 [%]",
        yaxis_title="件数",
        template="plotly_white",
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"dist_{ticker}.html")
    fig.write_html(path)
    logger.info("分布チャート保存: %s", path)
    return path


def generate_all_charts(
    results: list,
    output_dir: str = "charts",
) -> list[str]:
    """バックテスト結果から全チャートを一括生成する。

    Args:
        results: BacktestResult のリスト
        output_dir: 出力ディレクトリ

    Returns:
        生成した HTML ファイルのパスリスト
    """
    generated: list[str] = []
    all_pnl: list[float] = []

    for result in results:
        pnl_list = [t.pnl_pct for t in result.trades if t.pnl_pct is not None]
        all_pnl.extend(pnl_list)

        if pnl_list:
            p = plot_equity_curve(pnl_list, ticker=result.ticker, output_dir=output_dir)
            if p:
                generated.append(p)
            p = plot_trade_distribution(pnl_list, ticker=result.ticker, output_dir=output_dir)
            if p:
                generated.append(p)

    # 全銘柄統合エクイティカーブ
    if all_pnl:
        p = plot_equity_curve(all_pnl, ticker="ALL", output_dir=output_dir)
        if p:
            generated.append(p)
        p = plot_trade_distribution(all_pnl, ticker="ALL", output_dir=output_dir)
        if p:
            generated.append(p)

    logger.info("チャート生成完了: %d ファイル", len(generated))
    return generated
