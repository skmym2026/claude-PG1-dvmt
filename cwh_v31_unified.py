"""
cwh_v31_unified.py

CWH / C-NH 統合検出エンジン v3.1
- CWH と C-NH を統一インターフェースで処理
- コード重複削減
- 混合スクリーニング対応
"""

from __future__ import annotations

from typing import Dict, List, Optional

from cwh_cnh_v30 import CWHCNHBacktest, find_patterns, load_config


class CWHCNHUnified:
    """
    CWH / C-NH 統合検出・スクリーニングクラス

    使用例:
        engine = CWHCNHUnified("cwh_config.yaml")
        patterns = engine.detect_patterns(df, pattern_type="both")
        results  = engine.backtest_unified(stocks, "2020-01-01", "2026-03-29")
    """

    def __init__(self, config_file: str = "cwh_config.yaml") -> None:
        self.cfg = load_config(config_file)
        self._backtest_engine = CWHCNHBacktest(config_file=config_file)

    # ------------------------------------------------------------------
    # パターン検出
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        df,
        pattern_type: str = "both",
    ) -> List[Dict]:
        """
        DataFrame から CWH / C-NH パターンを検出する。

        Args:
            df:           pandas DataFrame（columns: High, Low, Close, Volume）
            pattern_type: "cwh" / "cnh" / "both"

        Returns:
            検出されたパターン情報のリスト
        """
        highs   = df["High"].tolist()
        lows    = df["Low"].tolist()
        closes  = df["Close"].tolist()
        volumes = df["Volume"].tolist()

        result = find_patterns(highs, lows, closes, volumes, self.cfg)

        detected = []
        if pattern_type in ("cwh", "both") and result.get("cwh"):
            detected.append({"pattern": "CWH", "entry_price": closes[-1]})
        if pattern_type in ("cnh", "both") and result.get("cnh"):
            detected.append({"pattern": "C-NH", "entry_price": closes[-1]})

        return detected

    # ------------------------------------------------------------------
    # 統合バックテスト
    # ------------------------------------------------------------------

    def backtest_unified(
        self,
        stocks: Dict[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        """
        全銘柄の統合バックテストを実行する。

        Args:
            stocks:     {"cwh": ["5253.T", ...], "cnh": ["5803.T", ...]}
            start_date: バックテスト開始日（省略時は設定ファイルの値を使用）
            end_date:   バックテスト終了日（省略時は設定ファイルの値を使用）

        Returns:
            パフォーマンス指標と全トレードリスト
        """
        if start_date:
            self._backtest_engine.start_date = start_date
        if end_date:
            self._backtest_engine.end_date = end_date

        return self._backtest_engine.run_all(stocks)

    # ------------------------------------------------------------------
    # CWH / C-NH 個別検出メソッド
    # ------------------------------------------------------------------

    def find_cwh_patterns(self, df) -> List[Dict]:
        """CWH パターンのみ検出する。"""
        return self.detect_patterns(df, pattern_type="cwh")

    def find_cnh_patterns(self, df) -> List[Dict]:
        """C-NH パターンのみ検出する。"""
        return self.detect_patterns(df, pattern_type="cnh")
