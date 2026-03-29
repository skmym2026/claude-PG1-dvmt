"""
cwh_pattern_analyzer.py - パターン判定モジュール（CWH / C-SH / C-NH）

CWHCandidate の属性値をもとに、以下の 3 種類のパターンに分類する。

  CWH  : Cup with Handle        標準（4 <= N_handle <= 25、5% <= D_handle <= 15%）
  C-SH : Cup with Shallow Handle 浅いハンドル（D_handle < 10%）
  C-NH : Cup with No Handle      ハンドル無し（N_handle < 4 営業日）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cwh_screener import CWHCandidate


class PatternType(str, Enum):
    """CWH パターン種別。"""

    CWH = "CWH"    # Cup with Handle
    C_SH = "C-SH"  # Cup with Shallow Handle
    C_NH = "C-NH"  # Cup with No Handle
    NONE = "NONE"  # 判定不能


@dataclass
class PatternResult:
    """パターン分類結果。"""

    ticker: str
    pattern_type: PatternType
    candidate: CWHCandidate
    reason: str

    @property
    def is_valid(self) -> bool:
        """有効パターン（CWH / C-SH / C-NH のいずれか）か。"""
        return self.pattern_type != PatternType.NONE


def classify_pattern(
    candidate: CWHCandidate,
    cfg: dict,
) -> PatternResult:
    """CWHCandidate を CWH / C-SH / C-NH に分類する。

    分類ルール:
      1. N_handle < c_nh_handle_bars_max  => C-NH
      2. D_handle < c_sh_handle_depth_max => C-SH
      3. それ以外（全条件 PASS）          => CWH

    Args:
        candidate: 検出済みの CWH 候補
        cfg: cwh_config.yaml の patterns セクション

    Returns:
        PatternResult
    """
    c_nh_max: int = cfg.get("c_nh_handle_bars_max", 3)
    c_sh_depth_max: float = cfg.get("c_sh_handle_depth_max", 0.10)

    # C-NH 判定：ハンドル期間が短すぎる
    if candidate.n_handle < c_nh_max:
        return PatternResult(
            ticker=candidate.ticker,
            pattern_type=PatternType.C_NH,
            candidate=candidate,
            reason=f"N_handle={candidate.n_handle} < {c_nh_max}（ハンドル無し）",
        )

    # C-SH 判定：ハンドル深さが浅い
    if candidate.d_handle < c_sh_depth_max:
        return PatternResult(
            ticker=candidate.ticker,
            pattern_type=PatternType.C_SH,
            candidate=candidate,
            reason=f"D_handle={candidate.d_handle:.2%} < {c_sh_depth_max:.0%}（浅いハンドル）",
        )

    # 通常の CWH
    return PatternResult(
        ticker=candidate.ticker,
        pattern_type=PatternType.CWH,
        candidate=candidate,
        reason="標準 CWH パターン",
    )


def analyze_patterns(
    candidates: list[CWHCandidate],
    cfg: dict,
) -> list[PatternResult]:
    """複数の候補を一括でパターン分類する。

    Args:
        candidates: CWHCandidate のリスト（all_pass=True のもの）
        cfg: cwh_config.yaml の patterns セクション

    Returns:
        PatternResult のリスト
    """
    return [classify_pattern(c, cfg) for c in candidates if c.all_pass]
