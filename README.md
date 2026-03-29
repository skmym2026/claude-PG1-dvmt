# CWH統合型バックテストシステム v3.0

## 📊 プロジェクト概要

**Cup with Handle（CWH）パターン検出・バックテストシステム**

William O'Neil が提唱した CWH パターンを、日本株市場（東証）に適用した全自動バックテスト・リアルタイム監視システムです。

v3.0 では以下を実装：
- ✅ 条件1〜10 による厳密な CWH 判定
- ✅ パターン拡張：**CWH** / **C-SH**（浅いハンドル） / **C-NH**（ハンドル無し）
- ✅ 時価総額・信用残フィルター
- ✅ ピーク検知による即売却
- ✅ Streamlit ダッシュボード
- ✅ SBI証券注文生成
- ✅ Intel Core i3 環境対応

## 🎯 実績（v2.3）

| 銘柄 | 期間 | 損益率 |
|------|------|--------|
| カバー（5253） | 2024-10 ～ 2025-06 | +22.4% |
| JET（6228） | 2024-10 ～ 2025-08 | +17.7% |
| ラボロ（5586） | 2025-06 ～ 2025-11 | +19.1% |
| アイズ（5242） | 2025-10-01 ～ 2026-03-03 | +20.1% |
| ジーディープ（5885） | 2025-10-01 ～ 2026-03-03 | +13.1% |
| クオリプス（4894） | 2025-10-01 ～ 2026-03-03 | +17.5% |

## 📁 ファイル構成

```
claude-PG1-dvmt/
├── README.md                          # このファイル
├── requirements.txt                   # 依存ライブラリ
├── cwh_config.yaml                    # パラメータ設定
├── cwh_v30_main.py                    # メインシステム
├── cwh_screener.py                    # CWH検出エンジン（条件1〜10）
├── cwh_pattern_analyzer.py            # パターン判定（CWH/C-SH/C-NH）
├── cwh_backtest_engine.py             # バックテストエンジン
├── cwh_trade_rules.py                 # トレードルール
├── cwh_data_fetcher.py                # データ取得（yfinance）
├── cwh_performance_metrics.py         # パフォーマンス指標
├── cwh_realtime_monitor.py            # リアルタイム監視
├── cwh_growth_market_screener.py      # 東証グロース市場スクリーニング
├── cwh_dashboard.py                   # Streamlit ダッシュボード
├── cwh_sbi_order_generator.py         # SBI証券注文生成
├── tests/
│   ├── test_cwh_detector.py
│   ├── test_backtest_engine.py
│   └── test_exit_strategy.py
└── charts/                             # 出力チャート
```

## 🚀 セットアップ

```bash
pip install -r requirements.txt
```

## 📊 使用方法

### バックテスト実行
```bash
python cwh_v30_main.py
```

### ダッシュボード起動
```bash
streamlit run cwh_dashboard.py
```

### リアルタイム監視
```bash
python cwh_realtime_monitor.py
```

### テスト実行
```bash
pytest tests/ -v
```

## 🔍 CWH パターン定義

### 時系列フロー
```
Ps ──── PL ──── BC ──── PR ──── BH ──── PV
(開始)  (左リム) (底)   (右リム) (ハンドル底) (ピボット)
```

### パターン種別

| パターン | 定義 | 判定基準 |
|---------|------|---------|
| **CWH** | Cup with Handle | 標準ハンドル（4≤N_handle≤25、5%≤D_handle≤15%） |
| **C-SH** | Cup with Shallow Handle | 浅いハンドル（D_handle < 10%） |
| **C-NH** | Cup with No Handle | ハンドル無し（N_handle < 4営業日） |

## 📐 判定条件（条件1〜10）

| # | 条件 | 式 |
|---|------|-----|
| 1 | 事前上昇トレンド | Rprior ≥ 25% |
| 2 | カップ深さ | Dcup ∈ [15%, 35%] |
| 3 | カップ長さ | Ncup ≥ 25営業日 |
| 4 | ハンドル深さ | Dhandle ∈ [5%, 15%] |
| 5 | ハンドル長さ | Nhandle ∈ [4, 25]営業日 |
| 6 | 出来高縮小 | Vhandle < Vcup |
| 7 | リム対称性 | Srim ≤ 15% |
| 8 | MA50との関係 | C(BH) ≥ MA50(BH) |
| 9 | ピボット有効性 | Epivot ≤ 10% |
| 10 | ブレイクアウト出来高 | V(PV) ≥ 1.5·V20(PV) |

## 💰 トレードルール

| ルール | 条件 |
|--------|------|
| エントリー | Pt ≥ 1.03·HL かつ Vt ≥ 1.5·V25 |
| ストップロス | PSL = Pentry × 0.92（-8%） |
| トレーリングストップ | PTS(t) = Pmax(t) × 0.92（+5%以上利益時） |
| テイクプロフィット | PTP = Pentry × 1.20（+20%） |
| 期間終了 | 60営業日 |

## 🔬 検証対象

- **マキタ（6586）**：2020-10 ～ 2021-09 → C-SH vs C-NH 判定
- **フジクラ（5803）**：2025-01 ～ 2026-03 → C-NH 実績検証
- **三菱重工業（7011）**：2025-01 ～ 2026-03 → C-NH 実績検証
- **監視銘柄**：153A, 485A, 6501 の現在進行系パターン

---

**バージョン**：v3.0 | **更新日**：2026-03-29
