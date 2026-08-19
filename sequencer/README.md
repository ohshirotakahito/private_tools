# sequencer

ナノポア型シーケンサーの信号読み取り〜ベースコール〜アセンブリのプロセスを模した、
展示会向けのリアルタイム可視化ツール（コードネーム: **Molecule Caller**）。

合成した波形データをストリーミング再生しながら、以下をリアルタイムに可視化する。

- 波形（Raw / Assigned Signal）と、読み取り中のコード（アミノ酸コード）のHUD表示
- 配列トラック（ゲノムブラウザ風の色分けブロック + 塩基ごとのQV数字）
- 判定確信度ランキング（CONFIDENCEパネル）
- アセンブリ結果（reference配列へのアラインメント、Depth、コンセンサス精度）
- Phred Qスコア、N50、Depth CV、Pass率などのQC指標
- 精度ゲージ、トレンド矢印、警告バッジなどのダッシュボード演出

## Directory structure

```
sequencer/
├── batch_generate.py              # 合成波形データの生成スクリプト
├── visualize_signals_stream_v4.py # 現行版（展示会用・本番）
├── visualize_signals_stream_v5.py # 実験/派生版（要整理）
├── ...
├── visualize_signals_stream_v10.py
└── seq_data/                      # batch_generate.py の出力先（Git管理外）
    ├── manifest.csv
    └── *.csv
```

> **v5〜v10について**: 現状は実験・派生バージョンが未整理のまま残っている。
> 現行の完成版は **v4** 。他バージョンの位置づけが整理でき次第、
> 不要なものは削除するか `archive/` 以下に退避する予定。

## Usage

### 1. データを生成する

```bash
python batch_generate.py
```

`seq_data/manifest.csv` と、run ごとの波形CSV（`Code`列を含む）が生成される。

### 2. ストリーミング可視化を実行する

```bash
python visualize_signals_stream_v4.py
```

2つのウインドウが開く。

- **Figure 1 (MOLECULE CALLER)**: 波形 + 配列トラック + CONFIDENCEパネル
- **Figure 2 (SEQUENCE ASSEMBLY)**: 精度ゲージ + KPIカード + reference配列 +
  コンセンサストレース + Depthチャート + Yieldグラフ

GIFとして保存したい場合は、スクリプト冒頭の `save_as_gif = True` に変更する。

## Requirements

```
numpy
pandas
matplotlib
```

## Notes

- `seq_data/` 以下の生成データは実験データにあたるため、リポジトリの
  Development policy に従いGit管理には含めない（`.gitignore`推奨）。
- ダークテーマ・ネオン配色・展示会向けの大きめフォントを前提にレイアウトを
  組んでいるため、通常の分析用途で使う場合はフォントサイズ定数
  （`FS_*`, `LW_*`）を調整するとよい。
