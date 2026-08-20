# sequencer

ナノポア型シーケンサーの信号読み取り〜ベースコール〜アセンブリのプロセスを模した、
展示会向けのリアルタイム可視化ツール（コードネーム: **Molecule Caller**）。

合成した波形データをストリーミング再生しながら、単一ウインドウのダッシュボードで
以下をリアルタイムに可視化する。

- **Real-time signal**: 波形（Raw / Assigned Signal）、Eventsトラック（色分けブロック）、
  直近デコード配列のピル表示、表示時間幅を変えられるWindowスライダー
- **Current call**: 現在読み取り中のコードと確信度バー、確信度ランキング
  （区別困難ペアを読んでいるときは警告色で表示）
- **Sequence assembly**: Mean consensus accuracy（トレンド矢印付き）、Phred Qスコア、
  コンセンサストレース、Depthチャート、Yieldグラフ
- **KPIタイル**: Reads / Yield / Mean Q / Pass率
- **トップバー**: Run選択、LIVEインジケーター、経過時間、Pause/Stop（Stopは確認
  ダイアログ付きでアプリを終了）
- **Export**: その時点の画面のスクリーンショットと、時刻・配列情報・各コードの
  検出時刻をまとめたテキストを `copy/` フォルダへ保存
- **Settings**: `seq_data/` 内のバッチ（生成データ一式）を一覧から選び直して
  アプリを再起動できる
- **Docs**: サイドバーからこの README を日本語/英語で表示

## Directory structure

```
sequencer/
├── batch_generate.py              # 合成波形データの生成スクリプト
├── signal_formation.py            # 波形生成・ノイズ付与などのコア関数
├── sequence_stream_pyqtgraph.py   # ダッシュボード本体（PyQtGraph製）
├── history/                       # 過去バージョンの履歴保管
├── assets/
│   └── jin_mark_white.png         # サイドバーに表示するロゴマーク（なくても動作する）
├── README.md / README_en.md       # このファイル（日本語版・英語版）
├── copy/                          # Export機能の保存先（Git管理外）
└── seq_data/                      # batch_generate.py の出力先（Git管理外）
    ├── _latest_batch.txt          # 自動選択される「最新バッチ」のフォルダ名
    ├── _selected_batch.txt        # Settingsで明示的に選んだバッチ（あれば最優先）
    └── {experiment}_{sample}_{sequence_name}_{timestamp}/
        ├── manifest.csv
        └── *.csv
```


## Usage

### 1. データを生成する

```bash
uv run batch_generate.py
```

スクリプト冒頭のパラメータで、配列とその識別情報を設定できる。

```python
sequence = 'GADGVGKSAL'          # 実際の配列（文字列）
sequence_name = 'sample_seqA'    # 配列の呼び名（空文字なら未設定）
sample_name = ''                 # 試料名（空文字なら未設定）
experiment_name = 'exp01'        # 実験名（空文字なら未設定）
```

実行すると `seq_data/{experiment}_{sample}_{sequence_name}_{timestamp}/` に
run ごとの波形CSV（`Code`列を含む）と `manifest.csv` が生成され、
`seq_data/_latest_batch.txt` が自動更新される（可視化側が次回起動時に
自動でこのバッチを読み込む）。

サンプリングは 1データ点 = 0.1ms（10kHz）。

### 2. ダッシュボードを実行する

```bash
uv run sequence_stream_pyqtgraph.py
```

何も設定しなくても、直近に生成した最新バッチを自動的に読み込む。過去のバッチを
見たい場合は、アプリ内の **Settings** から選び直すか、スクリプト冒頭の
`BATCH_DIR_OVERRIDE` にフォルダ名を指定する。

## Requirements

```
pyqtgraph
PyQt5  (または PyQt6)
numpy
pandas
```

## Notes

- `seq_data/` `copy/` 以下の生成データ・出力ファイルは実験データにあたるため、
  リポジトリの Development policy に従いGit管理には含めない（`.gitignore`推奨）。
- PyQt5とPyQt6の両方で動くよう、Qtの列挙型まわりに簡易的な互換レイヤーを
  入れている（環境によってどちらが使われるか変わるため）。
- ダークテーマ・展示会向けの配色を前提にレイアウトを組んでいる。
- `assets/jin_mark_white.png` が見つからない場合でも、ロゴなしでそのまま
  起動する（起動時にターミナルへ見つかった/見つからなかったを表示する）。
