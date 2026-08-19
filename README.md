# Private Tools

個人用のPythonプログラムおよび解析ツールを開発・管理するためのリポジトリ。

## Directory structure

- `src/` : Pythonプログラム本体
- `sequencer/` : ナノポア型シーケンサー可視化ツール（詳細は [sequencer/README.md](sequencer/README.md) を参照）
- `tests/` : テストコード
- `docs/` : 設計・仕様・開発メモ

## Tools

- **sequencer** : シーケンサーの信号読み取り〜アセンブリ過程をリアルタイム可視化する展示会向けツール（Molecule Caller）。詳細は [sequencer/README.md](sequencer/README.md)。

## Development policy

- 共有研究コードとは分離して管理する。
- 実験データや大容量データはGitに保存しない。
- 解析結果や学習済みモデルは原則としてGitに保存しない。
- APIキーやパスワードなどの秘密情報はGitに保存しない。
- 新しい機能は原則として `src/` 以下に作成する。

## Repository status

This repository is tracked on GitHub (`origin/main`).
