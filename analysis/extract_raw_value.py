# -*- coding: utf-8 -*-
"""
extract_raw_value.py

目的
----
sequencer/seq_data 配下にある解析済みCSV
(列: Time, Raw Value, Assigned, Code) から
"Time" と "Raw Value" だけを抜き出し、
test_data フォルダに「生信号だけ」のファイルとして保存する。

これは信号ピックアップ(パルス検出)アルゴリズムを
  1. 人工データで検証
  2. 実データから切り出した短い区間で検証   <- ここで使うデータを作る
  3. 実際のTDMSファイルに本適用
という段階を踏んで作っていくための、ステップ2用の準備スクリプト。

元CSVにある Assigned / Code 列は正解ラベルと思われるため、
抽出ファイルには含めない（＝アルゴリズムには見せない）。
ただし検出結果を後で答え合わせできるよう、
どの元ファイルから来たかを追跡できるようにしている。

使い方
------
1. 下の CONFIG セクションのパスを自分の環境に合わせて書き換える
2. python extract_raw_value.py を実行
3. test_data/<サンプルフォルダ名>/<元ファイル名>_raw.csv と .npy が生成される
"""

import os
import csv
import numpy as np
from pathlib import Path


# ========= CONFIG =========

# seq_data のルートフォルダ（sequencer/seq_data）
SEQ_DATA_ROOT = Path("sequencer/seq_data")

# 出力先のルートフォルダ
TEST_DATA_ROOT = Path("test_data")

# 元CSVの列名
TIME_COL = "Time"
VALUE_COL = "Raw Value"

# サブフォルダ（サンプルフォルダ）ごとにtest_data内も分けるか
KEEP_SUBFOLDER_STRUCTURE = True


# ========= Utility =========

def list_seq_csv_files(seq_data_root: Path, recursive: bool = True):
    """seq_data 配下のCSVファイルを列挙する。"""
    if not seq_data_root.is_dir():
        print(f"[WARN] フォルダが存在しません: {seq_data_root}")
        return []

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(seq_data_root.glob(pattern))
    return files


def read_time_and_raw_value(csv_path: Path, time_col=TIME_COL, value_col=VALUE_COL):
    """CSVから Time, Raw Value の2列だけを読み込む。"""
    times = []
    values = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if time_col not in reader.fieldnames or value_col not in reader.fieldnames:
            raise ValueError(
                f"列 '{time_col}' または '{value_col}' が見つかりません: "
                f"{csv_path} (列: {reader.fieldnames})"
            )

        for row in reader:
            times.append(float(row[time_col]))
            values.append(float(row[value_col]))

    return np.asarray(times), np.asarray(values)


def resolve_output_dir(csv_path: Path, seq_data_root: Path, test_data_root: Path):
    """元CSVの位置に応じた出力先フォルダを決める。"""
    if not KEEP_SUBFOLDER_STRUCTURE:
        out_dir = test_data_root
    else:
        try:
            rel_dir = csv_path.parent.relative_to(seq_data_root)
        except ValueError:
            rel_dir = Path(csv_path.parent.name)
        out_dir = test_data_root / rel_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_raw_value(out_dir: Path, base_name: str, times: np.ndarray, values: np.ndarray):
    """Time, Raw Value を csv と npy の両方で保存する。"""
    csv_path = out_dir / f"{base_name}_raw.csv"
    npy_path = out_dir / f"{base_name}_raw.npy"

    arr = np.column_stack([times, values])
    np.savetxt(csv_path, arr, delimiter=",", header="Time,Raw Value", comments="")
    np.save(npy_path, values)  # 信号値のみ（アルゴリズム側で高速に読み込む用）

    return csv_path, npy_path


# ========= main =========

def main():
    csv_files = list_seq_csv_files(SEQ_DATA_ROOT, recursive=True)

    if not csv_files:
        print(f"[WARN] CSVファイルが見つかりませんでした: {SEQ_DATA_ROOT}")
        return

    print(f"[INFO] {len(csv_files)} 件のCSVファイルを処理します。")

    n_ok = 0
    n_ng = 0

    for csv_path in csv_files:
        try:
            times, values = read_time_and_raw_value(csv_path)
        except Exception as e:
            print(f"[ERROR] 読み込み失敗: {csv_path}\n  -> {e}")
            n_ng += 1
            continue

        out_dir = resolve_output_dir(csv_path, SEQ_DATA_ROOT, TEST_DATA_ROOT)
        base_name = csv_path.stem

        out_csv, out_npy = save_raw_value(out_dir, base_name, times, values)

        print(f"[SAVED] {out_csv}")
        print(f"[SAVED] {out_npy}")
        n_ok += 1

    print("=" * 80)
    print(f"[INFO] 完了: 成功 {n_ok} 件 / 失敗 {n_ng} 件")


if __name__ == "__main__":
    main()
