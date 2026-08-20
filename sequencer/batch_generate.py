# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.11.1",
#     "numpy>=2.4.6",
#     "pandas>=3.0.5",
# ]
# ///

# -*- coding: utf-8 -*-
"""
signal_formation.py の関数を使って、波形データを複数回まとめて生成・保存するスクリプト。

実行すると seq_data/ フォルダに
  - 各runごとのCSV (Time, Raw Value, Assigned, Code)
  - manifest.csv (どのファイルがどの条件で作られたかの一覧)
が保存されます。

Code列は、その時点の値がどの配列文字（アミノ酸コードなど）に対応するかを
そのまま残したもの。可視化側でこれを使って波形と配列情報を紐づけられる。

visualize_signals.py / visualize_signals_stream.py はCode列を使わないので
そのまま動く（余分な列は無視される）。visualize_signals_stream_v2.py が
Code列を使って波形と配列を同期表示する。
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from signal_formation import g_list, raw_form, datalist

# スクリプト自身の場所を基準にする（どのディレクトリから実行しても動くように）
BASE_DIR = Path(__file__).resolve().parent


def assigned_form_with_codes(sequence, values):
    """assigned_form と同じロジックだが、各時点の元コード(文字)も一緒に返す版。"""
    durations = np.random.randint(1, 11, size=len(sequence))

    timestamps = []
    signal_values = []
    codes = []
    current_time = 0

    for char, duration in zip(sequence, durations):
        if char not in values:
            raise KeyError(f"'{char}' が BC CSV の Code 列にありません。")

        for _ in range(duration):
            # 1データ点 = 0.1ms (10kHz) 。以前は 0.001 (=1ms, 1kHz) になっていた。
            timestamps.append(current_time * 0.0001)
            signal_values.append(values[char])
            codes.append(char)
            current_time += 1

    return timestamps, signal_values, codes, current_time

# ============================
# パラメータ（ここを変更して使う）
# ============================
experiment_name = 'Tsuzikawa'    # 実験名。空文字 '' なら未設定。

sample_name = 'VHH'             # 試料名。空文字 '' なら未設定。

sequence_name = 'VHH_spt50_001'  # 配列の呼び名(例: 'vassp@resin', 'osytosine')。空文字 '' なら未設定。

#sequence = "CYFQNCPRG"    # vassp@resin       # 元配列
#sequence = "CYIQNCPLG" #osytosine
sequence = 'PAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNG'      # 実際の配列(文字列)。BC CSVのCode列に対応する文字で構成する。

selectBC = 'Amino_01phos'   # BC/{selectBC}.csv を参照
fn = 1                      # 部分配列の最小長
sn = 100                    # 1回あたりのシグナル数

num_runs = 100                # 生成する波形の本数
noise_seed_base = 42         # 各runで seed=base+run_idx とする。Noneなら毎回ランダム
noise_amplitude = 0.05
drift_strength = 0.002
signal_dependent_noise = False

save_dir_root = BASE_DIR / 'seq_data'
save_dir_root.mkdir(exist_ok=True)


def _sanitize_for_folder(name):
    """フォルダ名に使えない文字(空白・スラッシュなど)を安全な文字に置き換える"""
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(name)).strip("_")


# 配列ごとに散らからないよう、実行ごとに専用のサブフォルダへ保存する。
# フォルダ名は 実験名 → 試料名 → 配列の呼び名 → 日時 の順。
# 実際の配列(sequence文字列)は長くなりがちなのでフォルダ名には含めず、
# manifest.csv 側にだけ記録する。
#   例: seq_data/exp01_sampleA_sample_seqA_20260820_124127/
#   すべて空文字なら: seq_data/20260820_124127/
_batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_name_parts = [
    p for p in (
        _sanitize_for_folder(experiment_name),
        _sanitize_for_folder(sample_name),
        _sanitize_for_folder(sequence_name),
    ) if p
]
batch_dir_name = "_".join(_name_parts + [_batch_timestamp])
save_dir = save_dir_root / batch_dir_name
save_dir.mkdir(exist_ok=True)

values_lookup = datalist(selectBC)  # Code -> R_conductance の辞書（1回だけ読み込み）

# ============================
# 生成ループ
# ============================
manifest_rows = []

for run_idx in range(num_runs):
    sequences = g_list(sequence, fn, sn)
    timestamps, signal_values, codes, current_time = assigned_form_with_codes(sequences, values_lookup)
    signal_data = pd.DataFrame({'Time': timestamps, 'Value': signal_values, 'Code': codes})

    seed = None if noise_seed_base is None else noise_seed_base + run_idx
    raw_data = raw_form(
        signal_data,
        seed=seed,
        noise_amplitude=noise_amplitude,
        drift_strength=drift_strength,
        signal_dependent_noise=signal_dependent_noise,
    )

    plot_data = raw_data[['Time', 'Raw Value']].copy()
    plot_data['Assigned'] = signal_data['Value'].values
    plot_data['Code'] = signal_data['Code'].values

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f'{sequence}_run{run_idx:03d}_{timestamp_str}.csv'
    file_path = save_dir / file_name
    plot_data.to_csv(file_path, index=False)

    manifest_rows.append({
        'run_idx': run_idx,
        'file_name': file_name,
        'sequence': sequence,
        'sequence_name': sequence_name,
        'sample_name': sample_name,
        'experiment_name': experiment_name,
        'selectBC': selectBC,
        'seed': seed,
        'noise_amplitude': noise_amplitude,
        'drift_strength': drift_strength,
        'n_points': len(plot_data),
        'created_at': timestamp_str,
    })

    print(f"[{run_idx + 1}/{num_runs}] saved: {file_name}")

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = save_dir / 'manifest.csv'
manifest_df.to_csv(manifest_path, index=False)

# 可視化スクリプト(sequence_stream_pyqtgraph.py)が、何も設定しなくても
# 自動的にこのバッチを読みに行けるよう、「最新バッチ」を指すポインタを書いておく。
# 過去のバッチを見たいときは、可視化スクリプト側の BATCH_DIR_OVERRIDE に
# batch_dir_name をそのまま指定すれば切り替えられる。
(save_dir_root / '_latest_batch.txt').write_text(batch_dir_name, encoding='utf-8')

print(f"\n{num_runs} 件の波形データを生成しました。")
print(f"保存先フォルダ: {save_dir}")
print(f"マニフェストを保存しました: {manifest_path}")
