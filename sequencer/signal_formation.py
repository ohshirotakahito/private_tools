# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:39:46 2025

@author: ohshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
from pathlib import Path

# データリストを作成する関数
def datalist(selectBC):
    # 指定されたCSVファイルを読み込む
    file_path = Path('BC/'+ selectBC +'.csv')

    if not file_path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {file_path}")

    # データフレームを作成し、カラム名を指定
    df = pd.read_csv(file_path, names=["Index", "Code", "Name", "Colar", "R_conductance", 
                                       "species", "Description", "Extra1", "Extra2", "Extra3"], skiprows=1)
    
    # 'Code'列をキー、'R_conductance'列を値とした辞書を作成
    values = dict(zip(df['Code'], df['R_conductance']))

    # デフォルト値として 'B' を追加
    values['B'] = 0.0
    return values

# 指定された配列に基づいて時間と値を生成する関数
def assigned_form(sequence,selectBC):
    values = datalist(selectBC)  # datalist関数から辞書を取得

    # 各文字に対してランダムな持続時間を生成
    durations = np.random.randint(1, 11, size=len(sequence))
    
    timestamps = []  # 時間リスト
    signal_values = []  # 値リスト
    current_time = 0  # 現在の時間

    for char, duration in zip(sequence, durations):
        if char not in values:
            raise KeyError(f"'{char}' が BC/005.csv の Code 列にありません。")

        for t in range(duration):
            timestamps.append(current_time * 0.001)  # 時間をミリ秒単位で追加
            signal_values.append(values[char])  # 対応する値を追加
            current_time += 1

    return timestamps, signal_values, current_time

# シグナルデータに熱ノイズを追加する関数
def raw_form(signal_data,
             seed=None,
             noise_amplitude=0.10,
             drift_strength=0.002,
             signal_dependent_noise=False):

    # 乱数生成器を設定
    rng = np.random.default_rng(seed)

    # データ点数
    n = len(signal_data)

    # -----------------------------
    # 熱ノイズ（ガウス白色ノイズ）
    # -----------------------------
    
    # signal_dependent_noise=True の場合
    # シグナル強度に比例してノイズを増やす
    if signal_dependent_noise:

        thermal_noise = rng.normal(
            loc=0,
            scale=noise_amplitude * (
                np.abs(signal_data['Value']) + 1e-12
            ),
            size=n
        )

    # 固定ノイズの場合
    else:

        thermal_noise = rng.normal(
            loc=0,
            scale=noise_amplitude,
            size=n
        )

    # -----------------------------
    # ゆっくりしたベースライン揺らぎ
    # -----------------------------
    
    drift = np.cumsum(
        rng.normal(
            loc=0,
            scale=drift_strength / n,
            size=n
        )
    )

    # -----------------------------
    # ノイズを加算
    # -----------------------------
    
    noisy_values = (
        signal_data['Value']
        + thermal_noise
        + drift
    )

    # -----------------------------
    # ノイズ値を新しいデータフレームに追加
    # -----------------------------
    
    noisy_signal_data = signal_data.copy()
    noisy_signal_data['Raw Value'] = noisy_values

    return noisy_signal_data

# シグナルデータをプロットする関数
def data_plt_only_raw_data(signal_data, raw_data):
    plt.figure(figsize=(10, 6))

    # 元の信号をプロット
    plt.plot(raw_data['Time'], raw_data['Raw Value'], label='Raw Signal', linestyle='--', linewidth=1)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Original and Raw Signals (Line Graph)')
    plt.grid(True)
    plt.legend()
    plt.show()

def data_plt(signal_data, raw_data):

    plt.figure(figsize=(10, 6))

    # ノイズ込みデータ
    plt.plot(
        raw_data['Time'],
        raw_data['Raw Value'],
        label='Raw Signal',
        linestyle='-',
        linewidth=1,
        alpha=0.7
    )

    # 元シグナル（赤線）
    plt.plot(
        signal_data['Time'],
        signal_data['Value'],
        color='red',
        label='Assigned Signal',
        linewidth=2
    )

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Signal and Raw Signal')

    plt.grid(True)
    plt.legend()

    plt.show()

# データを保存し、処理済みデータを表示する関数
def bsave_data(signal_data, raw_data):
    # プロット用データを抽出
    plot_data = raw_data[['Time', 'Raw Value']].copy()
    print(plot_data)
    return plot_data

# データを保存し、処理済みデータを表示する関数
def asave_data(signal_data, raw_data):
    # プロット用データを抽出
    plot_data = raw_data[['Time', 'Raw Value']].copy()
    plot_data['Assigned'] = signal_data['Value'].values

    #print(plot_data)
    return plot_data

# 配列の部分配列を生成する関数
def generate_subsequences(seq):
    subsequences = []
    n = len(seq)

    for start in range(n):
        for end in range(start + 1, n + 1):
            subsequences.append(seq[start:end])

    return subsequences

# 部分配列をフィルタリングする関数
def filter_longer_subsequences(subsequences, n):
    return [seq for seq in subsequences if len(seq) >= n]

# 部分配列リストを作成する関数
def create_seqlist(sequence, n):
    reversed_sequence = sequence[::-1]  # 配列を逆順に
    subsequences = generate_subsequences(sequence)
    r_subsequences = generate_subsequences(reversed_sequence)
    combined_sequences = subsequences + r_subsequences
    subsequences_sorted = sorted(combined_sequences, key=len, reverse=True)
    return filter_longer_subsequences(subsequences_sorted, n)

# 部分配列リストからランダムに選択して配列を生成する関数
def generate_random_sequence_with_repeats(t_seqlist, sn, power=2):
    sequence = ''

    # 短い配列ほど出やすくする重み
    # power が大きいほど短い配列が強く優遇される
    weights = [1 / (len(seq) ** power) for seq in t_seqlist]

    #print("=== Weight Check ===")
    #for seq, w in zip(t_seqlist, weights):
    #    print(f"{seq} : length={len(seq)} weight={w:.4f}")

    for i in range(sn):
        seq = random.choices(
            t_seqlist,
            weights=weights,
            k=1
        )[0]

        sequence += seq

        if i < sn - 1:
            num_B = random.randint(1, 12)
            sequence += 'B' * num_B

    sequence = 'BBB' + sequence + 'BBB'

    return sequence

# 部分配列リストを生成し、ランダムな配列を作成する関数
def g_list(sequence, fn, sn):
    t_seqlist = create_seqlist(sequence, fn)

    if len(t_seqlist) == 0:
        raise ValueError("部分配列リストが空です。fn が sequence の長さより大きくないか確認してください。")

    sequences = generate_random_sequence_with_repeats(t_seqlist, sn)
    return sequences

# 連続した重複文字を削除する関数
def remove_consecutive_duplicates(s):
    return ''.join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])

# メイン処理
if __name__ == '__main__':
    # 元の配列を設定
    # 部分配列やランダムな配列生成の基となる文字列を指定
    sequence = "IPP"  # 元の配列（任意に変更可能）
    
    selectBC = 'Amino_01phos'
    
    # 部分配列の最小長を設定
    # `fn` はフィルタリング条件となり、長さが `fn` 以上の部分配列のみを対象とする
    fn = 1  # 最小長を1に設定
    
    # シグナル数を設定
    # `sn` は生成するランダム配列内で部分配列を選択する回数を指定
    sn = 100  # シグナル数（任意に変更可能）
    
    # ランダムな配列を生成
    # g_list関数を使用して、部分配列リストからランダムなシーケンスを生成
    sequences = g_list(sequence, fn, sn)

    #print('Generated Sequence:', sequences)  # 生成されたランダムなシーケンスを表示
    
    # 'B' で挟まれた部分文字列を抽出
    assigned_seqs = [segment for segment in sequences.split('B') if segment]
    
    # 各部分文字列から連続する重複文字を削除
    cleaned_assigned_seqs = [remove_consecutive_duplicates(assigned_seq) for assigned_seq in assigned_seqs]
    
    # Count the length of each string
    lengths = [len(s) for s in cleaned_assigned_seqs]
    
    # Find the maximum length
    max_length = max(lengths)
    
    print(len(cleaned_assigned_seqs), max_length, len(sequence), cleaned_assigned_seqs)
    
    # シーケンスに基づいてタイムスタンプと値を生成
    # assigned_form関数で、シーケンスの各文字に対応する時間と値を生成
    timestamps, signal_values, current_time = assigned_form(sequences,selectBC)
    
    # データフレームに変換
    # 生成されたタイムスタンプと値をDataFrame形式に格納
    signal_data = pd.DataFrame({'Time': timestamps, 'Value': signal_values})
    
    # ノイズを加えたデータを生成
    # raw_form関数を使用して、元のデータにランダムノイズを加える
    raw_data = raw_form(signal_data, seed=42)
    
    # データをプロット
    # data_plt関数で、元データとノイズが加えられたデータをプロット
    data_plt(signal_data, raw_data)
    
    # プロット用データを取得
    # bsave_data関数で、プロット用のデータを抽出し確認
    plot_data = asave_data(signal_data, raw_data)
    
    # 現在の日時を取得して、保存するファイル名に利用
    # YYYYMMDD_HHMMSS形式で日時を取得
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # プロットデータをCSVファイルとして保存
    # ランダムに生成されたデータをCSV形式で保存
    save_dir = Path('seq_data')
    save_dir.mkdir(exist_ok=True)

    plot_csv_file_path = save_dir / f'{sequence}_{current_time_str}_plot_data.csv'
    plot_data.to_csv(plot_csv_file_path, index=False)
    
    # 保存先を表示
    print(f"Plot data saved to: {plot_csv_file_path}")