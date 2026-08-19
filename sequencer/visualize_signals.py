# -*- coding: utf-8 -*-
"""
batch_generate.py で保存した複数の波形データ (seq_data/manifest.csv) を可視化するスクリプト。

- グリッド表示: 各runを小さいサブプロットで並べて一覧できる
- 重ね書き表示: 全runのRaw Signalを1枚に重ねて、ばらつきを俯瞰できる

batch_generate.py とは完全に独立しているので、データを生成し直さなくても
何度でも気軽に再実行して見え方を変えられます。
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

save_dir = Path('seq_data')
manifest_path = save_dir / 'manifest.csv'

if not manifest_path.exists():
    raise FileNotFoundError(
        f"マニフェストが見つかりません: {manifest_path}\n"
        "先に batch_generate.py を実行してください。"
    )

manifest_df = pd.read_csv(manifest_path)
print(f"{len(manifest_df)} 件のデータが見つかりました。")

# ============================
# 表示するrunを選択（任意）
# None なら全件、リストで指定するとその番号だけ表示
# 例: selected_runs = [0, 1, 2, 5]
# ============================
selected_runs = None

if selected_runs is not None:
    manifest_df = manifest_df[manifest_df['run_idx'].isin(selected_runs)].reset_index(drop=True)

n = len(manifest_df)
if n == 0:
    raise ValueError("表示対象のrunが0件です。selected_runs の指定を確認してください。")

# ============================
# グリッド表示（run毎に個別プロット）
# ============================
ncols = 4
nrows = -(-n // ncols)  # 切り上げ

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4 * ncols, 2.5 * nrows),
    sharey=True,
    squeeze=False,
)
axes = axes.flatten()

for ax, (_, row) in zip(axes, manifest_df.iterrows()):
    df = pd.read_csv(save_dir / row['file_name'])
    ax.plot(df['Time'], df['Raw Value'], linewidth=0.8, alpha=0.7, label='Raw')
    ax.plot(df['Time'], df['Assigned'], color='red', linewidth=1.2, label='Assigned')
    ax.set_title(f"run {row['run_idx']}", fontsize=9)
    ax.grid(True, alpha=0.3)

for ax in axes[n:]:
    ax.axis('off')

axes[0].legend(fontsize=7)
fig.suptitle(f"Signal Runs ({manifest_df.iloc[0]['sequence']}, n={n})")
fig.tight_layout()
plt.show()

# ============================
# 重ね書き表示（全runのRaw Signalをまとめて俯瞰）
# ============================
plt.figure(figsize=(10, 5))
for _, row in manifest_df.iterrows():
    df = pd.read_csv(save_dir / row['file_name'])
    plt.plot(df['Time'], df['Raw Value'], linewidth=0.7, alpha=0.35)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'All Runs Overlaid - Raw Signal (n={n})')
plt.grid(True, alpha=0.3)
plt.show()
