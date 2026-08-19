# -*- coding: utf-8 -*-
"""
batch_generate.py (Code列対応版) で保存した複数の波形データを、
配列情報（アミノ酸コード）と同期させながらストリーミング表示するスクリプト。

構成:
  - 上段: 波形（Raw / Assigned）。背景をそのコード固有の色で塗って区間を可視化
  - 下段: 配列トラック。ゲノムブラウザ / ピアノロール風に、色ブロック+文字ラベルが
          波形と完全に同期してスクロールする
  - 再生ヘッド（縦線）: 「今読んでいる位置」を示す固定ライン。通過中のコードを
          左上のパネルに大きく表示する

BC/{selectBC}.csv の 'Colar' 列（10進数エンコードのRGB値）をそのまま
色として使うので、コードごとに色分けされる。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# ============================
# 見た目パラメータ
# ============================
plt.style.use('dark_background')

ACCENT_COLOR = '#39FF14'      # 再生ヘッドなどのネオン色
GRID_COLOR = '#333333'
FONT_MONO = 'DejaVu Sans Mono'

# スクリプト自身の場所を基準にする（どのディレクトリから実行しても動くように）
BASE_DIR = Path(__file__).resolve().parent

save_dir = BASE_DIR / 'seq_data'
manifest_path = save_dir / 'manifest.csv'

if not manifest_path.exists():
    raise FileNotFoundError(
        f"マニフェストが見つかりません: {manifest_path}\n"
        "先に batch_generate.py を実行してください（Code列対応版）。"
    )

manifest_df = pd.read_csv(manifest_path)
selectBC = manifest_df.iloc[0]['selectBC']
print(f"{len(manifest_df)} 件のデータを連結してアニメーション表示します。 (BC = {selectBC})")


# ============================
# BC CSVからコードの色・名前情報を読み込む
# ============================
def load_code_info(selectBC):
    file_path = BASE_DIR / 'BC' / f'{selectBC}.csv'
    df = pd.read_csv(
        file_path,
        names=["Index", "Code", "Name", "Colar", "R_conductance",
               "species", "Description", "Extra1", "Extra2", "Extra3"],
        skiprows=1,
    )
    info = {}
    for _, row in df.iterrows():
        hex_color = '#{:06X}'.format(int(row['Colar']))
        info[row['Code']] = {
            'name': row['Name'] if isinstance(row['Name'], str) else str(row['Code']),
            'color': hex_color,
            'description': row['Description'] if isinstance(row['Description'], str) else '',
        }
    info.setdefault('B', {'name': 'Baseline', 'color': '#969696', 'description': 'baseline'})
    return info


def contrasting_text_color(hex_color):
    """背景色の明るさに応じて、読みやすい文字色（黒or白）を返す"""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000000' if luminance > 140 else '#FFFFFF'


code_info = load_code_info(selectBC)

# ============================
# 全runのデータを時間軸で連結
# ============================
all_time = []
all_raw = []
all_assigned = []
all_codes = []
run_boundaries = []  # (run_idx, 開始時刻)

time_offset = 0.0
gap = 0.01  # run間の隙間

for _, row in manifest_df.iterrows():
    df = pd.read_csv(save_dir / row['file_name'])
    if 'Code' not in df.columns:
        raise ValueError(
            f"{row['file_name']} に 'Code' 列がありません。"
            "batch_generate.py を最新版（Code列対応）で再実行してください。"
        )

    t = df['Time'].values + time_offset
    run_boundaries.append((row['run_idx'], t[0]))
    all_time.append(t)
    all_raw.append(df['Raw Value'].values)
    all_assigned.append(df['Assigned'].values)
    all_codes.append(df['Code'].astype(str).values)

    time_offset = t[-1] + gap

all_time = np.concatenate(all_time)
all_raw = np.concatenate(all_raw)
all_assigned = np.concatenate(all_assigned)
all_codes = np.concatenate(all_codes)
n_total = len(all_time)
dt = float(np.median(np.diff(all_time))) if n_total > 1 else 0.001

# ============================
# 連続する同一コード区間をセグメント化（毎フレーム計算しないよう事前計算）
# ============================
change_idx = np.where(all_codes[1:] != all_codes[:-1])[0] + 1
seg_start_idx = np.concatenate(([0], change_idx))
seg_codes = all_codes[seg_start_idx]
seg_start_t = all_time[seg_start_idx]

seg_end_t = np.empty_like(seg_start_t)
seg_end_t[:-1] = seg_start_t[1:]
seg_end_t[-1] = all_time[-1] + dt

n_seg = len(seg_start_t)


def visible_segment_range(t_start, t_now):
    """[t_start, t_now] に重なるセグメントのインデックス範囲 [lo, hi) を返す"""
    lo = np.searchsorted(seg_end_t, t_start, side='right')
    hi = np.searchsorted(seg_start_t, t_now, side='right')
    return lo, max(hi, lo)


def segment_at_time(t):
    idx = np.searchsorted(seg_start_t, t, side='right') - 1
    idx = int(np.clip(idx, 0, n_seg - 1))
    return seg_codes[idx]


# ============================
# アニメーションパラメータ
# ============================
window_width = 0.3      # 表示する時間幅
step = 5                # 1フレームで進めるデータ点数
interval_ms = 20        # フレーム間隔(ms)
playhead_frac = 0.8     # 窓の中で再生ヘッドを置く位置（0=左端, 1=右端）
min_label_frac = 0.02   # セグメント幅がこの割合未満ならラベル文字を省略

save_as_gif = False
gif_fps = 30

# ============================
# 描画準備
# ============================
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(3, 1, height_ratios=[5, 1, 0.01], hspace=0.05)
ax_wave = fig.add_subplot(gs[0])
ax_track = fig.add_subplot(gs[1], sharex=ax_wave)

# --- 波形パネル ---
line_raw, = ax_wave.plot([], [], linewidth=0.8, alpha=0.6, color='#4FC3F7', label='Raw Signal')
line_assigned, = ax_wave.plot([], [], linewidth=1.8, color='#FFFFFF', label='Assigned Signal')

margin = 0.1 * (all_raw.max() - all_raw.min() + 1e-9)
ax_wave.set_ylim(all_raw.min() - margin, all_raw.max() + margin)
ax_wave.set_ylabel('Value')
ax_wave.grid(True, color=GRID_COLOR, alpha=0.5)
ax_wave.legend(loc='upper right', framealpha=0.3)
plt.setp(ax_wave.get_xticklabels(), visible=False)
title = ax_wave.set_title('', fontsize=11, fontfamily=FONT_MONO)

playhead_wave = ax_wave.axvline(0, color=ACCENT_COLOR, linewidth=1.5, alpha=0.9, zorder=10)

# --- 配列トラックパネル ---
ax_track.set_ylim(0, 1)
ax_track.set_yticks([])
ax_track.set_xlabel('Time')
ax_track.grid(False)
playhead_track = ax_track.axvline(0, color=ACCENT_COLOR, linewidth=1.5, alpha=0.9, zorder=10)

# --- 「現在読み取り中」表示パネル ---
readout_box = fig.text(
    0.13, 0.90, '', fontsize=22, fontweight='bold', fontfamily=FONT_MONO,
    color=ACCENT_COLOR, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', fc='#111111', ec=ACCENT_COLOR, lw=1.5, alpha=0.85),
)
readout_sub = fig.text(
    0.13, 0.82, '', fontsize=10, fontfamily=FONT_MONO,
    color='#AAAAAA', va='top', ha='left',
)

# 動的に追加/削除する描画要素をまとめて管理
dynamic_artists = []


def clear_dynamic_artists():
    for artist in dynamic_artists:
        artist.remove()
    dynamic_artists.clear()


def init():
    line_raw.set_data([], [])
    line_assigned.set_data([], [])
    return line_raw, line_assigned


def update(frame):
    idx = min(frame * step, n_total - 1)
    t_now = all_time[idx]
    t_start = max(0.0, t_now - window_width)

    mask = (all_time >= t_start) & (all_time <= t_now)
    line_raw.set_data(all_time[mask], all_raw[mask])
    line_assigned.set_data(all_time[mask], all_assigned[mask])

    ax_wave.set_xlim(t_start, t_start + window_width)
    ax_track.set_xlim(t_start, t_start + window_width)

    clear_dynamic_artists()

    # --- 波形背景の色分け & 配列トラックの色ブロック ---
    lo, hi = visible_segment_range(t_start, t_now)
    for si in range(lo, hi):
        code = seg_codes[si]
        info = code_info.get(code, {'name': code, 'color': '#888888'})
        color = info['color']
        seg_l = max(seg_start_t[si], t_start)
        seg_r = min(seg_end_t[si], t_start + window_width)
        width = seg_r - seg_l
        if width <= 0:
            continue

        span = ax_wave.axvspan(seg_l, seg_r, color=color, alpha=0.18, lw=0, zorder=0)
        dynamic_artists.append(span)

        rect = mpatches.Rectangle(
            (seg_l, 0), width, 1, facecolor=color, edgecolor='#000000', linewidth=0.5, zorder=1,
        )
        ax_track.add_patch(rect)
        dynamic_artists.append(rect)

        if code != 'B' and width >= window_width * min_label_frac:
            txt_color = contrasting_text_color(color)
            label = ax_track.text(
                seg_l + width / 2, 0.5, code,
                ha='center', va='center', fontsize=9, fontweight='bold',
                fontfamily=FONT_MONO, color=txt_color, zorder=2,
            )
            dynamic_artists.append(label)

    # --- run境界の縦線 ---
    for run_idx, start_t in run_boundaries:
        if t_start <= start_t <= t_start + window_width:
            ln1 = ax_wave.axvline(start_t, color='#888888', linestyle='--', alpha=0.5, zorder=3)
            ln2 = ax_track.axvline(start_t, color='#888888', linestyle='--', alpha=0.5, zorder=3)
            dynamic_artists.extend([ln1, ln2])

    # --- 再生ヘッド ---
    playhead_t = t_start + window_width * playhead_frac
    playhead_wave.set_xdata([playhead_t, playhead_t])
    playhead_track.set_xdata([playhead_t, playhead_t])

    # --- 現在読み取り中のコードを表示 ---
    current_code = segment_at_time(playhead_t)
    info = code_info.get(current_code, {'name': current_code, 'description': ''})
    if current_code == 'B':
        readout_box.set_text('—')
        readout_box.set_color('#888888')
        readout_sub.set_text('baseline')
    else:
        readout_box.set_text(f"{current_code}  {info['name']}")
        readout_box.set_color(info['color'])
        readout_sub.set_text(info.get('description', ''))

    current_run = 0
    for run_idx, start_t in run_boundaries:
        if start_t <= t_now:
            current_run = run_idx
    title.set_text(f"run {current_run} / {len(manifest_df) - 1}   |   time = {t_now:.3f}")

    return [line_raw, line_assigned, title, playhead_wave, playhead_track] + dynamic_artists


n_frames = n_total // step + 1

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=interval_ms, blit=False, repeat=False,
)

fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1)

if save_as_gif:
    out_path = save_dir / 'streaming_signal_v2.gif'
    ani.save(out_path, writer='pillow', fps=gif_fps)
    print(f"アニメーションを保存しました: {out_path}")
else:
    plt.show()
