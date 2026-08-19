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
import matplotlib.patheffects as pe

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
            'value': float(row['R_conductance']) if pd.notna(row['R_conductance']) else None,
        }
    info.setdefault('B', {'name': 'Baseline', 'color': '#969696', 'description': 'baseline', 'value': 0.0})
    return info


def contrasting_text_color(hex_color):
    """背景色の明るさに応じて、読みやすい文字色（黒or白）を返す"""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000000' if luminance > 140 else '#FFFFFF'


def _logit(p, eps=1e-3):
    """確率pを対数オッズに変換（コンセンサス精度をベイズ的に積み上げるため）"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _prob_to_qscore(p, q_cap=60.0):
    """確信度(0-1)をPhredクオリティスコアに変換: Q = -10*log10(1-p)"""
    p = np.clip(p, 0.0, 1.0 - 10 ** (-q_cap / 10.0))
    error = np.clip(1.0 - p, 10 ** (-q_cap / 10.0), 1.0)
    return -10.0 * np.log10(error)


PASS_Q_THRESHOLD = 9.0  # ONT等でよく使われる合格ライン（Q9 ≈ 87.4%）に合わせる
PASS_CONF_THRESHOLD = 1.0 - 10 ** (-PASS_Q_THRESHOLD / 10.0)


code_info = load_code_info(selectBC)

# 確率密度（信頼度）計算用: 基準値(R_conductance)が定義されているコードだけを候補にする
_prob_codes = [c for c, v in code_info.items() if v.get('value') is not None]
prob_codes = np.array(_prob_codes)
prob_values = np.array([code_info[c]['value'] for c in _prob_codes])
prob_colors = [code_info[c]['color'] for c in _prob_codes]

# ノイズ振幅（生成時と同じ値をmanifestから取得）→ 尤度計算のσに使う
noise_amplitude = float(manifest_df.iloc[0]['noise_amplitude'])

# アセンブリパネル用の基準配列（元配列。例: "IPP"）
reference_sequence = str(manifest_df.iloc[0]['sequence'])

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

# ============================
# セグメントごとの「アサイン確信度」を事前計算
# 全候補コードでsoftmaxを取ると、reference_sequenceで実際には使わない
# コードも含めて確率が分散してしまい、正解コードでも確信度が低く出がち
# （候補が多いほど不利になる）。そこで、候補全体でのsoftmaxではなく、
# 「正解コード」と「その次に尤もらしかった1つの競合コード」の
# "2択"だけで確信度を計算する（二項ロジスティック / シグモイド）。
# 　confidence = sigmoid(LL_true - LL_best_rival)
# これなら無関係な候補が何個あっても影響を受けず、実際に紛らわしい
# 相手とどれだけ差がついたかだけで確信度が決まる。
#
# さらに、正解コードはアラインメント（文字列一致）で既に確定している
# ため、「その回のノイズでたまたま競合コードの方が尤もらしく見えた
# （confidence<0.5）」としても、それは「間違っている証拠」ではなく
# 単に「その1回の読みでは情報が弱かった」というだけ。ベルヌーイ試行
# ベースの多数決モデル（e<0.5を仮定するCondorcetの陪審定理）と同じ
# 前提に合わせるため、confidenceは0.5を下限としてクリップする。
# これにより対数オッズへの寄与は必ず0以上になり、Depthが増えるほど
# 単調にコンセンサス精度が上がっていく（マイナスに転じない）。
# ============================
seg_end_idx = np.empty_like(seg_start_idx)
seg_end_idx[:-1] = seg_start_idx[1:] - 1
seg_end_idx[-1] = n_total - 1

_code_to_prob_idx = {c: i for i, c in enumerate(prob_codes)}
seg_confidence = np.full(n_seg, np.nan)
_seg_rival_code = np.full(n_seg, '', dtype=object)  # 診断用: どのコードに負けたか
_seg_raw_conf = np.full(n_seg, np.nan)               # 診断用: クリップ前の確信度
_CONF_FLOOR = 0.5  # 二択ロジスティックの下限（これを下回る分は「情報なし」扱い）

for _si in range(n_seg):
    _code = seg_codes[_si]
    if _code not in _code_to_prob_idx:
        continue  # 基準値が無いコード（'B'など）は確信度の対象外
    _s, _e = seg_start_idx[_si], seg_end_idx[_si]
    _obs_mean = float(all_raw[_s:_e + 1].mean())
    _n_obs = max(_e - _s + 1, 1)
    _sigma = max(noise_amplitude / np.sqrt(_n_obs), 1e-4)
    _log_lik = -0.5 * ((_obs_mean - prob_values) / _sigma) ** 2

    _true_idx = _code_to_prob_idx[_code]
    _ll_true = _log_lik[_true_idx]
    if len(prob_codes) > 1:
        _rest_idx = np.delete(np.arange(len(prob_codes)), _true_idx)
        _best_rival_pos = _rest_idx[np.argmax(_log_lik[_rest_idx])]
        _ll_best_rival = float(_log_lik[_best_rival_pos])
        _seg_rival_code[_si] = str(prob_codes[_best_rival_pos])
    else:
        _ll_best_rival = -np.inf  # 候補が1つしかない場合は無条件で確信度1

    _raw_conf = _sigmoid(_ll_true - _ll_best_rival)
    _seg_raw_conf[_si] = _raw_conf
    seg_confidence[_si] = max(_raw_conf, _CONF_FLOOR)

# --- 診断用: 確信度の分布を確認（原因切り分けのため） ---
_valid_conf = seg_confidence[~np.isnan(seg_confidence)]
if len(_valid_conf) > 0:
    print(
        f"[diag] 候補コード数={len(prob_codes)}   "
        f"seg_confidence(2択版・0.5下限クリップ後): min={_valid_conf.min():.3f} "
        f"median={np.median(_valid_conf):.3f} "
        f"mean={_valid_conf.mean():.3f} "
        f"max={_valid_conf.max():.3f}"
    )

# --- 診断用: 基準値(R_conductance)の一覧をソートして表示 ---
# 近接している(差がnoise_amplitudeに対して小さい)コード同士は、原理的に
# 混同されやすい（物理的に紛らわしい）ペアである可能性が高い。
print(f"[diag] noise_amplitude = {noise_amplitude:.4f}")
_order = np.argsort(prob_values)
print("[diag] R_conductance 一覧（昇順）と隣接コードとの差:")
for _k in range(len(_order)):
    _oi = _order[_k]
    _c, _v = prob_codes[_oi], prob_values[_oi]
    if _k > 0:
        _prev_v = prob_values[_order[_k - 1]]
        _gap = _v - _prev_v
        _gap_str = f"  (直前との差: {_gap:.4f} / noise比: {_gap / noise_amplitude:.2f})"
    else:
        _gap_str = ""
    print(f"    {_c}: {_v:.4f}{_gap_str}")

# --- 診断用: 実際にreference_sequenceで使われているコードごとに
#     「本当の確信度」と「誰に負け続けているか」を集計 ---
print("[diag] reference_sequenceで使われるコードごとの内訳:")
for _c in sorted(set(reference_sequence)):
    _mask = (seg_codes == _c) & ~np.isnan(_seg_raw_conf)
    if not np.any(_mask):
        continue
    _raws = _seg_raw_conf[_mask]
    _rivals = _seg_rival_code[_mask]
    _rival_vals, _rival_counts = np.unique(_rivals[_rivals != ''], return_counts=True)
    _top_rival = _rival_vals[np.argmax(_rival_counts)] if len(_rival_vals) > 0 else '-'
    _lose_rate = float((_raws < 0.5).mean()) * 100
    print(
        f"    {_c}: 生confidence mean={_raws.mean():.3f}  "
        f"0.5未満の割合={_lose_rate:.1f}%  "
        f"最頻の競合相手={_top_rival}"
    )




def visible_segment_range(t_start, t_now):
    """[t_start, t_now] に重なるセグメントのインデックス範囲 [lo, hi) を返す"""
    lo = np.searchsorted(seg_end_t, t_start, side='right')
    hi = np.searchsorted(seg_start_t, t_now, side='right')
    return lo, max(hi, lo)


def current_segment_index(t):
    idx = np.searchsorted(seg_start_t, t, side='right') - 1
    return int(np.clip(idx, 0, n_seg - 1))


# ============================
# 読み取り断片（フラグメント）の検出 & 元配列へのアラインメント
# ============================
# 'B'（ベースライン）で区切られた、連続する非Bセグメントのまとまりを
# 1つの「読み取り断片」とみなす（例: "I","P" が連続していれば断片 "IP"）。
# 各断片は生成時の仕組み上、必ず reference_sequence かその逆順のどこかの
# 部分文字列と完全一致するので、文字列検索でそのまま位置を特定できる。
# 同じ文字が配列中に複数回出現する場合（例: "IPP" の P）は一致位置が複数
# あり得るため、その候補の中からランダムに1つを選んで割り当てる
# （実際のアセンブリでも短い/曖昧な読みは複数箇所に整合し得ることの再現）。
_ref_fwd = reference_sequence
_ref_rev = reference_sequence[::-1]
_ref_len = len(reference_sequence)
_frag_rng = np.random.default_rng()


def _find_all_occurrences(haystack, needle):
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


_i = 0
fragments = []
while _i < n_seg:
    if seg_codes[_i] == 'B':
        _i += 1
        continue
    _j = _i
    frag_chars = []
    frag_confs = []
    while _j < n_seg and seg_codes[_j] != 'B':
        frag_chars.append(seg_codes[_j])
        frag_confs.append(seg_confidence[_j])
        _j += 1
    frag_str = ''.join(frag_chars)

    fwd_positions = _find_all_occurrences(_ref_fwd, frag_str)
    rev_positions = _find_all_occurrences(_ref_rev, frag_str)
    candidates = [('fwd', p) for p in fwd_positions] + [('rev', p) for p in rev_positions]

    if candidates:
        orientation, pos = candidates[_frag_rng.integers(0, len(candidates))]
        if orientation == 'fwd':
            align_start, align_end = pos, pos + len(frag_str)
        else:
            align_start = _ref_len - pos - len(frag_str)
            align_end = _ref_len - pos
    else:
        orientation = None  # 理論上は起きないはずだが念のため
        align_start = align_end = None

    # 文字ごとの確信度を、対応する参照配列上の位置と対にしておく
    # （'rev'の場合は時間順とゲノム座標順が逆になる点に注意）
    if align_start is not None:
        if orientation == 'fwd':
            pos_conf = list(zip(range(align_start, align_end), frag_confs))
        else:
            pos_conf = list(zip(range(align_end - 1, align_start - 1, -1), frag_confs))
    else:
        pos_conf = []

    fragments.append({
        'text': frag_str,
        'start_t': seg_start_t[_i],
        'end_t': seg_end_t[_j - 1],
        'align_start': align_start,
        'align_end': align_end,
        'orientation': orientation,
        'pos_conf': pos_conf,
        'length': len(frag_str),
        'mean_conf': float(np.mean(frag_confs)) if len(frag_confs) > 0 else np.nan,
    })
    _i = _j

fragments.sort(key=lambda f: f['end_t'])
frag_end_times = np.array([f['end_t'] for f in fragments])
n_frag = len(fragments)


# ============================
# アニメーションパラメータ
# ============================
window_width = 0.3      # 表示する時間幅
step = 5                # 1フレームで進めるデータ点数
interval_ms = 20        # フレーム間隔(ms)
playhead_frac = 0.8     # 窓の中で再生ヘッドを置く位置（0=左端, 1=右端）
min_label_frac = 0.02   # セグメント幅がこの割合未満ならラベル文字を省略
pulse_speed = 0.25       # 現在セグメントの明滅（パルス）の速さ

save_as_gif = False
gif_fps = 30

# ============================
# 描画準備
# ============================
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[3.4, 1.1], height_ratios=[5, 1, 0.9], hspace=0.35, wspace=0.28)
ax_wave = fig.add_subplot(gs[0, 0])
ax_track = fig.add_subplot(gs[1, 0], sharex=ax_wave)
ax_trail = fig.add_subplot(gs[2, 0])
ax_prob = fig.add_subplot(gs[0:2, 1])

# --- 波形パネル ---
line_raw, = ax_wave.plot([], [], linewidth=0.8, alpha=0.6, color='#4FC3F7', label='Raw Signal')
line_assigned, = ax_wave.plot([], [], linewidth=1.8, color='#FFFFFF', label='Assigned Signal')

# Raw信号に重ねる「電子ノイズ」風の点滅パーティクル
_particle_rng = np.random.default_rng()
n_particles = 45
particle_scatter = ax_wave.scatter(
    [], [], s=[], c='#B3E5FC', alpha=0.85, zorder=6, edgecolors='none',
)

margin = 0.1 * (all_raw.max() - all_raw.min() + 1e-9)
ax_wave.set_ylim(all_raw.min() - margin, all_raw.max() + margin)
ax_wave.set_ylabel('Value')
ax_wave.grid(True, color=GRID_COLOR, alpha=0.5)
ax_wave.legend(loc='upper right', framealpha=0.3)
plt.setp(ax_wave.get_xticklabels(), visible=False)
title = ax_wave.set_title('', fontsize=11, fontfamily=FONT_MONO)

# プレイヘッドを「グロー（発光）」に見せるため、太さ違いの半透明レイヤーを重ねる
playhead_wave_glow = [
    ax_wave.axvline(0, color=ACCENT_COLOR, linewidth=w, alpha=a, zorder=9, solid_capstyle='round')
    for w, a in [(10, 0.05), (6, 0.09), (3, 0.18)]
]
playhead_wave_core = ax_wave.axvline(0, color=ACCENT_COLOR, linewidth=1.1, alpha=0.9, zorder=10)

# 波形とAssigned信号の交点＝「今まさに判定されている値」を示すマーカー
marker_glow = ax_wave.scatter([0], [0], s=500, color=ACCENT_COLOR, alpha=0.12, zorder=11, linewidths=0)
marker_ring = ax_wave.scatter([0], [0], s=160, color=ACCENT_COLOR, alpha=0.30, zorder=11, linewidths=0)
marker_core = ax_wave.scatter(
    [0], [0], s=55, color=ACCENT_COLOR, edgecolor='white', linewidth=1.2, zorder=12,
)

# 波形上部に「スキャナーの照準」のようなキャレットマーカー
scan_caret = ax_wave.scatter([0], [0], marker='v', s=90, color=ACCENT_COLOR, zorder=13)

# --- 配列トラックパネル ---
ax_track.set_ylim(0, 1)
ax_track.set_yticks([])
ax_track.set_xlabel('Time')
ax_track.grid(False)

playhead_track_glow = [
    ax_track.axvline(0, color=ACCENT_COLOR, linewidth=w, alpha=a, zorder=9, solid_capstyle='round')
    for w, a in [(10, 0.05), (6, 0.09), (3, 0.18)]
]
playhead_track_core = ax_track.axvline(0, color=ACCENT_COLOR, linewidth=1.1, alpha=0.9, zorder=10)

# --- 読み取り履歴トレイルパネル（字幕のように過去に読んだコードが薄く流れていく） ---
trail_window = 8 * window_width  # どれだけ過去まで遡って表示するか（時間幅）

ax_trail.set_xlim(0, trail_window)
ax_trail.set_ylim(0, 1)
ax_trail.set_xticks([])
ax_trail.set_yticks([])
ax_trail.set_facecolor('#050505')
for spine in ax_trail.spines.values():
    spine.set_visible(False)
ax_trail.text(
    trail_window, 1.15, 'READ HISTORY', ha='right', va='bottom',
    fontsize=8, color='#666666', fontfamily=FONT_MONO,
)
ax_trail.axvline(trail_window, color=ACCENT_COLOR, alpha=0.4, linewidth=1.0)
ax_trail.text(
    trail_window, -0.25, 'now', ha='right', va='top',
    fontsize=7, color=ACCENT_COLOR, alpha=0.6, fontfamily=FONT_MONO,
)

# 事前計算: 'B'（ベースライン）以外のセグメントだけを履歴対象にする
_nonB_mask = seg_codes != 'B'
trail_codes = seg_codes[_nonB_mask]
trail_start_t = seg_start_t[_nonB_mask]
trail_artists = []

# --- 確率密度（信頼度）パネル ---
# 「基準値との距離 → ガウス尤度 → 正規化して確率」というアナログな判定過程を
# 上位候補のランキング形式（動的に入れ替わるバーチャート）で可視化する。
n_prob_rows = 7  # 表示する上位候補の数

ax_prob.set_xlim(0, 1.0)
ax_prob.set_ylim(-0.6, n_prob_rows - 0.4)
ax_prob.invert_yaxis()  # 1位を一番上に
ax_prob.set_xticks([0, 0.5, 1.0])
ax_prob.set_xticklabels(['0%', '50%', '100%'], fontsize=8, color='#888888')
ax_prob.set_yticks([])
ax_prob.set_facecolor('#0A0A0A')
for spine in ax_prob.spines.values():
    spine.set_color('#333333')
ax_prob.grid(axis='x', color=GRID_COLOR, alpha=0.4)
ax_prob.set_title('CONFIDENCE', fontsize=10, fontfamily=FONT_MONO, color='#888888', pad=10)

prob_status_text = ax_prob.text(
    0.5, n_prob_rows - 0.55, '', ha='center', va='top',
    fontsize=8, fontfamily=FONT_MONO, color='#666666',
)

# パフォーマンス対策: 毎フレーム生成・削除せず、固定個数のオブジェクトを
# 使い回す（add_patch/removeの再計算コストを避けるため）
prob_bars = [
    mpatches.Rectangle((0, r - 0.32), 0.001, 0.64, facecolor='#888888', alpha=0.6, edgecolor='none', zorder=2)
    for r in range(n_prob_rows)
]
for b in prob_bars:
    ax_prob.add_patch(b)

prob_glow = mpatches.Rectangle((0, -0.36), 1.0, 0.72, fill=False, edgecolor=ACCENT_COLOR, linewidth=0, zorder=3)
ax_prob.add_patch(prob_glow)

prob_name_labels = [
    ax_prob.text(0.02, r, '', ha='left', va='center', fontsize=8.5,
                 fontfamily=FONT_MONO, color='#DDDDDD', zorder=4)
    for r in range(n_prob_rows)
]
prob_pct_labels = [
    ax_prob.text(0.985, r, '', ha='right', va='center', fontsize=8,
                 fontfamily=FONT_MONO, color='#888888', zorder=4)
    for r in range(n_prob_rows)
]

# --- 「現在読み取り中」表示パネル（HUD風） ---
_glow_stroke = [pe.withStroke(linewidth=4, foreground='black')]

readout_box = fig.text(
    0.13, 0.94, '', fontsize=24, fontweight='bold', fontfamily=FONT_MONO,
    color=ACCENT_COLOR, va='top', ha='left', path_effects=_glow_stroke,
    bbox=dict(boxstyle='round,pad=0.4', fc='#0A0A0A', ec=ACCENT_COLOR, lw=2.0, alpha=0.9),
)
readout_sub = fig.text(
    0.135, 0.825, '', fontsize=10, fontfamily=FONT_MONO,
    color='#BBBBBB', va='top', ha='left',
)
readout_val = fig.text(
    0.135, 0.788, '', fontsize=9, fontfamily=FONT_MONO,
    color='#888888', va='top', ha='left',
)

# ============================
# アセンブリパネル（別ウインドウ）
# 完成した読み取り断片を元配列にアラインメントし、pileup（重なり depth）として
# 積み上げて表示する。メインウインドウと同じ update() ループで同期させる。
# ============================
fig2 = plt.figure(figsize=(8.6, 6.4))
fig2.patch.set_facecolor('#000000')
fig2.suptitle('SEQUENCE ASSEMBLY', fontsize=12, fontfamily=FONT_MONO, color='#888888')

gs2 = fig2.add_gridspec(3, 1, height_ratios=[1, 2.6, 1.1], hspace=0.65)
ax_ref = fig2.add_subplot(gs2[0])
ax_depth = fig2.add_subplot(gs2[1], sharex=ax_ref)
ax_yield = fig2.add_subplot(gs2[2])

# --- 基準配列（reference）トラック ---
ax_ref.set_xlim(0, _ref_len)
ax_ref.set_ylim(0, 1)
ax_ref.set_xticks([])
ax_ref.set_yticks([])
for spine in ax_ref.spines.values():
    spine.set_visible(False)
ax_ref.set_facecolor('none')

ref_boxes = []
for _p, _c in enumerate(reference_sequence):
    _info = code_info.get(_c, {'color': '#888888', 'name': _c})
    _box = mpatches.FancyBboxPatch(
        (_p + 0.06, 0.1), 0.88, 0.8, boxstyle='round,pad=0.02,rounding_size=0.08',
        facecolor=_info['color'], edgecolor='white', linewidth=1.2, zorder=2,
    )
    ax_ref.add_patch(_box)
    ax_ref.text(
        _p + 0.5, 0.5, _c, ha='center', va='center', fontsize=16, fontweight='bold',
        fontfamily=FONT_MONO, color=contrasting_text_color(_info['color']), zorder=3,
    )
    ref_boxes.append(_box)
ax_ref.set_title('reference', fontsize=8, fontfamily=FONT_MONO, color='#666666', loc='left')

# 現在ハイライト中の断片を示す縁取り（毎フレーム位置とアルファを更新）
ref_highlight = mpatches.FancyBboxPatch(
    (0, 0.05), 1, 0.9, boxstyle='round,pad=0.02,rounding_size=0.10',
    facecolor='none', edgecolor=ACCENT_COLOR, linewidth=0, zorder=4,
)
ax_ref.add_patch(ref_highlight)

# --- カバレッジ深度（depth）チャート ---
ax_depth.set_xlim(0, _ref_len)
ax_depth.set_ylim(0, 5)
ax_depth.set_xticks(np.arange(_ref_len) + 0.5)
ax_depth.set_xticklabels(list(reference_sequence), fontfamily=FONT_MONO, fontsize=10)
ax_depth.set_ylabel('depth', fontsize=9, color='#888888')
ax_depth.grid(axis='y', color=GRID_COLOR, alpha=0.4)
ax_depth.set_facecolor('#0A0A0A')
for spine in ax_depth.spines.values():
    spine.set_color('#333333')

depth_counts = np.zeros(_ref_len)
depth_bars = ax_depth.bar(
    np.arange(_ref_len) + 0.5, depth_counts, width=0.7,
    color=[code_info.get(c, {'color': '#888888'})['color'] for c in reference_sequence],
    edgecolor='white', linewidth=0.8, alpha=0.85, zorder=2,
)

# --- コンセンサス精度（達成精度）の折れ線 ---
# Depth（本数）だけでなく、各リードのアサイン確信度を対数オッズで積み上げた
# 「その位置の判定がどれだけ正しそうか」を0〜100%の折れ線として重ねる。
depth_logodds = np.zeros(_ref_len)  # 位置ごとに対数オッズを蓄積（コンセンサス精度の元）

ax_acc = ax_depth.twinx()
ax_acc.set_ylim(0, 1.02)
ax_acc.set_yticks([0, 0.5, 1.0])
ax_acc.set_yticklabels(['0%', '50%', '100%'], fontsize=8, color=ACCENT_COLOR)
ax_acc.tick_params(axis='y', colors=ACCENT_COLOR)
ax_acc.set_ylabel('accuracy', fontsize=9, color=ACCENT_COLOR)
for spine in ax_acc.spines.values():
    spine.set_visible(False)

acc_x = np.arange(_ref_len) + 0.5
acc_line, = ax_acc.plot(
    acc_x, np.full(_ref_len, np.nan), color=ACCENT_COLOR, marker='o', markersize=4,
    linewidth=1.4, alpha=0.9, zorder=5, label='consensus accuracy',
)
ax_acc.axhline(0.9, color=ACCENT_COLOR, linestyle=':', linewidth=1.0, alpha=0.3)

# --- Yield / スループット（累積アサイン塩基数の推移） ---
ax_yield.set_xlabel('time', fontsize=9, color='#888888')
ax_yield.set_ylabel('yield (bases)', fontsize=9, color='#888888')
ax_yield.set_facecolor('#0A0A0A')
ax_yield.grid(True, color=GRID_COLOR, alpha=0.4)
ax_yield.tick_params(axis='both', labelsize=8, colors='#888888')
for spine in ax_yield.spines.values():
    spine.set_color('#333333')

_yield_t_hist = []  # 診断/描画用に時刻を蓄積
_yield_v_hist = []  # 累積アサイン塩基数を蓄積
yield_line, = ax_yield.plot(
    [], [], color='#4FC3F7', linewidth=1.6, alpha=0.9, zorder=3,
)
yield_head = ax_yield.scatter([0], [0], s=25, color=ACCENT_COLOR, zorder=4)

# --- 統計ダッシュボード（2段のテキスト行にまとめる） ---
assembly_status_text = fig2.text(
    0.02, 0.965, '', fontsize=9, fontfamily=FONT_MONO, color='#AAAAAA', va='top', ha='left',
)
assembly_status_text2 = fig2.text(
    0.02, 0.945, '', fontsize=9, fontfamily=FONT_MONO, color='#AAAAAA', va='top', ha='left',
)
latest_frag_text = fig2.text(
    0.98, 0.965, '', fontsize=9, fontfamily=FONT_MONO, color=ACCENT_COLOR, va='top', ha='right',
)

fig2.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.08)

# アセンブリの状態（フレームをまたいで保持する）
_next_frag_idx = 0
_reveal_time = -999.0
_reveal_frag = None
_frag_reveal_decay = 1.2  # ハイライトが消えるまでの時間幅
_cumulative_yield = 0     # Yield計算用: これまでにアサインされた塩基数の累積

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

    # 再生ヘッド位置。まだ生成済みデータの先端（t_now）を超えないようクランプする
    # （アニメーション開始直後、窓がまだデータで埋まっていない間の対策）
    playhead_t = min(t_start + window_width * playhead_frac, t_now)

    mask = (all_time >= t_start) & (all_time <= t_now)
    line_raw.set_data(all_time[mask], all_raw[mask])

    # Assigned Signal（白線）は再生ヘッドより右（＝まだ読み取っていない未来）は隠す。
    # 左側（読み取り済み）は今まで通り表示し続ける。
    mask_assigned = mask & (all_time <= playhead_t)
    line_assigned.set_data(all_time[mask_assigned], all_assigned[mask_assigned])

    # --- 電子ノイズ風パーティクル（毎フレームランダムに点滅） ---
    # Assigned Signalと同様、再生ヘッドより左（読み取り済み）側だけに表示する
    visible_idxs = np.where(mask_assigned)[0]
    if len(visible_idxs) > 0:
        n_pick = min(n_particles, len(visible_idxs))
        sel = _particle_rng.choice(visible_idxs, size=n_pick, replace=False)
        pts = np.column_stack([all_time[sel], all_raw[sel]])
        sizes = _particle_rng.uniform(3, 24, size=n_pick)
        alphas = _particle_rng.uniform(0.15, 0.95, size=n_pick)
        base_rgb = np.array([0.70, 0.90, 1.0])
        colors = np.column_stack([np.tile(base_rgb, (n_pick, 1)), alphas])
        particle_scatter.set_offsets(pts)
        particle_scatter.set_sizes(sizes)
        particle_scatter.set_facecolor(colors)
    else:
        particle_scatter.set_offsets(np.empty((0, 2)))

    ax_wave.set_xlim(t_start, t_start + window_width)
    ax_track.set_xlim(t_start, t_start + window_width)

    clear_dynamic_artists()

    # --- 波形背景の色分け & 配列トラックの色ブロック ---
    # どちらも「再生ヘッドより右（＝まだ読み取っていない未来）は一切見せない」で統一する。
    # セグメントがまだ始まっていなければ完全にスキップし、読み取り中のセグメントは
    # プレイヘッドの位置までだけを描画する（ネタバレなし）。
    lo, hi = visible_segment_range(t_start, t_now)
    for si in range(lo, hi):
        code = seg_codes[si]
        info = code_info.get(code, {'name': code, 'color': '#888888'})
        color = info['color']
        seg_l = max(seg_start_t[si], t_start)
        seg_r_full = min(seg_end_t[si], t_start + window_width)

        if seg_l >= playhead_t:
            continue  # まだ再生ヘッドが到達していない区間は完全に非表示

        seg_r = min(seg_r_full, playhead_t)
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

    # 再生ヘッドより右側の「未読ゾーン」を軽く暗く塗って、境界をより分かりやすくする
    unread_l = max(playhead_t, t_start)
    unread_r = t_start + window_width
    if unread_r > unread_l:
        unread_span1 = ax_wave.axvspan(unread_l, unread_r, color='black', alpha=0.15, lw=0, zorder=0.1)
        unread_span2 = ax_track.axvspan(unread_l, unread_r, color='black', alpha=0.35, lw=0, zorder=0.1)
        dynamic_artists.extend([unread_span1, unread_span2])

    # --- run境界の縦線 ---
    for run_idx, start_t in run_boundaries:
        if t_start <= start_t <= t_start + window_width:
            ln1 = ax_wave.axvline(start_t, color='#888888', linestyle='--', alpha=0.5, zorder=3)
            ln2 = ax_track.axvline(start_t, color='#888888', linestyle='--', alpha=0.5, zorder=3)
            dynamic_artists.extend([ln1, ln2])

    # --- 読み取り履歴トレイル（字幕のように過去に読んだコードが右から左へ薄く流れる） ---
    for artist in trail_artists:
        artist.remove()
    trail_artists.clear()

    trail_hi = np.searchsorted(trail_start_t, playhead_t, side='right')
    trail_lo = np.searchsorted(trail_start_t, playhead_t - trail_window, side='left')
    edge_margin = trail_window * 0.02  # 右端でラベルが軸境界に切れないための余白
    for ti in range(trail_lo, trail_hi):
        code = trail_codes[ti]
        elapsed = playhead_t - trail_start_t[ti]
        if elapsed < 0 or elapsed > trail_window:
            continue
        x = trail_window - max(elapsed, edge_margin)  # 新しいほど右、古いほど左
        age_frac = elapsed / trail_window
        alpha = float(np.clip(1.0 - age_frac, 0.05, 1.0))
        fontsize = 13 - 4 * age_frac
        info = code_info.get(code, {'color': '#AAAAAA'})
        txt = ax_trail.text(
            x, 0.5, code, ha='center', va='center', clip_on=False,
            fontsize=fontsize, fontweight='bold', fontfamily=FONT_MONO,
            color=info['color'], alpha=alpha, zorder=2,
        )
        trail_artists.append(txt)

    # --- 現在読み取り中のセグメントをパルス（明滅）ハイライト ---
    pulse = 0.5 + 0.5 * np.sin(frame * pulse_speed)
    cur_seg_idx = current_segment_index(playhead_t)
    cur_code = seg_codes[cur_seg_idx]
    if cur_code != 'B':
        hl_l = max(seg_start_t[cur_seg_idx], t_start)
        hl_r = min(seg_end_t[cur_seg_idx], t_start + window_width)
        hl1 = ax_wave.axvspan(hl_l, hl_r, color='white', alpha=0.10 * pulse, zorder=0.5)
        hl2 = ax_track.axvspan(hl_l, hl_r, color='white', alpha=0.28 * pulse, zorder=1.5)
        border = mpatches.Rectangle(
            (hl_l, 0), hl_r - hl_l, 1, fill=False,
            edgecolor=ACCENT_COLOR, linewidth=1.5 + pulse, alpha=0.6 + 0.4 * pulse, zorder=2.5,
        )
        ax_track.add_patch(border)
        dynamic_artists.extend([hl1, hl2, border])

    # --- 確率密度（信頼度）パネル ---
    # 現在のセグメント開始〜プレイヘッドまでの生データ平均を「観測値」とし、
    # 各候補コードの基準値とのガウス尤度を正規化して確率を得る。
    # サンプル数が増えるほど標準誤差が縮み、確率がシャープに収束していく。
    seg_l_cur = seg_start_t[cur_seg_idx]
    obs_mask = (all_time >= seg_l_cur) & (all_time <= playhead_t)
    n_obs = int(np.count_nonzero(obs_mask))
    if n_obs > 0:
        obs_mean = float(all_raw[obs_mask].mean())
    else:
        obs_mean = float(np.interp(playhead_t, all_time, all_raw))
        n_obs = 1
    sigma = max(noise_amplitude / np.sqrt(n_obs), 1e-4)

    log_lik = -0.5 * ((obs_mean - prob_values) / sigma) ** 2
    log_lik -= log_lik.max()
    weights = np.exp(log_lik)
    probs = weights / weights.sum()

    order = np.argsort(-probs)[:n_prob_rows]
    top_probs = probs[order]
    top_codes = prob_codes[order]
    top_colors = [prob_colors[i] for i in order]

    for rank in range(n_prob_rows):
        bar = prob_bars[rank]
        name_lbl = prob_name_labels[rank]
        pct_lbl = prob_pct_labels[rank]

        if rank >= len(top_probs):
            bar.set_width(0.0)
            name_lbl.set_text('')
            pct_lbl.set_text('')
            continue

        p = float(top_probs[rank])
        code = top_codes[rank]
        color = top_colors[rank]
        is_top = (rank == 0)

        bar.set_bounds(0, rank - 0.32, max(p, 0.002), 0.64)
        bar.set_facecolor(color)
        bar.set_alpha(0.85 if is_top else 0.55)

        name = code_info.get(code, {'name': code}).get('name', code)
        label_color = contrasting_text_color(color) if p > 0.22 else '#DDDDDD'
        label_x = 0.02 if p > 0.22 else min(p + 0.03, 0.97)
        name_lbl.set_position((label_x, rank))
        name_lbl.set_text(f"{code} {name}")
        name_lbl.set_color(label_color)
        name_lbl.set_fontweight('bold' if is_top else 'normal')

        pct_lbl.set_text(f"{p * 100:4.1f}%")
        pct_lbl.set_color(ACCENT_COLOR if is_top else '#888888')

    if len(top_probs) > 0:
        prob_glow.set_bounds(0, -0.36, 1.0, 0.72)
        prob_glow.set_linewidth(1.5 + pulse)
        prob_glow.set_alpha(0.5 + 0.4 * pulse)
        top1 = float(top_probs[0])
    else:
        prob_glow.set_linewidth(0)
        top1 = 0.0

    locked = top1 > 0.9
    status = f"n={n_obs}  \u03c3={sigma:.3f}"
    if cur_code != 'B':
        status += "   \u2713 LOCKED" if locked else "   ...reading"
    prob_status_text.set_text(status)
    prob_status_text.set_color(ACCENT_COLOR if locked else '#666666')

    # --- 再生ヘッド（グロー含む全レイヤーを同期させる） ---
    for ln in playhead_wave_glow + [playhead_wave_core]:
        ln.set_xdata([playhead_t, playhead_t])
    for ln in playhead_track_glow + [playhead_track_core]:
        ln.set_xdata([playhead_t, playhead_t])

    # --- 判定マーカー（波形とAssigned信号の交点） ---
    # Assigned信号は区間内で一定値なので、線形補間ではなくセグメントの値をそのまま使う
    # （境界ちょうどの浮動小数点誤差で値がぶれるのを防ぐため）
    current_value = float(all_assigned[seg_start_idx[cur_seg_idx]])
    info = code_info.get(cur_code, {'name': cur_code, 'color': ACCENT_COLOR, 'description': ''})
    marker_color = info['color'] if cur_code != 'B' else ACCENT_COLOR

    for m in (marker_glow, marker_ring, marker_core):
        m.set_offsets([[playhead_t, current_value]])
    marker_glow.set_color(marker_color)
    marker_ring.set_color(marker_color)
    marker_core.set_facecolor(marker_color)

    y_top = ax_wave.get_ylim()[1]
    scan_caret.set_offsets([[playhead_t, y_top - 0.03 * (y_top - ax_wave.get_ylim()[0])]])
    scan_caret.set_color(marker_color)

    # --- 判定マーカー → 読み取りパネルへのリーダーライン（点線） ---
    leader = ax_wave.annotate(
        '', xy=(playhead_t, current_value), xycoords='data',
        xytext=(0.035, 0.965), textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='-', color=marker_color, alpha=0.55, linestyle=(0, (2, 2)),
            linewidth=1.3, connectionstyle='arc3,rad=-0.25', shrinkA=0, shrinkB=6,
        ),
        zorder=8,
    )
    dynamic_artists.append(leader)

    # --- 現在読み取り中のコードを表示 ---
    if cur_code == 'B':
        readout_box.set_text('—')
        readout_box.set_color('#888888')
        readout_box.get_bbox_patch().set_edgecolor('#888888')
        readout_sub.set_text('baseline')
        readout_val.set_text('')
    else:
        readout_box.set_text(f"{cur_code}  {info['name']}")
        readout_box.set_color(info['color'])
        readout_box.get_bbox_patch().set_edgecolor(info['color'])
        readout_sub.set_text(info.get('description', ''))
        readout_val.set_text(f"value \u2248 {current_value:.3f}")

    current_run = 0
    for run_idx, start_t in run_boundaries:
        if start_t <= t_now:
            current_run = run_idx
    title.set_text(f"run {current_run} / {len(manifest_df) - 1}   |   time = {t_now:.3f}")

    # ============================
    # アセンブリパネル（別ウインドウ fig2）の更新
    # ============================
    global _next_frag_idx, _reveal_time, _reveal_frag, _cumulative_yield

    while _next_frag_idx < n_frag and frag_end_times[_next_frag_idx] <= playhead_t:
        frag = fragments[_next_frag_idx]
        _cumulative_yield += frag['length']
        if frag['align_start'] is not None:
            depth_counts[frag['align_start']:frag['align_end']] += 1
            for _pos, _conf in frag['pos_conf']:
                if not np.isnan(_conf):
                    depth_logodds[_pos] += _logit(_conf)
            _reveal_time = frag['end_t']
            _reveal_frag = frag
        _next_frag_idx += 1

    for bar, h in zip(depth_bars, depth_counts):
        bar.set_height(h)
    ax_depth.set_ylim(0, max(5.0, float(depth_counts.max()) * 1.25))

    # --- コンセンサス精度（対数オッズ合算 → シグモイドで確率に戻す） ---
    covered = depth_counts > 0
    consensus_accuracy = _sigmoid(depth_logodds)
    acc_y = np.where(covered, consensus_accuracy, np.nan)
    acc_line.set_data(acc_x, acc_y)
    mean_accuracy = float(consensus_accuracy[covered].mean()) if covered.any() else 0.0
    mean_q = _prob_to_qscore(mean_accuracy) if mean_accuracy > 0 else 0.0

    # --- Depthの均一性（変動係数 CV = 標準偏差 / 平均。小さいほどムラが少ない） ---
    if covered.sum() >= 2:
        _d = depth_counts[covered]
        depth_cv = float(_d.std() / _d.mean())
    else:
        depth_cv = 0.0

    # --- N50（読み取り断片の長さの代表値） ---
    _lengths = np.array([f['length'] for f in fragments[:_next_frag_idx]])
    if len(_lengths) > 0:
        _sorted_len = np.sort(_lengths)[::-1]
        _cum = np.cumsum(_sorted_len)
        _half = _cum[-1] / 2.0
        n50 = int(_sorted_len[np.searchsorted(_cum, _half)])
    else:
        n50 = 0

    # --- Pass率（PASS_Q_THRESHOLDを超えたリードの割合） ---
    _confs = np.array([
        f['mean_conf'] for f in fragments[:_next_frag_idx] if not np.isnan(f['mean_conf'])
    ])
    if len(_confs) > 0:
        pass_rate = float((_confs >= PASS_CONF_THRESHOLD).mean()) * 100
    else:
        pass_rate = 0.0

    # --- Yield / スループット（時間 vs 累積アサイン塩基数） ---
    _yield_t_hist.append(playhead_t)
    _yield_v_hist.append(_cumulative_yield)
    yield_line.set_data(_yield_t_hist, _yield_v_hist)
    ax_yield.set_xlim(0, max(playhead_t * 1.05, 1e-3))
    ax_yield.set_ylim(0, max(_cumulative_yield * 1.15, 5))
    yield_head.set_offsets([[playhead_t, _cumulative_yield]])

    if _reveal_frag is not None:
        age = playhead_t - _reveal_time
        fade = float(np.clip(1.0 - age / _frag_reveal_decay, 0.0, 1.0))
        a_s, a_e = _reveal_frag['align_start'], _reveal_frag['align_end']
        ref_highlight.set_bounds(a_s, 0.05, a_e - a_s, 0.9)
        ref_highlight.set_linewidth(1.5 + 2.5 * fade)
        ref_highlight.set_alpha(fade)

        arrow = '\u2192' if _reveal_frag['orientation'] == 'fwd' else '\u2190'
        latest_frag_text.set_text(
            f"latest: {_reveal_frag['text']} {arrow}  "
            f"@ ref[{a_s}:{a_e}]"
        )
    assembly_status_text.set_text(
        f"reads assembled: {_next_frag_idx} / {n_frag}   "
        f"total depth: {int(depth_counts.sum())}   "
        f"yield: {_cumulative_yield} bases"
    )
    assembly_status_text2.set_text(
        f"mean accuracy: {mean_accuracy * 100:5.1f}%  (Q{mean_q:4.1f})   "
        f"N50: {n50}   "
        f"depth CV: {depth_cv:.2f}   "
        f"pass rate (\u2265Q{PASS_Q_THRESHOLD:.0f}): {pass_rate:5.1f}%"
    )

    fig2.canvas.draw_idle()

    return (
        [line_raw, line_assigned, particle_scatter, title]
        + playhead_wave_glow + [playhead_wave_core]
        + playhead_track_glow + [playhead_track_core]
        + [marker_glow, marker_ring, marker_core, scan_caret]
        + dynamic_artists + trail_artists
        + prob_bars + [prob_glow] + prob_name_labels + prob_pct_labels
        + [acc_line, yield_line, yield_head]
    )


n_frames = n_total // step + 1

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=interval_ms, blit=False, repeat=False,
)

fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08)


# ============================
# ウィンドウ位置の調整
# 2つのFigure（メイン画面 + アセンブリ画面）が画面上の同じ位置に開いて
# 重なってしまうのを防ぐため、それぞれ別のスクリーン座標に配置する。
# バックエンド（Qt/Tk/WX等）によって位置指定の方法が異なるため、
# 対応していない場合は何もせず無視する。
# ============================
def _position_figure_window(fig, x, y):
    try:
        mgr = fig.canvas.manager
        backend = plt.get_backend().lower()
        window = getattr(mgr, 'window', None)
        if window is None:
            return
        if 'qt' in backend:
            window.move(x, y)
        elif 'tk' in backend:
            window.wm_geometry(f"+{x}+{y}")
        elif 'wx' in backend:
            window.SetPosition((x, y))
    except Exception:
        pass  # 対応していないバックエンド／環境では何もしない


_position_figure_window(fig, 40, 40)     # メインウインドウ: 左上
_position_figure_window(fig2, 980, 60)   # アセンブリウインドウ: その右側

if save_as_gif:
    out_path = save_dir / 'streaming_signal_v2.gif'
    ani.save(out_path, writer='pillow', fps=gif_fps)
    print(f"アニメーションを保存しました: {out_path}")
else:
    plt.show()
