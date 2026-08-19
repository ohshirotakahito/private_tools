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
import matplotlib.colors as mcolors

# ============================
# 見た目パラメータ
# ============================
plt.style.use('dark_background')

ACCENT_COLOR = '#39FF14'      # 再生ヘッドなどのネオン色
ACCENT_COLOR2 = '#00E5FF'     # サブアクセント（シアン系）
GRID_COLOR = '#2A2A2A'
FONT_MONO = 'DejaVu Sans Mono'

# --- 展示会向け: 遠目からでも見やすいよう文字/線を全体的に太く大きく ---
FS_TITLE = 20      # 大見出し
FS_HUD = 34         # 読み取り中コードのHUD表示
FS_SUB = 14         # HUDの補足行
FS_LABEL = 13       # 軸ラベル・パネルタイトル
FS_TICK = 11        # 目盛りラベル
FS_TRACK = 13       # 配列トラックの文字ラベル
FS_STATUS = 12      # ステータス行（reads assembled等）
FS_LEGEND = 12

LW_THICK = 2.6      # 主要ライン
LW_MED = 1.8
_GLOW_STROKE = [pe.withStroke(linewidth=5, foreground='black')]  # 発光文字の縁取り


def _glow_line(ax, x_or_fn, color, base_lw, base_alpha=0.9, layers=((9, 0.06), (5, 0.12), (2.5, 0.22))):
    """ネオン管のような発光ラインを作るため、太さ違いの半透明レイヤーを重ねて返す"""
    glow_artists = [
        ax.axvline(0, color=color, linewidth=w, alpha=a, zorder=9, solid_capstyle='round')
        for w, a in layers
    ]
    core = ax.axvline(0, color=color, linewidth=base_lw, alpha=base_alpha, zorder=10)
    return glow_artists, core


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

# --- クオリティ値 → 信号色（赤〜黄〜緑）のグラデーション ---
QUALITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'quality', ['#FF3B30', '#FFD400', '#39FF14']
)


def _quality_color(accuracy_frac):
    """精度(0-1)を 赤(低)→黄(中)→緑(高) のグラデーション色に変換"""
    return QUALITY_CMAP(float(np.clip(accuracy_frac, 0.0, 1.0)))


def _draw_gauge(ax, value_frac, zones=((0.0, 0.9, '#FF3B30'), (0.9, 0.99, '#FFD400'), (0.99, 1.0, ACCENT_COLOR))):
    """0-1の値を、色分けされた半円メーター（速度計スタイル）で描画する。
    戻り値: (needle_line, center_text) — 毎フレームこれらを更新する。"""
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    # 色分けされた帯（外側の弧）
    for lo, hi, color in zones:
        theta1 = 180 * (1 - lo)
        theta2 = 180 * (1 - hi)
        wedge = mpatches.Wedge((0, 0), 1.0, theta2, theta1, width=0.22,
                                facecolor=color, edgecolor='none', alpha=0.85, zorder=2)
        ax.add_patch(wedge)
    # 内側の暗い背景円（針の土台）
    ax.add_patch(mpatches.Wedge((0, 0), 0.78, 0, 180, facecolor='#0A0A0A', edgecolor='none', zorder=1))

    needle_line, = ax.plot([0, 0], [0, 0.72], color='white', linewidth=3.2, zorder=5,
                            solid_capstyle='round')
    ax.add_patch(mpatches.Circle((0, 0), 0.06, facecolor='white', edgecolor=ACCENT_COLOR,
                                  linewidth=1.5, zorder=6))
    center_text = ax.text(0, 0.32, '', ha='center', va='center', fontsize=FS_HUD,
                           fontweight='bold', fontfamily=FONT_MONO, color='white', zorder=7,
                           path_effects=[pe.withStroke(linewidth=5, foreground='black')])
    ax.text(-1.05, -0.08, '0%', ha='left', va='top', fontsize=FS_TICK, color='#888888',
            fontfamily=FONT_MONO)
    ax.text(1.05, -0.08, '100%', ha='right', va='top', fontsize=FS_TICK, color='#888888',
            fontfamily=FONT_MONO)
    return needle_line, center_text


def _update_gauge(needle_line, center_text, value_frac, label_text=None, needle_color='white'):
    value_frac = float(np.clip(value_frac, 0.0, 1.0))
    angle = np.pi * (1 - value_frac)  # 0%→左端(180°) / 100%→右端(0°)
    needle_line.set_data([0, 0.72 * np.cos(angle)], [0, 0.72 * np.sin(angle)])
    needle_line.set_color(needle_color)
    center_text.set_text(label_text if label_text is not None else f"{value_frac * 100:.0f}%")


def _add_corner_brackets(fig, color=ACCENT_COLOR, size=0.028, lw=2.4, margin=0.012, alpha=0.8):
    """画面四隅にHUD風のコーナーブラケット（L字マーク）を追加する（展示会向けの演出）"""
    import matplotlib.lines as mlines
    corners = [
        (margin, 1 - margin, 1, -1), (1 - margin, 1 - margin, -1, -1),
        (margin, margin, 1, 1), (1 - margin, margin, -1, 1),
    ]
    for x, y, dx, dy in corners:
        fig.add_artist(mlines.Line2D([x, x + dx * size], [y, y], transform=fig.transFigure,
                                      color=color, linewidth=lw, alpha=alpha, zorder=50,
                                      solid_capstyle='round'))
        fig.add_artist(mlines.Line2D([x, x], [y, y + dy * size], transform=fig.transFigure,
                                      color=color, linewidth=lw, alpha=alpha, zorder=50,
                                      solid_capstyle='round'))


def _add_scanlines(fig, n_lines=90, color='white', alpha=0.018):
    """全画面にうっすらとした走査線テクスチャを敷く（SFモニター風の質感）"""
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-20)
    ax_bg.axis('off')
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    for y in np.linspace(0, 1, n_lines):
        ax_bg.axhline(y, color=color, alpha=alpha, linewidth=0.6)
    return ax_bg


def _add_live_indicator(fig, x, y, fontsize=FS_TICK):
    """点滅する赤丸+「LIVE」表示（毎フレームdotのalphaを更新して使う）"""
    dot = fig.add_artist(mpatches.Circle((x, y), 0.009, transform=fig.transFigure,
                                          facecolor='#FF3B30', edgecolor='none', zorder=30))
    label = fig.text(x + 0.018, y, 'LIVE', fontsize=fontsize, fontweight='bold',
                      fontfamily=FONT_MONO, color='#FF3B30', va='center', ha='left', zorder=30)
    return dot, label


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

# ============================
# 「区別困難ペア」の静的判定（警告バッジ用）
# 隣接コードとの値の差がnoise_amplitudeに対して小さい（ratio未満）場合、
# 物理的に混同されやすいコードとみなし、警告バッジの対象にする。
# ============================
CONFUSION_RATIO_THRESHOLD = 3.0  # このratio未満なら「紛らわしい」と判定
confusable_codes = {}  # code -> 最も近い競合コード名
for _ci, _c in enumerate(prob_codes):
    if _c not in set(reference_sequence):
        continue
    _others = np.delete(np.arange(len(prob_codes)), _ci)
    _dists = np.abs(prob_values[_others] - prob_values[_ci])
    _nearest_i = _others[np.argmin(_dists)]
    _nearest_gap = float(_dists.min())
    if _nearest_gap / noise_amplitude < CONFUSION_RATIO_THRESHOLD:
        confusable_codes[_c] = str(prob_codes[_nearest_i])
if confusable_codes:
    print(f"[diag] 区別困難と判定されたコード: {confusable_codes}")


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
fig = plt.figure(figsize=(18, 9.2))
gs = fig.add_gridspec(3, 2, width_ratios=[3.4, 1.1], height_ratios=[5, 1, 0.9], hspace=0.4, wspace=0.26)
ax_wave = fig.add_subplot(gs[0, 0])
ax_track = fig.add_subplot(gs[1, 0], sharex=ax_wave)
ax_trail = fig.add_subplot(gs[2, 0])
ax_prob = fig.add_subplot(gs[0:2, 1])

# --- 波形パネル ---
line_raw, = ax_wave.plot([], [], linewidth=1.1, alpha=0.55, color=ACCENT_COLOR2, label='Raw Signal')
line_assigned, = ax_wave.plot([], [], linewidth=LW_THICK, color='#FFFFFF', label='Assigned Signal',
                               path_effects=[pe.withStroke(linewidth=LW_THICK + 2.5, foreground='#0a3a2a')])

# Raw信号に重ねる「電子ノイズ」風の点滅パーティクル
_particle_rng = np.random.default_rng()
n_particles = 45
particle_scatter = ax_wave.scatter(
    [], [], s=[], c='#B3E5FC', alpha=0.85, zorder=6, edgecolors='none',
)

margin = 0.1 * (all_raw.max() - all_raw.min() + 1e-9)
ax_wave.set_ylim(all_raw.min() - margin, all_raw.max() + margin)
ax_wave.set_ylabel('Value', fontsize=FS_LABEL, fontfamily=FONT_MONO)
ax_wave.tick_params(axis='both', labelsize=FS_TICK)
ax_wave.grid(True, color=GRID_COLOR, alpha=0.5)
ax_wave.legend(loc='upper right', framealpha=0.25, fontsize=FS_LEGEND, edgecolor=GRID_COLOR)
plt.setp(ax_wave.get_xticklabels(), visible=False)
title = ax_wave.set_title(
    '', fontsize=FS_TITLE, fontfamily=FONT_MONO, color=ACCENT_COLOR, fontweight='bold',
    path_effects=_GLOW_STROKE,
)

# プレイヘッドを「グロー（発光）」に見せるため、太さ違いの半透明レイヤーを重ねる
playhead_wave_glow = [
    ax_wave.axvline(0, color=ACCENT_COLOR, linewidth=w, alpha=a, zorder=9, solid_capstyle='round')
    for w, a in [(14, 0.05), (9, 0.10), (4.5, 0.22)]
]
playhead_wave_core = ax_wave.axvline(0, color=ACCENT_COLOR, linewidth=1.6, alpha=0.95, zorder=10)

# 波形とAssigned信号の交点＝「今まさに判定されている値」を示すマーカー
marker_glow = ax_wave.scatter([0], [0], s=750, color=ACCENT_COLOR, alpha=0.12, zorder=11, linewidths=0)
marker_ring = ax_wave.scatter([0], [0], s=230, color=ACCENT_COLOR, alpha=0.32, zorder=11, linewidths=0)
marker_core = ax_wave.scatter(
    [0], [0], s=80, color=ACCENT_COLOR, edgecolor='white', linewidth=1.6, zorder=12,
)

# 波形上部に「スキャナーの照準」のようなキャレットマーカー
scan_caret = ax_wave.scatter([0], [0], marker='v', s=130, color=ACCENT_COLOR, zorder=13)

# --- 配列トラックパネル ---
ax_track.set_ylim(0, 1.6)
ax_track.set_yticks([])
ax_track.set_xlabel('Time', fontsize=FS_LABEL, fontfamily=FONT_MONO)
ax_track.tick_params(axis='x', labelsize=FS_TICK)
ax_track.grid(False)
ax_track.axhline(1.0, color=GRID_COLOR, linewidth=0.8, alpha=0.6, zorder=1)


playhead_track_glow = [
    ax_track.axvline(0, color=ACCENT_COLOR, linewidth=w, alpha=a, zorder=9, solid_capstyle='round')
    for w, a in [(14, 0.05), (9, 0.10), (4.5, 0.22)]
]
playhead_track_core = ax_track.axvline(0, color=ACCENT_COLOR, linewidth=1.6, alpha=0.95, zorder=10)

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
    trail_window, 1.18, 'READ HISTORY', ha='right', va='bottom',
    fontsize=FS_TICK, color='#888888', fontfamily=FONT_MONO, fontweight='bold',
)
ax_trail.axvline(trail_window, color=ACCENT_COLOR, alpha=0.5, linewidth=1.6)
ax_trail.text(
    trail_window, -0.3, 'now', ha='right', va='top',
    fontsize=FS_TICK, color=ACCENT_COLOR, alpha=0.75, fontfamily=FONT_MONO, fontweight='bold',
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
ax_prob.set_xticklabels(['0%', '50%', '100%'], fontsize=FS_TICK, color='#999999')
ax_prob.set_yticks([])
ax_prob.set_facecolor('#0A0A0A')
for spine in ax_prob.spines.values():
    spine.set_color('#3A3A3A')
ax_prob.grid(axis='x', color=GRID_COLOR, alpha=0.4)
ax_prob.set_title(
    'CONFIDENCE', fontsize=FS_LABEL + 1, fontfamily=FONT_MONO, color=ACCENT_COLOR,
    fontweight='bold', pad=12, path_effects=_GLOW_STROKE,
)

prob_status_text = ax_prob.text(
    0.5, n_prob_rows - 0.55, '', ha='center', va='top',
    fontsize=FS_TICK, fontfamily=FONT_MONO, color='#888888', fontweight='bold',
)

# パフォーマンス対策: 毎フレーム生成・削除せず、固定個数のオブジェクトを
# 使い回す（add_patch/removeの再計算コストを避けるため）
prob_bars = [
    mpatches.Rectangle((0, r - 0.34), 0.001, 0.68, facecolor='#888888', alpha=0.6, edgecolor='none', zorder=2)
    for r in range(n_prob_rows)
]
for b in prob_bars:
    ax_prob.add_patch(b)

prob_glow = mpatches.Rectangle((0, -0.36), 1.0, 0.72, fill=False, edgecolor=ACCENT_COLOR, linewidth=0, zorder=3)
ax_prob.add_patch(prob_glow)

prob_name_labels = [
    ax_prob.text(0.02, r, '', ha='left', va='center', fontsize=FS_TICK + 1, fontweight='bold',
                 fontfamily=FONT_MONO, color='#EEEEEE', zorder=4)
    for r in range(n_prob_rows)
]
prob_pct_labels = [
    ax_prob.text(0.985, r, '', ha='right', va='center', fontsize=FS_TICK, fontweight='bold',
                 fontfamily=FONT_MONO, color='#999999', zorder=4)
    for r in range(n_prob_rows)
]

# --- 展示会向けHUD演出（コーナーブラケット・走査線・LIVEインジケーター） ---
_add_scanlines(fig)
_add_corner_brackets(fig)
main_live_dot, main_live_label = _add_live_indicator(fig, 0.945, 0.99)

# --- ブランドタイトル（展示会向け・大きく発光させる） ---
brand_title = fig.text(
    0.5, 0.995, 'MOLECULE CALLER', fontsize=FS_TITLE + 12, fontweight='bold',
    fontfamily=FONT_MONO, color=ACCENT_COLOR, va='top', ha='center',
    path_effects=[pe.withStroke(linewidth=7, foreground='black')],
)
brand_subtitle = fig.text(
    0.5, 0.925, 'REAL-TIME NANOPORE SIGNAL DECODER', fontsize=FS_LABEL, fontweight='bold',
    fontfamily=FONT_MONO, color='#999999', va='top', ha='center',
    alpha=0.85,
)

# --- 「現在読み取り中」表示パネル（HUD風） ---
readout_box = fig.text(
    0.13, 0.86, '', fontsize=FS_HUD, fontweight='bold', fontfamily=FONT_MONO,
    color=ACCENT_COLOR, va='top', ha='left', path_effects=_GLOW_STROKE,
    bbox=dict(boxstyle='round,pad=0.5', fc='#0A0A0A', ec=ACCENT_COLOR, lw=2.6, alpha=0.92),
)
readout_sub = fig.text(
    0.137, 0.71, '', fontsize=FS_SUB + 1, fontfamily=FONT_MONO, fontweight='bold',
    color='#DDDDDD', va='top', ha='left',
)
readout_val = fig.text(
    0.137, 0.663, '', fontsize=FS_SUB, fontfamily=FONT_MONO,
    color=ACCENT_COLOR2, va='top', ha='left',
)

# HUD右上に出す警告バッジ（区別困難ペアを読み取り中の時だけ表示）
main_warn_circle = fig.add_artist(
    mpatches.Circle((0.395, 0.895), 0.014, transform=fig.transFigure,
                     facecolor='#FF3B30', edgecolor='white', linewidth=1.2, zorder=20)
)
main_warn_text = fig.text(
    0.395, 0.895, '!', fontsize=FS_LABEL - 1, fontweight='bold', color='white',
    ha='center', va='center', zorder=21,
)
main_warn_circle.set_visible(False)
main_warn_text.set_visible(False)

# ============================
# アセンブリパネル（別ウインドウ）
# 完成した読み取り断片を元配列にアラインメントし、pileup（重なり depth）として
# 積み上げて表示する。メインウインドウと同じ update() ループで同期させる。
# ============================
fig2 = plt.figure(figsize=(12, 18))
fig2.patch.set_facecolor('#000000')
_add_scanlines(fig2)
_add_corner_brackets(fig2)
fig2_live_dot, fig2_live_label = _add_live_indicator(fig2, 0.90, 0.99)
fig2.suptitle(
    'SEQUENCE ASSEMBLY', fontsize=FS_TITLE + 2, fontfamily=FONT_MONO, color=ACCENT_COLOR,
    fontweight='bold', path_effects=_GLOW_STROKE, y=0.99,
)

gs2 = fig2.add_gridspec(4, 1, height_ratios=[1, 1.5, 2.0, 1.0], hspace=0.75)
ax_ref = fig2.add_subplot(gs2[0])
ax_trace = fig2.add_subplot(gs2[1], sharex=ax_ref)
ax_depth = fig2.add_subplot(gs2[2], sharex=ax_ref)
ax_yield = fig2.add_subplot(gs2[3])

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
        facecolor=_info['color'], edgecolor='white', linewidth=1.6, zorder=2,
    )
    ax_ref.add_patch(_box)
    ax_ref.text(
        _p + 0.5, 0.5, _c, ha='center', va='center', fontsize=22, fontweight='bold',
        fontfamily=FONT_MONO, color=contrasting_text_color(_info['color']), zorder=3,
    )
    ref_boxes.append(_box)
ax_ref.set_title('reference', fontsize=FS_LABEL, fontfamily=FONT_MONO, color='#999999',
                  fontweight='bold', loc='left')

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
ax_depth.set_xticklabels(list(reference_sequence), fontfamily=FONT_MONO, fontsize=FS_TICK + 2,
                          fontweight='bold')
ax_depth.tick_params(axis='y', labelsize=FS_TICK)
ax_depth.set_ylabel('depth', fontsize=FS_LABEL, color='#AAAAAA', fontweight='bold')
ax_depth.grid(axis='y', color=GRID_COLOR, alpha=0.4)
ax_depth.set_facecolor('#0A0A0A')
for spine in ax_depth.spines.values():
    spine.set_color('#3A3A3A')

depth_counts = np.zeros(_ref_len)
depth_bars = ax_depth.bar(
    np.arange(_ref_len) + 0.5, depth_counts, width=0.7,
    color=[code_info.get(c, {'color': '#888888'})['color'] for c in reference_sequence],
    edgecolor='white', linewidth=1.2, alpha=0.9, zorder=2,
)

# --- コンセンサス精度（達成精度）を「塩基ごとのガウシアン波形」で表示 ---
# Depthバーと重なって見づらくならないよう、本物のクロマトグラムビューアのように
# 専用のトレースパネル(ax_trace)を独立させて表示する。
# 各リードのアサイン確信度を対数オッズで積み上げた「その位置の判定がどれだけ
# 正しそうか」を、コードごとに色分けされたガウシアンの山として表示する。
# 波のピーク高さ = 現在のコンセンサス精度、波の色 = そのコードの色。
depth_logodds = np.zeros(_ref_len)  # 位置ごとに対数オッズを蓄積（コンセンサス精度の元）

ax_trace.set_xlim(0, _ref_len)
ax_trace.set_ylim(0, 1.08)
ax_trace.set_yticks([0, 0.5, 1.0])
ax_trace.set_yticklabels(['0%', '50%', '100%'], fontsize=FS_TICK, color=ACCENT_COLOR, fontweight='bold')
ax_trace.tick_params(axis='y', colors=ACCENT_COLOR)
ax_trace.set_ylabel('accuracy', fontsize=FS_LABEL, color=ACCENT_COLOR, fontweight='bold')
ax_trace.set_facecolor('#050505')
plt.setp(ax_trace.get_xticklabels(), visible=False)
for spine in ax_trace.spines.values():
    spine.set_color('#3A3A3A')
ax_trace.set_title('CONSENSUS TRACE', fontsize=FS_TICK, fontfamily=FONT_MONO, color='#888888',
                    fontweight='bold', loc='left', pad=6)

acc_x = np.arange(_ref_len) + 0.5

# ガウシアン形状の共通カーネル（各塩基のセル幅=1.0の中に収まるように設計）
_GAUSS_SIGMA = 0.15
_gauss_x_local = np.linspace(-0.5, 0.5, 61)
_gauss_kernel = np.exp(-0.5 * (_gauss_x_local / _GAUSS_SIGMA) ** 2)

# 塩基ごとに色分けされた波形ライン（グロー+本体の2層）を1本ずつ用意し、
# 毎フレーム高さ(ピーク値)だけを更新する
acc_wave_lines = []
acc_wave_glow = []
for _pi in range(_ref_len):
    _code = reference_sequence[_pi]
    _wcolor = code_info.get(_code, {'color': ACCENT_COLOR})['color']
    _wx = _pi + 0.5 + _gauss_x_local
    _glow, = ax_trace.plot(_wx, np.zeros_like(_wx), color=_wcolor, linewidth=7, alpha=0.20,
                            zorder=4, solid_capstyle='round')
    _line, = ax_trace.plot(_wx, np.zeros_like(_wx), color=_wcolor, linewidth=LW_MED, alpha=0.95,
                            zorder=5, solid_capstyle='round')
    acc_wave_glow.append(_glow)
    acc_wave_lines.append(_line)

ax_trace.axhline(0.9, color=ACCENT_COLOR, linestyle=':', linewidth=1.4, alpha=0.35)

# --- Yield / スループット（累積アサイン塩基数の推移） ---
ax_yield.set_xlabel('time', fontsize=FS_LABEL, color='#AAAAAA', fontweight='bold')
ax_yield.set_ylabel('yield (bases)', fontsize=FS_LABEL, color='#AAAAAA', fontweight='bold')
ax_yield.set_facecolor('#0A0A0A')
ax_yield.grid(True, color=GRID_COLOR, alpha=0.4)
ax_yield.tick_params(axis='both', labelsize=FS_TICK, colors='#AAAAAA')
for spine in ax_yield.spines.values():
    spine.set_color('#3A3A3A')

_yield_t_hist = []  # 診断/描画用に時刻を蓄積
_yield_v_hist = []  # 累積アサイン塩基数を蓄積
yield_line, = ax_yield.plot(
    [], [], color=ACCENT_COLOR2, linewidth=LW_MED, alpha=0.95, zorder=3,
    path_effects=[pe.withStroke(linewidth=LW_MED + 2.5, foreground='#00303a')],
)
yield_head = ax_yield.scatter([0], [0], s=55, color=ACCENT_COLOR2, edgecolor='white',
                               linewidth=1.0, zorder=4)

# --- 精度のヒーロー表示: 半円ゲージ + トレンド矢印 ---
hero_label = fig2.text(
    0.5, 0.945, 'MEAN CONSENSUS ACCURACY', fontsize=FS_LABEL, fontweight='bold',
    fontfamily=FONT_MONO, color='#999999', va='top', ha='center', alpha=0.9,
)
ax_gauge = fig2.add_axes([0.30, 0.72, 0.40, 0.19])
gauge_needle, gauge_center_text = _draw_gauge(ax_gauge, 0.0)

hero_qsub = fig2.text(
    0.5, 0.685, '', fontsize=FS_SUB + 2, fontweight='bold', fontfamily=FONT_MONO,
    color=ACCENT_COLOR2, va='top', ha='center',
)
trend_text = fig2.text(
    0.5, 0.65, '', fontsize=FS_SUB, fontweight='bold', fontfamily=FONT_MONO,
    color='#AAAAAA', va='top', ha='center',
)

# トレンド計算用の履歴（時刻, mean_accuracy）を保持
_acc_history = []
TREND_WINDOW = 1.5  # どれだけ過去と比較してトレンドを見るか（時間幅）

# --- 警告バッジ（区別困難ペアを読み取り中の時だけ表示） ---
warn_badge_circle = fig2.add_artist(
    mpatches.Circle((0.73, 0.948), 0.016, transform=fig2.transFigure,
                     facecolor='#FF3B30', edgecolor='white', linewidth=1.2, zorder=20)
)
warn_badge_text = fig2.text(
    0.73, 0.948, '!', fontsize=FS_LABEL - 1, fontweight='bold', color='white',
    ha='center', va='center', zorder=21,
)
warn_badge_circle.set_visible(False)
warn_badge_text.set_visible(False)

# --- 副指標をKPIカードスタイルで表示（Geckoboard風タイル） ---
def _make_kpi_card(fig, x, y, w, h, label, value_color=ACCENT_COLOR2, fs_value=None,
                    accent=ACCENT_COLOR2):
    """角丸カード背景（上部アクセントバー付き） + ラベル(上) + 値(下) のKPIタイルを作る。
    値のTextオブジェクトを返す。"""
    fig.add_artist(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
        transform=fig.transFigure, facecolor='#111111', edgecolor=GRID_COLOR,
        linewidth=1.3, zorder=10,
    ))
    # 上部アクセントバー（カードごとの色でひと目でジャンル分けできるように）
    fig.add_artist(mpatches.FancyBboxPatch(
        (x + w * 0.06, y + h - 0.012), w * 0.88, 0.006, boxstyle='round,pad=0.0,rounding_size=0.003',
        transform=fig.transFigure, facecolor=accent, edgecolor='none', alpha=0.9, zorder=11,
    ))
    fig.text(x + w / 2, y + h - 0.024, label, ha='center', va='top', fontsize=FS_TICK - 1,
              fontweight='bold', fontfamily=FONT_MONO, color='#888888', zorder=11)
    value_text = fig.text(
        x + w / 2, y + h * 0.34, '', ha='center', va='center',
        fontsize=fs_value or FS_STATUS + 3, fontweight='bold', fontfamily=FONT_MONO,
        color=value_color, zorder=11,
    )
    return value_text


_kpi_cols = 4
_kpi_x0, _kpi_gap, _kpi_w, _kpi_h = 0.035, 0.015, 0.2175, 0.085
_kpi_row_y = [0.525, 0.42]

kpi_reads = _make_kpi_card(fig2, _kpi_x0 + 0 * (_kpi_w + _kpi_gap), _kpi_row_y[0], _kpi_w, _kpi_h, 'READS')
kpi_depth = _make_kpi_card(fig2, _kpi_x0 + 1 * (_kpi_w + _kpi_gap), _kpi_row_y[0], _kpi_w, _kpi_h, 'TOTAL DEPTH')
kpi_yield = _make_kpi_card(fig2, _kpi_x0 + 2 * (_kpi_w + _kpi_gap), _kpi_row_y[0], _kpi_w, _kpi_h, 'YIELD')
kpi_latest = _make_kpi_card(fig2, _kpi_x0 + 3 * (_kpi_w + _kpi_gap), _kpi_row_y[0], _kpi_w, _kpi_h,
                             'LATEST READ', fs_value=FS_STATUS, accent=ACCENT_COLOR)
kpi_n50 = _make_kpi_card(fig2, _kpi_x0 + 0 * (_kpi_w + _kpi_gap), _kpi_row_y[1], _kpi_w, _kpi_h, 'N50')
kpi_cv = _make_kpi_card(fig2, _kpi_x0 + 1 * (_kpi_w + _kpi_gap), _kpi_row_y[1], _kpi_w, _kpi_h, 'DEPTH CV')
kpi_pass = _make_kpi_card(fig2, _kpi_x0 + 2 * (_kpi_w + _kpi_gap), _kpi_row_y[1], _kpi_w, _kpi_h,
                           f'PASS RATE (\u2265Q{PASS_Q_THRESHOLD:.0f})', accent=ACCENT_COLOR)
kpi_conf = _make_kpi_card(fig2, _kpi_x0 + 3 * (_kpi_w + _kpi_gap), _kpi_row_y[1], _kpi_w, _kpi_h,
                           'AMBIGUOUS?', fs_value=FS_STATUS, accent='#FF3B30')

fig2.subplots_adjust(left=0.10, right=0.90, top=0.365, bottom=0.06)

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

    # --- LIVEインジケーターの点滅 ---
    _live_alpha = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(frame * 0.35))
    main_live_dot.set_alpha(_live_alpha)
    fig2_live_dot.set_alpha(_live_alpha)

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
                ha='center', va='center', fontsize=FS_TRACK, fontweight='bold',
                fontfamily=FONT_MONO, color=txt_color, zorder=2,
            )
            dynamic_artists.append(label)

            # --- 本物のベースコーラー風: QV数字 + 信頼度カラーバー ---
            _seg_conf = seg_confidence[si]
            if not np.isnan(_seg_conf):
                _seg_q = _prob_to_qscore(_seg_conf)
                _qcolor = _quality_color(_seg_conf)
                qv_bar = mpatches.Rectangle(
                    (seg_l, 1.05), width, 0.16, facecolor=_qcolor, edgecolor='none',
                    alpha=0.9, zorder=2,
                )
                ax_track.add_patch(qv_bar)
                dynamic_artists.append(qv_bar)
                if width >= window_width * (min_label_frac * 1.5):
                    qv_text = ax_track.text(
                        seg_l + width / 2, 1.35, f"Q{_seg_q:.0f}",
                        ha='center', va='center', fontsize=FS_TICK - 1, fontweight='bold',
                        fontfamily=FONT_MONO, color=_qcolor, zorder=2,
                    )
                    dynamic_artists.append(qv_text)

    # 未読ゾーンでは信頼度カラーバー(y=1.0付近)も見えないよう、後段のunread_spanが
    # y方向をカバーしないため、明示的に隠す必要はない（unread_spanはaxvspanで全域を覆う）

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
            ln1 = ax_wave.axvline(start_t, color='#999999', linestyle='--', linewidth=1.4, alpha=0.55, zorder=3)
            ln2 = ax_track.axvline(start_t, color='#999999', linestyle='--', linewidth=1.4, alpha=0.55, zorder=3)
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
        fontsize = 17 - 5 * age_frac
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

    if cur_code in confusable_codes:
        main_warn_circle.set_visible(True)
        main_warn_text.set_visible(True)
        main_warn_circle.set_alpha(0.6 + 0.4 * np.sin(frame * pulse_speed * 2))
    else:
        main_warn_circle.set_visible(False)
        main_warn_text.set_visible(False)

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
    for _pi in range(_ref_len):
        _peak = float(consensus_accuracy[_pi]) if covered[_pi] else 0.0
        _wy = _peak * _gauss_kernel
        acc_wave_lines[_pi].set_ydata(_wy)
        acc_wave_glow[_pi].set_ydata(_wy)
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
        latest_read_str = f"{_reveal_frag['text']} {arrow} ref[{a_s}:{a_e}]"
    else:
        latest_read_str = '-'

    # --- 精度のヒーロー表示: ゲージ ---
    if mean_accuracy >= 0.99:
        _needle_color = ACCENT_COLOR
    elif mean_accuracy >= 0.9:
        _needle_color = '#B6FF3B'
    else:
        _needle_color = '#FFB020'
    _update_gauge(gauge_needle, gauge_center_text, mean_accuracy,
                  label_text=f"{mean_accuracy * 100:.1f}%", needle_color=_needle_color)
    gauge_center_text.set_color(_needle_color)
    hero_qsub.set_text(f"Phred Q {mean_q:.1f}")

    # --- トレンド矢印（TREND_WINDOW前との差分） ---
    _acc_history.append((playhead_t, mean_accuracy))
    while len(_acc_history) > 2 and _acc_history[0][0] < playhead_t - TREND_WINDOW - 1.0:
        _acc_history.pop(0)
    _baseline_acc = None
    for _t_hist, _a_hist in _acc_history:
        if _t_hist <= playhead_t - TREND_WINDOW:
            _baseline_acc = _a_hist
        else:
            break
    if _baseline_acc is not None:
        _delta_pp = (mean_accuracy - _baseline_acc) * 100
        if abs(_delta_pp) < 0.05:
            trend_text.set_text(f"\u2192 \u00b10.0pp ({TREND_WINDOW:.1f}s)")
            trend_text.set_color('#888888')
        elif _delta_pp > 0:
            trend_text.set_text(f"\u25b2 +{_delta_pp:.1f}pp ({TREND_WINDOW:.1f}s)")
            trend_text.set_color(ACCENT_COLOR)
        else:
            trend_text.set_text(f"\u25bc {_delta_pp:.1f}pp ({TREND_WINDOW:.1f}s)")
            trend_text.set_color('#FF3B30')
    else:
        trend_text.set_text('...')
        trend_text.set_color('#666666')

    # --- 警告バッジ（区別困難ペアを読み取り中のときだけ点滅表示） ---
    if cur_code in confusable_codes:
        warn_badge_circle.set_visible(True)
        warn_badge_text.set_visible(True)
        _badge_pulse = 0.6 + 0.4 * np.sin(frame * pulse_speed * 2)
        warn_badge_circle.set_alpha(_badge_pulse)
    else:
        warn_badge_circle.set_visible(False)
        warn_badge_text.set_visible(False)

    # --- 副指標カード（KPIタイル）の更新 ---
    kpi_reads.set_text(f"{_next_frag_idx}/{n_frag}")
    kpi_depth.set_text(f"{int(depth_counts.sum())}")
    kpi_yield.set_text(f"{_cumulative_yield} bp")
    kpi_latest.set_text(latest_read_str)
    kpi_n50.set_text(f"{n50}")
    kpi_cv.set_text(f"{depth_cv:.2f}")
    kpi_pass.set_text(f"{pass_rate:.1f}%")
    if cur_code in confusable_codes:
        kpi_conf.set_text(f"{cur_code}\u2194{confusable_codes[cur_code]}")
        kpi_conf.set_color('#FF3B30')
    else:
        kpi_conf.set_text('clear')
        kpi_conf.set_color(ACCENT_COLOR)

    fig2.canvas.draw_idle()

    return (
        [line_raw, line_assigned, particle_scatter, title]
        + playhead_wave_glow + [playhead_wave_core]
        + playhead_track_glow + [playhead_track_core]
        + [marker_glow, marker_ring, marker_core, scan_caret]
        + dynamic_artists + trail_artists
        + prob_bars + [prob_glow] + prob_name_labels + prob_pct_labels
        + [yield_line, yield_head] + acc_wave_lines + acc_wave_glow
    )


n_frames = n_total // step + 1

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=interval_ms, blit=False, repeat=False,
)

fig.subplots_adjust(left=0.06, right=0.98, top=0.60, bottom=0.07)


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


_position_figure_window(fig, 20, 20)       # メインウインドウ: 左上（大画面/プロジェクタ想定でやや大きめ）
_position_figure_window(fig2, 1500, 40)    # アセンブリウインドウ: その右側（デュアルモニタ推奨）

if save_as_gif:
    out_path = save_dir / 'streaming_signal_v2.gif'
    ani.save(out_path, writer='pillow', fps=gif_fps)
    print(f"アニメーションを保存しました: {out_path}")
else:
    plt.show()
