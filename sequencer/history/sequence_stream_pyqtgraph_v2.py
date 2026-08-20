# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyqtgraph>=0.13",
#     "PyQt5>=5.15",
#     "numpy>=2.0",
#     "pandas>=2.0",
# ]
# ///

# -*- coding: utf-8 -*-
"""
sequence_stream_pyqtgraph.py
=============================
visualize_signals_stream_v10.py (matplotlib版) の完全移行版。
matplotlibへの依存を完全に排除し、PyQtGraph + Qt(PyQt5)のネイティブ描画に
置き換えている。データ層(manifest/BC CSV読み込み、セグメント化、フラグメント
のアラインメント、コンセンサス計算)は元スクリプトとロジック・変数名を
そのまま踏襲しており、同じディレクトリ構成(seq_data/manifest.csv, BC/*.csv)
であればそのまま動作する。

対応関係(matplotlib -> PyQtGraph/Qt):
  fig, ax = plt.subplots()        -> pg.GraphicsLayoutWidget().addPlot()
  ax.plot / line.set_data          -> PlotDataItem.setData()
  ax.axvline (グロー重ね書き)       -> InfiniteLineを複数重ねる(_glow_vline)
  ax.axvspan                       -> QGraphicsRectItemをViewBoxに直接追加(_rect_item)
  ax.text                          -> pg.TextItem
  animation.FuncAnimation           -> QTimer + update_frame()
  fig.text (HUDパネル)              -> QLabel (スタイルシートで縁取り・色を制御)
  半円ゲージ(Wedge)                 -> GaugeWidget(QWidget.paintEvent自前実装)
  KPIカード(FancyBboxPatch)         -> QFrame + QLabelタイル
  plt.show()                       -> app.exec()

※ 展示会向けの純粋に装飾的な要素(走査線テクスチャ、コーナーブラケット、
   電子ノイズ風パーティクル)は今回のコアな移植対象からは外し、機能的な
   要素(発光波形、プレイヘッド、確信度ランキング、アセンブリ、ゲージ、KPI)
   を優先して完全移植している。QSS(Qtスタイルシート)で背景に薄いグリッド
   線を敷く程度の代替演出は入れてある。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ============================
# 見た目パラメータ
# ============================
BG_COLOR = "#0A0A0A"
PANEL_BG = "#0A0A0A"
ACCENT_COLOR = "#39FF14"      # 再生ヘッドなどのネオン色
ACCENT_COLOR2 = "#00E5FF"     # サブアクセント(シアン系)
GRID_COLOR = "#2A2A2A"
WARN_COLOR = "#FF3B30"
FONT_MONO = "DejaVu Sans Mono"

FS_TITLE = 20
FS_HUD = 22
FS_SUB = 12
FS_LABEL = 11
FS_TICK = 10
FS_TRACK = 12
FS_STATUS = 11

pg.setConfigOptions(antialias=True, background=BG_COLOR, foreground="#CCCCCC")


# ============================
# PyQt5 / PyQt6 互換レイヤー
# PyQt6ではQtCore.Qt.AlignTopのようなフラットな列挙値が、
# QtCore.Qt.AlignmentFlag.AlignTop のようにスコープ付きEnumへ変更されている。
# インストールされているバインディングによってどちらか変わるため、
# 実行環境を問わず動くように吸収しておく。
# ============================
class _QtCompat:
    _ENUM_GROUPS = (
        "AlignmentFlag", "PenStyle", "BrushStyle", "PenCapStyle",
        "PenJoinStyle", "GlobalColor", "ItemDataRole", "Orientation",
        "MouseButton", "KeyboardModifier",
    )

    def __getattr__(self, name):
        if hasattr(QtCore.Qt, name):
            return getattr(QtCore.Qt, name)
        for group in self._ENUM_GROUPS:
            sub = getattr(QtCore.Qt, group, None)
            if sub is not None and hasattr(sub, name):
                return getattr(sub, name)
        raise AttributeError(f"QtCore.Qt(互換レイヤー)に '{name}' が見つかりません")


_Qt = _QtCompat()

try:
    _FONT_BOLD = QtGui.QFont.Weight.Bold
except AttributeError:
    _FONT_BOLD = QtGui.QFont.Bold

try:
    _RENDER_HINT_AA = QtGui.QPainter.RenderHint.Antialiasing
except AttributeError:
    _RENDER_HINT_AA = QtGui.QPainter.Antialiasing


# ============================
# 汎用ユーティリティ(matplotlib非依存。元スクリプトから移植)
# ============================
def contrasting_text_color(hex_color):
    """背景色の明るさに応じて、読みやすい文字色(黒or白)を返す"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#FFFFFF"


def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _prob_to_qscore(p, q_cap=60.0):
    p = np.clip(p, 0.0, 1.0 - 10 ** (-q_cap / 10.0))
    error = np.clip(1.0 - p, 10 ** (-q_cap / 10.0), 1.0)
    return -10.0 * np.log10(error)


PASS_Q_THRESHOLD = 9.0
PASS_CONF_THRESHOLD = 1.0 - 10 ** (-PASS_Q_THRESHOLD / 10.0)

# --- クオリティ値(0-1) -> 赤/黄/緑グラデーション。
#     matplotlibのLinearSegmentedColormapの代わりに手動で線形補間する。
_QUALITY_STOPS = [(0.0, (255, 59, 48)), (0.5, (255, 212, 0)), (1.0, (57, 255, 20))]


def _quality_rgb(accuracy_frac):
    a = float(np.clip(accuracy_frac, 0.0, 1.0))
    for i in range(len(_QUALITY_STOPS) - 1):
        p0, c0 = _QUALITY_STOPS[i]
        p1, c1 = _QUALITY_STOPS[i + 1]
        if a <= p1:
            t = 0.0 if p1 == p0 else (a - p0) / (p1 - p0)
            r = c0[0] + (c1[0] - c0[0]) * t
            g = c0[1] + (c1[1] - c0[1]) * t
            b = c0[2] + (c1[2] - c0[2]) * t
            return r, g, b
    return _QUALITY_STOPS[-1][1]


def _quality_hex(accuracy_frac):
    r, g, b = _quality_rgb(accuracy_frac)
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


# ============================
# Qtグラフィックスの薄いヘルパー
# 「太さ違いの半透明レイヤーを重ねて発光に見せる」という元スクリプトの
# 発想をそのままPyQtGraphに移植する。
# ============================
def _rect_item(viewbox, brush=None, pen=None, z=0):
    """データ座標系の矩形をViewBoxに直接追加する(ax.axvspan / Rectangle相当)"""
    item = QtWidgets.QGraphicsRectItem()
    item.setBrush(QtGui.QBrush(QtGui.QColor(brush)) if brush else QtGui.QBrush(_Qt.NoBrush))
    item.setPen(pen if pen is not None else QtGui.QPen(_Qt.NoPen))
    item.setZValue(z)
    viewbox.addItem(item, ignoreBounds=True)
    return item


def _set_rect(item, x, y, w, h):
    item.setRect(QtCore.QRectF(x, y, w, h))


class _ItemPool:
    """毎フレームのGraphicsItem生成/破棄コストを避けるための使い回しプール。

    実データのように1フレームで表示するセグメント数が数百に達すると、
    「毎フレーム作って毎フレーム消す」方式(元のmatplotlib版と同じ発想)は
    QtではUIスレッドの処理落ち(=ウインドウが「応答なし」になる)を招く。
    そこで、一度作ったアイテムは隠すだけにして使い回す。

    使い方:
        pool.begin()
        for seg in visible_segments:
            item = pool.acquire()
            item.setRect(...)/setText(...) など更新
        pool.end()   # このフレームで使わなかった余りは自動で非表示にする
    """

    def __init__(self, factory):
        self._factory = factory
        self._pool = []
        self._used = 0

    def begin(self):
        self._used = 0

    def acquire(self):
        if self._used < len(self._pool):
            item = self._pool[self._used]
            item.setVisible(True)
        else:
            item = self._factory()
            self._pool.append(item)
        self._used += 1
        return item

    def end(self):
        for i in range(self._used, len(self._pool)):
            self._pool[i].setVisible(False)


def _glow_vline(plot_item, color, base_width=2.4,
                 layers=((14, 40), (8, 80), (3, 160))):
    """ネオン管風の発光縦線(プレイヘッド用)。InfiniteLineを重ねて作る。
    戻り値: (グローInfiniteLineのリスト, 本体InfiniteLine)"""
    glow_lines = []
    for width, alpha in layers:
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        line = pg.InfiniteLine(angle=90, pen=pg.mkPen(c, width=width))
        line.setZValue(9)
        plot_item.addItem(line)
        glow_lines.append(line)
    core = pg.InfiniteLine(angle=90, pen=pg.mkPen(color, width=base_width))
    core.setZValue(10)
    plot_item.addItem(core)
    return glow_lines, core


def _glow_curve(plot_item, color, base_width=2.6, base_alpha=255,
                 layers=((9, 15), (5, 30), (2.5, 55))):
    """発光ライン(波形本体用)。戻り値: (グローPlotDataItemのリスト, 本体PlotDataItem)"""
    glow_curves = []
    for width, alpha in layers:
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        curve = pg.PlotDataItem(pen=pg.mkPen(c, width=width))
        curve.setZValue(9)
        plot_item.addItem(curve)
        glow_curves.append(curve)
    c0 = QtGui.QColor(color)
    c0.setAlpha(base_alpha)
    core = pg.PlotDataItem(pen=pg.mkPen(c0, width=base_width))
    core.setZValue(10)
    plot_item.addItem(core)
    return glow_curves, core


def _set_glow_curve_data(glow_curves, core, x, y):
    for gc in glow_curves:
        gc.setData(x, y)
    core.setData(x, y)


# ============================
# スクリプト自身の場所を基準にする
# ============================
BASE_DIR = Path(__file__).resolve().parent
save_dir = BASE_DIR / "seq_data"
manifest_path = save_dir / "manifest.csv"

if not manifest_path.exists():
    raise FileNotFoundError(
        f"マニフェストが見つかりません: {manifest_path}\n"
        "先に batch_generate.py を実行してください(Code列対応版)。"
    )

manifest_df = pd.read_csv(manifest_path)
selectBC = manifest_df.iloc[0]["selectBC"]
print(f"{len(manifest_df)} 件のデータを連結してアニメーション表示します。 (BC = {selectBC})")


# ============================
# BC CSVからコードの色・名前情報を読み込む(元スクリプトから流用)
# ============================
def load_code_info(selectBC):
    file_path = BASE_DIR / "BC" / f"{selectBC}.csv"
    df = pd.read_csv(
        file_path,
        names=["Index", "Code", "Name", "Colar", "R_conductance",
               "species", "Description", "Extra1", "Extra2", "Extra3"],
        skiprows=1,
    )
    info = {}
    for _, row in df.iterrows():
        hex_color = "#{:06X}".format(int(row["Colar"]))
        info[row["Code"]] = {
            "name": row["Name"] if isinstance(row["Name"], str) else str(row["Code"]),
            "color": hex_color,
            "description": row["Description"] if isinstance(row["Description"], str) else "",
            "value": float(row["R_conductance"]) if pd.notna(row["R_conductance"]) else None,
        }
    info.setdefault("B", {"name": "Baseline", "color": "#969696", "description": "baseline", "value": 0.0})
    return info


code_info = load_code_info(selectBC)

_prob_codes = [c for c, v in code_info.items() if v.get("value") is not None]
prob_codes = np.array(_prob_codes)
prob_values = np.array([code_info[c]["value"] for c in _prob_codes])
prob_colors = [code_info[c]["color"] for c in _prob_codes]

noise_amplitude = float(manifest_df.iloc[0]["noise_amplitude"])
reference_sequence = str(manifest_df.iloc[0]["sequence"])

# ============================
# 全runのデータを時間軸で連結(元スクリプトから流用)
# ============================
all_time, all_raw, all_assigned, all_codes = [], [], [], []
run_boundaries = []
time_offset = 0.0
gap = 0.01

for _, row in manifest_df.iterrows():
    df = pd.read_csv(save_dir / row["file_name"])
    if "Code" not in df.columns:
        raise ValueError(
            f"{row['file_name']} に 'Code' 列がありません。"
            "batch_generate.py を最新版(Code列対応)で再実行してください。"
        )
    t = df["Time"].values + time_offset
    run_boundaries.append((row["run_idx"], t[0]))
    all_time.append(t)
    all_raw.append(df["Raw Value"].values)
    all_assigned.append(df["Assigned"].values)
    all_codes.append(df["Code"].astype(str).values)
    time_offset = t[-1] + gap

all_time = np.concatenate(all_time)
all_raw = np.concatenate(all_raw)
all_assigned = np.concatenate(all_assigned)
all_codes = np.concatenate(all_codes)
n_total = len(all_time)
dt = float(np.median(np.diff(all_time))) if n_total > 1 else 0.001

# ============================
# 連続する同一コード区間をセグメント化(元スクリプトから流用)
# ============================
change_idx = np.where(all_codes[1:] != all_codes[:-1])[0] + 1
seg_start_idx = np.concatenate(([0], change_idx))
seg_codes = all_codes[seg_start_idx]
seg_start_t = all_time[seg_start_idx]
seg_end_t = np.empty_like(seg_start_t)
seg_end_t[:-1] = seg_start_t[1:]
seg_end_t[-1] = all_time[-1] + dt
n_seg = len(seg_start_t)

seg_end_idx = np.empty_like(seg_start_idx)
seg_end_idx[:-1] = seg_start_idx[1:] - 1
seg_end_idx[-1] = n_total - 1

_code_to_prob_idx = {c: i for i, c in enumerate(prob_codes)}
seg_confidence = np.full(n_seg, np.nan)
_seg_rival_code = np.full(n_seg, "", dtype=object)
_seg_raw_conf = np.full(n_seg, np.nan)
_CONF_FLOOR = 0.5

for _si in range(n_seg):
    _code = seg_codes[_si]
    if _code not in _code_to_prob_idx:
        continue
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
        _ll_best_rival = -np.inf
    _raw_conf = _sigmoid(_ll_true - _ll_best_rival)
    _seg_raw_conf[_si] = _raw_conf
    seg_confidence[_si] = max(_raw_conf, _CONF_FLOOR)

CONFUSION_RATIO_THRESHOLD = 3.0
confusable_codes = {}
for _ci, _c in enumerate(prob_codes):
    if _c not in set(reference_sequence):
        continue
    _others = np.delete(np.arange(len(prob_codes)), _ci)
    _dists = np.abs(prob_values[_others] - prob_values[_ci])
    _nearest_i = _others[np.argmin(_dists)]
    _nearest_gap = float(_dists.min())
    if _nearest_gap / noise_amplitude < CONFUSION_RATIO_THRESHOLD:
        confusable_codes[_c] = str(prob_codes[_nearest_i])


def visible_segment_range(t_start, t_now):
    lo = np.searchsorted(seg_end_t, t_start, side="right")
    hi = np.searchsorted(seg_start_t, t_now, side="right")
    return lo, max(hi, lo)


def current_segment_index(t):
    idx = np.searchsorted(seg_start_t, t, side="right") - 1
    return int(np.clip(idx, 0, n_seg - 1))


# ============================
# 読み取り断片(フラグメント)の検出 & 元配列へのアラインメント(元スクリプトから流用)
# ============================
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
    if seg_codes[_i] == "B":
        _i += 1
        continue
    _j = _i
    frag_chars, frag_confs = [], []
    while _j < n_seg and seg_codes[_j] != "B":
        frag_chars.append(seg_codes[_j])
        frag_confs.append(seg_confidence[_j])
        _j += 1
    frag_str = "".join(frag_chars)

    fwd_positions = _find_all_occurrences(_ref_fwd, frag_str)
    rev_positions = _find_all_occurrences(_ref_rev, frag_str)
    candidates = [("fwd", p) for p in fwd_positions] + [("rev", p) for p in rev_positions]

    if candidates:
        orientation, pos = candidates[_frag_rng.integers(0, len(candidates))]
        if orientation == "fwd":
            align_start, align_end = pos, pos + len(frag_str)
        else:
            align_start = _ref_len - pos - len(frag_str)
            align_end = _ref_len - pos
    else:
        orientation = None
        align_start = align_end = None

    if align_start is not None:
        if orientation == "fwd":
            pos_conf = list(zip(range(align_start, align_end), frag_confs))
        else:
            pos_conf = list(zip(range(align_end - 1, align_start - 1, -1), frag_confs))
    else:
        pos_conf = []

    fragments.append({
        "text": frag_str,
        "start_t": seg_start_t[_i],
        "end_t": seg_end_t[_j - 1],
        "align_start": align_start,
        "align_end": align_end,
        "orientation": orientation,
        "pos_conf": pos_conf,
        "length": len(frag_str),
        "mean_conf": float(np.mean(frag_confs)) if len(frag_confs) > 0 else np.nan,
    })
    _i = _j

fragments.sort(key=lambda f: f["end_t"])
frag_end_times = np.array([f["end_t"] for f in fragments])
n_frag = len(fragments)


# ============================
# アニメーションパラメータ
# ============================
window_width = 0.3
step = 5
interval_ms = 20
playhead_frac = 0.8
min_label_frac = 0.02
pulse_speed = 0.25
n_prob_rows = 7
trail_window = 8 * window_width
TREND_WINDOW = 1.5
_frag_reveal_decay = 1.2
n_frames = n_total // step + 1


# ============================
# GaugeWidget: matplotlib版の半円メーター(_draw_gauge/_update_gauge)のQt移植
# ============================
class GaugeWidget(QtWidgets.QWidget):
    """0-1の値を色分けされた半円メーター(速度計スタイル)で描画するQtウィジェット"""

    ZONES = ((0.0, 0.9, "#FF3B30"), (0.9, 0.99, "#FFD400"), (0.99, 1.0, ACCENT_COLOR))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._value = 0.0
        self._label = "0%"
        self._needle_color = QtGui.QColor("white")

    def set_value(self, value_frac, label_text=None, needle_color="white"):
        self._value = float(np.clip(value_frac, 0.0, 1.0))
        self._label = label_text if label_text is not None else f"{self._value * 100:.0f}%"
        self._needle_color = QtGui.QColor(needle_color)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(_RENDER_HINT_AA)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h * 0.90
        radius = min(w * 0.46, h * 0.80)
        band_w = radius * 0.22

        for lo, hi, color in self.ZONES:
            start_angle = 180 * (1 - lo)
            span_angle = -180 * (hi - lo)
            rect = QtCore.QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            pen = QtGui.QPen(QtGui.QColor(color), band_w)
            pen.setCapStyle(_Qt.FlatCap)
            p.setPen(pen)
            p.drawArc(rect, int(start_angle * 16), int(span_angle * 16))

        angle = np.pi * (1 - self._value)
        nx = cx + (radius * 0.72) * np.cos(angle)
        ny = cy - (radius * 0.72) * np.sin(angle)
        pen = QtGui.QPen(self._needle_color, 4)
        pen.setCapStyle(_Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(nx, ny))
        p.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        p.setPen(QtGui.QPen(QtGui.QColor(ACCENT_COLOR), 2))
        p.drawEllipse(QtCore.QPointF(cx, cy), 6, 6)

        p.setPen(QtGui.QPen(self._needle_color))
        font = QtGui.QFont(FONT_MONO, 18, _FONT_BOLD)
        p.setFont(font)
        p.drawText(QtCore.QRectF(0, cy - radius * 0.55, w, radius * 0.5),
                   _Qt.AlignCenter, self._label)

        p.setPen(QtGui.QPen(QtGui.QColor("#888888")))
        font2 = QtGui.QFont(FONT_MONO, 9)
        p.setFont(font2)
        p.drawText(QtCore.QRectF(cx - radius - 22, cy - 14, 44, 20), _Qt.AlignCenter, "0%")
        p.drawText(QtCore.QRectF(cx + radius - 22, cy - 14, 44, 20), _Qt.AlignCenter, "100%")


def make_kpi_card(parent_layout, label, accent=ACCENT_COLOR2, value_fs=18):
    """KPIカード(Geckoboard風タイル)。matplotlib版の _make_kpi_card のQt移植。
    戻り値: 値を表示するQLabel(毎フレーム .setText() で更新する)"""
    frame = QtWidgets.QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background-color:#111111; border:1px solid {GRID_COLOR}; border-radius:6px; }}"
    )
    v = QtWidgets.QVBoxLayout(frame)
    v.setContentsMargins(8, 6, 8, 8)
    v.setSpacing(4)

    top_bar = QtWidgets.QFrame()
    top_bar.setFixedHeight(3)
    top_bar.setStyleSheet(f"background-color:{accent}; border-radius:1px; border:none;")
    v.addWidget(top_bar)

    lbl = QtWidgets.QLabel(label)
    lbl.setStyleSheet("color:#888888; font-size:10px; font-family:'DejaVu Sans Mono',monospace; font-weight:bold; border:none;")
    lbl.setAlignment(_Qt.AlignCenter)
    v.addWidget(lbl)

    val = QtWidgets.QLabel("--")
    val.setStyleSheet(
        f"color:{accent}; font-size:{value_fs}px; font-weight:bold; font-family:'DejaVu Sans Mono',monospace; border:none;"
    )
    val.setAlignment(_Qt.AlignCenter)
    val.setWordWrap(True)
    v.addWidget(val)

    parent_layout.addWidget(frame)
    return val


# ============================
# ダッシュボード配色パレット(参考画像のネイビー系デザインに合わせる)
# ============================
PAGE_BG = "#0B1220"
SIDEBAR_BG = "#0D1526"
CARD_BG = "#101A2E"
CARD_BORDER = "#1E2A44"
TEXT_PRIMARY = "#E7ECF5"
TEXT_MUTED = "#8592AC"
ACCENT_BLUE = "#38BDF8"
GREEN_OK = "#4ADE80"


class HBar(QtWidgets.QWidget):
    """角丸の水平プログレスバー(参考画像の「Confidence」バー相当)。
    matplotlib版の半円ゲージ(_draw_gauge)より参考画像のUIに近いシンプルな
    バー表現に置き換えたもの。"""

    def __init__(self, height=8, track_color=CARD_BORDER, fill_color=ACCENT_BLUE, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._track_color = QtGui.QColor(track_color)
        self._fill_color = QtGui.QColor(fill_color)
        self.setFixedHeight(height)
        self.setMinimumWidth(40)

    def set_value(self, frac, color=None):
        self._value = float(np.clip(frac, 0.0, 1.0))
        if color is not None:
            self._fill_color = QtGui.QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(_RENDER_HINT_AA)
        w, h = self.width(), self.height()
        r = h / 2.0
        p.setPen(QtGui.QPen(_Qt.NoPen))
        p.setBrush(QtGui.QBrush(self._track_color))
        p.drawRoundedRect(QtCore.QRectF(0, 0, w, h), r, r)
        if self._value > 0:
            fw = max(h, w * self._value)
            p.setBrush(QtGui.QBrush(self._fill_color))
            p.drawRoundedRect(QtCore.QRectF(0, 0, fw, h), r, r)
        p.end()


def make_card(title, parent_layout=None, stretch=0):
    """角丸のネイビーカードを1つ作る。戻り値: (QFrame, その中身用QVBoxLayout)"""
    frame = QtWidgets.QFrame()
    frame.setObjectName("card")
    frame.setStyleSheet(
        f"QFrame#card {{ background-color:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px; }}"
    )
    v = QtWidgets.QVBoxLayout(frame)
    v.setContentsMargins(18, 16, 18, 16)
    v.setSpacing(10)
    if title:
        head = QtWidgets.QLabel(title)
        head.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:14px; font-weight:600; font-family:'DejaVu Sans Mono',monospace; border:none;"
        )
        v.addWidget(head)
    if parent_layout is not None:
        parent_layout.addWidget(frame, stretch=stretch)
    return frame, v


# ============================
# ダッシュボードウインドウ(参考画像のレイアウトに合わせた単一ウインドウ版)
# ============================
class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOLECULE CALLER")
        self.resize(1700, 960)
        self.setStyleSheet(f"background-color:{PAGE_BG};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        content = QtWidgets.QWidget()
        content_v = QtWidgets.QVBoxLayout(content)
        content_v.setContentsMargins(24, 18, 24, 18)
        content_v.setSpacing(16)
        root.addWidget(content, stretch=1)

        content_v.addLayout(self._build_topbar())

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(16)
        content_v.addLayout(row1, stretch=5)
        row1.addWidget(self._build_signal_card(), stretch=7)
        row1.addWidget(self._build_current_call_card(), stretch=3)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(16)
        content_v.addLayout(row2, stretch=4)
        row2.addWidget(self._build_assembly_card(), stretch=7)
        row2.addWidget(self._build_kpi_grid(), stretch=3)

        content_v.addLayout(self._build_status_bar())

        # --- 状態(フレームをまたいで保持) ---
        self.frame = 0
        self._paused = False
        self.next_frag_idx = 0
        self.cumulative_yield = 0
        self.depth_logodds = np.zeros(_ref_len)
        self.timer = None

    # ----------------------------------------------------------------
    # レイアウト構築
    # ----------------------------------------------------------------
    def _build_sidebar(self):
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(120)
        sidebar.setStyleSheet(f"background-color:{SIDEBAR_BG}; border:none;")
        v = QtWidgets.QVBoxLayout(sidebar)
        v.setContentsMargins(0, 24, 0, 20)
        v.setSpacing(2)

        items = [("\u2764", "Live Run", True), ("\u2261", "Sequence", False),
                 ("\u25A4", "Analysis", False), ("\u2913", "Export", False),
                 ("\u2699", "Settings", False)]
        for icon, label, active in items:
            btn = QtWidgets.QFrame()
            btn.setFixedHeight(64)
            bg = "#16233D" if active else "transparent"
            border_left = f"3px solid {ACCENT_BLUE}" if active else "3px solid transparent"
            btn.setStyleSheet(f"QFrame {{ background-color:{bg}; border:none; border-left:{border_left}; }}")
            bl = QtWidgets.QVBoxLayout(btn)
            bl.setContentsMargins(0, 8, 0, 0)
            bl.setSpacing(2)
            color = ACCENT_BLUE if active else TEXT_MUTED
            icon_lbl = QtWidgets.QLabel(icon)
            icon_lbl.setAlignment(_Qt.AlignCenter)
            icon_lbl.setStyleSheet(f"color:{color}; font-size:18px; border:none;")
            text_lbl = QtWidgets.QLabel(label)
            text_lbl.setAlignment(_Qt.AlignCenter)
            text_lbl.setStyleSheet(f"color:{color}; font-size:10px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            bl.addWidget(icon_lbl)
            bl.addWidget(text_lbl)
            v.addWidget(btn)
        v.addStretch(1)
        return sidebar

    def _build_topbar(self):
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("MOLECULE CALLER")
        title.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:20px; font-weight:700; font-family:'DejaVu Sans Mono',monospace;")
        row.addWidget(title)

        self.run_selector = QtWidgets.QComboBox()
        self.run_selector.addItems([f"Run {int(r):03d}" for r in manifest_df["run_idx"]])
        self.run_selector.setStyleSheet(
            f"QComboBox {{ background-color:{CARD_BG}; color:{TEXT_PRIMARY}; border:1px solid {CARD_BORDER}; "
            "border-radius:6px; padding:4px 10px; font-family:'DejaVu Sans Mono',monospace; }}"
        )
        self.run_selector.setFixedWidth(130)
        row.addSpacing(20)
        row.addWidget(self.run_selector)

        self.live_dot = QtWidgets.QLabel("\u25CF")
        self.live_dot.setStyleSheet(f"color:{ACCENT_COLOR}; font-size:12px;")
        self.live_text = QtWidgets.QLabel("LIVE")
        self.live_text.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:12px; font-weight:600; font-family:'DejaVu Sans Mono',monospace;")
        row.addSpacing(14)
        row.addWidget(self.live_dot)
        row.addWidget(self.live_text)

        row.addStretch(1)

        elapsed_lbl = QtWidgets.QLabel("Elapsed")
        elapsed_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace;")
        self.elapsed_value = QtWidgets.QLabel("00:00:00")
        self.elapsed_value.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:13px; font-weight:600; font-family:'DejaVu Sans Mono',monospace;")
        row.addWidget(elapsed_lbl)
        row.addSpacing(6)
        row.addWidget(self.elapsed_value)
        row.addSpacing(20)

        self.pause_btn = QtWidgets.QPushButton("\u23F8  Pause")
        self.pause_btn.setStyleSheet(
            f"QPushButton {{ background-color:{CARD_BG}; color:{TEXT_PRIMARY}; border:1px solid {CARD_BORDER}; "
            "border-radius:6px; padding:8px 16px; font-family:'DejaVu Sans Mono',monospace; font-weight:600; }}\n"
            "QPushButton:hover { background-color:#182842; }"
        )
        self.pause_btn.clicked.connect(self._toggle_pause)
        row.addWidget(self.pause_btn)

        self.stop_btn = QtWidgets.QPushButton("\u25A0  Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color:#3A1620; color:#FF6B6B; border:1px solid #5A2230; "
            "border-radius:6px; padding:8px 16px; font-family:'DejaVu Sans Mono',monospace; font-weight:600; }\n"
            "QPushButton:hover { background-color:#4A1A28; }"
        )
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.stop_btn)
        return row

    def _build_signal_card(self):
        frame, v = make_card("Real-time signal")

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(CARD_BG)
        v.addWidget(self.glw, stretch=5)

        self.plot_wave = self.glw.addPlot(row=0, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 5)
        self.plot_wave.showGrid(x=True, y=True, alpha=0.12)
        self.plot_wave.setLabel("left", "Current (nA)")
        self.plot_wave.addLegend(offset=(-10, 10))
        for ax_name in ("left", "bottom"):
            ax = self.plot_wave.getAxis(ax_name)
            ax.setPen(pg.mkPen(CARD_BORDER))
            ax.setTextPen(pg.mkPen(TEXT_MUTED))

        self.glw.nextRow()
        self.plot_track = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.plot_track.setXLink(self.plot_wave)
        self.plot_track.setYRange(0, 1)
        self.plot_track.hideAxis("left")
        self.plot_track.getAxis("bottom").setStyle(showValues=False)
        self.plot_track.setMouseEnabled(x=False, y=False)
        self.plot_track.setTitle("Events", color=TEXT_MUTED, size="9pt")

        margin = 0.1 * (all_raw.max() - all_raw.min() + 1e-9)
        self.plot_wave.setYRange(all_raw.min() - margin, all_raw.max() + margin)
        self.vb_wave = self.plot_wave.getViewBox()
        self.vb_track = self.plot_track.getViewBox()

        self.line_raw = pg.PlotDataItem(pen=pg.mkPen(ACCENT_BLUE, width=1.1), name="Raw signal")
        self.line_raw.setOpacity(0.6)
        self.plot_wave.addItem(self.line_raw)
        self.assigned_glow, self.line_assigned = _glow_curve(
            self.plot_wave, "#FFFFFF", base_width=2.2, layers=((6, 25), (3, 55))
        )

        self.playhead_wave_glow, self.playhead_wave_core = _glow_vline(
            self.plot_wave, ACCENT_BLUE, layers=((10, 30), (5, 70), (2, 150))
        )
        self.playhead_track_glow, self.playhead_track_core = _glow_vline(
            self.plot_track, ACCENT_BLUE, layers=((10, 30), (5, 70), (2, 150))
        )

        self.marker_core = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(ACCENT_BLUE),
                                              pen=pg.mkPen("white", width=1.4))
        self.marker_core.setZValue(11)
        self.plot_wave.addItem(self.marker_core)

        self._wave_seg_pool = _ItemPool(lambda: _rect_item(self.vb_wave, brush="#888888", z=0))
        self._track_seg_pool = _ItemPool(lambda: _rect_item(self.vb_track, brush="#888888", z=1))
        self._unread_wave = _rect_item(self.vb_wave, brush=QtGui.QColor(0, 0, 0, 60), z=0.1)
        self._unread_track = _rect_item(self.vb_track, brush=QtGui.QColor(0, 0, 0, 90), z=0.1)
        for it in (self._unread_wave, self._unread_track):
            it.setVisible(False)

        # --- 直近デコード配列(色付きピル表示) ---
        seq_lbl = QtWidgets.QLabel("Decoded sequence (latest)")
        seq_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace;")
        v.addWidget(seq_lbl)

        pill_row = QtWidgets.QHBoxLayout()
        pill_row.setSpacing(8)
        self.pill_labels = []
        N_PILLS = 12
        for _ in range(N_PILLS):
            lbl = QtWidgets.QLabel("")
            lbl.setFixedSize(42, 42)
            lbl.setAlignment(_Qt.AlignCenter)
            lbl.setStyleSheet(
                f"background-color:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:8px;"
            )
            pill_row.addWidget(lbl)
            self.pill_labels.append(lbl)
        pill_row.addStretch(1)
        v.addLayout(pill_row)

        return frame

    def _build_current_call_card(self):
        frame, v = make_card("Current call")

        self.call_letter = QtWidgets.QLabel("\u2014")
        self.call_letter.setAlignment(_Qt.AlignCenter)
        self.call_letter.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:56px; font-weight:800; font-family:'DejaVu Sans Mono',monospace; border:none;"
        )
        v.addWidget(self.call_letter)

        self.call_name = QtWidgets.QLabel("baseline")
        self.call_name.setAlignment(_Qt.AlignCenter)
        self.call_name.setStyleSheet(f"color:{TEXT_MUTED}; font-size:14px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        v.addWidget(self.call_name)

        v.addSpacing(8)
        conf_lbl = QtWidgets.QLabel("Confidence")
        conf_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        v.addWidget(conf_lbl)
        self.call_conf_value = QtWidgets.QLabel("0.0%")
        self.call_conf_value.setStyleSheet(
            f"color:{ACCENT_BLUE}; font-size:26px; font-weight:700; font-family:'DejaVu Sans Mono',monospace; border:none;"
        )
        v.addWidget(self.call_conf_value)
        self.call_conf_bar = HBar(height=10, fill_color=ACCENT_BLUE)
        v.addWidget(self.call_conf_bar)
        bar_labels = QtWidgets.QHBoxLayout()
        l0 = QtWidgets.QLabel("0%")
        l0.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        l1 = QtWidgets.QLabel("100%")
        l1.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        bar_labels.addWidget(l0)
        bar_labels.addStretch(1)
        bar_labels.addWidget(l1)
        v.addLayout(bar_labels)

        v.addSpacing(14)
        conf_head = QtWidgets.QLabel("Confidence")
        conf_head.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:13px; font-weight:600; font-family:'DejaVu Sans Mono',monospace; border:none;"
        )
        v.addWidget(conf_head)

        self.prob_rows = []
        for i in range(n_prob_rows):
            row = QtWidgets.QHBoxLayout()
            idx_lbl = QtWidgets.QLabel(str(i + 1))
            idx_lbl.setFixedWidth(16)
            idx_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            name_lbl = QtWidgets.QLabel("")
            name_lbl.setMinimumWidth(90)
            name_lbl.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:11px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            bar = HBar(height=6, fill_color=ACCENT_BLUE)
            pct_lbl = QtWidgets.QLabel("0.0%")
            pct_lbl.setFixedWidth(48)
            pct_lbl.setAlignment(_Qt.AlignRight | _Qt.AlignVCenter)
            pct_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            row.addWidget(idx_lbl)
            row.addWidget(name_lbl)
            row.addWidget(bar, stretch=1)
            row.addWidget(pct_lbl)
            v.addLayout(row)
            self.prob_rows.append((name_lbl, bar, pct_lbl))
        v.addStretch(1)
        return frame

    def _build_assembly_card(self):
        frame, v = make_card("Sequence assembly")
        h = QtWidgets.QHBoxLayout()
        v.addLayout(h, stretch=1)

        left_glw = pg.GraphicsLayoutWidget()
        left_glw.setBackground(CARD_BG)
        h.addWidget(left_glw, stretch=1)

        self.plot_trace = left_glw.addPlot(row=0, col=0)
        left_glw.ci.layout.setRowStretchFactor(0, 1)
        self.plot_trace.setXRange(0, _ref_len)
        self.plot_trace.setYRange(0, 1.08)
        self.plot_trace.setLabel("left", "Accuracy")
        self.plot_trace.getAxis("bottom").setStyle(showValues=False)
        self.plot_trace.setTitle("Consensus trace", color=TEXT_MUTED, size="9pt")
        ax = self.plot_trace.getAxis("left")
        ax.setPen(pg.mkPen(CARD_BORDER))
        ax.setTextPen(pg.mkPen(TEXT_MUTED))

        _GAUSS_SIGMA = 0.15
        self._gauss_x_local = np.linspace(-0.5, 0.5, 61)
        self._gauss_kernel = np.exp(-0.5 * (self._gauss_x_local / _GAUSS_SIGMA) ** 2)
        self.acc_wave_lines = []
        for pi in range(_ref_len):
            code = reference_sequence[pi]
            wcolor = code_info.get(code, {"color": ACCENT_BLUE})["color"]
            wx = pi + 0.5 + self._gauss_x_local
            line = pg.PlotDataItem(wx, np.zeros_like(wx), pen=pg.mkPen(wcolor, width=2))
            self.plot_trace.addItem(line)
            self.acc_wave_lines.append(line)

        left_glw.nextRow()
        self.plot_depth = left_glw.addPlot(row=1, col=0)
        left_glw.ci.layout.setRowStretchFactor(1, 1)
        self.plot_depth.setXLink(self.plot_trace)
        self.plot_depth.setYRange(0, 5)
        self.plot_depth.setLabel("left", "Depth")
        axb = self.plot_depth.getAxis("bottom")
        axb.setTicks([[(i + 0.5, c) for i, c in enumerate(reference_sequence)]])
        for ax_name in ("left", "bottom"):
            axo = self.plot_depth.getAxis(ax_name)
            axo.setPen(pg.mkPen(CARD_BORDER))
            axo.setTextPen(pg.mkPen(TEXT_MUTED))
        depth_colors = [code_info.get(c, {"color": "#888888"})["color"] for c in reference_sequence]
        self.depth_counts = np.zeros(_ref_len)
        self.depth_bars = pg.BarGraphItem(
            x=np.arange(_ref_len) + 0.5, height=self.depth_counts, width=0.6,
            brushes=[pg.mkBrush(c) for c in depth_colors], pen=pg.mkPen(None),
        )
        self.plot_depth.addItem(self.depth_bars)

        right_glw = pg.GraphicsLayoutWidget()
        right_glw.setBackground(CARD_BG)
        h.addWidget(right_glw, stretch=1)
        self.plot_yield = right_glw.addPlot()
        self.plot_yield.setLabel("bottom", "Time (min)")
        self.plot_yield.setLabel("left", "Yield (bases)")
        self.plot_yield.setTitle("Yield (bases)", color=TEXT_MUTED, size="9pt")
        for ax_name in ("left", "bottom"):
            axo = self.plot_yield.getAxis(ax_name)
            axo.setPen(pg.mkPen(CARD_BORDER))
            axo.setTextPen(pg.mkPen(TEXT_MUTED))
        self.yield_glow, self.yield_line = _glow_curve(
            self.plot_yield, ACCENT_BLUE, base_width=1.8, layers=((6, 30), (3, 60))
        )
        self._yield_t_hist = []
        self._yield_v_hist = []

        return frame

    def _kpi_tile(self, grid, r, c, label, color):
        tile = QtWidgets.QFrame()
        tile.setStyleSheet(
            f"QFrame {{ background-color:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px; }}"
        )
        tv = QtWidgets.QVBoxLayout(tile)
        tv.setContentsMargins(16, 14, 16, 14)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        tv.addWidget(lbl)
        val = QtWidgets.QLabel("--")
        val.setStyleSheet(f"color:{color}; font-size:22px; font-weight:800; font-family:'DejaVu Sans Mono',monospace; border:none;")
        tv.addWidget(val)
        sub = QtWidgets.QLabel("")
        sub.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; font-family:'DejaVu Sans Mono',monospace; border:none;")
        tv.addWidget(sub)
        grid.addWidget(tile, r, c)
        return val, sub

    def _build_kpi_grid(self):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setSpacing(14)
        self.kpi_reads_val, self.kpi_reads_sub = self._kpi_tile(grid, 0, 0, "Reads", ACCENT_BLUE)
        self.kpi_yield_val, self.kpi_yield_sub = self._kpi_tile(grid, 0, 1, "Yield", ACCENT_BLUE)
        self.kpi_q_val, self.kpi_q_sub = self._kpi_tile(grid, 1, 0, "Mean Q", ACCENT_BLUE)
        self.kpi_pass_val, self.kpi_pass_sub = self._kpi_tile(grid, 1, 1, "Pass rate", GREEN_OK)
        return container

    def _build_status_bar(self):
        row = QtWidgets.QHBoxLayout()

        def item(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px; font-family:'DejaVu Sans Mono',monospace;")
            return lbl

        sampling_hz = (1.0 / dt) if dt > 0 else 0.0
        row.addWidget(item(f"Model: Phred Q  |  BC: {selectBC}"))
        row.addSpacing(24)
        row.addWidget(item(f"Sampling rate: {sampling_hz / 1000:.2f} kHz"))
        row.addSpacing(24)
        row.addWidget(item(f"Reference: {reference_sequence}"))
        row.addSpacing(24)
        self.status_conn_dot = QtWidgets.QLabel("\u25CF")
        self.status_conn_dot.setStyleSheet(f"color:{ACCENT_COLOR}; font-size:11px;")
        row.addWidget(self.status_conn_dot)
        row.addWidget(item("Connection: OK"))
        row.addStretch(1)
        export_btn = QtWidgets.QPushButton("\u2913  Export")
        export_btn.setStyleSheet(
            f"QPushButton {{ background-color:{CARD_BG}; color:{TEXT_PRIMARY}; border:1px solid {CARD_BORDER}; "
            "border-radius:6px; padding:6px 14px; font-family:'DejaVu Sans Mono',monospace; }}"
        )
        row.addWidget(export_btn)
        return row

    # ----------------------------------------------------------------
    # Pause / Stop
    # ----------------------------------------------------------------
    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("\u25B6  Resume" if self._paused else "\u23F8  Pause")

    def _stop(self):
        if self.timer is not None:
            self.timer.stop()
        self._paused = True
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.live_text.setText("STOPPED")
        self.live_dot.setStyleSheet(f"color:{WARN_COLOR}; font-size:12px;")

    # ----------------------------------------------------------------
    # 毎フレームの更新
    # ----------------------------------------------------------------
    def tick(self):
        if self._paused:
            return
        if self.frame >= n_frames:
            self.timer.stop()
            return
        self.update_frame()
        self.frame += 1

    def update_frame(self):
        frame = self.frame
        idx = min(frame * step, n_total - 1)
        t_now = all_time[idx]
        t_start = max(0.0, t_now - window_width)
        playhead_t = min(t_start + window_width * playhead_frac, t_now)

        # --- LIVE点滅 ---
        live_alpha = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(frame * 0.35))
        c = QtGui.QColor(ACCENT_COLOR)
        c.setAlphaF(live_alpha)
        self.live_dot.setStyleSheet(
            f"color:rgba({c.red()},{c.green()},{c.blue()},{c.alphaF():.2f}); font-size:12px;"
        )

        # --- Elapsed ---
        total_s = int(playhead_t)
        hh, rem = divmod(total_s, 3600)
        mm, ss = divmod(rem, 60)
        self.elapsed_value.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")

        # --- 波形 ---
        mask = (all_time >= t_start) & (all_time <= t_now)
        self.line_raw.setData(all_time[mask], all_raw[mask])
        mask_assigned = mask & (all_time <= playhead_t)
        _set_glow_curve_data(self.assigned_glow, self.line_assigned,
                              all_time[mask_assigned], all_assigned[mask_assigned])
        self.plot_wave.setXRange(t_start, t_start + window_width, padding=0)

        vb_wave, vb_track = self.vb_wave, self.vb_track
        y0, y1 = vb_wave.viewRange()[1]

        self._wave_seg_pool.begin()
        self._track_seg_pool.begin()
        lo, hi = visible_segment_range(t_start, t_now)
        MAX_VISIBLE_SEGMENTS = 500
        if hi - lo > MAX_VISIBLE_SEGMENTS:
            lo = hi - MAX_VISIBLE_SEGMENTS
        for si in range(lo, hi):
            code = seg_codes[si]
            info = code_info.get(code, {"color": "#888888"})
            color = info["color"]
            seg_l = max(seg_start_t[si], t_start)
            seg_r_full = min(seg_end_t[si], t_start + window_width)
            if seg_l >= playhead_t:
                continue
            seg_r = min(seg_r_full, playhead_t)
            width = seg_r - seg_l
            if width <= 0:
                continue

            c_bg = QtGui.QColor(color)
            c_bg.setAlpha(50)
            span = self._wave_seg_pool.acquire()
            span.setBrush(QtGui.QBrush(c_bg))
            _set_rect(span, seg_l, y0, width, y1 - y0)

            rect = self._track_seg_pool.acquire()
            c_rect = QtGui.QColor(color)
            if code == "B":
                c_rect.setAlpha(70)
            rect.setBrush(QtGui.QBrush(c_rect))
            _set_rect(rect, seg_l, 0, width, 1)
        self._wave_seg_pool.end()
        self._track_seg_pool.end()

        unread_l = max(playhead_t, t_start)
        unread_r = t_start + window_width
        if unread_r > unread_l:
            _set_rect(self._unread_wave, unread_l, y0, unread_r - unread_l, y1 - y0)
            _set_rect(self._unread_track, unread_l, 0, unread_r - unread_l, 1)
            self._unread_wave.setVisible(True)
            self._unread_track.setVisible(True)
        else:
            self._unread_wave.setVisible(False)
            self._unread_track.setVisible(False)

        for ln in self.playhead_wave_glow + [self.playhead_wave_core]:
            ln.setValue(playhead_t)
        for ln in self.playhead_track_glow + [self.playhead_track_core]:
            ln.setValue(playhead_t)

        cur_seg_idx = current_segment_index(playhead_t)
        cur_code = seg_codes[cur_seg_idx]
        current_value = float(all_assigned[seg_start_idx[cur_seg_idx]])
        info = code_info.get(cur_code, {"name": cur_code, "color": ACCENT_BLUE, "description": ""})
        marker_color = info["color"] if cur_code != "B" else ACCENT_BLUE
        self.marker_core.setData([playhead_t], [current_value], brush=pg.mkBrush(marker_color))

        # --- Run選択ドロップダウンを現在位置に同期(表示のみ、操作はまだ受け付けない) ---
        current_run = 0
        for run_idx, start_t in run_boundaries:
            if start_t <= t_now:
                current_run = run_idx
        if self.run_selector.currentIndex() != current_run:
            self.run_selector.blockSignals(True)
            self.run_selector.setCurrentIndex(min(current_run, self.run_selector.count() - 1))
            self.run_selector.blockSignals(False)

        # --- 直近デコード配列のピル表示 ---
        _nonB_mask = seg_codes != "B"
        seq_codes_nb = seg_codes[_nonB_mask]
        seq_start_nb = seg_start_t[_nonB_mask]
        n_shown = np.searchsorted(seq_start_nb, playhead_t, side="right")
        n_pills = len(self.pill_labels)
        latest = seq_codes_nb[max(0, n_shown - n_pills):n_shown]
        n_missing = n_pills - len(latest)
        for i, lbl in enumerate(self.pill_labels):
            j = i - n_missing
            if 0 <= j < len(latest):
                code = latest[j]
                cinfo = code_info.get(code, {"color": CARD_BORDER})
                is_last = (j == len(latest) - 1)
                border = f"2px solid {ACCENT_BLUE}" if is_last else "2px solid transparent"
                lbl.setText(code)
                lbl.setStyleSheet(
                    f"background-color:{cinfo['color']}; color:{contrasting_text_color(cinfo['color'])}; "
                    f"border-radius:8px; font-size:16px; font-weight:700; font-family:'DejaVu Sans Mono',monospace; border:{border};"
                )
            else:
                lbl.setText("")
                lbl.setStyleSheet(f"background-color:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:8px;")

        # --- Current call カード ---
        if cur_code == "B":
            self.call_letter.setText("\u2014")
            self.call_letter.setStyleSheet(
                f"color:{TEXT_MUTED}; font-size:56px; font-weight:800; font-family:'DejaVu Sans Mono',monospace; border:none;"
            )
            self.call_name.setText("baseline")
        else:
            self.call_letter.setText(cur_code)
            self.call_letter.setStyleSheet(
                f"color:{info['color']}; font-size:56px; font-weight:800; font-family:'DejaVu Sans Mono',monospace; border:none;"
            )
            name_text = info.get("name", cur_code)
            if cur_code in confusable_codes:
                name_text += f"  \u26A0 vs {confusable_codes[cur_code]}"
        if cur_code != "B" and cur_code in confusable_codes:
            self.call_name.setStyleSheet(f"color:{WARN_COLOR}; font-size:14px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            self.call_name.setText(name_text)
        elif cur_code != "B":
            self.call_name.setStyleSheet(f"color:{TEXT_MUTED}; font-size:14px; font-family:'DejaVu Sans Mono',monospace; border:none;")
            self.call_name.setText(info.get("name", cur_code))

        # --- 確信度(確率密度)計算 ---
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

        top1 = float(top_probs[0]) if len(top_probs) > 0 else 0.0
        self.call_conf_value.setText(f"{top1 * 100:.1f}%")
        self.call_conf_bar.set_value(top1, color=(marker_color if cur_code != "B" else ACCENT_BLUE))

        for i, (name_lbl, bar, pct_lbl) in enumerate(self.prob_rows):
            if i >= len(top_probs):
                name_lbl.setText("")
                bar.set_value(0.0)
                pct_lbl.setText("")
                continue
            p = float(top_probs[i])
            code = top_codes[i]
            name = code_info.get(code, {"name": code}).get("name", code)
            name_lbl.setText(f"{code} ({name})")
            color = code_info.get(code, {"color": ACCENT_BLUE})["color"]
            bar.set_value(p, color=color)
            pct_lbl.setText(f"{p * 100:.1f}%")

        # --- アセンブリ(depth / consensus / yield) ---
        while self.next_frag_idx < n_frag and frag_end_times[self.next_frag_idx] <= playhead_t:
            frag = fragments[self.next_frag_idx]
            self.cumulative_yield += frag["length"]
            if frag["align_start"] is not None:
                self.depth_counts[frag["align_start"]:frag["align_end"]] += 1
                for _pos, _conf in frag["pos_conf"]:
                    if not np.isnan(_conf):
                        self.depth_logodds[_pos] += _logit(_conf)
            self.next_frag_idx += 1

        self.depth_bars.setOpts(height=self.depth_counts)
        self.plot_depth.setYRange(0, max(5.0, float(self.depth_counts.max()) * 1.25))

        covered = self.depth_counts > 0
        consensus_accuracy = _sigmoid(self.depth_logodds)
        for pi in range(_ref_len):
            peak = float(consensus_accuracy[pi]) if covered[pi] else 0.0
            wy = peak * self._gauss_kernel
            self.acc_wave_lines[pi].setData(self.acc_wave_lines[pi].xData, wy)
        mean_accuracy = float(consensus_accuracy[covered].mean()) if covered.any() else 0.0
        mean_q = _prob_to_qscore(mean_accuracy) if mean_accuracy > 0 else 0.0

        confs = np.array([
            f["mean_conf"] for f in fragments[:self.next_frag_idx] if not np.isnan(f["mean_conf"])
        ])
        pass_rate = float((confs >= PASS_CONF_THRESHOLD).mean()) * 100 if len(confs) > 0 else 0.0

        self._yield_t_hist.append(playhead_t / 60.0)
        self._yield_v_hist.append(self.cumulative_yield)
        _set_glow_curve_data(self.yield_glow, self.yield_line, self._yield_t_hist, self._yield_v_hist)
        self.plot_yield.setXRange(0, max(playhead_t / 60.0 * 1.05, 1e-3))
        self.plot_yield.setYRange(0, max(self.cumulative_yield * 1.15, 5))

        self.kpi_reads_val.setText(f"{self.next_frag_idx:,} / {n_frag:,}")
        self.kpi_reads_sub.setText(f"{self.next_frag_idx / max(n_frag, 1) * 100:.1f}% of target")
        self.kpi_yield_val.setText(f"{self.cumulative_yield} bp")
        self.kpi_yield_sub.setText("Live yield")
        self.kpi_q_val.setText(f"{mean_q:.1f}")
        self.kpi_q_sub.setText("Phred quality score")
        self.kpi_pass_val.setText(f"{pass_rate:.1f}%")
        self.kpi_pass_sub.setText(f"\u2265 Q{PASS_Q_THRESHOLD:.0f}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = DashboardWindow()
    win.show()

    win.timer = QtCore.QTimer()
    win.timer.timeout.connect(win.tick)
    win.timer.start(interval_ms)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
