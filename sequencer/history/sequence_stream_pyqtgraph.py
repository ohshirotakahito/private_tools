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
    lbl.setStyleSheet("color:#888888; font-size:10px; font-family:monospace; font-weight:bold; border:none;")
    lbl.setAlignment(_Qt.AlignCenter)
    v.addWidget(lbl)

    val = QtWidgets.QLabel("--")
    val.setStyleSheet(
        f"color:{accent}; font-size:{value_fs}px; font-weight:bold; font-family:monospace; border:none;"
    )
    val.setAlignment(_Qt.AlignCenter)
    val.setWordWrap(True)
    v.addWidget(val)

    parent_layout.addWidget(frame)
    return val


# ============================
# メインウインドウ: 波形パネル + 配列トラック + 確信度ランキング + HUD
# ============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOLECULE CALLER — REAL-TIME NANOPORE SIGNAL DECODER")
        self.resize(1500, 820)
        self.setStyleSheet(f"background-color:{BG_COLOR};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # --- ブランドタイトル + LIVEインジケーター ---
        header = QtWidgets.QHBoxLayout()
        brand = QtWidgets.QLabel("MOLECULE CALLER")
        brand.setStyleSheet(
            f"color:{ACCENT_COLOR}; font-size:22px; font-weight:bold; font-family:monospace;"
        )
        sub = QtWidgets.QLabel("REAL-TIME NANOPORE SIGNAL DECODER")
        sub.setStyleSheet("color:#999999; font-size:11px; font-family:monospace; font-weight:bold;")
        title_box = QtWidgets.QVBoxLayout()
        title_box.addWidget(brand)
        title_box.addWidget(sub)
        header.addLayout(title_box)
        header.addStretch(1)
        self.live_label = QtWidgets.QLabel("\u25CF LIVE")
        self.live_label.setStyleSheet(
            f"color:{WARN_COLOR}; font-size:13px; font-weight:bold; font-family:monospace;"
        )
        header.addWidget(self.live_label, alignment=_Qt.AlignTop)
        root.addLayout(header)

        # --- 本体グリッド: 左(HUD+波形+トラック+トレイル) / 右(確信度ランキング) ---
        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, stretch=1)

        left_col = QtWidgets.QVBoxLayout()
        body.addLayout(left_col, stretch=4)

        # --- HUD読み取りパネル ---
        self.readout_frame = QtWidgets.QFrame()
        self.readout_frame.setStyleSheet(
            f"QFrame {{ background-color:#0A0A0A; border:2.5px solid {ACCENT_COLOR}; border-radius:8px; }}"
        )
        ro_layout = QtWidgets.QVBoxLayout(self.readout_frame)
        ro_layout.setContentsMargins(14, 8, 14, 8)
        self.readout_box = QtWidgets.QLabel("\u2014")
        self.readout_box.setStyleSheet(
            f"color:{ACCENT_COLOR}; font-size:{FS_HUD}px; font-weight:bold; font-family:monospace; border:none;"
        )
        self.readout_sub = QtWidgets.QLabel("")
        self.readout_sub.setStyleSheet("color:#DDDDDD; font-size:12px; font-family:monospace; font-weight:bold; border:none;")
        self.readout_sub.setWordWrap(True)
        self.readout_val = QtWidgets.QLabel("")
        self.readout_val.setStyleSheet(f"color:{ACCENT_COLOR2}; font-size:11px; font-family:monospace; border:none;")
        ro_layout.addWidget(self.readout_box)
        ro_layout.addWidget(self.readout_sub)
        ro_layout.addWidget(self.readout_val)
        self.readout_frame.setFixedHeight(110)

        hud_row = QtWidgets.QHBoxLayout()
        hud_row.addWidget(self.readout_frame, stretch=3)
        self.warn_badge = QtWidgets.QLabel("\u26A0 AMBIGUOUS")
        self.warn_badge.setFixedWidth(170)
        self.warn_badge.setFixedHeight(32)
        self.warn_badge.setAlignment(_Qt.AlignCenter)
        # 表示/非表示をsetVisible()で切り替えるとレイアウトが再計算されて
        # 画面がガタつくため、常にスペースは確保したまま色を透明にして
        # 「見えなくする」方式にする(サイズは常に固定)。
        self._warn_badge_style_on = (
            f"color:white; background-color:{WARN_COLOR}; border-radius:6px; padding:4px 8px; "
            "font-size:12px; font-weight:bold; font-family:monospace;"
        )
        self._warn_badge_style_off = (
            "color:transparent; background-color:transparent; border-radius:6px; padding:4px 8px; "
            "font-size:12px; font-weight:bold; font-family:monospace;"
        )
        self.warn_badge.setStyleSheet(self._warn_badge_style_off)
        hud_row.addWidget(self.warn_badge, stretch=0, alignment=_Qt.AlignVCenter)
        left_col.addLayout(hud_row)

        # --- 波形パネル + 配列トラック(GraphicsLayoutWidgetでリンクさせる) ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG_COLOR)
        left_col.addWidget(self.glw, stretch=6)

        self.plot_wave = self.glw.addPlot(row=0, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 5)
        self.plot_wave.showGrid(x=True, y=True, alpha=0.15)
        self.plot_wave.setLabel("left", "Value")
        self.plot_wave.getAxis("bottom").setStyle(showValues=False)
        self.plot_wave.addLegend(offset=(-10, 10))

        self.glw.nextRow()
        self.plot_track = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.plot_track.setXLink(self.plot_wave)
        self.plot_track.setYRange(0, 1.6)
        self.plot_track.getAxis("left").setStyle(showValues=False)
        self.plot_track.setLabel("bottom", "Time")
        self.plot_track.showGrid(x=False, y=False)

        margin = 0.1 * (all_raw.max() - all_raw.min() + 1e-9)
        self.plot_wave.setYRange(all_raw.min() - margin, all_raw.max() + margin)
        vb_wave = self.plot_wave.getViewBox()
        vb_track = self.plot_track.getViewBox()
        self.vb_wave = vb_wave
        self.vb_track = vb_track

        # 波形本体(Raw: 薄いシアン細線 / Assigned: 白の発光線)
        self.line_raw = pg.PlotDataItem(pen=pg.mkPen(ACCENT_COLOR2, width=1.1), name="Raw Signal")
        self.line_raw.setOpacity(0.55)
        self.plot_wave.addItem(self.line_raw)
        self.assigned_glow, self.line_assigned = _glow_curve(
            self.plot_wave, "#FFFFFF", base_width=2.6,
            layers=((7, 25), (3.5, 55)),
        )
        self.line_assigned.opts["name"] = "Assigned Signal"

        # プレイヘッド(波形/トラック両方で同期させる発光縦線)
        self.playhead_wave_glow, self.playhead_wave_core = _glow_vline(self.plot_wave, ACCENT_COLOR)
        self.playhead_track_glow, self.playhead_track_core = _glow_vline(self.plot_track, ACCENT_COLOR)

        # 判定マーカー(波形とAssigned信号の交点)
        self.marker_glow = pg.ScatterPlotItem(size=34, brush=pg.mkBrush(255, 255, 255, 30), pen=None)
        self.marker_ring = pg.ScatterPlotItem(size=18, brush=pg.mkBrush(255, 255, 255, 70), pen=None)
        self.marker_core = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(ACCENT_COLOR),
                                              pen=pg.mkPen("white", width=1.6))
        for m in (self.marker_glow, self.marker_ring, self.marker_core):
            m.setZValue(11)
            self.plot_wave.addItem(m)
        self.scan_caret = pg.ScatterPlotItem(size=13, symbol="t1", brush=pg.mkBrush(ACCENT_COLOR), pen=None)
        self.scan_caret.setZValue(13)
        self.plot_wave.addItem(self.scan_caret)

        # --- 動的アイテム用プール(実データのように1フレームに大量のセグメントが
        #     詰まっているケースでも、生成/破棄を繰り返さず使い回すことで
        #     UIスレッドの処理落ち(=ウインドウが「応答なし」になる)を防ぐ ---
        self._wave_seg_pool = _ItemPool(lambda: _rect_item(vb_wave, brush="#888888", z=0))
        self._track_seg_pool = _ItemPool(
            lambda: _rect_item(vb_track, brush="#888888", pen=pg.mkPen("#000000", width=0.5), z=1)
        )
        self._track_label_pool = _ItemPool(self._make_track_text)
        self._track_qv_pool = _ItemPool(lambda: _rect_item(vb_track, brush="#888888", z=2))
        self._track_qvtext_pool = _ItemPool(self._make_track_text)
        self._run_wave_line_pool = _ItemPool(lambda: self._make_vline(self.plot_wave))
        self._run_track_line_pool = _ItemPool(lambda: self._make_vline(self.plot_track))
        self._trail_pool = _ItemPool(self._make_trail_text)

        # 未読ゾーン・パルスハイライトは常に高々1個ずつなのでプール不要(使い回しの固定アイテム)
        self._unread_wave = _rect_item(vb_wave, brush=QtGui.QColor(0, 0, 0, 38), z=0.1)
        self._unread_track = _rect_item(vb_track, brush=QtGui.QColor(0, 0, 0, 90), z=0.1)
        self._pulse_wave = _rect_item(vb_wave, brush=QtGui.QColor(255, 255, 255, 26), z=0.5)
        self._pulse_track_border = _rect_item(vb_track, brush=None, pen=pg.mkPen(ACCENT_COLOR, width=1.5), z=2.5)
        for it in (self._unread_wave, self._unread_track, self._pulse_wave, self._pulse_track_border):
            it.setVisible(False)

        # --- 読み取り履歴トレイル ---
        self.trail_plot = self.glw.addPlot(row=2, col=0)
        self.glw.ci.layout.setRowStretchFactor(2, 1)
        self.trail_plot.setXRange(0, trail_window)
        self.trail_plot.setYRange(0, 1)
        self.trail_plot.hideAxis("left")
        self.trail_plot.hideAxis("bottom")
        self.trail_plot.setMouseEnabled(x=False, y=False)
        now_line = pg.InfiniteLine(pos=trail_window, angle=90,
                                    pen=pg.mkPen(ACCENT_COLOR, width=1.6, style=_Qt.SolidLine))
        self.trail_plot.addItem(now_line)
        header_txt = pg.TextItem("READ HISTORY", color="#888888", anchor=(1, 1))
        header_txt.setPos(trail_window, 1.15)
        self.trail_plot.addItem(header_txt)

        # --- 確信度ランキングパネル ---
        right_col = QtWidgets.QVBoxLayout()
        body.addLayout(right_col, stretch=2)
        self.prob_glw = pg.GraphicsLayoutWidget()
        self.prob_glw.setBackground(BG_COLOR)
        right_col.addWidget(self.prob_glw, stretch=1)
        self.prob_plot = self.prob_glw.addPlot()
        self.prob_plot.setXRange(0, 1.0)
        self.prob_plot.setYRange(-0.6, n_prob_rows - 0.4)
        self.prob_plot.getViewBox().invertY(True)
        self.prob_plot.hideAxis("left")
        self.prob_plot.getAxis("bottom").setTicks([[(0, "0%"), (0.5, "50%"), (1.0, "100%")]])
        self.prob_plot.setMouseEnabled(x=False, y=False)
        self.prob_plot.setTitle("CONFIDENCE", color=ACCENT_COLOR, size="13pt")

        vb = self.prob_plot.getViewBox()
        self.prob_bars = [_rect_item(vb, brush="#888888", z=2) for _ in range(n_prob_rows)]
        self.prob_glow_border = _rect_item(
            vb, brush=None, pen=pg.mkPen(ACCENT_COLOR, width=0), z=3
        )
        self.prob_name_labels = [pg.TextItem("", anchor=(0, 0.5)) for _ in range(n_prob_rows)]
        self.prob_pct_labels = [pg.TextItem("", anchor=(1, 0.5)) for _ in range(n_prob_rows)]
        for lbl in self.prob_name_labels + self.prob_pct_labels:
            self.prob_plot.addItem(lbl)
        self.prob_status_label = QtWidgets.QLabel("")
        self.prob_status_label.setStyleSheet("color:#888888; font-size:11px; font-family:monospace; font-weight:bold;")
        self.prob_status_label.setAlignment(_Qt.AlignCenter)
        right_col.addWidget(self.prob_status_label)

        # 状態(フレームをまたいで保持)
        self.frame = 0

    # --------------------------------------------------------------
    def _make_track_text(self):
        item = pg.TextItem("", anchor=(0.5, 0.5))
        self.plot_track.addItem(item)
        return item

    def _make_trail_text(self):
        item = pg.TextItem("", anchor=(0.5, 0.5))
        self.trail_plot.addItem(item)
        return item

    def _make_vline(self, plot_item):
        pen = pg.mkPen("#999999", width=1.4, style=_Qt.DashLine)
        line = pg.InfiniteLine(angle=90, pen=pen)
        plot_item.addItem(line)
        return line

    def update_frame(self, frame):
        self.frame = frame
        idx = min(frame * step, n_total - 1)
        t_now = all_time[idx]
        t_start = max(0.0, t_now - window_width)
        playhead_t = min(t_start + window_width * playhead_frac, t_now)

        # --- LIVE点滅 ---
        live_alpha = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(frame * 0.35))
        c = QtGui.QColor(WARN_COLOR)
        c.setAlphaF(live_alpha)
        self.live_label.setStyleSheet(
            f"color:rgba({c.red()},{c.green()},{c.blue()},{c.alphaF():.2f}); "
            "font-size:13px; font-weight:bold; font-family:monospace;"
        )

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
        self._track_label_pool.begin()
        self._track_qv_pool.begin()
        self._track_qvtext_pool.begin()

        # --- 波形背景の色分け & 配列トラックの色ブロック(再生ヘッドより右は非表示) ---
        # 実データはセグメント数が非常に多くなり得るため、1フレームに描画する数の
        # 上限を設けて安全弁にする(古いものから間引く)。
        lo, hi = visible_segment_range(t_start, t_now)
        MAX_VISIBLE_SEGMENTS = 500
        if hi - lo > MAX_VISIBLE_SEGMENTS:
            lo = hi - MAX_VISIBLE_SEGMENTS
        for si in range(lo, hi):
            code = seg_codes[si]
            info = code_info.get(code, {"name": code, "color": "#888888"})
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
            c_bg.setAlpha(46)
            span = self._wave_seg_pool.acquire()
            span.setBrush(QtGui.QBrush(c_bg))
            _set_rect(span, seg_l, y0, width, y1 - y0)

            rect = self._track_seg_pool.acquire()
            rect.setBrush(QtGui.QBrush(QtGui.QColor(color)))
            _set_rect(rect, seg_l, 0, width, 1)

            if code != "B" and width >= window_width * min_label_frac:
                txt_color = contrasting_text_color(color)
                label = self._track_label_pool.acquire()
                label.setText(code, color=txt_color)
                label.setPos(seg_l + width / 2, 0.5)

                seg_conf = seg_confidence[si]
                if not np.isnan(seg_conf):
                    seg_q = _prob_to_qscore(seg_conf)
                    qcolor = _quality_hex(seg_conf)
                    qv_bar = self._track_qv_pool.acquire()
                    qv_bar.setBrush(QtGui.QBrush(QtGui.QColor(qcolor)))
                    _set_rect(qv_bar, seg_l, 1.05, width, 0.16)
                    if width >= window_width * (min_label_frac * 1.5):
                        qv_text = self._track_qvtext_pool.acquire()
                        qv_text.setText(f"Q{seg_q:.0f}", color=qcolor)
                        qv_text.setPos(seg_l + width / 2, 1.35)

        self._wave_seg_pool.end()
        self._track_seg_pool.end()
        self._track_label_pool.end()
        self._track_qv_pool.end()
        self._track_qvtext_pool.end()

        # --- 未読ゾーンを暗く塗る(常に1個の固定アイテムを使い回す) ---
        unread_l = max(playhead_t, t_start)
        unread_r = t_start + window_width
        if unread_r > unread_l:
            _set_rect(self._unread_wave, unread_l, y0, unread_r - unread_l, y1 - y0)
            _set_rect(self._unread_track, unread_l, 0, unread_r - unread_l, 1.6)
            self._unread_wave.setVisible(True)
            self._unread_track.setVisible(True)
        else:
            self._unread_wave.setVisible(False)
            self._unread_track.setVisible(False)

        # --- run境界の縦線 ---
        self._run_wave_line_pool.begin()
        self._run_track_line_pool.begin()
        for run_idx, start_t in run_boundaries:
            if t_start <= start_t <= t_start + window_width:
                ln1 = self._run_wave_line_pool.acquire()
                ln1.setValue(start_t)
                ln2 = self._run_track_line_pool.acquire()
                ln2.setValue(start_t)
        self._run_wave_line_pool.end()
        self._run_track_line_pool.end()

        # --- 読み取り履歴トレイル ---
        # 実データは密度が高く、trail_window内に大量のセグメントが入り得るため
        # 直近の一定数だけに間引いて描画する(可読性とパフォーマンス両方の対策)。
        self._trail_pool.begin()
        _nonB_mask = seg_codes != "B"
        trail_codes = seg_codes[_nonB_mask]
        trail_start_t = seg_start_t[_nonB_mask]
        trail_hi = np.searchsorted(trail_start_t, playhead_t, side="right")
        trail_lo = np.searchsorted(trail_start_t, playhead_t - trail_window, side="left")
        MAX_TRAIL_ITEMS = 150
        if trail_hi - trail_lo > MAX_TRAIL_ITEMS:
            trail_lo = trail_hi - MAX_TRAIL_ITEMS
        edge_margin = trail_window * 0.02
        for ti in range(trail_lo, trail_hi):
            code = trail_codes[ti]
            elapsed = playhead_t - trail_start_t[ti]
            if elapsed < 0 or elapsed > trail_window:
                continue
            x = trail_window - max(elapsed, edge_margin)
            age_frac = elapsed / trail_window
            alpha = float(np.clip(1.0 - age_frac, 0.08, 1.0))
            info = code_info.get(code, {"color": "#AAAAAA"})
            col = QtGui.QColor(info["color"])
            col.setAlphaF(alpha)
            txt = self._trail_pool.acquire()
            txt.setText(code, color=col)
            txt.setPos(x, 0.5)
            font = QtGui.QFont(FONT_MONO, int(17 - 5 * age_frac), _FONT_BOLD)
            txt.setFont(font)
        self._trail_pool.end()

        # --- 現在読み取り中のセグメントをパルスハイライト(固定アイテムを使い回す) ---
        pulse = 0.5 + 0.5 * np.sin(frame * pulse_speed)
        cur_seg_idx = current_segment_index(playhead_t)
        cur_code = seg_codes[cur_seg_idx]
        if cur_code != "B":
            hl_l = max(seg_start_t[cur_seg_idx], t_start)
            hl_r = min(seg_end_t[cur_seg_idx], t_start + window_width)
            self._pulse_wave.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, int(26 * pulse))))
            _set_rect(self._pulse_wave, hl_l, y0, hl_r - hl_l, y1 - y0)
            self._pulse_track_border.setPen(pg.mkPen(ACCENT_COLOR, width=1.5 + pulse))
            _set_rect(self._pulse_track_border, hl_l, 0, hl_r - hl_l, 1)
            self._pulse_wave.setVisible(True)
            self._pulse_track_border.setVisible(True)
        else:
            self._pulse_wave.setVisible(False)
            self._pulse_track_border.setVisible(False)

        # --- 確信度(確率密度)パネル ---
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
            bar = self.prob_bars[rank]
            name_lbl = self.prob_name_labels[rank]
            pct_lbl = self.prob_pct_labels[rank]
            if rank >= len(top_probs):
                _set_rect(bar, 0, rank - 0.32, 0.0, 0.64)
                name_lbl.setText("")
                pct_lbl.setText("")
                continue
            p = float(top_probs[rank])
            code = top_codes[rank]
            color = top_colors[rank]
            is_top = (rank == 0)
            c_bar = QtGui.QColor(color)
            c_bar.setAlphaF(0.85 if is_top else 0.55)
            bar.setBrush(QtGui.QBrush(c_bar))
            _set_rect(bar, 0, rank - 0.32, max(p, 0.002), 0.64)

            name = code_info.get(code, {"name": code}).get("name", code)
            label_color = contrasting_text_color(color) if p > 0.22 else "#DDDDDD"
            label_x = 0.02 if p > 0.22 else min(p + 0.03, 0.97)
            name_lbl.setPos(label_x, rank)
            name_lbl.setText(f"{code} {name}", color=label_color)
            pct_lbl.setPos(0.985, rank)
            pct_lbl.setText(f"{p * 100:4.1f}%", color=(ACCENT_COLOR if is_top else "#888888"))

        if len(top_probs) > 0:
            self.prob_glow_border.setPen(pg.mkPen(ACCENT_COLOR, width=1.5 + pulse))
            _set_rect(self.prob_glow_border, 0, -0.36, 1.0, 0.72)
            top1 = float(top_probs[0])
        else:
            self.prob_glow_border.setPen(QtGui.QPen(_Qt.NoPen))
            top1 = 0.0

        locked = top1 > 0.9
        status = f"n={n_obs}  \u03c3={sigma:.3f}"
        if cur_code != "B":
            status += "   \u2713 LOCKED" if locked else "   ...reading"
        self.prob_status_label.setText(status)
        self.prob_status_label.setStyleSheet(
            f"color:{ACCENT_COLOR if locked else '#666666'}; font-size:11px; "
            "font-family:monospace; font-weight:bold;"
        )

        # --- プレイヘッド同期 ---
        for ln in self.playhead_wave_glow + [self.playhead_wave_core]:
            ln.setValue(playhead_t)
        for ln in self.playhead_track_glow + [self.playhead_track_core]:
            ln.setValue(playhead_t)

        # --- 判定マーカー ---
        current_value = float(all_assigned[seg_start_idx[cur_seg_idx]])
        info = code_info.get(cur_code, {"name": cur_code, "color": ACCENT_COLOR, "description": ""})
        marker_color = info["color"] if cur_code != "B" else ACCENT_COLOR
        for m in (self.marker_glow, self.marker_ring):
            m.setData([playhead_t], [current_value], brush=pg.mkBrush(QtGui.QColor(marker_color).lighter(150)))
        self.marker_core.setData([playhead_t], [current_value], brush=pg.mkBrush(marker_color))
        self.scan_caret.setData([playhead_t], [y1 - 0.03 * (y1 - y0)], brush=pg.mkBrush(marker_color))

        # --- HUD読み取りパネル ---
        if cur_code == "B":
            self.readout_box.setText("\u2014")
            self.readout_box.setStyleSheet(
                "color:#888888; font-size:%dpx; font-weight:bold; font-family:monospace; border:none;" % FS_HUD
            )
            self.readout_frame.setStyleSheet(
                "QFrame { background-color:#0A0A0A; border:2.5px solid #888888; border-radius:8px; }"
            )
            self.readout_sub.setText("baseline")
            self.readout_val.setText("")
        else:
            self.readout_box.setText(f"{cur_code}  {info['name']}")
            self.readout_box.setStyleSheet(
                f"color:{info['color']}; font-size:{FS_HUD}px; font-weight:bold; "
                "font-family:monospace; border:none;"
            )
            self.readout_frame.setStyleSheet(
                f"QFrame {{ background-color:#0A0A0A; border:2.5px solid {info['color']}; border-radius:8px; }}"
            )
            self.readout_sub.setText(info.get("description", ""))
            self.readout_val.setText(f"value \u2248 {current_value:.3f}")

        # --- 警告バッジ(setVisible()せず、常時表示のまま色だけ透明⇔赤に切り替える) ---
        if cur_code in confusable_codes:
            badge_pulse = 0.6 + 0.4 * np.sin(frame * pulse_speed * 2)
            self.warn_badge.setText(f"\u26A0 {cur_code}\u2194{confusable_codes[cur_code]}")
            self.warn_badge.setStyleSheet(
                f"color:white; background-color:rgba(255,59,48,{badge_pulse:.2f}); border-radius:6px; "
                "padding:4px 8px; font-size:12px; font-weight:bold; font-family:monospace;"
            )
        else:
            self.warn_badge.setText("\u26A0 AMBIGUOUS")
            self.warn_badge.setStyleSheet(self._warn_badge_style_off)

        return cur_code, cur_seg_idx, playhead_t, t_now


# ============================
# アセンブリウインドウ: 基準配列 + コンセンサストレース + Depth + Yield + ゲージ + KPI
# ============================
class AssemblyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEQUENCE ASSEMBLY")
        self.resize(760, 1000)
        self.setStyleSheet(f"background-color:{BG_COLOR};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("SEQUENCE ASSEMBLY")
        title.setStyleSheet(f"color:{ACCENT_COLOR}; font-size:18px; font-weight:bold; font-family:monospace;")
        header.addWidget(title)
        header.addStretch(1)
        self.live_label = QtWidgets.QLabel("\u25CF LIVE")
        self.live_label.setStyleSheet(f"color:{WARN_COLOR}; font-size:12px; font-weight:bold; font-family:monospace;")
        header.addWidget(self.live_label)
        root.addLayout(header)

        # --- 基準配列トラック(色付きボックス+文字) ---
        self.ref_row = QtWidgets.QHBoxLayout()
        self.ref_row.setSpacing(3)
        self.ref_boxes = []
        for c in reference_sequence:
            info = code_info.get(c, {"color": "#888888", "name": c})
            box = QtWidgets.QLabel(c)
            box.setAlignment(_Qt.AlignCenter)
            box.setFixedHeight(46)
            box.setStyleSheet(
                f"background-color:{info['color']}; color:{contrasting_text_color(info['color'])}; "
                "font-size:18px; font-weight:bold; font-family:monospace; border:2px solid white; border-radius:6px;"
            )
            self._ref_default_style = box.styleSheet()
            self.ref_row.addWidget(box)
            self.ref_boxes.append((box, info["color"]))
        root.addLayout(self.ref_row)

        # --- コンセンサストレース + Depthチャート(GraphicsLayoutWidget) ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(BG_COLOR)
        root.addWidget(self.glw, stretch=4)

        self.plot_trace = self.glw.addPlot(row=0, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 2)
        self.plot_trace.setXRange(0, _ref_len)
        self.plot_trace.setYRange(0, 1.08)
        self.plot_trace.getAxis("left").setTicks([[(0, "0%"), (0.5, "50%"), (1.0, "100%")]])
        self.plot_trace.setLabel("left", "accuracy")
        self.plot_trace.getAxis("bottom").setStyle(showValues=False)
        self.plot_trace.setTitle("CONSENSUS TRACE", color="#888888", size="10pt")
        ref_line = pg.InfiniteLine(pos=0.9, angle=0, pen=pg.mkPen(ACCENT_COLOR, width=1.4, style=_Qt.DotLine))
        self.plot_trace.addItem(ref_line)

        _GAUSS_SIGMA = 0.15
        self._gauss_x_local = np.linspace(-0.5, 0.5, 61)
        self._gauss_kernel = np.exp(-0.5 * (self._gauss_x_local / _GAUSS_SIGMA) ** 2)
        self.acc_wave_lines = []
        self.acc_wave_glow = []
        for pi in range(_ref_len):
            code = reference_sequence[pi]
            wcolor = code_info.get(code, {"color": ACCENT_COLOR})["color"]
            wx = pi + 0.5 + self._gauss_x_local
            glow = pg.PlotDataItem(wx, np.zeros_like(wx), pen=pg.mkPen(QtGui.QColor(wcolor), width=7))
            glow.setOpacity(0.20)
            line = pg.PlotDataItem(wx, np.zeros_like(wx), pen=pg.mkPen(wcolor, width=1.8))
            self.plot_trace.addItem(glow)
            self.plot_trace.addItem(line)
            self.acc_wave_glow.append(glow)
            self.acc_wave_lines.append(line)

        self.glw.nextRow()
        self.plot_depth = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(1, 3)
        self.plot_depth.setXLink(self.plot_trace)
        self.plot_depth.setXRange(0, _ref_len)
        self.plot_depth.setYRange(0, 5)
        self.plot_depth.setLabel("left", "depth")
        ax = self.plot_depth.getAxis("bottom")
        ax.setTicks([[(i + 0.5, c) for i, c in enumerate(reference_sequence)]])
        depth_colors = [code_info.get(c, {"color": "#888888"})["color"] for c in reference_sequence]
        self.depth_counts = np.zeros(_ref_len)
        self.depth_bars = pg.BarGraphItem(
            x=np.arange(_ref_len) + 0.5, height=self.depth_counts, width=0.7,
            brushes=[pg.mkBrush(c) for c in depth_colors], pen=pg.mkPen("white", width=1.2),
        )
        self.plot_depth.addItem(self.depth_bars)

        self.glw.nextRow()
        self.plot_yield = self.glw.addPlot(row=2, col=0)
        self.glw.ci.layout.setRowStretchFactor(2, 2)
        self.plot_yield.setLabel("bottom", "time")
        self.plot_yield.setLabel("left", "yield (bases)")
        self.yield_glow, self.yield_line = _glow_curve(
            self.plot_yield, ACCENT_COLOR2, base_width=1.8, layers=((6, 30), (3, 60))
        )
        self.yield_head = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(ACCENT_COLOR2), pen=pg.mkPen("white", width=1))
        self.plot_yield.addItem(self.yield_head)
        self._yield_t_hist, self._yield_v_hist = [], []

        # 現在ハイライト中の断片(基準配列上の縁取り)を示すオーバーレイ
        self._ref_highlight_prev = set()

        # --- 精度ヒーロー表示: ゲージ + トレンド ---
        hero_row = QtWidgets.QVBoxLayout()
        hero_label = QtWidgets.QLabel("MEAN CONSENSUS ACCURACY")
        hero_label.setAlignment(_Qt.AlignCenter)
        hero_label.setStyleSheet("color:#999999; font-size:11px; font-weight:bold; font-family:monospace;")
        hero_row.addWidget(hero_label)
        self.gauge = GaugeWidget()
        hero_row.addWidget(self.gauge)
        self.hero_qsub = QtWidgets.QLabel("")
        self.hero_qsub.setAlignment(_Qt.AlignCenter)
        self.hero_qsub.setStyleSheet(f"color:{ACCENT_COLOR2}; font-size:13px; font-weight:bold; font-family:monospace;")
        hero_row.addWidget(self.hero_qsub)
        self.trend_text = QtWidgets.QLabel("...")
        self.trend_text.setAlignment(_Qt.AlignCenter)
        self.trend_text.setStyleSheet("color:#666666; font-size:11px; font-weight:bold; font-family:monospace;")
        hero_row.addWidget(self.trend_text)
        self.warn_badge = QtWidgets.QLabel("\u26A0 AMBIGUOUS")
        self.warn_badge.setAlignment(_Qt.AlignCenter)
        self.warn_badge.setFixedHeight(26)
        # MainWindowと同様、setVisible()の出し入れによるレイアウトのガタつきを避け、
        # 色の透明⇔赤で表示/非表示を表現する。
        self._warn_badge_style_on = (
            f"color:white; background-color:{WARN_COLOR}; border-radius:6px; padding:3px 8px; "
            "font-size:11px; font-weight:bold; font-family:monospace;"
        )
        self._warn_badge_style_off = (
            "color:transparent; background-color:transparent; border-radius:6px; padding:3px 8px; "
            "font-size:11px; font-weight:bold; font-family:monospace;"
        )
        self.warn_badge.setStyleSheet(self._warn_badge_style_off)
        hero_row.addWidget(self.warn_badge)
        root.addLayout(hero_row)

        # --- KPIカード ---
        kpi_grid = QtWidgets.QGridLayout()
        kpi_grid.setSpacing(8)
        self.kpi_reads = make_kpi_card(_GridProxy(kpi_grid, 0, 0), "READS")
        self.kpi_depth = make_kpi_card(_GridProxy(kpi_grid, 0, 1), "TOTAL DEPTH")
        self.kpi_yield = make_kpi_card(_GridProxy(kpi_grid, 0, 2), "YIELD")
        self.kpi_latest = make_kpi_card(_GridProxy(kpi_grid, 0, 3), "LATEST READ", accent=ACCENT_COLOR, value_fs=12)
        self.kpi_n50 = make_kpi_card(_GridProxy(kpi_grid, 1, 0), "N50")
        self.kpi_cv = make_kpi_card(_GridProxy(kpi_grid, 1, 1), "DEPTH CV")
        self.kpi_pass = make_kpi_card(_GridProxy(kpi_grid, 1, 2), f"PASS RATE (\u2265Q{PASS_Q_THRESHOLD:.0f})", accent=ACCENT_COLOR)
        self.kpi_conf = make_kpi_card(_GridProxy(kpi_grid, 1, 3), "AMBIGUOUS?", accent=WARN_COLOR, value_fs=12)
        root.addLayout(kpi_grid)

        # --- アセンブリの状態(フレームをまたいで保持) ---
        self.next_frag_idx = 0
        self.reveal_time = -999.0
        self.reveal_frag = None
        self.cumulative_yield = 0
        self.depth_logodds = np.zeros(_ref_len)
        self.acc_history = []

    # --------------------------------------------------------------
    def update_frame(self, frame, playhead_t, cur_code):
        live_alpha = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(frame * 0.35))
        c = QtGui.QColor(WARN_COLOR)
        c.setAlphaF(live_alpha)
        self.live_label.setStyleSheet(
            f"color:rgba({c.red()},{c.green()},{c.blue()},{c.alphaF():.2f}); "
            "font-size:12px; font-weight:bold; font-family:monospace;"
        )

        while self.next_frag_idx < n_frag and frag_end_times[self.next_frag_idx] <= playhead_t:
            frag = fragments[self.next_frag_idx]
            self.cumulative_yield += frag["length"]
            if frag["align_start"] is not None:
                self.depth_counts[frag["align_start"]:frag["align_end"]] += 1
                for _pos, _conf in frag["pos_conf"]:
                    if not np.isnan(_conf):
                        self.depth_logodds[_pos] += _logit(_conf)
                self.reveal_time = frag["end_t"]
                self.reveal_frag = frag
            self.next_frag_idx += 1

        self.depth_bars.setOpts(height=self.depth_counts)
        self.plot_depth.setYRange(0, max(5.0, float(self.depth_counts.max()) * 1.25))

        covered = self.depth_counts > 0
        consensus_accuracy = _sigmoid(self.depth_logodds)
        for pi in range(_ref_len):
            peak = float(consensus_accuracy[pi]) if covered[pi] else 0.0
            wy = peak * self._gauss_kernel
            self.acc_wave_lines[pi].setData(self.acc_wave_lines[pi].xData, wy)
            self.acc_wave_glow[pi].setData(self.acc_wave_glow[pi].xData, wy)
        mean_accuracy = float(consensus_accuracy[covered].mean()) if covered.any() else 0.0
        mean_q = _prob_to_qscore(mean_accuracy) if mean_accuracy > 0 else 0.0

        if covered.sum() >= 2:
            d = self.depth_counts[covered]
            depth_cv = float(d.std() / d.mean())
        else:
            depth_cv = 0.0

        lengths = np.array([f["length"] for f in fragments[:self.next_frag_idx]])
        if len(lengths) > 0:
            sorted_len = np.sort(lengths)[::-1]
            cum = np.cumsum(sorted_len)
            half = cum[-1] / 2.0
            n50 = int(sorted_len[np.searchsorted(cum, half)])
        else:
            n50 = 0

        confs = np.array([
            f["mean_conf"] for f in fragments[:self.next_frag_idx] if not np.isnan(f["mean_conf"])
        ])
        pass_rate = float((confs >= PASS_CONF_THRESHOLD).mean()) * 100 if len(confs) > 0 else 0.0

        self._yield_t_hist.append(playhead_t)
        self._yield_v_hist.append(self.cumulative_yield)
        _set_glow_curve_data(self.yield_glow, self.yield_line, self._yield_t_hist, self._yield_v_hist)
        self.plot_yield.setXRange(0, max(playhead_t * 1.05, 1e-3))
        self.plot_yield.setYRange(0, max(self.cumulative_yield * 1.15, 5))
        self.yield_head.setData([playhead_t], [self.cumulative_yield])

        # --- 基準配列上のハイライト(現在アサインされた断片の範囲を縁取り) ---
        new_highlight = set()
        if self.reveal_frag is not None:
            age = playhead_t - self.reveal_time
            fade = float(np.clip(1.0 - age / _frag_reveal_decay, 0.0, 1.0))
            a_s, a_e = self.reveal_frag["align_start"], self.reveal_frag["align_end"]
            if fade > 0.01:
                new_highlight = set(range(a_s, a_e))
                for p in new_highlight:
                    box, base_color = self.ref_boxes[p]
                    alpha = int(255 * (0.4 + 0.6 * fade))
                    box.setStyleSheet(
                        f"background-color:{base_color}; color:{contrasting_text_color(base_color)}; "
                        f"font-size:18px; font-weight:bold; font-family:monospace; "
                        f"border:{2 + int(2 * fade)}px solid rgba(57,255,20,{alpha}); border-radius:6px;"
                    )
            arrow = "\u2192" if self.reveal_frag["orientation"] == "fwd" else "\u2190"
            latest_read_str = f"{self.reveal_frag['text']} {arrow} ref[{a_s}:{a_e}]"
        else:
            latest_read_str = "-"
        for p in self._ref_highlight_prev - new_highlight:
            box, base_color = self.ref_boxes[p]
            box.setStyleSheet(
                f"background-color:{base_color}; color:{contrasting_text_color(base_color)}; "
                "font-size:18px; font-weight:bold; font-family:monospace; border:2px solid white; border-radius:6px;"
            )
        self._ref_highlight_prev = new_highlight

        # --- 精度ゲージ ---
        if mean_accuracy >= 0.99:
            needle_color = ACCENT_COLOR
        elif mean_accuracy >= 0.9:
            needle_color = "#B6FF3B"
        else:
            needle_color = "#FFB020"
        self.gauge.set_value(mean_accuracy, label_text=f"{mean_accuracy * 100:.1f}%", needle_color=needle_color)
        self.hero_qsub.setText(f"Phred Q {mean_q:.1f}")

        # --- トレンド矢印 ---
        self.acc_history.append((playhead_t, mean_accuracy))
        while len(self.acc_history) > 2 and self.acc_history[0][0] < playhead_t - TREND_WINDOW - 1.0:
            self.acc_history.pop(0)
        baseline_acc = None
        for t_hist, a_hist in self.acc_history:
            if t_hist <= playhead_t - TREND_WINDOW:
                baseline_acc = a_hist
            else:
                break
        if baseline_acc is not None:
            delta_pp = (mean_accuracy - baseline_acc) * 100
            if abs(delta_pp) < 0.05:
                self.trend_text.setText(f"\u2192 \u00b10.0pp ({TREND_WINDOW:.1f}s)")
                self.trend_text.setStyleSheet("color:#888888; font-size:11px; font-weight:bold; font-family:monospace;")
            elif delta_pp > 0:
                self.trend_text.setText(f"\u25b2 +{delta_pp:.1f}pp ({TREND_WINDOW:.1f}s)")
                self.trend_text.setStyleSheet(f"color:{ACCENT_COLOR}; font-size:11px; font-weight:bold; font-family:monospace;")
            else:
                self.trend_text.setText(f"\u25bc {delta_pp:.1f}pp ({TREND_WINDOW:.1f}s)")
                self.trend_text.setStyleSheet(f"color:{WARN_COLOR}; font-size:11px; font-weight:bold; font-family:monospace;")
        else:
            self.trend_text.setText("...")
            self.trend_text.setStyleSheet("color:#666666; font-size:11px; font-weight:bold; font-family:monospace;")

        # --- 警告バッジ(色の透明⇔赤で切り替え。setVisible()は使わない) ---
        if cur_code in confusable_codes:
            self.warn_badge.setStyleSheet(self._warn_badge_style_on)
        else:
            self.warn_badge.setStyleSheet(self._warn_badge_style_off)

        # --- KPIカード ---
        self.kpi_reads.setText(f"{self.next_frag_idx}/{n_frag}")
        self.kpi_depth.setText(f"{int(self.depth_counts.sum())}")
        self.kpi_yield.setText(f"{self.cumulative_yield} bp")
        self.kpi_latest.setText(latest_read_str)
        self.kpi_n50.setText(f"{n50}")
        self.kpi_cv.setText(f"{depth_cv:.2f}")
        self.kpi_pass.setText(f"{pass_rate:.1f}%")
        if cur_code in confusable_codes:
            self.kpi_conf.setText(f"{cur_code}\u2194{confusable_codes[cur_code]}")
            self.kpi_conf.setStyleSheet(f"color:{WARN_COLOR}; font-size:12px; font-weight:bold; font-family:monospace;")
        else:
            self.kpi_conf.setText("clear")
            self.kpi_conf.setStyleSheet(f"color:{ACCENT_COLOR}; font-size:12px; font-weight:bold; font-family:monospace;")


class _GridProxy:
    """make_kpi_cardのaddWidget呼び出しを、QGridLayoutの特定セルに転送するための薄いプロキシ"""

    def __init__(self, grid, row, col):
        self.grid = grid
        self.row = row
        self.col = col

    def addWidget(self, widget):
        self.grid.addWidget(widget, self.row, self.col)


# ============================
# アプリケーション本体: 2つのウインドウを同じQTimerで同期させて更新する
# ============================
def main():
    app = QtWidgets.QApplication(sys.argv)

    main_win = MainWindow()
    asm_win = AssemblyWindow()

    main_win.show()
    asm_win.show()
    main_win.move(20, 20)
    asm_win.move(1560, 40)   # デュアルモニタ推奨。メインウインドウの右側に配置

    frame_counter = {"n": 0}

    def on_tick():
        frame = frame_counter["n"]
        if frame >= n_frames:
            timer.stop()
            return
        cur_code, cur_seg_idx, playhead_t, t_now = main_win.update_frame(frame)
        asm_win.update_frame(frame, playhead_t, cur_code)
        frame_counter["n"] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(on_tick)
    timer.start(interval_ms)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
