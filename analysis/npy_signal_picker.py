# -*- coding: utf-8 -*-
"""
npy_signal_picker.py

目的
----
extract_raw_value.py で test_data 配下に作った "*_raw.npy"
(Time,Raw Value のうち Raw Value のみを保存した信号データ) を
リスト化して1件ずつ読み込み、パルス検出(信号ピックアップ)結果を
可視化して確認するビューア。

■ 検出ロジック（ヒステリシス2閾値方式）
----------------------------------------
単一の閾値だけで判定すると、パルスの途中で信号が閾値付近を
ちょうど上下してしまい、1つのパルスが複数に分裂して検出されて
しまうことがある。これを避けるため、2本の閾値を使う
ヒステリシス(シュミットトリガ)方式にしている。

  t1(t) = baseline(t) + k1 * σ   … 低いほうの閾値（開始/終了判定用）
  t2(t) = baseline(t) + k2 * σ   … 高いほうの閾値（本物のシグナルか確定する用）
  （通常 k1 < k2）

状態遷移:
  IDLE      : y <= t1 の間はここ
  CANDIDATE : y が t1 を上回った時点(①)でここに遷移。まだ「仮」の状態。
              t2 に届かないまま t1 を下回ったら(=ノイズの単発超過とみなし)
              破棄して IDLE に戻る。
  CONFIRMED : CANDIDATE 中に y が t2 を上回った時点(②)で確定。
              その後 y が t2 を下回っても(③)ここに留まり続ける
              （パルスの裾を落とさない）。
  終了      : CONFIRMED 中に y が t1 を下回った時点(④)でパルスを確定して終了。

  記録する開始点は①（t1を上回った時点）、終了点は④（t1を下回った時点）。

baseline(t) はSavitzky-Golay平滑化＋移動下位パーセンタイルフィルタ、
σ(ノイズレベル)はMADベースの反復外れ値除去で頑健に推定している。
どちらもファイル全体で1回計算し、可視化にも検出にも同じものを使う
（以前は可視化用と検出用でベースラインの計算が別々で不整合だった）。

3種類のスライダーを用意:
  - View range スライダー: 表示するTime範囲をドラッグで拡大/縮小
  - Threshold k1 スライダー: 開始/終了判定用の閾値(低いほう)
  - Threshold k2 スライダー: シグナル確定用の閾値(高いほう)
    どちらも動かすたびに現在のファイルに対して検出をやり直すが、
    表示中のズーム範囲は維持したまま更新する。

最終的に「信号の位置(時間)・高さ・相対的な高さ(=高さ-ベースライン)・
ノイズレベル(σ)」をCSVに書き出せるようにしている(Exportボタン)。

使い方
------
1. 下の CONFIG セクションの TEST_DATA_ROOT を自分の環境に合わせて変更
2. python npy_signal_picker.py を実行
3. Prev/Next File で npy ファイルを切り替えながらパルス検出結果を確認
4. View range スライダーで見たい時間範囲を拡大/縮小
5. Threshold k1 / Threshold k2 スライダーで検出感度を調整
6. Reset View で全範囲表示に戻す
7. Export (This File) / Export CSV でCSVに書き出す

必要パッケージ: numpy, scipy, matplotlib(>=3.5, RangeSliderを使用)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter, label, find_objects
from pathlib import Path
from datetime import datetime


# ========= CONFIG =========

TEST_DATA_ROOT = Path("analysis/test_data")   # 環境に合わせて変更
NPY_PATTERN = "*_raw.npy"

DT = 0.0001                # サンプリング間隔[s]（extract_raw_value.py と同じ想定）

THRESHOLD_K1 = 3.0         # t1 = baseline + k1*sigma （開始/終了判定用、低いほう）
THRESHOLD_K2 = 6.0         # t2 = baseline + k2*sigma （シグナル確定用、高いほう）
THRESHOLD_K_MIN = 0.5
THRESHOLD_K_MAX = 15.0

MIN_WIDTH_MS = 0.2         # ①→④の幅がこれより短いパルスは無視
BASELINE_WINDOW = 501      # ベースライン推定の窓幅（奇数, サンプル数, 初期値）
BASELINE_PERCENTILE = 10   # 窓内の下位何%点をベースラインとするか（初期値。パルスは正方向のみなので低め推奨。
                            # パルスの時間占有率(duty比)が高いデータほど、さらに下げるか
                            # BASELINE_WINDOWを広げる必要がある）
BASELINE_WINDOW_MS_MIN = 5.0
BASELINE_WINDOW_MS_MAX = 150.0
BASELINE_PERCENTILE_MIN = 1.0
BASELINE_PERCENTILE_MAX = 40.0
SG_WINDOW = 51             # Savitzky-Golayフィルタの窓幅（奇数）
SG_POLY = 3

INITIAL_VIEW_WIDTH_S = 0.05  # 起動時・ファイル切替時の初期表示幅[s]

# ベースライン推定窓の半分に当たる端部を灰色で可視化する。
# 端部のシグナルも集計・CSV出力の対象にし、edge_region列で識別できるようにする。
EXCLUDE_EDGE_PULSES = False

MAX_TABLE_ROWS = 20        # 右側の一覧表に表示するパルス数の上限（表示範囲内のもの）

OUT_CSV_DEFAULT = "pulse_summary.csv"
EXPORT_CSV_DIR = Path("analysis/export_csv")  # Export CSV のファイル別出力先

PULSE_CSV_FIELDS = [
    "file", "pulse_index", "start_time_s", "end_time_s", "duration_ms",
    "width_ms", "peak", "baseline", "relative_height", "noise_std",
    "threshold_k1", "threshold_k2", "threshold_t1", "threshold_t2",
    "edge_region",
]


# ========= Utility =========

def robust_baseline_and_std(y, n_iter=5, clip_k=4.0):
    """外れ値（パルス）を反復的に除いてから baseline / ノイズレベルを推定する。

    通常の np.std(y) は、パルスがデータの大きな割合を占める場合に
    パルス自体の分散まで「ノイズ」として拾ってしまい、閾値が
    実データの最大値を超えるほど過大評価される問題があった。

    ここでは MAD(中央絶対偏差)ベースの頑健なσ推定から始め、
    中央値から clip_k * σ 以上外れた点（＝パルス候補）を除外して
    再計算する、を n_iter 回繰り返すことでベースライン領域だけの
    ノイズレベルに収束させる。
    """
    y = np.asarray(y, dtype=float)
    mask = np.ones(len(y), dtype=bool)
    med = float(np.median(y))
    sigma = float(np.std(y))

    for _ in range(n_iter):
        yy = y[mask]
        if len(yy) < 10:
            break
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(yy))
        new_mask = np.abs(y - med) < clip_k * sigma
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask

    return med, sigma


def list_npy_files(root_folder, pattern=NPY_PATTERN, recursive=True):
    root = Path(root_folder)
    if not root.is_dir():
        print(f"[WARN] フォルダが存在しません: {root}")
        return []

    files = sorted(root.rglob(pattern)) if recursive else sorted(root.glob(pattern))
    return files


def compute_baseline_curve(y, baseline_window=BASELINE_WINDOW,
                            baseline_percentile=BASELINE_PERCENTILE,
                            sg_window=SG_WINDOW, sg_poly=SG_POLY):
    """時間的に変動するベースライン曲線を推定する（可視化にも検出にも共通で使う）。

    ベースラインは単純な移動平均ではなく、窓内の下位パーセンタイル
    (percentile_filter)で推定する。パルスは常に正方向にしか出ないため、
    低いパーセンタイルを取ればパルスの影響をほぼ受けずに
    本来の静止レベル(≒ノイズだけの区間の中心)を追跡できる。
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.array([])

    if len(y) >= sg_window and sg_window % 2 == 1:
        # mode="nearest": デフォルトの"interp"は端を多項式で外挿するため、
        # 端にスパイク（パルスの途中でファイルが切れている場合など）があると
        # 大きくオーバーシュートすることがある。"nearest"で安全側にする。
        y_f = savgol_filter(y, window_length=sg_window, polyorder=sg_poly, mode="nearest")
    else:
        y_f = y.copy()

    w = baseline_window
    if w % 2 == 0:
        w += 1

    if len(y_f) >= w:
        # mode="reflect": "nearest"は端の1点を複製し続けるため
        # その1点の値に偏った推定になりやすい。"reflect"はデータを
        # 鏡写しに延長するので、より妥当な値になる。
        baseline = percentile_filter(y_f, percentile=baseline_percentile, size=w, mode="reflect")
    else:
        baseline = np.full_like(y_f, np.percentile(y_f, baseline_percentile))

    return baseline


def detect_pulses_hysteresis(y, baseline_curve, noise, dt=DT,
                              threshold_k1=THRESHOLD_K1, threshold_k2=THRESHOLD_K2,
                              min_width_ms=MIN_WIDTH_MS):
    """ヒステリシス(2段階)閾値によるパルス検出。

    t1 = baseline + k1*noise （開始/終了判定用、低いほう）
    t2 = baseline + k2*noise （シグナル確定用、高いほう）

    ①y>t1 で仮開始 → ②y>t2 で本物のシグナルと確定 →
    ③y<t2 に戻っても継続 → ④y<=t1 で終了、①〜④の区間をパルスとする。
    ②に一度も到達しないまま④相当(t1割れ)になった場合は破棄する。
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        t1_curve = baseline_curve + threshold_k1 * noise
        t2_curve = baseline_curve + threshold_k2 * noise
        return [], t1_curve, t2_curve

    t1_curve = baseline_curve + threshold_k1 * noise
    t2_curve = baseline_curve + threshold_k2 * noise

    # above_t1 の連続領域（候補）をラベリングして、各領域ごとに
    # above_t2 が一度でも真になっているかで確定判定をする。
    above_t1 = y > t1_curve
    pulses = []

    if not np.any(above_t1):
        return pulses, t1_curve, t2_curve

    labeled, nlabels = label(above_t1)
    objs = find_objects(labeled)

    for obj in objs:
        if obj is None:
            continue
        a = obj[0].start
        b = obj[0].stop  # b は exclusive

        # 元実装の挙動を模倣: 開始点は一つ前のサンプルを含める
        start = a - 1 if a > 0 else a
        end = b

        # 確定条件: 区間内に t2 を超えたサンプルが存在すること
        if not np.any(y[a:b] > t2_curve[a:b]):
            continue

        width_ms = (end - start) * dt * 1000
        if width_ms < min_width_ms:
            continue

        segment = y[start:end]
        local_baseline = float(baseline_curve[start])
        peak = float(np.max(segment))
        mean = float(np.mean(segment))
        area = float(np.sum(segment - local_baseline) * dt)

        pulses.append({
            "start_index": int(start),
            "end_index": int(end),
            "start_time_s": float(start * dt),
            "end_time_s": float(end * dt),
            "width_ms": float(width_ms),
            "peak": peak,
            "mean": mean,
            "area": area,
            "baseline": local_baseline,
            "threshold_t1": float(t1_curve[start]),
            "threshold_t2": float(t2_curve[start]),
        })

    return pulses, t1_curve, t2_curve


# ========= Core UI =========

class NpySignalPicker:
    def __init__(self, npy_files, dt=DT, threshold_k1=THRESHOLD_K1, threshold_k2=THRESHOLD_K2,
                 min_width_ms=MIN_WIDTH_MS, out_csv=OUT_CSV_DEFAULT,
                 initial_view_width=INITIAL_VIEW_WIDTH_S,
                 exclude_edge_pulses=EXCLUDE_EDGE_PULSES):
        if not npy_files:
            raise RuntimeError("npyファイルが1つも見つかりませんでした。")

        self.npy_files = npy_files
        self.dt = float(dt)
        self.threshold_k1 = float(threshold_k1)
        self.threshold_k2 = float(threshold_k2)
        self.min_width_ms = float(min_width_ms)
        self.out_csv = out_csv
        self.initial_view_width = float(initial_view_width)
        self.exclude_edge_pulses = bool(exclude_edge_pulses)
        # キャッシュ: キー=(path_str, mtime, length, baseline_window, baseline_percentile, sg_window, sg_poly)
        # 値={'baseline': array, 'std': float}
        self._baseline_cache = {}
        # ベースライン推定のパラメータ（スライダーで調整可能にする）
        self.baseline_window = BASELINE_WINDOW          # サンプル数（奇数）
        self.baseline_percentile = float(BASELINE_PERCENTILE)
        # ベースライン推定窓(baseline_window)の半分は、窓の片側がデータの
        # 外に出るため原理的に信頼性が低い。この時間幅を「端の不確実領域」
        # として可視化・デフォルトでは検出結果から除外する。
        self.edge_margin_s = (self.baseline_window // 2) * self.dt

        self.file_idx = 0
        self.data = None
        self.pulses = []
        self.t = np.array([])
        self.y = np.array([])
        self.baseline_curve = np.array([])
        self.t1_curve = np.array([])
        self.t2_curve = np.array([])
        self.std = 0.0
        self.pulse_artists = []
        self.table_offset = 0
        self._last_visible_pulses = []
        self._table_max_offset = 0
        self._scroll_thumb = None      # ドラッグ可能なつまみ(Rectangle)
        self._scroll_dragging = False
        self._scroll_drag_anchor_y = None   # ドラッグ開始時のマウスY(axes座標)
        self._scroll_drag_anchor_offset = None  # ドラッグ開始時のoffset

        self.show_signal = True
        self.show_baseline = True
        self.show_threshold_t1 = True
        self.show_threshold_t2 = True
        self.show_pulse = True
        self.show_peak_value = True

        self.fig, self.ax = plt.subplots(figsize=(12, 7.5))
        plt.subplots_adjust(left=0.13, right=0.70, bottom=0.34, top=0.94)

        # 右側: 表示範囲内のパルス一覧表（＋縦スクロールバー）
        self.ax_table = plt.axes([0.73, 0.34, 0.22, 0.60])
        self.ax_table.axis("off")

        self.ax_table_scroll = plt.axes([0.965, 0.34, 0.015, 0.60])
        self.ax_table_scroll.set_xlim(0, 1)
        self.ax_table_scroll.set_ylim(0, 1)
        self.ax_table_scroll.axis("off")
        # うっすらしたトラック（背景の細い帯）
        self.ax_table_scroll.add_patch(
            Rectangle((0.30, 0.0), 0.4, 1.0, facecolor="#e8e8e8",
                     edgecolor="none", zorder=1)
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_scroll_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_scroll_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_scroll_release)

        ax_check = plt.axes([0.01, 0.55, 0.11, 0.28])
        self.check = CheckButtons(
            ax_check,
            ["Signal", "Baseline", "Threshold t1", "Threshold t2", "Pulse", "Peak value"],
            [self.show_signal, self.show_baseline, self.show_threshold_t1,
             self.show_threshold_t2, self.show_pulse, self.show_peak_value],
        )
        self.check.on_clicked(self.on_check)

        # 表ポップアウト & 表示範囲評価ボタン（簡易配置）
        ax_show_table = plt.axes([0.01, 0.49, 0.11, 0.04])
        self.btn_show_table = Button(ax_show_table, "Show Table")
        self.btn_show_table.on_clicked(self.on_show_table)

        ax_eval_view = plt.axes([0.01, 0.44, 0.11, 0.04])
        self.btn_eval_view = Button(ax_eval_view, "Evaluate View")
        self.btn_eval_view.on_clicked(self.on_evaluate_view)

        # 表示範囲スライダーはファイルごとに作り直す（ウィンドウサイズと開始位置の2スライダ）
        self.window_size_slider = None
        self.window_start_slider = None

        # 閾値スライダー（k1, k2 独立、ファイルをまたいで値を維持する）
        ax_k1 = plt.axes([0.18, 0.20, 0.65, 0.03])
        self.k1_slider = Slider(
            ax_k1, "Threshold k1 (start/end)",
            valmin=THRESHOLD_K_MIN, valmax=THRESHOLD_K_MAX,
            valinit=self.threshold_k1, valstep=0.1,
        )
        self.k1_slider.on_changed(self.on_k1_change)

        ax_k2 = plt.axes([0.18, 0.16, 0.65, 0.03])
        self.k2_slider = Slider(
            ax_k2, "Threshold k2 (confirm)",
            valmin=THRESHOLD_K_MIN, valmax=THRESHOLD_K_MAX,
            valinit=self.threshold_k2, valstep=0.1,
        )
        self.k2_slider.on_changed(self.on_k2_change)

        # ベースライン推定パラメータ（窓幅・パーセンタイル）
        init_window_ms = self.baseline_window * self.dt * 1000
        ax_bw = plt.axes([0.18, 0.12, 0.65, 0.03])
        self.bw_slider = Slider(
            ax_bw, "Baseline window (ms)",
            valmin=BASELINE_WINDOW_MS_MIN, valmax=BASELINE_WINDOW_MS_MAX,
            valinit=init_window_ms, valstep=1.0,
        )
        self.bw_slider.on_changed(self.on_baseline_window_change)

        ax_bp = plt.axes([0.18, 0.08, 0.65, 0.03])
        self.bp_slider = Slider(
            ax_bp, "Baseline percentile (%)",
            valmin=BASELINE_PERCENTILE_MIN, valmax=BASELINE_PERCENTILE_MAX,
            valinit=self.baseline_percentile, valstep=1.0,
        )
        self.bp_slider.on_changed(self.on_baseline_percentile_change)

        # ボタン行
        ax_prev = plt.axes([0.14, 0.005, 0.10, 0.05])
        ax_next = plt.axes([0.25, 0.005, 0.10, 0.05])
        ax_reset = plt.axes([0.36, 0.005, 0.10, 0.05])
        ax_exp_cur = plt.axes([0.47, 0.005, 0.16, 0.05])
        ax_exp_all = plt.axes([0.65, 0.005, 0.16, 0.05])
        ax_stop = plt.axes([0.83, 0.005, 0.10, 0.05])

        self.btn_prev = Button(ax_prev, "Prev File")
        self.btn_next = Button(ax_next, "Next File")
        self.btn_reset = Button(ax_reset, "Reset View")
        self.btn_exp_cur = Button(ax_exp_cur, "Export (This File)")
        self.btn_exp_all = Button(ax_exp_all, "Export CSV")
        self.btn_stop = Button(ax_stop, "Stop")

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset_view)
        self.btn_exp_cur.on_clicked(self.on_export_current)
        self.btn_exp_all.on_clicked(self.on_export_all)
        self.btn_stop.on_clicked(self.on_stop)

        self.load_file(0)
        self.full_reload()

    # ----- data handling -----

    def current_path(self):
        return self.npy_files[self.file_idx]

    def load_file(self, idx):
        self.file_idx = idx % len(self.npy_files)
        path = self.current_path()
        try:
            self.data = np.load(path)
        except Exception as e:
            print(f"[ERROR] 読み込み失敗: {path}\n  -> {e}")
            self.data = np.array([])

    def build_pulse_rows(self, path, y, pulses, baseline_curve, std):
        """CSV出力用の行リストを作る（位置・持続時間・高さ・相対的な高さ・ノイズレベル）。"""
        file_duration = len(y) * self.dt
        margin = self.edge_margin_s
        rows = []
        for i, p in enumerate(pulses):
            local_baseline = p["baseline"]
            relative_height = p["peak"] - local_baseline
            is_edge = (p["start_time_s"] < margin) or (p["end_time_s"] > file_duration - margin)
            # 持続時間 = (終了点 - 開始点) + 1サンプル分[ms]
            duration_ms = (p["end_time_s"] - p["start_time_s"]) * 1000 + self.dt * 1000
            rows.append({
                "file": str(path),
                "pulse_index": i,
                "start_time_s": p["start_time_s"],
                "end_time_s": p["end_time_s"],
                "duration_ms": duration_ms,
                "width_ms": p["width_ms"],
                "peak": p["peak"],
                "baseline": local_baseline,
                "relative_height": relative_height,
                "noise_std": std,
                "threshold_k1": self.threshold_k1,
                "threshold_k2": self.threshold_k2,
                "threshold_t1": p["threshold_t1"],
                "threshold_t2": p["threshold_t2"],
                "edge_region": is_edge,
            })
        return rows

    # ----- view range slider -----

    def create_view_slider(self, total_duration):
        # ウィンドウサイズ(表示幅) と ウィンドウ開始位置 の2スライダを生成
        total_duration = max(total_duration, 1e-6)
        init_width = min(self.initial_view_width, total_duration)
        # 以前のスライダがあれば軸を削除してから作り直す
        try:
            if self.window_size_slider is not None:
                self.window_size_slider.ax.remove()
        except Exception:
            pass
        try:
            if self.window_start_slider is not None:
                self.window_start_slider.ax.remove()
        except Exception:
            pass

        ax_size = plt.axes([0.18, 0.28, 0.65, 0.03])
        self.window_size_slider = Slider(
            ax_size, "Window size (s)",
            valmin=DT, valmax=total_duration, valinit=init_width, valstep=DT,
        )
        self.window_size_slider.on_changed(self.on_window_size_change)

        ax_start = plt.axes([0.18, 0.24, 0.65, 0.03])
        max_start = max(0.0, total_duration - init_width)
        self.window_start_slider = Slider(
            ax_start, "Window start (s)",
            valmin=0.0, valmax=max_start, valinit=0.0, valstep=DT,
        )
        self.window_start_slider.on_changed(self.on_window_start_change)

    def on_view_change(self, val):
        view_start, view_end = val
        self.update_view(view_start, view_end)

    def on_reset_view(self, event):
        # Reset view to start=0, size=initial or total_duration
        if self.window_size_slider is None or self.window_start_slider is None:
            return
        total_duration = self.window_size_slider.valmax
        init_width = min(self.initial_view_width, total_duration)
        self.window_size_slider.set_val(init_width)
        self.window_start_slider.set_val(0.0)

    def current_view_range(self):
        if self.window_size_slider is not None and self.window_start_slider is not None:
            start = float(self.window_start_slider.val)
            size = float(self.window_size_slider.val)
            return start, min(start + size, self.window_size_slider.valmax)
        n = len(self.y)
        return 0.0, n * self.dt

    def on_window_size_change(self, val):
        # ウィンドウサイズが変わったら開始スライダの上限を調整し、ビュー更新
        if self.window_start_slider is None:
            return
        total = self.window_size_slider.valmax
        size = float(val)
        # update start slider max
        new_max = max(0.0, total - size)
        # recreate start slider axis range by setting valmax attribute
        try:
            self.window_start_slider.ax.set_xlim(0.0, new_max)
        except Exception:
            pass
        # clamp start value
        if self.window_start_slider.val > new_max:
            self.window_start_slider.set_val(new_max)

        view_start, view_end = self.current_view_range()
        self.update_view(view_start, view_end)

    def on_window_start_change(self, val):
        view_start, view_end = self.current_view_range()
        self.update_view(view_start, view_end)

    # ----- threshold sliders -----

    def on_k1_change(self, val):
        self.threshold_k1 = float(val)
        self._recompute_and_redraw_keep_view()

    def on_k2_change(self, val):
        self.threshold_k2 = float(val)
        self._recompute_and_redraw_keep_view()

    def on_baseline_window_change(self, val):
        # ms -> 奇数サンプル数に変換
        n_samples = int(round((val / 1000.0) / self.dt))
        if n_samples < 3:
            n_samples = 3
        if n_samples % 2 == 0:
            n_samples += 1
        self.baseline_window = n_samples
        self.edge_margin_s = (self.baseline_window // 2) * self.dt
        self._recompute_and_redraw_keep_view()

    def on_baseline_percentile_change(self, val):
        self.baseline_percentile = float(val)
        self._recompute_and_redraw_keep_view()

    def _recompute_and_redraw_keep_view(self):
        if len(self.data) == 0:
            return
        view_start, view_end = self.current_view_range()
        self.recompute_detection()
        self.redraw_plot()
        self.update_view(view_start, view_end)

    # ----- detection / drawing -----

    def recompute_detection(self):
        """現在のファイル(self.data)に対して、現在の k1/k2 で
        ベースライン曲線・閾値曲線・パルスを計算し直す。"""
        y = self.data
        n = len(y)

        if n == 0:
            self.t = np.array([])
            self.y = np.array([])
            self.baseline_curve = np.array([])
            self.t1_curve = np.array([])
            self.t2_curve = np.array([])
            self.pulses = []
            self.std = 0.0
            return

        t = np.arange(n) * self.dt
        path = self.current_path()
        baseline_curve, std = self._get_baseline_and_std(y, path)
        pulses, t1_curve, t2_curve = detect_pulses_hysteresis(
            y, baseline_curve, std, dt=self.dt,
            threshold_k1=self.threshold_k1, threshold_k2=self.threshold_k2,
            min_width_ms=self.min_width_ms,
        )

        self.t = t
        self.y = y
        self.baseline_curve = baseline_curve
        self.t1_curve = t1_curve
        self.t2_curve = t2_curve
        self.pulses = pulses
        self.std = std

    def _get_baseline_and_std(self, y, path):
        """ベースライン曲線とノイズ標準偏差をキャッシュ付きで取得する。

        キャッシュキーはファイルパス・mtime・長さ・ベースライン関連パラメータで構成。
        """
        p = Path(path)
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = None

        key = (str(path), mtime, len(y), int(self.baseline_window), float(self.baseline_percentile), SG_WINDOW, SG_POLY)
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached["baseline"], cached["std"]

        baseline_curve = compute_baseline_curve(
            y, baseline_window=self.baseline_window,
            baseline_percentile=self.baseline_percentile,
            sg_window=SG_WINDOW, sg_poly=SG_POLY,
        )
        _, std = robust_baseline_and_std(y)

        # キャッシュ保存
        try:
            # keep a reference to arrays
            self._baseline_cache[key] = {"baseline": baseline_curve, "std": std}
        except Exception:
            pass

        return baseline_curve, std

    def redraw_plot(self):
        """信号/ベースライン/閾値の線を描き直す（パルス注釈は含まない）。"""
        self.ax.clear()
        self.pulse_artists = []

        path = self.current_path()

        if len(self.y) == 0:
            self.ax.text(0.5, 0.5, f"Failed to read:\n{path}",
                         ha="center", va="center", transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return

        if self.show_signal:
            self.ax.plot(self.t, self.y, label="Signal", linewidth=1.0)
        if self.show_baseline:
            self.ax.plot(self.t, self.baseline_curve, label="Baseline", linewidth=2.0)
        if self.show_threshold_t1:
            self.ax.plot(self.t, self.t1_curve,
                         label=f"Threshold t1 (+{self.threshold_k1:.1f}σ)",
                         linewidth=1.5, linestyle="--")
        if self.show_threshold_t2:
            self.ax.plot(self.t, self.t2_curve,
                         label=f"Threshold t2 (+{self.threshold_k2:.1f}σ)",
                         linewidth=2.0)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Signal Amplitude")
        self.ax.grid(True, alpha=0.2)

        self.ax.text(0.01, 0.99, str(path), transform=self.ax.transAxes,
                     fontsize=8, va="top", bbox=dict(boxstyle="round", alpha=0.2))
        # 出力CSVパスを表示
        try:
            self.ax.text(0.01, 0.94, f"Out CSV: {self.out_csv}", transform=self.ax.transAxes,
                         fontsize=8, va="top", bbox=dict(boxstyle="round", alpha=0.15))
        except Exception:
            pass

        duration = self.t[-1] + self.dt if len(self.t) else 0.0
        margin = self.edge_margin_s
        if margin > 0 and duration > 0:
            self.ax.axvspan(0, min(margin, duration), color="gray", alpha=0.15,
                            label="Edge (unreliable baseline)")
            self.ax.axvspan(max(duration - margin, 0), duration, color="gray", alpha=0.15)

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right", fontsize=8)

    def full_reload(self):
        """ファイル切替時のフル再構築。検出のやり直し＋表示範囲スライダーの
        作り直し（そのファイルの長さに合わせる）を行う。"""
        self.recompute_detection()
        self.redraw_plot()

        if len(self.y) == 0:
            return

        total_duration = len(self.y) * self.dt
        self.create_view_slider(total_duration)

        view_start, view_end = self.current_view_range()
        self.update_view(view_start, view_end)

    def update_view(self, view_start, view_end):
        """表示範囲の変更に伴う軽量更新。パルス注釈と軸範囲だけ更新する。"""
        for artist in self.pulse_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.pulse_artists = []

        if len(self.y) == 0:
            self.fig.canvas.draw_idle()
            return

        visible_pulses = [
            p for p in self.pulses
            if not (p["end_time_s"] < view_start or p["start_time_s"] > view_end)
        ]

        if self.show_pulse:
            for i, p in enumerate(visible_pulses):
                span = self.ax.axvspan(p["start_time_s"], p["end_time_s"], alpha=0.25, color="orange")
                self.pulse_artists.append(span)

                seg = self.y[p["start_index"]:p["end_index"]]
                peak_local_idx = p["start_index"] + int(np.argmax(seg))
                peak_time = self.t[peak_local_idx]
                peak_value = self.y[peak_local_idx]
                rel_height = peak_value - self.baseline_curve[peak_local_idx]

                start_idx = p["start_index"]
                end_idx = min(p["end_index"], len(self.y) - 1)
                start_time = self.t[start_idx]
                end_time = self.t[end_idx]
                start_value = self.y[start_idx]
                end_value = self.y[end_idx]

                # 開始点・終了点・ピーク点を色分けして表示
                start_marker, = self.ax.plot(
                    start_time, start_value, marker="^", markersize=9,
                    linestyle="None", color="limegreen",
                    label="Start" if i == 0 else None,
                )
                self.pulse_artists.append(start_marker)

                end_marker, = self.ax.plot(
                    end_time, end_value, marker="v", markersize=9,
                    linestyle="None", color="dodgerblue",
                    label="End" if i == 0 else None,
                )
                self.pulse_artists.append(end_marker)

                peak_marker, = self.ax.plot(
                    peak_time, peak_value, marker="o", markersize=8,
                    linestyle="None", color="red",
                    label="Peak" if i == 0 else None,
                )
                self.pulse_artists.append(peak_marker)

                if self.show_peak_value:
                    txt = self.ax.text(
                        peak_time, peak_value,
                        f"h={peak_value:.3f}\nΔ={rel_height:.3f}",
                        fontsize=8, ha="center", va="bottom",
                    )
                    self.pulse_artists.append(txt)

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right", fontsize=8)

        self.ax.set_xlim(view_start, view_end)

        # 表示範囲内のデータに合わせてy軸を自動調整
        i0 = int(np.floor(view_start / self.dt))
        i1 = int(np.ceil(view_end / self.dt))
        i0 = max(0, min(i0, len(self.y)))
        i1 = max(0, min(i1, len(self.y)))

        if i1 > i0:
            y_view = self.y[i0:i1]
            ymin, ymax = float(np.min(y_view)), float(np.max(y_view))
            margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
            self.ax.set_ylim(ymin - margin, ymax + margin)

        path = self.current_path()
        bw_ms = self.baseline_window * self.dt * 1000
        self.ax.set_title(
            f"File {self.file_idx+1}/{len(self.npy_files)} | {path.name} | "
            f"Pulses in view: {len(visible_pulses)}/{len(self.pulses)} | "
            f"k1={self.threshold_k1:.1f} k2={self.threshold_k2:.1f} std≈{self.std:.4g} | "
            f"bw={bw_ms:.0f}ms bp={self.baseline_percentile:.0f}% | "
            f"view=[{view_start:.4f}, {view_end:.4f}]s",
            fontsize=9,
        )

        self._update_table(visible_pulses)

        self.fig.canvas.draw_idle()

    def _update_table(self, visible_pulses):
        """表示範囲内のパルスについて、開始時刻・ピーク時刻・終了時刻・
        持続時間・絶対強度(ピーク値)・相対強度・ノイズレベルの
        7項目を一覧表として右側パネルに描画する。
        件数がMAX_TABLE_ROWSを超える場合は縦スクロールバー(つまみ)で送れるようにする。"""
        self._last_visible_pulses = sorted(visible_pulses, key=lambda p: p["start_time_s"])
        n_total = len(self._last_visible_pulses)

        self.table_offset = 0
        self._table_max_offset = max(0, n_total - MAX_TABLE_ROWS)

        self._draw_scroll_thumb()
        self._render_table_rows()

    def _draw_scroll_thumb(self):
        """スクロールバーのつまみ(小さい丸みのある四角)を、現在の
        table_offset に応じた位置・サイズで描き直す。"""
        if self._scroll_thumb is not None:
            try:
                self._scroll_thumb.remove()
            except Exception:
                pass
            self._scroll_thumb = None

        n_total = len(self._last_visible_pulses)
        if n_total <= MAX_TABLE_ROWS or self._table_max_offset <= 0:
            self.fig.canvas.draw_idle()
            return

        # つまみの高さ = 表示できている割合（最低でも見える程度の高さは確保）
        thumb_h = max(0.08, MAX_TABLE_ROWS / n_total)
        thumb_h = min(thumb_h, 1.0)

        # offset=0(先頭)のとき上端、offset=max(末尾)のとき下端に来るようにする
        frac_scrolled = self.table_offset / self._table_max_offset if self._table_max_offset else 0.0
        thumb_top = 1.0 - frac_scrolled * (1.0 - thumb_h)
        thumb_bottom = thumb_top - thumb_h

        self._scroll_thumb = FancyBboxPatch(
            (0.15, thumb_bottom), 0.7, thumb_h,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor="#8a8a8a", edgecolor="none", zorder=2,
            mutation_aspect=0.05,
        )
        self.ax_table_scroll.add_patch(self._scroll_thumb)
        self.fig.canvas.draw_idle()

    def _scroll_offset_from_axes_y(self, y_axes):
        """スクロールバーaxes内のy座標(0=下,1=上)から table_offset を計算する。"""
        n_total = len(self._last_visible_pulses)
        if n_total <= MAX_TABLE_ROWS or self._table_max_offset <= 0:
            return 0
        thumb_h = max(0.08, MAX_TABLE_ROWS / n_total)
        thumb_h = min(thumb_h, 1.0)
        travel = max(1e-6, 1.0 - thumb_h)
        # y_axes=1(一番上)がoffset0、y_axes=0(一番下)がoffset最大
        frac = (1.0 - thumb_h / 2 - y_axes) / travel
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * self._table_max_offset))

    def _on_scroll_press(self, event):
        if event.inaxes != self.ax_table_scroll or self._scroll_thumb is None:
            return
        if event.ydata is None:
            return

        contains, _ = self._scroll_thumb.contains(event)
        if contains:
            # つまみを直接ドラッグ開始
            self._scroll_dragging = True
            self._scroll_drag_anchor_y = event.ydata
            self._scroll_drag_anchor_offset = self.table_offset
        else:
            # トラックの余白をクリック -> その位置へジャンプ
            self.table_offset = self._scroll_offset_from_axes_y(event.ydata)
            self._draw_scroll_thumb()
            self._render_table_rows()
            self.fig.canvas.draw_idle()

    def _on_scroll_motion(self, event):
        if not self._scroll_dragging:
            return
        if event.ydata is None or event.inaxes != self.ax_table_scroll:
            return

        n_total = len(self._last_visible_pulses)
        if n_total <= MAX_TABLE_ROWS or self._table_max_offset <= 0:
            return
        thumb_h = max(0.08, MAX_TABLE_ROWS / n_total)
        thumb_h = min(thumb_h, 1.0)
        travel = max(1e-6, 1.0 - thumb_h)

        dy = event.ydata - self._scroll_drag_anchor_y
        # 上方向(+y)へ動かすとoffsetが減る(先頭側)、下方向でoffsetが増える(末尾側)
        d_offset = -(dy / travel) * self._table_max_offset
        new_offset = self._scroll_drag_anchor_offset + d_offset
        self.table_offset = int(round(max(0, min(self._table_max_offset, new_offset))))

        self._draw_scroll_thumb()
        self._render_table_rows()
        self.fig.canvas.draw_idle()

    def _on_scroll_release(self, event):
        self._scroll_dragging = False
        self._scroll_drag_anchor_y = None
        self._scroll_drag_anchor_offset = None

    def _render_table_rows(self):
        """self._last_visible_pulses のうち、現在のスクロール位置に応じた
        MAX_TABLE_ROWS件だけを表として描画する。"""
        self.ax_table.clear()
        self.ax_table.axis("off")

        n_total = len(self._last_visible_pulses)

        if n_total == 0:
            self.ax_table.text(0.5, 0.5, "No pulses in view",
                               ha="center", va="center", fontsize=9)
            return

        offset = max(0, min(self.table_offset, max(0, n_total - MAX_TABLE_ROWS)))
        # 表に収まる最大行数まで表示する。件数が上限を超える場合は、
        # スクロール位置(offset)を使ってすべてのパルスを参照できるようにする。
        shown = self._last_visible_pulses[offset:offset + MAX_TABLE_ROWS]

        col_labels = ["Start\n(ms)", "Peak\n(ms)", "End\n(ms)", "Duration\n(ms)",
                     "Peak\n(abs)", "Δ\n(rel)", "σ\n(noise)"]
        cell_text = []
        for p in shown:
            seg = self.y[p["start_index"]:p["end_index"]]
            peak_local_idx = p["start_index"] + int(np.argmax(seg))
            peak_time = self.t[peak_local_idx]
            peak_value = self.y[peak_local_idx]
            rel_height = peak_value - self.baseline_curve[peak_local_idx]
            # 持続時間 = (終了点 - 開始点) + 1サンプル分[ms]
            # （開始点は「t1を超える直前」のサンプルを基準にしているため、
            #   実際にt1以上だった区間の長さに合わせて+1サンプル分を加える）
            duration_ms = (p["end_time_s"] - p["start_time_s"]) * 1000 + self.dt * 1000

            cell_text.append([
                f"{p['start_time_s']*1000:.2f}",
                f"{peak_time*1000:.2f}",
                f"{p['end_time_s']*1000:.2f}",
                f"{duration_ms:.2f}",
                f"{peak_value:.3f}",
                f"{rel_height:.3f}",
                f"{self.std:.4f}",
            ])

        table = self.ax_table.table(
            cellText=cell_text, colLabels=col_labels,
            loc="upper center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.2)

        first_row = offset + 1
        last_row = offset + len(shown)
        title = f"Pulses in view: {n_total}"
        if n_total > len(shown):
            title += f"  (showing {first_row}-{last_row})"
        self.ax_table.set_title(title, fontsize=9)

    # ----- events -----

    def on_check(self, label):
        if label == "Signal":
            self.show_signal = not self.show_signal
        elif label == "Baseline":
            self.show_baseline = not self.show_baseline
        elif label == "Threshold t1":
            self.show_threshold_t1 = not self.show_threshold_t1
        elif label == "Threshold t2":
            self.show_threshold_t2 = not self.show_threshold_t2
        elif label == "Pulse":
            self.show_pulse = not self.show_pulse
        elif label == "Peak value":
            self.show_peak_value = not self.show_peak_value

        view_start, view_end = self.current_view_range()
        self.redraw_plot()
        self.update_view(view_start, view_end)

    def on_prev(self, event):
        self.load_file(self.file_idx - 1)
        self.full_reload()

    def on_next(self, event):
        self.load_file(self.file_idx + 1)
        self.full_reload()

    def on_export_current(self, event):
        path = self.current_path()
        if len(self.y) == 0:
            print("[INFO] このファイルは読めていないので出力できません。")
            return

        rows = self.build_pulse_rows(path, self.y, self.pulses, self.baseline_curve, self.std)
        n_total = len(rows)
        if self.exclude_edge_pulses:
            rows = [r for r in rows if not r["edge_region"]]

        self._write_csv(rows, self.out_csv)
        print(f"[SAVED] {self.out_csv} ({len(rows)}/{n_total} pulses from this file, "
              f"k1={self.threshold_k1:.1f}, k2={self.threshold_k2:.1f}, "
              f"edge_excluded={self.exclude_edge_pulses})")

    def on_export_all(self, event):
        export_dir = EXPORT_CSV_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        total_pulses = 0

        for idx, path in enumerate(self.npy_files):
            try:
                y = np.load(path)
            except Exception as e:
                print(f"[ERROR] 読み込み失敗: {path}\n  -> {e}")
                continue

            baseline_curve, std = self._get_baseline_and_std(y, path)
            pulses, _, _ = detect_pulses_hysteresis(
                y, baseline_curve, std, dt=self.dt,
                threshold_k1=self.threshold_k1, threshold_k2=self.threshold_k2,
                min_width_ms=self.min_width_ms,
            )
            rows = self.build_pulse_rows(path, y, pulses, baseline_curve, std)
            n_before = len(rows)
            if self.exclude_edge_pulses:
                rows = [r for r in rows if not r["edge_region"]]
            # 同名ファイルの上書きを避けるため、test_data 以下のフォルダ構成を維持する。
            try:
                relative_path = path.relative_to(TEST_DATA_ROOT)
            except ValueError:
                relative_path = Path(path.name)
            out_csv = export_dir / relative_path.with_suffix(".csv")
            self._write_csv(rows, out_csv, fieldnames=PULSE_CSV_FIELDS)
            saved_count += 1
            total_pulses += len(rows)
            print(f"[SAVED] {idx+1}/{len(self.npy_files)} {out_csv}: "
                  f"{len(rows)}/{n_before} pulses (edge excluded={self.exclude_edge_pulses})")

        print(f"[SAVED] {saved_count} CSV files in {export_dir} "
              f"({total_pulses} pulses total, k1={self.threshold_k1:.1f}, "
              f"k2={self.threshold_k2:.1f})")

    def _write_csv(self, rows, out_csv, fieldnames=None):
        if fieldnames is None:
            if not rows:
                print("[INFO] 出力するパルスがありません。")
                return
            fieldnames = list(rows[0].keys())

        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def on_stop(self, event):
        try:
            plt.close(self.fig)
        except Exception as e:
            print("[ERROR]", e)

    def on_show_table(self, event):
        """ポップアウトウインドウでフルのパルス表を表示する。"""
        pulses = self._last_visible_pulses
        n_total = len(pulses)
        if n_total == 0:
            print("[INFO] No pulses in view to show")
            return

        # 新しい Figure にテーブル表示
        fig, ax = plt.subplots(figsize=(6, min(0.4 + 0.25 * min(n_total, 40), 12)))
        ax.axis('off')

        col_labels = ["Start (ms)", "Peak (ms)", "End (ms)", "Duration (ms)", "Peak (abs)", "Δ (rel)", "σ"]
        cell_text = []
        for p in pulses:
            seg = self.y[p["start_index"]:p["end_index"]]
            peak_local_idx = p["start_index"] + int(np.argmax(seg))
            peak_time = self.t[peak_local_idx]
            peak_value = self.y[peak_local_idx]
            rel_height = peak_value - self.baseline_curve[peak_local_idx]
            duration_ms = (p["end_time_s"] - p["start_time_s"]) * 1000 + self.dt * 1000
            cell_text.append([
                f"{p['start_time_s']*1000:.2f}", f"{peak_time*1000:.2f}", f"{p['end_time_s']*1000:.2f}",
                f"{duration_ms:.2f}", f"{peak_value:.3f}", f"{rel_height:.3f}", f"{self.std:.4f}",
            ])

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        ax.set_title(f"Pulses in view: {n_total}")
        fig.tight_layout()
        fig.show()

    # --- Minimal inline evaluation for visible view (no import of evaluate_detection to avoid circular import)
    def _read_csv_with_labels(self, csv_path, time_col="Time", value_col="Raw Value", assigned_col="Assigned", code_col="Code"):
        times, values, assigned, codes = [], [], [], []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for col in (time_col, value_col, assigned_col, code_col):
                if col not in fieldnames:
                    raise ValueError(f"列 '{col}' が見つかりません: {csv_path} (列: {fieldnames})")
            for row in reader:
                times.append(float(row[time_col]))
                values.append(float(row[value_col]))
                assigned.append(float(row[assigned_col]))
                codes.append(row[code_col].strip())
        return np.asarray(times), np.asarray(values), np.asarray(assigned), codes

    def _ground_truth_intervals(self, codes, assigned, baseline_code='B'):
        n = len(codes)
        intervals = []
        in_signal = False
        start = None
        for i in range(n):
            is_signal = codes[i] != baseline_code
            if is_signal and not in_signal:
                start = i
                in_signal = True
            elif not is_signal and in_signal:
                intervals.append({"start_index": start, "end_index": i, "assigned": float(np.max(assigned[start:i]))})
                in_signal = False
        if in_signal:
            intervals.append({"start_index": start, "end_index": n, "assigned": float(np.max(assigned[start:n]))})
        return intervals

    def _match_pulses(self, detected_pulses, gt_intervals, iou_threshold=0.3):
        pairs = []
        for di, dp in enumerate(detected_pulses):
            d0, d1 = dp["start_index"], dp["end_index"]
            for gi, gt in enumerate(gt_intervals):
                g0, g1 = gt["start_index"], gt["end_index"]
                inter = max(0, min(d1, g1) - max(d0, g0))
                if inter == 0:
                    continue
                union = max(d1, g1) - min(d0, g0)
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    pairs.append((iou, di, gi))
        pairs.sort(key=lambda x: x[0], reverse=True)
        matched_d, matched_g = set(), set()
        matches = []
        for iou, di, gi in pairs:
            if di in matched_d or gi in matched_g:
                continue
            matched_d.add(di)
            matched_g.add(gi)
            matches.append((di, gi, iou))
        tp = len(matches)
        fp = len(detected_pulses) - tp
        fn = len(gt_intervals) - tp
        return matches, tp, fp, fn

    def on_evaluate_view(self, event):
        """現在表示中の time 範囲で、元のラベル付きCSVがあれば簡易評価を行う。"""
        import math
        view_start, view_end = self.current_view_range()
        npy_path = Path(self.current_path())
        base = npy_path.stem
        orig_base = base[:-4] if base.endswith("_raw") else base
        seq_root = Path("sequencer/seq_data")
        matches = list(seq_root.rglob(orig_base + ".csv"))
        if not matches:
            print(f"[WARN] Labeled CSV not found for {npy_path.name}. Looked for {orig_base}.csv under {seq_root}")
            return
        csv_path = matches[0]

        try:
            times, values, assigned, codes = self._read_csv_with_labels(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read labeled CSV {csv_path}: {e}")
            return

        # map view time to sample indices (assume same sampling dt and alignment)
        i0 = int(math.floor(view_start / self.dt))
        i1 = int(math.ceil(view_end / self.dt))
        i0 = max(0, i0)
        i1 = min(len(values), i1)

        # ground truth intervals restricted to view
        gt_intervals = self._ground_truth_intervals(codes, assigned)
        gt_in_view = []
        for gt in gt_intervals:
            if gt['end_index'] <= i0 or gt['start_index'] >= i1:
                continue
            # clip to view
            g0 = max(gt['start_index'], i0)
            g1 = min(gt['end_index'], i1)
            gt_in_view.append({'start_index': g0, 'end_index': g1, 'assigned': gt['assigned']})

        # detected pulses in view
        detected_in_view = [
            p for p in self.pulses if not (p['end_time_s'] < view_start or p['start_time_s'] > view_end)
        ]

        matches, tp, fp, fn = self._match_pulses(detected_in_view, gt_in_view)

        print(f"[EVAL] View {view_start:.4f}-{view_end:.4f}s | CSV={csv_path.name} | TP={tp} FP={fp} FN={fn} (gt={len(gt_in_view)} det={len(detected_in_view)})")
        # show a small dialog figure with summary
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis('off')
        txt = f"Evaluate View:\nCSV: {csv_path.name}\nview={view_start:.4f}-{view_end:.4f}s\nTP={tp} FP={fp} FN={fn}\nGT={len(gt_in_view)} Det={len(detected_in_view)}"
        ax.text(0.01, 0.5, txt, fontsize=10, va='center')
        fig.tight_layout()
        fig.show()


# ========= main =========

if __name__ == "__main__":
    npy_files = list_npy_files(TEST_DATA_ROOT, pattern=NPY_PATTERN, recursive=True)

    print("=" * 80)
    print("[INFO] TEST_DATA_ROOT =", TEST_DATA_ROOT)
    print("[INFO] npy files found =", len(npy_files))
    for p in npy_files[:10]:
        print("   ", p)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"pulse_summary_{ts}.csv"

    picker = NpySignalPicker(
        npy_files,
        dt=DT,
        threshold_k1=THRESHOLD_K1,
        threshold_k2=THRESHOLD_K2,
        min_width_ms=MIN_WIDTH_MS,
        out_csv=out_csv,
    )
    plt.show()
