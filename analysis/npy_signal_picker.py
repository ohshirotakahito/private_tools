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
7. Export (This File) / Export (All Files) でCSVに書き出す

必要パッケージ: numpy, scipy, matplotlib(>=3.5, RangeSliderを使用)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
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

# ベースライン推定窓の半分は原理的に信頼できない端の領域になるため、
# その範囲を可視化・デフォルトでは検出結果からも除外する。
EXCLUDE_EDGE_PULSES = True

MAX_TABLE_ROWS = 20        # 右側の一覧表に表示するパルス数の上限（表示範囲内のもの）

OUT_CSV_DEFAULT = "pulse_summary.csv"


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
        return []

    t1_curve = baseline_curve + threshold_k1 * noise
    t2_curve = baseline_curve + threshold_k2 * noise

    pulses = []
    state = "IDLE"       # IDLE -> CANDIDATE -> CONFIRMED -> IDLE
    cand_start = None

    def _finalize(start, end):
        width_ms = (end - start) * dt * 1000
        if width_ms < min_width_ms:
            return None
        segment = y[start:end]
        local_baseline = float(baseline_curve[start])
        return {
            "start_index": start,
            "end_index": end,
            "start_time_s": start * dt,
            "end_time_s": end * dt,
            "width_ms": width_ms,
            "peak": float(np.max(segment)),
            "mean": float(np.mean(segment)),
            "area": float(np.sum(segment - local_baseline) * dt),
            "baseline": local_baseline,
            "threshold_t1": float(t1_curve[start]),
            "threshold_t2": float(t2_curve[start]),
        }

    for i in range(n):
        yi = y[i]
        t1 = t1_curve[i]
        t2 = t2_curve[i]

        if state == "IDLE":
            if yi > t1:
                state = "CANDIDATE"
                # 開始点は「t1を超えた最初のサンプル」ではなく、
                # 「t1を超える直前(まだt1以下)の1つ前のサンプル」とする。
                # 立ち上がりが急峻なデータでは、超えた最初のサンプルが
                # 既にピーク付近まで跳ね上がっていることが多く、
                # 開始点がt1のラインよりずっと高く見えてしまうため。
                cand_start = i - 1 if i > 0 else i

        elif state == "CANDIDATE":
            if yi > t2:
                state = "CONFIRMED"          # ② 確定
            elif yi <= t1:
                state = "IDLE"               # t2に届かず終わった候補 -> 破棄
                cand_start = None

        elif state == "CONFIRMED":
            if yi <= t1:                     # ④ 終了
                p = _finalize(cand_start, i)
                if p is not None:
                    pulses.append(p)
                state = "IDLE"
                cand_start = None
            # yi<=t2 (③)でも yi>t1 である限りは CONFIRMED のまま継続

    # ファイル終端で CONFIRMED のまま終わった場合、そこまでを1パルスとして確定する
    if state == "CONFIRMED" and cand_start is not None:
        p = _finalize(cand_start, n)
        if p is not None:
            pulses.append(p)

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

        self.fig, self.ax = plt.subplots(figsize=(17, 10.5))
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

        # 表示範囲スライダー（ファイルごとに作り直す）
        self.ax_slider = plt.axes([0.18, 0.28, 0.65, 0.03])
        self.range_slider = None

        # 閾値スライダー（k1, k2 独立、ファイルをまたいで値を維持する）
        ax_k1 = plt.axes([0.18, 0.23, 0.65, 0.03])
        self.k1_slider = Slider(
            ax_k1, "Threshold k1 (start/end)",
            valmin=THRESHOLD_K_MIN, valmax=THRESHOLD_K_MAX,
            valinit=self.threshold_k1, valstep=0.1,
        )
        self.k1_slider.on_changed(self.on_k1_change)

        ax_k2 = plt.axes([0.18, 0.18, 0.65, 0.03])
        self.k2_slider = Slider(
            ax_k2, "Threshold k2 (confirm)",
            valmin=THRESHOLD_K_MIN, valmax=THRESHOLD_K_MAX,
            valinit=self.threshold_k2, valstep=0.1,
        )
        self.k2_slider.on_changed(self.on_k2_change)

        # ベースライン推定パラメータ（窓幅・パーセンタイル）
        init_window_ms = self.baseline_window * self.dt * 1000
        ax_bw = plt.axes([0.18, 0.13, 0.65, 0.03])
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
        self.btn_exp_all = Button(ax_exp_all, "Export (All Files)")
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
        self.ax_slider.clear()
        total_duration = max(total_duration, 1e-6)
        init_width = min(self.initial_view_width, total_duration)

        self.range_slider = RangeSlider(
            self.ax_slider, "View range (s)",
            valmin=0.0, valmax=total_duration,
            valinit=(0.0, init_width),
        )
        self.range_slider.on_changed(self.on_view_change)

    def on_view_change(self, val):
        view_start, view_end = val
        self.update_view(view_start, view_end)

    def on_reset_view(self, event):
        if self.range_slider is None:
            return
        total_duration = self.range_slider.valmax
        self.range_slider.set_val((0.0, total_duration))

    def current_view_range(self):
        if self.range_slider is not None:
            return self.range_slider.val
        n = len(self.y)
        return 0.0, n * self.dt

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
        baseline_curve = compute_baseline_curve(
            y, baseline_window=self.baseline_window,
            baseline_percentile=self.baseline_percentile,
        )
        _, std = robust_baseline_and_std(y)
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

        view_start, view_end = self.range_slider.val
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
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        title = f"Pulses in view: {n_total}"
        if n_total > MAX_TABLE_ROWS:
            title += f"  (showing {offset+1}-{offset+len(shown)})"
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
        all_rows = []
        for idx, path in enumerate(self.npy_files):
            try:
                y = np.load(path)
            except Exception as e:
                print(f"[ERROR] 読み込み失敗: {path}\n  -> {e}")
                continue

            baseline_curve = compute_baseline_curve(
                y, baseline_window=self.baseline_window,
                baseline_percentile=self.baseline_percentile,
            )
            _, std = robust_baseline_and_std(y)
            pulses, _, _ = detect_pulses_hysteresis(
                y, baseline_curve, std, dt=self.dt,
                threshold_k1=self.threshold_k1, threshold_k2=self.threshold_k2,
                min_width_ms=self.min_width_ms,
            )
            rows = self.build_pulse_rows(path, y, pulses, baseline_curve, std)
            n_before = len(rows)
            if self.exclude_edge_pulses:
                rows = [r for r in rows if not r["edge_region"]]
            all_rows.extend(rows)
            print(f"[INFO] {idx+1}/{len(self.npy_files)} {path.name}: "
                  f"{len(rows)}/{n_before} pulses (edge excluded={self.exclude_edge_pulses})")

        self._write_csv(all_rows, self.out_csv)
        print(f"[SAVED] {self.out_csv} ({len(all_rows)} pulses total from {len(self.npy_files)} files, "
              f"k1={self.threshold_k1:.1f}, k2={self.threshold_k2:.1f})")

    def _write_csv(self, rows, out_csv):
        if not rows:
            print("[INFO] 出力するパルスがありません。")
            return

        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def on_stop(self, event):
        try:
            plt.close(self.fig)
        except Exception as e:
            print("[ERROR]", e)


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
