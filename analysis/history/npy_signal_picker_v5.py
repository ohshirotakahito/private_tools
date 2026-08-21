# -*- coding: utf-8 -*-
"""
npy_signal_picker.py

目的
----
extract_raw_value.py で test_data 配下に作った "*_raw.npy"
(Time,Raw Value のうち Raw Value のみを保存した信号データ) を
リスト化して1件ずつ読み込み、パルス検出(信号ピックアップ)結果を
可視化して確認するビューア。

view_tdms_signal.py の detect_pulses / ベースライン計算ロジックを
そのまま流用しているので、ここで閾値やウィンドウ幅を調整して
アルゴリズムを詰めれば、そのまま実TDMSファイルにも使い回せる。

ノイズレベル(std)は単純な np.std() だとパルス自体の分散を
「ノイズ」として拾ってしまい過大評価されるため、MADベースの
反復的なσクリッピングで頑健に推定している(robust_baseline_and_std)。

2種類のスライダーを用意:
  - View range スライダー: 表示するTime範囲をドラッグで拡大/縮小
  - Threshold スライダー: パルス検出の閾値(baseline + kσ の k)を調整
    動かすたびに現在のファイルに対してパルス検出をやり直すが、
    表示中のズーム範囲は維持したまま更新する。

最終的に「信号の位置(時間)・高さ・相対的な高さ(=高さ-ベースライン)・
ノイズレベル(σ)」をCSVに書き出せるようにしている(Exportボタン)。
CSVへの出力は表示範囲に関わらず、現在の threshold_k で検出された
全パルスが対象。

使い方
------
1. 下の CONFIG セクションの TEST_DATA_ROOT を自分の環境に合わせて変更
2. python npy_signal_picker.py を実行
3. Prev/Next File で npy ファイルを切り替えながらパルス検出結果を確認
4. View range スライダーで見たい時間範囲を拡大/縮小
5. Threshold スライダーで検出感度(kσ)を調整
6. Reset View で全範囲表示に戻す
7. Export (This File) / Export (All Files) でCSVに書き出す

必要パッケージ: numpy, scipy, matplotlib(>=3.5, RangeSliderを使用)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
from pathlib import Path
from datetime import datetime


# ========= CONFIG =========

TEST_DATA_ROOT = Path("analysis/test_data")   # 環境に合わせて変更
NPY_PATTERN = "*_raw.npy"

DT = 0.0001                # サンプリング間隔[s]（extract_raw_value.py と同じ想定）
THRESHOLD_K = 6.0          # baseline + K * noise を閾値とする（初期値）
THRESHOLD_K_MIN = 1.0
THRESHOLD_K_MAX = 15.0
MIN_WIDTH_MS = 0.2         # これより短いパルスは無視
BASELINE_WINDOW = 501      # ベースライン推定の窓幅（奇数, サンプル数）
BASELINE_PERCENTILE = 10   # 窓内の下位何%点をベースラインとするか（パルスは正方向のみなので低め推奨。
                            # パルスの時間占有率(duty比)が高いデータほど、さらに下げるか
                            # BASELINE_WINDOWを広げる必要がある）
SG_WINDOW = 51             # Savitzky-Golayフィルタの窓幅（奇数）
SG_POLY = 3

INITIAL_VIEW_WIDTH_S = 0.05  # 起動時・ファイル切替時の初期表示幅[s]

# ベースライン推定窓の半分は原理的に信頼できない端の領域になるため、
# その範囲を可視化・デフォルトでは検出結果からも除外する。
EXCLUDE_EDGE_PULSES = True

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


def detect_pulses(y, dt=DT, threshold_k=THRESHOLD_K, min_width_ms=MIN_WIDTH_MS):
    """閾値超え区間をパルスとして検出する。ノイズレベルは頑健推定を使用。"""
    y = np.asarray(y)
    if len(y) == 0:
        return [], 0.0, 0.0

    baseline, noise = robust_baseline_and_std(y)
    threshold = baseline + threshold_k * noise

    above = y > threshold
    pulses = []
    in_pulse = False
    start = None

    def _finalize(start, end):
        width_ms = (end - start) * dt * 1000
        if width_ms < min_width_ms:
            return None
        segment = y[start:end]
        return {
            "start_index": start,
            "end_index": end,
            "start_time_s": start * dt,
            "end_time_s": end * dt,
            "width_ms": width_ms,
            "peak": float(np.max(segment)),
            "mean": float(np.mean(segment)),
            "area": float(np.sum(segment - baseline) * dt),
            "baseline": float(baseline),
            "threshold": float(threshold),
        }

    for i, flag in enumerate(above):
        if flag and not in_pulse:
            start = i
            in_pulse = True
        elif not flag and in_pulse:
            p = _finalize(start, i)
            if p is not None:
                pulses.append(p)
            in_pulse = False

    if in_pulse:
        p = _finalize(start, len(y))
        if p is not None:
            pulses.append(p)

    return pulses, baseline, threshold


def compute_baseline_threshold(y, threshold_k=THRESHOLD_K,
                                baseline_window=BASELINE_WINDOW,
                                baseline_percentile=BASELINE_PERCENTILE,
                                sg_window=SG_WINDOW, sg_poly=SG_POLY):
    """描画用のベースライン曲線・閾値曲線・ノイズレベルを計算する。

    ベースラインは単純な移動平均ではなく、窓内の下位パーセンタイル
    (percentile_filter)で推定する。パルスは常に正方向にしか出ないため、
    低いパーセンタイルを取ればパルスの影響をほぼ受けずに
    本来の静止レベル(≒ノイズだけの区間の中心)を追跡できる。
    """
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
    elif len(y_f) > 0:
        baseline = np.full_like(y_f, np.percentile(y_f, baseline_percentile))
    else:
        baseline = np.array([])

    if len(y):
        _, std = robust_baseline_and_std(y)
    else:
        std = 0.0
    threshold = baseline + threshold_k * std
    return baseline, threshold, std


# ========= Core UI =========

class NpySignalPicker:
    def __init__(self, npy_files, dt=DT, threshold_k=THRESHOLD_K,
                 min_width_ms=MIN_WIDTH_MS, out_csv=OUT_CSV_DEFAULT,
                 initial_view_width=INITIAL_VIEW_WIDTH_S,
                 exclude_edge_pulses=EXCLUDE_EDGE_PULSES):
        if not npy_files:
            raise RuntimeError("npyファイルが1つも見つかりませんでした。")

        self.npy_files = npy_files
        self.dt = float(dt)
        self.threshold_k = float(threshold_k)
        self.min_width_ms = float(min_width_ms)
        self.out_csv = out_csv
        self.initial_view_width = float(initial_view_width)
        self.exclude_edge_pulses = bool(exclude_edge_pulses)
        # ベースライン推定窓(BASELINE_WINDOW)の半分は、窓の片側がデータの
        # 外に出るため原理的に信頼性が低い。この時間幅を「端の不確実領域」
        # として可視化・デフォルトでは検出結果から除外する。
        self.edge_margin_s = (BASELINE_WINDOW // 2) * self.dt

        self.file_idx = 0
        self.data = None
        self.pulses = []
        self.t = np.array([])
        self.y = np.array([])
        self.baseline_curve = np.array([])
        self.threshold_curve = np.array([])
        self.std = 0.0
        self.pulse_artists = []
        self.line_artists = []

        self.show_signal = True
        self.show_baseline = True
        self.show_threshold = True
        self.show_pulse = True
        self.show_peak_value = True

        self.fig, self.ax = plt.subplots(figsize=(14, 8.5))
        plt.subplots_adjust(left=0.16, bottom=0.34)

        ax_check = plt.axes([0.02, 0.55, 0.12, 0.22])
        self.check = CheckButtons(
            ax_check,
            ["Signal", "Baseline", "Threshold", "Pulse", "Peak value"],
            [self.show_signal, self.show_baseline, self.show_threshold,
             self.show_pulse, self.show_peak_value],
        )
        self.check.on_clicked(self.on_check)

        # 表示範囲スライダー（ファイルごとに作り直す）
        self.ax_slider = plt.axes([0.18, 0.21, 0.65, 0.035])
        self.range_slider = None

        # 閾値スライダー（ファイルをまたいで値を維持する）
        ax_thresh = plt.axes([0.18, 0.15, 0.65, 0.035])
        self.thresh_slider = Slider(
            ax_thresh, "Threshold (kσ)",
            valmin=THRESHOLD_K_MIN, valmax=THRESHOLD_K_MAX,
            valinit=self.threshold_k, valstep=0.1,
        )
        self.thresh_slider.on_changed(self.on_threshold_change)

        # ボタン行
        ax_prev = plt.axes([0.14, 0.04, 0.10, 0.07])
        ax_next = plt.axes([0.25, 0.04, 0.10, 0.07])
        ax_reset = plt.axes([0.36, 0.04, 0.10, 0.07])
        ax_exp_cur = plt.axes([0.47, 0.04, 0.16, 0.07])
        ax_exp_all = plt.axes([0.65, 0.04, 0.16, 0.07])
        ax_stop = plt.axes([0.83, 0.04, 0.10, 0.07])

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
        """CSV出力用の行リストを作る（位置・高さ・相対的な高さ・ノイズレベル）。"""
        duration = len(y) * self.dt
        margin = self.edge_margin_s
        rows = []
        for i, p in enumerate(pulses):
            local_baseline = baseline_curve[p["start_index"]]
            relative_height = p["peak"] - local_baseline
            is_edge = (p["start_time_s"] < margin) or (p["end_time_s"] > duration - margin)
            rows.append({
                "file": str(path),
                "pulse_index": i,
                "start_time_s": p["start_time_s"],
                "end_time_s": p["end_time_s"],
                "width_ms": p["width_ms"],
                "peak": p["peak"],
                "baseline": local_baseline,
                "relative_height": relative_height,
                "noise_std": std,
                "threshold_k": self.threshold_k,
                "threshold": p["threshold"],
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

    # ----- threshold slider -----

    def on_threshold_change(self, val):
        self.threshold_k = float(val)
        if len(self.data) == 0:
            return

        # 表示範囲(ズーム)は維持したまま、検出だけやり直す
        view_start, view_end = self.current_view_range()
        self.recompute_detection()
        self.redraw_plot()
        self.update_view(view_start, view_end)

    # ----- detection / drawing -----

    def recompute_detection(self):
        """現在のファイル(self.data)に対して、現在の threshold_k で
        ベースライン曲線・閾値曲線・パルスを計算し直す。"""
        y = self.data
        n = len(y)

        if n == 0:
            self.t = np.array([])
            self.y = np.array([])
            self.baseline_curve = np.array([])
            self.threshold_curve = np.array([])
            self.pulses = []
            self.std = 0.0
            return

        t = np.arange(n) * self.dt
        baseline_curve, threshold_curve, std = compute_baseline_threshold(
            y, threshold_k=self.threshold_k
        )
        pulses, _, _ = detect_pulses(
            y, dt=self.dt, threshold_k=self.threshold_k, min_width_ms=self.min_width_ms
        )

        self.t = t
        self.y = y
        self.baseline_curve = baseline_curve
        self.threshold_curve = threshold_curve
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
        if self.show_threshold:
            self.ax.plot(self.t, self.threshold_curve,
                         label=f"Threshold (+{self.threshold_k:.1f}σ)", linewidth=2.0)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Signal Amplitude")
        self.ax.grid(True, alpha=0.2)

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right")

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
                self.ax.legend(loc="upper right")

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
            for p in visible_pulses:
                span = self.ax.axvspan(p["start_time_s"], p["end_time_s"], alpha=0.25)
                self.pulse_artists.append(span)

                seg = self.y[p["start_index"]:p["end_index"]]
                peak_local_idx = p["start_index"] + int(np.argmax(seg))
                peak_time = self.t[peak_local_idx]
                peak_value = self.y[peak_local_idx]
                rel_height = peak_value - self.baseline_curve[peak_local_idx]

                marker, = self.ax.plot(
                    peak_time, peak_value, marker="o", markersize=8,
                    linestyle="None", color="red",
                )
                self.pulse_artists.append(marker)

                if self.show_peak_value:
                    txt = self.ax.text(
                        peak_time, peak_value,
                        f"h={peak_value:.3f}\nΔ={rel_height:.3f}",
                        fontsize=8, ha="center", va="bottom",
                    )
                    self.pulse_artists.append(txt)

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
        self.ax.set_title(
            f"File {self.file_idx+1}/{len(self.npy_files)} | {path.name} | "
            f"Pulses in view: {len(visible_pulses)}/{len(self.pulses)} | "
            f"k={self.threshold_k:.1f} std≈{self.std:.4g} | "
            f"view=[{view_start:.4f}, {view_end:.4f}]s"
        )

        self.fig.canvas.draw_idle()

    # ----- events -----

    def on_check(self, label):
        if label == "Signal":
            self.show_signal = not self.show_signal
        elif label == "Baseline":
            self.show_baseline = not self.show_baseline
        elif label == "Threshold":
            self.show_threshold = not self.show_threshold
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
              f"k={self.threshold_k:.1f}, edge_excluded={self.exclude_edge_pulses})")

    def on_export_all(self, event):
        all_rows = []
        for idx, path in enumerate(self.npy_files):
            try:
                y = np.load(path)
            except Exception as e:
                print(f"[ERROR] 読み込み失敗: {path}\n  -> {e}")
                continue

            baseline_curve, _, std = compute_baseline_threshold(y, threshold_k=self.threshold_k)
            pulses, _, _ = detect_pulses(
                y, dt=self.dt, threshold_k=self.threshold_k, min_width_ms=self.min_width_ms
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
              f"k={self.threshold_k:.1f})")

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
        threshold_k=THRESHOLD_K,
        min_width_ms=MIN_WIDTH_MS,
        out_csv=out_csv,
    )
    plt.show()
