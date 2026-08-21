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

パルス数が多いファイルでも見やすいよう、下部の RangeSlider で
表示するTime範囲をドラッグして拡大/縮小できるようにしている。
パルスの注釈(網掛け・ピークマーカー・数値ラベル)は表示範囲内の
ものだけを描画するので、ズームインするほど見やすくなる。

最終的に「信号の位置(時間)・高さ・相対的な高さ(=高さ-ベースライン)・
ノイズレベル(σ)」をCSVに書き出せるようにしている(Exportボタン)。
CSVへの出力は表示範囲に関わらず、検出された全パルスが対象。

使い方
------
1. 下の CONFIG セクションの TEST_DATA_ROOT を自分の環境に合わせて変更
2. python npy_signal_picker.py を実行
3. Prev/Next File で npy ファイルを切り替えながらパルス検出結果を確認
4. 下部の RangeSlider をドラッグして見たい時間範囲を拡大/縮小
5. Reset View で全範囲表示に戻す
6. Export (This File) / Export (All Files) でCSVに書き出す

必要パッケージ: numpy, scipy, matplotlib(>=3.5, RangeSliderを使用)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider
from scipy.signal import savgol_filter
from pathlib import Path
from datetime import datetime


# ========= CONFIG =========

TEST_DATA_ROOT = Path("analysis/test_data")   # 環境に合わせて変更
NPY_PATTERN = "*_raw.npy"

DT = 0.0001                # サンプリング間隔[s]（extract_raw_value.py と同じ想定）
THRESHOLD_K = 6.0          # baseline + K * noise を閾値とする
MIN_WIDTH_MS = 0.2         # これより短いパルスは無視
BASELINE_WINDOW = 501      # ベースライン移動平均の窓幅（奇数）
SG_WINDOW = 51             # Savitzky-Golayフィルタの窓幅（奇数）
SG_POLY = 3

INITIAL_VIEW_WIDTH_S = 0.05  # 起動時・ファイル切替時の初期表示幅[s]

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
                                sg_window=SG_WINDOW, sg_poly=SG_POLY):
    """描画用のベースライン曲線・閾値曲線・ノイズレベルを計算する。"""
    if len(y) >= sg_window and sg_window % 2 == 1:
        y_f = savgol_filter(y, window_length=sg_window, polyorder=sg_poly)
    else:
        y_f = y.copy()

    w = baseline_window
    if w < 3:
        baseline = np.zeros_like(y_f) + (np.mean(y_f) if len(y_f) else 0.0)
    else:
        if w % 2 == 0:
            w += 1
        half = w // 2
        padded = np.pad(y_f, (half, half), mode="edge")
        baseline = np.convolve(padded, np.ones(w) / w, mode="valid")

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
                 initial_view_width=INITIAL_VIEW_WIDTH_S):
        if not npy_files:
            raise RuntimeError("npyファイルが1つも見つかりませんでした。")

        self.npy_files = npy_files
        self.dt = float(dt)
        self.threshold_k = float(threshold_k)
        self.min_width_ms = float(min_width_ms)
        self.out_csv = out_csv
        self.initial_view_width = float(initial_view_width)

        self.file_idx = 0
        self.data = None
        self.pulses = []
        self.t = np.array([])
        self.y = np.array([])
        self.baseline_curve = np.array([])
        self.pulse_artists = []

        self.show_signal = True
        self.show_baseline = True
        self.show_threshold = True
        self.show_pulse = True
        self.show_peak_value = True

        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.16, bottom=0.30)

        ax_check = plt.axes([0.02, 0.55, 0.12, 0.22])
        self.check = CheckButtons(
            ax_check,
            ["Signal", "Baseline", "Threshold", "Pulse", "Peak value"],
            [self.show_signal, self.show_baseline, self.show_threshold,
             self.show_pulse, self.show_peak_value],
        )
        self.check.on_clicked(self.on_check)

        # 表示範囲スライダー（ファイルごとに作り直す）
        self.ax_slider = plt.axes([0.18, 0.17, 0.65, 0.04])
        self.range_slider = None

        # ボタン行
        ax_prev = plt.axes([0.14, 0.05, 0.10, 0.07])
        ax_next = plt.axes([0.25, 0.05, 0.10, 0.07])
        ax_reset = plt.axes([0.36, 0.05, 0.10, 0.07])
        ax_exp_cur = plt.axes([0.47, 0.05, 0.16, 0.07])
        ax_exp_all = plt.axes([0.65, 0.05, 0.16, 0.07])
        ax_stop = plt.axes([0.83, 0.05, 0.10, 0.07])

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
        self.draw()

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
        rows = []
        for i, p in enumerate(pulses):
            local_baseline = baseline_curve[p["start_index"]]
            relative_height = p["peak"] - local_baseline
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
                "threshold": p["threshold"],
            })
        return rows

    # ----- slider -----

    def create_slider(self, total_duration):
        self.ax_slider.clear()
        total_duration = max(total_duration, 1e-6)
        init_width = min(self.initial_view_width, total_duration)

        self.range_slider = RangeSlider(
            self.ax_slider, "View range (s)",
            valmin=0.0, valmax=total_duration,
            valinit=(0.0, init_width),
        )
        self.range_slider.on_changed(self.on_slider_change)

    def on_slider_change(self, val):
        view_start, view_end = val
        self.update_view(view_start, view_end)

    def on_reset_view(self, event):
        if self.range_slider is None:
            return
        total_duration = self.range_slider.valmax
        self.range_slider.set_val((0.0, total_duration))

    # ----- drawing -----

    def draw(self):
        """ファイル切替時などに呼ぶフル再描画。信号線・パルス検出をやり直す。"""
        self.ax.clear()
        self.pulse_artists = []

        path = self.current_path()
        y = self.data
        n = len(y)

        if n == 0:
            self.ax.text(0.5, 0.5, f"Failed to read:\n{path}",
                         ha="center", va="center", transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
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
        self.pulses = pulses
        self.std = std

        if self.show_signal:
            self.ax.plot(t, y, label="Signal", linewidth=1.0)
        if self.show_baseline:
            self.ax.plot(t, baseline_curve, label="Baseline", linewidth=2.0)
        if self.show_threshold:
            self.ax.plot(t, threshold_curve, label=f"Threshold (+{self.threshold_k}σ)", linewidth=2.0)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Signal Amplitude")
        self.ax.grid(True, alpha=0.2)

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right")

        self.ax.text(0.01, 0.99, str(path), transform=self.ax.transAxes,
                     fontsize=8, va="top", bbox=dict(boxstyle="round", alpha=0.2))

        total_duration = n * self.dt
        self.create_slider(total_duration)

        view_start, view_end = self.range_slider.val
        self.update_view(view_start, view_end)

    def update_view(self, view_start, view_end):
        """スライダー変更時に呼ぶ軽量更新。パルス注釈と表示範囲だけ更新する。"""
        for artist in self.pulse_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.pulse_artists = []

        visible_pulses = [
            p for p in self.pulses
            if not (p["end_time_s"] < view_start or p["start_time_s"] > view_end)
        ]

        if self.show_pulse:
            for i, p in enumerate(visible_pulses):
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
            f"std≈{self.std:.4g} | view=[{view_start:.4f}, {view_end:.4f}]s"
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
        self.draw()

    def on_prev(self, event):
        self.load_file(self.file_idx - 1)
        self.draw()

    def on_next(self, event):
        self.load_file(self.file_idx + 1)
        self.draw()

    def on_export_current(self, event):
        path = self.current_path()
        y = self.data
        if len(y) == 0:
            print("[INFO] このファイルは読めていないので出力できません。")
            return

        baseline_curve, _, std = compute_baseline_threshold(y, threshold_k=self.threshold_k)
        rows = self.build_pulse_rows(path, y, self.pulses, baseline_curve, std)
        self._write_csv(rows, self.out_csv)
        print(f"[SAVED] {self.out_csv} ({len(rows)} pulses from this file)")

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
            all_rows.extend(self.build_pulse_rows(path, y, pulses, baseline_curve, std))
            print(f"[INFO] {idx+1}/{len(self.npy_files)} {path.name}: {len(pulses)} pulses")

        self._write_csv(all_rows, self.out_csv)
        print(f"[SAVED] {self.out_csv} ({len(all_rows)} pulses total from {len(self.npy_files)} files)")

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
