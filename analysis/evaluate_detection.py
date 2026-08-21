# -*- coding: utf-8 -*-
"""
evaluate_detection.py

目的
----
sequencer/seq_data 配下の元CSV(列: Time, Raw Value, Assigned, Code)に
含まれる正解ラベル(Code: 'B'=ベースライン、それ以外=シグナル区間、
Assigned=正解の高さ)を使って、npy_signal_picker.py と同じロジック
(ベースライン推定＋ヒステリシス2閾値検出)の性能を定量評価する。

評価する項目
-----------
1. ベースライン推定の精度
   Code=='B'(正解:シグナルなし)区間だけを取り出し、
   Raw Value と 推定baseline との残差の bias / RMSE / std を計算する。
   B区間では Raw Value ≈ 真のベースライン + ノイズ のはずなので、
   系統的なズレ(bias)や過大な残差(RMSE)がないかを見る。

2. パルス検出のPrecision / Recall / F1
   Code!='B' の連続区間を「正解パルス」とみなし、
   検出したパルス(ヒステリシス検出結果)と時間的重なり(IoU)で
   貪欲マッチングし、TP / FP / FN を数える。

3. マッチしたパルスの精度
   - タイミング誤差(開始・終了時刻の絶対誤差, ms)
   - 高さの誤差(Assigned列 vs 検出したpeak/relative_height)

使い方
------
1. 下のCONFIGを環境に合わせて変更(SEQ_DATA_ROOTは元CSVのあるフォルダ)
2. python evaluate_detection.py を実行
3. ファイルごとの評価結果 + 全体サマリーが表示され、
   詳細はCSV(evaluation_detail.csv, evaluation_summary.csv)に保存される

npy_signal_picker.py と同じフォルダに置いて実行してください
(そこから検出ロジックをそのままインポートして使う)。
"""

import csv
import numpy as np
from scipy.stats import norm
from pathlib import Path
from datetime import datetime

from npy_signal_picker import (
    compute_baseline_curve, robust_baseline_and_std, detect_pulses_hysteresis,
    DT, THRESHOLD_K1, THRESHOLD_K2, BASELINE_WINDOW, BASELINE_PERCENTILE, MIN_WIDTH_MS,
)


# ========= CONFIG =========

SEQ_DATA_ROOT = Path("sequencer/seq_data")   # 正解ラベル付きの元CSVがあるフォルダ
CSV_PATTERN = "*.csv"

TIME_COL = "Time"
VALUE_COL = "Raw Value"
ASSIGNED_COL = "Assigned"
CODE_COL = "Code"
BASELINE_CODE = "B"

IOU_THRESHOLD = 0.3   # これ以上重なっていればマッチとみなす

# 評価結果の出力先: analysis/evaluation_results/run_YYYYMMDD_HHMMSS/ に格納する
# (パラメータを変えて何度も実行する想定なので、実行ごとにフォルダを分けて履歴を残す)
EVALUATION_RESULTS_ROOT = Path("analysis/evaluation_results")
OUT_DETAIL_FILENAME = "evaluation_detail.csv"
OUT_SUMMARY_FILENAME = "evaluation_summary.csv"


# ========= Utility =========

def list_labeled_csv_files(root_folder, pattern=CSV_PATTERN, recursive=True,
                           exclude_names=("manifest.csv",)):
    root = Path(root_folder)
    if not root.is_dir():
        print(f"[WARN] フォルダが存在しません: {root}")
        return []
    files = sorted(root.rglob(pattern)) if recursive else sorted(root.glob(pattern))
    files = [f for f in files if f.name not in exclude_names]
    return files


def read_csv_with_labels(csv_path, time_col=TIME_COL, value_col=VALUE_COL,
                          assigned_col=ASSIGNED_COL, code_col=CODE_COL):
    """正解ラベル付きの元CSVを読み込む。"""
    times, values, assigned, codes = [], [], [], []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
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


def ground_truth_intervals(codes, assigned, baseline_code=BASELINE_CODE):
    """Code列から、連続した非ベースライン区間(=正解パルス)を求める。
    各区間には代表的なAssigned値(区間内の最大値)も付与する。"""
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
            intervals.append({"start_index": start, "end_index": i,
                              "assigned": float(np.max(assigned[start:i]))})
            in_signal = False

    if in_signal:
        intervals.append({"start_index": start, "end_index": n,
                          "assigned": float(np.max(assigned[start:n]))})

    return intervals


def evaluate_baseline(values, codes, baseline_curve, baseline_code=BASELINE_CODE):
    """Code=='B'区間だけでベースライン推定の残差を評価する。"""
    mask = np.array([c == baseline_code for c in codes])
    if mask.sum() == 0:
        return {"n_baseline_samples": 0, "bias": None, "rmse": None, "std": None}

    residual = values[mask] - baseline_curve[mask]
    return {
        "n_baseline_samples": int(mask.sum()),
        "bias": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "std": float(np.std(residual)),
    }


def match_pulses(detected_pulses, gt_intervals, iou_threshold=IOU_THRESHOLD):
    """検出パルスと正解区間をIoU(時間的重なり)ベースで貪欲マッチングする。"""
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


def classify_false_negatives(detected_pulses, gt_intervals, matches):
    """見逃した正解パルス(FN)を原因別に分類する。

    - "merged": 検出パルスとは重なっているが、IoUがマッチ閾値未満
                (＝近くの別パルスと融合して検出されたか、部分的にしか
                  拾えなかった可能性が高い)
    - "missed": 検出パルスと全く重なっていない
                (＝k1にすら到達しなかった、弱すぎるパルス)
    """
    matched_g = {gi for _, gi, _ in matches}
    reasons = []

    for gi, gt in enumerate(gt_intervals):
        if gi in matched_g:
            continue
        g0, g1 = gt["start_index"], gt["end_index"]
        has_overlap = False
        for dp in detected_pulses:
            d0, d1 = dp["start_index"], dp["end_index"]
            if max(0, min(d1, g1) - max(d0, g0)) > 0:
                has_overlap = True
                break
        reasons.append({
            "start_index": g0, "end_index": g1, "assigned": gt["assigned"],
            "reason": "merged_or_partial" if has_overlap else "not_detected_at_all",
        })

    return reasons


def evaluate_file(csv_path, threshold_k1=THRESHOLD_K1, threshold_k2=THRESHOLD_K2,
                  baseline_window=BASELINE_WINDOW, baseline_percentile=BASELINE_PERCENTILE,
                  min_width_ms=MIN_WIDTH_MS, dt=DT):
    times, values, assigned, codes = read_csv_with_labels(csv_path)

    baseline_curve = compute_baseline_curve(
        values, baseline_window=baseline_window, baseline_percentile=baseline_percentile
    )
    _, noise_std = robust_baseline_and_std(values)
    detected, _, _ = detect_pulses_hysteresis(
        values, baseline_curve, noise_std, dt=dt,
        threshold_k1=threshold_k1, threshold_k2=threshold_k2, min_width_ms=min_width_ms,
    )

    gt_intervals = ground_truth_intervals(codes, assigned)
    baseline_eval = evaluate_baseline(values, codes, baseline_curve)
    matches, tp, fp, fn = match_pulses(detected, gt_intervals)
    fn_reasons = classify_false_negatives(detected, gt_intervals, matches)
    n_fn_merged = sum(1 for r in fn_reasons if r["reason"] == "merged_or_partial")
    n_fn_missed = sum(1 for r in fn_reasons if r["reason"] == "not_detected_at_all")

    # ベースラインの理論的なバイアス補正量（percentile法による既知の系統誤差）
    # 標準正規分布のp点(0<p<1)におけるz値: norm.ppf(p/100)
    z = norm.ppf(baseline_percentile / 100.0)
    expected_bias = -z * noise_std   # baseline は true baseline より -z*std だけ低くなる想定

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
         if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
         else float("nan"))

    start_errs_ms, end_errs_ms, height_errs = [], [], []
    for di, gi, iou in matches:
        dp = detected[di]
        gt = gt_intervals[gi]
        start_errs_ms.append(abs(dp["start_time_s"] - gt["start_index"] * dt) * 1000)
        end_errs_ms.append(abs(dp["end_time_s"] - gt["end_index"] * dt) * 1000)
        height_errs.append(abs(dp["peak"] - gt["assigned"]))

    result = {
        "file": str(csv_path),
        "n_samples": len(values),
        "n_gt_pulses": len(gt_intervals),
        "n_detected_pulses": len(detected),
        "n_fn_merged_or_partial": n_fn_merged,
        "n_fn_not_detected_at_all": n_fn_missed,
        "noise_std": noise_std,
        "expected_baseline_bias": expected_bias,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "baseline_bias": baseline_eval["bias"],
        "baseline_rmse": baseline_eval["rmse"],
        "start_time_mae_ms": float(np.mean(start_errs_ms)) if start_errs_ms else None,
        "end_time_mae_ms": float(np.mean(end_errs_ms)) if end_errs_ms else None,
        "peak_height_mae": float(np.mean(height_errs)) if height_errs else None,
    }
    return result


def main():
    files = list_labeled_csv_files(SEQ_DATA_ROOT)
    print("=" * 80)
    print("[INFO] SEQ_DATA_ROOT =", SEQ_DATA_ROOT)
    print("[INFO] labeled csv files found =", len(files))

    if not files:
        print("[WARN] 評価対象のCSVが見つかりませんでした。SEQ_DATA_ROOTを確認してください。")
        return

    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVALUATION_RESULTS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_detail_csv = out_dir / OUT_DETAIL_FILENAME
    out_summary_csv = out_dir / OUT_SUMMARY_FILENAME
    print("[INFO] output dir =", out_dir)

    results = []
    for idx, path in enumerate(files):
        try:
            r = evaluate_file(path)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            continue
        results.append(r)
        print(f"[{idx+1}/{len(files)}] {path.name}: "
              f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
              f"(TP={r['tp']} FP={r['fp']} FN={r['fn']}: "
              f"merged={r['n_fn_merged_or_partial']} missed={r['n_fn_not_detected_at_all']}) "
              f"baseline_bias={r['baseline_bias']:.4f}(理論値≈{r['expected_baseline_bias']:.4f})")

    if not results:
        print("[WARN] 評価できたファイルがありませんでした。")
        return

    # 詳細CSV
    fieldnames = list(results[0].keys())
    with open(out_detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[SAVED] {out_detail_csv}")

    # 全体サマリー
    def _nanmean(key):
        vals = [r[key] for r in results if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_files": len(results),
        "total_tp": sum(r["tp"] for r in results),
        "total_fp": sum(r["fp"] for r in results),
        "total_fn": sum(r["fn"] for r in results),
        "total_fn_merged_or_partial": sum(r["n_fn_merged_or_partial"] for r in results),
        "total_fn_not_detected_at_all": sum(r["n_fn_not_detected_at_all"] for r in results),
        "mean_precision": _nanmean("precision"),
        "mean_recall": _nanmean("recall"),
        "mean_f1": _nanmean("f1"),
        "mean_baseline_bias": _nanmean("baseline_bias"),
        "mean_expected_baseline_bias": _nanmean("expected_baseline_bias"),
        "mean_baseline_rmse": _nanmean("baseline_rmse"),
        "mean_start_time_mae_ms": _nanmean("start_time_mae_ms"),
        "mean_end_time_mae_ms": _nanmean("end_time_mae_ms"),
        "mean_peak_height_mae": _nanmean("peak_height_mae"),
    }

    # マイクロ平均のPrecision/Recall/F1(全体のTP/FP/FNから計算)
    tp, fp, fn = summary["total_tp"], summary["total_fp"], summary["total_fn"]
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) > 0 else float("nan"))
    summary["micro_precision"] = micro_p
    summary["micro_recall"] = micro_r
    summary["micro_f1"] = micro_f1

    with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("=" * 80)
    print("[SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[SAVED] {out_summary_csv}")
    print(f"[INFO] このパラメータでの結果一式: {out_dir}")


if __name__ == "__main__":
    main()
