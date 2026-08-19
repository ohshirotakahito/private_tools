# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:30:06 2025

@author: ohshi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time  # 時間計測用モジュールのインポート
from datetime import datetime  # ファイル名に時刻を付けるため

# ============================================================
# パラメータ一覧（ここを変えると結果が変わります）
# ============================================================

# グリッドサイズ：盤面の一辺のマス数。大きいほど密で複雑な模様になるが重くなる
GRID_SIZE = 254

# アニメーションのフレーム数（世代数）。大きいほど長い動画になるが時間がかかる
FRAMES = 300

# 1フレームの表示間隔（ミリ秒）。GIF内の1コマあたりの待ち時間。小さいほど速い動きに見える
INTERVAL_MS = 100

# GIF保存時のフレームレート（fps）。値を大きくすると再生が速くなる
FPS = 10

# 初期グリッドの各状態の初期割合 [死んでいる, 弱い生存, 強い生存]（合計が1になるようにする）
# 例：強い生存の割合を増やすと最初から活発な盤面になる
INIT_PROBS = [0.1, 0.3, 0.6]

# 誕生条件：死んでいるセルの周囲に何個生きたセルがあれば誕生するか（標準のライフゲームは3）
BIRTH_NEIGHBORS = 3

# 生存条件の範囲：この範囲外の隣接数だと弱い/強い状態が変化する（標準は2〜3）
SURVIVE_MIN = 2
SURVIVE_MAX = 3

# 弱い生存状態が強い生存状態へ進化する確率（0〜1）。大きいほど強い状態が増えやすい
EVOLVE_PROB = 1 - 0.998  # 0.002 = 0.2%の確率で進化

# 出力先（このスクリプトが置かれているフォルダ基準にする）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gif")

# 実行するたびに日時付きのファイル名にして、以前の結果を上書きしないようにする
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"game_of_life_{TIMESTAMP}.gif")


# ライフゲームの初期化
def initialize_grid(size):
    # 0: 死んでいる, 1: 弱い生存, 2: 強い生存
    return np.random.choice([0, 1, 2], size=(size, size), p=INIT_PROBS)


# 隣接セルを数える（ベクトル化版：盤面の端は折り返さない = 0埋め）
def count_neighbors_grid(grid):
    alive = (grid > 0).astype(np.int32)
    padded = np.pad(alive, 1, mode="constant")
    h, w = grid.shape
    neighbors = np.zeros((h, w), dtype=np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbors += padded[1 + dx : 1 + dx + h, 1 + dy : 1 + dy + w]
    return neighbors


# ゲームの進行（ベクトル化版）
def update_grid(grid):
    neighbors = count_neighbors_grid(grid)
    new_grid = grid.copy()

    weak = grid == 1
    strong = grid == 2
    dead = grid == 0
    out_of_range = (neighbors < SURVIVE_MIN) | (neighbors > SURVIVE_MAX)

    # 弱い生存状態：過疎・過密で死亡。生存継続時にごく低確率で強い状態へ進化
    new_grid[weak & out_of_range] = 0
    evolve_roll = np.random.random(grid.shape)
    evolve_mask = weak & ~out_of_range & (evolve_roll < EVOLVE_PROB)
    new_grid[evolve_mask] = 2

    # 強い生存状態：過疎・過密でも死なずに弱い状態に戻るだけ
    new_grid[strong & out_of_range] = 1

    # 死んでいるセル：隣接がBIRTH_NEIGHBORS個で誕生（弱い状態として）
    new_grid[dead & (neighbors == BIRTH_NEIGHBORS)] = 1

    return new_grid


# 可視化のためのアニメーション
def animate(frame, img, grid, ax):
    new_grid = update_grid(grid)
    img.set_data(new_grid)

    # セルの状態ごとのカウント
    dead_cells = np.sum(new_grid == 0)
    weak_cells = np.sum(new_grid == 1)
    strong_cells = np.sum(new_grid == 2)

    ax.set_title(f"Frame: {frame}  Dead: {dead_cells}  Weak: {weak_cells}  Strong: {strong_cells}")

    grid[:] = new_grid
    return (img,)


if __name__ == "__main__":
    # 実行時間計測の開始
    start_time = time.time()

    # 出力ディレクトリを用意
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初期グリッドの設定
    grid = initialize_grid(GRID_SIZE)

    # プロット設定
    fig, ax = plt.subplots()

    # 3状態（死・弱・強）をはっきり区別する離散カラーマップ
    cmap = ListedColormap(["black", "yellow", "blue"])

    img = ax.imshow(grid, interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_title("Game of Life with Strong and Weak States")

    # アニメーションの作成
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(img, grid, ax),
        frames=FRAMES,
        interval=INTERVAL_MS,
        repeat=False,
        blit=False,
    )

    # アニメーションを保存する（imagemagick不要のpillowライターを使用）
    ani.save(OUTPUT_PATH, writer="pillow", fps=FPS)

    plt.close(fig)

    # 実行時間計測の終了
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.2f}秒")
    print(f"保存先: {OUTPUT_PATH}")
