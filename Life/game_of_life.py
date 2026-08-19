# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:30:06 2025

@author: ohshi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time  # 時間計測用モジュールのインポート

# グリッドサイズ
GRID_SIZE = 254

# ライフゲームの初期化
def initialize_grid(size):
    # 0: 死んでいる, 1: 弱い生存, 2: 強い生存
    return np.random.choice([0, 1, 2], size=(size, size), p=[0.3, 0.3, 0.4])

# 隣接セルを数える
def count_neighbors(grid, x, y):
    neighbors = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] > 0:  # 生きているセル
                    neighbors += 1
    return neighbors

# ゲームの進行
def update_grid(grid):
    new_grid = grid.copy()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            neighbors = count_neighbors(grid, x, y)
            if grid[x, y] == 1:  # 弱い生存状態
                # 生存率を調整（弱い生存状態は生存しにくい）
                if neighbors < 2 or neighbors > 3:
                    new_grid[x, y] = 0  # 死亡
                elif np.random.random() > 0.998:  # 99%で強い状態に進化
                    new_grid[x, y] = 2
            elif grid[x, y] == 2:  # 強い生存状態
                # 強い状態は生存しやすい
                if neighbors < 2 or neighbors > 3:
                    new_grid[x, y] = 1  # 弱い状態に戻る
            else:  # 死んでいるセル
                if neighbors == 3:
                    new_grid[x, y] = 1  # 弱い生存状態が誕生
    return new_grid

# 可視化のためのアニメーション
def animate(frame, img, grid):
    new_grid = update_grid(grid)
    img.set_data(new_grid)
    
    # セルの状態ごとのカウント
    dead_cells = np.sum(new_grid == 0)  # 死んでいるセルの数
    weak_cells = np.sum(new_grid == 1)  # 弱い生存状態のセルの数
    strong_cells = np.sum(new_grid == 2)  # 強い生存状態のセルの数
    
    # タイトルにセルの数を表示
    ax.set_title(f"Frame: {frame}  Dead: {dead_cells}  Weak: {weak_cells}  Strong: {strong_cells}")
    
    grid[:] = new_grid
    return img

if __name__ =='__main__':
    # 実行時間計測の開始
    start_time = time.time()
    
    # 初期グリッドの設定
    grid = initialize_grid(GRID_SIZE)
    
    # プロット設定
    fig, ax = plt.subplots()
    
    # 明確に色を区別するカスタムカラーマップを設定
    cmap = plt.cm.get_cmap('coolwarm', 3)  # 3段階の色
    cmap.set_under('black')  # 死んでいるセルの色を黒に設定
    cmap.set_over('yellow')  # 弱い生存状態の色を暖色系に設定
    cmap.set_bad('blue')  # 強い生存状態の色を寒色系に設定
    
    # カラーマップを変更した画像を表示
    img = ax.imshow(grid, interpolation='nearest', cmap=cmap, vmin=0, vmax=2)
    ax.set_title('Game of Life with Strong and Weak States')
    
    # アニメーションの作成
    ani = animation.FuncAnimation(fig, animate, fargs=(img, grid), frames=3000, interval=100, repeat=False)
    
    # アニメーションを保存する
    ani.save('gif/game_of_life26.gif', writer='imagemagick', fps=10)
    
    # 実行時間計測の終了
    end_time = time.time()
    
    # 実行時間を計算して表示
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time:.2f}秒")
