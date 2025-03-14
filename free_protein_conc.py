import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# パラメータの設定
Kd_values = [1e-6, 1e-7, 1e-8]  # Dissociation constants [M]
L0_values = [5*1e-5, 10*1e-6, 5*1e-6]   # Ligand concentrations [M]

# タンパク質濃度の範囲：1e-6 M から 1e-4 M（線形に100点）
P0_range = np.linspace(1*1e-6, 5*1e-5, num=20)
# μM単位に変換: 1e-6 M = 1 μM, 1e-4 M = 100 μM
P0_range_uM = P0_range * 1e6

# constrained_layoutを使ってレイアウト調整
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9), constrained_layout=True)

# Kd, L0 の組み合わせごとに計算・プロット
for i, Kd in enumerate(Kd_values):
    for j, L0 in enumerate(L0_values):
        ax = axes[i, j]
        free_fraction = []  # 自由タンパク質比率のリスト
        
        # 各タンパク質濃度で計算（単位はM）
        for P0 in P0_range:
            # 二次方程式を解いて[PL]を求める
            PL = 0.5 * (P0 + L0 + Kd - np.sqrt((P0 + L0 + Kd)**2 - 4 * P0 * L0))
            free_frac = (P0 - PL) / P0  # 自由タンパク質比率 = [P] / ( [P] + [PL] )
            free_fraction.append(free_frac)
        
        # μM単位に変換した横軸でプロット（線形スケール）
        ax.plot(P0_range_uM, free_fraction, marker='o', linestyle='-', label='Free Protein Fraction')
        ax.grid(True)
        # 現在のパラメータ値をタイトルに表示
        ax.set_title(f"Kd = {Kd:.1e} M, L0 = {L0:.1e} M", fontsize=10)
        ax.set_xlabel('Protein Concentration (μM)')
        ax.set_ylabel('Free Protein Fraction')
        # x軸の目盛りをプレーンな表示に
        ax.ticklabel_format(axis='x', style='plain')
        # x軸の表示範囲をμM単位に合わせる
        ax.set_xlim(P0_range_uM[0], P0_range_uM[-1])

plt.show()
