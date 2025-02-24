from iotbx import pdb

# PDBファイルの読み込み
pdb_file = "your_structure.pdb"  # ここを対象のPDBファイル名に置き換えてください
pdb_input = pdb.input(file_name=pdb_file)
hierarchy = pdb_input.construct_hierarchy()

# 各軸の最小・最大値の初期化
x_min = y_min = z_min = float('inf')
x_max = y_max = z_max = float('-inf')

# 原子座標の取得と最小・最大値の更新
for atom in hierarchy.atoms():
    x, y, z = atom.xyz
    if x < x_min: x_min = x
    if x > x_max: x_max = x
    if y < y_min: y_min = y
    if y > y_max: y_max = y
    if z < z_min: z_min = z
    if z > z_max: z_max = z

# 各軸の長さ（サイズ）の計算
x_length = x_max - x_min
y_length = y_max - y_min
z_length = z_max - z_min

# 結果の表示
print(f"分子のサイズ:")
print(f"X軸: {x_length:.2f} Å")
print(f"Y軸: {y_length:.2f} Å")
print(f"Z軸: {z_length:.2f} Å")
