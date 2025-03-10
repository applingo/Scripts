#!/usr/bin/env python3
import gemmi
import numpy as np
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input_map.ccp4 input.pdb output_map.ccp4".format(sys.argv[0]))
        sys.exit(1)

    ccp4_file = sys.argv[1]
    pdb_file = sys.argv[2]
    output_file = sys.argv[3]
    mask_radius = 2.0  # マスクする半径（Å単位、必要に応じて変更）

    # CCP4マップの読み込み
    print("CCP4マップを読み込み中:", ccp4_file)
    map_obj = gemmi.read_ccp4_map(ccp4_file)
    grid = map_obj.grid

    # 原点はグリッドの (0,0,0) の位置
    origin = grid.get_position(0, 0, 0)
    # 各軸方向のステップは、隣接グリッド点間の距離として計算
    pos_x1 = grid.get_position(1, 0, 0)
    pos_y1 = grid.get_position(0, 1, 0)
    pos_z1 = grid.get_position(0, 0, 1)
    step_x = pos_x1.x - origin.x
    step_y = pos_y1.y - origin.y
    step_z = pos_z1.z - origin.z
    step = gemmi.Vec3(step_x, step_y, step_z)
    
    size = grid.size  # グリッドサイズ (nu, nv, nw)

    # 全体の平均密度を算出
    density_array = grid.array
    mean_density = float(np.mean(density_array))
    print("全体の平均密度 =", mean_density)

    # PDBファイルの読み込みと対象原子の抽出
    print("PDBファイルを読み込み中:", pdb_file)
    structure = gemmi.read_structure(pdb_file)
    heavy_atoms = []
    # マスク対象とする元素のリスト
    mask_elements = {"ZN", "NA", "CA", "K", "MG"}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name.upper() in mask_elements:
                        heavy_atoms.append(atom)
    print("検出されたマスク対象原子の数 =", len(heavy_atoms))

    # 各対象原子周囲を走査し、mask_radius内のボクセルを平均密度に置換
    print("対象原子周囲をマスクして平均密度に置換中...")
    for atom in heavy_atoms:
        pos = atom.pos  # 原子の実空間座標

        # 原子位置をグリッドインデックスに変換（浮動小数点値）
        center_i = (pos.x - origin.x) / step.x
        center_j = (pos.y - origin.y) / step.y
        center_k = (pos.z - origin.z) / step.z

        # mask_radiusに対応するグリッド上の範囲を決定
        di = int(np.ceil(mask_radius / step.x))
        dj = int(np.ceil(mask_radius / step.y))
        dk = int(np.ceil(mask_radius / step.z))
        i_min = max(0, int(center_i - di))
        i_max = min(size[0]-1, int(center_i + di))
        j_min = max(0, int(center_j - dj))
        j_max = min(size[1]-1, int(center_j + dj))
        k_min = max(0, int(center_k - dk))
        k_max = min(size[2]-1, int(center_k + dk))

        for i in range(i_min, i_max+1):
            for j in range(j_min, j_max+1):
                for k in range(k_min, k_max+1):
                    # グリッドインデックスから実空間座標を再計算
                    point = gemmi.Vec3(origin.x + i * step.x,
                                        origin.y + j * step.y,
                                        origin.z + k * step.z)
                    # 距離が mask_radius 内なら値を平均密度に更新
                    if (point - pos).length() <= mask_radius:
                        grid.set_value(i, j, k, mean_density)

    # マスク処理後のマップをCCP4形式で出力
    print("マスク処理後のマップを出力中:", output_file)
    map_obj.write_ccp4_map(output_file)
    print("処理完了。")

if __name__ == "__main__":
    main()

