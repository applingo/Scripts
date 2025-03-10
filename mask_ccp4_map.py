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
    mask_radius = 2.0  # マスクする半径 (Å)

    # CCP4マップの読み込み
    print("Reading CCP4 map:", ccp4_file)
    map_obj = gemmi.read_ccp4_map(ccp4_file)
    grid = map_obj.grid

    # 原点はグリッドの (0,0,0) の実空間座標
    origin = grid.get_position(0, 0, 0)
    # 各軸方向のグリッド間隔は隣接点との差分から計算
    pos_100 = grid.get_position(1, 0, 0)
    pos_010 = grid.get_position(0, 1, 0)
    pos_001 = grid.get_position(0, 0, 1)
    step_x = pos_100.x - origin.x
    step_y = pos_010.y - origin.y
    step_z = pos_001.z - origin.z
    # （step 値は負になる可能性もあるため、距離計算では絶対値を使います）

    # グリッドサイズは grid.nu, grid.nv, grid.nw で取得
    nu = grid.nu
    nv = grid.nv
    nw = grid.nw
    print("Grid dimensions: {} x {} x {}".format(nu, nv, nw))
    
    # 全体の平均密度を算出
    density_array = grid.array
    mean_density = float(np.mean(density_array))
    print("Mean density =", mean_density)

    # PDBファイルの読み込みと対象原子の抽出
    print("Reading PDB file:", pdb_file)
    structure = gemmi.read_structure(pdb_file)
    heavy_atoms = []
    mask_elements = {"ZN", "NA", "CA", "K", "MG"}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name.upper() in mask_elements:
                        heavy_atoms.append(atom)
    print("Number of heavy atoms to mask:", len(heavy_atoms))

    # 各対象原子周囲のグリッド点を平均密度に置換
    print("Masking grid points around heavy atoms...")
    for atom in heavy_atoms:
        pos = atom.pos  # 原子の実空間座標

        # 原子位置をグリッドインデックスに変換（浮動小数点値）
        center_i = (pos.x - origin.x) / step_x
        center_j = (pos.y - origin.y) / step_y
        center_k = (pos.z - origin.z) / step_z

        # mask_radius に対応するインデックス幅（各軸方向）
        di = int(np.ceil(mask_radius / abs(step_x))) if step_x != 0 else 0
        dj = int(np.ceil(mask_radius / abs(step_y))) if step_y != 0 else 0
        dk = int(np.ceil(mask_radius / abs(step_z))) if step_z != 0 else 0

        # 考慮するグリッドインデックスの範囲
        i_min = max(0, int(center_i) - di)
        i_max = min(nu - 1, int(center_i) + di)
        j_min = max(0, int(center_j) - dj)
        j_max = min(nv - 1, int(center_j) + dj)
        k_min = max(0, int(center_k) - dk)
        k_max = min(nw - 1, int(center_k) + dk)

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    # グリッド点の実空間座標を取得
                    grid_point = grid.get_position(i, j, k)
                    # 原子との距離が mask_radius 内なら、そのボクセルを平均密度に置換
                    if (grid_point - pos).length() <= mask_radius:
                        grid.set_value(i, j, k, mean_density)

    # マスク処理後のマップを CCP4 形式で出力
    print("Writing output map to:", output_file)
    map_obj.write_ccp4_map(output_file)
    print("Done.")

if __name__ == "__main__":
    main()
