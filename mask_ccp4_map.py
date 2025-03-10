#!/usr/bin/env python3
import gemmi
import numpy as np
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input_map.ccp4 input.pdb output_map.ccp4".format(sys.argv[0]))
        sys.exit(1)

    input_map = sys.argv[1]
    pdb_file = sys.argv[2]
    output_map = sys.argv[3]
    mask_radius = 2.0  # マスクする半径 (Å)

    print("Reading CCP4 map from:", input_map)
    # setup=True により、グリッドがユニットセル全体を覆うように再構成され、軸が X, Y, Z に並び替えられる
    ccp4_map = gemmi.read_ccp4_map(input_map, setup=True)
    grid = ccp4_map.grid

    # グリッドサイズは grid.nu, grid.nv, grid.nw で取得
    nu = grid.nu
    nv = grid.nv
    nw = grid.nw
    print("Grid dimensions: {} x {} x {}".format(nu, nv, nw))

    # 原点はグリッドの (0,0,0) の実空間座標
    origin = grid.get_position(0, 0, 0)

    # 各軸方向のグリッド間隔は、隣接グリッド点との差分から計算
    pos_100 = grid.get_position(1, 0, 0)
    pos_010 = grid.get_position(0, 1, 0)
    pos_001 = grid.get_position(0, 0, 1)
    step_x = pos_100.x - origin.x
    step_y = pos_010.y - origin.y
    step_z = pos_001.z - origin.z
    print("Grid spacing: step_x = {:.3f}, step_y = {:.3f}, step_z = {:.3f}".format(step_x, step_y, step_z))

    # 全体の平均密度を算出
    density_array = grid.array
    mean_density = float(np.mean(density_array))
    print("Mean density =", mean_density)

    # PDBファイルの読み込みと対象原子（重元素）の抽出
    print("Reading PDB file:", pdb_file)
    structure = gemmi.read_structure(pdb_file)
    heavy_atoms = []
    # マスク対象とする元素のリスト（大文字で指定）
    mask_elements = {"ZN", "NA", "CA", "K", "MG"}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name.upper() in mask_elements:
                        heavy_atoms.append(atom)
    print("Number of heavy atoms to mask:", len(heavy_atoms))

    # 各対象原子周囲のグリッド点を平均密度に置換
    print("Masking grid points within radius {:.2f} Å...".format(mask_radius))
    for atom in heavy_atoms:
        atom_pos = atom.pos  # gemmi.Position (Cartesian)

        # 原子位置に対応するグリッドインデックス（浮動小数点値）
        center_i = (atom_pos.x - origin.x) / step_x
        center_j = (atom_pos.y - origin.y) / step_y
        center_k = (atom_pos.z - origin.z) / step_z

        # mask_radius に対応するインデックス幅（各軸方向）
        di = int(np.ceil(mask_radius / abs(step_x))) if step_x != 0 else 0
        dj = int(np.ceil(mask_radius / abs(step_y))) if step_y != 0 else 0
        dk = int(np.ceil(mask_radius / abs(step_z))) if step_z != 0 else 0

        # グリッドインデックスの範囲を決定（境界チェックあり）
        i_min = max(0, int(center_i) - di)
        i_max = min(nu - 1, int(center_i) + di)
        j_min = max(0, int(center_j) - dj)
        j_max = min(nv - 1, int(center_j) + dj)
        k_min = max(0, int(center_k) - dk)
        k_max = min(nw - 1, int(center_k) + dk)

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    # グリッド点の実空間座標
                    grid_point = grid.get_position(i, j, k)
                    # 原子との距離が mask_radius 内なら、グリッド点の値を平均密度に置換
                    if (grid_point - atom_pos).length() <= mask_radius:
                        grid.set_value(i, j, k, mean_density)

    # マスク処理後のマップを CCP4 形式で出力
    print("Writing masked CCP4 map to:", output_map)
    ccp4_map.write_ccp4_map(output_map)
    print("Done.")

if __name__ == "__main__":
    main()
