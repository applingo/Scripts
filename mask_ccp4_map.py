#!/usr/bin/env python3
import gemmi
import numpy as np
import sys
import math

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input_map.ccp4 input.pdb output_map.ccp4".format(sys.argv[0]))
        sys.exit(1)

    input_map = sys.argv[1]
    pdb_file = sys.argv[2]
    output_map = sys.argv[3]
    mask_radius = 2.0  # マスクする半径 (Å)

    print("Reading CCP4 map from:", input_map)
    m = gemmi.read_ccp4_map(input_map)
    # マップをユニットセル全体に合わせ、軸を X,Y,Z に整列
    m.setup(float('nan'))

    # Origin（原点オフセット）はヘッダーのワード 49～51 から取得
    origin = gemmi.Position(m.header_float(49), m.header_float(50), m.header_float(51))
    # グリッドサイズ（各軸のボクセル数）
    nu, nv, nw = m.grid.nu, m.grid.nv, m.grid.nw
    # ボクセルサイズ（各軸方向）は、単位セルの a, b, c をグリッドサイズで割る
    step_x = m.grid.unit_cell.a / nu
    step_y = m.grid.unit_cell.b / nv
    step_z = m.grid.unit_cell.c / nw
    step = (step_x, step_y, step_z)
    size = (nu, nv, nw)

    print(f"Origin: {origin}")
    print(f"Step size: {step}")
    print(f"Grid size: {size}")

    # 全体の平均密度を計算
    density_array = m.grid.array
    mean_density = float(np.mean(density_array))
    print("Mean density =", mean_density)

    # PDBファイルから重元素（Zn, Na, Ca, K, Mg）を抽出
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

    # 各重元素周囲のグリッド点を走査し、mask_radius 内なら全体平均密度に置換
    print(f"Masking grid points within {mask_radius:.2f} Å of heavy atoms...")
    for atom in heavy_atoms:
        atom_pos = atom.pos  # gemmi.Position (実空間座標)

        # 原子位置をグリッドインデックス（浮動小数点値）に変換
        center_i = (atom_pos.x - origin.x) / step_x
        center_j = (atom_pos.y - origin.y) / step_y
        center_k = (atom_pos.z - origin.z) / step_z

        # mask_radius に相当するインデックス幅を計算
        di = int(math.ceil(mask_radius / abs(step_x))) if step_x != 0 else 0
        dj = int(math.ceil(mask_radius / abs(step_y))) if step_y != 0 else 0
        dk = int(math.ceil(mask_radius / abs(step_z))) if step_z != 0 else 0

        # チェックするグリッドインデックスの範囲（境界チェック付き）
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
                    grid_point = m.grid.get_position(i, j, k)
                    # 原子からの距離が mask_radius 内なら、そのグリッド点を平均密度に置換
                    if (grid_point - atom_pos).length() <= mask_radius:
                        m.grid.set_value(i, j, k, mean_density)

    # マスク処理後のマップを CCP4 形式で出力
    print("Writing masked CCP4 map to:", output_map)
    m.write_ccp4_map(output_map)
    print("Done.")

if __name__ == "__main__":
    main()
