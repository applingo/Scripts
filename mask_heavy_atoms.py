#!/usr/bin/env python3
import gemmi
import numpy as np
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input.mtz input.pdb output_map.ccp4".format(sys.argv[0]))
        sys.exit(1)

    mtz_file = sys.argv[1]
    pdb_file = sys.argv[2]
    output_map = sys.argv[3]
    mask_radius = 2.0  # マスクする半径（Å単位、必要に応じて変更）

    # MTZファイルの読み込みと密度マップ生成
    print("MTZファイルを読み込み、密度マップを作成しています...")
    mtz = gemmi.read_mtz_file(mtz_file)
    # FWT, PHWT列からマップへ変換。grid_stepは必要に応じて変更してください
    map_obj = mtz.transform_to_map("FWT", "PHWT", grid_step=0.5)
    grid = map_obj.grid

    # 全体の平均密度を算出
    print("全体の平均密度を算出しています...")
    array = grid.array  # gemmi.FloatGrid の内部は numpy 配列になっています
    mean_density = float(np.mean(array))
    print("平均密度 =", mean_density)

    # マップのグリッドパラメータ（原点、格子間隔、サイズ）を取得
    origin = grid.origin  # grid[0,0,0] に対応する実空間座標
    step = grid.step      # 各方向のグリッド間隔（gemmi.Vec3）
    size = grid.size      # グリッドサイズ (nx, ny, nz)

    # PDBファイルの読み込みと Zn 原子の抽出
    print("PDBファイルを読み込み、Zn 原子を抽出しています...")
    structure = gemmi.read_structure(pdb_file)
    heavy_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # 原子記号が Zn のものを対象（大文字・小文字に注意）
                    if atom.element.name.upper() == "ZN":
                        heavy_atoms.append(atom)
    print("検出された Zn 原子の数 =", len(heavy_atoms))

    # 各 Zn 原子の周囲をマスクして平均密度に置換
    print("Zn 原子周囲をマスクして平均密度に置換中...")
    for atom in heavy_atoms:
        pos = atom.pos  # 原子の実空間座標（gemmi.Vec3）

        # grid 内での中心位置を求める（各軸ごとに）
        center_i = (pos.x - origin.x) / step.x
        center_j = (pos.y - origin.y) / step.y
        center_k = (pos.z - origin.z) / step.z

        # マスク対象とする範囲をグリッドインデックスで決定（± mask_radius に相当する範囲）
        di = int(np.ceil(mask_radius / step.x))
        dj = int(np.ceil(mask_radius / step.y))
        dk = int(np.ceil(mask_radius / step.z))
        i_min = max(0, int(center_i - di))
        i_max = min(size[0] - 1, int(center_i + di))
        j_min = max(0, int(center_j - dj))
        j_max = min(size[1] - 1, int(center_j + dj))
        k_min = max(0, int(center_k - dk))
        k_max = min(size[2] - 1, int(center_k + dk))

        # 上記範囲内を走査し、実際の距離が mask_radius 以下なら値を平均密度に置換
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    # 現在のグリッド点の実空間座標を計算
                    point = gemmi.Vec3(origin.x + i * step.x,
                                        origin.y + j * step.y,
                                        origin.z + k * step.z)
                    if (point - pos).length() <= mask_radius:
                        grid.set_value(i, j, k, mean_density)

    # マスク処理後のマップをCCP4形式で出力
    print("マスク処理後のマップを {} に書き出しています...".format(output_map))
    map_obj.write_ccp4_map(output_map)
    print("完了しました。")

if __name__ == "__main__":
    main()
