#!/usr/bin/env python
import os, sys
import numpy as np
from iotbx import pdb
from iotbx import file_reader

def compute_bounding_box(pdb_file):
    """
    入力 PDB ファイルから原子座標の最小・最大座標を計算し、
    中心座標と各軸方向の寸法を返す
    """
    pdb_in = pdb.input(file_name=pdb_file)
    hierarchy = pdb_in.construct_hierarchy()
    sites = hierarchy.atoms().extract_xyz()
    min_coords = np.min(sites, axis=0)
    max_coords = np.max(sites, axis=0)
    center = (min_coords + max_coords) / 2.0
    dims = max_coords - min_coords
    return center, dims

def main():
    if len(sys.argv) < 3:
        print("Usage: {} input.mtz input.pdb".format(sys.argv[0]))
        sys.exit(1)

    mtz_file = sys.argv[1]
    pdb_file = sys.argv[2]

    # PDB ファイルから分子のバウンディングボックスを計算
    center, dims = compute_bounding_box(pdb_file)
    # 立方体（等方的な領域）とするため、最大寸法に padding を加える
    max_dim = max(dims)
    padding = 5.0  # 必要に応じて調整（単位：Å）
    cube_dim = max_dim + 2 * padding

    # phenix.cut_out_density を呼び出すためのコマンド文字列を作成
    # ※ここでは PDB の重心 (center) と、各軸が同一長さの cube_dim を指定
    cut_cmd = ("phenix.cut_out_density {pdb} {mtz} "
               "cutout_center={x:.3f},{y:.3f},{z:.3f} "
               "cutout_dimensions={d:.3f},{d:.3f},{d:.3f}").format(
                   pdb=pdb_file, mtz=mtz_file,
                   x=center[0], y=center[1], z=center[2],
                   d=cube_dim)
    print("Running command:\n", cut_cmd)
    ret = os.system(cut_cmd)
    if ret != 0:
        print("Error running phenix.cut_out_density.")
        sys.exit(1)

    # phenix.cut_out_density の出力は通常 "cutout.mtz" (および cutout.pdb)
    # これを phenix.mtz2map で CCP4 マップに変換
    map_cmd = "phenix.mtz2map cutout.mtz"
    print("Running command:\n", map_cmd)
    ret = os.system(map_cmd)
    if ret != 0:
        print("Error running phenix.mtz2map.")
        sys.exit(1)

    print("Map extraction complete. 出力ファイルを確認してください。")

if __name__ == "__main__":
    main()
