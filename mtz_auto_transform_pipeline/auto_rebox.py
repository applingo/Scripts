#!/usr/bin/env phenix.python
import sys, math
from iotbx import ccp4_map
from cctbx import maptbx
from scitbx.array_family import flex

def main():
    if len(sys.argv) < 3:
        print("Usage: auto_rebox.py input_map.ccp4 output_cubic_map.ccp4")
        sys.exit(1)
    input_map = sys.argv[1]
    output_map = sys.argv[2]
    
    # 元のマップ読み込み
    map_reader = ccp4_map.map_reader(input_map)
    map_data = map_reader.map_data.as_double()  # 3D flex.double array
    header = map_reader.header
    # 単位セルパラメータ (a,b,c,alpha,beta,gamma)
    a, b, c, alpha, beta, gamma = header.unit_cell_parameters
    nx, ny, nz = map_data.all()
    
    # 各軸の格子間隔
    spacing_x = a / nx
    spacing_y = b / ny
    spacing_z = c / nz
    # 解像度として最大の格子間隔を採用（保守的に）
    spacing = max(spacing_x, spacing_y, spacing_z)
    
    # 立方体とするため，新たなボックスサイズをL = max(a,b,c)に設定
    L = max(a, b, c)
    new_n = int(math.ceil(L / spacing))
    new_grid = (new_n, new_n, new_n)
    
    # 元のマップの物理中心を計算
    # 元のオフセット（header.origin）は、一般に物理的な原点位置（Å）として与えられている
    orig_origin = header.origin  # タプル (ox, oy, oz)
    center_phys = (orig_origin[0] + (nx*spacing_x)/2.0,
                   orig_origin[1] + (ny*spacing_y)/2.0,
                   orig_origin[2] + (nz*spacing_z)/2.0)
    
    # 新たな立方体マップのグリッド間隔は元と同じとし，物理中心を保つよう新オリジンを決定
    new_spacing = spacing
    new_origin = (center_phys[0] - new_n/2.0*new_spacing,
                  center_phys[1] - new_n/2.0*new_spacing,
                  center_phys[2] - new_n/2.0*new_spacing)
    
    # 新たな立方体マップ用の空配列を作成
    new_map = flex.double(flex.grid(new_grid))
    
    # 新たなグリッド各点の物理座標を計算し、元のマップから補間して新マップに格納（trilinear補間）
    for i in range(new_n):
        for j in range(new_n):
            for k in range(new_n):
                # 新グリッド点の物理座標
                x = new_origin[0] + i * new_spacing
                y = new_origin[1] + j * new_spacing
                z = new_origin[2] + k * new_spacing
                # 対応する元のグリッド内の位置（連続値）
                fx = (x - orig_origin[0]) / spacing_x
                fy = (y - orig_origin[1]) / spacing_y
                fz = (z - orig_origin[2]) / spacing_z
                # maptbxの補間関数を利用（trilinear補間）
                val = maptbx.interpolate_map_values(map_data, (fx, fy, fz))
                new_map[i,j,k] = val

    # 新たなマップヘッダー：立方体ユニットセル（L, L, L, 90,90,90）と新グリッド，新オリジンを設定
    new_unit_cell = (L, L, L, 90, 90, 90)
    new_header = header.customized_copy(unit_cell_parameters=new_unit_cell, origin=new_origin, grid=new_grid)
    
    # 新たな立方体マップを出力
    from iotbx import ccp4_map_writer
    ccp4_map_writer.write_ccp4_map_file(file_name=output_map, map_data=new_map, header=new_header)
    
    # 適用されたシフトは new_origin - orig_origin (物理座標での平行移動)
    shift_vector = (new_origin[0] - orig_origin[0],
                    new_origin[1] - orig_origin[1],
                    new_origin[2] - orig_origin[2])
    print("Applied shift vector (Å): {} {} {}".format(*shift_vector))

if __name__=="__main__":
    main()
