import sys
import os
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from scipy.linalg import svd
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R

def load_pdb(pdb_file):
    """
    PDBファイルを読み込み、全ての原子とCα原子の座標を取得する。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_file)
    atoms = list(structure.get_atoms())
    ca_atoms = [atom for atom in atoms if atom.get_id() == 'CA']
    coordinates = np.array([atom.get_coord() for atom in ca_atoms])
    return structure, ca_atoms, coordinates

def perform_pca(coordinates):
    """
    主成分分析（PCA）を実行し、主な平面を特定する。
    """
    # 中心化
    centroid = np.mean(coordinates, axis=0)
    centered_coords = coordinates - centroid

    # SVDによるPCA
    U, S, Vt = svd(centered_coords)
    normal_vector = Vt[2]  # 第3主成分が平面の法線
    plane_vectors = Vt[:2]  # 第1および第2主成分が平面の基底

    return centroid, plane_vectors, normal_vector

def project_to_plane(coordinates, centroid, plane_vectors):
    """
    3D座標を主成分平面に投影し、2D座標を取得する。
    """
    centered_coords = coordinates - centroid
    x = np.dot(centered_coords, plane_vectors[0])
    y = np.dot(centered_coords, plane_vectors[1])
    return np.vstack((x, y)).T

def fit_ellipse(x, y):
    """
    2Dのポイントに対して楕円をフィッティングする。
    最小二乗法を使用。
    """
    # パラメータは [a, b, xc, yc, theta]
    def ellipse_residuals(params, x, y):
        a, b, xc, yc, theta = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_shift = x - xc
        y_shift = y - yc
        term1 = ((x_shift * cos_theta + y_shift * sin_theta) / a) ** 2
        term2 = ((-x_shift * sin_theta + y_shift * cos_theta) / b) ** 2
        return term1 + term2 - 1

    # 初期パラメータの推定
    a_init = (np.max(x) - np.min(x)) / 2
    b_init = (np.max(y) - np.min(y)) / 2
    xc_init = np.mean(x)
    yc_init = np.mean(y)
    theta_init = 0

    initial_params = [a_init, b_init, xc_init, yc_init, theta_init]

    # 楕円フィッティングの実行
    params_opt, _ = leastsq(ellipse_residuals, initial_params, args=(x, y))

    return params_opt  # [a, b, xc, yc, theta]

def compute_normal_vectors(a, b, theta, points):
    """
    フィットした楕円上の各点での法線ベクトルを計算する。
    pointsは楕円のパラメトリック角度（t）ではなく、実際の点の座標。
    """
    # 楕円の微分から法線ベクトルを計算
    # 楕円方程式: (x')^2 / a^2 + (y')^2 / b^2 = 1
    # 法線ベクトルは [2x'/a^2, 2y'/b^2]
    normals = []
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for (x, y) in points:
        # 回転前の座標
        x_shift = x - 0  # フィッティング後はxc=0, yc=0
        y_shift = y - 0

        # 法線ベクトルの計算
        nx = (2 * x_shift) / (a ** 2)
        ny = (2 * y_shift) / (b ** 2)
        normal = np.array([nx, ny])

        # 回転
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        normal_rotated = rotation_matrix.dot(normal)

        # 正規化
        norm_length = np.linalg.norm(normal_rotated)
        if norm_length == 0:
            normals.append(np.array([0, 0]))
        else:
            normals.append(normal_rotated / norm_length)

    return np.array(normals)

def rotate_structure(structure, rotation_axis, angle_deg, center):
    """
    構造全体を指定された軸と角度で回転させる。
    rotation_axis: 回転軸の3Dベクトル
    angle_deg: 回転角度（度単位）
    center: 回転の中心点（3D座標）
    """
    # 正規化された回転軸
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # 回転オブジェクトの作成
    rot = R.from_rotvec(np.deg2rad(angle_deg) * rotation_axis)

    # 新しい構造の作成
    rotated_structure = Structure.Structure('rotated')
    for model in structure:
        new_model = Model.Model(model.get_id())
        for chain in model:
            new_chain = Chain.Chain(chain.get_id())
            for residue in chain:
                new_residue = Residue.Residue(residue.get_id(), residue.get_resname(), residue.get_segid())
                for atom in residue:
                    coord = atom.get_coord()
                    # 中心を原点に移動
                    coord_centered = coord - center
                    # 回転適用
                    coord_rotated = rot.apply(coord_centered)
                    # 元の位置に戻す
                    coord_new = coord_rotated + center
                    # 新しい原子を作成
                    new_atom = Atom.Atom(atom.get_id(),
                                         coord_new,
                                         atom.get_bfactor(),
                                         atom.get_occupancy(),
                                         atom.get_altloc(),
                                         atom.get_fullname(),
                                         atom.get_serial_number(),
                                         atom.element)
                    new_residue.add(new_atom)
                new_chain.add(new_residue)
            new_model.add(new_chain)
        rotated_structure.add(new_model)
    return rotated_structure

def save_pdb(structure, filename):
    """
    構造をPDBファイルとして保存する。
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

def main(pdb_input):
    # ステップ1: PDBファイルの読み込み
    structure, ca_atoms, coordinates = load_pdb(pdb_input)
    print(f"Loaded PDB file '{pdb_input}' with {len(ca_atoms)} Cα atoms.")

    # ステップ2: PCAによる主成分平面の決定
    centroid, plane_vectors, normal_vector = perform_pca(coordinates)
    print("Performed PCA to determine the principal plane.")

    # ステップ3: 平面への投影
    projected_2d = project_to_plane(coordinates, centroid, plane_vectors)
    print("Projected Cα coordinates onto the principal plane.")

    # ステップ4: 楕円のフィッティング
    x = projected_2d[:, 0]
    y = projected_2d[:, 1]
    a, b, xc, yc, theta = fit_ellipse(x, y)
    print(f"Fitted ellipse parameters:\n a={a:.3f}, b={b:.3f}, xc={xc:.3f}, yc={yc:.3f}, theta={np.degrees(theta):.3f} degrees")

    # ステップ5: 法線ベクトルの計算
    # 楕円の中心を原点と仮定
    ellipse_points = projected_2d  # [x, y]
    normals_2d = compute_normal_vectors(a, b, theta, ellipse_points)
    print("Computed normal vectors for each Cα atom on the ellipse.")

    # ステップ6: 回転の実行とPDBファイルの保存
    angle_deg = 360 / 11  # 約32.727度
    print(f"Rotation angle set to {angle_deg:.3f} degrees.")

    # 楕円の中心を3D空間に戻す
    center_3d = centroid

    # 10個の回転を実行
    num_rotations = 10
    for i in range(num_rotations):
        normal_2d = normals_2d[i % len(normals_2d)]
        # 回転軸を3Dに拡張（平面の法線に垂直）
        # 平面の基底ベクトルを使用して3D回転軸を計算
        rotation_axis_3d = normal_2d[0] * plane_vectors[0] + normal_2d[1] * plane_vectors[1]
        rotation_axis_3d = rotation_axis_3d / np.linalg.norm(rotation_axis_3d)

        # 構造の回転
        rotated_struct = rotate_structure(structure, rotation_axis_3d, angle_deg, center_3d)
        output_filename = f"rotated_structure_{i+1}.pdb"
        save_pdb(rotated_struct, output_filename)
        print(f"Saved rotated structure {i+1} to '{output_filename}'.")

    print("All rotated PDB files have been successfully created.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rotate_peptide.py <input_pdb_file>")
        sys.exit(1)

    pdb_input = sys.argv[1]
    if not os.path.isfile(pdb_input):
        print(f"Error: File '{pdb_input}' does not exist.")
        sys.exit(1)

    main(pdb_input)
