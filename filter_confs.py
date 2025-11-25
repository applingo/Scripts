#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    python filter_confs.py --smi peptide.smi --pdb_dir ./pdbs \
        --out_all energies_all.txt --out_good energies_filtered.txt

機能要約:
- .smi から SMILES を1つ読み込む
- SMILES から H 付きテンプレート分子を作成
- 指定ディレクトリ内の *.pdb をすべて処理：
    - PDB を読み込み、H を付加
    - フラグメント分解して「テンプレートと重原子数が最も近いフラグメント」を選択
      → 水やイオンなどの余計な分子は自動的に削除
    - テンプレートとの原子数一致を試み、合えば AssignBondOrdersFromTemplate
    - 幾何クラッシュチェック（非結合 heavy atom 間距離）
    - MMFF（ダメなら UFF）でエネルギー計算:
        * E_raw: 最小化前
        * E_min: 軽い最小化(maxIts=50)後
    - per-heavy-atom エネルギー e = E_min / N_heavy を計算
- 全構造の e の分布から e_min を求め、Δe/atom が閾値以内のものだけ GOOD と判定
- energies_all.txt に全構造、energies_filtered.txt に GOOD 構造のみ出力
"""

import os
import glob
import argparse
from math import sqrt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem import rdDistGeom


# ----------------- ユーザー調整しやすいパラメータ -----------------

# 非結合 heavy atom ペアのクラッシュ判定用閾値（絶対距離）
ABS_CLASH_DIST = 1.0  # Å

# vdW 半径の和に対する縮小係数（d < SCALE * (r_i + r_j) でクラッシュとみなす）
VDW_SCALE = 0.6

# 1 heavy atom あたりの許容エネルギー差 (kcal/mol)
DELTA_E_PER_ATOM_MAX = 1.5

# E_raw - E_min がこれより大きい場合は「強いクラッシュあり」とみなす (kcal/mol)
DELTA_E_RELAX_MAX = 50.0

# 最小化のステップ数
MINIMIZE_STEPS = 50


# ----------------- ユーティリティ関数群 -----------------

def read_smiles_from_smi(smi_path):
    """最初の有効行から SMILES と名前を読み取る"""
    with open(smi_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else "mol"
            return smiles, name
    raise ValueError(f"No valid SMILES found in {smi_path}")


def build_template_mol(smiles):
    """SMILES から H 付きテンプレート分子を作る"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # 力場計算用に H を付加
    Chem.SanitizeMol(mol)
    return mol


def count_heavy_atoms(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def pick_main_fragment(pdb_mol, template_heavy):
    """
    PDB 分子をフラグメント分解して、
    テンプレートの重原子数に最も近いフラグメントを返す。
    """
    frags = Chem.GetMolFrags(pdb_mol, asMols=True, sanitize=False)
    if len(frags) == 1:
        frag = frags[0]
        Chem.SanitizeMol(frag)
        return frag

    best_frag = None
    best_score = None

    for fr in frags:
        try:
            Chem.SanitizeMol(fr)
        except Exception:
            continue

        heavy = count_heavy_atoms(fr)
        score = abs(heavy - template_heavy)
        if best_score is None or score < best_score:
            best_score = score
            best_frag = fr

    return best_frag


def load_pdb_with_template(pdb_path, template, allow_partial=True):
    """
    PDB を読み込み、余計な分子を削除した上で、
    可能ならテンプレートから結合次数をコピーして Mol を返す。

    - 原子数がテンプレートと一致 → AssignBondOrdersFromTemplate
    - 原子数が異なる（欠損ありなど）→ テンプレートは使わず、そのフラグメントをそのまま返す
    """
    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if pdb_mol is None:
        print(f"[WARN] Failed to read PDB: {pdb_path}")
        return None

    pdb_mol = Chem.AddHs(pdb_mol, addCoords=True)

    template_heavy = count_heavy_atoms(template)
    main_frag = pick_main_fragment(pdb_mol, template_heavy)

    if main_frag is None:
        print(f"[WARN] Could not pick main fragment for {pdb_path}")
        return None

    n_atoms_template = template.GetNumAtoms()
    n_atoms_frag = main_frag.GetNumAtoms()

    # 原子数が一致 → テンプレートの結合次数をコピー
    if n_atoms_frag == n_atoms_template:
        try:
            mol_with_bonds = AllChem.AssignBondOrdersFromTemplate(template, main_frag)
            Chem.SanitizeMol(mol_with_bonds)
            return mol_with_bonds
        except Exception as e:
            print(f"[WARN] AssignBondOrdersFromTemplate failed for {pdb_path}: {e}")
            if not allow_partial:
                return None
            mol_with_bonds = main_frag
    else:
        # 欠損あり → 部分構造としてそのまま使う
        if allow_partial:
            print(
                f"[INFO] Atom count mismatch (template={n_atoms_template}, "
                f"frag={n_atoms_frag}) for {os.path.basename(pdb_path)}; "
                f"using fragment as-is (partial structure)."
            )
            mol_with_bonds = main_frag
        else:
            print(
                f"[WARN] Atom count mismatch for {os.path.basename(pdb_path)}, skipping."
            )
            return None

    try:
        Chem.SanitizeMol(mol_with_bonds)
    except Exception as e:
        print(f"[WARN] Sanitize failed for {pdb_path}: {e}")
        return None

    return mol_with_bonds


def has_severe_clash(mol, abs_thresh=ABS_CLASH_DIST, vdw_scale=VDW_SCALE):
    """
    非結合 heavy atom 間のクラッシュを簡易判定。
    - 結合していない heavy atom ペアについて:
        * 距離 d < abs_thresh ならクラッシュ
        * または d < vdw_scale * (r_i + r_j) ならクラッシュ
    """
    pt = Chem.GetPeriodicTable()
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # 結合しているペアはスキップするためのセット
    bonded = set()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < j:
            bonded.add((i, j))
        else:
            bonded.add((j, i))

    for i in range(n_atoms):
        ai = mol.GetAtomWithIdx(i)
        if ai.GetAtomicNum() <= 1:
            continue
        pos_i = conf.GetAtomPosition(i)
        ri = pt.GetRvdw(ai.GetAtomicNum())  # vdW radius

        for j in range(i + 1, n_atoms):
            aj = mol.GetAtomWithIdx(j)
            if aj.GetAtomicNum() <= 1:
                continue
            if (i, j) in bonded:
                continue

            pos_j = conf.GetAtomPosition(j)
            dx = pos_i.x - pos_j.x
            dy = pos_i.y - pos_j.y
            dz = pos_i.z - pos_j.z
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= 0.0:
                return True
            d = sqrt(d2)

            if d < abs_thresh:
                return True

            rj = pt.GetRvdw(aj.GetAtomicNum())
            if d < vdw_scale * (ri + rj):
                return True

    return False


def calc_energy_with_relax(mol, maxIts=MINIMIZE_STEPS):
    """
    1コンフォマーだけ入った Mol を受け取り、
    MMFF（使えなければ UFF）で
      - 最小化前エネルギー E_raw
      - 最小化後エネルギー E_min
    を返す。
    戻り値: (method, E_raw, E_min)
    """
    conf_id = 0

    # まず MMFF を試す
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        e_raw = ff.CalcEnergy()
        ff.Minimize(maxIts=maxIts)
        e_min = ff.CalcEnergy()
        method = "MMFF"
        return method, e_raw, e_min

    # ダメなら UFF
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        e_raw = ff.CalcEnergy()
        ff.Minimize(maxIts=maxIts)
        e_min = ff.CalcEnergy()
        method = "UFF"
        return method, e_raw, e_min
    except Exception as e:
        print(f"[WARN] UFF failed: {e}")
        return None, None, None


# ----------------- メイン -----------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter PDB conformers using RDKit FF energies and clash checks."
    )
    parser.add_argument("--smi", required=True, help=".smi file containing SMILES")
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files (1 conf per file)")
    parser.add_argument("--out_all", default="energies_all.txt", help="Output text file (all)")
    parser.add_argument("--out_good", default="energies_filtered.txt", help="Output text file (filtered good only)")

    args = parser.parse_args()

    # 1) SMILES 読み込み & テンプレート作成
    smiles, name = read_smiles_from_smi(args.smi)
    print(f"[INFO] SMILES: {smiles} (name: {name})")

    template = build_template_mol(smiles)
    n_atoms_template = template.GetNumAtoms()
    template_heavy = count_heavy_atoms(template)
    print(f"[INFO] Template atoms: total={n_atoms_template}, heavy={template_heavy}")

    # 2) PDB ファイル一覧
    pdb_pattern = os.path.join(args.pdb_dir, "*.pdb")
    pdb_files = sorted(glob.glob(pdb_pattern))
    if not pdb_files:
        print(f"[ERROR] No PDB files found in {args.pdb_dir}")
        return

    print(f"[INFO] Found {len(pdb_files)} PDB files.")

    # 一時ストレージ: 全結果
    # (fname, status, reason, method, E_raw, E_min, e_per_atom, n_heavy)
    all_results = []

    # 3) 各 PDB について処理
    for pdb_path in pdb_files:
        fname = os.path.basename(pdb_path)
        print(f"[INFO] Processing {fname} ...")

        status = "OK"
        reason = ""

        mol = load_pdb_with_template(pdb_path, template, allow_partial=True)
        if mol is None:
            status = "FAIL"
            reason = "load_or_template_error"
            all_results.append((fname, status, reason, None, None, None, None, None))
            print(f"[WARN] Skipping {fname} due to loading/template error.")
            continue

        n_heavy = count_heavy_atoms(mol)

        # 幾何クラッシュチェック
        if has_severe_clash(mol):
            status = "BAD"
            reason = "severe_clash"
            all_results.append((fname, status, reason, None, None, None, None, n_heavy))
            print(f"[WARN] Severe clash detected in {fname}. Marked as BAD.")
            continue

        # 力場エネルギー（軽い最小化込み）
        method, e_raw, e_min = calc_energy_with_relax(mol)
        if method is None:
            status = "FAIL"
            reason = "ff_error"
            all_results.append((fname, status, reason, None, None, None, None, n_heavy))
            print(f"[WARN] Energy calculation failed for {fname}.")
            continue

        delta_relax = e_raw - e_min
        if delta_relax > DELTA_E_RELAX_MAX:
            status = "BAD"
            reason = f"too_large_relax({delta_relax:.1f})"

        e_per_atom = e_min / n_heavy if n_heavy > 0 else None

        all_results.append(
            (fname, status, reason, method, e_raw, e_min, e_per_atom, n_heavy)
        )
        print(f"    -> {method} E_raw={e_raw:.2f}, E_min={e_min:.2f}, "
              f"per_atom={e_per_atom:.3f}, status={status}")

    # 4) GOOD 候補の中でエネルギー外れ値をカット
    #    まず "OK" のものだけから e_min/atom の最小値を求める
    ok_e_per_atom = [r[6] for r in all_results if r[1] == "OK" and r[6] is not None]

    if not ok_e_per_atom:
        print("[WARN] No 'OK' molecules to compute energy threshold. "
              "Filtered output will be empty.")
        e_min_per_atom = None
    else:
        e_min_per_atom = min(ok_e_per_atom)
        print(f"[INFO] min(E_min/atom) among OK = {e_min_per_atom:.3f} kcal/mol")

    # GOOD/ BAD を最終確定
    final_results = []
    for (fname, status, reason, method, e_raw, e_min, e_per_atom, n_heavy) in all_results:
        label = "BAD"
        final_reason = reason

        if status == "FAIL":
            label = "BAD"
            if not final_reason:
                final_reason = "fail"
        elif status == "BAD":
            label = "BAD"
            if not final_reason:
                final_reason = "pre_filter_bad"
        elif status == "OK":
            # エネルギー外れ値チェック
            if e_min_per_atom is None or e_per_atom is None:
                label = "BAD"
                final_reason = "no_energy_baseline"
            else:
                delta_e_per_atom = e_per_atom - e_min_per_atom
                if delta_e_per_atom <= DELTA_E_PER_ATOM_MAX:
                    label = "GOOD"
                    final_reason = "within_energy_threshold"
                else:
                    label = "BAD"
                    final_reason = f"high_energy(delta_per_atom={delta_e_per_atom:.2f})"

        final_results.append(
            (fname, label, final_reason, method, e_raw, e_min, e_per_atom, n_heavy)
        )

    # 5) 出力: 全体 & GOOD のみ
    with open(args.out_all, "w") as f_all:
        f_all.write("# file_name\tlabel\treason\tmethod\tE_raw\tE_min\tE_min_per_atom\tn_heavy\n")
        for r in final_results:
            fname, label, reason, method, e_raw, e_min, e_per_atom, n_heavy = r
            f_all.write(
                f"{fname}\t{label}\t{reason}\t"
                f"{method or 'NA'}\t"
                f"{'' if e_raw is None else f'{e_raw:.6f}'}\t"
                f"{'' if e_min is None else f'{e_min:.6f}'}\t"
                f"{'' if e_per_atom is None else f'{e_per_atom:.6f}'}\t"
                f"{'' if n_heavy is None else n_heavy}\n"
            )

    with open(args.out_good, "w") as f_good:
        f_good.write("# file_name\tmethod\tE_min\tE_min_per_atom\tn_heavy\n")
        for r in final_results:
            fname, label, reason, method, e_raw, e_min, e_per_atom, n_heavy = r
            if label != "GOOD":
                continue
            f_good.write(
                f"{fname}\t{method}\t{e_min:.6f}\t{e_per_atom:.6f}\t{n_heavy}\n"
            )

    n_good = sum(1 for r in final_results if r[1] == "GOOD")
    print(f"[INFO] Finished. GOOD={n_good}, total={len(final_results)}")
    print(f"[INFO] Wrote all results to {args.out_all}")
    print(f"[INFO] Wrote GOOD-only results to {args.out_good}")


if __name__ == "__main__":
    main()
