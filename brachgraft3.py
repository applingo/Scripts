#!/usr/bin/env python3
"""
CycPep Revive - 環状ペプチドのPDB修正・補完システム

【機能概要】
- 入力：
    ・不完全な環状ペプチドのPDBファイル（元構造：主にバックボーン情報を保持）
    ・完全な分子情報を示す正確なSMILES文字列
- 処理：
    1. 元PDB構造をBio.PDBで読み込み
    2. SMILES文字列からRDKitで完全な3D構造を生成・最適化
    3. Bio.PDBのSuperimposerを用いて、SMILES構造を元PDBのバックボーン（CA原子）にアライメント
    4. アライメント後、残基順序が一致していない場合は、各残基のCA原子間距離に基づいて最適な対応付けを行い、
       対応する元構造のバックボーン原子（N, CA, C, O）の座標でSMILES構造の対応原子を上書き
    5. 全補完対象残基の残基名を「BRC」に変更
- 出力：補完済みのPDBファイル（元の環状構造を保持しつつ側鎖などを補完）
"""

import sys
import os
from io import StringIO
import numpy as np

# RDKit 関連
from rdkit import Chem
from rdkit.Chem import AllChem

# Bio.PDB 関連
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Superimposer import Superimposer

def read_pdb_file(pdb_filepath):
    """
    PDBファイルを読み込み、Bio.PDBのStructureオブジェクトとして返す。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_filepath)
    return structure

def generate_smiles_structure(smiles_string):
    """
    SMILES文字列からRDKitで3D構造を生成・最適化し、
    生成したPDBブロックをBio.PDBのStructureオブジェクトとして返す。
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError("無効なSMILES文字列です。")
    
    # 水素追加と3D座標生成
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        print("【WARNING】3D構造生成に警告が発生しました。")
    AllChem.UFFOptimizeMolecule(mol)
    
    pdb_block = Chem.MolToPDBBlock(mol)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("smiles", StringIO(pdb_block))
    return structure

def align_smiles_to_original(orig_structure, smiles_structure, chain_id="A"):
    """
    元構造とSMILES由来構造をバックボーンのCA原子を用いてアライメントする。
    アライメントされたSMILES構造を返す。
    
    ※ 単一チェーン（デフォルトは "A"）であり、両構造でCA原子が存在していることを前提とします。
    """
    orig_model = list(orig_structure.get_models())[0]
    smiles_model = list(smiles_structure.get_models())[0]

    if chain_id not in orig_model or chain_id not in smiles_model:
        raise ValueError(f"チェーンID '{chain_id}' が両構造に存在しません。")

    orig_chain = orig_model[chain_id]
    smiles_chain = smiles_model[chain_id]

    orig_atoms = []
    smiles_atoms = []
    for orig_res, smiles_res in zip(orig_chain, smiles_chain):
        if "CA" in orig_res and "CA" in smiles_res:
            orig_atoms.append(orig_res["CA"])
            smiles_atoms.append(smiles_res["CA"])
    if len(orig_atoms) == 0:
        raise ValueError("CA原子が見つかりません。アライメントできません。")
    
    sup = Superimposer()
    sup.set_atoms(orig_atoms, smiles_atoms)
    print(f"【INFO】CA原子でのアライメントRMSD: {sup.rms:.3f} Å")
    
    # SMILES構造内の全原子に対して回転・平行移動を適用
    for atom in smiles_structure.get_atoms():
        atom.transform(sup.rotran[0], sup.rotran[1])
    
    return smiles_structure

def graft_backbone(orig_structure, smiles_structure, chain_id="A", threshold=3.0):
    """
    アライメント済みのSMILES構造の各残基について、元構造のバックボーン原子（N, CA, C, O）の座標で上書きする。
    残基順序が一致していない場合は、各元残基のCA原子とSMILES構造内の各残基のCA原子との距離により
    最適な対応を決定します。対応が見つかった場合、そのSMILES残基のバックボーン原子座標を元構造で上書きし、
    残基名を"BRC"に変更します。対応が見つからなかったSMILES残基についても、残基名は"BRC"に変更されます。
    """
    orig_model = list(orig_structure.get_models())[0]
    smiles_model = list(smiles_structure.get_models())[0]

    if chain_id not in orig_model or chain_id not in smiles_model:
        raise ValueError(f"チェーンID '{chain_id}' が両構造に存在しません。")
    
    orig_chain = orig_model[chain_id]
    smiles_chain = list(smiles_model[chain_id])
    
    orig_residues = list(orig_chain.get_residues())
    smiles_residues = list(smiles_chain)
    
    # CA原子の距離に基づいた対応付け（グリーディー法）
    matches = {}  # key: 元残基オブジェクト, value: SMILES残基オブジェクト
    used_smiles_indices = set()
    for orig_res in orig_residues:
        if "CA" not in orig_res:
            continue
        orig_ca = orig_res["CA"].get_coord()
        best_match_index = None
        best_distance = float('inf')
        for j, smiles_res in enumerate(smiles_residues):
            if j in used_smiles_indices:
                continue
            if "CA" not in smiles_res:
                continue
            smiles_ca = smiles_res["CA"].get_coord()
            dist = np.linalg.norm(orig_ca - smiles_ca)
            if dist < best_distance:
                best_distance = dist
                best_match_index = j
        if best_match_index is not None and best_distance <= threshold:
            matches[orig_res] = smiles_residues[best_match_index]
            used_smiles_indices.add(best_match_index)
        else:
            print(f"【WARNING】元構造の残基 {orig_res.get_id()} のCAが適切にマッチしませんでした（最小距離: {best_distance:.2f} Å）")
    
    # 各対応ペアについて、バックボーン原子 (N, CA, C, O) の座標を上書き
    backbone_atoms = {"N", "CA", "C", "O"}
    for orig_res, smiles_res in matches.items():
        for atom in smiles_res:
            atom_name = atom.get_name().strip()
            if atom_name in backbone_atoms and atom_name in orig_res:
                atom.set_coord(orig_res[atom_name].get_coord())
        # 対応があった残基の残基名を"BRC"に変更
        smiles_res.resname = "BRC"
    
    # 対応がなかったSMILES残基についても、残基名を"BRC"に変更
    for i, smiles_res in enumerate(smiles_residues):
        if i not in used_smiles_indices:
            smiles_res.resname = "BRC"
    
    return smiles_structure

def write_structure(structure, output_filepath):
    """
    Bio.PDB の PDBIO を用いて、Structure オブジェクトをファイルに出力する。
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filepath)
    print(f"【INFO】補完済みPDBファイルを出力しました: {output_filepath}")

def main():
    if len(sys.argv) != 4:
        print("使い方: python cycpep_revive.py input.pdb input_smiles.txt output_corrected.pdb")
        sys.exit(1)
    
    pdb_filepath = sys.argv[1]
    smiles_filepath = sys.argv[2]
    output_filepath = sys.argv[3]
    
    if not os.path.exists(pdb_filepath):
        print(f"【ERROR】PDBファイルが存在しません: {pdb_filepath}")
        sys.exit(1)
    if not os.path.exists(smiles_filepath):
        print(f"【ERROR】SMILESファイルが存在しません: {smiles_filepath}")
        sys.exit(1)
    
    # SMILES文字列の読み込み
    with open(smiles_filepath, "r") as f:
        smiles_string = f.read().strip()
    
    print("【INFO】元PDBファイルの読み込み開始")
    orig_structure = read_pdb_file(pdb_filepath)
    print("【INFO】元PDBファイルの読み込み完了")
    
    print("【INFO】SMILES文字列から完全構造生成開始")
    smiles_structure = generate_smiles_structure(smiles_string)
    print("【INFO】SMILES由来構造の生成・最適化完了")
    
    print("【INFO】元構造に合わせたアライメント開始")
    smiles_aligned = align_smiles_to_original(orig_structure, smiles_structure, chain_id="A")
    print("【INFO】アライメント完了")
    
    print("【INFO】バックボーン座標のグラフト処理開始")
    corrected_structure = graft_backbone(orig_structure, smiles_aligned, chain_id="A", threshold=3.0)
    print("【INFO】グラフト処理完了")
    
    # 出力
    write_structure(corrected_structure, output_filepath)

if __name__ == "__main__":
    main()
