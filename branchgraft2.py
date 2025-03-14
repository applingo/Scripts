#!/usr/bin/env python3
"""
CycPep Revive - 環状ペプチドのPDB修正・補完システム

【機能概要】
- 入力：
    ・不完全な環状ペプチドのPDBファイル（元構造、主にバックボーン情報を保持）
    ・完全な分子情報を示す正確なSMILES文字列
- 処理：
    1. 元PDB構造をBio.PDBで読み込み
    2. SMILES文字列からRDKitで完全な3D構造を生成・最適化
    3. Bio.PDBのSuperimposerを用いて、SMILES構造を元PDBのバックボーン（CA原子）にアライメント
    4. アライメント後、各残基ごとに、元PDBのバックボーン座標（N, CA, C, O）でSMILES構造の対応原子を上書き
    5. 全残基の残基名を「BRC」に統一
- 出力：補完済みのPDBファイル（元の環状構造を保持しながら側鎖などを補完）
"""

import sys
import os
from io import StringIO

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
    
    ※ 単一チェーン（デフォルトは "A"）であり、両構造で残基順序が一致していることを前提としています。
    """
    orig_model = list(orig_structure.get_models())[0]
    smiles_model = list(smiles_structure.get_models())[0]

    if chain_id not in orig_model or chain_id not in smiles_model:
        raise ValueError(f"チェーンID '{chain_id}' が両構造に存在しません。")

    orig_chain = orig_model[chain_id]
    smiles_chain = smiles_model[chain_id]

    # CA原子のみを用いて対応点リストを作成
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
    # sup.rms がアライメント後の RMSD
    print(f"【INFO】CA原子でのアライメントRMSD: {sup.rms:.3f} Å")
    
    # SMILES構造内の全原子に対して回転・平行移動を適用
    for atom in smiles_structure.get_atoms():
        atom.transform(sup.rotran[0], sup.rotran[1])
    
    return smiles_structure

def graft_backbone(orig_structure, smiles_structure, chain_id="A"):
    """
    アライメント済みのSMILES構造の各残基について、元構造のバックボーン原子（N, CA, C, O）の座標で上書きする。
    また、全残基の残基名を「BRC」に変更します。
    
    ※ 残基順序が一致している前提。
    """
    orig_model = list(orig_structure.get_models())[0]
    smiles_model = list(smiles_structure.get_models())[0]

    if chain_id not in orig_model or chain_id not in smiles_model:
        raise ValueError(f"チェーンID '{chain_id}' が両構造に存在しません。")
    
    orig_chain = orig_model[chain_id]
    smiles_chain = smiles_model[chain_id]

    orig_residues = list(orig_chain.get_residues())
    smiles_residues = list(smiles_chain.get_residues())
    
    if len(orig_residues) != len(smiles_residues):
        print("【WARNING】元PDBとSMILES構造の残基数が一致しません。対応は可能な範囲で行います。")
    
    # 各残基ごとにバックボーン原子の座標上書きと残基名変更
    backbone_atoms = {"N", "CA", "C", "O"}
    for i, smiles_res in enumerate(smiles_residues):
        if i < len(orig_residues):
            orig_res = orig_residues[i]
            for atom in smiles_res:
                atom_name = atom.get_name().strip()
                if atom_name in backbone_atoms and atom_name in orig_res:
                    # 元構造の対応原子座標で上書き
                    atom.set_coord(orig_res[atom_name].get_coord())
        # 補完（グラフト）対象として残基名を"BRC"に変更
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
    corrected_structure = graft_backbone(orig_structure, smiles_aligned, chain_id="A")
    print("【INFO】グラフト処理完了")
    
    # 出力
    write_structure(corrected_structure, output_filepath)

if __name__ == "__main__":
    main()
