#!/usr/bin/env python3
"""
CycPep Revive
環状ペプチドのPDB修正・補完システム 完全実装例

【機能概要】
- 入力：環状ペプチドの不完全なPDBファイル（非天然アミノ酸、特殊3文字コード、欠損側鎖の可能性有）と、
         正確なSMILES文字列（完全な分子構造情報）
- 処理：
  1. 元PDB構造を読み込み（Bio.PDB）
  2. SMILESから3D構造を生成しエネルギー最適化（RDKit）
  3. SMILES由来構造を Bio.PDB の Structure として読み込み
  4. 各残基について、バックボーン（N, CA, C）の座標を元PDBから置換
  5. 補完された残基の残基名を「BRC」に変更
- 出力：補完済みのPDBファイル
"""

import sys
import os
from io import StringIO

# RDKit 関連
from rdkit import Chem
from rdkit.Chem import AllChem

# Bio.PDB 関連
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Atom import Atom

def read_pdb_file(pdb_filepath):
    """
    PDBファイルを読み込み、Bio.PDBのStructureオブジェクトとして返す。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_filepath)
    return structure

def generate_smiles_structure(smiles_string):
    """
    SMILES文字列から分子オブジェクトを生成、3D構造を作成し最適化した後、
    PDB形式の文字列に変換して、Bio.PDBのStructureオブジェクトとして返す。
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError("無効なSMILES文字列です。")
    
    # 水素の追加と3D構造生成
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        print("【WARNING】3D構造生成に警告が発生しました。")
    AllChem.UFFOptimizeMolecule(mol)
    
    pdb_block = Chem.MolToPDBBlock(mol)
    # RDKit が出力する PDB ブロックは単一チェーン "A" で出力される場合が多い
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("smiles", StringIO(pdb_block))
    return structure

def graft_backbone(orig_structure, smiles_structure):
    """
    元のPDB構造（orig_structure）とSMILES由来構造（smiles_structure）を
    残基ごとに対応付け、バックボーン原子（N, CA, C）の座標を
    元の構造から SMILES構造に移植（グラフト）する。
    その後、全残基の残基名を "BRC" に変更する。
    
    ※ 本実装例では、単一モデル・単一チェーン（例："A"）かつ
       残基順序が一致している前提としています。
    """
    # モデルの取得（先頭モデルのみを使用）
    orig_model = list(orig_structure.get_models())[0]
    smiles_model = list(smiles_structure.get_models())[0]
    
    # チェーンID "A" を前提（必要に応じて拡張してください）
    if "A" not in orig_model or "A" not in smiles_model:
        raise ValueError("チェーンID 'A' がどちらかの構造に存在しません。")
    orig_chain = orig_model["A"]
    smiles_chain = smiles_model["A"]
    
    # 残基のリスト（順序が一致していると仮定）
    orig_residues = list(orig_chain.get_residues())
    smiles_residues = list(smiles_chain.get_residues())
    
    if len(orig_residues) != len(smiles_residues):
        print("【WARNING】元PDBとSMILES構造の残基数が一致しません。単純対応できない可能性があります。")
    
    # 対応する残基間でバックボーン (N, CA, C) の座標を入れ替え
    for i, smiles_res in enumerate(smiles_residues):
        if i < len(orig_residues):
            orig_res = orig_residues[i]
            for atom in smiles_res:
                atom_name = atom.get_name().strip()
                if atom_name in ['N', 'CA', 'C']:
                    # 対応する原子を元構造から探す
                    if atom_name in orig_res:
                        orig_atom = orig_res[atom_name]
                        atom.set_coord(orig_atom.get_coord())
        # 補完された残基の名前を "BRC" に変更
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
    
    print("【INFO】SMILES文字列から3D構造生成開始")
    smiles_structure = generate_smiles_structure(smiles_string)
    print("【INFO】SMILES由来構造の生成完了")
    
    print("【INFO】バックボーン座標のグラフト処理開始")
    corrected_structure = graft_backbone(orig_structure, smiles_structure)
    print("【INFO】グラフト処理完了")
    
    # 出力
    write_structure(corrected_structure, output_filepath)

if __name__ == "__main__":
    main()
