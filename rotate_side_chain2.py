#!/usr/bin/env python3
"""
rotate_sidechains.py  ‐ side-chain 180° flip generator
（クラッシュ除外・RDKit 2023⇔2025 両対応・プロリン様自動判定付き）

使い方例
--------
    python rotate_sidechains.py cyclicpeptide.pdb          # 全組合せ
    python rotate_sidechains.py cyclicpeptide.pdb -t 2 7   # 残基 2,7 のみ
    python rotate_sidechains.py cyclicpeptide.pdb --max 50 # 上限 50 件
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rmt
from rdkit.Geometry import Point3D

# ────────────────────────── 設定 ──────────────────────────
SKIP_RESIDUES: Set[str] = {              # 常に回さない三文字コード
    "PRO", "MVA", "MAA", "SAR", "7TK",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}

# ─ RotateBond 互換ラッパ（RDKit 2023…2025）─────────────────
def _has_rotatebond() -> bool:
    return hasattr(rmt, "RotateBond") and rmt.RotateBond.__code__.co_argcount == 5

if _has_rotatebond():                       # RDKit ≤ 2024
    def rotate_bond(conf, a1, a2, ang, atom_ids):  # type: ignore
        rmt.RotateBond(conf, a1, a2, ang, atom_ids)
else:                                       # RDKit ≥ 2025
    def rotate_bond(conf, a1, a2, ang, atom_ids):
        p1, p2 = conf.GetAtomPosition(a1), conf.GetAtomPosition(a2)
        axis = p2 - p1; axis /= axis.Length()
        ux, uy, uz = axis.x, axis.y, axis.z
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c+ux*ux*(1-c),   ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                      [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),   uy*uz*(1-c)-ux*s],
                      [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]])
        for idx in atom_ids:
            v = conf.GetAtomPosition(idx) - p1
            v_rot = Point3D(*(R @ np.array([v.x, v.y, v.z])))
            conf.SetAtomPosition(idx, p1 + v_rot)

# ───────────────────── ユーティリティ ──────────────────────
def residues_by_atom(mol) -> Dict[Tuple[str,int,str], List[int]]:
    tbl: Dict[Tuple[str,int,str], List[int]] = {}
    for i, a in enumerate(mol.GetAtoms()):
        info = a.GetPDBResidueInfo()
        if info:
            key = (info.GetChainId().strip(),
                   info.GetResidueNumber(),
                   info.GetResidueName().strip())
            tbl.setdefault(key, []).append(i)
    return tbl

def atoms_beyond(mol, start: int, stop: int, res_key) -> List[int]:
    todo, seen, out = [start], {stop}, []
    while todo:
        cur = todo.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nbr.GetIdx()
            if j in seen: continue
            inf = nbr.GetPDBResidueInfo()
            k = (inf.GetChainId().strip(), inf.GetResidueNumber(), inf.GetResidueName().strip())
            if k != res_key: continue
            seen.add(j); out.append(j); todo.append(j)
    return out

def heavy_clash_exists(mol, cutoff: float) -> bool:
    conf = mol.GetConformer()
    pos = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    heavy = [i for i,a in enumerate(mol.GetAtoms()) if a.GetAtomicNum()>1]
    for ii, ai in enumerate(heavy):
        for aj in heavy[ii+1:]:
            if mol.GetBondBetweenAtoms(ai,aj): continue
            if pos[ai].Distance(pos[aj]) < cutoff: return True
    return False

# ─────────────── プロリン様厳密判定 ───────────────
def is_proline_like(mol, atom_idxs: List[int], res_key) -> bool:
    """同一残基内で N と CA (→ 望ましくは CB も) が同一環に含まれるか"""
    name2idx = {mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip(): i
                for i in atom_idxs}
    n_idx, ca_idx = name2idx.get("N"), name2idx.get("CA")
    if n_idx is None or ca_idx is None:       # そもそも backbone 欠損
        return False

    ringinfo = mol.GetRingInfo().AtomRings()
    atom_set = set(atom_idxs)
    for ring in ringinfo:
        ring_set = set(ring)
        # 同一残基に限定
        if not ring_set <= atom_set:
            continue
        if n_idx in ring_set and ca_idx in ring_set:
            return True
    return False

# ─────────────── 回転サイト抽出 ───────────────
def find_sites(mol, cutoff: float, targets: set[int]|None,
               verbose=False) -> List[Tuple[int,int,List[int]]]:

    raw: List[Tuple[int,int,List[int],int,str]] = []

    for res_key, atom_idxs in residues_by_atom(mol).items():
        _chain, resno, resname = res_key
        if targets and resno not in targets: continue

        # スキップ判定
        if (resname in SKIP_RESIDUES
            or is_proline_like(mol, atom_idxs, res_key)):
            if verbose:
                print(f"  ⚠︎  {resname}:{resno}  → skipped (proline-like)")
            continue

        # CA/CB 必須
        name2idx = {mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip(): i
                    for i in atom_idxs}
        ca, cb = name2idx.get("CA"), name2idx.get("CB")
        if ca is None or cb is None: continue

        # CA–CB 軸
        rot1 = [i for i in atom_idxs
                if i not in (ca, cb)
                and mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
                not in BACKBONE]
        if rot1:
            raw.append((ca, cb, rot1, resno, resname))

        # CB–CG/CG1 軸
        cg = name2idx.get("CG") or name2idx.get("CG1")
        if cg is not None:
            rot2 = atoms_beyond(mol, cg, cb, res_key)
            if rot2:
                raw.append((cb, cg, rot2, resno, resname))

    # クラッシュフィルタ
    sites: List[Tuple[int,int,List[int]]] = []
    for a1,a2,rot,resno,resname in raw:
        test = Chem.Mol(mol)
        rotate_bond(test.GetConformer(), a1, a2, np.pi, rot)
        if heavy_clash_exists(test, cutoff):
            if verbose:
                axis = (mol.GetAtomWithIdx(a1).GetPDBResidueInfo().GetName().strip(),
                        mol.GetAtomWithIdx(a2).GetPDBResidueInfo().GetName().strip())
                print(f"  ⚠︎  {resname}:{resno} axis {axis[0]}–{axis[1]}  → clash")
            continue
        sites.append((a1,a2,rot))
    return sites

# ─────────────────────── CLI ───────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Side-chain 180° flip generator")
    ap.add_argument("pdb", help="input PDB file")
    ap.add_argument("--clash", type=float, default=1.8,
                    help="clash cutoff Å (default 1.8)")
    ap.add_argument("--max", type=int,
                    help="max variants to output")
    ap.add_argument("-t","--target", nargs="+", metavar="N",
                    help="restrict to residue numbers")
    ap.add_argument("-v","--verbose", action="store_true",
                    help="verbose (show exclusions)")
    return ap.parse_args()

def collect_targets(raw) -> set[int]|None:
    if not raw: return None
    out:set[int]=set()
    for token in raw:
        for part in token.replace(","," ").split():
            out.add(int(part))
    return out

# ─────────────────────── main ───────────────────────
def main():
    args = parse_args()
    targets = collect_targets(args.target)
    pdb = Path(args.pdb).expanduser().resolve()
    if not pdb.is_file():
        sys.exit(f"PDB not found: {pdb}")

    mol = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
    if mol is None:
        sys.exit("RDKit failed to read PDB.")
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)

    sites = find_sites(mol, args.clash, targets, args.verbose)
    if not sites:
        sys.exit("No rotatable (clash-free) sites detected.")

    print(f"\nSelected {len(sites)} rotation site(s)"
          f" (clash {args.clash} Å"
          + (f", targets {sorted(targets)}" if targets else "") + "):")
    for i,(a1,a2,rot) in enumerate(sites):
        info = mol.GetAtomWithIdx(a1).GetPDBResidueInfo()
        axis = (info.GetName().strip(),
                mol.GetAtomWithIdx(a2).GetPDBResidueInfo().GetName().strip())
        tag  = f"{info.GetResidueName().strip()}:{info.GetResidueNumber()}"
        print(f"  {i}: {tag} axis {axis[0]}–{axis[1]} (rot {len(rot)} atom)")

    outdir = pdb.with_name(pdb.stem+"_rotamers"); outdir.mkdir(exist_ok=True)
    n = len(sites); limit = args.max or (1<<n); written = 0
    for bits in range(1<<n):
        if written >= limit: break
        var = Chem.Mol(mol); conf = var.GetConformer()
        for i,(a1,a2,rot) in enumerate(sites):
            if (bits>>i)&1: rotate_bond(conf,a1,a2,np.pi,rot)
        if heavy_clash_exists(var,args.clash): continue
        tag = format(bits,f"0{n}b")
        Chem.MolToPDBFile(var, str(outdir/f"variant_{tag}.pdb"))
        written += 1
    print(f"\n✔  {written} variant(s) written to {outdir}"
          + (f" (limited to {limit})" if args.max else ""))

if __name__ == "__main__":
    main()

