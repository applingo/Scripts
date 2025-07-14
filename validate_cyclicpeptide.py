#!/usr/bin/env python3
"""
validate_cyclic_peptide.py ── sanity‑check a cyclic‑peptide PDB
================================================================

Version **2025‑07‑14**

This script inspects a cyclic peptide structure (Amber‑generated, possibly
with non‑canonical amino acids) for obvious artefacts:

1. **Abnormally long or short covalent bonds** (vs. covalent‑radius sum)
2. **Severe steric clashes** (vdW overlap > 0.4 Å)
3. **Bonds that pierce aromatic rings** (ring‑crossing)
4. **Residue connectivity breaks** – *NEW*: if SMILES is **not** supplied,
   each residue should form a single connected component; disconnected
   fragments are flagged.

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------
    python validate_cyclic_peptide.py peptide.pdb                 # auto bond detect
    python validate_cyclic_peptide.py peptide.pdb -s "SMILES"      # use SMILES bonds

The script prints a report and exits **1** when any issue is detected, so you
can plug it into CI pipelines.

Dependencies: `numpy`, `rdkit-pypi` (2025.03+).
"""
from __future__ import annotations
import argparse
import sys
from typing import List, Dict, Set, Tuple
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

###############################################################################
# Constants
###############################################################################

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "B": 0.85, "Si": 1.11,
}
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98, "B": 1.92, "Si": 2.10,
}
BOND_LEN_FACTOR_MIN = 0.8
BOND_LEN_FACTOR_MAX = 1.3
MAX_BOND_FACTOR = 1.5  # for bond inference
CLASH_OVERLAP = -0.4   # Å

###############################################################################
# Issue wrapper
###############################################################################

class Issue:
    def __init__(self, msg: str):
        self.msg = msg
    def __str__(self):
        return self.msg

###############################################################################
# PDB → Mol helpers
###############################################################################

def mol_from_pdb(pdb_path: str, smiles: str | None) -> Chem.Mol:
    """Load PDB; if SMILES provided, use it as connectivity template."""
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            sys.exit("[ERROR] Could not parse supplied SMILES.")
        template = Chem.AddHs(template, addCoords=False)
    else:
        template = None

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        sys.exit("[ERROR] Failed to read PDB: no atoms found.")

    if template and mol.GetNumAtoms() == template.GetNumAtoms():
        rw = Chem.RWMol(mol)
        rw.RemoveAllBonds()
        for b in template.GetBonds():
            rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
        mol = rw.GetMol()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    else:
        mol = infer_bonds(mol)
    return mol


def infer_bonds(mol: Chem.Mol) -> Chem.Mol:
    """Add bonds heuristically via covalent radii."""
    rw = Chem.RWMol(mol)
    conf = mol.GetConformer()
    nat = mol.GetNumAtoms()
    for i in range(nat):
        ei = mol.GetAtomWithIdx(i).GetSymbol()
        ri = COVALENT_RADII.get(ei, 0.77)
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, nat):
            ej = mol.GetAtomWithIdx(j).GetSymbol()
            rj = COVALENT_RADII.get(ej, 0.77)
            pj = conf.GetAtomPosition(j)
            d = pi.Distance(Point3D(pj))
            if d < (ri + rj) * MAX_BOND_FACTOR:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, sanitizeOps=Chem.SANITIZE_PROPERTIES)
    return out

###############################################################################
# Validation functions
###############################################################################

def check_bond_lengths(mol: Chem.Mol) -> List[Issue]:
    issues: List[Issue] = []
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ea, eb = mol.GetAtomWithIdx(a).GetSymbol(), mol.GetAtomWithIdx(b).GetSymbol()
        pa, pb = conf.GetAtomPosition(a), conf.GetAtomPosition(b)
        d = pa.Distance(Point3D(pb))
        exp = COVALENT_RADII.get(ea, 0.77) + COVALENT_RADII.get(eb, 0.77)
        if not (BOND_LEN_FACTOR_MIN * exp <= d <= BOND_LEN_FACTOR_MAX * exp):
            issues.append(Issue(f"Abnormal bond length {d:.2f} Å between {a}:{ea}‑{b}:{eb} (expected ~{exp:.2f} Å)"))
    return issues


def check_clashes(mol: Chem.Mol) -> List[Issue]:
    issues: List[Issue] = []
    conf = mol.GetConformer()
    nat = mol.GetNumAtoms()
    for i in range(nat):
        ei = mol.GetAtomWithIdx(i).GetSymbol()
        ri = VDW_RADII.get(ei, 1.7)
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, nat):
            if mol.GetBondBetweenAtoms(i, j):
                continue
            ej = mol.GetAtomWithIdx(j).GetSymbol()
            rj = VDW_RADII.get(ej, 1.7)
            pj = conf.GetAtomPosition(j)
            d = pi.Distance(Point3D(pj))
            if d < (ri + rj) + CLASH_OVERLAP:
                issues.append(Issue(f"Steric clash {i}:{ei}‑{j}:{ej} dist {d:.2f} Å (vdW sum {ri+rj:.2f} Å)"))
    return issues

###############################################################################
# Aromatic ring piercing
###############################################################################

def detect_ring_piercing(mol: Chem.Mol) -> List[Issue]:
    issues: List[Issue] = []
    conf = mol.GetConformer()
    rings = [list(r) for r in Chem.GetSymmSSSR(mol) if len(r) == 6]
    for ring in rings:
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        pts = np.array([conf.GetAtomPosition(i) for i in ring])
        centroid = pts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(pts - centroid)
        normal = vv[2]
        r_rad = np.linalg.norm(pts[0] - centroid)
        axes = vv[:2]
        poly2d = (pts - centroid) @ axes.T
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring or b in ring:
                continue
            pa, pb = np.array(conf.GetAtomPosition(a)), np.array(conf.GetAtomPosition(b))
            denom = np.dot(normal, pb - pa)
            if abs(denom) < 1e-3:
                continue
            t = np.dot(normal, centroid - pa) / denom
            if not (0 <= t <= 1):
                continue
            pint = pa + t * (pb - pa)
            if np.linalg.norm(pint - centroid) > r_rad * 0.9:
                continue
            p2 = (pint - centroid) @ axes.T
            if point_in_polygon(p2, poly2d):
                issues.append(Issue(f"Bond {a}-{b} pierces aromatic ring ({'/'.join(map(str, ring))})"))
    return issues


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    wn = 0
    n = len(poly)
    for i in range(n):
        p1, p2 = poly[i], poly[(i+1)%n]
        if p1[1] <= p[1]:
            if p2[1] > p[1] and is_left(p1, p2, p) > 0:
                wn += 1
        else:
            if p2[1] <= p[1] and is_left(p1, p2, p) < 0:
                wn -= 1
    return wn != 0

def is_left(p0, p1, p2):
    return (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])

###############################################################################
# Residue connectivity (NEW)
###############################################################################

def check_residue_connectivity(mol: Chem.Mol) -> List[Issue]:
    """Ensure all atoms belonging to the same residue form one connected graph.
    Triggered **only** when SMILES not provided (inferred bonds)."""
    issues: List[Issue] = []
    # Build residue → atom indices mapping using PDBResidueInfo
    res_atoms: Dict[Tuple[str,int,str], List[int]] = {}
    for idx in range(mol.GetNumAtoms()):
        info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if info is None:
            continue  # fallback if no PDB info
        key = (info.GetChainId(), info.GetResidueNumber(), info.GetResidueName())
        res_atoms.setdefault(key, []).append(idx)

    for key, atoms in res_atoms.items():
        if len(atoms) <= 1:
            continue  # single‑atom residue → trivial
        # Make a subgraph of bonds within this residue
        remaining: Set[int] = set(atoms)
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                v = stack.pop()
                for neigh in [nb.GetIdx() for nb in mol.GetAtomWithIdx(v).GetNeighbors()]:
                    if neigh in remaining:
                        remaining.remove(neigh)
                        stack.append(neigh)
        if components > 1:
            chain, resnum, resname = key
            issues.append(Issue(f"Residue {resname}{resnum}{chain} is split into {components} fragments"))
    return issues

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Validate cyclic peptide PDB for obvious artefacts.")
    parser.add_argument("pdb", help="Input PDB file.")
    parser.add_argument("-s", "--smiles", help="Peptide SMILES (optional). If omitted, bonds are inferred and residue-connectivity check is enabled.")
    args = parser.parse_args()

    mol = mol_from_pdb(args.pdb, args.smiles)

    issues: List[Issue] = []
    issues.extend(check_bond_lengths(mol))
    issues.extend(check_clashes(mol))
    issues.extend(detect_ring_piercing(mol))
    if args.smiles is None:
        issues.extend(check_residue_connectivity(mol))

    if issues:
        print("Found the following potential problems:\n")
        for i, iss in enumerate(issues, 1):
            print(f" {i:2d}. {iss}")
        print("\n[FAIL] Structure appears to contain artefacts!", file=sys.stderr)
        sys.exit(1)
    else:
        print("[OK] No obvious structural artefacts detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
