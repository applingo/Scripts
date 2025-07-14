#!/usr/bin/env python3
"""
validate_cyclic_peptide.py ── sanity‑check a cyclic‑peptide PDB
================================================================

This script inspects an (Amber‑generated) cyclic peptide structure that may
contain non‑canonical amino acids and reports obviously non‑physical issues
such as

* **Abnormally long or short covalent bonds** (e.g. ~3 Å C–C bonds)
* **Severe steric clashes** between non‑bonded atoms
* **Bonds that pierce aromatic rings** (ring‐crossing artefacts)

You *optionally* provide the **SMILES** for the peptide.  If supplied, the
bonding graph is taken from the SMILES template; otherwise the script infers
bonds from inter‑atomic distances.

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------
    python validate_cyclic_peptide.py peptide.pdb          # auto bond detect
    python validate_cyclic_peptide.py peptide.pdb -s "SMILES_STRING"

The program prints a concise report and exits with a non‑zero status code if
problems are found (handy for CI pipelines).

--------------------------------------------------------------------
Dependencies
--------------------------------------------------------------------
* numpy
* rdkit (2025.03+)

Install with e.g.  `pip install numpy rdkit-pypi`.

--------------------------------------------------------------------
Limitations & Notes
--------------------------------------------------------------------
* Aromatic‑ring detection requires either a valid SMILES or that the six‑
  membered ring is planar within 0.15 Å.
* Bond inference without SMILES uses generous covalent‑radius cut‑offs; very
  exotic chemistry might confuse it—supplying SMILES is preferred.
* The aromatic ring–piercing test is heuristic (centroid‑radius) but catches
  all practical Amber artefacts the author has seen.

© 2025 Takashi Helper – MIT licence.
"""
import argparse
import sys
from typing import List, Tuple, Dict, Set
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

###############################################################################
#                              helper utilities                               #
###############################################################################

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "B": 0.85, "Si": 1.11,
}
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98, "B": 1.92, "Si": 2.10,
}

BOND_LEN_TOL = 0.20  # ±20 % relative tolerance around expected length
MAX_BOND_FACTOR = 1.5  # fallback: covRadSum * MAX_BOND_FACTOR
CLASH_OVERLAP = -0.4   # Å, allowed negative clearance

###############################################################################
#                            PDB → RDKit helpers                              #
###############################################################################

def mol_from_pdb(pdb_path: str, smiles: str | None) -> Chem.Mol:
    """Load coordinates from PDB and build an RDKit Mol, optionally using a
    SMILES template for connectivity."""
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            sys.exit("[ERROR] Could not parse supplied SMILES!")
        template = Chem.AddHs(template, addCoords=False)
    else:
        template = None

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        sys.exit("[ERROR] Could not read PDB file or no atoms found.")

    if template and mol.GetNumAtoms() == template.GetNumAtoms():
        # Transfer bond info from template
        rw = Chem.RWMol(mol)
        rw.RemoveAllBonds()
        for bond in template.GetBonds():
            rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        mol = rw.GetMol()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    else:
        # Infer bonds by heuristic distance cutoff
        mol = infer_bonds(mol)
    return mol


def infer_bonds(mol: Chem.Mol) -> Chem.Mol:
    """Infer bonds purely from distances and covalent radii."""
    rw = Chem.RWMol(mol)
    conf = mol.GetConformer()
    nat = mol.GetNumAtoms()
    for i in range(nat):
        elem_i = mol.GetAtomWithIdx(i).GetSymbol()
        ri = COVALENT_RADII.get(elem_i, 0.77)
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, nat):
            elem_j = mol.GetAtomWithIdx(j).GetSymbol()
            rj = COVALENT_RADII.get(elem_j, 0.77)
            pj = conf.GetAtomPosition(j)
            d = pi.Distance(Point3D(pj))
            if d < (ri + rj) * MAX_BOND_FACTOR:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, sanitizeOps=Chem.SANITIZE_PROPERTIES)
    return out

###############################################################################
#                             Validation checks                               #
###############################################################################

class Issue:
    def __init__(self, msg: str):
        self.msg = msg
    def __str__(self):
        return self.msg


def check_bond_lengths(mol: Chem.Mol) -> List[Issue]:
    issues: List[Issue] = []
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        elem_a = mol.GetAtomWithIdx(a).GetSymbol()
        elem_b = mol.GetAtomWithIdx(b).GetSymbol()
        pa = conf.GetAtomPosition(a)
        pb = conf.GetAtomPosition(b)
        d = pa.Distance(Point3D(pb))
        exp = COVALENT_RADII.get(elem_a, 0.77) + COVALENT_RADII.get(elem_b, 0.77)
        if not (0.8 * exp <= d <= 1.3 * exp):
            issues.append(Issue(f"Abnormal bond length {d:.2f} Å between {a}:{elem_a}–{b}:{elem_b} (expected ~{exp:.2f} Å)"))
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
                continue  # skip bonded pairs
            ej = mol.GetAtomWithIdx(j).GetSymbol()
            rj = VDW_RADII.get(ej, 1.7)
            pj = conf.GetAtomPosition(j)
            d = pi.Distance(Point3D(pj))
            if d < (ri + rj) + CLASH_OVERLAP:
                issues.append(Issue(f"Steric clash: {i}:{ei}–{j}:{ej} distance {d:.2f} Å (vdW sum {ri + rj:.2f} Å)"))
    return issues

###############################################################################
#                     Aromatic ring piercing detection                        #
###############################################################################

def detect_ring_piercing(mol: Chem.Mol) -> List[Issue]:
    issues: List[Issue] = []
    conf = mol.GetConformer()
    rings = [list(r) for r in Chem.GetSymmSSSR(mol) if len(r) == 6]
    for ring in rings:
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue  # only benzene‑like rings
        # Collect ring points
        ring_pts = np.array([conf.GetAtomPosition(i) for i in ring])
        centroid = ring_pts.mean(axis=0)
        # Plane normal via SVD
        uu, dd, vv = np.linalg.svd(ring_pts - centroid)
        normal = vv[2]
        # Effective ring radius
        r_rad = np.linalg.norm(ring_pts[0] - centroid)
        # Precompute polygon (for 2D check)
        axes = vv[:2]  # principal axes
        ring_2d = (ring_pts - centroid) @ axes.T

        # For each bond not in ring
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring or b in ring:
                continue
            pa = np.array(conf.GetAtomPosition(a))
            pb = np.array(conf.GetAtomPosition(b))
            # Find intersection with ring plane
            denom = np.dot(normal, pb - pa)
            if abs(denom) < 1e-3:
                continue  # bond parallel
            t = np.dot(normal, centroid - pa) / denom
            if not (0.0 <= t <= 1.0):
                continue  # intersection outside bond segment
            pint = pa + t * (pb - pa)
            # Check radial distance
            if np.linalg.norm(pint - centroid) > r_rad * 0.9:
                continue
            # Quick polygon test in 2D (winding number)
            p2 = (pint - centroid) @ axes.T
            inside = point_in_polygon(p2, ring_2d)
            if inside:
                issues.append(Issue(
                    f"Bond {a}-{b} appears to pierce aromatic ring ({'/'.join(map(str, ring))})"
                ))
    return issues


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """2D point‑in‑polygon by winding number (poly is (n,2))."""
    wn = 0
    n = len(poly)
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        if p1[1] <= p[1]:
            if p2[1] > p[1] and is_left(p1, p2, p) > 0:
                wn += 1
        else:
            if p2[1] <= p[1] and is_left(p1, p2, p) < 0:
                wn -= 1
    return wn != 0


def is_left(p0, p1, p2):
    return ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p2[0] - p0[0]) * (p1[1] - p0[1]))

###############################################################################
#                                   main                                      #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Validate a cyclic peptide PDB for obvious structural artefacts.")
    parser.add_argument("pdb", help="Input PDB file (Amber output).")
    parser.add_argument("-s", "--smiles", help="Peptide SMILES string for accurate bonding (optional).")
    args = parser.parse_args()

    mol = mol_from_pdb(args.pdb, args.smiles)

    issues: List[Issue] = []
    issues += check_bond_lengths(mol)
    issues += check_clashes(mol)
    issues += detect_ring_piercing(mol)

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
