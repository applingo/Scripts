#!/usr/bin/env python
"""
flip_sampler.py  ───────────────────────────────────────────────────────────────
Generate every combination of exocyclic‑substituent flips (±180°) for a cyclic
peptide that may contain non‑canonical residues.

• Input  : PDB file whose residue names may include lowercase letters.
• Output : One PDB per flip‑combination, written to ./flipped_models/.
            File names encode which residues were flipped.

How we decide a “flippable” bond
────────────────────────────────
For each residue:
   1. Does it contain at least one ring of size 3‑6?            → Ring atoms R
   2. Does some atom in R have a non‑hydrogen neighbour that is not in R?
      ( = an exocyclic substituent root S )                     → eligible
When those conditions are met, the dihedral (P‑R‑S‑F) is defined as:
   - **R** : the first ring atom that owns an exocyclic neighbour
   - **P** : a different ring atom directly bonded to R (any one)
   - **S** : the exocyclic neighbour itself (root)
   - **F** : the heaviest atom bonded to S that is *not* R        (far end)

Flipping means rotating that dihedral by +180 °.  Combinations are the cartesian
product of {no‑flip, flip} across all eligible residues → 2^N PDB files.

Usage
-----
$ python flip_sampler.py input.pdb           # writes ./flipped_models/*.pdb
$ python flip_sampler.py input.pdb --angle -90  # choose alternative angle

Requires: RDKit 2024.03+  (conda install -c conda-forge rdkit)
"""

from __future__ import annotations
import argparse
import itertools
import pathlib
from typing import Dict, List, NamedTuple, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

# ────────────────────────────────────────────────────────────────────────────────
class FlipSite(NamedTuple):
    residkey: str              # e.g. "A45" (chain + residue number)
    atom_p: int                # preceding ring atom
    atom_r: int                # ring atom bearing the substituent
    atom_s: int                # exocyclic root (bond from ring)
    atom_f: int                # far atom defining rota table side

# ────────────────────────────────────────────────────────────────────────────────

def find_flip_sites(mol: Chem.Mol) -> List[FlipSite]:
    """Return a list of FlipSite for which a 180° flip will alter an exocyclic
    substituent orientation around a 3‑6‑membered ring in *mol*."""
    ringinfo = mol.GetRingInfo()
    sites: List[FlipSite] = []

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        residkey = f"{info.GetChainId().strip()}{info.GetResidueNumber()}"

        # Consider this atom only if it belongs to a 3‑,4‑,5‑,or 6‑membered ring
        if not any(atom.IsInRingSize(n) for n in (3, 4, 5, 6)):
            continue

        # neighbours classified as exocyclic roots (S)  = non‑H outside the ring
        for nbor in atom.GetNeighbors():
            if nbor.GetSymbol() == "H":
                continue
            if nbor.IsInRing():
                continue

            # define P (another ring neighbour)
            ring_neighbours = [x for x in atom.GetNeighbors() if x.GetIdx() != nbor.GetIdx()]
            if not ring_neighbours:
                continue  # safety
            atom_p = ring_neighbours[0].GetIdx()

            # define F  (heaviest atom bonded to S that is not R)
            candidates_f = [x for x in nbor.GetNeighbors() if x.GetIdx() != atom.GetIdx()]
            if not candidates_f:
                continue  # substituent is a single atom – flipping moot
            atom_f = max(candidates_f, key=lambda x: x.GetAtomicNum()).GetIdx()

            sites.append(
                FlipSite(residkey, atom_p, atom.GetIdx(), nbor.GetIdx(), atom_f)
            )
            break  # one site per residue is usually enough

    # Deduplicate by residue (some residues may have several rings)
    unique: Dict[str, FlipSite] = {}
    for s in sites:
        unique.setdefault(s.residkey, s)
    return list(unique.values())

# ────────────────────────────────────────────────────────────────────────────────

def generate_models(mol: Chem.Mol, sites: List[FlipSite], angle: float, outdir: pathlib.Path):
    """Write PDBs for every flip combination (2^N) defined by *sites*.

    *angle* is applied *additively* to the current dihedral value.
    File naming scheme:  base_Fxxx_Fyyy.pdb  where xxx,yyy are flipped residues."""
    outdir.mkdir(exist_ok=True)
    n = len(sites)
    if n == 0:
        print("No flippable sites detected – nothing to do.")
        return

    print(f"Found {n} flippable residues (→ {2**n} combinations). Writing to {outdir}/")

    base_name = "model"
    for bits in itertools.product([0, 1], repeat=n):
        # copy mol for this combination
        newmol = Chem.Mol(mol)
        conf = newmol.GetConformer()
        flipped_residues = []

        for bit, site in zip(bits, sites):
            if bit == 1:
                # current dihedral plus angle
                rdMT.SetDihedralDeg(conf, site.atom_p, site.atom_r, site.atom_s, site.atom_f,
                                     rdMT.GetDihedralDeg(conf, site.atom_p, site.atom_r,
                                                         site.atom_s, site.atom_f) + angle)
                flipped_residues.append(site.residkey)

        tag = "_".join(f"F{r}" for r in flipped_residues) if flipped_residues else "NF"  # NF = no flip
        outfile = outdir / f"{base_name}_{tag}.pdb"
        Chem.MolToPDBFile(newmol, str(outfile), flavor=0)

# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pdb", help="input PDB file containing the cyclic peptide")
    ap.add_argument("--angle", type=float, default=180.0,
                    help="rotation to apply in degrees (default: 180)")
    ap.add_argument("--outdir", default="flipped_models", help="output directory")
    args = ap.parse_args()

    mol = Chem.MolFromPDBFile(args.pdb, removeHs=False)
    if mol is None:
        raise SystemExit("Unable to read PDB – check file path and format")

    sites = find_flip_sites(mol)
    generate_models(mol, sites, args.angle, pathlib.Path(args.outdir))

if __name__ == "__main__":
    main()
