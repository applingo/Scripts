#!/usr/bin/env python3
"""
chain_clash_remover.py
======================
Detect and remove clashing atoms (or whole residues) from a **target** PDB file
by comparing it against a **reference** PDB file with a matching chain layout.

Key change (2025‑04‑18)
----------------------
By default, *identical* chain IDs are **not** compared (e.g. Chain A vs Chain A),
because these chains usually encode the same molecule in the two files and would
produce overwhelming self‑overlap.  Instead, each chain in the **target** is
checked against **all *other* chains** in the reference structure.  You can
restore the previous behaviour with the new flag `--self-compare`.

Example
-------
$ python chain_clash_remover.py \
        --ref    file1.pdb \
        --target file2.pdb \
        --out    file2_cleaned.pdb \
        --cutoff 2.0 \
        --ignore-h \
        --remove-level residue

Optional self‑comparison:
$ python chain_clash_remover.py ... --self-compare

Requirements
------------
* Biopython ≥ 1.79 (installs NumPy as dependency)

Install with:
    pip install biopython
"""

import argparse
import sys
from pathlib import Path
from typing import Set, List, Tuple

from Bio.PDB import PDBParser, NeighborSearch, Select, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_hydrogen(atom: Atom) -> bool:
    """True if *atom* is a hydrogen (H or D)."""
    return atom.element.strip().upper() in {"H", "D"}


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def detect_clashes(ref_chain: Chain, tgt_chain: Chain, cutoff: float, ignore_h: bool,
                   residue_level: bool) -> Tuple[Set[Atom], Set[Residue]]:
    """Return atoms or residues from *tgt_chain* that sit within *cutoff* Å from
    any atom in *ref_chain*.
    """
    ref_atoms: List[Atom] = [a for a in ref_chain.get_atoms()
                             if not (ignore_h and is_hydrogen(a))]
    if not ref_atoms:
        return set(), set()

    ns = NeighborSearch(ref_atoms)
    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    for atom in tgt_chain.get_atoms():
        if ignore_h and is_hydrogen(atom):
            continue
        if ns.search(atom.coord, cutoff, level='A'):
            if residue_level:
                residues_to_remove.add(atom.get_parent())
            else:
                atoms_to_remove.add(atom)

    return atoms_to_remove, residues_to_remove


class ClashSelect(Select):
    """PDBIO Select subclass that masks marked atoms/residues."""

    def __init__(self, atoms_skip: Set[Atom], residues_skip: Set[Residue]):
        super().__init__()
        self.atoms_skip = atoms_skip
        self.residues_skip = residues_skip

    def accept_atom(self, atom):  # noqa: D401  (Biopython convention)
        if atom in self.atoms_skip or atom.get_parent() in self.residues_skip:
            return 0
        return 1


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove clashing atoms/residues from target PDB compared to reference PDB.")
    parser.add_argument("--ref", required=True, help="Reference PDB file")
    parser.add_argument("--target", required=True, help="Target PDB file to clean")
    parser.add_argument("--out", default="file2_cleaned.pdb", help="Output filename")
    parser.add_argument("--cutoff", type=float, default=2.0, help="Distance cutoff in Å (default 2.0)")
    parser.add_argument("--ignore-h", action="store_true", help="Ignore hydrogens in clash detection")
    parser.add_argument("--remove-level", choices=["atom", "residue"], default="residue",
                        help="Delete clashing 'atom' or whole 'residue' (default)")
    parser.add_argument("--self-compare", action="store_true",
                        help="Also compare identical chain IDs (e.g. A vs A). By default they are skipped.")

    args = parser.parse_args(argv)

    # Parse PDB files
    pdb_parser = PDBParser(QUIET=True)
    ref_struct = pdb_parser.get_structure("ref", Path(args.ref))
    tgt_struct = pdb_parser.get_structure("tgt", Path(args.target))

    # Accumulate items to remove
    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    tgt_chains: List[Chain] = list(tgt_struct.get_chains())
    ref_chains: List[Chain] = list(ref_struct.get_chains())

    for tgt_chain in tgt_chains:
        for ref_chain in ref_chains:
            if not args.self_compare and (tgt_chain.id == ref_chain.id):
                # Skip self‑ID clashes by default
                continue
            a_rem, r_rem = detect_clashes(ref_chain, tgt_chain, args.cutoff,
                                          args.ignore_h, args.remove_level == "residue")
            atoms_to_remove.update(a_rem)
            residues_to_remove.update(r_rem)

    # Write cleaned PDB
    io = PDBIO()
    io.set_structure(tgt_struct)
    io.save(Path(args.out), select=ClashSelect(atoms_to_remove, residues_to_remove))

    removed_desc = (f"{len(residues_to_remove)} residue(s)" if args.remove_level == "residue"
                    else f"{len(atoms_to_remove)} atom(s)")
    print(f"[INFO] Removed {removed_desc}. Saved cleaned structure to '{args.out}'.")


if __name__ == "__main__":
    main()
