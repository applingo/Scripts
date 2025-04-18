#!/usr/bin/env python3
"""
chain_clash_remover.py
======================
Detect and remove clashing atoms (or whole residues) from a **target** PDB file
by comparing it against a **reference** PDB file with a matching chain layout.

The script reads two PDB files, measures all inter‑atomic distances between
corresponding chains, and drops the atoms/residues in the target structure
that lie closer than a user‑defined cut‑off (default 2.0 Å) to any atom in the
reference.

Example
-------
$ python chain_clash_remover.py \
        --ref   file1.pdb \
        --target file2.pdb \
        --out    file2_cleaned.pdb \
        --cutoff 2.0 \
        --ignore-h \
        --remove-level residue

Requirements
------------
* Biopython ≥ 1.79
* NumPy (indirectly used by Biopython)

Install with:
    pip install biopython
"""

import argparse
import sys
from pathlib import Path
from typing import Set, Dict, List

from Bio.PDB import PDBParser, NeighborSearch, Select, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_hydrogen(atom: Atom) -> bool:
    """Return True if *atom* represents a hydrogen (H or D)."""
    element = atom.element.strip().upper()
    return element in {"H", "D"}


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def detect_clashes(ref_chain: Chain, tgt_chain: Chain, cutoff: float, ignore_h: bool,
                   residue_level: bool):
    """Return sets of atoms or residues in *tgt_chain* that clash with *ref_chain*.

    Parameters
    ----------
    ref_chain : Biopython Chain (reference)
    tgt_chain : Biopython Chain (target)
    cutoff    : float, Å distance defining a clash
    ignore_h  : bool, skip hydrogens when *searching* and *marking*
    residue_level : bool, if True mark entire residues else individual atoms
    """
    # Filter atoms for the NeighborSearch object (reference)
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
        # Any reference atom within cutoff?
        close = ns.search(atom.coord, cutoff, level='A')
        if close:
            if residue_level:
                residues_to_remove.add(atom.get_parent())
            else:
                atoms_to_remove.add(atom)

    return atoms_to_remove, residues_to_remove


class ClashSelect(Select):
    """PDBIO Select subclass that skips marked residues/atoms."""

    def __init__(self, atoms_to_skip: Set[Atom], residues_to_skip: Set[Residue]):
        super().__init__()
        self._atoms_skip = atoms_to_skip
        self._residues_skip = residues_to_skip

    def accept_atom(self, atom):
        if atom in self._atoms_skip or atom.get_parent() in self._residues_skip:
            return 0
        return 1


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Remove clashing atoms/residues from target PDB compared to reference PDB.")
    parser.add_argument("--ref", required=True, help="Reference PDB file")
    parser.add_argument("--target", required=True, help="Target PDB file to clean")
    parser.add_argument("--out", default="file2_cleaned.pdb", help="Output filename")
    parser.add_argument("--cutoff", type=float, default=2.0, help="Distance cutoff in Å to define a clash (default: 2.0 Å)")
    parser.add_argument("--ignore-h", action="store_true", help="Ignore hydrogen atoms when detecting clashes")
    parser.add_argument("--remove-level", choices=["atom", "residue"], default="residue",
                        help="Remove individual atoms or whole residues that clash (default: residue)")

    args = parser.parse_args(argv)

    parser1 = PDBParser(QUIET=True)
    parser2 = PDBParser(QUIET=True)

    ref_struct = parser1.get_structure("ref", args.ref)
    tgt_struct = parser2.get_structure("tgt", args.target)

    # Collect chains present in both structures
    ref_chains: Dict[str, Chain] = {c.id: c for c in ref_struct.get_chains()}
    tgt_chains: Dict[str, Chain] = {c.id: c for c in tgt_struct.get_chains()}

    common_chain_ids = sorted(set(ref_chains).intersection(tgt_chains))
    if not common_chain_ids:
        sys.exit("No common chain IDs between reference and target structures.")

    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    for cid in common_chain_ids:
        ref_chain = ref_chains[cid]
        tgt_chain = tgt_chains[cid]

        a_rem, r_rem = detect_clashes(ref_chain, tgt_chain, args.cutoff,
                                      args.ignore_h, args.remove_level == "residue")
        atoms_to_remove.update(a_rem)
        residues_to_remove.update(r_rem)

    # Write cleaned PDB
    io = PDBIO()
    io.set_structure(tgt_struct)
    io.save(args.out, select=ClashSelect(atoms_to_remove, residues_to_remove))

    removed = (f"{len(residues_to_remove)} residues" if args.remove_level == "residue" else
               f"{len(atoms_to_remove)} atoms")
    print(f"[INFO] Removed {removed} across {len(common_chain_ids)} chain(s). Saved cleaned structure to '{args.out}'.")


if __name__ == "__main__":
    main()