#!/usr/bin/env python3
"""
chain_clash_remover.py
======================
Detect and remove clashing atoms (or whole residues) from a **target** PDB file
by comparing it against a **reference** PDB file.

2025‑04‑18 — v1.1
-----------------
* **Default removal granularity changed to *atom*-level** so entire chains
  are no longer wiped just because most residues clash.
* Added verbose per‑chain clash report (`-v / --verbose`).
* Added `--max‑fraction` to switch to residue‑level removal only when more than
  a given fraction (0‒1) of atoms in the residue clash (useful for side‑chain
  pruning).
* Bug‑fix: residues/chains with zero remaining atoms are now skipped in output
  to avoid ghost chains.

Example
-------
```bash
python chain_clash_remover.py \
    --ref    file1.pdb \
    --target file2.pdb \
    --out    file2_cleaned.pdb \
    --cutoff  2.5         # Å
    --ignore-h            # skip hydrogens
    --max-fraction 0.50   # remove residue only if >50 % atoms clash
    --verbose
```

Installation
------------
```bash
pip install biopython
```
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from Bio.PDB import PDBParser, NeighborSearch, PDBIO, Select
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_hydrogen(atom: Atom) -> bool:
    """Return *True* if *atom* is a hydrogen (H/D)."""
    return atom.element.strip().upper() in {"H", "D"}


# ---------------------------------------------------------------------------
# Clash detection core
# ---------------------------------------------------------------------------

def detect_clashes(
    ref_chain: Chain,
    tgt_chain: Chain,
    cutoff: float,
    ignore_h: bool,
    max_fraction: float,
) -> Tuple[Set[Atom], Set[Residue]]:
    """Identify atoms/residues in *tgt_chain* that lie closer than *cutoff* Å to
    any atom in *ref_chain*.

    If *max_fraction* < 1, residues are only flagged when the fraction of
    clashing atoms > *max_fraction*.
    """
    ref_atoms = [a for a in ref_chain.get_atoms() if not (ignore_h and is_hydrogen(a))]
    if not ref_atoms:
        return set(), set()

    ns = NeighborSearch(ref_atoms)

    # Collect clashing atoms grouped by residue
    residue_hits: Dict[Residue, List[Atom]] = defaultdict(list)

    for atom in tgt_chain.get_atoms():
        if ignore_h and is_hydrogen(atom):
            continue
        if ns.search(atom.coord, cutoff, level="A"):
            residue_hits[atom.get_parent()].append(atom)

    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    for residue, atoms in residue_hits.items():
        if len(atoms) / len([a for a in residue.get_atoms() if not (ignore_h and is_hydrogen(a))]) >= max_fraction:
            residues_to_remove.add(residue)
        else:
            atoms_to_remove.update(atoms)

    return atoms_to_remove, residues_to_remove


class ClashSelect(Select):
    """Biopython *Select* that excludes marked atoms/residues and drops empty chains."""

    def __init__(self, atoms_skip: Set[Atom], residues_skip: Set[Residue]):
        super().__init__()
        self._atoms_skip = atoms_skip
        self._residues_skip = residues_skip

    def accept_residue(self, residue: Residue):  # noqa: D401
        if residue in self._residues_skip:
            return 0
        # If residue will have no atoms after filtering, drop it to prevent ghost TER lines
        remaining = [a for a in residue.get_atoms() if a not in self._atoms_skip]
        return 1 if remaining else 0

    def accept_atom(self, atom: Atom):  # noqa: D401
        return 0 if (atom in self._atoms_skip or atom.get_parent() in self._residues_skip) else 1

    def accept_chain(self, chain: Chain):  # noqa: D401
        # Skip chain if every residue is filtered out
        for residue in chain.get_residues():
            if self.accept_residue(residue):
                return 1
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Remove clashing atoms/residues from target PDB compared to reference PDB.")
    parser.add_argument("--ref", required=True, help="Reference PDB file")
    parser.add_argument("--target", required=True, help="Target PDB file to clean")
    parser.add_argument("--out", default="file2_cleaned.pdb", help="Output filename")
    parser.add_argument("--cutoff", type=float, default=2.0, help="Distance cutoff in Å (default 2.0)")
    parser.add_argument("--ignore-h", action="store_true", help="Ignore hydrogens when detecting clashes")
    parser.add_argument("--max-fraction", type=float, default=1.0,
                        help="Fraction (0‒1) of atoms that must clash before a whole residue is removed. "
                             "Default 1.0 (never remove whole residue)")
    parser.add_argument("--self-compare", action="store_true", help="Also compare identical chain IDs (A vs A).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per‑chain clash statistics.")

    args = parser.parse_args(argv)

    if not (0.0 <= args.max_fraction <= 1.0):
        sys.exit("--max-fraction must be between 0 and 1")

    # Parse PDBs
    pdb_parser = PDBParser(QUIET=True)
    ref_struct = pdb_parser.get_structure("ref", Path(args.ref))
    tgt_struct = pdb_parser.get_structure("tgt", Path(args.target))

    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    # Verbose statistics
    per_chain_removed: Dict[str, int] = defaultdict(int)

    tgt_chains: List[Chain] = list(tgt_struct.get_chains())
    ref_chains: List[Chain] = list(ref_struct.get_chains())

    for tgt_chain in tgt_chains:
        for ref_chain in ref_chains:
            if not args.self_compare and tgt_chain.id == ref_chain.id:
                continue
            a_rem, r_rem = detect_clashes(ref_chain, tgt_chain, args.cutoff, args.ignore_h, args.max_fraction)
            atoms_to_remove.update(a_rem)
            residues_to_remove.update(r_rem)
            per_chain_removed[tgt_chain.id] += len(a_rem) + len(r_rem)

    if args.verbose:
        for cid in sorted(per_chain_removed):
            print(f"[INFO] Chain {cid}: removed {per_chain_removed[cid]} entities (atoms + residues)")

    # Write cleaned PDB
    io = PDBIO()
    io.set_structure(tgt_struct)
    io.save(Path(args.out), select=ClashSelect(atoms_to_remove, residues_to_remove))

    print(f"[INFO] Cleaned structure saved to '{args.out}'.")


if __name__ == "__main__":
    main()

