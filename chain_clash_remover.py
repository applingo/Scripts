#!/usr/bin/env python3
"""
chain_clash_remover.py
======================
Detect and remove clashing atoms (or whole residues) from a **target** PDB file
relative to a **reference** PDB file.

2025‑04‑18 — v1.3
-----------------
* **NEW:** `--list-chains` prints chain IDs found in *target* and *reference*
  PDBs and **exits**, useful for迅速なデバッグ (確認だけで何も削除しない)。
* Keeps `--debug-pairs`, `--verbose`, etc.

Usage examples
--------------
```bash
# 1) チェーン一覧だけ確認
python chain_clash_remover.py --target file2.pdb --ref file1.pdb --list-chains

# 2) 競合除去を実行 & ペアも表示
python chain_clash_remover.py \
    --ref file1.pdb --target file2.pdb \
    --cutoff 2.0 --ignore-h \
    --debug-pairs --verbose
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
# Clash detection
# ---------------------------------------------------------------------------

def detect_clashes(
    ref_chain: Chain,
    tgt_chain: Chain,
    cutoff: float,
    ignore_h: bool,
    max_fraction: float,
) -> Tuple[Set[Atom], Set[Residue]]:
    """Return atoms / residues in *tgt_chain* that clash with *ref_chain*."""
    ref_atoms = [a for a in ref_chain.get_atoms() if not (ignore_h and is_hydrogen(a))]
    if not ref_atoms:
        return set(), set()

    ns = NeighborSearch(ref_atoms)
    residue_hits: Dict[Residue, List[Atom]] = defaultdict(list)

    for atom in tgt_chain.get_atoms():
        if ignore_h and is_hydrogen(atom):
            continue
        if ns.search(atom.coord, cutoff, level="A"):
            residue_hits[atom.get_parent()].append(atom)

    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()

    for residue, atoms in residue_hits.items():
        n_atoms = len([a for a in residue.get_atoms() if not (ignore_h and is_hydrogen(a))])
        if n_atoms and len(atoms) / n_atoms >= max_fraction:
            residues_to_remove.add(residue)
        else:
            atoms_to_remove.update(atoms)

    return atoms_to_remove, residues_to_remove

# ---------------------------------------------------------------------------
# PDBIO helpers
# ---------------------------------------------------------------------------

class ClashSelect(Select):
    """Mask atoms/residues to be skipped and drop empty chains."""

    def __init__(self, atoms_skip: Set[Atom], residues_skip: Set[Residue]):
        super().__init__()
        self._atoms_skip = atoms_skip
        self._residues_skip = residues_skip

    def accept_atom(self, atom):  # noqa: D401
        return 0 if (atom in self._atoms_skip or atom.get_parent() in self._residues_skip) else 1

    def accept_residue(self, residue):  # noqa: D401
        if residue in self._residues_skip:
            return 0
        remaining = [a for a in residue.get_atoms() if a not in self._atoms_skip]
        return 1 if remaining else 0

    def accept_chain(self, chain):  # noqa: D401
        return 1 if any(self.accept_residue(r) for r in chain.get_residues()) else 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Remove clashing atoms/residues from target PDB compared to reference PDB.")
    p.add_argument("--ref", required=True, help="Reference PDB file")
    p.add_argument("--target", required=True, help="Target PDB file to clean")
    p.add_argument("--out", default="file2_cleaned.pdb", help="Output filename")
    p.add_argument("--cutoff", type=float, default=2.0, help="Distance cutoff in Å (default 2.0)")
    p.add_argument("--ignore-h", action="store_true", help="Ignore hydrogens when detecting clashes")
    p.add_argument("--max-fraction", type=float, default=1.0,
                   help="Fraction (0‑1) of atoms that must clash before a whole residue is removed (default 1.0)")
    p.add_argument("--self-compare", action="store_true", help="Also compare identical chain IDs (A vs A)")
    p.add_argument("--debug-pairs", action="store_true", help="Print every chain‑comparison pair evaluated")
    p.add_argument("--list-chains", action="store_true", help="List chain IDs in target and reference, then exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Per‑chain removal statistics")

    args = p.parse_args(argv)

    pdb_parser = PDBParser(QUIET=True)
    ref_struct = pdb_parser.get_structure("ref", Path(args.ref))
    tgt_struct = pdb_parser.get_structure("tgt", Path(args.target))

    # List‑only mode --------------------------------------------------------
    if args.list_chains:
        tgt_ids = [c.id for c in tgt_struct.get_chains()]
        ref_ids = [c.id for c in ref_struct.get_chains()]
        print(f"[INFO] target chains:    {' '.join(tgt_ids) if tgt_ids else '(none)'}")
        print(f"[INFO] reference chains: {' '.join(ref_ids) if ref_ids else '(none)'}")
        sys.exit(0)

    # Normal clash‑removal mode -------------------------------------------
    if not 0.0 <= args.max_fraction <= 1.0:
        sys.exit("--max-fraction must be between 0 and 1")

    atoms_to_remove: Set[Atom] = set()
    residues_to_remove: Set[Residue] = set()
    per_chain_removed: Dict[str, int] = defaultdict(int)

    tgt_chains = list(tgt_struct.get_chains())
    ref_chains = list(ref_struct.get_chains())

    for tgt_chain in tgt_chains:
        for ref_chain in ref_chains:
            if not args.self_compare and tgt_chain.id == ref_chain.id:
                continue
            if args.debug_pairs:
                print(f"[DEBUG] Comparing target Chain {tgt_chain.id} → reference Chain {ref_chain.id}")
            a_rem, r_rem = detect_clashes(ref_chain, tgt_chain, args.cutoff, args.ignore_h, args.max_fraction)
            atoms_to_remove.update(a_rem)
            residues_to_remove.update(r_rem)
            per_chain_removed[tgt_chain.id] += len(a_rem) + len(r_rem)

    if args.verbose:
        for cid in sorted(per_chain_removed):
            print(f"[INFO] Chain {cid}: removed {per_chain_removed[cid]} entities")

    io = PDBIO()
    io.set_structure(tgt_struct)
    io.save(Path(args.out), select=ClashSelect(atoms_to_remove, residues_to_remove))

    print(f"[INFO] Cleaned structure saved to '{args.out}'.")

if __name__ == "__main__":
    main()
