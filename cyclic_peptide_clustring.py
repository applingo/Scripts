#!/usr/bin/env python3
"""
Cluster backbone conformations of a cyclic 11‑mer peptide and
extract N representative PDBs.

pip install "MDAnalysis>=2.9.0" scikit-learn numpy

Usage
-----
python cluster_cyclic_peptide.py \
        --indir  path/to/input_pdbs \
        --outdir path/to/representatives \
        --nclusters 10
"""
import argparse, pathlib, shutil, warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import PhiPsi
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

warnings.filterwarnings("ignore", category=UserWarning)  # noisy PDB remarks

def load_universes(pdb_dir):
    pdb_files = sorted(pathlib.Path(pdb_dir).glob("*.pdb"))
    return [mda.Universe(str(f)) for f in pdb_files], pdb_files

def backbone_phi_psi(univ):
    """Return flattened [sinφ,cosφ,sinψ,cosψ]*n_res (shape = (4*n,))."""
    sel = univ.select_atoms("backbone and name N CA C")
    if sel.n_atoms < 3: raise ValueError("Backbone atoms not found.")
    phipsi = PhiPsi(univ).run()
    angles = np.deg2rad(phipsi.angles[0])  # (n_res, 2)
    sincos = np.column_stack((np.sin(angles), np.cos(angles)))  # (n_res, 4)
    return sincos.flatten()

def feature_matrix(universes):
    feats = [backbone_phi_psi(u) for u in universes]
    return np.vstack(feats)

def kmeans_cluster(X, k, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    reps_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    return labels, reps_idx

def save_representatives(pdb_files, reps_idx, outdir):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(reps_idx, 1):
        src = pdb_files[idx]
        dst = outdir / f"rep_{i:02d}.pdb"
        shutil.copy(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="folder with conf.pdb.* files")
    ap.add_argument("--outdir", required=True, help="folder for representatives")
    ap.add_argument("--nclusters", type=int, default=10, help="number of clusters")
    args = ap.parse_args()

    universes, pdb_files = load_universes(args.indir)
    X = feature_matrix(universes)
    _, reps_idx = kmeans_cluster(X, args.nclusters)
    save_representatives(pdb_files, reps_idx, args.outdir)

    print(f"✅  Saved {len(reps_idx)} representatives to {args.outdir}")

if __name__ == "__main__":
    main()
