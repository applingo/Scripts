#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  generate_isolde_ff.sh
#  ---------------------------------------------------------------------------
#  One‑shot script to generate an ISOLDE‑compatible force‑field description for
#  a residue named MOL that is present in a PDB structure.  The script produces:
#     * <basename>.mol2  – GAFF2 atom‑typed molecule with AM1‑BCC charges
#     * <basename>.frcmod – any missing GAFF2 parameters
#     * <basename>.ffxml  – OpenMM/ISOLDE reusable residue template
#     * <basename>_system.xml + <basename>_clean.pdb – ready‑to‑simulate system
#
#  Requirements (AmberTools 25 or later installed via conda is assumed):
#     • pdb4amber, antechamber, parmchk2, mdgx (AmberTools)
#     • ParmEd (conda‑forge parmed)
#     • OpenMM (for ISOLDE)
#     • Quick GPU module (optional, boosts AM1‑BCC on NVIDIA GPUs)
#
#  Parallel/GPU acceleration 
#     • Set NCPU   – no. of CPU threads (default: all cores)
#     • Set QUICK_CUDA=1 before running if you have quick.cuda in your PATH.
#       Otherwise the script detects quick.cuda automatically.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pdbfile> [RESNAME]" >&2
  echo "       <pdbfile>  – input structure containing the residue" >&2
  echo "       [RESNAME]  – residue name to parameterise (default: MOL)" >&2
  exit 1
fi

PDB=$1               # original PDB file
RES=${2:-MOL}        # residue name to treat (defaults to MOL)
BASE=$(basename "${PDB%.*}")_${RES}  # prefix for outputs, e.g. ligand_MOL

# ---------------------------------------------------------------------------
# 1.  Thread / GPU detection
# ---------------------------------------------------------------------------
NCPU=${NCPU:-$(nproc || sysctl -n hw.logicalcpu)}
export OMP_NUM_THREADS=$NCPU
export QUICK_THREADS=$NCPU               # quick.cuda uses this variable

if command -v quick.cuda >/dev/null 2>&1 ; then
  export QUICK_CUDA=1                    # tell Amber’s sqm to call Quick GPU
  printf "[Info] Quick GPU detected – AM1‑BCC will use GPU (%d threads)\n" "$NCPU"
else
  unset QUICK_CUDA                       # fall back to CPU
  printf "[Info] Quick GPU not found – AM1‑BCC will run on CPU (%d threads)\n" "$NCPU"
fi

# ---------------------------------------------------------------------------
# 2.  Clean PDB (remove altlocs, add hydrogens, renumber nicely)
# ---------------------------------------------------------------------------
printf "[Step] Cleaning PDB with pdb4amber…\n"
pdb4amber -i "$PDB" \
          -o "${BASE}_clean.pdb" \
          --nohyd  \
          --reduce  # auto‑protonate via Reduce

# ---------------------------------------------------------------------------
# 3.  antechamber – GAFF2 atom‑typing & AM1‑BCC charges (parallel/GPU aware)
# ---------------------------------------------------------------------------
printf "[Step] antechamber GAFF2 + AM1‑BCC (this may take a while)…\n"
antechamber -i "${BASE}_clean.pdb" \
            -fi pdb \
            -o "${BASE}.mol2" \
            -fo mol2 \
            -at gaff2 \
            -c bcc \
            -s 2 \
            -ncpu "$NCPU"

# ---------------------------------------------------------------------------
# 4.  parmchk2 – fill in any GAFF2 gaps
# ---------------------------------------------------------------------------
printf "[Step] parmchk2 missing‑parameter scan…\n"
parmchk2 -i "${BASE}.mol2" -f mol2 -o "${BASE}.frcmod" -ncpu "$NCPU"

# ---------------------------------------------------------------------------
# 5.  mdgx – build reusable ffxml (+ .lib) for OpenMM/ISOLDE
# ---------------------------------------------------------------------------
printf "[Step] mdgx → ffxml generation…\n"
cat > mdgx_ffxml.in <<EOF
&files
  -par $BASE
  -mol ${BASE}.mol2
  -frcmod ${BASE}.frcmod
  -ffxml  ${BASE}.ffxml
&end
&param
  forcefield = gaff2,
  charge     = bcc,
&end
EOF
mdgx -i mdgx_ffxml.in

# ---------------------------------------------------------------------------
# 6.  ParmEd – create single‑system XML for instant ISOLDE drag‑and‑drop
# ---------------------------------------------------------------------------
printf "[Step] ParmEd – writing OpenMM System XML…\n"
python - <<PY
import parmed as pmd, os
mol2 = "${BASE}.mol2"; frc = "${BASE}.frcmod"
parm = pmd.load_file(mol2, frc)           # build topology in‑memory
parm.save(f"{mol2}.prmtop", overwrite=True)
parm.save(f"{mol2}.inpcrd", overwrite=True)
parm = pmd.load_file(f"{mol2}.prmtop", f"{mol2}.inpcrd")
parm.save("${BASE}_system.xml", format="xml", overwrite=True)
PY

# ---------------------------------------------------------------------------
# 7.  Summary
# ---------------------------------------------------------------------------
cat <<EOF

------------------------------------------------------------
Finished!  Key artefacts for ISOLDE:
   • ${BASE}.ffxml            (re‑usable residue template)
   • ${BASE}_system.xml       + ${BASE}_clean.pdb  (ready‑to‑run system)
Optional usage inside ChimeraX/ISOLDE:
   isolde loadResidueParameters ${BASE}.ffxml      # registers MOL template
   isolde sim ${BASE}_system.xml ${BASE}_clean.pdb # starts MD immediately
------------------------------------------------------------
EOF
