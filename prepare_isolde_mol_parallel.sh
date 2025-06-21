#!/usr/bin/env bash
############################################################################################################
# prepare_isolde_mol_parallel.sh
#
# Generate force‑field files for a small‑molecule residue called “MOL” so it can be simulated in ISOLDE,
# but with **CPU multi‑threading and (optional) GPU acceleration** to shorten the charge‑derivation step.
#
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────
# ❶  Environment setup (do **once**):
#     ▸ CPU‑only build via conda (OpenMP‑enabled):
#         conda create -n ambertools -c conda-forge ambertools=25 openmm
#         conda activate ambertools
#
#     ▸ (Optional) GPU build for `mdgx` (requires NVIDIA GPU, CUDA ≥ 11, compute capability ≥ 6.0):
#         git clone https://github.com/ambermd/amber.git
#         cd amber
#         ./configure -cuda gnu              # or -cuda clang / intel if preferred
#         make -j$(nproc)
#         echo 'export AMBERHOME=$PWD' >> ~/.bashrc
#         echo 'export PATH=$AMBERHOME/bin:$PATH' >> ~/.bashrc
#         source ~/.bashrc
#
# ❷  Usage
#        ./prepare_isolde_mol_parallel.sh mol.pdb  <threads>  [bcc|abcg2]  [--mdgx]
#
#        mol.pdb     – input PDB with residue name MOL
#        <threads>   – number of CPU threads to use (OpenMP). 1 ⇒ no threading.
#        bcc|abcg2   – AM1‑BCC (default) or ABCG2 charges
#        --mdgx      – (optional) route B: build GAFF2 *.ffxml + *.lib via mdgx (GPU‑aware)
#
# ❸  Outputs
#        Route A (default)  ─▶  mol.xml                     (OpenMM System XML)
#        Route B (--mdgx)   ─▶  mol.ffxml  and  mol.lib     (for ISOLDE)
#
# ❹  Example (12‑thread CPU run, AM1‑BCC, Route‑A)
#        ./prepare_isolde_mol_parallel.sh mol.pdb 12 bcc
#
#     Example (GPU id 0 for mdgx, ABCG2, Route‑B)
#        CUDA_VISIBLE_DEVICES=0 ./prepare_isolde_mol_parallel.sh mol.pdb 1 abcg2 --mdgx
#
############################################################################################################
set -euo pipefail

# ---------------------------- argument parsing -----------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 mol.pdb <threads> [bcc|abcg2] [--mdgx]" >&2
    exit 1
fi

PDB_IN="$1"
THREADS="$2"
CHARGE="${3:-bcc}"
ROUTE="${4:-tleap}"   # internal flag: 'tleap' or '--mdgx'

if [[ ! -f "$PDB_IN" ]]; then
    echo "ERROR: file '$PDB_IN' not found." >&2; exit 1; fi

[[ "$CHARGE" =~ ^(bcc|abcg2)$ ]] || { echo "Charge must be 'bcc' or 'abcg2'." >&2; exit 1; }

BASE="${PDB_IN%.*}"
export OMP_NUM_THREADS="$THREADS"            # OpenMP threads used by `sqm`
export MKL_NUM_THREADS="$THREADS"            # if MKL is present
LOG="${BASE}_timings.log"

echo "==================================================================" | tee  "$LOG"
echo "  Parallel MOL build started  : $(date)" | tee -a "$LOG"
echo "  Threads (OMP)               : $OMP_NUM_THREADS" | tee -a "$LOG"
echo "  Charge model                : $CHARGE" | tee -a "$LOG"
echo "  Route                       : $ROUTE" | tee -a "$LOG"
echo "==================================================================" | tee -a "$LOG"

timer () {    # helper for timing blocks
    local label="$1"; shift
    echo -e "\n>>> $label" | tee -a "$LOG"
    /usr/bin/time -f "ELAPSED(s)	%e" -o tmp.time "$@"
    cat tmp.time | tee -a "$LOG"
    rm -f tmp.time
}

# ---------------------------- 1. PDB cleanup -------------------------------------------------------------
timer "Step 1: pdb4amber cleanup"       pdb4amber -i "$PDB_IN" -o "${BASE}_clean.pdb" --nohyd

# ---------------------------- 2. antechamber + sqm -------------------------------------------------------
timer "Step 2: antechamber + sqm (GAFF2, $CHARGE)"       antechamber -i "${BASE}_clean.pdb" -fi pdb                   -o "${BASE}.mol2"      -fo mol2                   -s 2                   -at gaff2                   -c "$CHARGE"           -nc 0                   -rn MOL

timer "Step 3: parmchk2 (missing types)"       parmchk2 -i "${BASE}.mol2" -f mol2 -o "${BASE}.frcmod" -s gaff2

# ---------------------------- 3A. Route A – TLEaP + ParmEd -----------------------------------------------
if [[ "$ROUTE" != "--mdgx" ]]; then
    cat > tleap.in <<EOF
source leaprc.gaff2
loadmol2  ${BASE}.mol2
loadamberparams ${BASE}.frcmod
saveamberparm  MOL  ${BASE}.prmtop ${BASE}.inpcrd
quit
EOF
    timer "Step 4A: tleap build prmtop/inpcrd" tleap -f tleap.in

    cat > parmed.in <<EOF
parm ${BASE}.prmtop
loadRestrt ${BASE}.inpcrd
outparm ${BASE}.xml
quit
EOF
    timer "Step 5A: ParmEd convert to OpenMM XML" parmed -O -i parmed.in
    FINAL_MSG="Created ${BASE}.xml (OpenMM System XML)"
else
# ---------------------------- 3B. Route B – mdgx GPU/CPU -----------------------------------------------
    # mdgx control file (GAFF2, ffxml/lib export)
    cat > ${BASE}_mdgx.in <<EOF
&files
  -parmout      ${BASE}.ffxml
  -mol2out      ${BASE}_reordered.mol2
  -libout       ${BASE}.lib
&end
&parameter
  -library      gaff2
  -frcmod       ${BASE}.frcmod
  -inputmol2    ${BASE}.mol2
&end
EOF
    timer "Step 4B: mdgx build ffxml/lib (GPU aware)"           mdgx -O -i ${BASE}_mdgx.in

    FINAL_MSG="Created ${BASE}.ffxml and ${BASE}.lib (GAFF2)"
fi

echo -e "\n$FINAL_MSG – ready for ISOLDE.\n" | tee -a "$LOG"
echo "Timing summary saved to $LOG"
echo "Done." | tee -a "$LOG"
