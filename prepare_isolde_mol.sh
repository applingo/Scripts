#!/usr/bin/env bash
# ============================================================
# prepare_isolde_mol.sh
#
# Generate an OpenMM System XML of a small-molecule residue
# “MOL” for use in ISOLDE, using AmberTools 25 (GAFF2).
#
#  Usage:
#     ./prepare_isolde_mol.sh mol.pdb [bcc|abcg2]
#
#  Requirements:
#     - AmberTools 25 installed via conda (antechamber, parmchk2,
#       tleap, pdb4amber, parmed all on PATH)
# ============================================================

set -euo pipefail

# ------------- 1. Parse command-line ------------------------
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 mol.pdb [bcc|abcg2]" >&2
    exit 1
fi

PDB_IN="$1"
CHARGE_METHOD="${2:-bcc}"   # default AM1-BCC
[[ "$CHARGE_METHOD" =~ ^(bcc|abcg2)$ ]] || {
    echo "Charge method must be 'bcc' or 'abcg2'." >&2
    exit 1
}

BASE="${PDB_IN%.*}"         # strip extension, e.g. mol
WORKDIR="$(pwd)"

echo ">>> Cleaning PDB with pdb4amber"
pdb4amber -i "$PDB_IN" -o "${BASE}_clean.pdb" --nohyd

echo ">>> Running antechamber (GAFF2, -c $CHARGE_METHOD)"
antechamber \
    -i "${BASE}_clean.pdb" -fi pdb \
    -o "${BASE}.mol2"      -fo mol2 \
    -s 2                   -at gaff2 \
    -c "$CHARGE_METHOD"    -nc 0 \
    -rn MOL

echo ">>> Generating missing parameters with parmchk2"
parmchk2 -i "${BASE}.mol2" -f mol2 -o "${BASE}.frcmod" -s gaff2

# ---------- 2. Build Amber topology with tleap --------------
cat > tleap.in <<EOF
source leaprc.gaff2
loadmol2  ${BASE}.mol2
loadamberparams ${BASE}.frcmod
saveamberparm  MOL  ${BASE}.prmtop ${BASE}.inpcrd
quit
EOF

echo ">>> Creating prmtop/inpcrd with tleap"
tleap -f tleap.in

# ---------- 3. Convert to OpenMM System XML -----------------
cat > parmed.in <<EOF
parm ${BASE}.prmtop
loadRestrt ${BASE}.inpcrd
outparm ${BASE}.xml
quit
EOF

echo ">>> Converting to OpenMM System XML with ParmEd"
parmed -O -i parmed.in

# ---------- 4. Finished -------------------------------------
echo "--------------------------------------------------"
echo "Created ${BASE}.xml – ready for import into ISOLDE"
echo "Intermediate files: ${BASE}.mol2  ${BASE}.frcmod"
echo "--------------------------------------------------"
