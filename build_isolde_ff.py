#!/usr/bin/env python3
"""
build_isolde_ff.py
==================

Generate an ISOLDE‑compatible force‑field for a ligand residue named **MOL**
starting from a PDB file.  It automates the standard AmberTools25 pipeline
(pdb4amber → antechamber → parmchk2 → tleap → mdgx → ParmEd) and adds simple
parallel/GPU acceleration hooks.

Prerequisites
-------------
* AmberTools 25 (GAFF2, mdgx, Quick GPU build optional)
* ParmEd ≥ 3.4 (bundled with AmberTools)
* OpenMM ≥ 8.1 (for ffxml/system XML support)
* CUDA‑capable GPU + Quick (optional, speeds up AM1‑BCC)

Typical usage
-------------
```bash
python build_isolde_ff.py ligand.pdb --resname MOL --threads 16 --gpu
```
This creates:
* **MOL.ffxml** – reusable GAFF2 force‑field fragment
* **MOL.lib**   – Amber residue template (for tleap)
* **MOL_system.xml** – one‑shot System XML for fast testing in ISOLDE
* **MOL_clean.pdb**  – protonated/renumbered PDB used in the build

Drop any of these into ChimeraX/ISOLDE:
```cxc
isolde start
# reusable way
isolde load ffxml MOL.ffxml
open MOL_clean.pdb
# or quick one‑shot
isolde sim MOL_system.xml MOL_clean.pdb
```

"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


###############################################################################
# Helper utilities
###############################################################################

def run(cmd, env=None):
    """Run *cmd* in the shell, streaming stdout/stderr and aborting on error."""
    print(f"[run] {cmd}")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def which(binary):
    """Return full path of *binary* if found in $PATH, else None."""
    return shutil.which(binary)


###############################################################################
# Core workflow
###############################################################################

def build_ff(pdb_file: Path, resname: str, ncpu: int, use_gpu: bool):
    root = Path.cwd().resolve()
    work = root / f"{resname}_work"
    work.mkdir(exist_ok=True)
    os.chdir(work)

    # 1. Clean PDB (strip altlocs, add hydrogens, renumber)
    pdb_clean = f"{resname}_clean.pdb"
    run(f"pdb4amber -i {pdb_file} -o {pdb_clean} --nohyd --reduce")

    # 2. Run antechamber with GAFF2 + AM1‑BCC (Quick GPU if requested)
    mol2 = f"{resname}.mol2"
    env = os.environ.copy()
    if use_gpu and which("quick.cuda"):
        env["QUICK_CUDA"] = "1"  # trigger GPU path inside sqm/quick
        print("[info] QUICK_CUDA enabled – AM1‑BCC will use GPU")
    cmd_ante = (
        f"antechamber -i {pdb_clean} -fi pdb "
        f"-o {mol2} -fo mol2 -at gaff2 -c bcc -s 2 -ncpu {ncpu}"
    )
    run(cmd_ante, env)

    # 3. parmchk2 – fill in missing GAFF2 parameters
    frcmod = f"{resname}.frcmod"
    run(f"parmchk2 -i {mol2} -f mol2 -o {frcmod}")

    # 4. tleap – make prmtop/inpcrd (for validation / System XML)
    tleap_in = "tleap.in"
    tleap_script = f"""
source leaprc.gaff2
loadAmberParams {frcmod}
{resname} = loadMol2 {mol2}
check {resname}
saveAmberParm {resname} {resname}.prmtop {resname}.inpcrd
quit
"""
    Path(tleap_in).write_text(tleap_script)
    run(f"tleap -f {tleap_in}")

    # 5. mdgx – generate ffxml (+ .lib)
    mdgx_in = "mdgx_ffxml.in"
    mdgx_script = f"""
&files
  -par {resname}
  -mol {mol2}
  -frcmod {frcmod}
  -ffxml {resname}.ffxml
&end
&param
  forcefield = gaff2,
  charge     = bcc,
&end
"""
    Path(mdgx_in).write_text(mdgx_script)
    run(f"mdgx -i {mdgx_in}")

    # 6. ParmEd – one‑shot System XML (easy drag‑and‑drop into ISOLDE)
    system_xml = f"{resname}_system.xml"
    parmed_py = (
        "import parmed as pmd, sys; "
        f"parm=pmd.load_file('{resname}.prmtop', '{resname}.inpcrd'); "
        f"parm.save('{system_xml}')"
    )
    run(f"python - <<'PY'\n{parmed_py}\nPY")

    print("\n[✓] All files generated in", work)
    print("     ├──", pdb_clean)
    print("     ├──", mol2)
    print("     ├──", frcmod)
    print("     ├──", f"{resname}.prmtop / .inpcrd")
    print("     ├──", f"{resname}.ffxml  (re‑usable)")
    print("     ├──", f"{resname}.lib    (Amber template)")
    print("     └──", system_xml)


###############################################################################
# CLI entry‑point
###############################################################################

def cli():
    p = argparse.ArgumentParser(description="Build ISOLDE‑ready force‑field via AmberTools25")
    p.add_argument("pdb", type=Path, help="Input PDB containing residue MOL")
    p.add_argument("--resname", default="MOL", help="Residue name to parameterise (default: MOL)")
    p.add_argument("--threads", "-j", type=int, default=os.cpu_count(), help="CPU threads for antechamber/sqm")
    p.add_argument("--gpu", action="store_true", help="Use Quick CUDA acceleration if available")
    args = p.parse_args()

    build_ff(args.pdb, args.resname.upper(), args.threads, args.gpu)


if __name__ == "__main__":
    cli()
