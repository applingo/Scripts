#!/usr/bin/env python3
import subprocess
import sys


#phenix.polder your_model.pdb your_data.mtz omit_atom_selection="element Zn or element Na or element Ca or element K or element Mg" output_file=omit_map.ccp4

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input.mtz input.pdb output_map.ccp4".format(sys.argv[0]))
        sys.exit(1)

    mtz_file = sys.argv[1]
    pdb_file = sys.argv[2]
    output_map = sys.argv[3]

    # 除外する重元素の指定（必要に応じて拡張可能）
    omit_selection = 'element Zn or element Na or element Ca or element K or element Mg'

    # phenix.polder を呼び出すコマンド
    cmd = [
        "phenix.polder",
        pdb_file,
        mtz_file,
        "omit_atom_selection=" + omit_selection,
        "output_file=" + output_map
    ]
    print("Executing command:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("phenix.polder の実行中にエラーが発生しました。")
        sys.exit(1)
    print("オミットマップが生成されました:", output_map)

if __name__ == "__main__":
    main()
