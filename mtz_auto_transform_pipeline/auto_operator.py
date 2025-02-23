#!/usr/bin/env phenix.python
import sys
from iotbx import mtz, pdb
from cctbx import crystal

def main():
  if len(sys.argv) < 3:
    print("Usage: auto_operator.py <input.mtz> <input.pdb>")
    sys.exit(1)
  mtz_file = sys.argv[1]
  pdb_file = sys.argv[2]

  # MTZファイルから結晶対称性情報を取得
  mtz_obj = mtz.object(mtz_file)
  mtz_sym = mtz_obj.crystal_symmetry()

  # PDBファイルから結晶対称性情報を取得
  pdb_inp = pdb.input(file_name=pdb_file)
  pdb_sym = pdb_inp.crystal_symmetry()

  # もし両者が一致していれば、変換は不要（identity operator）
  if mtz_sym.unit_cell() == pdb_sym.unit_cell() and \
     mtz_sym.space_group().type().lookup_symbol() == pdb_sym.space_group().type().lookup_symbol():
    operator_str = "a,b,c"
  else:
    try:
      # PDBからMTZへの変換演算子を自動算出
      op = pdb_sym.change_of_basis_op(mtz_sym)
      operator_str = op.as_string()
    except Exception as e:
      print("自動算出に失敗しました: ", e)
      operator_str = "a,b,c"

  # 逆変換演算子を算出
  op = crystal.change_of_basis_op(operator_str)
  inv_op = op.inverse()
  inverse_operator_str = inv_op.as_string()

  print("Determined change-of-basis operator: {}".format(operator_str))
  print("Inverse operator: {}".format(inverse_operator_str))

if __name__=="__main__":
  main()
