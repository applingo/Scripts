#!/bin/bash
# 使用例:
#   ./auto_transform.sh input.mtz input.pdb refined_model.pdb
# ※ refined_model.pdb は、shifted_model.pdbを用いて精密化済みのモデルファイルとします。

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input.mtz> <input.pdb> <refined_model.pdb>"
    exit 1
fi

MTZ_FILE=$1
PDB_FILE=$2
REFINED_MODEL=$3

echo "=== Step 1: MTZファイルから電子密度マップ (map.ccp4) を作成 ==="
phenix.mtz2map ${MTZ_FILE} d_min=2.0 keep_origin=True map_file=map.ccp4

echo "=== Step 2: マップを立方体 (cubic_map.ccp4) に再グリッド ==="
phenix.python auto_rebox.py map.ccp4 cubic_map.ccp4 > rebox_info.txt
# 出力例の行:
# Applied shift vector (Å): 5.0  -3.2  2.7
SHIFT_LINE=$(grep "Applied shift vector" rebox_info.txt)
Sx=$(echo $SHIFT_LINE | awk '{print $4}')
Sy=$(echo $SHIFT_LINE | awk '{print $5}')
Sz=$(echo $SHIFT_LINE | awk '{print $6}')
echo "適用されたシフトベクトル: ${Sx} ${Sy} ${Sz} (Å)"

echo "=== Step 3: PDBファイルの座標を変換（再配列演算子＋平行移動） ==="
# 自動で再配列演算子を算出
OP_INFO=$(phenix.python auto_operator.py ${MTZ_FILE} ${PDB_FILE})
OPERATOR=$(echo "$OP_INFO" | grep "Determined change-of-basis operator:" | awk '{print $5}')
INVERSE_OPERATOR=$(echo "$OP_INFO" | grep "Inverse operator:" | awk '{print $3}')

echo "算出された変換演算子: ${OPERATOR}"
echo "算出された逆変換演算子: ${INVERSE_OPERATOR}"

# ※ここでは、MTZ→map変換時にkeep_origin=Trueなら座標系は一致していると仮定し、
# さらにauto_rebox.pyで立方体化した際に適用された平行移動 (shift_vector) を反映します。
# そのため、PDB座標に対しては「change_of_basis」オプションで再配列演算子を（必要なら）適用し、
# さらに sites.translate で -shift_vector を適用します。
phenix.pdbtools ${PDB_FILE} change_of_basis="${OPERATOR}" sites.translate="-${Sx},-${Sy},-${Sz}" out=shifted_model.pdb

echo "shifted_model.pdb を作成しました。"
echo "=== Step 4: （ここで shifted_model.pdb を用いてモデル精密化を実施してください） ==="
echo "    ※精密化後のモデルは、引数で指定された ${REFINED_MODEL} とします。"

echo "=== Step 5: 精密化後のモデルを元の座標系に戻す ==="
# 逆変換：まず sites.translate で +shift_vector を適用し、その後逆の再配列演算子を適用
phenix.pdbtools ${REFINED_MODEL} sites.translate="${Sx},${Sy},${Sz}" change_of_basis="${INVERSE_OPERATOR}" out=final_model.pdb

echo "最終モデル (元のMTZ座標系に戻したもの) は final_model.pdb に出力されました。"
