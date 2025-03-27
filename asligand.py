#!/usr/bin/env python3
import sys

# 使用する34進数用の文字（I, Oを除く）
BASE34 = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 全34文字

def int_to_base34(n):
    """
    1から始まる番号nを、34進数の2桁文字列に変換する関数
    例：1 -> "01", 2 -> "02", 34 -> "10"
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    digits = []
    temp = n
    while temp:
        temp, r = divmod(temp, 34)
        digits.append(BASE34[r])
    if not digits:
        digits.append("0")
    result = ''.join(reversed(digits))
    # 桁数が足りなければ先頭に0を付加（必ず2桁になるように）
    if len(result) < 2:
        result = "0" + result
    return result[-2:]

def extract_element(atom_name_field):
    """
    PDBファイルのATOM名フィールド（columns 13-16）の先頭2文字（columns 13-14）を
    抽出して元素記号とする関数。
    ※ ここでは13-14列に既に元素記号が記載されている前提なので、特別な処理は行わない。
    """
    return atom_name_field[:2].strip().upper()

def process_pdb(input_file, output_file):
    """
    入力PDBファイルを読み込み、各ATOM/HETATM行について以下の処理を行う：
      - 残基名 (columns 18-20) を "MOL" に変更
      - ATOM名 (columns 13-16) を「抽出した元素記号＋34進数2桁の番号」に変更
         ※ 元素記号は columns 13-14（右詰）、残りの部分は位置識別子として扱われる
      - chain名（列22, index21）が空欄の場合は "X" を設定
      - Elementフィールド (columns 77-78) に抽出した元素記号（右詰2文字）を設定
    その他の行はそのまま出力する
    """
    atom_counter = 1  # ATOM/HETATM行ごとにカウンタ（1からスタート）
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # ATOM名フィールド（columns 13-16）を取り出す
                orig_atom_name_field = line[12:16]
                # 13-14列から元素記号を抽出
                element = extract_element(orig_atom_name_field)
                # カウンタを34進数2桁文字列に変換
                num_str = int_to_base34(atom_counter)
                atom_counter += 1
                # 新しいATOM名の作成
                # 元素記号が1文字の場合は右詰め（先頭に空白）で作成
                if len(element) == 1:
                    new_atom_name = " " + element + num_str  # 例: " C01"
                else:
                    new_atom_name = element + num_str        # 例: "CL03"
                # ATOM名フィールド (columns 13-16) を置換
                new_line = line[:12] + new_atom_name + line[16:]
                # 残基名 (columns 18-20) を "MOL" に変更
                new_line = new_line[:17] + "MOL" + new_line[20:]
                # chain名（列22, index21）が空欄なら "X" を設定
                if new_line[21].strip() == "":
                    new_line = new_line[:21] + "X" + new_line[22:]
                # Elementフィールド (columns 77-78) に抽出した元素記号（右詰2文字）を設定
                new_line = new_line[:76] + element.rjust(2) + new_line[78:]
                outfile.write(new_line)
            else:
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdb_processor.py input.pdb output.pdb")
        sys.exit(1)
    process_pdb(sys.argv[1], sys.argv[2])
