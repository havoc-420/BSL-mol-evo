#!/usr/bin/env python3
"""
从 qm9-evo-pairs JSON 文件中提取 smiles_from / smiles_to 去重后输出为 CSV。
"""

import json
import csv
import sys
from pathlib import Path


def main():
    input_file = Path(__file__).parent / "data" / "qm9-evo-pairs-step-1-pairs.json"
    output_file = Path(__file__).parent / "qm9_evo_smiles_pairs_dedup.csv"

    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = set()
    smiles_list = []
    for item in data:
        for key in ("smiles_from", "smiles_to"):
            s = item[key]
            if s not in seen:
                seen.add(s)
                smiles_list.append(s)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        writer.writerows([[s] for s in smiles_list])

    print(f"原始条目: {len(data)}")
    print(f"去重后 SMILES 数量: {len(smiles_list)}")
    print(f"已保存至: {output_file}")


if __name__ == "__main__":
    main()
