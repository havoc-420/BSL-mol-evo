#!/usr/bin/env python3
"""
Markdown 表格 ↔ CSV ↔ XLSX 三向转换工具。

用法:
    # MD → CSV
    python3 md_csv.py to-csv input.md [output.csv]

    # CSV → MD
    python3 md_csv.py to-md input.csv [output.md]

    # XLSX → CSV
    python3 md_csv.py xlsx-to-csv input.xlsx [output.csv]

    # XLSX → MD（一步到位）
    python3 md_csv.py xlsx-to-md input.xlsx [output.md]

默认输出路径: 同目录下同名切换扩展名，如未指定则自动生成。
MD 中的空行分隔行在 CSV 中以全空行保留。
"""

import csv
import re
import sys
import os
from datetime import datetime


# ─────────────────────────── MD → CSV ───────────────────────────

def parse_md_table(md_text: str) -> list[list[str]]:
    """解析 Markdown 表格行，返回单元格列表的列表。"""
    rows = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        rows.append(cells)
    return rows


def md_to_csv(md_path: str, csv_path: str | None = None) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    rows = parse_md_table(md_text)
    if not rows:
        print("未找到 Markdown 表格数据", file=sys.stderr)
        sys.exit(1)

    num_cols = len(rows[0])

    if csv_path is None:
        base_dir = os.path.dirname(md_path) or "."
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        csv_path = os.path.join(base_dir, f"{base_name}.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            if all(c == "" for c in row):
                writer.writerow([""] * num_cols)
            else:
                padded = row + [""] * (num_cols - len(row))
                writer.writerow(padded)

    print(f"MD → CSV: {md_path} -> {csv_path}")
    return csv_path


# ─────────────────────────── CSV → MD ───────────────────────────

def csv_to_md(csv_path: str, md_path: str | None = None) -> str:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV 文件为空", file=sys.stderr)
        sys.exit(1)

    md_content = rows_to_md(rows)

    if md_path is None:
        base_dir = os.path.dirname(csv_path) or "."
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        md_path = os.path.join(base_dir, f"{base_name}-{ts}.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"CSV → MD: {csv_path} -> {md_path}")
    return md_path


def rows_to_md(rows: list[list[str]]) -> str:
    """将行数据（含表头）格式化为紧凑 Markdown 表格文本。

    格式风格：
    - 单元格紧凑，不做列宽填充对齐
    - 对齐行简洁: | --- | (左对齐) 或 | ---: | (右对齐)
    - 组间空行用 |  |  | ... | 表格空行
    """
    num_cols = len(rows[0]) if rows else 0

    # 过滤尾部全空行
    while rows and all(c == "" for c in rows[-1]):
        rows.pop()

    # 判断数值列（右对齐）
    numeric_cols: set[int] = set()
    for i in range(num_cols):
        is_numeric = True
        has_value = False
        for row in rows:
            if i < len(row) and row[i].strip():
                has_value = True
                try:
                    float(row[i].replace("%", ""))
                except ValueError:
                    is_numeric = False
                    break
        if has_value and is_numeric:
            numeric_cols.add(i)

    def format_row(row: list[str]) -> str:
        parts = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            parts.append(cell)
        return "| " + " | ".join(parts) + " |"

    def format_empty_row() -> str:
        return "| " + " | ".join([""] * num_cols) + " |"

    def format_align_row() -> str:
        parts = []
        for i in range(num_cols):
            if i in numeric_cols:
                parts.append("---:")
            else:
                parts.append("---")
        return "| " + " | ".join(parts) + " |"

    # 分组：空行分隔
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in rows:
        if all(c == "" for c in row):
            if current:
                groups.append(current)
                current = []
        else:
            current.append(row)
    if current:
        groups.append(current)

    lines: list[str] = []

    for gi, group in enumerate(groups):
        if gi > 0:
            lines.append(format_empty_row())

        for ri, row in enumerate(group):
            if ri == 0 and gi == 0:
                lines.append(format_row(row))
                lines.append(format_align_row())
            else:
                lines.append(format_row(row))

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────── XLSX → CSV ───────────────────────────

def xlsx_read_rows(xlsx_path: str, sheet_name: str | int | None = None) -> list[list[str]]:
    """读取 XLSX 为字符串行列表，自动还原 Excel 百分格式的百分比显示。"""
    import pandas as pd
    from openpyxl import load_workbook

    if sheet_name is None:
        df = pd.read_excel(xlsx_path, sheet_name=0, header=None, dtype=str)
        sn = 0
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str)
        sn = sheet_name

    df = df.fillna("")

    # 用 openpyxl 检测百分格式的列，将小数还原为百分比字符串
    try:
        wb = load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.worksheets[sn] if isinstance(sn, int) else wb[sn]
        pct_cols: set[int] = set()
        for row in ws.iter_rows(min_row=2):  # 跳过表头
            for cell in row:
                if cell.number_format and "%" in cell.number_format:
                    pct_cols.add(cell.column - 1)  # 0-indexed
        wb.close()

        if pct_cols:
            for col_idx in pct_cols:
                for row_idx in range(len(df)):
                    val = df.iat[row_idx, col_idx]
                    if val.strip():
                        try:
                            df.iat[row_idx, col_idx] = f"{float(val) * 100:.2f}%"
                        except (ValueError, TypeError):
                            pass
    except Exception:
        pass  # openpyxl 读取失败则跳过百分比还原

    rows = []
    for _, row in df.iterrows():
        rows.append([str(v).strip() for v in row])
    return rows


def xlsx_to_csv(xlsx_path: str, csv_path: str | None = None, sheet_name: str | int | None = None) -> str:
    """将 XLSX 文件转为 CSV。sheet_name 可指定工作表名或索引，默认首个。"""
    rows = xlsx_read_rows(xlsx_path, sheet_name)
    if not rows:
        print("XLSX 文件为空", file=sys.stderr)
        sys.exit(1)

    num_cols = len(rows[0])

    if csv_path is None:
        base_dir = os.path.dirname(xlsx_path) or "."
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
        csv_path = os.path.join(base_dir, f"{base_name}.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            padded = row + [""] * (num_cols - len(row))
            writer.writerow(padded)

    print(f"XLSX → CSV: {xlsx_path} -> {csv_path}")
    return csv_path


# ─────────────────────────── XLSX → MD ───────────────────────────

def xlsx_to_md(xlsx_path: str, md_path: str | None = None, sheet_name: str | int | None = None) -> str:
    """将 XLSX 文件转为 Markdown 表格。"""
    rows = xlsx_read_rows(xlsx_path, sheet_name)
    if not rows:
        print("XLSX 文件为空", file=sys.stderr)
        sys.exit(1)

    # 过滤尾部全空行
    while rows and all(c == "" for c in rows[-1]):
        rows.pop()

    md_content = rows_to_md(rows)

    if md_path is None:
        base_dir = os.path.dirname(xlsx_path) or "."
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        md_path = os.path.join(base_dir, f"{base_name}-{ts}.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"XLSX → MD: {xlsx_path} -> {md_path}")
    return md_path


# ─────────────────────────── CLI ───────────────────────────

def main():
    if len(sys.argv) < 3:
        print("用法:")
        print("  python md_csv.py to-csv       input.md  [output.csv]")
        print("  python md_csv.py to-md        input.csv [output.md]")
        print("  python md_csv.py xlsx-to-csv  input.xlsx [output.csv]")
        print("  python md_csv.py xlsx-to-md   input.xlsx [output.md]")
        sys.exit(1)

    direction = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if direction == "to-csv":
        md_to_csv(input_path, output_path)
    elif direction == "to-md":
        csv_to_md(input_path, output_path)
    elif direction == "xlsx-to-csv":
        xlsx_to_csv(input_path, output_path)
    elif direction == "xlsx-to-md":
        xlsx_to_md(input_path, output_path)
    else:
        print(f"未知方向: {direction}，请用 to-csv / to-md / xlsx-to-csv / xlsx-to-md", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
