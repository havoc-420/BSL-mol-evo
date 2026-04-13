# Paper Table Scripts

实验结果表格的格式转换与排名计算工具集。

## md_csv.py — 表格格式互转

Markdown 表格 ↔ CSV ↔ XLSX 三向转换。

```bash
# MD → CSV
python3 md_csv.py to-csv input.md [output.csv]

# CSV → MD
python3 md_csv.py to-md input.csv [output.md]

# XLSX → CSV
python3 md_csv.py xlsx-to-csv input.xlsx [output.csv]

# XLSX → MD（一步到位）
python3 md_csv.py xlsx-to-md input.xlsx [output.md]
```

**特性：**
- MD 输出为紧凑格式（无列宽填充），对齐行简洁（`| --- | ---: |`）
- 组间空行保留为 `|  |  | ... |` 表格空行
- XLSX 读取自动还原 Excel 百分格式的百分比显示（如 `0.9458` → `94.58%`）
- 未指定输出路径时自动生成（同目录同名切换扩展名，CSV→MD 加时间戳）

## rank_table.py — 分区排名计算

按"目标属性"分区，计算改善值排名、改善率排名、加权排名和总排名。

```bash
# 默认输出带时间戳的新文件
python3 rank_table.py input.csv

# 指定输出路径
python3 rank_table.py input.csv output.csv
```

**排名规则：**
1. 按目标属性值分区（LUMO(D)、LUMO(U)、HOMO(D)、HOMO(U) 等）
2. 分区内按"平均改善值"绝对值排名（越大 → rank 越小）
3. 分区内按"改善百分比"排名（越高 → rank 越小）
4. **rank 加权** = 改善值 rank + 改善率 rank
5. **总排名** = 按 rank 加权从小到大排名
6. 并列采用竞赛排名（1 2 2 4 风格）
7. 无数据分区（如 GAP）自动跳过

**依赖：** 无额外依赖，仅使用 Python 标准库（`md_csv.py` 的 XLSX 功能需 `pandas` + `openpyxl`）。

## 典型工作流

```bash
# 1. XLSX → MD（从 Excel 导出）
python3 md_csv.py xlsx-to-md task-1.xlsx

# 2. 编辑 MD 中的数据

# 3. MD → CSV
python3 md_csv.py to-csv task-1.md

# 4. 计算排名
python3 rank_table.py task-1.csv

# 5. 排名后 CSV → MD（回写）
python3 md_csv.py to-md task-1-ranked.csv
```
