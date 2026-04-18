### `case_paths` 目录说明

这里把案例图按用途拆成了三层：

- **`panels/`**：整版路径图
  - **`main_cases/`**：当前主案例图（如 `case_a`、`case_b`）
  - **`chain_to_ring_candidates/`**：链到环备选图，以及总览缩略图
- **`nodes/`**：单个分子节点图
  - 按 **案例分组 / case slug** 继续拆目录，例如 `nodes/chain_to_ring_candidates/case_e_lumo_up_nco_terminal_ring/`
- **`metadata/`**：说明和表格
  - `case_path_notes.md`
  - `chain_to_ring_shortlist.md`
  - `chain_to_ring_candidates.json`
  - `chain_to_ring_candidates.tsv`

### 当前推荐查看顺序

1. `panels/chain_to_ring_candidates/chain_to_ring_contact_sheet.png`
2. `metadata/chain_to_ring_shortlist.md`
3. 你感兴趣的单张 `*_path_panel.png`
4. 对应 `nodes/.../` 里的单节点 PNG

### 说明

- 之后重新运行 `export_case_path_figures.py` 时，会继续写入上面的分层目录，不会再把所有 PNG 混在同一层。
- 重新运行 `find_chain_to_ring_candidates.py` 时，候选表会更新到 `metadata/`。
