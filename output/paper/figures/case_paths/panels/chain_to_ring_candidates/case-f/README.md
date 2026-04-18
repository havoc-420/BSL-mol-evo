### Case F 资料包

- **case slug**: `case_f_lumo_up_alkoxy_terminal_ring`
- **task**: `Target: raise LUMO`
- **title**: Case F · LUMO increase · terminal ring view · start = CCO[C@@H](C)CO
- **source json**: `/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/lumo_up/full/seed42/search/CCO[C@@H](C)CO_5c52cf46.json`
- **panel**: `panel/case_f_lumo_up_alkoxy_terminal_ring_path_panel.png`
- **single mol 汇总图**: `case_f_full_path_single_mols_contact_sheet.png`
- **styled 汇总图**: `case_f_full_path_single_mols_contact_sheet_styled.png`

### 关键信息

- **起始分子**: `CCO[C@@H](C)CO`
- **最终分子**: `C[C@@H]1OCCCO1`
- **full true gain**: `0.818041`
- **random_topb true gain**: `0.792448`
- **advantage vs random**: `+0.025594`
- **true eval start value**: `1.656601`
- **true eval best value**: `2.474642`
- **tree final cum. Δ_pred**: `3.101765`
- **完整路径长度**: `5` edits / `6` states
- **panel 当前展示节点**: `[0, 1, 13, 14]`
- **本次补全导出节点**: `[0, 1, 11, 12, 13, 14]`

### 路径解读

- **图面风格**: 起点是链态含氧骨架，终点收成一个含氧六元环。
- **真实完整路径**: 这条路径不是单调‘一直链态到最后’，中间先短暂成环，再开环，再在最后一步重新成环。
- **为什么这个 case 仍然好看**: 中间的含氧重排比较集中，末端闭环后的结构风格和前面明显不同，适合做视觉型备选。

### 完整路径节点表

| step | node id | depth | ring? | operation | Δ_pred | cum. Δ_pred | property_value | smiles | image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | chain | start | +0.000 | +0.000 | 2.2068 | `CCO[C@@H](C)CO` | `single_mols/step00_node00_start.png` |
| 1 | 1 | 1 | chain | replace atom O | -0.118 | -0.118 | 2.0887 | `CCO[C@@H](C)OO` | `single_mols/step01_node01_step-1.png` |
| 2 | 11 | 2 | ring | form ring (4, 6) | +1.856 | +1.738 | 3.9444 | `CCO[C@H]1COO1` | `single_mols/step02_node11_step-2.png` |
| 3 | 12 | 3 | ring | replace atom C | +1.155 | +2.893 | 5.0997 | `CCO[C@H]1CCO1` | `single_mols/step03_node12_step-3.png` |
| 4 | 13 | 4 | chain | open ring (4, 5) | +0.075 | +2.968 | 5.1748 | `CCO[C@@H](C)OC` | `single_mols/step04_node13_step-4.png` |
| 5 | 14 | 5 | ring | form ring (0, 6) | +0.134 | +3.102 | 5.3086 | `C[C@@H]1OCCCO1` | `single_mols/step05_node14_final.png` |

### 操作序列

- **edit chain**: `replace atom O` → `form ring (4, 6)` → `replace atom C` → `open ring (4, 5)` → `form ring (0, 6)`

### 文件说明

- **panel**: `panel/case_f_lumo_up_alkoxy_terminal_ring_path_panel.png`
- **single mol images**: `single_mols/`
- **styled single mol images**: `single_mols_styled/`
- **contact sheet**: `case_f_full_path_single_mols_contact_sheet.png`
- **styled contact sheet**: `case_f_full_path_single_mols_contact_sheet_styled.png`

### 备注

- **panel 里缺的两个中间态** 已在这次 bundle 里补齐：`node11` 和 `node12`。
- **true-eval best smiles**: `C[C@@H]1OCCCO1`，与 panel 终点一致。
- **数值判断**: 这条 case 的真实优势不大，更偏视觉叙事备选，而不是 strongest quantitative example。
