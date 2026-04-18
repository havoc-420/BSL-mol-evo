# Paper Images 索引

本目录存放论文相关的所有图片与说明，按用途分目录。

---

## 目录结构

```
images/
├── case_paths/                    # 图3：案例路径图（链→环等）
│   ├── panels/                    #   整版路径图
│   │   ├── main_cases/            #     主案例（Case A/B）
│   │   └── chain_to_ring_candidates/  # 链→环备选（Case C–I）
│   ├── nodes/                     #   单个分子节点图
│   └── metadata/                  #   说明和候选表
├── scatter_plots/                 # 散点图 & 2D test case 图
│   ├── scatter_*.png/pdf          #   HOMO/LUMO 变化散点图
│   └── test_cases_2d/             #   2D 案例网格图
├── qm9_step1_op_cases_2d/         # QM9 step-1 各操作类型的 2D 示例图
├── ofo_case_f_*.png               # Case F 编辑前后卡片图
└── render_case_f_edit_pair_3d.py  # Case F 3D 渲染脚本
```

---

## 案例路径图 (`case_paths/`)

### 主案例

| Case | 任务 | 起始分子 | full 真值收益 | random_topb 真值收益 | 图版文件 |
| --- | --- | --- | --- | --- | --- |
| A | HOMO down | `C#CCOCC` | 3.87 | 2.35 | `case_a_homo_down_c3ccocc_path_panel.png` |
| B | LUMO up | `NC(=O)C#CCO` | 3.44 | 2.90 | `case_b_lumo_up_nco_c3cco_path_panel.png` |

### 链→环备选案例

| Case | 任务 | 起始分子 | full | random_topb | 推荐优先级 | 图版文件 |
| --- | --- | --- | --- | --- | --- | --- |
| E | LUMO up | `NC(=O)C#CCO` | 3.44 | 2.90 | ★★★ | `case_e_lumo_up_nco_terminal_ring_path_panel.png` |
| D | LUMO up | `CC(C)(C)CC#N` | 1.55 | 0.84 | ★★★ | `case_d_lumo_up_delayed_visible_ring_path_panel.png` |
| C | LUMO up | `C#CCOCC` | 1.57 | 1.71 | ★★ | `case_c_lumo_up_chain_to_ring_c3ccocc_path_panel.png` |
| F | LUMO up | `CCO[C@@H](C)CO` | 0.82 | 0.79 | ★★ | `case_f_lumo_up_alkoxy_terminal_ring_path_panel.png` |
| G | LUMO up | `C#CC(C)(C)CC` | 0.68 | 0.56 | ★ | `case_g_lumo_up_c3cc_tbutyl_terminal_ring_path_panel.png` |
| H | LUMO up | `CC#CC(C)(C)C` | 0.45 | 0.51 | ★ | `case_h_lumo_up_cc3cc_tbutyl_terminal_ring_path_panel.png` |
| I | HOMO down | `CC#CCCCO` | 0.89 | 2.10 | 备胎 | `case_i_homo_down_hexynol_terminal_ring_path_panel.png` |

> 完整候选说明见 `case_paths/metadata/chain_to_ring_shortlist.md` 和 `case_paths/metadata/case_path_notes.md`

---

## 散点图 (`scatter_plots/`)

来源于 `output/scatter_plots/20260417_075604/`。

| 文件 | 说明 |
| --- | --- |
| `scatter_combined.png` | HOMO + LUMO 综合散点图 |
| `scatter_grid.png/.pdf` | 散点网格总览 |
| `scatter_homo_change_{train,val,test}.png` | HOMO 变化散点图（训练/验证/测试） |
| `scatter_lumo_change_{train,val,test}.png` | LUMO 变化散点图（训练/验证/测试） |
| `results_homo_change_{train,val,test}.json` | HOMO 变化统计结果 |
| `results_lumo_change_{train,val,test}.json` | LUMO 变化统计结果 |
| `test_cases_2d/` | 2D 案例网格图（含 assembled 拼合版） |

---

## QM9 Step-1 操作示例 (`qm9_step1_op_cases_2d/`)

展示 QM9 数据集 step-1 中 12 种编辑操作类型的代表性示例，每类一张。

> 详见 `qm9_step1_op_cases_2d/README.md`

---

## 来源

- **case_paths**: `output/paper/figures/case_paths/`
- **scatter_plots**: `output/scatter_plots/20260417_075604/`
- **qm9_step1_op_cases_2d**: 原已存在于 `docs/paper/images/`
