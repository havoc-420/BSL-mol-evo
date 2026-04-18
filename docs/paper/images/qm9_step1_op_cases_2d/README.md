### QM9 step-1 不同 op 的 2D case 图

- **数据源**: `/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-pairs-127730.json`
- **选择规则**: 每个 `op` 选 1 个代表性 pair，优先选择更短、更规整、括号/电荷更少的分子。
- **导出总数**: `12` 个不同 `op` case
- **汇总图**: `qm9_step1_op_cases_contact_sheet.png`
- **清单**: `selected_cases.json`

### case 列表

| # | op | count | row | from | to | detail | image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `add_atom` | 6572 | 22766 | `CC(=O)C(C)(C)C` | `CC(=O)C(C)(C)C=O` | `add atom O @ 6` | `case_01_add_atom.png` |
| 2 | `add_stereo` | 7415 | 2039 | `CC(=O)C(C)=O` | `CC(=O)[C@@H](C)O` | `add stereo @ 3` | `case_02_add_stereo.png` |
| 3 | `form_double_bond` | 7491 | 62885 | `CC(=O)C(O)C(C)=O` | `CC(=O)C(=O)C(C)=O` | `form double bond @ 3-4` | `case_03_form_double_bond.png` |
| 4 | `form_double_ring` | 265 | 35547 | `CC(=O)CC(C)(C)C` | `CC1=C(C=O)CCC1` | `form double ring @ 2-6` | `case_04_form_double_ring.png` |
| 5 | `form_ring` | 6497 | 66185 | `O=CCC12CC(C1)C2` | `CC(=O)C12CC(C1)C2` | `form ring @ 1-3` | `case_05_form_ring.png` |
| 6 | `form_triple_bond` | 9366 | 44352 | `CC(=O)C(C)(C)C=O` | `C#CC(=O)C(C)(C)O` | `form triple bond @ 2-3` | `case_06_form_triple_bond.png` |
| 7 | `remove_add_stereo` | 7415 | 2107 | `CC(=O)[C@@H](C)O` | `CC(=O)C(C)=O` | `remove stereo @ 3` | `case_07_remove_add_stereo.png` |
| 8 | `remove_form_double_bond` | 7491 | 62832 | `CC(=O)C(=O)C(C)=O` | `CC(=O)C(O)C(C)=O` | `double→single @ 3-4` | `case_08_remove_form_double_bond.png` |
| 9 | `remove_form_double_ring` | 265 | 47654 | `CC1=CC(C)(C)OC1` | `C1CCC2(CC2)OC1` | `open double ring @ 2-4` | `case_09_remove_form_double_ring.png` |
| 10 | `remove_form_ring` | 6497 | 66300 | `CC(=O)C12CC(C1)C2` | `O=CCC12CC(C1)C2` | `open ring @ 1-3` | `case_10_remove_form_ring.png` |
| 11 | `remove_form_triple_bond` | 9366 | 36908 | `C#CC(=O)C(C)(C)O` | `CC(=O)C(C)(C)C=O` | `triple→single @ 2-3` | `case_11_remove_form_triple_bond.png` |
| 12 | `replace_atom` | 59090 | 62845 | `CC(=O)C(=O)C(N)=O` | `NC(=O)C(=O)C(N)=O` | `replace atom C→N @ 5` | `case_12_replace_atom.png` |

### 说明

- **图面风格**: 延续论文 2D 分子图思路，每张图展示 `Mfrom → Mto`、操作类型和简要 edit 细节。
- **row**: 使用 1-based JSON 数组行号，便于回到原始 `qm9-evo-pairs-step-1-pairs-127730.json` 查对应样本。
- **可扩展**: 如需只画某几类 `op`，可以直接改脚本里的筛选逻辑或加 CLI 参数。

