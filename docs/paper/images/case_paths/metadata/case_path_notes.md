### 图 3 案例路径图：已选 case

#### Case A：`homo_down / C#CCOCC`
- **选择理由**：`full` 的真值收益为 **3.87**，明显高于 `random_topb` 的 **2.35**；路径前 8 步几乎全是可解释的结构编辑，只有最后 2 步才进入立体修饰。
- **图版文件**：`case_a_homo_down_c3ccocc_path_panel.png`
- **关键节点**：
  - `node00`：起始分子 `C#CCOCC`
  - `node01`：`CCOCC#N`
  - `node02`：`CCCCC#N`
  - `node03`：`N#CCCCF`
  - `node05`：`N#CC(F)(F)CCF`
  - `node08`：最终分子 `N#CC(F)(F)OC(F)(F)F`
- **图意重点**：路径表现出明显的“腈化 → 碳链化 → 多步氟化 → 含氧氟代末端重构”的局部变换逻辑，累计 OFO 预测收益单调下降（更优）。

#### Case B：`lumo_up / NC(=O)C#CCO`
- **选择理由**：`full` 的真值收益为 **3.44**，高于 `random_topb` 的 **2.90**；路径没有依赖 `add_stereo` 灌水，主要由成环、键型调整和碳骨架重排组成。
- **图版文件**：`case_b_lumo_up_nco_c3cco_path_panel.png`
- **关键节点**：
  - `node00`：起始分子 `NC(=O)C#CCO`
  - `node01`：`C=C(N)C#CCO`
  - `node02`：`NC1=CC(O)C#C1`
  - `node03`：`CC(N)=CC(C)O`
  - `node06`：`CC(O)=C(C)C(C)C`
  - `node09`：最终分子 `CCOCC1CC1C`
- **图意重点**：路径表现出“去羰基/电子重分布 → 成环 → 去三键 → 烃化增疏水 → 环化收束”的连续局部优化过程，累计 OFO 预测收益单调上升（更优）。

#### Case C：`lumo_up / C#CCOCC`（更纯粹的链→环叙事）
- **选择理由**：如果你的首要目标是让读者一眼看出“**开链骨架如何被逐步收束成环**”，这个 case 比 `NC(=O)C#CCO` 更直观：主线只有 **成环 → 三键松弛 → 再成环 → 立体定型**。
- **图版文件**：`case_c_lumo_up_chain_to_ring_c3ccocc_path_panel.png`
- **关键节点**：
  - `node00`：起始分子 `C#CCOCC`
  - `node01`：首次成环 `CC1C#CCO1`
  - `node02`：三键松弛 `CCOC(C)C`
  - `node03`：再次成环 `CCOC1CC1`
  - `node04`：最终分子 `CCO[C@@H]1CC1`
- **图意重点**：这条路径更适合强调“局部编辑如何把链状分子一步步收束为小环”。不过它是 **视觉叙事优先** 的选择：该单 case 的 `full` 真值收益为 **1.57**，略低于 `random_topb` 的 **1.71**。

#### Case D：`lumo_up / CC(C)(C)CC#N`（避免图中段突然见环）
- **选择理由**：真实最佳路径里先经历两步链态编辑，再通过一次临时成环与键型松弛到达更稳定链态，最后才闭成最终小环。图中只展示链态关键节点和最终环态节点，因此**视觉上不会在中段突然看到一个环分子**。
- **图版文件**：`case_d_lumo_up_delayed_visible_ring_path_panel.png`
- **关键节点**：
  - `node00`：起始分子 `CC(C)(C)CC#N`
  - `node01`：`C#CCC(C)(C)C`
  - `node02`：`CC#CCC(C)(C)C`
  - `node28`：链态收束后 `CCCC(C)(C)CC`
  - `node29`：最终分子 `CCCC1(C)CC1C`
- **图意重点**：图版中把“临时成环 + 三键松弛”压缩到箭头说明里，仅在终点展示环结构；这样保留了真实 OFO 路径，又更符合“先链、后环”的阅读体验。该单 case 的 `full` 真值收益为 **1.55**，高于 `random_topb` 的 **0.84**。

#### 脚本
- 候选筛选：`docs/paper/sweep/v2/select_case_candidates.py`
- 路径出图：`docs/paper/sweep/v2/export_case_path_figures.py`
