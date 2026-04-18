### 链到环备选图清单

下面这批都按“**尽量让环只在终点或接近终点出现**”的思路出过图，可以直接在 `case_paths/` 里逐个看 PNG。

#### 优先推荐

- **Case E · `lumo_up / NC(=O)C#CCO`**
  - **图版**：`case_e_lumo_up_nco_terminal_ring_path_panel.png`
  - **展示节点**：`NC(=O)C#CCO → C=C(N)C#CCO → CC(O)=CC(C)O → CCOCC(C)CC → CCOCC1CC1C`
  - **优点**：数值最稳，`full = 3.44`，`random_topb = 2.90`；图里可以做到前面一直是链态，最后才看到环。
  - **注意**：真实路径内部仍有一次隐含环化/去环化，只是展示上被压进箭头说明。

- **Case D · `lumo_up / CC(C)(C)CC#N`**
  - **图版**：`case_d_lumo_up_delayed_visible_ring_path_panel.png`
  - **展示节点**：`CC(C)(C)CC#N → C#CCC(C)(C)C → CC#CCC(C)(C)C → CCCC(C)(C)CC → CCCC1(C)CC1C`
  - **优点**：链态阶段最清楚，`full = 1.55`，`random_topb = 0.84`，对比也不错。
  - **注意**：真实最优链里中间有一次临时成环，但图面不会直接显示出来。

- **Case F · `lumo_up / CCO[C@@H](C)CO`**
  - **图版**：`case_f_lumo_up_alkoxy_terminal_ring_path_panel.png`
  - **展示节点**：`CCO[C@@H](C)CO → CCO[C@@H](C)OO → CCO[C@@H](C)OC → C[C@@H]1OCCCO1`
  - **优点**：风格和前两个不一样，含氧骨架比较柔和，终点成一个含氧六元环。
  - **注意**：数值优势很弱，`full = 0.82`，`random_topb = 0.79`，更像视觉备选。

#### 短路径备选

- **Case G · `lumo_up / C#CC(C)(C)CC`**
  - **图版**：`case_g_lumo_up_c3cc_tbutyl_terminal_ring_path_panel.png`
  - **展示节点**：`C#CC(C)(C)CC → CCCC(C)(C)C → CCC1CC1(C)C`
  - **优点**：最短最干净，三张图就能讲完“链态重排后闭环”。
  - **注意**：真实路径中第一步其实先成过一个环，再被压回链态；`full = 0.68`，`random_topb = 0.56`。

- **Case H · `lumo_up / CC#CC(C)(C)C`**
  - **图版**：`case_h_lumo_up_cc3cc_tbutyl_terminal_ring_path_panel.png`
  - **展示节点**：`CC#CC(C)(C)C → CCCC(C)(C)C → CCC1CC1(C)C`
  - **优点**：和 Case G 很像，适合当同构替补。
  - **注意**：数值略输随机，`full = 0.45`，`random_topb = 0.51`。

#### 非 `lumo_up` 备胎

- **Case I · `homo_down / CC#CCCCO`**
  - **图版**：`case_i_homo_down_hexynol_terminal_ring_path_panel.png`
  - **展示节点**：`CC#CCCCO → OCCCC#CF → FC#CCC1OO1`
  - **优点**：非常短，视觉上几乎就是“链态改造后收成小环”。
  - **注意**：这是 `HOMO decrease` 任务，而且单 case 对比差，`full = 0.89`，`random_topb = 2.10`。

#### 我当前建议的看图顺序

1. `case_e_lumo_up_nco_terminal_ring_path_panel.png`
2. `case_d_lumo_up_delayed_visible_ring_path_panel.png`
3. `case_f_lumo_up_alkoxy_terminal_ring_path_panel.png`
4. `case_g_lumo_up_c3cc_tbutyl_terminal_ring_path_panel.png`
5. `case_h_lumo_up_cc3cc_tbutyl_terminal_ring_path_panel.png`
6. `case_i_homo_down_hexynol_terminal_ring_path_panel.png`
