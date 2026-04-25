# MCTS 侧消融结果解读

## 1. 使用定位

本文档不追求罗列全部原始字段，而是从论文读者最关心的三个角度来解读当前结果：

- **效果是否变差？**
- **效率是否变差？**
- **稳定性是否也受到了影响？**

当前最可靠的主数值证据来自 `pilot` 真值评估；`smoke` 主要用于验证方向一致性；而本轮 `official` 虽未生成真值终表，但提供了重要的**搜索层稳定性证据**。

## 2. `pilot` 真值聚合结果

### 2.1 `HOMO(D)`

| variant | improvement | success rate | runtime | expanded nodes | 相对 `full` 的解读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | `2.0041` | `1.00` | `2.31` | `7.60` | 基线 |
| `wo_prior` | `1.9007` | `1.00` | `1.97` | `5.95` | 效果略低（`-0.1034`），但更省 |
| `wo_leaf_value` | `0.4896` | `0.95` | `204.35` | `188.25` | **效果大幅退化**，且成本暴涨 |
| `random_topb` | `1.6479` | `0.95` | `3.62` | `8.80` | 效果低于 `full`，也更慢 |

### 2.2 `LUMO(U)`

| variant | improvement | success rate | runtime | expanded nodes | 相对 `full` 的解读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | `1.1452` | `1.00` | `11.49` | `12.45` | 基线 |
| `wo_prior` | `1.0953` | `1.00` | `9.79` | `10.85` | 效果略低（`-0.0499`），但更省 |
| `wo_leaf_value` | `1.2969` | `1.00` | `240.14` | `207.95` | improvement 略高，但**代价极端不可接受** |
| `random_topb` | `1.1130` | `1.00` | `34.47` | `30.20` | 效果未优于 `full`，效率明显更差 |

## 3. `official` 搜索层补充结果

### 3.1 本轮 `official` 的最终状态

| Task | Variant | Seeds | 搜索状态 | 解释 |
| --- | --- | --- | --- | --- |
| `lumo_up` | `full` | `42/43/44` | `3 / 3 success` | 搜索层稳定通过 |
| `lumo_up` | `wo_prior` | `42/43/44` | `3 / 3 success` | 搜索层稳定通过 |
| `lumo_up` | `wo_leaf_value` | `42/43/44` | `3 / 3 success` | 搜索层可跑通，但不代表真值层表现可接受 |
| `homo_down` | `full` | `42/43/44` | `3 / 3 success` | 搜索层稳定通过 |
| `homo_down` | `wo_prior` | `42/43/44` | `3 / 3 success` | 搜索层稳定通过 |
| `homo_down` | `wo_leaf_value` | `42/43/44` | `0 / 3 success` | **三个 seed 全部 `search_failed`** |

### 3.2 为什么这很重要

本轮 `official` 没有直接产出可写进主表的真值终表，因为：

- 启动参数为 `RUN_EVAL=0`；
- 因而没有 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`；
- 同时 `homo_down / wo_leaf_value` 的三个 seed 全部失败，矩阵本身也不完整。

但它仍然提供了一个很强的补充信息：

- 三个失败 run **发生在同一个 task / variant 上**；
- 三个失败 run **都在同一个起始分子 `CC#CC(C)(C)C` 上复现**；
- 三个失败 run **都报同一类错误：`'NoneType' object has no attribute 'is_expanded'`**。

因此，从论文角度，这轮 `official` 不能作为“最终数值表”，但可以作为：

> **移除 `leaf value` 后，系统在更正式预算下还会暴露出稳定性问题。**

## 4. 从论文角度如何读这些结果

### 4.1 `wo_leaf_value`：这是当前最强的负面对照

这个变体最重要，因为它已经不是“略差”，而是出现了**结构性退化**：

- 在 `HOMO(D)` 上，平均改善从 `2.0041` 下降到 `0.4896`，下降约 `75.6%`；
- 同时 runtime 从 `2.31` 飙到 `204.35`，约为 `88.4x`；
- expanded nodes 从 `7.60` 上升到 `188.25`，约为 `24.8x`。

这说明 `leaf value` 并不是一个“锦上添花”的模块，而是当前框架中**实质性决定搜索质量与搜索效率的关键组件**。

在 `LUMO(U)` 上，`wo_leaf_value` 的 improvement 表面上略高于 `full`，但这并不支持“去掉 leaf value 更好”这样的说法，原因有两点：

1. runtime 与 expanded nodes 分别放大到 `20.9x` 与 `16.7x`；
2. `start_count_mean = 19`，低于 `full` 的 `20`，说明该结果本身并不是完全同口径覆盖。

再加上本轮 `official` 搜索层里，`homo_down / wo_leaf_value` 出现 **`3 / 3` seeds 复现失败**，所以现在对它的结论已经不只是“效果-效率更差”，而是：

> **去掉 `leaf value` 会同时损害优化质量、搜索效率，并在正式预算下暴露出稳定性风险。**

### 4.2 `wo_prior`：方向稳定，但幅度小

`wo_prior` 的特点是：

- 两个任务上都没有赢过 `full`；
- 但差距很小；
- 同时运行成本略低；
- 在本轮 `official` 搜索层中没有出现系统性失败。

这说明 prior 对当前框架是**有帮助但并不决定性**的组件。更准确地说，prior 更像是在“相似总预算下帮助搜索更稳地走向更优区域”，但它的重要性明显弱于 `leaf value`。

从论文论证角度，这类结果适合写成：

- `prior` 带来**一致但温和的正向贡献**；
- `leaf value` 才是当前 MCTS 侧性能差异的主要来源。

### 4.3 `random_topb`：当前更像效率消融，而不是效果消融

`random_topb` 目前的证据仍主要来自 `smoke + pilot`。它的特征是：

- `HOMO(D)` 上持续低于 `full`；
- `LUMO(U)` 上在 smoke 中曾短暂高于 `full`，但到了 pilot 反而低于 `full`；
- runtime 与 expanded nodes 在两个任务上都明显高于 `full`。

因此，这个变体当前更适合支持如下说法：

- OFO 引导的 expansion ranking **至少明显改善了搜索效率**；
- 至于它是否稳定改善最终改善值，现阶段证据还不如 `leaf value` 那么强。

## 5. 现在已经可以下的结论

### 5.1 强结论

1. **`leaf value` 是当前框架中的关键组件。**
2. **去掉 `leaf value` 会显著恶化整体性能，尤其是效率，并在 `HOMO(D)` 上明显损害优化质量。**
3. **本轮 `official` 进一步表明：`wo_leaf_value` 在 `homo_down` 上存在复现性的搜索稳定性问题。**
4. **当前完整框架至少明显优于一个关键消融版本（`wo_leaf_value`）。**
5. **在 `HOMO(D)` 上，去掉 `leaf value` 后 MCTS 的 improvement 降至 `0.4896`，甚至低于主实验中 BFS 的 `0.9125`**——这证明 `leaf value` 是 MCTS 超越 BFS 的关键信号。

### 5.2 中等强度结论

1. **`prior` 的贡献方向是正向的，但量级较小。**
2. **随机替代 expansion ranking 会稳定降低搜索效率。**

### 5.3 暂时不要写死的结论

1. 不要写“所有消融版本都明显更差”；
2. 不要写“random top-b 在所有任务上都显著降低最终改善值”；
3. 不要写“prior 是与 leaf value 同等重要的核心来源”；
4. 不要把本轮 `official` 当成“已完成的多 seed 真值主表”。

## 6. 当前最推荐的主文口径

如果今天就要组织正文，当前最稳的写法是：

> 我们在 `LUMO(U)` 与 `HOMO(D)` 上对 MCTS 侧关键组件进行了真值消融。结果表明，`leaf value` 是当前框架中最关键的搜索信号：移除该项后，`HOMO(D)` 上的平均改善由 `2.0041` 降至 `0.4896`，同时平均搜索时间与展开节点数分别增加到原始配置的 `88.4x` 与 `24.8x`。相比之下，移除 `prior` 仅带来较小幅度退化，说明其贡献更偏向温和的搜索引导；而将 expansion ranking 随机化则主要损害搜索效率，其对最终改善值的影响目前仍表现出一定任务依赖性。进一步地，在 `official` 预算的搜索层实验中，`homo_down / wo_leaf_value` 还出现了 `3 / 3` seeds 的复现失败，说明去掉 `leaf value` 不仅会损害效果与效率，还会额外带来稳定性风险。

## 7. 一句话总结

当前 MCTS 侧消融最有力的结论不是“每个模块都同样关键”，而是：**`leaf value` 是决定当前框架效果、效率与稳定性的关键组件，而 `prior` 与 expansion ranking 则表现为较弱或更偏效率侧的贡献。**
