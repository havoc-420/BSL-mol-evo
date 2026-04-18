# P1：从 REINFORCE 升级到 PPO + GAE
<!-- last-updated: 2026-04-12 -->

本文档描述在 P0 闭环修正完成后，如何把当前 `astar_demo` 在线训练从 `REINFORCE + Value baseline` 升级为更稳的 `PPO + GAE`。

---

## 为什么首选 PPO + GAE

当前任务满足 PPO 最适合的一类条件：

- 动作空间是**离散候选集合**；
- 每个状态本来就会生成一批 candidate actions；
- 已经有 `PolicyNet` 与 `ValueNet`；
- 当前最大的痛点是高方差、checkpoint 波动大、`episode_return` 与 holdout 不稳定。

相比继续使用 REINFORCE，PPO 的价值在于：

- 对极端 return 更稳；
- 更新幅度可控；
- 能重复利用 on-policy 样本做多 epoch 更新；
- 与现有代码的结构兼容性较好。

---

## 目标

把当前训练升级成一套标准的 on-policy 流程：

1. rollout 收集 trajectory
2. 计算 `returns` 与 `advantages`
3. 使用 `old_log_prob` 构造 PPO clipped objective
4. 多轮 mini-batch 更新 `policy / value`
5. 用固定 holdout 而不是 `episode_return` 选 best checkpoint

---

## 核心设计

### 1. rollout 数据协议

建议每条样本至少记录：

- `state_vec`
- `action_vecs`
- `selected_idx`
- `old_log_prob`
- `reward`
- `value_pred`
- `done`

如果实现上更方便，也可以在 episode 结束后统一补算 `value_pred`。

### 2. Advantage 计算

建议采用 `GAE(lambda)`：

- 降低方差；
- 比纯 Monte Carlo return 更稳；
- 更适合 episode 长短不完全一致的情况。

建议预留超参：

- `gamma`
- `gae_lambda`
- `normalize_advantage`

### 3. PPO loss

建议最小化以下组合：

- **policy loss**：clipped surrogate objective
- **value loss**：MSE 或 clipped value loss
- **entropy bonus**：保持必要探索

建议预留超参：

- `clip_ratio`
- `value_coef`
- `entropy_coef`
- `update_epochs`
- `minibatch_size`
- `max_grad_norm`

### 4. checkpoint 选择规则

不要继续只看 `episode_return`。

建议优先级：

1. 固定 holdout 的 `top1_median`
2. `trimmed_mean`
3. `win_rate_vs_bc`
4. 训练曲线只作为辅助观察

---

## 实施步骤

### Step 1：补全 `PPORLTrainer`

当前代码里已经有 `PPORLTrainer` 入口，但仍是 fallback。

P1 的第一步不是新起炉灶，而是：

- 复用当前 `RLTrainer` 的公共结构；
- 新增 rollout buffer；
- 正式实现 PPO 的 update 部分。

### Step 2：加入 GAE 与 advantage normalization

这是 PPO 稳定性的关键部分。

最小要求：

- 按 episode 或按 rollout 计算 advantage；
- 支持 standardization；
- logging 中输出 advantage 均值、方差、极值。

### Step 3：支持多 epoch / mini-batch 更新

相较当前“一次 episode 一次更新”，建议升级为：

- 收集一定量 rollout；
- 拆成 mini-batch；
- 多轮更新 policy/value；
- 每轮记录 KL、clip fraction、entropy 等稳定性指标。

### Step 4：补实验与回滚机制

建议每次升级保留与 REINFORCE 的直接对照：

- 相同 seed
- 相同 BC 初始权重
- 相同 holdout
- 相同 `budget / prefilter`

确保 PPO 失败时可以快速回滚，而不是把训练环境一起搞乱。

---

## 代码落点建议

建议主要修改：

- `mol_evo/core/models/astar_rl/rl_trainer.py`
- `mol_evo/train_astar_rl_demo.py`
- `mol_evo/core/models/astar_rl/value_network.py`
- `mol_evo/docs/arl/testing.md`
- `mol_evo/docs/arl/experiments.md`

如需要，也可增加：

- `rollout_buffer` 相关辅助类；
- PPO 专属日志汇总模块。

---

## 评估矩阵

### 必做对照

- **REINFORCE vs PPO**：固定同一 BC 初始化与 holdout；
- **PPO 不同 seeds**：至少 `3` 个；
- **是否做 GAE**：有条件可做小消融。

### 主要指标

- holdout `top1_mean`
- holdout `top1_median`
- holdout `trimmed_mean`
- `win_rate_vs_bc`
- `nonempty_topk`
- 训练稳定性指标：entropy、clip fraction、approx KL、value loss

---

## 风险与止损

### 风险

- PPO 实现变复杂后，调参空间明显增大；
- 若 P0 没做好，PPO 只会更稳定地学习偏目标；
- 若搜索自身支配结果太强，PPO 的收益可能被淹没。

### 止损条件

若出现以下任一情况，应暂停继续调 PPO，而回头检查 P0 或转向 P2：

- PPO 多 seed 下对 holdout 没有稳定优于 REINFORCE；
- entropy 快速塌缩，策略退化成近乎纯 greedy；
- `episode_return` 看起来变好，但 holdout 稳健指标不涨。

---

## 完成标准

P1 完成时，至少应达到：

- `PPORLTrainer` 不再 fallback 到 REINFORCE；
- 训练过程能稳定输出 PPO 关键诊断指标；
- 固定 holdout 上结果波动小于当前 REINFORCE 基线；
- 至少一个配置在 `top1_median / trimmed_mean / win_rate_vs_bc` 上表现出正信号。

---

## 一句话结论

> **在当前任务形态下，`PPO + GAE` 是最值得优先落地的算法升级，因为它能在不推翻现有 `PolicyNet / ValueNet / A*` 架构的前提下，显著改善高方差和 checkpoint 不稳定的问题。**
