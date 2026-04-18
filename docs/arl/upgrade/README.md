# A* RL Demo — 升级计划总览
<!-- last-updated: 2026-04-12 -->

本文档整理当前 `astar_demo` RL 路线的升级建议，目标不是简单“换一个更高级的 RL 名字”，
而是先把**真实决策闭环、训练稳定性、搜索-学习耦合方式**理顺，再决定是否进入更重的算法升级。

---

## 当前判断

基于目前代码与实验现状，当前链路的主要问题不是“完全没有 RL”，而是：

- **当前在线训练本质仍是 `BC -> REINFORCE + Value baseline` 的 demo 版本**。
- **policy 没有真正决定被训练的动作**：搜索里更像是在给 `top_scored[0]` 背书，而不是对 policy 的真实采样结果做 credit assignment。
- **reward 与 policy 决策没有完全对齐**，导致即便继续堆 episode，也可能只是更稳定地学习到偏掉的目标。
- **搜索本身已经很强**，因此后续优化不应只盯着 RL loss，而要同时考虑 `policy/value` 与 `A* / MCTS / OFO` 的耦合方式。

一句话说：

> **当前优先级不是“立刻上更复杂算法”，而是先把 RL 变成真正的 RL 闭环；在这个前提下，首选升级方向是 `PPO + GAE`；如果继续强调搜索主导，则应进一步走向 search-guided policy improvement。**

---

## 推荐优先级

### P0：先修训练闭环

先解决“谁做决策、谁拿 reward、谁被更新”这三件事的一致性。

- 把当前固定 `selected_idx = 0` 的逻辑改成**真实采样或受控采样**。
- 保证 reward 对应的是 **policy 实际选中的 action**，而不是搜索排序后的 top1。
- 让 trajectory 里的 `state / action / log_prob / reward / done` 真正构成可学习样本。

> 这是后续一切 PPO / search-guided / ranking 升级的前置条件。

### P1：把 REINFORCE 升级为 `PPO + GAE`

这是最推荐的主线升级。

原因：

- 当前动作空间是**每步一组离散候选动作**，很适合 categorical policy；
- 已经有 `PolicyNet`、`ValueNet`、trajectory buffer 与训练脚手架；
- 相比 REINFORCE，PPO 对高方差回报和偶发大样本更稳。

### P2：引入 search-guided policy improvement

如果后续继续把 `MCTS / A*` 当作核心能力，而不是把搜索退化成 RL 环境，那么更自然的升级是：

- 用搜索输出更强的 action distribution / visit counts；
- policy 学习搜索改进后的 target；
- value 学习状态的未来可达收益；
- 在线 RL 退居为辅助而不是唯一学习信号。

### P3：把 bandit / ranking 作为务实备选路线

如果目标更偏向“在固定 OFO 预算下提升 prefilter 质量”，那么 contextual bandit / learning-to-rank 可能比 full RL 更划算。

适用场景：

- 更关心 `top_n_prefilter` 质量；
- horizon 不长；
- 主要收益来自候选排序，而不是长程 credit assignment。

---

## 我不建议优先投入的方向

### 不建议优先：`SAC / DDPG / TD3`

原因：

- 当前动作空间是**变长离散候选集合**，不是连续控制；
- 需要额外做 candidate-Q 设计，工程复杂度高；
- 很可能比 PPO 更难训，也更难解释。

### 不建议继续：单纯把 REINFORCE 的 episode 数继续堆大

原因：

- 高方差问题不会因为多跑就自动消失；
- 若 reward 对齐仍有偏差，只会更稳定地放大偏差；
- 容易出现 `episode_return` 好看但 holdout 不涨的情况。

---

## 建议拆分为三份执行文档

| 文档 | 目的 | 什么时候做 |
|------|------|-------------|
| [01-close-rl-loop.md](./01-close-rl-loop.md) | 修正当前在线 RL 的真实闭环 | **立刻开始** |
| [02-ppo-gae-upgrade.md](./02-ppo-gae-upgrade.md) | 在闭环正确后升级成更稳的 on-policy 训练 | P0 完成后 |
| [03-search-guided-policy.md](./03-search-guided-policy.md) | 把搜索结果变成 policy/value 的更强监督信号 | PPO 稳定后或并行预研 |

---

## 推荐执行顺序

1. **先完成 P0**：让 policy 真采样、reward 真对齐、trajectory 真可学。
2. **再完成 P1**：落地 `PPO + GAE`，并用固定 holdout 替代 `episode_return` 作为主选优标准。
3. **最后决定 P2/P3**：
   - 如果继续强调搜索主导，优先做 search-guided；
   - 如果更看重低风险增益与预算控制，可转 bandit / ranking。

---

## 验收标准

升级计划不是以“loss 下降”作为完成标准，而是以下结果：

- **训练闭环正确**：policy 的 sampled action 能对应上 log-prob、reward 与 update。
- **评估更稳**：holdout 上 `top1_median / trimmed_mean / win_rate_vs_bc` 更稳定。
- **搜索耦合更合理**：policy/value 与 `A* / MCTS` 的作用边界更清楚，不再互相背锅。
- **实验可迭代**：每一步升级都能独立回滚、独立评估、独立复现实验结论。

---

## 一句话结论

> **先把当前 demo 版 REINFORCE 修成“真正的 RL 闭环”，再优先升级到 `PPO + GAE`；如果长期仍以搜索为核心，则应把重点逐步转到 search-guided policy improvement，而不是继续死磕高方差在线 REINFORCE。**
