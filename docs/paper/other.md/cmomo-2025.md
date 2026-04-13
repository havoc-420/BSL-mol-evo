## CMOMO

`CMOMO` 是一个面向**受约束多性质分子优化**的深度多目标优化框架，重点解决“同时提升多个性质”与“满足药物可行性约束”之间的冲突。它将优化过程划分为两个阶段：先在 unconstrained scenario 中强调多个目标性质的联合改进，再在 constrained scenario 中通过**动态约束处理策略**逐步强化对约束条件的满足，以在收敛性与可行性之间取得平衡。方法上，`CMOMO` 先基于 lead molecule 和高性质相似分子构建 `Bank library` 初始化种群，再借助预训练编码器把分子映射到连续潜空间，在该空间中采用 **latent vector fragmentation based evolutionary reproduction** 生成新候选，并结合 `NSGA-II` 的环境选择维护 Pareto 收敛性与群体多样性。总体来看，`CMOMO` 的优势在于把多目标优化、约束满足和潜空间进化统一到同一框架中，因此相比只关注性质提升或简单丢弃不可行分子的方案，更适合实际药物设计中复杂、受限的多属性优化任务。
