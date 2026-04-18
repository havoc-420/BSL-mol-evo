# A* RL Demo 文档索引
<!-- last-updated: 2026-04-11 -->

基于 A* 搜索 + PolicyNet/ValueNet 引导的分子优化 RL Demo 文档目录。

| 文档 | 内容 |
|------|------|
| [overview.md](./overview.md) | 系统概览：模块地图、数据流、f_score 公式、关键技术决策 |
| [usage.md](./usage.md) | 使用指南：各阶段详细命令、所有 CLI 参数说明、输出文件结构、常见问题 |
| [testing.md](./testing.md) | 测试指南：测试文件列表、每个 case 验证点、运行方式、手动 smoke test |
| [experiments.md](./experiments.md) | 实验推进手册：Step-by-Step 路线、消融方向、结果记录模板、失败排查 |
| [stage_scores.md](./stage_scores.md) | 分阶段实验得分台账：按阶段记录 BC、RL、网格评估与后续扩规模结果 |
| [plan_scaleup_a.md](./plan_scaleup_a.md) | 方案 A：低风险扩规模计划，包含当前结果冻结、实验矩阵、指标与交付物 |
| [upgrade/README.md](./upgrade/README.md) | RL 升级计划总览：闭环修正、`PPO + GAE`、search-guided policy improvement 三阶段路线 |

---

## 当前状态（2026-04-11 18:16）

- **离线导出桥已修复**：`export_rl_demo_transitions.py` 已修复 `property_change` / `predicted_change` 字段错读，并统一 `operation/details` 格式；当前所有 go / no-go 判断已切换到 fixed 口径。
- **fixed BFS 基线已全部完成**：
  - `a0_fixed`：`top1_mean=3.0514`，`top1_median=3.1228`，`trimmed_mean=2.9509`
  - `a1_small_fixed`：`top1_mean=3.0438`，`top1_median=2.9572`，`trimmed_mean=3.0420`，相对 `a0_fixed`：`win_rate=54%`，`trimmed_delta_mean=+0.1221`
  - `a1_main_fixed`：`top1_mean=2.9425`，`top1_median=2.9894`，`trimmed_mean=2.9669`，相对 `a0_fixed`：`win_rate=54%`，`trimmed_delta_mean=+0.0457`
- **MCTS 小规模基石结果已重对齐到 fixed A0**：`A1-mcts-main` 只有 `1089` 条 transition，但在 fixed 口径下 `top1_mean=3.0971`，`top1_median=3.0167`，相对 `a0_fixed`：`win_rate=50%`，`trimmed_delta_mean=+0.0538`。这说明它**不再明显弱于 BFS 基线**，而是“数据偏少但已经接近持平”。
- **当前主任务仍在推进**：`mcts-expand-rl` 已跑到训练池 `376 / 983`（`38.3%`），执行 `MCTS 扩规模 -> BC -> holdout -> RL(300 ep) -> holdout` 全流程。
- **当前判断**：修复后没有任何一条 BFS 扩规模支线对 `a0_fixed` 形成压倒性优势，因此继续推进更厚的 `MCTS` 数据源是合理的下一步。
- **建议入口**：最新进度先看 `stage_scores.md`，实验判断看 `experiments.md`，执行计划看 `plan_scaleup_a.md`。

---

## 5 分钟快速入口

```bash
# 1. smoke check（无需模型权重）
python mol_evo/tests/test_rl_training.py
python mol_evo/tests/test_astar_demo_search_mode.py

# 2. 用随机权重跑 5 个分子，验证流程
python -m mol_evo.scripts.batch_optimizer \
  --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path $OFO_MODEL --model-dir $OFO_MODEL_DIR \
  --config-file $OFO_CONFIG \
  --search-mode astar_demo --rl-eval \
  --open-set-budget 30 --max-depth 2 \
  --start-index 0 --end-index 5

# 3. 导出 BC 数据 → 冷启动训练 → 评估
#    详见 usage.md
```
