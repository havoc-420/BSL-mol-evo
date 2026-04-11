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

---

## 当前状态（2026-04-11 15:53）

- **离线导出桥已修复**：`export_rl_demo_transitions.py` 已修复 `property_change` / `predicted_change` 字段错读，并统一 `operation/details` 格式；因此 `2026-04-09 ~ 2026-04-10` 的早期 BFS / A1 BC 数据不再作为最新基线。
- **fixed BFS 重跑中**：
  - `fix-a0-bc`：holdout 原始 JSON 已写出，当前快照 `40 / 50` 分子，`top1_mean=3.1797`，`top1_median=3.1843`。
  - `fix-a1-small-bc`：holdout 原始 JSON 已写出，当前快照 `15 / 50` 分子，`top1_mean=3.1099`，`top1_median=2.9432`。
  - `fix-a1-main-bc`：`53269` 条 fixed transition 已导出，BC 重训进行中。
- **MCTS 基石试跑已完成**：`A1-mcts-main` 导出 `1089` 条全非零 transition，holdout `top1_mean=3.0971`，`top1_median=3.0167`，相对参考 `win_rate=0.34`；当前判断是**数据质量正常，但树太稀、样本量偏少**。
- **新主任务已启动**：`mcts-expand-rl` 已在 `tmux` 中启动，执行 `MCTS 扩规模 -> BC -> holdout -> RL(300 ep) -> holdout` 全流程。
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
