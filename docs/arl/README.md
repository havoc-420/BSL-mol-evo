# A* RL Demo 文档索引
<!-- last-updated: 2026-04-10 -->

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

## 当前状态（2026-04-10）

- **BC 冷启动已完成**：15 棵 BFS 树 → 9090 条样本 → `bc_20260409_145233`
- **在线 RL 已跑通**：100 episode run 位于 `lumo_rl_bfs15_depth3_run100/rl_20260409_161937`
- **P0 + P1 已完成**：`50` 分子、`12` 组网格（`BC / RL × budget × prefilter`）结果已落在 `lumo_eval_grid50_parallel_20260409_194105`
- **当前结论**：`top_n_prefilter=50` 明显优于 `20`；RL 在部分搜索参数上已能超过 BC 的**均值**，但最佳 `rl 50/50` 仍受少数大样本拉动，尚不能判定为“稳定全面领先”
- **结果归档入口**：看 `experiments.md` 中的“实验结果归档（2026-04-10）”与 `Step 6`
- **得分台账入口**：看 `stage_scores.md`，集中记录不同阶段的实验分数与主指标
- **下一步计划**：看 `plan_scaleup_a.md`，先执行方案 A 的低风险扩规模版本

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
