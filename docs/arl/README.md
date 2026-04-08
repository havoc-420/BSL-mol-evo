# A* RL Demo 文档索引
<!-- last-updated: 2026-04-09 -->

基于 A* 搜索 + PolicyNet/ValueNet 引导的分子优化 RL Demo 文档目录。

| 文档 | 内容 |
|------|------|
| [overview.md](./overview.md) | 系统概览：模块地图、数据流、f_score 公式、关键技术决策 |
| [usage.md](./usage.md) | 使用指南：各阶段详细命令、所有 CLI 参数说明、输出文件结构、常见问题 |
| [testing.md](./testing.md) | 测试指南：测试文件列表、每个 case 验证点、运行方式、手动 smoke test |
| [experiments.md](./experiments.md) | 实验推进手册：Step-by-Step 路线、消融方向、结果记录模板、失败排查 |

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
