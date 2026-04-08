# A* RL Demo — 使用指南
<!-- last-updated: 2026-04-09 -->

所有命令以**项目根目录**（`mol-ofo/`）为工作目录执行。

---

## 快速上手（跳过 BC，直接评估）

如果只想验证 `astar_demo` 搜索能否跑通，不需要训练，直接：

```bash
python -m mol_evo.scripts.batch_optimizer \
  --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path /path/to/ofo_model.pth \
  --model-dir  /path/to/ofo_model_dir \
  --config-file /path/to/config.yaml \
  --search-mode astar_demo \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --open-set-budget 50 \
  --start-index 0 --end-index 5
```

> 此时 PolicyNet / ValueNet 使用随机初始化，结果质量不保证，仅验证流程正确性。

---

## Phase 0 — 准备 BC 数据

先用 BFS 跑一批分子，保存搜索树 JSON：

```bash
python -m mol_evo.scripts.batch_optimizer \
  --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path /path/to/ofo_model.pth \
  --model-dir  /path/to/ofo_model_dir \
  --config-file /path/to/config.yaml \
  --search-mode bfs \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --start-index 0 --end-index 200
# 输出写入 mol_evo/output/evo-mo/batch_optimization_<timestamp>/
```

从搜索树提取 BC 训练样本：

```bash
python mol_evo/dataset/export_rl_demo_transitions.py \
  --input-dir  mol_evo/output/evo-mo/batch_optimization_<timestamp> \
  --output-json mol_evo/dataset/rl_demo/bc_transitions.json \
  --direction decrease \
  --max-depth 3 \
  --logp-min 0.0 --logp-max 5.0
```

导出脚本参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | 必填 | `batch_optimizer` 的输出目录，含 `*.json` 树文件 |
| `--output-json` | `mol_evo/dataset/rl_demo/bc_transitions.json` | BC 样本输出路径 |
| `--direction` | `decrease` | 属性优化方向 |
| `--max-depth` | `4` | 与搜索时的 `--max-depth` 一致 |
| `--max-trees` | 不限 | 最多处理的树文件数量，调试时可设小值 |
| `--target-property` | `lumo` | 仅用于日志，不影响数据内容 |

---

## Phase 1 — BC 冷启动

```bash
python mol_evo/train_bc_pretrain.py \
  --data-json   mol_evo/dataset/rl_demo/bc_transitions.json \
  --output-dir  mol_evo/output/astar_rl/bc \
  --direction   decrease \
  --epochs      100 \
  --batch-size  64 \
  --lr-policy   1e-4 \
  --lr-value    1e-3 \
  --patience    20 \
  --val-ratio   0.1 \
  --device      auto
```

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-json` | 必填 | Phase 0 导出的 BC 样本 JSON |
| `--output-dir` | `mol_evo/output/astar_rl/bc` | 权重和日志输出目录 |
| `--direction` | `decrease` | 与数据导出时一致 |
| `--epochs` | `100` | 训练轮数 |
| `--batch-size` | `64` | 批大小，in-batch 负样本数 = batch_size - 1 |
| `--lr-policy` | `1e-4` | PolicyNet 学习率 |
| `--lr-value` | `1e-3` | ValueNet 学习率 |
| `--patience` | `20` | 验证集早停耐心值 |
| `--hidden-dim` | `128` | 两个网络的隐层宽度 |
| `--num-layers` | `3` | 隐层数 |
| `--dropout` | `0.1` | Dropout 比例 |
| `--val-ratio` | `0.1` | 验证集划分比例 |
| `--device` | `auto` | `cpu` / `cuda` / `auto` |

训练完成后输出：

```
mol_evo/output/astar_rl/bc/
├── policy_best.pth       # 验证集最优 PolicyNet 权重
├── value_best.pth        # 验证集最优 ValueNet 权重
├── bc_history.json       # 每轮 train/val loss 记录
└── bc_config.json        # 训练配置快照
```

---

## Phase 2 — 在线 RL 微调

```bash
python mol_evo/train_astar_rl_demo.py \
  --input-csv   mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path  /path/to/ofo_model.pth \
  --model-dir   /path/to/ofo_model_dir \
  --config-file /path/to/config.yaml \
  --policy-path mol_evo/output/astar_rl/bc/policy_best.pth \
  --value-path  mol_evo/output/astar_rl/bc/value_best.pth \
  --output-dir  mol_evo/output/astar_rl/rl \
  --direction   decrease \
  --num-episodes 200 \
  --max-depth   4 \
  --max-branching 8 \
  --open-set-budget 200 \
  --top-n-prefilter 20 \
  --lr-policy   1e-4 \
  --lr-value    1e-3 \
  --gamma       0.99 \
  --entropy-coef 0.01 \
  --checkpoint-every 50 \
  --device auto
```

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-csv` | 必填 | 含 `smiles` 列的分子 CSV |
| `--model-path` | 必填 | OFO 模型权重（.pth） |
| `--model-dir` | 必填 | OFO 模型目录 |
| `--config-file` | 必填 | OFO 配置文件（.yaml） |
| `--policy-path` | `None` | BC 预训练权重，不指定则随机初始化 |
| `--value-path` | `None` | BC 预训练权重，不指定则随机初始化 |
| `--num-episodes` | `200` | 在线训练 episode 总数（每个 = 一次完整搜索） |
| `--open-set-budget` | `200` | A* open set 展开预算 |
| `--top-n-prefilter` | `20` | PolicyNet 每步预筛保留的候选数 |
| `--gamma` | `0.99` | 折扣因子 |
| `--entropy-coef` | `0.01` | 熵正则系数，防止 policy 过早收敛 |
| `--checkpoint-every` | `50` | 每 N 个 episode 保存一次 checkpoint |
| `--target-property` | `lumo` | CSV 中目标属性列名 |
| `--optimization-mode` | `sub` | `sub`（绝对值改善）或 `pct`（百分比改善） |

训练完成后输出：

```
mol_evo/output/astar_rl/rl/rl_<timestamp>/
├── policy_best.pth       # episode_return 最优时的 PolicyNet 权重
├── value_best.pth        # 对应的 ValueNet 权重
├── policy_last.pth       # 最后一个 episode 的 PolicyNet 权重
├── value_last.pth        # 最后一个 episode 的 ValueNet 权重
├── checkpoints/
│   └── rl_ckpt_ep<N>.pth # 每隔 checkpoint_every 保存的快照
├── rl_history.json       # 每个 episode 的 return/loss/steps 记录
├── rl_config.json        # 训练配置快照
└── train_rl.log          # 日志（console + file 双输出）
```

---

## Phase 3 — 批量评估

加载训练好的权重，与 BFS / MCTS 做对比：

```bash
# astar_demo 纯评估（不更新权重）
python -m mol_evo.scripts.batch_optimizer \
  --input-csv   mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path  /path/to/ofo_model.pth \
  --model-dir   /path/to/ofo_model_dir \
  --config-file /path/to/config.yaml \
  --search-mode astar_demo \
  --rl-eval \
  --policy-path mol_evo/output/astar_rl/rl/rl_<timestamp>/policy_best.pth \
  --value-path  mol_evo/output/astar_rl/rl/rl_<timestamp>/value_best.pth \
  --direction   decrease \
  --max-depth   4 \
  --max-branching 8 \
  --open-set-budget 200 \
  --topK 20

# BFS 基线（同等参数）
python -m mol_evo.scripts.batch_optimizer \
  --search-mode bfs \
  --max-depth 4 --max-branching 8 \
  ...

# MCTS 基线
python -m mol_evo.scripts.batch_optimizer \
  --search-mode mcts \
  --num-simulations 200 \
  ...
```

`astar_demo` 专属 batch_optimizer 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--search-mode astar_demo` | — | 启用 A* RL 搜索 |
| `--rl-eval` | flag | 纯评估模式，加载权重不更新 |
| `--rl-train` | flag | 在线 RL 模式，边搜索边更新 |
| `--policy-path` | `None` | PolicyNet 权重路径 |
| `--value-path` | `None` | ValueNet 权重路径 |
| `--top-n-prefilter` | `20` | PolicyNet 预筛保留候选数 |
| `--open-set-budget` | `200` | A* 展开预算上限 |

---

## 常见问题

**Q: `policy_path` / `value_path` 不指定会怎样？**  
A: 两个网络均随机初始化。搜索能跑通，但排序质量差。BC 冷启动后质量会明显提升。

**Q: `--rl-train` 和 `--rl-eval` 可以同时用吗？**  
A: 不可以。`--rl-train` 优先级更高。一般评估实验用 `--rl-eval`；消融时用 `--rl-train` 观察在线效果。

**Q: 搜索很慢，如何加速？**  
A: 减小 `--open-set-budget`（如 50）和 `--top-n-prefilter`（如 10）。搜索预算越小越快，但质量下降。

**Q: `e3nn` / `torch_geometric` 缺失导致 import 失败？**  
A: `astar_rl` 包不依赖这些库。失败通常来自 `mol_evo/core/models/__init__.py` 里的其他模型注册。可以直接 `import mol_evo.core.models.astar_rl` 跳过主 `__init__`。
