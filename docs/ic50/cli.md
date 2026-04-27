# IC50 批量分子优化 CLI 速查

脚本：`mol_evo/scripts/optimization/batch_optimizer_ic50.py`
入口函数：`EvolutionTreeOptimizer.optimize_evolution_tree`（`mol_evo/core/evolution_optimizer_ic50.py`）

## 1. 运行前提（MANDATORY）

- **工作目录**：必须是仓库根 `mol-ofo/`（不是 `mol-ofo/mol_evo/`）。输出目录按 `os.getcwd()/mol_evo/output/evo-mo/ic50/batch_optimization_<ts>/` 创建。
- **PYTHONPATH**：MCTS 版脚本位于 `scripts/optimization/`，`sys.path[0]` 不指向仓库根，必须显式：
  ```bash
  export PYTHONPATH=/root/autodl-tmp/projects/mol-ofo:$PYTHONPATH
  ```
- **Conda 环境**：`mol-edit`（不是 `schnet_drp`）。SSH 非交互式 shell 需先：
  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate mol-edit
  ```
- **长任务**：走 tmux，避免 SSH 断连杀进程。单分子硬超时 `3h`（`timeout_seconds=600*6*3`，见 `batch_process`）。

## 2. 参数清单

### 2.1 必填 / 数据与模型路径
s
|---|---|---|---|
| `--input-json` | str | 示例路径 | 输入 JSON：`{cell_name: [{smiles, ic50, ...}, ...]}` |
| `--cell-name` | str | **必填** | 从 JSON 中取的细胞系 key（如 `906792`） |
| `--model-path` | str | 示例路径 | DRP 模型权重 `.pth` |
| `--model-dir` | str | 示例路径 | 模型目录（含 config） |
| `--config-file` | str | `mol_evo/core/drp_ic50_p/ic50.yaml` | DRP 推理配置 |
| `--output-json` | str | 自动 | 不填则落 `<output_dir>/batch_results.json` |

### 2.2 优化语义

| 参数 | 可选值 / 默认 | 说明 |
|---|---|---|
| `--target-property` | `ic50` | 目标属性名，从 `original_row[target_property]` 取初始值 |
| `--direction` | `decrease` / `increase`（默认 `decrease`） | 降 IC50 用 `decrease` |
| `--optimization-mode` | `sub` / `pct`（默认 `sub`） | 优化指标：绝对差 / 百分比 |
| `--topK` | int，默认 `100` | 每个分子保留的最好 K 个结果 |

### 2.3 BFS 剪枝相关

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-depth` | `2` | 演化最大深度（代数） |
| `--max-branching` | `8` | 每节点最大分支 |
| `--pruning-patience` | `2` | 属性无改进多少代即剪枝 |
| `--logp-min` / `--logp-max` | `0.0` / `5.0` | logP 约束区间 |
| `--logp-patience` | `3` | logP 连续越界多少代即剪枝 |

### 2.4 MCTS 搜索（`--search-mode mcts` 生效）

| 参数 | 可选值 / 默认 | 说明 |
|---|---|---|
| `--search-mode` | `bfs` / `mcts`（默认 `bfs`） | 切换搜索算法 |
| `--num-simulations` | int，默认 `200` | MCTS 模拟总轮数 |
| `--exploration-weight` | float，默认 `1.4` | PUCT 探索系数 c |
| `--mcts-prior-mode` | `softmax` / `uniform`（默认 `softmax`） | prior 构造：按 DRP 打分 softmax 或均匀 |
| `--mcts-value-mode` | `accumulated` / `zero` / `step`（默认 `accumulated`） | 叶节点价值：累积改进 / 0 / 单步改进 |
| `--mcts-expansion-mode` | `topk` / `random_topk` / `full`（默认 `topk`） | 扩展策略：Top-K / 随机 Top-K / 全扩展 |
| `--mcts-random-seed` | int，默认 `None` | 主要给 `random_topk` 用，保证复现 |

### 2.5 批处理 / 恢复

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch-size` | `10` | 当前脚本未显式消费，保留字段 |
| `--resume-from` | `None` | 传已有输出目录则进入恢复模式，跳过已处理分子，按 `initial_smiles` 去重 |

## 3. 输出目录结构

```
mol_evo/output/evo-mo/ic50/batch_optimization_<ts>/
├── batch_optimization_main.log      # 全局配置 + 总结
├── batch_optimization_total.log     # 每个分子开始/结束/状态
├── <smiles20>_<uuid8>.json          # 单分子完整优化树
├── <smiles20>_<uuid8>_topK.csv      # 单分子 topK
└── batch_results.json               # 聚合结果（每 5 个分子增量写一次）
```

- 单分子硬超时 `3h`，触发 `terminate()`/`kill()` + `result['status']='timeout'`。
- `SIGINT`（Ctrl+C）第一次优雅终止子进程，第二次强退。

## 4. 命令模板（复制即用）

### 4.1 BFS 基线

```bash
cd /root/autodl-tmp/projects/mol-ofo
export PYTHONPATH=$(pwd):$PYTHONPATH
python mol_evo/scripts/optimization/batch_optimizer_ic50.py \
  --input-json mol_evo/dataset/data/gdscv2/ic50_result_dict_906792.json \
  --cell-name 906792 \
  --model-path /root/autodl-tmp/projects/schnet_drp/outputs/SchNet_906792/0.9_N4_20260427_163833/checkpoints/best_model.pth \
  --model-dir  /root/autodl-tmp/projects/schnet_drp/outputs/SchNet_906792/0.9_N4_20260427_163833 \
  --config-file mol_evo/core/drp_ic50_p/ic50.yaml \
  --target-property ic50 --direction decrease \
  --max-depth 6 --max-branching 8 --topK 100 \
  --search-mode bfs
```

### 4.2 MCTS 推荐配置

```bash
python mol_evo/scripts/optimization/batch_optimizer_ic50.py \
  --input-json mol_evo/dataset/data/gdscv2/ic50_result_dict_906792.json \
  --cell-name 906792 \
  --model-path <.../best_model.pth> --model-dir <.../> \
  --config-file mol_evo/core/drp_ic50_p/ic50.yaml \
  --target-property ic50 --direction decrease \
  --max-depth 10 --max-branching 20 --topK 100 \
  --search-mode mcts \
  --num-simulations 1600 --exploration-weight 1.4 \
  --mcts-prior-mode softmax \
  --mcts-value-mode accumulated \
  --mcts-expansion-mode topk
```

### 4.3 远程 tmux 一行启动

```bash
ssh seetacloud-2 "tmux kill-session -t mo-mcts 2>/dev/null; \
tmux new-session -d -s mo-mcts -c /root/autodl-tmp/projects/mol-ofo \
'source /root/miniconda3/etc/profile.d/conda.sh && conda activate mol-edit && \
 export CUDA_VISIBLE_DEVICES=0 && \
 export PYTHONPATH=/root/autodl-tmp/projects/mol-ofo:\$PYTHONPATH && \
 python mol_evo/scripts/optimization/batch_optimizer_ic50.py \
   --input-json mol_evo/dataset/data/gdscv2/ic50_result_dict_906792.json \
   --cell-name 906792 \
   --model-path <...> --model-dir <...> \
   --config-file mol_evo/core/drp_ic50_p/ic50.yaml \
   --target-property ic50 --direction decrease \
   --max-depth 6 --max-branching 8 --topK 100 \
   --search-mode mcts --num-simulations 400 --exploration-weight 1.4 \
   --mcts-prior-mode softmax --mcts-value-mode accumulated --mcts-expansion-mode topk \
   2>&1 | tee -a mol_evo/logs/mcts_906792_\$(date +%Y%m%d_%H%M%S).log; exec bash'"
```

### 4.4 恢复未跑完的批次

```bash
python mol_evo/scripts/optimization/batch_optimizer_ic50.py \
  --input-json <same> --cell-name 906792 \
  --model-path <...> --model-dir <...> --config-file <...> \
  --resume-from mol_evo/output/evo-mo/ic50/batch_optimization_20260427_184255
```

## 5. 常见坑

| 现象 | 根因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'mol_evo'` | cwd 不是仓库根 或 脚本在 `scripts/optimization/`，`sys.path[0]` 指向 `optimization/` | `cd mol-ofo/` + `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| 日志只到 "读取到 N 个分子" 就卡住 | DRP 模型首次加载 + CUDA 初始化，属正常启动耗时 | 等 1~2 min，看 `batch_optimization_total.log` |
| 单分子跑满 3h 被 kill | 硬超时保护 | 调低 `--num-simulations` / `--max-branching` / `--max-depth` |
| MCTS 结果不可复现 | `random_topk` 扩展 + 无种子 | 指定 `--mcts-random-seed <int>` |
| KeyError `target_property` | JSON 条目缺该属性列 | 检查输入 JSON 每条是否有 `ic50` 字段 |
| MCTS smoke test 只跑一两步 | `--num-simulations` 太小 / `max-depth` 太小 | 最小验证建议 `num-simulations>=20, max-depth>=3` |

## 6. 代码入口对应关系

- CLI 参数 → `parse_args()`（脚本内）
- 每分子流程 → `run_evolution_optimizer` → `optimizer.optimize_evolution_tree(...)`
- MCTS 核心 → `evolver.generate_expansion_tree_mcts(...)`（`core/molecular_evolution_expansion_ic50.py`）
- BFS 核心 → `evolver.generate_expansion_tree(...)`
- 结果落盘 → `save_optimized_tree` / `get_topK_results` / `save_topK_results_to_csv` / `save_results`
