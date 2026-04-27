# `mu` case50：原版 ViSNet 基座 + OFO-guided MCTS 运行记录

## 1. 目标

使用目标服务器上的原版 ViSNet `mu` 基座模型：

- checkpoint：`/root/autodl-tmp/projects/visnet-qm9/logs/output_ngpus_1_bs_64_lr_0.0001_seed_1_reload_0_lmax_2_vnorm_max_min_vertex_None_L9_D512_H8_cutoff_5.0_E1.0_F1.0_loss_MSE/last.ckpt`

在 `mol_evo` 项目中，对 `QM9` 测试集前 `50` 个分子执行 `mu` 目标的 `MCTS` 优化试跑。

## 2. 启动命令

### `increase` 版本

```bash
ssh seetacloud-2 "tmux new-session -d -s mu-opt-inc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate mol-edit && cd /root/autodl-tmp/projects/mol-ofo && CUDA_VISIBLE_DEVICES=0 python -m mol_evo.scripts.optimization.batch_optimizer_visnet_native --input-csv /root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv --model-path /root/autodl-tmp/projects/visnet-qm9/logs/output_ngpus_1_bs_64_lr_0.0001_seed_1_reload_0_lmax_2_vnorm_max_min_vertex_None_L9_D512_H8_cutoff_5.0_E1.0_F1.0_loss_MSE/last.ckpt --model-dir /root/autodl-tmp/projects/visnet-qm9/logs/output_ngpus_1_bs_64_lr_0.0001_seed_1_reload_0_lmax_2_vnorm_max_min_vertex_None_L9_D512_H8_cutoff_5.0_E1.0_F1.0_loss_MSE --config-file /root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml --target-property mu --direction increase --start-index 0 --end-index 50 --max-depth 10 --max-branching 20 --search-mode mcts --num-simulations 1600 --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5 --topK 20 --visnet-project-path /root/autodl-tmp/projects/visnet-qm9 --output-dir /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase 2>&1 | tee /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase/tmux.log'"
```

### `decrease` 版本

```bash
ssh seetacloud-2 "tmux new-session -d -s mu-opt-dec 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate mol-edit && cd /root/autodl-tmp/projects/mol-ofo && CUDA_VISIBLE_DEVICES=0 python -m mol_evo.scripts.optimization.batch_optimizer_visnet_native --input-csv /root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv --model-path /root/autodl-tmp/projects/visnet-qm9/logs/output_ngpus_1_bs_64_lr_0.0001_seed_1_reload_0_lmax_2_vnorm_max_min_vertex_None_L9_D512_H8_cutoff_5.0_E1.0_F1.0_loss_MSE/last.ckpt --model-dir /root/autodl-tmp/projects/visnet-qm9/logs/output_ngpus_1_bs_64_lr_0.0001_seed_1_reload_0_lmax_2_vnorm_max_min_vertex_None_L9_D512_H8_cutoff_5.0_E1.0_F1.0_loss_MSE --config-file /root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml --target-property mu --direction decrease --start-index 0 --end-index 50 --max-depth 10 --max-branching 20 --search-mode mcts --num-simulations 1600 --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5 --topK 20 --visnet-project-path /root/autodl-tmp/projects/visnet-qm9 --output-dir /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease 2>&1 | tee /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease/tmux.log'"
```

## 3. 参数口径

- **目标属性**：`mu`
- **优化方向**：提供 `increase` / `decrease` 两个版本
- **样本范围**：`start-index=0`，`end-index=50`
- **搜索模式**：`mcts`
- **搜索预算**：`num-simulations=1600`
- **探索系数**：`exploration-weight=2`
- **搜索深度**：`max-depth=10`
- **分支宽度**：`max-branching=20`
- **剪枝设置**：`pruning-patience=3`
- **logP 约束**：`[-0.5, 6]`
- **TopK**：`20`

除优化方向和输出位置外，两条命令的其余配置完全一致，整体对齐当前论文 `MCTS` sweep 的主线设置。

## 4. 输入与输出路径

### 输入

- **输入分子集**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- **操作配置**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml`
- **ViSNet 项目路径**：`/root/autodl-tmp/projects/visnet-qm9`

### 输出

- **increase 输出目录**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase`
- **increase tmux 日志**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase/tmux.log`
- **decrease 输出目录**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease`
- **decrease tmux 日志**：`/root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease/tmux.log`

## 5. 运行后检查命令

### 查看 tmux 会话

```bash
ssh seetacloud-2 "tmux ls"
```

### 查看 `increase` 实时输出

```bash
ssh seetacloud-2 "tmux capture-pane -t mu-opt-inc -p | tail -40"
```

### 查看 `decrease` 实时输出

```bash
ssh seetacloud-2 "tmux capture-pane -t mu-opt-dec -p | tail -40"
```

### 查看 `increase` 日志末尾

```bash
ssh seetacloud-2 "tail -40 /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase/tmux.log"
```

### 查看 `decrease` 日志末尾

```bash
ssh seetacloud-2 "tail -40 /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease/tmux.log"
```

### 查看 `increase` 结果文件数量

```bash
ssh seetacloud-2 "find /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_increase -name '*.json' | wc -l"
```

### 查看 `decrease` 结果文件数量

```bash
ssh seetacloud-2 "find /root/autodl-tmp/projects/mol-ofo/mol_evo/output/evo-mo/mu_visnet_native_case50_mcts_decrease -name '*.json' | wc -l"
```

## 6. 当前说明

- 本次运行依赖脚本：`mol_evo/scripts/optimization/batch_optimizer_visnet_native.py`
- 该脚本用于把**原版 ViSNet 单分子真值预测模型**适配到 `mol_evo` 的分子编辑优化链路中，内部通过：

\[
\Delta \mu = \mu(\text{smiles\_to}) - \mu(\text{smiles\_from})
\]

来构造编辑步的属性变化值。

- 当前测试集 `csv` 本身不含 `mu` 列，因此脚本需要在运行时先对起始分子预测初始 `mu` 值，再继续做树搜索。

## 7. 风险提示

- 原版 ViSNet 的批量预测在 `GPU` 显存占用上明显高于现有 pair-based 预测器；如果服务器上同时有其他大任务占用显存，可能出现 `CUDA out of memory`。
- 若出现显存问题，优先检查：
  - `nvidia-smi`
  - 是否有其他 `python` 训练/推理进程占用同一张卡
  - `batch_optimizer_visnet_native.py` 内部的真值预测批量大小是否需要进一步收缩

## 8. 后续建议

如果这组 `case50` 能顺利跑通，下一步建议：

- 先对输出目录做一次结果汇总，确认 `mu increase` 是否产生稳定的正向改善；
- 再决定是否扩展到更大样本规模，或纳入 `docs/paper/sweep/v2` 的正式对比分析。 
