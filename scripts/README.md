# mol_evo/scripts

脚本按功能分为 5 个子目录：

## 目录结构

```
scripts/
├── optimization/          批量优化运行（核心生产脚本）
│   ├── batch_optimizer.py         QM9 属性批量优化入口（BFS/MCTS/A*RL）
│   └── batch_optimizer_ic50.py    IC50 药物响应批量优化入口
│
├── evaluation/            实验结果汇总
│   ├── eval_astar_rl_holdout.py           A* RL holdout 评估 + baseline 对比
│   ├── stat_generation_time.py            统计分子生成时间
│   ├── summarize_mcts_ablation_runs.py    汇总 ablation 模型预测口径指标
│   └── summarize_mcts_true_eval_runs.py   汇总 ablation 真值评估口径指标
│
├── visualization/         可视化 / 绘图
│   ├── test_scatter_plot.py           模型预测 vs 真值散点图
│   ├── plot_scatter_grid.py           2×3 拼接散点图
│   ├── plot_test_cases_2d.py          Source→Target 2D 分子对比图
│   └── reassemble_selected_cases.py   从 temp 收集挑选 case 重新渲染组装 PDF
│
├── data_prep/             数据准备
│   ├── convert_qm9_evo_to_paths.py   qm9-evo pairs → v0.3 路径格式
│   └── prepare_plan_a_holdout.py      切分 holdout / train-pool CSV
│
└── runners/               Shell 运行编排
    ├── run_mcts_ablations.sh          MCTS 消融实验编排
    ├── run_eval_mcts_ablations.sh     对已有 ablation 目录补充真值评估
    └── run_mcts_hparam_sweeps.sh      MCTS 超参扫描编排
```

## 模块调用

所有 Python 脚本均可通过 `-m` 方式调用，例如：

```bash
python -m mol_evo.scripts.optimization.batch_optimizer --help
python -m mol_evo.scripts.evaluation.summarize_mcts_ablation_runs --run-root <path>
```

Shell 脚本从 repo 根目录调用：

```bash
bash mol_evo/scripts/runners/run_mcts_ablations.sh
```
