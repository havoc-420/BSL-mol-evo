# Molecular Optimization (MO) 文档目录

该目录包含与分子优化任务相关的所有文档和说明。

## 目录内容

- [molecular_optimization_task.md](file:///home/data2/rhj/project/mol_editor/mol_evo/docs/mo/molecular_optimization_task.md) - 分子优化任务的目标与实现思路

## 任务概述

分子优化任务旨在基于已训练的分子属性预测模型，实现一个能够预测分子经过若干步操作后属性变化并筛选合适分子的系统。

主要功能包括：
1. 加载基于step=1数据集训练的模型
2. 预测分子经过多步演化后的属性变化
3. 根据用户需求筛选满足条件的分子

## 相关脚本

相关实现脚本位于项目目录：
- `mol_evo/molecule_optimizer.py` - 分子优化主脚本

## 使用方法

详细使用方法请参考 [molecular_optimization_task.md](file:///home/data2/rhj/project/mol_editor/mol_evo/docs/mo/molecular_optimization_task.md) 文档。