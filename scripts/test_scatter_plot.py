#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在指定数据集上使用训练好的 MoleculeEvolutionVisnetLinearPredictor 模型进行预测，
并绘制预测值 vs 真实值的散点图。

支持同时评估 lumo_change 和 homo_change 两个模型，支持 train/val/test 分割。

Usage:
    # 默认只评估测试集
    conda activate mol-ofo && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/test_scatter_plot.py

    # 评估 train + val + test 三个分割
    conda activate mol-ofo && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/test_scatter_plot.py --splits train val test

    # 只测试 lumo_change，使用 100 个样本
    conda activate mol-ofo && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/test_scatter_plot.py \
        --num-samples 100 --properties lumo_change

    # 全量测试（默认使用全部样本）
    conda activate mol-ofo && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/test_scatter_plot.py --num-samples 0 --splits train val test
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

try:
    from mol_evo.core.data.processing import (
        load_operation_config,
        prepare_edge_features,
    )
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
    from mol_evo.utils.predict.model_utils import load_property_stats, is_model_normalized
    from mol_evo.core.models.v0.visnet_linear_linear import MoleculeEvolutionVisnetLinearPredictor
except ImportError as e:
    print(f"导入模块失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


def load_visnet_model(model_path, model_dir):
    """
    加载 VisNet 模型，自动从 checkpoint 推断 edge_feature_dim 以避免配置错误。
    """
    config_file = os.path.join(model_dir, "model_config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)

    params = config.get('model_params', {})
    node_feature_dim = params.get('node_feature_dim', 11)
    edge_feature_dim = params.get('edge_feature_dim', 16)
    hidden_dims = params.get('hidden_dims', [128, 256, 256])
    output_dim = params.get('output_dim', 1)

    # 从 checkpoint 中推断实际的 edge_feature_dim
    state_dict = torch.load(model_path, map_location='cpu')
    edge_encoder_key = 'edge_encoder.edge_encoder.0.weight'
    if edge_encoder_key in state_dict:
        actual_edge_dim = state_dict[edge_encoder_key].shape[1]
        if actual_edge_dim != edge_feature_dim:
            print(f"   ⚠️  model_config edge_feature_dim={edge_feature_dim}, "
                  f"但 checkpoint 实际为 {actual_edge_dim}，使用 checkpoint 值")
            edge_feature_dim = actual_edge_dim

    model = MoleculeEvolutionVisnetLinearPredictor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"   模型参数: node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, "
          f"hidden={hidden_dims}, output={output_dim}")
    print(f"   推理设备: {device}")
    return model


# ==================== 默认配置 ====================
BASE_DIR = os.path.join(project_root, 'mol_evo')

# 模型目录
MODEL_DIRS = {
    'lumo_change': os.path.join(BASE_DIR, 'output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200'),
    'homo_change': os.path.join(BASE_DIR, 'output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200'),
}

# 数据文件
DATA_FILE = os.path.join(BASE_DIR, 'dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json')
INDICES_FILE = os.path.join(BASE_DIR, 'dataset/data/dataset_indices/indices_20251122_213847_seed42.json')
CONFIG_FILE = os.path.join(BASE_DIR, 'dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml')


def load_test_data(data_file, indices_file, splits=None):
    """
    加载数据集

    Args:
        data_file: 数据文件路径
        indices_file: 索引文件路径
        splits: 要加载的数据分割列表，如 ['train', 'val', 'test']。
                默认为 ['test']。None 也等同于 ['test']。

    Returns:
        dict: {split_name: data_list}，如 {'test': [...], 'val': [...]}
    """
    if splits is None:
        splits = ['test']

    print(f"📂 加载数据文件: {data_file}")
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    print(f"   总数据量: {len(all_data)}")

    print(f"📂 加载索引文件: {indices_file}")
    with open(indices_file, 'r') as f:
        indices_info = json.load(f)

    result = {}
    for split in splits:
        key = f'{split}_indices'
        if key not in indices_info:
            print(f"   ⚠️  索引文件中无 {key}，跳过")
            continue
        indices = indices_info[key]
        data = [all_data[i] for i in indices if i < len(all_data)]
        result[split] = data
        print(f"   {split} 集索引数量: {len(indices)}, 实际数据量: {len(data)}")

    return result


def predict_single(model, model_dir, item, target_prop, cache, property_stats, is_normalized):
    """
    对单个样本进行预测（直接调用模型，绕过 predict_property_changes 的 x 验证问题）

    Args:
        model: 已加载的模型
        model_dir: 模型目录
        item: 数据字典
        target_prop: 目标属性名
        cache: MoleculeCache 实例
        property_stats: 属性统计信息
        is_normalized: 模型是否使用标准化

    Returns:
        pred_value: 预测值 (反标准化后)
        true_value: 真实值
    """
    smiles_from = item['smiles_from']
    smiles_to = item['smiles_to']

    # 从 operations 中提取信息
    operations = item.get('operations', [])
    if not operations:
        return None, None

    atom_symbol = operations[0].get('atom', '') or ''
    operation_type = operations[0].get('operation', 'unknown')

    # 构建分子图数据
    from_data = smiles_to_graph_data(smiles_from, cache)
    to_data = smiles_to_graph_data(smiles_to, cache)

    if from_data is None or to_data is None:
        return None, None

    # 检查 VisNet 需要的字段
    for data in [from_data, to_data]:
        if not hasattr(data, 'z') or data.z is None:
            return None, None
        if not hasattr(data, 'pos') or data.pos is None:
            return None, None

    # 构建 edge 特征
    data_dict = {
        'smiles_from': [smiles_from],
        'smiles_to': [smiles_to],
        'to_atom_symbol': [atom_symbol],
        'operation_type': [operation_type],
    }
    df = pd.DataFrame(data_dict)
    edge_feat = prepare_edge_features(
        df.iloc[0], property_stats,
        include_property_changes=False,
        include_position_encoding=False
    )
    edge_attr = torch.FloatTensor(np.array([edge_feat]))

    # 添加 batch 信息 & 移到 GPU
    device = next(model.parameters()).device
    from_data.batch = torch.zeros(from_data.z.size(0), dtype=torch.long)
    to_data.batch = torch.zeros(to_data.z.size(0), dtype=torch.long)
    from_data = from_data.to(device)
    to_data = to_data.to(device)
    edge_attr = edge_attr.to(device)

    # 推理
    with torch.no_grad():
        predictions = model(from_data, to_data, edge_attr)

    # 兼容不同返回格式，确保转到 CPU 再转 numpy
    if isinstance(predictions, (list, tuple)):
        pred_tensor = predictions[0]
    else:
        pred_tensor = predictions
    pred_raw = pred_tensor.detach().cpu().numpy().flatten()[0]

    # 反标准化
    if is_normalized and property_stats and target_prop in property_stats:
        mean, std = property_stats[target_prop]
        pred_value = float(pred_raw * std + mean)
    else:
        pred_value = float(pred_raw)

    true_value = item.get(target_prop, None)
    if true_value is not None:
        true_value = float(true_value)

    return pred_value, true_value


def predict_on_dataset(model_path, model_dir, data, target_prop, num_samples=0, tag='test'):
    """
    在指定数据集上进行预测

    Args:
        model_path: 模型权重路径
        model_dir: 模型目录路径
        data: 数据列表
        target_prop: 目标属性名 (如 'lumo_change')
        num_samples: 使用的样本数，0 表示全部
        tag: 数据集标签 (如 'train', 'val', 'test')

    Returns:
        predictions: 预测值数组
        true_values: 真实值数组
    """
    tag_display = {'train': 'Train', 'val': 'Val', 'test': 'Test'}.get(tag, tag.capitalize())
    print(f"\n🔮 开始预测 [{target_prop}] [{tag_display}]...")
    print(f"   模型路径: {model_path}")

    # 加载模型
    model = load_visnet_model(model_path, model_dir)
    print(f"   模型加载完成")

    # 加载属性统计
    property_stats = load_property_stats(model_dir)
    is_normalized = is_model_normalized(model_dir)
    print(f"   标准化: {is_normalized}, property_stats: {property_stats}")

    # 创建分子缓存
    cache = MoleculeCache("test_predict")

    # 如果 num_samples > 0，则只使用前 num_samples 个
    data_to_use = data[:num_samples] if num_samples > 0 else data
    print(f"   预测样本数: {len(data_to_use)}")

    predictions = []
    true_values = []
    errors = 0

    for item in tqdm(data_to_use, desc=f"预测 {target_prop} [{tag_display}]"):
        try:
            pred_value, true_value = predict_single(
                model, model_dir, item, target_prop, cache, property_stats, is_normalized
            )

            if pred_value is not None and true_value is not None:
                predictions.append(pred_value)
                true_values.append(true_value)
            else:
                errors += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                import traceback
                print(f"  ❌ 错误 #{errors}: {e}")
                traceback.print_exc()
            continue

    print(f"   成功预测: {len(predictions)}, 错误/跳过: {errors}")
    return np.array(predictions), np.array(true_values)


def compute_metrics(predictions, true_values):
    """计算评估指标"""
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))

    # R²
    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # PCC
    if np.std(predictions) > 0 and np.std(true_values) > 0:
        pcc = np.corrcoef(predictions, true_values)[0, 1]
    else:
        pcc = 0

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'PCC': pcc,
        'N': len(predictions),
    }


def plot_scatter(predictions, true_values, metrics, target_prop, output_path, tag='test'):
    """
    绘制散点图 (Predicted vs Actual)

    Args:
        tag: 数据集标签，如 'train', 'val', 'test'
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)

    prop_display = target_prop.replace('_change', '').upper()
    tag_display = {'train': 'Train', 'val': 'Val', 'test': 'Test'}.get(tag, tag.capitalize())

    # 散点
    ax.scatter(true_values, predictions, alpha=0.3, s=12, c='#2196F3', edgecolors='none', label='Predictions')

    # 对角线 (y=x)
    all_vals = np.concatenate([true_values, predictions])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    margin = (vmax - vmin) * 0.05
    line_range = [vmin - margin, vmax + margin]
    ax.plot(line_range, line_range, 'r--', linewidth=1.5, alpha=0.8, label='y = x')

    ax.set_xlim(line_range)
    ax.set_ylim(line_range)

    # 标注指标
    metrics_text = (
        f"N = {metrics['N']}\n"
        f"R² = {metrics['R²']:.4f}\n"
        f"PCC = {metrics['PCC']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'True {prop_display} Change', fontsize=13)
    ax.set_ylabel(f'Predicted {prop_display} Change', fontsize=13)
    ax.set_title(f'{tag_display} Set: Predicted vs True ({prop_display} Change)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 散点图已保存: {output_path}")


def plot_combined_scatter(results, output_path):
    """
    绘制 splits(行) x properties(列) 的组合散点图

    results 的 key 格式为 "lumo_change[train]"
    """
    SPLIT_ORDER = ['train', 'val', 'test']
    SPLIT_COLORS = {'train': '#4CAF50', 'val': '#FF9800', 'test': '#2196F3'}
    SPLIT_DISPLAY = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    # 解析所有 result_key，提取唯一的属性和分割
    parsed = {}
    for result_key, val in results.items():
        if '[' in result_key and ']' in result_key:
            prop = result_key[:result_key.index('[')]
            split = result_key[result_key.index('[')+1:result_key.index(']')]
        else:
            prop = result_key
            split = 'test'
        parsed[(prop, split)] = val

    properties = list(dict.fromkeys(
        k[0] for k in parsed.keys()
    ))
    splits = [s for s in SPLIT_ORDER if any(k[1] == s for k in parsed.keys())]

    n_rows = len(splits)
    n_cols = len(properties)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows), dpi=150)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # 按属性统一坐标范围
    for col, prop in enumerate(properties):
        all_vals = []
        for split in splits:
            key = (prop, split)
            if key in parsed:
                preds, trues, _ = parsed[key]
                all_vals.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))
                all_vals.extend(trues.tolist() if hasattr(trues, 'tolist') else list(trues))

        if all_vals:
            vmin, vmax = min(all_vals), max(all_vals)
            margin = (vmax - vmin) * 0.05
            xlim = (vmin - margin, vmax + margin)
        else:
            xlim = (-1, 1)

        for row, split in enumerate(splits):
            ax = axes[row, col]
            key = (prop, split)

            if key not in parsed:
                ax.set_visible(False)
                continue

            preds, trues, metrics = parsed[key]
            color = SPLIT_COLORS.get(split, '#2196F3')

            ax.scatter(trues, preds, alpha=0.3, s=8, c=color, edgecolors='none', label='Predictions')
            ax.plot(xlim, xlim, 'r--', linewidth=1.5, alpha=0.7, label='y = x')
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
            ax.set_aspect('equal', adjustable='box')

            metrics_text = (
                f"R² = {metrics['R²']:.4f}\n"
                f"PCC = {metrics['PCC']:.4f}\n"
                f"RMSE = {metrics['RMSE']:.4f}\n"
                f"MAE = {metrics['MAE']:.4f}"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

            prop_display = prop.replace('_change', '').upper()
            split_display = SPLIT_DISPLAY.get(split, split.capitalize())

            if row == n_rows - 1:
                ax.set_xlabel(f'True {prop_display} Change', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Predicted {prop_display} Change', fontsize=12)

            ax.set_title(f'[{split_display}] {prop_display} Change', fontsize=13, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('MoleculeEvolutionVisnetLinearPredictor - Evaluation',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\n📊 组合散点图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='数据集预测与散点图绘制')
    parser.add_argument('--num-samples', type=int, default=0,
                        help='每个模型每个分割使用的样本数，0 表示全部 (默认: 0)')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['test'],
                        help='要评估的数据分割，可选 train val test (默认: test)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: mol_evo/output/scatter_plots/<timestamp>)')
    parser.add_argument('--properties', type=str, nargs='+',
                        default=['lumo_change', 'homo_change'],
                        help='要评估的属性列表 (默认: lumo_change homo_change)')
    parser.add_argument('--data-file', type=str, default=DATA_FILE,
                        help='数据文件路径')
    parser.add_argument('--indices-file', type=str, default=INDICES_FILE,
                        help='索引文件路径')
    parser.add_argument('--config-file', type=str, default=CONFIG_FILE,
                        help='操作配置文件路径')

    args = parser.parse_args()

    # 创建输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_DIR, 'output', 'scatter_plots', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    # 加载操作配置
    print(f"📂 加载操作配置: {args.config_file}")
    load_operation_config(config_path=args.config_file)

    # 加载数据（按分割）
    split_data = load_test_data(args.data_file, args.indices_file, splits=args.splits)

    # 对每个目标属性进行预测和绘图
    all_results = {}

    for target_prop in args.properties:
        if target_prop not in MODEL_DIRS:
            print(f"⚠️  未找到属性 {target_prop} 对应的模型目录，跳过")
            continue

        model_dir = MODEL_DIRS[target_prop]
        model_path = os.path.join(model_dir, 'last.pth')

        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}，跳过")
            continue

        for split, data in split_data.items():
            # 预测
            predictions, true_values = predict_on_dataset(
                model_path, model_dir, data, target_prop,
                num_samples=args.num_samples, tag=split
            )

            if len(predictions) == 0:
                print(f"⚠️  属性 {target_prop} [{split}] 没有有效的预测结果，跳过")
                continue

            # 计算指标
            metrics = compute_metrics(predictions, true_values)
            result_key = f"{target_prop}[{split}]"
            all_results[result_key] = (predictions, true_values, metrics)

            # 打印指标
            tag_display = {'train': 'Train', 'val': 'Val', 'test': 'Test'}.get(split, split.capitalize())
            print(f"\n📈 [{target_prop}] [{tag_display}] 评估指标:")
            print(f"   {'指标':<10} {'值'}")
            print(f"   {'-'*30}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"   {k:<10} {v:.6f}")
                else:
                    print(f"   {k:<10} {v}")

            # 单属性散点图 (文件名包含分割标签)
            scatter_path = os.path.join(output_dir, f'scatter_{target_prop}_{split}.png')
            plot_scatter(predictions, true_values, metrics, target_prop, scatter_path, tag=split)

            # 保存预测结果到 JSON
            result_data = {
                'target_property': target_prop,
                'split': split,
                'model_dir': model_dir,
                'num_samples': int(metrics['N']),
                'metrics': {k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in metrics.items()},
                'predictions': predictions.tolist(),
                'true_values': true_values.tolist(),
            }
            result_path = os.path.join(output_dir, f'results_{target_prop}_{split}.json')
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"   💾 预测结果已保存: {result_path}")

    # 组合散点图
    if len(all_results) > 0:
        combined_path = os.path.join(output_dir, 'scatter_combined.png')
        plot_combined_scatter(all_results, combined_path)

    print(f"\n✅ 全部完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
