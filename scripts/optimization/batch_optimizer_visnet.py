#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用evolution_optimizer.py的 MO 优化脚本
支持 prediction_type='change'（模型预测差值，默认）和
         prediction_type='true_value'（模型预测真值，需预处理 sub 差值）

当基座模型预测的是真值（absolute property value）而非差值时，
通过 TrueValueAdapter 在 predict_property_change / predict_batch 中
自动计算 pred(smiles_to) - pred(smiles_from) 作为属性变化值。

# INFO
--prediction-type change       # 默认，模型直接输出差值（原始行为）
--prediction-type true_value   # 模型输出真值，自动 sub 计算差值
"""

import os
import sys
import json
import argparse
import time
import pandas as pd
import uuid
import traceback
from rdkit import RDLogger
import signal
from datetime import datetime
from typing import List, Optional

# 全局中断标志
interrupted = False

# 信号处理函数
def signal_handler(sig, frame):
    """处理中断信号"""
    global interrupted
    if interrupted:
        print("再次收到中断信号，立即退出！")
        sys.exit(0)
    print("正在中断处理过程，请稍候...")
    interrupted = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)


def _torch_load_compat(path, *, map_location=None, weights_only=None):
    """兼容不同 PyTorch 版本的 `torch.load` 参数。"""
    import torch
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


try:
    from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
#  TrueValueAdapter: 当基座模型预测真值时，包装为差值预测器
# ---------------------------------------------------------------------------
class TrueValueAdapter(EvolutionTreeOptimizer):
    """
    适配器：当基座模型预测的是属性真值（而非差值）时，
    重写 predict_property_change / predict_batch，
    通过 pred(smiles_to) - pred(smiles_from) 计算差值。

    使用方式：
        optimizer = TrueValueAdapter(model_path, model_dir, ...)
        # 其余接口与 EvolutionTreeOptimizer 完全一致
    """

    def __init__(self, model_path, model_dir, config_file=None,
                 initial_smiles_csv=None, target_property=None,
                 initial_property_value=None, optimization_mode='pct'):
        super().__init__(
            model_path, model_dir, config_file,
            initial_smiles_csv, target_property,
            initial_property_value, optimization_mode,
        )
        # 缓存初始分子真值，避免重复推理
        self._initial_value_cache = {}

        print(f"[TrueValueAdapter] 模式已启用：基座模型预测真值，自动计算 sub 差值")

    # ------------------------------------------------------------------
    #  单对预测
    # ------------------------------------------------------------------
    def predict_property_change(self, smiles_from, smiles_to, operation_details):
        """
        用真值模型分别预测 from / to 分子的属性值，返回差值 pred_to - pred_from。
        """
        try:
            pred_from = self._predict_true_value(smiles_from)
            pred_to = self._predict_true_value(smiles_to)

            if pred_from is None or pred_to is None:
                return None

            change = pred_to - pred_from
            return change

        except Exception as e:
            print(f"[TrueValueAdapter] predict_property_change 出错: "
                  f"from={smiles_from}, to={smiles_to}, error={e}")
            return None

    # ------------------------------------------------------------------
    #  批量预测
    # ------------------------------------------------------------------
    def predict_batch(self, from_smiles_list, to_smiles_list,
                      operation_details_list, batch_size=64):
        """
        批量版：分别预测 from / to 真值，返回差值列表。
        复用父类的批量推理能力，但改为单分子输入。
        """
        if not from_smiles_list or not to_smiles_list or not operation_details_list:
            return []

        assert len(from_smiles_list) == len(to_smiles_list) == len(operation_details_list), \
            "输入列表长度必须相同"

        # 收集所有需要预测的 SMILES（去重）
        all_smiles = set(from_smiles_list) | set(to_smiles_list)

        # 批量推理真值
        value_map = self._predict_true_value_batch(all_smiles, batch_size=batch_size)

        # 计算差值
        results = []
        for s_from, s_to in zip(from_smiles_list, to_smiles_list):
            v_from = value_map.get(s_from)
            v_to = value_map.get(s_to)
            if v_from is not None and v_to is not None:
                results.append(v_to - v_from)
            else:
                results.append(None)

        return results

    # ------------------------------------------------------------------
    #  内部：单分子真值预测
    # ------------------------------------------------------------------
    def _predict_true_value(self, smiles: str) -> Optional[float]:
        """
        用基座模型预测单个分子的属性真值。

        对于 pair-based 模型 (from_data, to_data, edge_attr)，
        将 from 和 to 设为同一分子来获取真值。
        """
        if smiles in self._initial_value_cache:
            return self._initial_value_cache[smiles]

        try:
            import numpy as np
            import torch
            from mol_evo.core.data.data_v0 import smiles_to_graph_data
            from mol_evo.core.data.processing import prepare_edge_features

            graph_data = smiles_to_graph_data(smiles, self.molecule_cache)
            if graph_data is None:
                return None

            graph_data = self._add_batch_info(graph_data)

            # pair-based 模型：from = to = 同一分子
            edge_feat = prepare_edge_features(
                {'smiles_from': smiles, 'smiles_to': smiles, 'operations': []},
                self.property_stats,
                include_property_changes=False,
                include_position_encoding=False,
            )
            edge_attr = torch.FloatTensor(np.array([edge_feat]))

            # 数据移到设备
            graph_data = graph_data.to(self.device)
            edge_attr = edge_attr.to(self.device)

            with torch.no_grad():
                predictions = self.model(graph_data, graph_data, edge_attr)

            predicted_value = predictions[0].numpy()[0]

            # 反标准化
            if self.property_stats and self.target_property in self.property_stats:
                mean, std = self.property_stats[self.target_property]
                predicted_value = predicted_value * std + mean

            self._initial_value_cache[smiles] = predicted_value
            return predicted_value

        except Exception as e:
            print(f"[TrueValueAdapter] _predict_true_value 出错: smiles={smiles}, error={e}")
            return None

    # ------------------------------------------------------------------
    #  内部：批量单分子真值预测
    # ------------------------------------------------------------------
    def _predict_true_value_batch(self, smiles_set, batch_size=64):
        """
        批量预测一组 SMILES 的真值，返回 {smiles: value}。
        对缓存中已有的直接返回，未缓存的批量推理。
        """
        import torch
        import numpy as np
        from torch_geometric.data import Batch
        from mol_evo.core.data.data_v0 import smiles_to_graph_data
        from mol_evo.core.data.processing import prepare_edge_features

        value_map = {}
        to_predict = []

        # 先查缓存
        for s in smiles_set:
            if s in self._initial_value_cache:
                value_map[s] = self._initial_value_cache[s]
            else:
                to_predict.append(s)

        if not to_predict:
            return value_map

        # 准备批量数据
        graph_list = []
        edge_attr_list = []
        valid_smiles = []

        for smiles in to_predict:
            try:
                graph_data = smiles_to_graph_data(smiles, self.molecule_cache)
                if graph_data is None:
                    continue
                graph_data = self._add_batch_info(graph_data)

                edge_feat = prepare_edge_features(
                    {'smiles_from': smiles, 'smiles_to': smiles, 'operations': []},
                    self.property_stats,
                    include_property_changes=False,
                    include_position_encoding=False,
                )
                edge_attr = torch.FloatTensor(np.array(edge_feat))

                graph_list.append(graph_data)
                edge_attr_list.append(edge_attr)
                valid_smiles.append(smiles)
            except Exception as e:
                print(f"[TrueValueAdapter] 准备数据出错: smiles={smiles}, error={e}")
                continue

        if not graph_list:
            return value_map

        # 分批推理
        all_predictions = []
        num_samples = len(graph_list)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_graphs = graph_list[start_idx:end_idx]
            batch_edges = edge_attr_list[start_idx:end_idx]

            from_batch = Batch.from_data_list(batch_graphs).to(self.device)
            to_batch = from_batch  # 同一分子
            edge_attr_batch = torch.stack(batch_edges).to(self.device)

            try:
                with torch.no_grad():
                    predictions = self.model(from_batch, to_batch, edge_attr_batch)

                # 反标准化
                if self.property_stats and self.target_property in self.property_stats:
                    mean, std = self.property_stats[self.target_property]
                    predictions = predictions * std + mean

                batch_preds = self._prediction_tensor_to_list(predictions)
                all_predictions.extend(batch_preds)
            except Exception as e:
                print(f"[TrueValueAdapter] 批量推理出错: start={start_idx}, end={end_idx}, error={e}")
                all_predictions.extend([None] * (end_idx - start_idx))

        # 写入缓存
        for s, v in zip(valid_smiles, all_predictions):
            if v is not None:
                value_map[s] = v
                self._initial_value_cache[s] = v

        return value_map


# ---------------------------------------------------------------------------
#  命令行参数
# ---------------------------------------------------------------------------
def parse_args():
    """解析命令行参数"""
    default_model_dir = os.path.join(
        project_root,
        'mol_evo',
        'output',
        'v0',
        'MoleculeEvolutionVisnetLinearPredictor',
        'train-20251123_192921-lumo_change-120000-200',
    )
    default_input_csv = os.path.join(
        project_root,
        'mol_evo',
        'dataset',
        'eval-data',
        '20251205_131636',
        'qm9_test_molecules.csv',
    )
    default_config_file = os.path.join(
        project_root,
        'mol_evo',
        'dataset',
        'data',
        'qm9-evo-pairs-step-1-with-properties-pct-config.yaml',
    )

    parser = argparse.ArgumentParser(
        description='MO 批量优化分子属性（支持真值/差值预测模型）'
    )
    parser.add_argument('--input-csv', type=str,
                        default=default_input_csv,
                        help='输入的CSV文件路径')
    parser.add_argument('--output-json', type=str,
                        help='输出的JSON文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径；若不提供则自动创建时间戳目录')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(default_model_dir, 'last.pth'),
                        help='模型文件路径')
    parser.add_argument('--model-dir', type=str,
                        default=default_model_dir,
                        help='模型目录路径')
    parser.add_argument('--config-file', type=str,
                        default=default_config_file,
                        help='配置文件路径')
    parser.add_argument('--target-property', type=str, default='lumo',
                        help='目标属性名称')
    # ---- 核心新增参数 ----
    parser.add_argument('--prediction-type', type=str,
                        choices=['change', 'true_value'], default='change',
                        help='预测类型: change=模型直接预测差值(默认); '
                             'true_value=模型预测真值，自动 sub 计算差值')
    # ---- 优化参数 ----
    parser.add_argument('--optimization-mode', type=str, choices=['sub', 'pct'], default='sub',
                        help='优化模式')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='最大演化深度')
    parser.add_argument('--max-branching', type=int, default=8,
                        help='最大分支数')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], default='decrease',
                        help='优化方向')
    parser.add_argument('--pruning-patience', type=int, default=2,
                        help='剪枝耐心值')
    parser.add_argument('--logp-min', type=float, default=0.0,
                        help='logP的最小值')
    parser.add_argument('--logp-max', type=float, default=5.0,
                        help='logP的最大值')
    parser.add_argument('--logp-patience', type=int, default=3,
                        help='logP剪枝耐心值，连续多少代logP超出范围就剪枝')
    parser.add_argument('--topK', type=int, default=20,
                        help='保留效果最好的K个结果')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批处理大小')
    parser.add_argument('--start-index', type=int, default=0,
                        help='起始索引')
    parser.add_argument('--end-index', type=int, default=-1,
                        help='结束索引，-1表示处理到文件末尾')
    # MCTS 参数
    parser.add_argument('--search-mode', type=str, choices=['bfs', 'mcts', 'astar_demo'],
                        default='bfs', help='搜索模式: bfs(广度优先)、mcts(蒙特卡洛树搜索) 或 astar_demo(A* RL Demo)')
    parser.add_argument('--num-simulations', type=int, default=200,
                        help='MCTS 模拟轮数 (仅 mcts 模式)')
    parser.add_argument('--exploration-weight', type=float, default=1.4,
                        help='MCTS PUCT 探索系数 (仅 mcts 模式)')
    parser.add_argument('--mcts-prior-mode', type=str, choices=['softmax', 'uniform'], default='softmax',
                        help='MCTS prior 构造方式：softmax=默认 OFO prior，uniform=均匀先验')
    parser.add_argument('--mcts-value-mode', type=str, choices=['accumulated', 'zero', 'step'], default='accumulated',
                        help='MCTS 叶节点价值：accumulated=累计增益，zero=恒为0，step=仅当前步增益')
    parser.add_argument('--mcts-expansion-mode', type=str, choices=['topk', 'random_topk', 'full'], default='topk',
                        help='MCTS 扩展策略：topk=按 OFO 排序截断，random_topk=随机选 TopB，full=全展开')
    parser.add_argument('--mcts-random-seed', type=int, default=None,
                        help='MCTS 随机种子；主要用于 random_topk 的可复现采样')
    # astar_demo 参数
    parser.add_argument('--policy-path', type=str, default=None,
                        help='PolicyNet 权重路径 (仅 astar_demo 模式)')
    parser.add_argument('--value-path', type=str, default=None,
                        help='ValueNet 权重路径 (仅 astar_demo 模式)')
    parser.add_argument('--rl-train', action='store_true',
                        help='astar_demo 在线 RL 训练模式（搜索过程中更新 policy/value）')
    parser.add_argument('--rl-eval', action='store_true',
                        help='astar_demo 纯评估模式（加载权重，不更新）')
    parser.add_argument('--top-n-prefilter', type=int, default=20,
                        help='PolicyNet 预筛候选数 (仅 astar_demo 模式)')
    parser.add_argument('--open-set-budget', type=int, default=200,
                        help='A* open set 展开预算 (仅 astar_demo 模式)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  辅助函数
# ---------------------------------------------------------------------------
def create_output_dir(output_dir=None):
    """创建输出目录"""
    if output_dir:
        base_dir = os.path.abspath(output_dir)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.getcwd(), "mol_evo", "output", "evo-mo",
                            f"batch_optimization_mo_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def read_csv_data(csv_path, start_idx=0, end_idx=-1, target_property='lumo'):
    """读取CSV数据"""
    df = pd.read_csv(csv_path)
    if end_idx > 0:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]

    data_list = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        property_value = row[target_property]

        data_list.append({
            'smiles': smiles,
            'property_value': property_value,
            'original_row': row.to_dict()
        })
    return data_list


# ---------------------------------------------------------------------------
#  优化器工厂
# ---------------------------------------------------------------------------
def create_optimizer(args):
    """
    根据 --prediction-type 创建优化器实例。
    - 'change'       → EvolutionTreeOptimizer（原始行为）
    - 'true_value'   → TrueValueAdapter（自动 sub 计算差值）
    """
    common_kwargs = dict(
        model_path=args.model_path,
        model_dir=args.model_dir,
        config_file=args.config_file,
        initial_smiles_csv=None,
        target_property=args.target_property,
        initial_property_value=None,
        optimization_mode=args.optimization_mode,
    )

    if args.prediction_type == 'true_value':
        optimizer = TrueValueAdapter(**common_kwargs)
    else:
        optimizer = EvolutionTreeOptimizer(**common_kwargs)

    return optimizer


# ---------------------------------------------------------------------------
#  运行优化
# ---------------------------------------------------------------------------
def run_evolution_optimizer(optimizer, smiles, property_value, args, output_dir):
    """运行分子优化"""
    run_id = str(uuid.uuid4())[:8]

    print(f"\n=== 开始处理分子: {smiles} ===")
    print(f"初始属性值: {property_value}")

    try:
        start_time = time.time()

        # 设置优化器的初始属性值
        optimizer.initial_property_value = property_value

        # 优化进化树
        optimized_tree = optimizer.optimize_evolution_tree(
            smiles,
            args.max_depth,
            args.max_branching,
            args.direction,
            pruning_patience=args.pruning_patience,
            logp_range=(args.logp_min, args.logp_max),
            logp_patience=args.logp_patience,
            search_mode=args.search_mode,
            num_simulations=args.num_simulations,
            exploration_weight=args.exploration_weight,
            mcts_prior_mode=args.mcts_prior_mode,
            mcts_value_mode=args.mcts_value_mode,
            mcts_expansion_mode=args.mcts_expansion_mode,
            mcts_random_seed=args.mcts_random_seed,
            policy_net=getattr(args, '_policy_net', None),
            value_net=getattr(args, '_value_net', None),
            rl_trainer=getattr(args, '_rl_trainer', None),
            top_n_prefilter=getattr(args, 'top_n_prefilter', 20),
            open_set_budget=getattr(args, 'open_set_budget', 200),
        )

        # 保存优化结果
        output_json = os.path.join(output_dir,
                                   f"{smiles[:20].replace('/', '_')}_{run_id}.json")
        output_file = optimizer.save_optimized_tree(optimized_tree, output_json, output_dir)

        # 处理topK结果
        topK_results = optimizer.get_topK_results(optimized_tree, args.topK)
        topk_csv = os.path.splitext(output_json)[0] + "_topK.csv"
        optimizer.save_topK_results_to_csv(topK_results, topk_csv, output_dir)

        # 读取保存的优化结果
        with open(output_json, 'r') as f:
            optimized_result = json.load(f)

        end_time = time.time()

        print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"找到 {len(topK_results.get('topK_results', []))} 个topK结果")

        return {
            'status': 'success',
            'smiles': smiles,
            'initial_property': property_value,
            'optimized_result': optimized_result,
            'topk_results': topK_results,
            'runtime': end_time - start_time
        }

    except Exception as e:
        error_msg = str(e)
        print(f"处理失败: {smiles}, 错误: {error_msg}")
        traceback.print_exc()

        return {
            'status': 'error',
            'smiles': smiles,
            'initial_property': property_value,
            'error': error_msg
        }


# ---------------------------------------------------------------------------
#  批量处理
# ---------------------------------------------------------------------------
def batch_process(data_list, args, output_dir):
    """批量处理数据"""
    results_dict = {}
    total_count = len(data_list)

    # 创建总日志文件
    total_log_file = os.path.join(output_dir, "batch_optimization_total.log")

    print(f"=== 开始批量处理，共 {total_count} 个分子 ===")
    print(f"预测类型: {args.prediction_type}")
    print(f"总日志文件: {total_log_file}")

    # 记录开始时间
    batch_start_time = time.time()

    # 创建优化器（根据 prediction_type 选择）
    optimizer = create_optimizer(args)

    # --- astar_demo 模型加载 ---
    if args.search_mode == 'astar_demo':
        try:
            import torch
            from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RLTrainer
            from mol_evo.core.models.astar_rl.reward import RewardConfig
            from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _policy_net = PolicyNet(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(_device)
            _value_net = ValueNet(state_dim=STATE_DIM).to(_device)

            if args.policy_path and os.path.isfile(args.policy_path):
                _policy_net.load_state_dict(torch.load(args.policy_path, map_location=_device))
                print(f"[astar_demo] PolicyNet 权重已加载: {args.policy_path}")
                _policy_net.eval()
            else:
                print("[astar_demo] PolicyNet 权重未指定，使用随机初始化")

            if args.value_path and os.path.isfile(args.value_path):
                _value_net.load_state_dict(
                    _torch_load_compat(args.value_path, map_location=_device, weights_only=True)
                )
                print(f"[astar_demo] ValueNet 权重已加载: {args.value_path}")
                _value_net.eval()
            else:
                print("[astar_demo] ValueNet 权重未指定，使用随机初始化")

            _rl_trainer = None
            if args.rl_train:
                _rl_trainer = RLTrainer(
                    policy_net=_policy_net,
                    value_net=_value_net,
                    reward_config=RewardConfig(direction=args.direction),
                    device=str(_device),
                    checkpoint_dir=os.path.join(output_dir, "rl_checkpoints"),
                    checkpoint_every=50,
                )
                print("[astar_demo] RLTrainer 在线训练模式已初始化")

            args._policy_net = _policy_net
            args._value_net = _value_net
            args._rl_trainer = _rl_trainer
        except Exception as _e:
            print(f"[astar_demo] 警告：RL 模型加载失败，将在无 policy/value 的降级模式下运行: {_e}")
            args._policy_net = None
            args._value_net = None
            args._rl_trainer = None

    for i, data in enumerate(data_list):
        smiles = data['smiles']
        property_value = data['property_value']

        progress = (i + 1) / total_count * 100
        print(f"\n=== 处理进度: {i+1}/{total_count} ({progress:.1f}%) ===")

        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"\n=== 开始处理分子 {i+1}/{total_count}: {smiles} ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初始属性值: {property_value}\n")

        result = run_evolution_optimizer(
            optimizer, smiles, property_value, args, output_dir
        )

        results_dict[smiles] = {
            'original_data': data['original_row'],
            'optimization_result': result
        }

        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"状态: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"耗时: {result['runtime']:.2f} 秒\n")
                f.write(f"优化结果数量: "
                        f"{len(result['optimized_result'].get('results', [])) if 'optimized_result' in result else 0}\n")
                f.write(f"topK结果数量: "
                        f"{len(result['topk_results'].get('topK_results', [])) if 'topk_results' in result else 0}\n")
            else:
                f.write(f"错误信息: {result['error']}\n")

        # 定期保存结果（每5个分子保存一次）
        if (i + 1) % 5 == 0 or (i + 1) == total_count:
            save_path = save_results(results_dict, args.output_json, output_dir)

    # 记录总耗时
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time

    with open(total_log_file, 'a') as f:
        f.write(f"\n=== 批量处理完成 ===\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write(f"平均每个分子耗时: {total_time/total_count:.2f} 秒\n")

    return results_dict


def save_results(results_dict, output_json, output_dir):
    """保存结果到JSON文件"""
    if not output_json:
        output_json = os.path.join(output_dir, "batch_results.json")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    return output_json


# ---------------------------------------------------------------------------
#  主函数
# ---------------------------------------------------------------------------
def main():
    """主函数"""
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)

    # 创建主日志文件
    main_log_file = os.path.join(output_dir, "batch_optimization_main.log")

    print(f"=== MO 批量分子优化脚本 ===")
    print(f"输出目录: {output_dir}")
    print(f"主日志文件: {main_log_file}")
    print(f"读取CSV文件: {args.input_csv}")
    print(f"预测类型: {args.prediction_type}")

    # 记录配置信息到主日志
    with open(main_log_file, 'w') as f:
        f.write(f"=== MO 批量分子优化配置 ===\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"输入CSV文件: {args.input_csv}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"模型目录: {args.model_dir}\n")
        f.write(f"配置文件: {args.config_file}\n")
        f.write(f"目标属性: {args.target_property}\n")
        f.write(f"预测类型: {args.prediction_type}\n")
        f.write(f"优化模式: {args.optimization_mode}\n")
        f.write(f"最大深度: {args.max_depth}\n")
        f.write(f"最大分支数: {args.max_branching}\n")
        f.write(f"优化方向: {args.direction}\n")
        f.write(f"剪枝耐心值: {args.pruning_patience}\n")
        f.write(f"logP最小值: {args.logp_min}\n")
        f.write(f"logP最大值: {args.logp_max}\n")
        f.write(f"logP耐心值: {args.logp_patience}\n")
        f.write(f"topK值: {args.topK}\n")
        f.write(f"起始索引: {args.start_index}\n")
        f.write(f"结束索引: {args.end_index}\n")
        f.write(f"搜索模式: {args.search_mode}\n")
        if args.search_mode == 'mcts':
            f.write(f"MCTS 模拟轮数: {args.num_simulations}\n")
            f.write(f"MCTS 探索系数: {args.exploration_weight}\n")
            f.write(f"MCTS prior 模式: {args.mcts_prior_mode}\n")
            f.write(f"MCTS value 模式: {args.mcts_value_mode}\n")
            f.write(f"MCTS expansion 模式: {args.mcts_expansion_mode}\n")
            f.write(f"MCTS 随机种子: {args.mcts_random_seed}\n")
        if args.search_mode == 'astar_demo':
            f.write(f"PolicyNet 权重: {args.policy_path}\n")
            f.write(f"ValueNet 权重: {args.value_path}\n")
            f.write(f"RL 模式: {'train' if args.rl_train else 'eval'}\n")
            f.write(f"top_n_prefilter: {args.top_n_prefilter}\n")
            f.write(f"open_set_budget: {args.open_set_budget}\n")
        f.write("\n")

    # 读取数据
    data_list = read_csv_data(args.input_csv, args.start_index, args.end_index, args.target_property)
    print(f"读取到 {len(data_list)} 个分子数据")

    with open(main_log_file, 'a') as f:
        f.write(f"读取到 {len(data_list)} 个分子数据\n")

    # 执行批量处理
    start_time = time.time()
    results_dict = batch_process(data_list, args, output_dir)
    end_time = time.time()

    # 保存最终结果
    final_output = save_results(results_dict, args.output_json, output_dir)

    # 统计结果
    success_count = sum(1 for r in results_dict.values()
                       if r['optimization_result']['status'] == 'success')
    failure_count = len(results_dict) - success_count

    with open(main_log_file, 'a') as f:
        f.write(f"\n=== 批量处理完成 ===\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {end_time - start_time:.2f} 秒\n")
        f.write(f"成功: {success_count}, 失败: {failure_count}\n")
        f.write(f"成功率: {success_count/len(results_dict)*100:.1f}%\n")
        f.write(f"结果文件: {final_output}\n")

    print(f"\n=== 批量处理完成！ ===")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功: {success_count}, 失败: {failure_count}")
    print(f"成功率: {success_count/len(results_dict)*100:.1f}%")
    print(f"结果文件: {final_output}")
    print(f"详细日志已保存到: {output_dir}")


if __name__ == "__main__":
    main()
