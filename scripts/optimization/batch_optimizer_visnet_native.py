#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用evolution_optimizer.py的 MO 优化脚本 —— 原版 ViSNet 基座模型适配版

支持原版 ViSNet（torchgeom 版本）的 .ckpt 权重，
模型预测单分子属性真值（如 mu/dipole_moment），
通过 NativeVisnetAdapter 自动计算 pred(smiles_to) - pred(smiles_from) 作为属性变化值。

与 batch_optimizer_visnet.py 的区别：
- batch_optimizer_visnet.py：适配我们的 pair-based ViSNet（MoleculeEvolutionVisnetLinearPredictor）
- 本脚本：适配原版 ViSNet（visnet.models.model.ViSNet），需要 3D 坐标输入

使用示例：
    python mol_evo/scripts/optimization/batch_optimizer_visnet_native.py \
      --input-csv mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv \
      --model-path /root/autodl-tmp/projects/visnet-qm9/logs/.../last.ckpt \
      --target-property mu \
      --direction increase \
      --start-index 0 --end-index 50 \
      --search-mode mcts --num-simulations 800
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
import re

# 全局中断标志
interrupted = False

def signal_handler(sig, frame):
    """处理中断信号"""
    global interrupted
    if interrupted:
        print("再次收到中断信号，立即退出！")
        sys.exit(0)
    print("正在中断处理过程，请稍候...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)


try:
    from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
#  NativeVisnetAdapter: 原版 ViSNet 基座模型适配器
# ---------------------------------------------------------------------------
class NativeVisnetAdapter(EvolutionTreeOptimizer):
    """
    适配器：原版 ViSNet（预测单分子属性真值）适配到 EvolutionTreeOptimizer 接口。

    重写 predict_property_change / predict_batch，
    通过 pred(smiles_to) - pred(smiles_from) 计算差值。

    与 TrueValueAdapter 的区别：
    - TrueValueAdapter 仍使用 pair-based 模型架构（from_data, to_data, edge_attr）
    - NativeVisnetAdapter 使用原版 ViSNet（data.z, data.pos），只接收单分子输入
    """

    def __init__(self, model_path, model_dir, config_file=None,
                 initial_smiles_csv=None, target_property=None,
                 initial_property_value=None, optimization_mode='pct',
                 visnet_project_path=None):
        """
        Args:
            model_path: 原版 ViSNet .ckpt 文件路径
            model_dir: 模型目录路径（用于兼容接口，本适配器不使用）
            config_file: 配置文件路径（用于操作配置）
            visnet_project_path: visnet-qm9 项目路径，用于加载模型代码
        """
        # 不调用 super().__init__()，因为那会加载 pair-based 模型
        # 手动初始化必要属性
        self.model_path = model_path
        self.model_dir = model_dir
        self.config_file = config_file
        self.initial_smiles_csv = initial_smiles_csv
        self.target_property = target_property
        self.initial_property_value = initial_property_value
        self.optimization_mode = optimization_mode
        self.model = None
        self.property_stats = None
        self.molecule_cache = None
        self.initial_properties = {}
        self.generation_time = 0
        self.attempt_count = 0
        self._initial_value_cache = {}
        self._visnet_model = None  # 原版 ViSNet 模型

        # 加载必要组件
        self._init_components(visnet_project_path)

    def _init_components(self, visnet_project_path=None):
        """初始化所有必要组件"""
        import torch
        from mol_evo.core.utils.molecule import MoleculeCache

        # 初始化分子缓存
        self.molecule_cache = MoleculeCache("prediction_dataset")

        # 加载操作配置
        if self.config_file and os.path.exists(self.config_file):
            from mol_evo.core.data.processing import load_operation_config
            load_operation_config(config_path=self.config_file)
        elif self.model_dir:
            from mol_evo.core.data.processing import load_operation_config
            dataset_config_path = os.path.join(self.model_dir, "data_config.json")
            if os.path.exists(dataset_config_path):
                load_operation_config(dataset_path=dataset_config_path)

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载原版 ViSNet 模型
        self._load_native_visnet(visnet_project_path)

        # 加载初始属性
        if self.initial_smiles_csv:
            self._load_initial_properties()

        print(f"[NativeVisnetAdapter] 模式已启用：原版 ViSNet 预测真值，自动计算 sub 差值")
        print(f"[NativeVisnetAdapter] 目标属性: {self.target_property}")
        print(f"[NativeVisnetAdapter] 设备: {self.device}")

    def _load_native_visnet(self, visnet_project_path=None):
        """加载原版 ViSNet .ckpt 模型"""
        import torch

        # 添加 visnet-qm9 项目路径
        if visnet_project_path and os.path.isdir(visnet_project_path):
            sys.path.insert(0, visnet_project_path)

        try:
            from visnet.models.model import load_model
        except ImportError:
            print("[NativeVisnetAdapter] 错误：无法导入 visnet 模型代码。")
            print("  请通过 --visnet-project-path 指定 visnet-qm9 项目路径，")
            print("  或将其添加到 PYTHONPATH 中。")
            raise

        print(f"[NativeVisnetAdapter] 加载原版 ViSNet 模型: {self.model_path}")
        self._visnet_model = load_model(self.model_path, device=str(self.device))
        self._visnet_model.eval()
        print(f"[NativeVisnetAdapter] 模型加载成功")

        # 设置 property_stats 为空（原版模型不需要反标准化）
        self.property_stats = {}

    def _load_initial_properties(self):
        """从CSV文件加载起始分子的属性"""
        if not os.path.exists(self.initial_smiles_csv):
            print(f"警告: CSV文件不存在: {self.initial_smiles_csv}")
            return

        try:
            df = pd.read_csv(self.initial_smiles_csv)
            for _, row in df.iterrows():
                smiles = row['smiles']
                if self.target_property in row:
                    self.initial_properties[smiles] = row[self.target_property]
        except Exception as e:
            print(f"加载初始属性失败: {e}")

    # ------------------------------------------------------------------
    #  核心：单分子真值预测
    # ------------------------------------------------------------------
    def _predict_true_value(self, smiles: str) -> Optional[float]:
        """用原版 ViSNet 预测单个分子的属性真值"""
        if smiles in self._initial_value_cache:
            return self._initial_value_cache[smiles]

        try:
            import torch
            from torch_geometric.data import Batch
            from mol_evo.core.data.data_v0 import smiles_to_graph_data

            graph_data = smiles_to_graph_data(smiles, self.molecule_cache)
            if graph_data is None:
                return None

            batch_data = Batch.from_data_list([graph_data]).to(self.device)

            with torch.no_grad():
                out, _ = self._visnet_model(batch_data)

            predicted_value = out[0].item()

            self._initial_value_cache[smiles] = predicted_value
            return predicted_value

        except Exception as e:
            print(f"[NativeVisnetAdapter] _predict_true_value 出错: smiles={smiles}, error={e}")
            return None

    # ------------------------------------------------------------------
    #  核心：批量单分子真值预测
    # ------------------------------------------------------------------
    def _predict_true_value_batch(self, smiles_set, batch_size=32):
        """批量预测一组 SMILES 的真值，返回 {smiles: value}"""
        import torch
        from torch_geometric.data import Batch
        from mol_evo.core.data.data_v0 import smiles_to_graph_data

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
        valid_smiles = []

        for smiles in to_predict:
            try:
                graph_data = smiles_to_graph_data(smiles, self.molecule_cache)
                if graph_data is None:
                    continue
                graph_list.append(graph_data)
                valid_smiles.append(smiles)
            except Exception as e:
                print(f"[NativeVisnetAdapter] 准备数据出错: smiles={smiles}, error={e}")
                continue

        if not graph_list:
            return value_map

        # 分批推理
        num_samples = len(graph_list)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_graphs = graph_list[start_idx:end_idx]
            batch_data = Batch.from_data_list(batch_graphs).to(self.device)

            try:
                with torch.no_grad():
                    out, _ = self._visnet_model(batch_data)

                batch_preds = out.squeeze(-1).cpu().tolist()
                if isinstance(batch_preds, float):
                    batch_preds = [batch_preds]

                for s, v in zip(valid_smiles[start_idx:end_idx], batch_preds):
                    value_map[s] = v
                    self._initial_value_cache[s] = v

            except Exception as e:
                print(f"[NativeVisnetAdapter] 批量推理出错: start={start_idx}, end={end_idx}, error={e}")
                # 单个预测作为 fallback
                for s in valid_smiles[start_idx:end_idx]:
                    val = self._predict_true_value(s)
                    if val is not None:
                        value_map[s] = val

        return value_map

    # ------------------------------------------------------------------
    #  接口：单对预测
    # ------------------------------------------------------------------
    def predict_property_change(self, smiles_from, smiles_to, operation_details):
        """用真值模型分别预测 from / to 分子的属性值，返回差值"""
        try:
            pred_from = self._predict_true_value(smiles_from)
            pred_to = self._predict_true_value(smiles_to)

            if pred_from is None or pred_to is None:
                return None

            change = pred_to - pred_from
            return change

        except Exception as e:
            print(f"[NativeVisnetAdapter] predict_property_change 出错: "
                  f"from={smiles_from}, to={smiles_to}, error={e}")
            return None

    # ------------------------------------------------------------------
    #  接口：批量预测
    # ------------------------------------------------------------------
    def predict_batch(self, from_smiles_list, to_smiles_list,
                      operation_details_list, batch_size=32):
        """批量版：分别预测 from / to 真值，返回差值列表"""
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
    #  兼容方法：EvolutionTreeOptimizer 需要的方法
    # ------------------------------------------------------------------
    def _add_batch_info(self, data):
        """兼容接口（原版 ViSNet 不需要）"""
        return data

    def _prediction_tensor_to_list(self, predictions):
        """将预测张量转为列表"""
        if hasattr(predictions, 'numpy'):
            return predictions.squeeze(-1).numpy().tolist()
        elif isinstance(predictions, (list, tuple)):
            return predictions
        return [float(predictions)]


# ---------------------------------------------------------------------------
#  命令行参数
# ---------------------------------------------------------------------------
def parse_args():
    """解析命令行参数"""
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
        description='MO 批量优化分子属性 —— 原版 ViSNet 基座模型适配版'
    )
    parser.add_argument('--input-csv', type=str,
                        default=default_input_csv,
                        help='输入的CSV文件路径')
    parser.add_argument('--output-json', type=str,
                        help='输出的JSON文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径；若不提供则自动创建时间戳目录')
    parser.add_argument('--model-path', type=str, required=True,
                        help='原版 ViSNet .ckpt 模型文件路径')
    parser.add_argument('--model-dir', type=str, default='',
                        help='模型目录路径（本适配器不使用，但保留接口兼容）')
    parser.add_argument('--config-file', type=str,
                        default=default_config_file,
                        help='操作配置文件路径')
    parser.add_argument('--target-property', type=str, default='mu',
                        help='目标属性名称')
    parser.add_argument('--visnet-project-path', type=str, default=None,
                        help='visnet-qm9 项目路径（用于加载 ViSNet 模型代码）')
    # ---- 优化参数 ----
    parser.add_argument('--optimization-mode', type=str, choices=['sub', 'pct'], default='sub',
                        help='优化模式')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='最大演化深度')
    parser.add_argument('--max-branching', type=int, default=8,
                        help='最大分支数')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], default='increase',
                        help='优化方向')
    parser.add_argument('--pruning-patience', type=int, default=2,
                        help='剪枝耐心值')
    parser.add_argument('--logp-min', type=float, default=0.0,
                        help='logP的最小值')
    parser.add_argument('--logp-max', type=float, default=5.0,
                        help='logP的最大值')
    parser.add_argument('--logp-patience', type=int, default=3,
                        help='logP剪枝耐心值')
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
                        default='bfs', help='搜索模式')
    parser.add_argument('--num-simulations', type=int, default=200,
                        help='MCTS 模拟轮数')
    parser.add_argument('--exploration-weight', type=float, default=1.4,
                        help='MCTS PUCT 探索系数')
    parser.add_argument('--mcts-prior-mode', type=str, choices=['softmax', 'uniform'], default='softmax',
                        help='MCTS prior 构造方式')
    parser.add_argument('--mcts-value-mode', type=str, choices=['accumulated', 'zero', 'step'], default='accumulated',
                        help='MCTS 叶节点价值')
    parser.add_argument('--mcts-expansion-mode', type=str, choices=['topk', 'random_topk', 'full'], default='topk',
                        help='MCTS 扩展策略')
    parser.add_argument('--mcts-random-seed', type=int, default=None,
                        help='MCTS 随机种子')
    # astar_demo 参数
    parser.add_argument('--policy-path', type=str, default=None,
                        help='PolicyNet 权重路径')
    parser.add_argument('--value-path', type=str, default=None,
                        help='ValueNet 权重路径')
    parser.add_argument('--rl-train', action='store_true',
                        help='astar_demo 在线 RL 训练模式')
    parser.add_argument('--rl-eval', action='store_true',
                        help='astar_demo 纯评估模式')
    parser.add_argument('--top-n-prefilter', type=int, default=20,
                        help='PolicyNet 预筛候选数')
    parser.add_argument('--open-set-budget', type=int, default=200,
                        help='A* open set 展开预算')
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
                            f"batch_optimization_mu_visnet_native_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def read_csv_data(csv_path, start_idx=0, end_idx=-1, target_property='mu'):
    """读取CSV数据"""
    df = pd.read_csv(csv_path)
    if end_idx > 0:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]

    data_list = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        property_value = row.get(target_property, None)

        data_list.append({
            'smiles': smiles,
            'property_value': property_value,
            'original_row': row.to_dict()
        })
    return data_list


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
    print(f"基座模型: 原版 ViSNet (native)")
    print(f"总日志文件: {total_log_file}")

    # 记录开始时间
    batch_start_time = time.time()

    # 创建优化器
    optimizer = NativeVisnetAdapter(
        model_path=args.model_path,
        model_dir=args.model_dir,
        config_file=args.config_file,
        initial_smiles_csv=None,
        target_property=args.target_property,
        initial_property_value=None,
        optimization_mode=args.optimization_mode,
        visnet_project_path=args.visnet_project_path,
    )

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
                    torch.load(args.value_path, map_location=_device, weights_only=True)
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
            print(f"[astar_demo] 警告：RL 模型加载失败: {_e}")
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

        # 定期保存结果
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

    print(f"=== MO 批量分子优化脚本 (原版 ViSNet 基座) ===")
    print(f"输出目录: {output_dir}")
    print(f"主日志文件: {main_log_file}")
    print(f"读取CSV文件: {args.input_csv}")
    print(f"基座模型: 原版 ViSNet (native)")

    # 记录配置信息到主日志
    with open(main_log_file, 'w') as f:
        f.write(f"=== MO 批量分子优化配置 (原版 ViSNet 基座) ===\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"输入CSV文件: {args.input_csv}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"模型目录: {args.model_dir}\n")
        f.write(f"配置文件: {args.config_file}\n")
        f.write(f"目标属性: {args.target_property}\n")
        f.write(f"基座模型类型: 原版 ViSNet (native)\n")
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
        if args.visnet_project_path:
            f.write(f"ViSNet项目路径: {args.visnet_project_path}\n")
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
