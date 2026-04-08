#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2：在线 RL 训练脚本（A* RL Demo）

在 astar_demo 搜索过程中，以 REINFORCE 策略梯度持续微调
PolicyNet / ValueNet，实现从 BC 冷启动到在线进化的闭环。

每个 episode = 一次完整的 astar_demo 搜索（针对一个输入分子）。
训练结束后输出 rl_ckpt_ep<N>.pth 权重文件，可直接用于 --policy-path / --value-path。

使用方式：
    python mol_evo/train_astar_rl_demo.py \\
        --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \\
        --model-path /path/to/ofo_model.pth \\
        --model-dir  /path/to/ofo_model_dir \\
        --config-file /path/to/config.yaml \\
        --policy-path mol_evo/output/astar_rl/bc/policy_best.pth \\
        --value-path  mol_evo/output/astar_rl/bc/value_best.pth \\
        --output-dir  mol_evo/output/astar_rl/rl \\
        --num-episodes 200 \\
        --direction decrease
"""

import os
import sys
import json
import random
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# 项目根路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RLTrainer
    from mol_evo.core.models.astar_rl.reward import RewardConfig
    from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM
    from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
except ImportError as e:
    import traceback
    print(f"无法导入所需模块: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """设置 console + file 双输出 logger。"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_rl.log")
    logger = logging.getLogger("astar_rl_demo")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# 在线 RL 训练主函数
# ---------------------------------------------------------------------------

def train_astar_rl(
    input_csv: str,
    model_path: str,
    model_dir: str,
    config_file: str,
    output_dir: str,
    policy_path: Optional[str] = None,
    value_path: Optional[str] = None,
    direction: str = "decrease",
    target_property: str = "lumo",
    optimization_mode: str = "sub",
    num_episodes: int = 200,
    max_depth: int = 4,
    max_branching: int = 8,
    logp_min: float = 0.0,
    logp_max: float = 5.0,
    top_n_prefilter: int = 20,
    open_set_budget: int = 200,
    lr_policy: float = 1e-4,
    lr_value: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    checkpoint_every: int = 50,
    seed: int = 42,
    device_str: str = "auto",
) -> None:
    """在线 RL 训练主函数。"""

    # 随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 输出目录与日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"rl_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)

    # 设备
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"使用设备: {device}")

    # 读取分子数据
    logger.info(f"读取输入 CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    smiles_list = df["smiles"].tolist()
    property_values = (
        df[target_property].tolist()
        if target_property in df.columns
        else [None] * len(df)
    )
    logger.info(f"共 {len(smiles_list)} 个分子")

    # 模型网络
    policy_net = PolicyNet(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    value_net = ValueNet(state_dim=STATE_DIM).to(device)

    if policy_path and os.path.isfile(policy_path):
        policy_net.load_state_dict(torch.load(policy_path, map_location=device))
        logger.info(f"PolicyNet 权重加载: {policy_path}")
    else:
        logger.info("PolicyNet 使用随机初始化（未指定权重或文件不存在）")

    if value_path and os.path.isfile(value_path):
        value_net.load_state_dict(torch.load(value_path, map_location=device))
        logger.info(f"ValueNet 权重加载: {value_path}")
    else:
        logger.info("ValueNet 使用随机初始化（未指定权重或文件不存在）")

    logger.info(
        f"PolicyNet 参数量: {sum(p.numel() for p in policy_net.parameters()):,}"
    )
    logger.info(
        f"ValueNet  参数量: {sum(p.numel() for p in value_net.parameters()):,}"
    )

    # RLTrainer
    rl_trainer = RLTrainer(
        policy_net=policy_net,
        value_net=value_net,
        reward_config=RewardConfig(direction=direction),
        lr_policy=lr_policy,
        lr_value=lr_value,
        gamma=gamma,
        entropy_coef=entropy_coef,
        device=str(device),
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        checkpoint_every=checkpoint_every,
    )

    # OFO 优化器（不使用 initial_smiles_csv，逐个设置）
    optimizer = EvolutionTreeOptimizer(
        model_path=model_path,
        model_dir=model_dir,
        config_file=config_file,
        initial_smiles_csv=None,
        target_property=target_property,
        initial_property_value=None,
        optimization_mode=optimization_mode,
    )
    optimizer.optimization_direction = direction

    # 训练历史
    history: List[Dict] = []
    best_episode_return = float("-inf")

    logger.info(f"开始在线 RL 训练，共 {num_episodes} 个 episode")

    for ep in range(1, num_episodes + 1):
        # 随机选择一个分子
        idx = random.randint(0, len(smiles_list) - 1)
        smiles = smiles_list[idx]
        prop_val = property_values[idx]

        optimizer.initial_property_value = prop_val

        ep_start = time.time()
        try:
            _ = optimizer.optimize_evolution_tree(
                initial_smiles=smiles,
                max_depth=max_depth,
                max_branching=max_branching,
                optimization_direction=direction,
                logp_range=(logp_min, logp_max),
                search_mode="astar_demo",
                policy_net=policy_net,
                value_net=value_net,
                rl_trainer=rl_trainer,
                top_n_prefilter=top_n_prefilter,
                open_set_budget=open_set_budget,
            )
        except Exception as exc:
            logger.warning(f"Episode {ep} 搜索失败 (smiles={smiles[:20]}): {exc}")
            # 即使搜索失败，end_episode 也可能已被内部调用；此处跳过
            continue

        ep_time = time.time() - ep_start

        # end_episode 已在 generate_expansion_tree_astar_demo 内部调用
        # 从 history 中取最新记录
        ep_stats = rl_trainer.history[-1] if rl_trainer.history else {}
        ep_return = ep_stats.get("episode_return", 0.0)

        row = {
            "episode": ep,
            "smiles": smiles,
            "episode_return": ep_return,
            "policy_loss": ep_stats.get("policy_loss", 0.0),
            "value_loss": ep_stats.get("value_loss", 0.0),
            "episode_steps": ep_stats.get("episode_steps", 0),
            "time_sec": ep_time,
        }
        history.append(row)

        # 保存最佳 checkpoint
        if ep_return > best_episode_return:
            best_episode_return = ep_return
            torch.save(
                policy_net.state_dict(),
                os.path.join(run_dir, "policy_best.pth"),
            )
            torch.save(
                value_net.state_dict(),
                os.path.join(run_dir, "value_best.pth"),
            )

        if ep % 10 == 0 or ep == 1:
            logger.info(
                f"Episode {ep:>4d}/{num_episodes} | "
                f"return={ep_return:.4f} "
                f"p_loss={row['policy_loss']:.6f} "
                f"v_loss={row['value_loss']:.6f} "
                f"steps={row['episode_steps']} "
                f"t={ep_time:.1f}s"
            )

    # 保存最终 checkpoint
    torch.save(policy_net.state_dict(), os.path.join(run_dir, "policy_last.pth"))
    torch.save(value_net.state_dict(), os.path.join(run_dir, "value_last.pth"))

    # 保存训练历史
    with open(os.path.join(run_dir, "rl_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 保存配置
    config = {
        "input_csv": input_csv,
        "direction": direction,
        "target_property": target_property,
        "num_episodes": num_episodes,
        "max_depth": max_depth,
        "max_branching": max_branching,
        "logp_min": logp_min,
        "logp_max": logp_max,
        "top_n_prefilter": top_n_prefilter,
        "open_set_budget": open_set_budget,
        "lr_policy": lr_policy,
        "lr_value": lr_value,
        "gamma": gamma,
        "entropy_coef": entropy_coef,
        "seed": seed,
        "best_episode_return": best_episode_return,
    }
    with open(os.path.join(run_dir, "rl_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"训练完成！best_episode_return={best_episode_return:.4f}")
    logger.info(f"权重保存路径: {run_dir}")
    print(f"\n=== A* RL Demo 在线训练完成 ===")
    print(f"best_episode_return: {best_episode_return:.4f}")
    print(f"policy_best: {os.path.join(run_dir, 'policy_best.pth')}")
    print(f"value_best:  {os.path.join(run_dir, 'value_best.pth')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A* RL Demo - Phase 2: 在线 RL 训练入口"
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="输入分子 CSV 文件（含 smiles 列）",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="OFO 模型权重路径（.pth）",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="OFO 模型目录",
    )
    parser.add_argument(
        "--config-file", type=str, required=True,
        help="OFO 配置文件路径（.yaml）",
    )
    parser.add_argument(
        "--policy-path", type=str, default=None,
        help="BC 预训练 PolicyNet 权重路径（可选，不指定则随机初始化）",
    )
    parser.add_argument(
        "--value-path", type=str, default=None,
        help="BC 预训练 ValueNet 权重路径（可选，不指定则随机初始化）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="mol_evo/output/astar_rl/rl",
        help="输出目录（会在其下创建带时间戳的子目录）",
    )
    parser.add_argument("--direction", type=str, choices=["increase", "decrease"],
                        default="decrease")
    parser.add_argument("--target-property", type=str, default="lumo")
    parser.add_argument("--optimization-mode", type=str, choices=["sub", "pct"],
                        default="sub")
    parser.add_argument("--num-episodes", type=int, default=200,
                        help="在线 RL 训练 episode 总数")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-branching", type=int, default=8)
    parser.add_argument("--logp-min", type=float, default=0.0)
    parser.add_argument("--logp-max", type=float, default=5.0)
    parser.add_argument("--top-n-prefilter", type=int, default=20,
                        help="PolicyNet 预筛候选数")
    parser.add_argument("--open-set-budget", type=int, default=200,
                        help="A* open set 展开预算")
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="每多少个 episode 保存一次 checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', 或 'auto'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_astar_rl(
        input_csv=args.input_csv,
        model_path=args.model_path,
        model_dir=args.model_dir,
        config_file=args.config_file,
        output_dir=args.output_dir,
        policy_path=args.policy_path,
        value_path=args.value_path,
        direction=args.direction,
        target_property=args.target_property,
        optimization_mode=args.optimization_mode,
        num_episodes=args.num_episodes,
        max_depth=args.max_depth,
        max_branching=args.max_branching,
        logp_min=args.logp_min,
        logp_max=args.logp_max,
        top_n_prefilter=args.top_n_prefilter,
        open_set_budget=args.open_set_budget,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
