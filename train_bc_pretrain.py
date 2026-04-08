#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1：BC 冷启动预训练脚本（A* RL Demo）

从 export_rl_demo_transitions.py 导出的离线样本中：
  - BC 预训练 PolicyNet（cross-entropy，模仿 BFS/MCTS 的展开选择分布）
  - BC 预训练 ValueNet（MSE，回归 future_best_gain）

训练风格参考 train_v0_3_path.py：随机种子固定、DualLogger、EarlyStop、
周期性保存 best/last checkpoint。

使用方式:
    python mol_evo/train_bc_pretrain.py \\
        --data-json mol_evo/dataset/rl_demo/bc_transitions.json \\
        --output-dir mol_evo/output/astar_rl/bc \\
        --direction decrease \\
        --epochs 100 \\
        --batch-size 64 \\
        --lr-policy 1e-4 \\
        --lr-value 1e-3
"""

import os
import sys
import json
import random
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

# 项目根路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RewardConfig
    from mol_evo.core.data.rl_demo_processing import BCDataset, build_bc_dataloader, STATE_DIM, ACTION_DIM
except ImportError as e:
    import traceback
    print(f"无法导入所需模块: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 日志工具（复用 DualLogger 风格，不依赖项目内部模块）
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """设置 console + file 双输出 logger。"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_bc.log")
    logger = logging.getLogger("bc_pretrain")
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
# BC Policy loss
# ---------------------------------------------------------------------------

def compute_policy_bc_loss(
    policy_net: PolicyNet,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """BC 策略损失：给定一条 (state, action) 样本，把 action 视为正样本，
    其他 batch 内样本的 action 视为负样本（in-batch 对比）。

    简化版本：由于 BC 样本中每条样本只有 1 个动作（selected action），
    我们用自监督方式：把同一 batch 内其他样本的 actions 作为 "负样本候选"，
    计算 softmax cross-entropy。

    直觉：如果搜索时选了这个 action，则它在当前 state 下的 logit 应该最高。
    """
    B = states.shape[0]
    device = states.device

    # 对每个样本，将 batch 内所有 actions 作为候选
    # actions: (B, ACTION_DIM)
    logits_list = []
    for i in range(B):
        s_i = states[i]                    # (STATE_DIM,)
        # logit for each action in batch
        logits_i = policy_net(s_i, actions)  # (B,)
        logits_list.append(logits_i)

    logits = torch.stack(logits_list, dim=0)  # (B, B)
    # 正样本标签 = 对角线（第 i 个样本的正样本是第 i 个 action）
    targets = torch.arange(B, device=device)
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# 训练主函数
# ---------------------------------------------------------------------------

def train_bc(
    data_json: str,
    output_dir: str,
    direction: str = "decrease",
    epochs: int = 100,
    batch_size: int = 64,
    lr_policy: float = 1e-4,
    lr_value: float = 1e-3,
    val_ratio: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    max_depth: int = 4,
    device_str: str = "auto",
) -> None:
    """BC 预训练主函数。"""

    # 随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # 输出目录与日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"bc_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)

    # 设备
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"使用设备: {device}")

    # 数据加载
    logger.info(f"加载数据: {data_json}")
    with open(data_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info(f"共 {len(samples)} 条样本")

    full_dataset = BCDataset(samples, max_depth=max_depth, direction=direction)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    logger.info(f"训练集: {n_train}，验证集: {n_val}")

    # 模型
    policy_net = PolicyNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    value_net = ValueNet(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    logger.info(
        f"PolicyNet 参数量: {sum(p.numel() for p in policy_net.parameters()):,}"
    )
    logger.info(
        f"ValueNet  参数量: {sum(p.numel() for p in value_net.parameters()):,}"
    )

    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=lr_value)

    # 训练记录
    best_val_loss = float("inf")
    patience_counter = 0
    history: List[Dict] = []

    logger.info(f"开始 BC 预训练，共 {epochs} 轮")

    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        policy_net.train()
        value_net.train()
        train_ploss, train_vloss, train_count = 0.0, 0.0, 0

        for batch in train_loader:
            states = batch["state"].to(device)          # (B, STATE_DIM)
            actions = batch["action"].to(device)        # (B, ACTION_DIM)
            future_gains = batch["future_best_gain"].to(device)  # (B,)

            # Policy BC loss（in-batch cross-entropy）
            p_loss = compute_policy_bc_loss(policy_net, states, actions)

            # Value BC loss（MSE 回归 future_best_gain）
            v_loss = value_net.bc_loss(states, future_gains)

            policy_opt.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            policy_opt.step()

            value_opt.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), 5.0)
            value_opt.step()

            B = states.shape[0]
            train_ploss += p_loss.item() * B
            train_vloss += v_loss.item() * B
            train_count += B

        train_ploss /= max(train_count, 1)
        train_vloss /= max(train_count, 1)

        # --- 验证 ---
        policy_net.eval()
        value_net.eval()
        val_vloss, val_count = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(device)
                future_gains = batch["future_best_gain"].to(device)
                v_loss = value_net.bc_loss(states, future_gains)
                B = states.shape[0]
                val_vloss += v_loss.item() * B
                val_count += B
        val_vloss /= max(val_count, 1)

        # 总验证 loss（policy loss 在验证集上不直接适用，用 value loss 作早停信号）
        val_total = val_vloss

        row = {
            "epoch": epoch,
            "train_policy_loss": train_ploss,
            "train_value_loss": train_vloss,
            "val_value_loss": val_vloss,
        }
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4d}/{epochs} | "
                f"train_p={train_ploss:.6f} train_v={train_vloss:.6f} "
                f"val_v={val_vloss:.6f}"
            )

        # --- 保存 best ---
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(policy_net.state_dict(), os.path.join(run_dir, "policy_best.pth"))
            torch.save(value_net.state_dict(), os.path.join(run_dir, "value_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发（patience={patience}），epoch={epoch}")
                break

    # 保存最终 checkpoint
    torch.save(policy_net.state_dict(), os.path.join(run_dir, "policy_last.pth"))
    torch.save(value_net.state_dict(), os.path.join(run_dir, "value_last.pth"))

    # 保存训练历史
    with open(os.path.join(run_dir, "bc_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 保存配置
    config = {
        "data_json": data_json,
        "direction": direction,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_policy": lr_policy,
        "lr_value": lr_value,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "max_depth": max_depth,
        "seed": seed,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(run_dir, "bc_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"训练完成！best_val_loss={best_val_loss:.6f}")
    logger.info(f"权重保存路径: {run_dir}")
    print(f"\n=== BC 预训练完成 ===")
    print(f"best_val_loss: {best_val_loss:.6f}")
    print(f"policy_best:   {os.path.join(run_dir, 'policy_best.pth')}")
    print(f"value_best:    {os.path.join(run_dir, 'value_best.pth')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A* RL Demo - Phase 1: BC 预训练")
    parser.add_argument(
        "--data-json", type=str, required=True,
        help="export_rl_demo_transitions.py 输出的 JSON 样本文件",
    )
    parser.add_argument(
        "--output-dir", type=str, default="mol_evo/output/astar_rl/bc",
        help="输出目录（会在其下创建带时间戳的子目录）",
    )
    parser.add_argument("--direction", type=str, choices=["increase", "decrease"],
                        default="decrease")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', 或 'auto'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_bc(
        data_json=args.data_json,
        output_dir=args.output_dir,
        direction=args.direction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_depth=args.max_depth,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
