#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2：在线 RL 训练脚本（A* RL Demo）

在 astar_demo 搜索过程中，以 REINFORCE 或 PPO 策略梯度持续微调
PolicyNet / ValueNet，实现从 BC 冷启动到在线进化的闭环。

每个 episode = 一次完整的 astar_demo 搜索（针对一个输入分子）。
训练结束后输出 rl_ckpt_ep<N>.pth 权重文件，可直接用于 --policy-path / --value-path。
若提供 holdout，则优先按固定 holdout 指标选择 `policy_best.pth / value_best.pth`。
"""

import os
import sys
import json
import random
import argparse
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# 项目根路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

try:
    from mol_evo.core.models.astar_rl import (
        PolicyNet,
        ValueNet,
        RLTrainer,
        PPORLTrainer,
    )
    from mol_evo.core.models.astar_rl.reward import RewardConfig
    from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM
    from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
    from mol_evo.scripts.eval_astar_rl_holdout import (
        summarize_eval_json,
        summarize_against_reference,
        write_outputs,
    )
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
# Trainer builder
# ---------------------------------------------------------------------------

def build_rl_trainer(
    algo: str,
    policy_net: PolicyNet,
    value_net: ValueNet,
    direction: str,
    lr_policy: float,
    lr_value: float,
    gamma: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    device: torch.device,
    checkpoint_dir: str,
    checkpoint_every: int,
    action_selection_mode: str,
    action_selection_temperature: float,
    action_selection_epsilon: float,
    clip_ratio: float,
    gae_lambda: float,
    update_epochs: int,
    minibatch_size: int,
    normalize_advantage: bool,
):
    reward_config = RewardConfig(direction=direction)
    device_str = str(device)
    if algo == "ppo":
        return PPORLTrainer(
            policy_net=policy_net,
            value_net=value_net,
            reward_config=reward_config,
            lr_policy=lr_policy,
            lr_value=lr_value,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            device=device_str,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            action_selection_mode=action_selection_mode,
            action_selection_temperature=action_selection_temperature,
            action_selection_epsilon=action_selection_epsilon,
            clip_ratio=clip_ratio,
            gae_lambda=gae_lambda,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            normalize_advantage=normalize_advantage,
        )
    if algo == "reinforce":
        return RLTrainer(
            policy_net=policy_net,
            value_net=value_net,
            reward_config=reward_config,
            lr_policy=lr_policy,
            lr_value=lr_value,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            device=device_str,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            action_selection_mode=action_selection_mode,
            action_selection_temperature=action_selection_temperature,
            action_selection_epsilon=action_selection_epsilon,
        )
    raise ValueError(f"不支持的 algo: {algo}")


# ---------------------------------------------------------------------------
# Holdout 评估 / best 选优 helper
# ---------------------------------------------------------------------------

def save_model_pair(
    policy_net: PolicyNet,
    value_net: ValueNet,
    policy_path: str,
    value_path: str,
) -> None:
    """保存当前 policy/value 权重。"""
    os.makedirs(os.path.dirname(os.path.abspath(policy_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(value_path)), exist_ok=True)
    torch.save(policy_net.state_dict(), policy_path)
    torch.save(value_net.state_dict(), value_path)


def resolve_holdout_eval_every(
    holdout_csv: Optional[str],
    holdout_eval_every: int,
    checkpoint_every: int,
) -> int:
    """解析 holdout 评估频率；未显式指定时回退到 checkpoint 周期。"""
    if not holdout_csv:
        return 0
    if holdout_eval_every > 0:
        return int(holdout_eval_every)
    return max(int(checkpoint_every), 1)


def build_holdout_priority(summary_payload: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """按文档优先级构造 holdout 选优元组。"""
    summary = summary_payload.get("summary") or {}
    comparison = summary_payload.get("comparison_vs_reference") or {}
    return (
        float(summary.get("top1_median", 0.0)),
        float(summary.get("trimmed_mean", 0.0)),
        float(comparison.get("win_rate", 0.0)),
        float(summary.get("top1_mean", 0.0)),
    )


def flatten_holdout_metrics(summary_payload: Dict[str, Any]) -> Dict[str, float]:
    """将 holdout 汇总结果压平成 history/config 友好的标量字段。"""
    summary = summary_payload.get("summary") or {}
    comparison = summary_payload.get("comparison_vs_reference") or {}
    metrics = {
        "holdout_mols": int(summary.get("mols", 0)),
        "holdout_nonempty_topk": int(summary.get("nonempty_topk", 0)),
        "holdout_top1_mean": float(summary.get("top1_mean", 0.0)),
        "holdout_top1_median": float(summary.get("top1_median", 0.0)),
        "holdout_trimmed_mean": float(summary.get("trimmed_mean", 0.0)),
        "holdout_actual_expansions_mean": float(summary.get("actual_expansions_mean", 0.0)),
        "holdout_ofo_calls_mean": float(summary.get("ofo_calls_mean", 0.0)),
    }
    if comparison:
        metrics.update({
            "holdout_shared_mols": int(comparison.get("shared_mols", 0)),
            "holdout_win_rate": float(comparison.get("win_rate", 0.0)),
            "holdout_delta_mean": float(comparison.get("delta_mean", 0.0)),
            "holdout_delta_median": float(comparison.get("delta_median", 0.0)),
            "holdout_trimmed_delta_mean": float(comparison.get("trimmed_delta_mean", 0.0)),
        })
    return metrics


def resolve_run_dir(
    output_dir: str,
    resume_checkpoint: Optional[str] = None,
    resume_run_dir: Optional[str] = None,
) -> Tuple[str, bool]:
    """解析本次训练使用的 run_dir；resume 时优先沿用旧目录。"""
    if resume_run_dir:
        return str(Path(resume_run_dir).resolve()), True
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint).resolve()
        if ckpt_path.parent.name == "checkpoints":
            return str(ckpt_path.parent.parent), True
        return str(ckpt_path.parent), True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str((Path(output_dir).resolve() / f"rl_{timestamp}")), False


def _read_json_if_exists(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_training_state(run_dir: str) -> Dict[str, Any]:
    """从 run_dir 恢复 history / best checkpoint / holdout 摘要等运行态。"""
    run_path = Path(run_dir).resolve()
    history_payload = _read_json_if_exists(run_path / "rl_history.json")
    history = history_payload if isinstance(history_payload, list) else []

    config_payload = _read_json_if_exists(run_path / "rl_config.json")
    config = config_payload if isinstance(config_payload, dict) else {}

    best_episode_return = float("-inf")
    try:
        best_episode_return = float(config.get("best_episode_return", float("-inf")))
    except (TypeError, ValueError):
        pass
    history_best_return = max(
        (float(row.get("episode_return", float("-inf"))) for row in history),
        default=float("-inf"),
    )
    best_episode_return = max(best_episode_return, history_best_return)

    best_holdout_payload_raw = _read_json_if_exists(run_path / "best_holdout_summary.json")
    best_holdout_payload = (
        best_holdout_payload_raw if isinstance(best_holdout_payload_raw, dict) else None
    )

    priority_raw = config.get("best_holdout_priority")
    best_holdout_priority: Optional[Tuple[float, float, float, float]] = None
    if isinstance(priority_raw, list) and len(priority_raw) == 4:
        try:
            best_holdout_priority = tuple(float(x) for x in priority_raw)
        except (TypeError, ValueError):
            best_holdout_priority = None
    elif best_holdout_payload is not None:
        best_holdout_priority = build_holdout_priority(best_holdout_payload)

    last_episode_index = 0
    try:
        last_episode_index = int(config.get("last_episode_index", 0))
    except (TypeError, ValueError):
        last_episode_index = 0
    history_last_episode = max(
        (int(row.get("episode", 0)) for row in history if row.get("episode") is not None),
        default=0,
    )
    last_episode_index = max(last_episode_index, history_last_episode)

    return {
        "history": history,
        "config": config,
        "best_episode_return": best_episode_return,
        "best_holdout_payload": best_holdout_payload,
        "best_holdout_priority": best_holdout_priority,
        "last_episode_index": last_episode_index,
    }


def build_training_config(
    *,
    base_config: Dict[str, Any],
    run_dir: str,
    is_resumed_run: bool,
    resume_checkpoint: Optional[str],
    episodes_completed: int,
    last_episode_index: int,
    best_episode_return: float,
    best_holdout_priority: Optional[Tuple[float, float, float, float]],
    best_holdout_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """构造可重复落盘的训练配置快照。"""
    best_checkpoint_metric = "holdout" if best_holdout_payload is not None else "episode_return"
    config: Dict[str, Any] = dict(base_config)
    config.update(
        {
            "run_dir": run_dir,
            "is_resumed_run": is_resumed_run,
            "resume_checkpoint": str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None,
            "episodes_completed": episodes_completed,
            "last_episode_index": last_episode_index,
            "best_episode_return": best_episode_return,
            "best_checkpoint_metric": best_checkpoint_metric,
        }
    )
    if best_holdout_payload is not None:
        config["best_holdout_priority"] = list(best_holdout_priority or ())
        config.update(flatten_holdout_metrics(best_holdout_payload))
        config["best_holdout_eval_dir"] = best_holdout_payload.get("eval_dir")
        config["best_holdout_summary_json"] = os.path.join(run_dir, "best_holdout_summary.json")
    return config


def persist_training_state(
    run_dir: str,
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    best_holdout_payload: Optional[Dict[str, Any]],
) -> None:
    """将 history/config/best holdout 摘要持久化到 run_dir，供断点续训恢复。"""
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    with (run_path / "rl_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with (run_path / "rl_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best_summary_path = run_path / "best_holdout_summary.json"
    if best_holdout_payload is not None:
        with best_summary_path.open("w", encoding="utf-8") as f:
            json.dump(best_holdout_payload, f, indent=2, ensure_ascii=False)
    elif best_summary_path.exists():
        best_summary_path.unlink()


def run_holdout_evaluation(
    *,
    episode: int,
    run_dir: str,
    logger: logging.Logger,
    policy_net: PolicyNet,
    value_net: ValueNet,
    holdout_csv: str,
    holdout_reference_json: Optional[str],
    holdout_trim_count: int,
    holdout_topk: int,
    model_path: str,
    model_dir: str,
    config_file: str,
    target_property: str,
    optimization_mode: str,
    direction: str,
    max_depth: int,
    max_branching: int,
    open_set_budget: int,
    top_n_prefilter: int,
) -> Dict[str, Any]:
    """运行一次固定 holdout 评估，并产出 summary payload。"""
    eval_root = Path(run_dir).resolve() / "holdout_eval"
    eval_dir = eval_root / f"ep{episode:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    label = f"holdout_ep{episode:06d}"
    eval_json = eval_dir / f"{label}.json"
    command_txt = eval_dir / f"{label}_command.txt"
    policy_eval_path = eval_dir / "policy_eval.pth"
    value_eval_path = eval_dir / "value_eval.pth"

    save_model_pair(
        policy_net=policy_net,
        value_net=value_net,
        policy_path=str(policy_eval_path),
        value_path=str(value_eval_path),
    )

    cmd = [
        sys.executable,
        "-m",
        "mol_evo.scripts.batch_optimizer",
        "--input-csv",
        str(Path(holdout_csv).resolve()),
        "--output-json",
        str(eval_json),
        "--model-path",
        str(Path(model_path).resolve()),
        "--model-dir",
        str(Path(model_dir).resolve()),
        "--config-file",
        str(Path(config_file).resolve()),
        "--target-property",
        target_property,
        "--optimization-mode",
        optimization_mode,
        "--search-mode",
        "astar_demo",
        "--rl-eval",
        "--policy-path",
        str(policy_eval_path),
        "--value-path",
        str(value_eval_path),
        "--direction",
        direction,
        "--max-depth",
        str(max_depth),
        "--max-branching",
        str(max_branching),
        "--open-set-budget",
        str(open_set_budget),
        "--top-n-prefilter",
        str(top_n_prefilter),
        "--topK",
        str(holdout_topk),
    ]
    command_txt.write_text(" ".join(cmd), encoding="utf-8")

    try:
        subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
        )
    finally:
        for temp_path in (policy_eval_path, value_eval_path):
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    summary = summarize_eval_json(eval_json, direction, holdout_trim_count)
    comparison = None
    reference_json = None
    if holdout_reference_json:
        reference_json = str(Path(holdout_reference_json).resolve())
        reference_summary = summarize_eval_json(
            Path(reference_json),
            direction,
            holdout_trim_count,
        )
        comparison = summarize_against_reference(summary, reference_summary, holdout_trim_count)

    payload: Dict[str, Any] = {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "eval_json": str(eval_json.resolve()),
        "reference_json": reference_json,
        "trim_count": holdout_trim_count,
        "summary": {k: v for k, v in summary.items() if k != "per_smiles"},
        "comparison_vs_reference": comparison,
    }
    write_outputs(eval_dir, label, payload)
    logger.info(
        "Holdout ep=%d | top1_median=%.6f trimmed=%.6f win_rate=%s",
        episode,
        float(payload["summary"].get("top1_median", 0.0)),
        float(payload["summary"].get("trimmed_mean", 0.0)),
        (
            f"{float((payload.get('comparison_vs_reference') or {}).get('win_rate', 0.0)):.6f}"
            if payload.get("comparison_vs_reference") is not None else "n/a"
        ),
    )
    return payload


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
    algo: str = "reinforce",
    lr_policy: float = 1e-4,
    lr_value: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 5.0,
    clip_ratio: float = 0.2,
    gae_lambda: float = 0.95,
    update_epochs: int = 4,
    minibatch_size: int = 0,
    normalize_advantage: bool = True,
    checkpoint_every: int = 50,
    seed: int = 42,
    device_str: str = "auto",
    action_selection_mode: str = "sample",
    action_selection_temperature: float = 1.0,
    action_selection_epsilon: float = 0.0,
    holdout_csv: Optional[str] = None,
    holdout_reference_json: Optional[str] = None,
    holdout_eval_every: int = 0,
    holdout_trim_count: int = 3,
    holdout_topk: int = 20,
    resume_checkpoint: Optional[str] = None,
    resume_run_dir: Optional[str] = None,
) -> None:
    """在线 RL 训练主函数。"""

    # 随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 输出目录与日志
    run_dir, is_resumed_run = resolve_run_dir(
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        resume_run_dir=resume_run_dir,
    )
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)

    # 设备
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"使用设备: {device}")
    logger.info(
        "训练配置: algo=%s, action_selection=%s, temp=%.3f, epsilon=%.3f",
        algo,
        action_selection_mode,
        action_selection_temperature,
        action_selection_epsilon,
    )
    if is_resumed_run:
        logger.info(
            "resume 模式: run_dir=%s, resume_checkpoint=%s",
            run_dir,
            resume_checkpoint or "None",
        )
    else:
        logger.info("新建训练目录: %s", run_dir)

    resolved_holdout_eval_every = resolve_holdout_eval_every(
        holdout_csv=holdout_csv,
        holdout_eval_every=holdout_eval_every,
        checkpoint_every=checkpoint_every,
    )
    if holdout_csv:
        logger.info(
            "启用 holdout 选优: csv=%s, eval_every=%d, trim=%d, topk=%d, reference=%s",
            holdout_csv,
            resolved_holdout_eval_every,
            holdout_trim_count,
            holdout_topk,
            holdout_reference_json or "None",
        )
    else:
        logger.info("未配置 holdout，best checkpoint 将回退到 episode_return")

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

    base_config: Dict[str, Any] = {
        "input_csv": input_csv,
        "algo": algo,
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
        "value_loss_coef": value_loss_coef,
        "max_grad_norm": max_grad_norm,
        "clip_ratio": clip_ratio,
        "gae_lambda": gae_lambda,
        "update_epochs": update_epochs,
        "minibatch_size": minibatch_size,
        "normalize_advantage": normalize_advantage,
        "action_selection_mode": action_selection_mode,
        "action_selection_temperature": action_selection_temperature,
        "action_selection_epsilon": action_selection_epsilon,
        "seed": seed,
        "checkpoint_every": checkpoint_every,
        "holdout_csv": holdout_csv,
        "holdout_reference_json": holdout_reference_json,
        "holdout_eval_every": resolved_holdout_eval_every,
        "holdout_trim_count": holdout_trim_count,
        "holdout_topk": holdout_topk,
    }

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

    rl_trainer = build_rl_trainer(
        algo=algo,
        policy_net=policy_net,
        value_net=value_net,
        direction=direction,
        lr_policy=lr_policy,
        lr_value=lr_value,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        checkpoint_every=checkpoint_every,
        action_selection_mode=action_selection_mode,
        action_selection_temperature=action_selection_temperature,
        action_selection_epsilon=action_selection_epsilon,
        clip_ratio=clip_ratio,
        gae_lambda=gae_lambda,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        normalize_advantage=normalize_advantage,
    )

    optimizer = EvolutionTreeOptimizer(
        model_path=model_path,
        model_dir=model_dir,
        config_file=config_file,
        initial_smiles_csv=None,
        target_property=target_property,
        initial_property_value=None,
        optimization_mode=optimization_mode,
    )
    setattr(optimizer, "optimization_direction", direction)

    history: List[Dict[str, Any]] = []
    best_episode_return = float("-inf")
    best_holdout_payload: Optional[Dict[str, Any]] = None
    best_holdout_priority: Optional[Tuple[float, float, float, float]] = None
    last_episode_index = 0

    policy_best_path = os.path.join(run_dir, "policy_best.pth")
    value_best_path = os.path.join(run_dir, "value_best.pth")
    policy_best_by_return_path = os.path.join(run_dir, "policy_best_by_return.pth")
    value_best_by_return_path = os.path.join(run_dir, "value_best_by_return.pth")

    if resume_checkpoint:
        resume_checkpoint = str(Path(resume_checkpoint).resolve())
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_checkpoint}")
        rl_trainer.load_checkpoint(resume_checkpoint)
        resumed_state = load_training_state(run_dir)
        history = list(resumed_state["history"])
        best_episode_return = float(resumed_state["best_episode_return"])
        best_holdout_payload = resumed_state["best_holdout_payload"]
        best_holdout_priority = resumed_state["best_holdout_priority"]
        last_episode_index = int(resumed_state["last_episode_index"])
        logger.info(
            "恢复断点成功: trainer_ep=%d, last_episode_index=%d, history=%d, best_metric=%s",
            rl_trainer.episode_count,
            last_episode_index,
            len(history),
            "holdout" if best_holdout_payload is not None else "episode_return",
        )

    start_episode = last_episode_index + 1
    if start_episode > num_episodes:
        logger.info(
            "当前断点已达到目标 episode：last_episode_index=%d, num_episodes=%d",
            last_episode_index,
            num_episodes,
        )
    else:
        logger.info(
            "开始在线 RL 训练：episode %d → %d",
            start_episode,
            num_episodes,
        )

    def snapshot_training_config() -> Dict[str, Any]:
        return build_training_config(
            base_config=base_config,
            run_dir=run_dir,
            is_resumed_run=is_resumed_run,
            resume_checkpoint=resume_checkpoint,
            episodes_completed=rl_trainer.episode_count,
            last_episode_index=last_episode_index,
            best_episode_return=best_episode_return,
            best_holdout_priority=best_holdout_priority,
            best_holdout_payload=best_holdout_payload,
        )

    for ep in range(start_episode, num_episodes + 1):
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
            last_episode_index = ep
            persist_training_state(
                run_dir=run_dir,
                history=history,
                config=snapshot_training_config(),
                best_holdout_payload=best_holdout_payload,
            )
            logger.warning(f"Episode {ep} 搜索失败 (smiles={smiles[:20]}): {exc}")
            continue

        ep_time = time.time() - ep_start
        ep_stats = rl_trainer.history[-1] if rl_trainer.history else {}
        ep_return = float(ep_stats.get("episode_return", 0.0))

        row: Dict[str, Any] = {
            "episode": ep,
            "algo": algo,
            "smiles": smiles,
            "episode_return": ep_return,
            "policy_loss": float(ep_stats.get("policy_loss", 0.0)),
            "value_loss": float(ep_stats.get("value_loss", 0.0)),
            "episode_steps": int(ep_stats.get("episode_steps", 0)),
            "action_selection_mode": ep_stats.get(
                "action_selection_mode",
                action_selection_mode,
            ),
            "time_sec": ep_time,
        }
        for metric_key in (
            "policy_entropy",
            "approx_kl",
            "clip_fraction",
            "adv_mean",
            "adv_std",
            "adv_min",
            "adv_max",
        ):
            if metric_key in ep_stats:
                row[metric_key] = float(ep_stats[metric_key])

        if ep_return > best_episode_return:
            best_episode_return = ep_return
            row["selected_as_best_by_episode_return"] = True
            save_model_pair(
                policy_net=policy_net,
                value_net=value_net,
                policy_path=policy_best_by_return_path,
                value_path=value_best_by_return_path,
            )
            if best_holdout_priority is None:
                save_model_pair(
                    policy_net=policy_net,
                    value_net=value_net,
                    policy_path=policy_best_path,
                    value_path=value_best_path,
                )
                row["best_selection_source"] = "episode_return"

        should_run_holdout_eval = (
            resolved_holdout_eval_every > 0
            and (ep % resolved_holdout_eval_every == 0 or ep == num_episodes)
        )
        if should_run_holdout_eval:
            try:
                holdout_payload = run_holdout_evaluation(
                    episode=ep,
                    run_dir=run_dir,
                    logger=logger,
                    policy_net=policy_net,
                    value_net=value_net,
                    holdout_csv=holdout_csv or "",
                    holdout_reference_json=holdout_reference_json,
                    holdout_trim_count=holdout_trim_count,
                    holdout_topk=holdout_topk,
                    model_path=model_path,
                    model_dir=model_dir,
                    config_file=config_file,
                    target_property=target_property,
                    optimization_mode=optimization_mode,
                    direction=direction,
                    max_depth=max_depth,
                    max_branching=max_branching,
                    open_set_budget=open_set_budget,
                    top_n_prefilter=top_n_prefilter,
                )
                row.update(flatten_holdout_metrics(holdout_payload))
                row["holdout_eval_dir"] = holdout_payload["eval_dir"]
                holdout_priority = build_holdout_priority(holdout_payload)
                row["holdout_priority"] = list(holdout_priority)
                if best_holdout_priority is None or holdout_priority > best_holdout_priority:
                    best_holdout_priority = holdout_priority
                    best_holdout_payload = holdout_payload
                    save_model_pair(
                        policy_net=policy_net,
                        value_net=value_net,
                        policy_path=policy_best_path,
                        value_path=value_best_path,
                    )
                    row["selected_as_best_by_holdout"] = True
                    row["best_selection_source"] = "holdout"
            except Exception as exc:
                logger.warning(f"Episode {ep} holdout 评估失败: {exc}")
                row["holdout_eval_error"] = str(exc)

        history.append(row)
        last_episode_index = ep
        persist_training_state(
            run_dir=run_dir,
            history=history,
            config=snapshot_training_config(),
            best_holdout_payload=best_holdout_payload,
        )

        if ep % 10 == 0 or ep == 1 or should_run_holdout_eval:
            message = (
                f"Episode {ep:>4d}/{num_episodes} | "
                f"algo={algo} "
                f"return={ep_return:.4f} "
                f"p_loss={row['policy_loss']:.6f} "
                f"v_loss={row['value_loss']:.6f} "
                f"steps={row['episode_steps']} "
                f"t={ep_time:.1f}s"
            )
            if "holdout_top1_median" in row:
                win_rate_text = (
                    f"{row['holdout_win_rate']:.4f}"
                    if "holdout_win_rate" in row else "n/a"
                )
                message += (
                    f" | holdout_median={row['holdout_top1_median']:.4f}"
                    f" holdout_trim={row['holdout_trimmed_mean']:.4f}"
                    f" holdout_win={win_rate_text}"
                )
            logger.info(message)

    torch.save(policy_net.state_dict(), os.path.join(run_dir, "policy_last.pth"))
    torch.save(value_net.state_dict(), os.path.join(run_dir, "value_last.pth"))

    config = snapshot_training_config()
    persist_training_state(
        run_dir=run_dir,
        history=history,
        config=config,
        best_holdout_payload=best_holdout_payload,
    )

    best_checkpoint_metric = str(config["best_checkpoint_metric"])
    logger.info(
        "训练完成！best_checkpoint_metric=%s, best_episode_return=%.4f",
        best_checkpoint_metric,
        best_episode_return,
    )
    logger.info(f"权重保存路径: {run_dir}")
    print("\n=== A* RL Demo 在线训练完成 ===")
    print(f"best_checkpoint_metric: {best_checkpoint_metric}")
    print(f"best_episode_return: {best_episode_return:.4f}")
    if best_holdout_payload is not None:
        best_holdout_metrics = flatten_holdout_metrics(best_holdout_payload)
        print(f"best_holdout_top1_median: {best_holdout_metrics['holdout_top1_median']:.6f}")
        print(f"best_holdout_trimmed_mean: {best_holdout_metrics['holdout_trimmed_mean']:.6f}")
        if "holdout_win_rate" in best_holdout_metrics:
            print(f"best_holdout_win_rate: {best_holdout_metrics['holdout_win_rate']:.6f}")
    print(f"policy_best: {policy_best_path}")
    print(f"value_best:  {value_best_path}")


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

    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"],
                        default="reinforce",
                        help="在线 RL 算法")
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="PPO GAE lambda")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="每个 rollout 的 PPO 更新轮数")
    parser.add_argument("--minibatch-size", type=int, default=0,
                        help="PPO mini-batch 大小，0 表示 full-batch")
    parser.add_argument("--normalize-advantage", dest="normalize_advantage",
                        action="store_true",
                        help="启用 advantage 标准化（默认启用）")
    parser.add_argument("--no-normalize-advantage", dest="normalize_advantage",
                        action="store_false",
                        help="关闭 advantage 标准化")
    parser.set_defaults(normalize_advantage=True)

    parser.add_argument("--action-selection-mode", type=str,
                        choices=["sample", "greedy", "epsilon_greedy"],
                        default="sample",
                        help="搜索阶段的动作选择模式")
    parser.add_argument("--action-selection-temperature", type=float, default=1.0,
                        help="sample 模式下的策略温度")
    parser.add_argument("--action-selection-epsilon", type=float, default=0.0,
                        help="epsilon-greedy 模式的 epsilon")

    parser.add_argument("--holdout-csv", type=str, default=None,
                        help="固定 holdout CSV；提供后优先按 holdout 指标选择 best checkpoint")
    parser.add_argument("--holdout-reference-json", type=str, default=None,
                        help="可选的参考评估 JSON，用于计算 win_rate_vs_bc")
    parser.add_argument("--holdout-eval-every", type=int, default=0,
                        help="每多少个 episode 跑一次 holdout；0 表示默认跟随 checkpoint_every")
    parser.add_argument("--holdout-trim-count", type=int, default=3,
                        help="holdout trimmed mean 两端裁掉的样本数")
    parser.add_argument("--holdout-topk", type=int, default=20,
                        help="holdout 评估时保存/汇总的 topK 数量")

    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="每多少个 episode 保存一次 checkpoint")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="从指定 rl_ckpt_*.pth 断点续训；会恢复 trainer/optimizer/history/best 状态")
    parser.add_argument("--resume-run-dir", type=str, default=None,
                        help="可选：显式指定 resume 使用的 run_dir；默认从 checkpoint 路径自动推断")
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
        algo=args.algo,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        clip_ratio=args.clip_ratio,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        normalize_advantage=args.normalize_advantage,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        device_str=args.device,
        action_selection_mode=args.action_selection_mode,
        action_selection_temperature=args.action_selection_temperature,
        action_selection_epsilon=args.action_selection_epsilon,
        holdout_csv=args.holdout_csv,
        holdout_reference_json=args.holdout_reference_json,
        holdout_eval_every=args.holdout_eval_every,
        holdout_trim_count=args.holdout_trim_count,
        holdout_topk=args.holdout_topk,
        resume_checkpoint=args.resume_checkpoint,
        resume_run_dir=args.resume_run_dir,
    )


if __name__ == "__main__":
    main()
