"""
A* RL Demo package for molecular optimization.

Provides a two-phase RL-augmented search mode (astar_demo) on top of the
existing BFS/MCTS OFO search framework.

Phase 1: Offline Behavioral Cloning (BC) pre-training of PolicyNet/ValueNet
         using trajectories from existing BFS/MCTS runs.
Phase 2: Online REINFORCE/PPO fine-tuning during astar_demo search episodes.
"""

from mol_evo.core.models.astar_rl.policy_network import PolicyNet
from mol_evo.core.models.astar_rl.value_network import ValueNet
from mol_evo.core.models.astar_rl.reward import (
    RewardConfig,
    compute_step_reward,
    compute_episode_reward,
)
from mol_evo.core.models.astar_rl.rl_trainer import RLTrainer

__all__ = [
    "PolicyNet",
    "ValueNet",
    "RewardConfig",
    "compute_step_reward",
    "compute_episode_reward",
    "RLTrainer",
]
