# A* RL Demo — 测试指南
<!-- last-updated: 2026-04-09 -->

## 测试文件一览

```
mol_evo/tests/
├── test_rl_training.py            # RLTrainer / BCDataset / RewardConfig 单元测试
└── test_astar_demo_search_mode.py # generate_expansion_tree_astar_demo() 集成测试
```

> 这两个测试文件**不依赖 OFO 模型权重**，也不依赖 e3nn / torch_geometric，
> 只需要 `torch` + `rdkit` 即可运行。

---

## 运行方式

### pytest（推荐）

```bash
# 安装 pytest（若环境中没有）
pip install pytest

# 运行全部 RL 测试
python -m pytest mol_evo/tests/test_rl_training.py \
                 mol_evo/tests/test_astar_demo_search_mode.py -v

# 只运行某一个测试类
python -m pytest mol_evo/tests/test_rl_training.py::TestRLTrainer -v

# 只运行某一个 case
python -m pytest mol_evo/tests/test_rl_training.py::TestRLTrainer::test_checkpoint_save_and_load -v
```

### 直接 python 运行

```bash
python mol_evo/tests/test_rl_training.py
python mol_evo/tests/test_astar_demo_search_mode.py
```

---

## test_rl_training.py — 测试内容

### TestRLTrainer（5 个 case）

| 测试名 | 验证点 |
|--------|--------|
| `test_collect_and_end_episode` | collect_step × 5 + end_episode 返回完整 stats 字典（含 policy_loss / value_loss / episode_return / episode_steps） |
| `test_episode_counter_increments` | 3 次 end_episode 后 `episode_count == 3` |
| `test_empty_episode_returns_empty` | 空 episode（未 collect_step）返回 `{}` 不报错 |
| `test_checkpoint_save_and_load` | save_checkpoint + load_checkpoint 恢复 episode_count / total_steps |
| `test_policy_loss_is_finite` | 更新后 policy_loss / value_loss 均为有限值（非 NaN/Inf） |

### TestBCDataset（3 个 case）

| 测试名 | 验证点 |
|--------|--------|
| `test_dataset_len` | `len(BCDataset(samples))` 与样本数一致 |
| `test_item_shapes` | `__getitem__` 返回 state(64,) / action(16,) / next_state(64,) / reward() / future_best_gain() |
| `test_dataloader_batch` | DataLoader batch 形状 `(4, 64)` / `(4, 16)` 正确 |

### TestRewardFunctions（4 个 case）

| 测试名 | 验证点 |
|--------|--------|
| `test_step_reward_decrease_improvement` | `direction=decrease` + `property_change=-0.5` → 正奖励 |
| `test_step_reward_logp_penalty` | `logp_in_range=False` 时奖励低于 `True` 时 |
| `test_discounted_returns_length` | `compute_discounted_returns` 输出长度 == 输入长度 |
| `test_normalize_returns` | `normalize_returns` 输出均值 ≈ 0 |

---

## test_astar_demo_search_mode.py — 测试内容

使用 `MockPredictor`（返回随机 float 列表）替代真实 OFO 模型，避免加载大型权重：

```python
class MockPredictor:
    def predict_batch(self, from_smiles_list, to_smiles_list, operations_list):
        import random
        return [random.uniform(-0.5, 0.5) for _ in from_smiles_list]
```

### TestAstarDemoSearchMode（6 个 case）

| 测试名 | 验证点 |
|--------|--------|
| `test_basic_tree_structure` | 返回 dict 含 `nodes / edges / initial_smiles`，nodes 非空 |
| `test_astar_stats_present` | `astar_stats` 含 5 个预期字段（expanded_nodes / open_set_peak / policy_prefilter_size / ofo_scored_candidates / actual_expansions） |
| `test_node_astar_fields` | 非根节点含 `g_score / h_score / f_score / policy_score` |
| `test_with_policy_and_value_net` | 接入随机初始化 PolicyNet / ValueNet 后搜索不崩溃，stats 字段存在 |
| `test_edge_references_valid_nodes` | 所有边的 `from` / `to` 均指向 `nodes` 中已存在的节点 |
| `test_open_set_budget_limits_expansions` | `expanded_nodes ≤ open_set_budget` |

---

## 预期输出示例

```
test_collect_and_end_episode ... ok
test_episode_counter_increments ... ok
test_empty_episode_returns_empty ... ok
test_checkpoint_save_and_load ... ok
test_policy_loss_is_finite ... ok
test_dataset_len ... ok
test_item_shapes ... ok
test_dataloader_batch ... ok
test_step_reward_decrease_improvement ... ok
test_step_reward_logp_penalty ... ok
test_discounted_returns_length ... ok
test_normalize_returns ... ok
----------------------------------------------------------------------
Ran 12 tests in X.XXXs
OK

test_basic_tree_structure ... ok
test_astar_stats_present ... ok
test_node_astar_fields ... ok
test_with_policy_and_value_net ... ok
test_edge_references_valid_nodes ... ok
test_open_set_budget_limits_expansions ... ok
----------------------------------------------------------------------
Ran 6 tests in X.XXXs
OK
```

---

## 跳过说明

两个测试文件的 `setUpClass` 均包含 try/except import 保护：

```python
@classmethod
def setUpClass(cls):
    try:
        from mol_evo.core.models.astar_rl import PolicyNet, ...
        cls._imports_ok = True
    except ImportError as e:
        cls._imports_ok = False
        cls._import_error = str(e)
```

若 `torch` 未安装或导入链路有问题，每个 case 会 `self.skipTest(...)` 而不是 ERROR，
方便在只有 CPU 环境下做 smoke check。

---

## 手动 smoke test（不依赖 pytest）

```python
# 在项目根目录下运行
import sys
sys.path.insert(0, '.')

import torch
from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RLTrainer
from mol_evo.core.models.astar_rl.reward import RewardConfig
from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM

print(f"STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}")

policy = PolicyNet(STATE_DIM, ACTION_DIM)
value  = ValueNet(STATE_DIM)
trainer = RLTrainer(policy, value, RewardConfig(direction="decrease"), device="cpu")

state   = torch.randn(STATE_DIM)
actions = torch.randn(4, ACTION_DIM)
lp = policy.get_log_probs(state, actions, 0)

from mol_evo.core.models.astar_rl.rl_trainer import TrajectoryStep
step = TrajectoryStep(
    state_tensor=state, action_tensors=actions, selected_action_idx=0,
    log_prob=lp, step_reward=0.5, next_state_tensor=torch.randn(STATE_DIM),
    value_estimate=0.1, done=False,
)
trainer.reset_episode()
trainer.collect_step(step)
stats = trainer.end_episode()
print("stats:", stats)
assert "policy_loss" in stats and "value_loss" in stats
print("Smoke test PASSED")
```
