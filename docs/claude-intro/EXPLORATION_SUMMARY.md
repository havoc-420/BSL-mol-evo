# MCTS Implementation Exploration - Executive Summary

**Date**: 2026-04-28  
**Explorer**: Claude  
**Status**: ✅ Complete

---

## What Was Requested

Explore the codebase to understand the MCTS (Monte Carlo Tree Search) implementation, focusing on:
1. ✅ Step/simulation counting methodology
2. ✅ Definition of "one step" in the search
3. ✅ Branching factor and depth interaction
4. ✅ Termination logic
5. ✅ Config file locations and parameters

---

## Key Findings

### 1. **ONE STEP = ONE SIMULATION = ONE ROOT VISIT**

In the main loop (line 1422 of `molecular_evolution_expansion.py`):
```python
for sim_idx in range(num_simulations):  # e.g., 200 iterations
    # Each iteration executes ONE complete MCTS cycle:
    # 1. Selection (traverse down using PUCT)
    # 2. Expansion (generate children on first visit)
    # 3. Evaluation (assign value to leaf)
    # 4. Backpropagation (update all nodes up to root)
    
    # Result: root.visit_count += 1
```

**Concrete Example**:
- `num_simulations=200` → exactly 200 iterations
- Each iteration → root visited once → `root.visit_count` incremented by 1
- After all iterations → `root.visit_count == 200`
- Output JSON reports: `"actual_simulations": 200` (line 1458, 1466)

---

### 2. **MCTS File Paths**

| Component | Location | Size | Purpose |
|-----------|----------|------|---------|
| **Main MCTS** | `core/molecular_evolution_expansion.py:1126-1532` | 407 lines | Core algorithm |
| **Wrapper API** | `core/evolution_optimizer.py:728-868` | 140 lines | Parameter passing |
| **Unit Tests** | `tests/evo-algorithm/tests/test_mcts_search_mode.py` | 345 lines | Validation |
| **Theory Doc** | `docs/paper/4-method-mo-mcts.md` | Full file | Mathematical foundation |

---

### 3. **Step Counting Logic**

**The implementation counts:**

✅ **One Root Visit** = One step  
❌ ~~Nodes Created~~ (orthogonal to steps)  
❌ ~~Depth Traversed~~ (varies per simulation)  
❌ ~~Leaf Evaluations~~ (typically 1-2 per simulation)  

**Why?** The root visit counter is the primary metric for resource consumption:
- Exactly one per MCTS iteration
- Deterministic and predictable
- Used to report actual simulations completed (line 1458)

---

### 4. **Termination Logic**

#### Hard Stops
1. **Main loop bound** (line 1422): `range(num_simulations)`
   - Loop exits after exactly `num_simulations` iterations
   - If `num_simulations=200`, loop runs 0-199, then stops

2. **Interrupt flag** (line 1424-1426):
   - Checked at start of each iteration
   - If `getattr(self, 'interrupted', False)`, breaks early
   - Reports: "收到中断信号，在第 {sim_idx+1} 轮停止" (interrupted at round X)

#### Soft Stops (Node Pruning)
Nodes are marked `is_terminal` during expansion (lines 1236-1266):

1. **Depth limit** (line 1236-1238):
   ```python
   if node.depth >= max_depth:
       node.is_terminal = True
       return
   ```

2. **Pruning patience** (line 1241-1253):
   - Counts consecutive non-improving steps from root to current node
   - If `stagnation >= pruning_patience`, marks terminal
   - Example: `pruning_patience=3` → prune after 3 steps without improvement

3. **LogP patience** (line 1256-1266):
   - Counts consecutive out-of-range logP values
   - If `logp_out >= logp_patience`, marks terminal

**Effect**: Terminal nodes are skipped during selection; they don't get expanded again.

---

### 5. **Four Phases Per Simulation**

#### Phase 1: Selection (Lines 1428-1431)
- Traverse downward from root using PUCT rule
- Stop at first unexpanded or terminal node

#### Phase 2: Expansion (Lines 1433-1435, 1229-1367)
- Generate child candidates from current molecule
- Batch predict with OFO model
- Sort by property change, keep top-B (max_branching)
- Compute priors via softmax
- Create child nodes

#### Phase 3: Evaluation (Lines 1437-1449, 1382-1396)
- Assign value to leaf based on `value_mode`:
  - `'accumulated'`: cumulative change from root
  - `'zero'`: always 0
  - `'step'`: only this node's change
- Sign-flip if `optimization_direction='decrease'`

#### Phase 4: Backpropagation (Lines 1398-1404)
- Update all nodes from leaf to root
- `node.visit_count += 1`
- `node.total_value += value`

---

### 6. **Branching Factor & Depth Impact**

#### Theoretical Max Nodes
Formula: `1 + B + B² + ... + B^D = (B^(D+1) - 1) / (B - 1)`

Example: `B=3, D=2` → `1 + 3 + 9 = 13` nodes

#### Interaction with Step Count
- **Step count = num_simulations** (always constant)
- **Nodes visited per simulation** = depth of tree traversal (varies)
- **Total nodes in output** = affected by revisits, pruning, caching

Example trace with `num_simulations=10, B=2, D=2`:
| Sim | Tree Depth | New Nodes | Root Visits | Steps |
|-----|-----------|-----------|-------------|-------|
| 1-3 | 0→2 | 5 | 1,2,3 | 1,1,1 |
| 4-5 | 1→2 | 2 | 4,5 | 1,1 |
| 6-10 | Revisit | 0 | 6-10 | 1×5 |
| **Total** | - | 7 | 10 | 10 |

**Key**: Each row = 1 simulation = 1 step, regardless of depth or node creation.

---

### 7. **Configuration Parameters**

#### MCTS Hyperparameters
No dedicated config file! Parameters are passed as CLI arguments or function parameters:

```bash
# CLI arguments (from evolution_optimizer.py run())
--search-mode mcts
--num-simulations 200                    # ← Step count
--exploration-weight 1.4                 # PUCT c parameter
--mcts-prior-mode softmax|uniform        # Prior construction
--mcts-value-mode accumulated|zero|step  # Leaf valuation
--mcts-expansion-mode topk|random_topk|full  # Candidate selection
--mcts-random-seed 42                    # RNG seed (for random_topk)
```

#### Defaults (in function signature, line 1138-1143)
- `num_simulations: int = 200`
- `exploration_weight: float = 1.4`
- `prior_mode: str = 'softmax'`
- `value_mode: str = 'accumulated'`
- `expansion_mode: str = 'topk'`
- `random_seed: Optional[int] = None`

#### Operation Config (Separate YAML)
- **File**: `dataset/data/*.yaml` or via `--config-file`
- **Keys**:
  ```yaml
  atom_types: [C, N, O, S, ...]
  operation_types: [add_atom, replace_atom, form_double_bond, ...]
  ```

---

### 8. **Output Structure**

JSON output includes MCTS-specific stats (lines 1464-1474):

```json
{
  "search_mode": "mcts",
  "mcts_stats": {
    "num_simulations": 200,        // Requested
    "actual_simulations": 200,     // Executed (may differ if interrupted)
    "exploration_weight": 1.4,
    "prior_mode": "softmax",
    "value_mode": "accumulated",
    "expansion_mode": "topk",
    "random_seed": null,
    "unique_states_expanded": 42,  // Unique SMILES explored
    "root_visits": 200             // == actual_simulations
  },
  "nodes": {
    "0": {
      "mcts_visits": 200,          // Root was visited 200 times
      "mcts_prior": 0.0,
      "mcts_q_value": -0.15,       // Average value
      ...
    },
    "1": {
      "mcts_visits": 45,           // This node visited 45 times
      "mcts_prior": 0.333,
      "mcts_q_value": 0.025,
      ...
    }
  }
}
```

---

## Common Misconceptions (Clarified)

| Misconception | Reality |
|---------------|---------|
| "Steps = Nodes created" | Steps = Root visits = num_simulations |
| "Simulations = Depth traversed" | Each simulation is 1 cycle regardless of depth |
| "Prior mode affects step count" | Prior only changes probability computation; steps = num_simulations always |
| "MCTS returns full tree" | Only nodes with visit_count > 0 are included (line 1520) |
| "Branching factor = steps" | Branching affects tree width/depth; steps = num_simulations always |

---

## Quick Reference

### To count steps:
```python
# Look at the output JSON
actual_steps = mcts_stats["actual_simulations"]
# or
actual_steps = nodes["0"]["mcts_visits"]
```

### To understand node visitation:
```python
# Root: visited once per simulation
root_visits = nodes["0"]["mcts_visits"]  # Should equal num_simulations

# Each other node: visited some number of times (< root)
node_visits = nodes["X"]["mcts_visits"]  # Subset of simulations that traversed this node
```

### To predict max nodes:
```python
max_branching = B
max_depth = D
theoretical_max = (B**(D+1) - 1) // (B - 1)
# Actual will be less due to pruning and revisits
```

---

## Files Created

✅ `/docs/claude-intro/MCTS_Implementation_Guide.md` (Comprehensive, 500+ lines)  
✅ `/docs/claude-intro/EXPLORATION_SUMMARY.md` (This file)

---

## Conclusion

The MCTS implementation clearly defines "steps" as root visits in a well-structured simulation loop. Each of the 200 (or N) simulations executes one complete MCTS cycle, incrementing the root visit counter. Node creation, tree depth, and branching are orthogonal concerns that don't affect the step count directly—they determine how efficiently the search explores the space within those fixed steps.

---

**End of Summary**
