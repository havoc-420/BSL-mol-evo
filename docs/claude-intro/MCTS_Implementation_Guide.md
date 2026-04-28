# MCTS Implementation Guide for Molecular Optimization

**Last Updated**: 2026-04-28
**Module**: `mol_evo.core.molecular_evolution_expansion`
**Main File**: `/core/molecular_evolution_expansion.py` (2369 lines)

## Executive Summary

This document provides a detailed understanding of the MCTS (Monte Carlo Tree Search) implementation in the molecular optimization framework. The implementation uses MCTS combined with an OFO (single-step response prediction) model to search for optimized molecular structures.

**Key Finding**: In this implementation, **one "simulation" equals one visit to the root node**. Each simulation executes the full MCTS pipeline: selection → expansion → evaluation → backpropagation.

---

## 1. MCTS File Locations

| Component | File Path | Lines | Purpose |
|-----------|-----------|-------|---------|
| **Main MCTS impl** | `core/molecular_evolution_expansion.py` | 1126-1532 | Core `generate_expansion_tree_mcts()` method |
| **Optimizer wrapper** | `core/evolution_optimizer.py` | 728-868 | High-level API & parameter passing |
| **Unit tests** | `tests/evo-algorithm/tests/test_mcts_search_mode.py` | 1-345 | Comprehensive MCTS behavior tests |
| **Documentation** | `docs/paper/4-method-mo-mcts.md` | Full method description | Theory & algorithm |

---

## 2. Entry Point: `generate_expansion_tree_mcts()`

### Location
- **File**: `core/molecular_evolution_expansion.py:1126-1532`
- **Class**: `MolecularEvolutionExpansion`
- **Method**: `generate_expansion_tree_mcts(...)`

### Method Signature
```python
def generate_expansion_tree_mcts(
    self,
    max_depth: int = 5,
    max_branching: int = 8,
    predictor=None,                           # OFO model wrapper
    optimization_direction: str = 'increase', # 'increase' | 'decrease'
    pruning_patience: int = 3,
    initial_property_value=None,
    optimization_mode=None,
    logp_range=(0, 5),
    logp_patience: int = 3,
    # --- MCTS-specific params ---
    num_simulations: int = 200,              # ⭐ Total simulations
    exploration_weight: float = 1.4,         # ⭐ PUCT c parameter
    prior_mode: str = 'softmax',             # 'softmax' | 'uniform'
    value_mode: str = 'accumulated',         # 'accumulated' | 'zero' | 'step'
    expansion_mode: str = 'topk',            # 'topk' | 'random_topk' | 'full'
    random_seed: Optional[int] = None,
) -> Dict:
```

### Key Parameters Explained

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `num_simulations` | int | 200 | **Total MCTS simulation rounds** (key for step counting) |
| `exploration_weight` | float | 1.4 | PUCT exploration coefficient (c in UCB formula) |
| `prior_mode` | str | 'softmax' | How prior probabilities are computed from OFO scores |
| `value_mode` | str | 'accumulated' | How leaf nodes are evaluated (see section 5) |
| `expansion_mode` | str | 'topk' | How candidates are selected after OFO scoring |
| `max_branching` | int | 8 | Max children per node (branching factor B) |
| `max_depth` | int | 5 | Max search depth |

---

## 3. What Counts as a "Step" in MCTS?

### Definition: One Simulation = One Root Visit

In this implementation:

```python
for sim_idx in range(num_simulations):  # Line 1422
    # Each iteration is ONE SIMULATION (ONE STEP)
    
    # 1. Selection (traverse downward using PUCT)
    node = root
    while node.is_expanded and node.children and not node.is_terminal:
        node = _select_child(node)
    
    # 2. Expansion (first visit: generate children)
    if not node.is_terminal and not node.is_expanded:
        _expand_node(node)
    
    # 3. Evaluation + Backpropagation
    if node.children:
        eval_node = _select_child(node)
        value = _evaluate_leaf(eval_node)
        _backpropagate(eval_node, value)  # Increments visit_count all the way up
    else:
        value = _evaluate_leaf(node)
        _backpropagate(node, value)
```

**Result**: After `num_simulations=200` iterations, `root.visit_count == 200`.

### Reported Metrics (Line 1458, 1466)
```python
actual_simulations = root.visit_count  # Number of times root was visited
mcts_stats = {
    "num_simulations": num_simulations,      # Requested count
    "actual_simulations": actual_simulations, # Real count (may differ if interrupted)
    "root_visits": root.visit_count,         # Same as actual_simulations
}
```

---

## 4. MCTS Node Structure

### Internal _MCTSNode Class (Lines 1184-1223)

```python
class _MCTSNode:
    __slots__ = (
        'smiles', 'depth', 'parent', 'children',
        'visit_count',      # ⭐ How many times this node was visited
        'total_value',      # ⭐ Accumulated value from backpropagation
        'prior',            # ⭐ Prior probability P(n) from OFO
        'property_value',   # Absolute property value (y_0 + G_t)
        'accumulated_change', # Cumulative property change from root (G_t)
        'logp', 'logp_in_range',
        'operation', 'operation_params',
        'is_expanded', 'is_terminal',
    )
    
    def q_value(self):
        """Average value = total_value / visit_count"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c: float):
        """PUCT score = Q(n) + c * P(n) * sqrt(N(parent)) / (1 + N(n))"""
        if self.parent is None:
            return 0.0
        exploration = c * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration
```

### Key Visit Count Semantics
- **`root.visit_count`** = Total simulations executed (equals `num_simulations` normally)
- **`node.visit_count`** = How many times this node was selected during tree traversal
- **Tree statistics are recorded per-node**: Each node in the output JSON has:
  - `"mcts_visits"`: The node's visit count
  - `"mcts_q_value"`: Its average value (for ablation analysis)
  - `"mcts_prior"`: Prior probability assigned during expansion

---

## 5. The Four Phases of Each Simulation

### Phase 1: Selection (Lines 1428-1431)
**What**: Traverse tree downward from root using PUCT rule  
**How**: Follow the child with highest UCB score until reaching unexpanded node

```python
node = root
while node.is_expanded and node.children and not node.is_terminal:
    node = _select_child(node)  # Uses ucb_score(exploration_weight)
```

**Stops when**: Node is not yet expanded OR has no children OR is terminal

---

### Phase 2: Expansion (Lines 1433-1435, 1229-1367)
**What**: First-time visit to a leaf node → generate child candidates  
**How**:

1. Get possible operations from current molecule
2. Apply each operation → generate candidate molecules
3. **Batch predict** with OFO model: `predictor.predict_batch(from_list, to_list, ops_list)`
4. **Sort candidates** by property change (descending if `maximize`, ascending if `minimize`)
5. **Select top-B** candidates (where B = `max_branching`)
6. **Compute priors** from OFO scores using softmax (or uniform if `prior_mode='uniform'`)
7. **Create child nodes** with accumulated property changes

```python
def _expand_node(node: _MCTSNode):
    if node.is_expanded or node.is_terminal:
        return
    
    node.is_expanded = True
    
    # Check terminal conditions (max_depth, pruning_patience, logp_patience)
    if node.depth >= max_depth:
        node.is_terminal = True
        return
    
    # Generate candidates (cached per SMILES to avoid redundant expansion)
    mol = Chem.MolFromSmiles(node.smiles)
    possible_ops = self._get_possible_operations(mol)
    
    # Apply operations → batch predict
    batch_from, batch_to, batch_ops, valid_pairs = [...build lists...]
    predictions = predictor.predict_batch(batch_from, batch_to, batch_ops)
    
    # Sort and truncate
    candidates = [(op, new_smi, pred_value) for ...]
    sorted_cands = sorted(candidates, key=lambda x: x[2], reverse=(dir=='increase'))
    
    if expansion_mode == 'topk':
        selected_cands = sorted_cands[:max_branching]
    elif expansion_mode == 'random_topk':
        selected_cands = rng.sample(candidates, max_branching)
    else:  # full
        selected_cands = sorted_cands
    
    # Compute priors (softmax on OFO scores)
    if prior_mode == 'uniform':
        priors = [1.0 / len(selected_cands)] * len(selected_cands)
    else:  # softmax
        raw_scores = [c[2] for c in selected_cands]
        max_score = max(raw_scores)
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        priors = [e / sum(exp_scores) for e in exp_scores]
    
    # Create child nodes with accumulated changes
    for (op, new_smi, prop_change), prior in zip(selected_cands, priors):
        new_acc = node.accumulated_change + prop_change
        new_val = node.property_value + prop_change if node.property_value else None
        child = _MCTSNode(
            smiles=new_smi,
            depth=node.depth + 1,
            parent=node,
            prior=prior,
            property_value=new_val,
            accumulated_change=new_acc,
            ...
        )
        node.children.append(child)
```

**Caching**: Expansion results are cached by canonical SMILES to avoid re-expanding identical molecules.

---

### Phase 3: Evaluation (Lines 1437-1449, 1382-1396)
**What**: Assign a value to the leaf node  
**How**: Depends on `value_mode`:

```python
def _evaluate_leaf(node: _MCTSNode) -> float:
    if value_mode == 'zero':
        # Constant zero value
        val = 0.0
    elif value_mode == 'step':
        # Only the current step's property change
        if node.parent is None:
            val = 0.0
        else:
            val = node.accumulated_change - node.parent.accumulated_change
    else:  # 'accumulated'
        # Entire path's cumulative change
        val = node.accumulated_change
    
    # Sign adjustment for optimization direction
    if optimization_direction == 'decrease':
        val = -val
    
    return val
```

**Value Mode Effects**:
- `'accumulated'`: Favors deep paths with high cumulative gains
- `'zero'`: Pure exploration (no heuristic guidance)
- `'step'`: Only rewards single-step improvements

---

### Phase 4: Backpropagation (Lines 1398-1404)
**What**: Update all nodes from leaf up to root with the evaluation value  
**How**:

```python
def _backpropagate(node: _MCTSNode, value: float):
    cur = node
    while cur is not None:
        cur.visit_count += 1          # ⭐ Count this as one visit
        cur.total_value += value      # Accumulate the value
        cur = cur.parent              # Walk upward to root
```

**Effect**: Each node along the path gets +1 visit and accumulates the value.

---

## 6. Termination Conditions

### Hard Stops
1. **`num_simulations` reached**: Main loop terminates after exactly `num_simulations` iterations (line 1422)
2. **`interrupted` flag set**: Check at start of each iteration (line 1424)

### Soft Stops (Node Pruning)
1. **Max depth**: `node.depth >= max_depth` → mark terminal
2. **Pruning patience**: `stagnation >= pruning_patience` → consecutive non-improving steps
3. **LogP patience**: `logp_out >= logp_patience` → consecutive out-of-range logP steps

```python
# Check terminal conditions during expansion
if node.depth >= max_depth:
    node.is_terminal = True
    return

if pruning_patience > 0 and node.parent is not None:
    stagnation = 0
    cur = node
    while cur is not None and cur.parent is not None:
        change = cur.accumulated_change - cur.parent.accumulated_change
        improved = (change > 0) if optimization_direction == 'increase' else (change < 0)
        if improved:
            break
        stagnation += 1
        cur = cur.parent
    if stagnation >= pruning_patience:
        node.is_terminal = True
        return
```

**Note**: Once a node is marked `is_terminal`, it won't be expanded further, even if selected.

---

## 7. How Branching Factor & Depth Impact Step Count

### Total Possible Nodes (Without Pruning)
- **Theoretical max**: `1 + B + B² + B³ + ... + B^D` (geometric series)
- **Simplifies to**: `(B^(D+1) - 1) / (B - 1)`
- **Example**: B=3, D=2 → `1 + 3 + 9 = 13` nodes

### Impact on Simulations

Since **each simulation is one root visit**, and root is visited at least once per simulation:

- **Step count = num_simulations** (always, by definition)
- **Nodes visited per simulation** varies:
  - If tree is deep/wide → many nodes traversed per simulation
  - If tree is pruned → fewer nodes available to visit
  - Nodes revisited → only increment their visit_count, don't create new steps

### Example Trace

Given: `num_simulations=10, max_branching=2, max_depth=2`

| Sim | Path | Root Visits | New Nodes Created | Total Steps |
|-----|------|-------------|-------------------|-------------|
| 1   | R → C1 → C1.1 | 1 | {C1, C1.1} | 1 |
| 2   | R → C1 → C1.1 | 2 | {} | 1 |
| 3   | R → C1 → C1.2 | 3 | {C1.2} | 1 |
| 4   | R → C2 → ... | 4 | {C2, C2.1} | 1 |
| ... | ... | ... | ... | 1 |
| 10  | ... | 10 | ... | 1 |

**Every row is one simulation, hence 10 simulations = 10 steps**. Node creation is independent.

---

## 8. Config Files & Hyperparameter Keys

### MCTS Hyperparameters (No Dedicated Config File)

MCTS parameters are passed **directly as function arguments**, not loaded from config:

```python
# In evolution_optimizer.py: optimize_evolution_tree()
evolution_tree = evolver.generate_expansion_tree_mcts(
    num_simulations=num_simulations,
    exploration_weight=exploration_weight,
    prior_mode=mcts_prior_mode,
    value_mode=mcts_value_mode,
    expansion_mode=mcts_expansion_mode,
    random_seed=mcts_random_seed,
)
```

### Where Parameters Come From

1. **CLI arguments** (from `evolution_optimizer.py:run()`):
   ```bash
   python evolution_optimizer.py \
     --search-mode mcts \
     --num-simulations 200 \
     --exploration-weight 1.4 \
     --mcts-prior-mode softmax \
     --mcts-value-mode accumulated \
     --mcts-expansion-mode topk \
     --mcts-random-seed 42
   ```

2. **Code defaults** (in `generate_expansion_tree_mcts()` signature):
   - `num_simulations: int = 200`
   - `exploration_weight: float = 1.4`
   - `prior_mode: str = 'softmax'`
   - `value_mode: str = 'accumulated'`
   - `expansion_mode: str = 'topk'`

### Operation Config (Separate YAML)

The **operation types** and **atom types** come from a config file (used by `_get_possible_operations()`):

- **File location**: Typically `dataset/data/*.yaml` or passed via `config-file` argument
- **Keys used**:
  ```yaml
  atom_types: [C, N, O, S, ...]
  operation_types: [add_atom, replace_atom, form_double_bond, ...]
  ```

---

## 9. Output Structure (nodes/edges/mcts_stats)

### Tree JSON Output

```json
{
  "initial_smiles": "CCO",
  "max_depth": 2,
  "max_branching": 3,
  "search_mode": "mcts",
  "mcts_stats": {
    "num_simulations": 200,
    "actual_simulations": 200,
    "exploration_weight": 1.4,
    "prior_mode": "softmax",
    "value_mode": "accumulated",
    "expansion_mode": "topk",
    "random_seed": null,
    "unique_states_expanded": 42,
    "root_visits": 200
  },
  "nodes": {
    "0": {
      "id": "0",
      "smiles": "CCO",
      "depth": 0,
      "parent_id": null,
      "operation": null,
      "details": {},
      "logP": 0.84,
      "logP_in_range": true,
      "mcts_visits": 200,        # ⭐ Root was visited 200 times
      "mcts_prior": 0.0,
      "mcts_q_value": -0.15,
      "property_value": 1.5,
      "accumulated_change": 0.0
    },
    "1": {
      "id": "1",
      "smiles": "CC(C)O",
      "depth": 1,
      "parent_id": "0",
      "operation": "add_atom",
      "mcts_visits": 45,         # ⭐ This node visited 45 times (out of 200)
      "mcts_prior": 0.333,       # Prior from softmax
      "mcts_q_value": 0.025,
      "property_value": 1.65,
      "accumulated_change": 0.15,
      "property_change": 0.15
    },
    ...
  },
  "edges": [
    { "from": "0", "to": "1", "operation": "add_atom", "details": {...} },
    ...
  ]
}
```

### Key MCTS Fields

| Field | Meaning |
|-------|---------|
| `mcts_stats.num_simulations` | Requested simulation count |
| `mcts_stats.actual_simulations` | Actual count executed (may differ if interrupted) |
| `mcts_stats.unique_states_expanded` | Number of unique SMILES expanded (cache size) |
| `node.mcts_visits` | How many times this node was visited |
| `node.mcts_prior` | Prior probability assigned during expansion |
| `node.mcts_q_value` | Average value (total_value / visit_count) |

---

## 10. Common Misconceptions Clarified

### ❌ "Steps" = Nodes Created
**Correct**: Steps = Root visits = `num_simulations`. Node creation is a side effect.

### ❌ "Simulations" = Depth Traversed
**Correct**: Each simulation is one complete MCTS cycle (select → expand → eval → backprop), regardless of depth.

### ❌ "Prior Mode" Affects Step Count
**Correct**: Prior mode only changes how probabilities are computed; step count = `num_simulations` always.

### ❌ MCTS Returns Full Tree
**Correct**: Only nodes with `visit_count > 0` are included in output (line 1520).

```python
visited_children = [ch for ch in mcts_node.children if ch.visit_count > 0]
```

---

## 11. Ablation Study Modes (For Comparison)

Three configurable dimensions for analyzing MCTS behavior:

### Prior Mode
- `'softmax'`: OFO scores → softmax → prior (DEFAULT, theory-grounded)
- `'uniform'`: All children get equal prior

### Value Mode
- `'accumulated'`: Leaf value = cumulative path property change
- `'zero'`: All leaves valued at 0 (pure exploration)
- `'step'`: Leaf value = only this step's change

### Expansion Mode
- `'topk'`: Sort candidates by OFO score, take top B (DEFAULT)
- `'random_topk'`: Randomly sample B from all candidates
- `'full'`: Expand all candidates (no truncation)

Each combination can be tested to understand their impact on search quality.

---

## 12. Calling MCTS from High-Level API

### Via `EvolutionTreeOptimizer.optimize_evolution_tree()`

```python
from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer

optimizer = EvolutionTreeOptimizer(
    model_path="path/to/model.pth",
    model_dir="path/to/model/dir",
)

tree = optimizer.optimize_evolution_tree(
    initial_smiles="CCO",
    max_depth=2,
    max_branching=3,
    search_mode='mcts',          # ⭐ Enables MCTS
    num_simulations=200,         # ⭐ Step count
    exploration_weight=1.4,
    optimization_direction='decrease',
)
```

### Via Direct Call

```python
from mol_evo.core.molecular_evolution_expansion import MolecularEvolutionExpansion

evolver = MolecularEvolutionExpansion("CCO", config_file="path/to/config.yaml")
tree = evolver.generate_expansion_tree_mcts(
    max_depth=2,
    max_branching=3,
    predictor=my_predictor,
    num_simulations=200,
    ...
)
```

---

## 13. Test Examples

See `tests/evo-algorithm/tests/test_mcts_search_mode.py` for comprehensive unit tests:

- `TestMCTSTreeStructure`: Validates output JSON structure
- `TestMCTSConstraints`: Confirms max_depth, pruning behavior
- `TestMCTSAblationModes`: Tests prior/value/expansion modes
- `TestMCTSInterrupt`: Tests early stopping via interrupt flag

Example:
```python
def test_max_depth_respected(self):
    tree = evolver.generate_expansion_tree_mcts(
        max_depth=1,
        max_branching=5,
        predictor=predictor,
        num_simulations=50,
    )
    for node in tree["nodes"].values():
        assert node["depth"] <= 1  # ✓ Depth constraint enforced
```

---

## 14. Quick Reference: Key Code Locations

| What | Where | Line(s) |
|------|-------|---------|
| MCTS main loop | `molecular_evolution_expansion.py` | 1422-1456 |
| Node expansion | `molecular_evolution_expansion.py` | 1229-1367 |
| PUCT selection | `molecular_evolution_expansion.py` | 1369-1380 |
| Leaf evaluation | `molecular_evolution_expansion.py` | 1382-1396 |
| Backpropagation | `molecular_evolution_expansion.py` | 1398-1404 |
| Tree-to-JSON conversion | `molecular_evolution_expansion.py` | 1482-1525 |
| Termination conditions | `molecular_evolution_expansion.py` | 1236-1266 |
| Parameter passing | `evolution_optimizer.py` | 813-830 |
| CLI interface | `evolution_optimizer.py` | 1135-1149 |

---

## 15. Summary: Step Counting

**ONE STEP = ONE SIMULATION = ONE ROOT VISIT**

```
for sim_idx in range(num_simulations):  # num_simulations iterations
    # Each iteration: select → expand → evaluate → backprop
    # Increments root.visit_count by 1
    # May create 0 to B new nodes
    # May revisit existing nodes
    # Exactly ONE step executed
```

**After 200 simulations**:
- `root.visit_count == 200`
- `mcts_stats["actual_simulations"] == 200`
- `mcts_stats["num_simulations"] == 200` (if not interrupted)
- Tree may have 10-50 nodes (varies based on pruning, caching, revisits)

---

## Appendix: Quick Examples

### Example 1: Run MCTS with Default Settings
```bash
python -m mol_evo.core.evolution_optimizer \
  --model-path path/to/model.pth \
  --model-dir path/to/model/dir \
  --initial-smiles "CCO" \
  --search-mode mcts \
  --num-simulations 200
```

### Example 2: Run MCTS with Custom Ablations
```bash
python -m mol_evo.core.evolution_optimizer \
  --model-path path/to/model.pth \
  --model-dir path/to/model/dir \
  --initial-smiles "CCO" \
  --search-mode mcts \
  --num-simulations 500 \
  --exploration-weight 0.5 \
  --mcts-prior-mode uniform \
  --mcts-value-mode zero \
  --mcts-expansion-mode full \
  --max-depth 3 \
  --max-branching 5
```

### Example 3: Analyze Results
```python
import json

with open("evolution_tree.json") as f:
    tree = json.load(f)

stats = tree["mcts_stats"]
print(f"Simulations: {stats['num_simulations']}")
print(f"Unique states: {stats['unique_states_expanded']}")
print(f"Nodes in tree: {len(tree['nodes'])}")

root = tree["nodes"]["0"]
print(f"Root visits: {root['mcts_visits']}")
```

---

**End of Document**
