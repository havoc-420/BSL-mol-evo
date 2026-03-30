"""
pair_to_path.py

将 qm9-evo-pairs-step-N-pairs.json 格式转换为 v0.3 路径格式。

输入格式（step-N）:
    {
        "smiles_from": "C",
        "smiles_to": "C#C",
        "operations": [
            {"position": "0", "atom": "C", "operation": "add_atom"},
            {"position": "0-1", "atom": null, "operation": "form_triple_bond"}
        ]
    }

输出格式（v0.3 path）:
    {
        "path_id": "path_0",
        "nodes": ["C", "CC", "C#C"],  # 包含中间推断节点
        "edges": [...],
        "target_property": null,  # 需要外部注入
        "property_change": null,
        "valid_nodes": [true, true, true]  # 节点合法性标记
    }

核心逻辑：
1. 解析 operations 序列
2. 用 RDKit 逐步应用操作，推断中间分子
3. 构建完整路径节点序列
4. 处理非法中间节点（操作失败）
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


# ============================================================================
# 分子操作应用器
# ============================================================================

class MoleculeOperationApplier:
    """
    用 RDKit 应用单个分子操作，返回新分子的 SMILES。
    
    支持的操作类型（来自 qm9-evo 数据）：
    - replace_atom: 替换原子
    - add_atom: 添加原子
    - remove_form_triple_bond: 移除/形成三键
    - remove_form_double_bond: 移除/形成双键
    - form_triple_bond: 形成三键
    - form_double_bond: 形成双键
    - form_ring: 形成环
    - remove_form_ring: 移除环
    """
    
    @staticmethod
    def apply_operation(smiles: str, operation: Dict[str, Any]) -> Optional[str]:
        """
        应用单个操作到分子，返回新 SMILES。
        
        Args:
            smiles: 输入分子 SMILES
            operation: 操作字典 {"position", "atom", "operation", ...}
            
        Returns:
            新分子 SMILES，失败返回 None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            op_type = operation.get("operation")
            
            if op_type == "replace_atom":
                return MoleculeOperationApplier._replace_atom(mol, operation)
            elif op_type == "add_atom":
                return MoleculeOperationApplier._add_atom(mol, operation)
            elif op_type == "remove_form_triple_bond":
                return MoleculeOperationApplier._modify_bond(mol, operation, bond_type=Chem.BondType.TRIPLE, remove=True)
            elif op_type == "form_triple_bond":
                return MoleculeOperationApplier._modify_bond(mol, operation, bond_type=Chem.BondType.TRIPLE, remove=False)
            elif op_type == "remove_form_double_bond":
                return MoleculeOperationApplier._modify_bond(mol, operation, bond_type=Chem.BondType.DOUBLE, remove=True)
            elif op_type == "form_double_bond":
                return MoleculeOperationApplier._modify_bond(mol, operation, bond_type=Chem.BondType.DOUBLE, remove=False)
            elif op_type == "form_ring":
                return MoleculeOperationApplier._form_ring(mol, operation)
            elif op_type == "remove_form_ring":
                return MoleculeOperationApplier._remove_ring(mol, operation)
            else:
                logger.warning(f"未知操作类型: {op_type}")
                return None
                
        except Exception as e:
            logger.debug(f"应用操作失败 ({smiles} + {operation}): {e}")
            return None
    
    @staticmethod
    def _replace_atom(mol: Chem.Mol, operation: Dict) -> Optional[str]:
        """替换指定位置的原子类型。"""
        pos = int(operation["position"])
        new_atom = operation["atom"]
        
        # 创建可编辑分子
        emol = Chem.RWMol(mol)
        
        # 获取原子并修改
        atom = emol.GetAtomWithIdx(pos)
        new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_atom)
        atom.SetAtomicNum(new_atomic_num)
        
        # 清理并返回
        try:
            emol.UpdatePropertyCache()
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except:
            return None
    
    @staticmethod
    def _add_atom(mol: Chem.Mol, operation: Dict) -> Optional[str]:
        """在指定位置添加原子。"""
        pos = int(operation["position"])
        new_atom = operation["atom"]
        
        emol = Chem.RWMol(mol)
        
        # 添加新原子
        new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_atom)
        new_idx = emol.AddAtom(Chem.Atom(new_atomic_num))
        
        # 连接到指定位置的原子
        emol.AddBond(pos, new_idx, Chem.BondType.SINGLE)
        
        try:
            emol.UpdatePropertyCache()
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except:
            return None
    
    @staticmethod
    def _modify_bond(mol: Chem.Mol, operation: Dict, bond_type, remove: bool) -> Optional[str]:
        """修改键类型（双键/三键）。"""
        position = operation["position"]
        pos_parts = position.split("-")
        
        if len(pos_parts) != 2:
            return None
        
        pos1, pos2 = int(pos_parts[0]), int(pos_parts[1])
        
        emol = Chem.RWMol(mol)
        
        # 获取现有键
        bond = emol.GetBondBetweenAtoms(pos1, pos2)
        if bond is None:
            return None
        
        if remove:
            # 降级键类型
            current_type = bond.GetBondType()
            if current_type == Chem.BondType.TRIPLE:
                emol.RemoveBond(pos1, pos2)
                emol.AddBond(pos1, pos2, Chem.BondType.DOUBLE)
            elif current_type == Chem.BondType.DOUBLE:
                emol.RemoveBond(pos1, pos2)
                emol.AddBond(pos1, pos2, Chem.BondType.SINGLE)
        else:
            # 升级键类型
            emol.RemoveBond(pos1, pos2)
            emol.AddBond(pos1, pos2, bond_type)
        
        try:
            emol.UpdatePropertyCache()
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except:
            return None
    
    @staticmethod
    def _form_ring(mol: Chem.Mol, operation: Dict) -> Optional[str]:
        """形成环。"""
        position = operation["position"]
        pos_parts = position.split("-")
        
        if len(pos_parts) < 2:
            return None
        
        emol = Chem.RWMol(mol)
        
        # 连接首尾原子形成环
        pos1, pos2 = int(pos_parts[0]), int(pos_parts[-1])
        
        # 检查是否已有键
        if emol.GetBondBetweenAtoms(pos1, pos2) is None:
            emol.AddBond(pos1, pos2, Chem.BondType.SINGLE)
        
        try:
            emol.UpdatePropertyCache()
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except:
            return None
    
    @staticmethod
    def _remove_ring(mol: Chem.Mol, operation: Dict) -> Optional[str]:
        """移除环。"""
        position = operation["position"]
        pos_parts = position.split("-")
        
        if len(pos_parts) < 2:
            return None
        
        emol = Chem.RWMol(mol)
        
        # 移除首尾原子之间的键
        pos1, pos2 = int(pos_parts[0]), int(pos_parts[-1])
        bond = emol.GetBondBetweenAtoms(pos1, pos2)
        
        if bond is not None:
            emol.RemoveBond(pos1, pos2)
        
        try:
            emol.UpdatePropertyCache()
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except:
            return None


# ============================================================================
# 路径构建器
# ============================================================================

def build_path_from_pair(
    pair_data: Dict[str, Any],
    path_id: str,
    target_property: Optional[str] = None,
    property_change: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    从 pair + operations 数据构建完整路径。
    
    Args:
        pair_data: {"smiles_from", "smiles_to", "operations"}
        path_id: 路径唯一标识
        target_property: 目标属性名
        property_change: 属性变化值
        
    Returns:
        v0.3 格式的路径字典，失败返回 None
    """
    smiles_from = pair_data.get("smiles_from")
    smiles_to = pair_data.get("smiles_to")
    operations = pair_data.get("operations", [])
    
    if not smiles_from or not operations:
        return None
    
    # 构建节点序列
    nodes = [smiles_from]
    valid_nodes = [True]  # 起点默认有效
    
    current_smiles = smiles_from
    
    for i, op in enumerate(operations):
        # 应用操作，推断下一步分子
        next_smiles = MoleculeOperationApplier.apply_operation(current_smiles, op)
        
        if next_smiles is None:
            # 操作失败，标记为非法节点，但用占位符继续
            logger.debug(f"路径 {path_id} 步骤 {i} 操作失败，使用占位符")
            nodes.append(f"<INVALID_{i}>")
            valid_nodes.append(False)
            # 仍然用当前 SMILES 继续尝试后续操作
        else:
            nodes.append(next_smiles)
            valid_nodes.append(True)
            current_smiles = next_smiles
    
    # 检查终点是否与预期一致
    if len(nodes) > 0 and nodes[-1] != smiles_to:
        if valid_nodes[-1]:
            logger.debug(f"路径 {path_id} 推断终点 {nodes[-1]} != 预期终点 {smiles_to}")
    
    # 构建 edge 特征（从 operations 提取）
    edges = []
    for i, op in enumerate(operations):
        edge_feat = {
            "operation_type": op.get("operation", "unknown"),
            "position": op.get("position", ""),
            "atom": op.get("atom"),
            "step_index": i,
        }
        edges.append(edge_feat)
    
    return {
        "path_id": path_id,
        "nodes": nodes,
        "edges": edges,
        "target_property": target_property,
        "property_change": property_change,
        "valid_nodes": valid_nodes,
        "num_steps": len(operations),
    }


def convert_pairs_file_to_paths(
    input_file: str,
    output_file: str,
    target_property: Optional[str] = None,
    property_changes_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    min_steps: int = 1,
    max_steps: int = 10,
) -> Dict[str, int]:
    """
    将 pairs JSON 文件转换为 paths JSON 文件。
    
    Args:
        input_file: 输入 pairs JSON 文件
        output_file: 输出 paths JSON 文件
        target_property: 目标属性名（如 "lumo_change"）
        property_changes_file: 属性变化值文件（可选，格式：{"path_id": property_change}）
        max_samples: 最大处理样本数
        min_steps: 最小步骤数（过滤太短的路径）
        max_steps: 最大步骤数（过滤太长的路径）
        
    Returns:
        统计信息字典
    """
    logger.info(f"开始转换: {input_file} -> {output_file}")
    
    # 加载属性变化值（如果提供）
    property_changes = {}
    if property_changes_file and Path(property_changes_file).exists():
        with open(property_changes_file, 'r') as f:
            property_changes = json.load(f)
    
    # 加载输入数据
    with open(input_file, 'r') as f:
        pairs = json.load(f)
    
    if max_samples:
        pairs = pairs[:max_samples]
    
    paths = []
    stats = {
        "total_pairs": len(pairs),
        "converted_paths": 0,
        "invalid_start": 0,
        "too_few_steps": 0,
        "too_many_steps": 0,
        "intermediate_invalid": 0,
    }
    
    for idx, pair in enumerate(pairs):
        operations = pair.get("operations", [])
        num_steps = len(operations)
        
        # 过滤步骤数
        if num_steps < min_steps:
            stats["too_few_steps"] += 1
            continue
        if num_steps > max_steps:
            stats["too_many_steps"] += 1
            continue
        
        # 检查起点是否合法
        smiles_from = pair.get("smiles_from")
        if not smiles_from or Chem.MolFromSmiles(smiles_from) is None:
            stats["invalid_start"] += 1
            continue
        
        # 构建路径
        path_id = f"path_{idx}"
        prop_change = property_changes.get(path_id)
        
        path = build_path_from_pair(
            pair,
            path_id=path_id,
            target_property=target_property,
            property_change=prop_change,
        )
        
        if path:
            # 统计非法中间节点
            if not all(path["valid_nodes"]):
                stats["intermediate_invalid"] += 1
            
            paths.append(path)
            stats["converted_paths"] += 1
    
    # 保存输出
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(paths, f, indent=2)
    
    logger.info(f"转换完成: {stats}")
    return stats


def convert_step_files_to_path_dataset(
    step_files: Dict[int, str],  # {step_count: file_path}
    output_file: str,
    target_property: Optional[str] = None,
    property_changes_file: Optional[str] = None,
    max_total_samples: Optional[int] = None,
    balance_steps: bool = True,
) -> Dict[str, Any]:
    """
    将多个 step-N 文件合并转换为单个路径数据集。
    
    Args:
        step_files: {步骤数: 文件路径} 映射
            例如: {1: "step-1-pairs.json", 2: "step-2-pairs.json"}
        output_file: 输出文件路径
        target_property: 目标属性名
        property_changes_file: 属性变化值文件
        max_total_samples: 最大总样本数
        balance_steps: 是否平衡不同步骤数的样本
        
    Returns:
        统计信息
    """
    all_paths = []
    stats = {"per_step": {}}
    
    # 计算每个 step 文件应该抽取的样本数
    if balance_steps and max_total_samples:
        samples_per_step = max_total_samples // len(step_files)
    else:
        samples_per_step = None
    
    for step_count, file_path in step_files.items():
        logger.info(f"处理 step-{step_count} 文件: {file_path}")
        
        step_stats = convert_pairs_file_to_paths(
            input_file=file_path,
            output_file="/tmp/temp_paths.json",  # 临时文件
            target_property=target_property,
            property_changes_file=property_changes_file,
            max_samples=samples_per_step,
            min_steps=step_count,
            max_steps=step_count,
        )
        
        # 加载临时文件并合并
        with open("/tmp/temp_paths.json", 'r') as f:
            step_paths = json.load(f)
        
        all_paths.extend(step_paths)
        stats["per_step"][step_count] = step_stats
    
    # 随机打乱
    import random
    random.shuffle(all_paths)
    
    # 截断到最大样本数
    if max_total_samples:
        all_paths = all_paths[:max_total_samples]
    
    # 保存最终文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_paths, f, indent=2)
    
    stats["total_paths"] = len(all_paths)
    logger.info(f"合并完成，总路径数: {stats['total_paths']}")
    
    return stats


# ============================================================================
# CLI 接口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 qm9-evo pairs 格式转换为 v0.3 paths 格式")
    parser.add_argument("-i", "--input", required=True, help="输入 pairs JSON 文件")
    parser.add_argument("-o", "--output", required=True, help="输出 paths JSON 文件")
    parser.add_argument("-p", "--property", default=None, help="目标属性名")
    parser.add_argument("--property-changes", default=None, help="属性变化值文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最大处理样本数")
    parser.add_argument("--min-steps", type=int, default=1, help="最小步骤数")
    parser.add_argument("--max-steps", type=int, default=10, help="最大步骤数")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    stats = convert_pairs_file_to_paths(
        input_file=args.input,
        output_file=args.output,
        target_property=args.property,
        property_changes_file=args.property_changes,
        max_samples=args.max_samples,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )
    
    print(f"\n转换统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
