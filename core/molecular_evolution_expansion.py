import argparse
import json
import sys
import math
import random
from collections import deque
from typing import List, Dict, Tuple, Optional, Set, Any
from rdkit import Chem
from rdkit.Chem import Crippen
from tqdm import tqdm
import yaml

# 设置RDKit日志级别，减少警告输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MolecularEvolutionExpansion:
    """多路径分子进化算法，支持生成多个不同的进化路径"""
    
    def __init__(self, initial_smiles: str = "CC", config_file: str = None):
        self.initial_smiles = initial_smiles
        self.initial_mol = Chem.MolFromSmiles(initial_smiles)
        
        # 初始化错误统计
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
        
        # 从配置文件加载原子类型和操作类型
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 加载原子类型 - 如果配置项不存在，将抛出KeyError异常
                self.common_atoms = config['atom_types']
                
                # 加载操作类型键列表 - 如果配置项不存在，将抛出KeyError异常
                operation_type_keys = config['operation_types']
                
                # 为操作类型创建带中文描述的字典
                operation_type_descriptions = {
                    "add_atom": "添加原子",
                    "replace_atom": "替换原子",
                    "form_double_bond": "形成双键",
                    "form_triple_bond": "形成三键",
                    "break_bond": "断开键",
                    "add_stereo": "添加立体化学",  # UPDATE 更新
                    "add_stereo_ccw": "添加R构型立体化学",
                    "add_stereo_cw": "添加S构型立体化学",
                    "form_ring": "成环",
                    "form_double_ring": "形成双键环",
                    "form_triple_ring": "形成三键环",
                    "form_aromatic_ring": "形成芳香环",
                    "remove_add_stereo": "移除立体化学添加操作",
                    "remove_form_double_bond": "对形成双键的反向操作",
                    "remove_form_triple_bond": "对形成三键的反向操作",
                    "remove_form_ring": "对成环的反向操作",
                    "remove_form_double_ring": "对形成双键环的反向操作",
                    "remove_form_triple_ring": "对形成三键环的反向操作",
                    "remove_form_aromatic_ring": "对形成芳香环的反向操作"
                }
                
                # 根据配置文件中的操作类型键创建实际使用的操作类型字典
                self.operation_types = {}
                for op_type in operation_type_keys:
                    if op_type in operation_type_descriptions:
                        self.operation_types[op_type] = operation_type_descriptions[op_type]
                
                # 保存操作类型键列表，用于在其他方法中确定启用哪些操作
                self.operation_type_keys = operation_type_keys
                
            except Exception as e:
                print(f"[ERROR] 加载配置文件失败: {e}, 使用默认配置")
                # 如果配置文件加载失败，使用默认值
                self._load_default_config()
        else:
            # 没有提供配置文件时使用默认值
            self._load_default_config()
            print("[WARNING] 没有提供配置文件，使用默认配置")
            
        # 与数据集保持一致的操作类型映射，移除 add_fragment
        self.reverse_operations = {
            "replace_atom": "replace_atom",
            "add_atom": "remove_atom",
            "form_double_bond": "remove_form_double_bond",
            "form_triple_bond": "remove_form_triple_bond",
            "form_ring": "remove_form_ring",
            "form_double_ring": "remove_form_double_ring",
            "form_triple_ring": "remove_form_triple_ring",
            "form_aromatic_ring": "remove_form_aromatic_ring",
            "add_stereo": "remove_add_stereo",
            "add_stereo_ccw": "remove_add_stereo",
            "add_stereo_cw": "remove_add_stereo"
        }
        
        # 输出当前可用的原子和操作类型
        print("[INFO] Available atoms:", self.common_atoms)
        print("[INFO] Available operations:")
        for op_key, op_desc in self.operation_types.items():
            print(f"  {op_key}: {op_desc}")
    
    def calculate_logP(self, smiles: str) -> Optional[float]:
        """计算分子的logP值
        
        Args:
            smiles: 分子的SMILES字符串
            
        Returns:
            logP值，如果计算失败则返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Crippen.MolLogP(mol)
        except Exception as e:
            print(f"计算logP时出错 ({smiles}): {e}")
            return None
    
    def validate_molecule(self, mol: Chem.Mol) -> bool:
        """验证分子是否有效"""
        self.error_stats["total_attempts"] += 1
        try:
            if mol is None:
                self.error_stats["other_errors"] += 1
                return False
            
            # 检查分子是否可以被Kekulize（芳香性检查）
            temp_mol = Chem.Mol(mol)
            Chem.SanitizeMol(temp_mol)
            
            # 检查是否有原子
            if temp_mol.GetNumAtoms() == 0:
                self.error_stats["other_errors"] += 1
                return False
                
            # 检查分子是否有多个碎片（例如断开环后产生的多部分分子如CC.N）  # UPDATE 现在 dataset 中并不包含跳过 碎片分子 的进化案例。
            # 通过将分子转换为SMILES并检查是否包含点号来判断
            smiles = Chem.MolToSmiles(temp_mol)
            if '.' in smiles:
                self.error_stats["other_errors"] += 1
                return False
                
            # 尝试计算分子量来确保分子完整性
            Chem.rdMolDescriptors.CalcExactMolWt(temp_mol)
                
            return True
        except Exception as e:
            # 记录特定类型的错误
            error_msg = str(e).lower()
            if "explicit valence" in error_msg:
                self.error_stats["valence_errors"] += 1
            elif "kekulize" in error_msg or "aromatic" in error_msg:
                self.error_stats["kekulization_errors"] += 1
            else:
                self.error_stats["other_errors"] += 1
            return False
    
    def _get_all_possible_operations(self, mol: Chem.Mol) -> List[Dict]:
        """
        获取所有可能的操作，而不是基于权重的随机选择
        实现全量扩展树生成
        """
        operations = []
        
        # 添加原子操作 - 对每个原子都尝试添加新原子
        if "add_atom" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                for element in self.common_atoms:
                    operations.append({
                        "type": "add_atom",
                        "atom_symbol": element,
                        "atom_idx": atom_idx
                    })
        
        # 替换原子操作 - 对每个原子尝试替换为其他元素
        if "replace_atom" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                current_symbol = atom.GetSymbol()
                for element in self.common_atoms:
                    if element != current_symbol:
                        operations.append({
                            "type": "replace_atom",
                            "atom_symbol": element,
                            "atom_idx": atom_idx,
                        })
        
        # 成键操作 - 尝试在合适的原子间形成各种类型的键
        atoms = list(mol.GetAtoms())
        if "form_double_bond" in self.operation_type_keys:
            for i in range(len(atoms)):
                for j in range(i+1, len(atoms)):
                    atom_idx = atoms[i].GetIdx()   # 为方便解析，这里把 atom_idx 命名为 atom_idx
                    atom2_idx = atoms[j].GetIdx()
                    
                    # 检查是否已经存在键
                    bond = mol.GetBondBetweenAtoms(atom_idx, atom2_idx)
                    if bond is None:
                        # 可以形成双键
                        operations.append({
                            "type": "form_double_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        # 单键可以升级为双键
                        operations.append({
                            "type": "form_double_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
        
        if "form_triple_bond" in self.operation_type_keys:
            for i in range(len(atoms)):
                for j in range(i+1, len(atoms)):
                    atom_idx = atoms[i].GetIdx()
                    atom2_idx = atoms[j].GetIdx()
                    
                    # 检查是否已经存在键
                    bond = mol.GetBondBetweenAtoms(atom_idx, atom2_idx)
                    if bond is None:
                        # 可以形成三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        # 单键可以升级为三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        # 双键可以升级为三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
        
        # 成环操作 - 尝试在合适的原子间形成环
        # 这里简化处理，只考虑形成5元环和6元环的可能性
        if "form_ring" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                # 简化处理，只添加示例操作
                operations.append({
                    "type": "form_ring",
                    "atom_idx": atom.GetIdx()
                })
        
        return operations

    def _apply_operation(self, mol: Chem.Mol, operation_type: str, **kwargs) -> Optional[Chem.Mol]:
        """应用指定的操作到分子上"""
        if operation_type == "add_atom" and "add_atom" in self.operation_type_keys:
            return self._add_atom_operation(mol, **kwargs)
        elif operation_type == "replace_atom" and "replace_atom" in self.operation_type_keys:
            return self._replace_atom_operation(mol, **kwargs)
        elif operation_type == "form_double_bond" and "form_double_bond" in self.operation_type_keys:
            return self._form_double_bond_operation(mol, **kwargs)
        elif operation_type == "form_triple_bond" and "form_triple_bond" in self.operation_type_keys:
            return self._form_triple_bond_operation(mol, **kwargs)
        # 注释掉断开键操作，因为这个操作不太可控
        # elif operation_type == "break_bond":
        #     return self._break_bond_operation(mol, **kwargs)
        elif operation_type == "add_stereo" and "add_stereo" in self.operation_type_keys:
            return self._add_stereo_operation(mol, **kwargs)
        elif (operation_type == "add_stereo_ccw" or operation_type == "add_stereo_cw") and "add_stereo" in self.operation_type_keys:
            # 从操作类型中提取立体构型信息
            stereo = "R" if operation_type.endswith("ccw") else "S"
            return self._add_stereo_operation(mol, stereo=stereo, **kwargs)
        elif operation_type == "form_ring" and "form_ring" in self.operation_type_keys:
            return self._form_ring_operation(mol, **kwargs)
        # 移除了 add_fragment 操作的处理
        else:
            # 其他操作保持不变
            return self._apply_other_operation(mol, operation_type, **kwargs)
    
    def _add_atom_operation(self, mol: Chem.Mol, atom_symbol: str = None, atom_idx: int = None) -> Optional[Chem.Mol]:
        """添加原子操作"""
        try:
            if atom_symbol is None:
                atom_symbol = random.choice(self.common_atoms)
            
            # 创建新原子
            new_atom = Chem.Atom(atom_symbol)
            new_mol = Chem.RWMol(mol)
            new_idx = new_mol.AddAtom(new_atom)
            
            # 如果指定了连接位置，则创建键
            if atom_idx is not None and atom_idx < new_mol.GetNumAtoms() - 1:
                new_mol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
            
            # 更新分子 - 内联 _sanitize_mol 方法的实现
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.Mol(new_mol)
            except:
                return None
            
        except Exception as e:
            self._update_error_stats(e)
            return None

    def _apply_other_operation(self, mol: Chem.Mol, operation_type: str, **kwargs) -> Optional[Chem.Mol]:
        """应用其他操作"""
        try:
            if operation_type.startswith("remove_") and operation_type in self.operation_type_keys:
                # 移除类操作先执行移除，再执行后续操作
                if operation_type == "remove_add_stereo" and "remove_add_stereo" in self.operation_type_keys:
                    return mol
                elif operation_type == "remove_form_double_bond" and "remove_form_double_bond" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_triple_bond" and "remove_form_triple_bond" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_ring" and "remove_form_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_double_ring" and "remove_form_double_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_triple_ring" and "remove_form_triple_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_aromatic_ring" and "remove_form_aromatic_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _replace_atom_operation(self, mol: Chem.Mol, atom_idx: int = None, atom_symbol: str = None) -> Optional[Chem.Mol]:
        """替换原子操作"""
        try:
            if atom_idx is None:
                atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            if atom_symbol is None:
                current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                available_symbols = [a for a in self.common_atoms if a != current_symbol]
                if available_symbols:
                    atom_symbol = random.choice(available_symbols)
                else:
                    return None
            
            mol_copy = Chem.Mol(mol)
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol))
            
            if self.validate_molecule(mol_copy):
                return mol_copy
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _form_bond_operation(self, mol: Chem.Mol, operation_type: str, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成键操作"""
        try:
            # 获取可能的原子对
            possible_pairs = []
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    if not mol.GetBondBetweenAtoms(i, j):
                        possible_pairs.append((i, j))
            
            if not possible_pairs:
                return None
            
            if atom_idx is None or atom2_idx is None:
                atom_idx, atom2_idx = random.choice(possible_pairs)
            
            # 检查是否已经存在键
            if mol.GetBondBetweenAtoms(atom_idx, atom2_idx):
                return None
            
            # 映射键类型
            bond_type_map = {
                "form_single_bond": Chem.BondType.SINGLE,
                "form_double_bond": Chem.BondType.DOUBLE,
                "form_triple_bond": Chem.BondType.TRIPLE,
                "form_ring": Chem.BondType.SINGLE,
                "form_double_ring": Chem.BondType.DOUBLE,
                "form_triple_ring": Chem.BondType.TRIPLE,
                "form_aromatic_ring": Chem.BondType.AROMATIC
            }
            
            bond_type = bond_type_map.get(operation_type, Chem.BondType.SINGLE)
            
            emol = Chem.EditableMol(mol)
            emol.AddBond(atom_idx, atom2_idx, bond_type)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _break_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """断开键操作"""
        try:
            bonds = []
            for bond in mol.GetBonds():
                bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
            if not bonds:
                return None
            
            if atom_idx is None or atom2_idx is None:
                atom_idx, atom2_idx = random.choice(bonds)
            
            if not mol.GetBondBetweenAtoms(atom_idx, atom2_idx):
                return None
            
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(atom_idx, atom2_idx)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _add_stereo_operation(self, mol: Chem.Mol, atom_idx: int = None, stereo: str = None) -> Optional[Chem.Mol]:
        """添加立体化学操作"""
        try:
            if atom_idx is None:
                atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            if stereo is None:
                stereo = random.choice(['R', 'S'])
            
            mol_copy = Chem.Mol(mol)
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW if stereo == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            if self.validate_molecule(mol_copy):
                return mol_copy
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _form_double_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成双键操作"""
        return self._form_bond_operation(mol, "form_double_bond", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _form_triple_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成三键操作"""
        return self._form_bond_operation(mol, "form_triple_bond", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _form_ring_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成环操作"""
        return self._form_bond_operation(mol, "form_ring", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _generate_single_path(self, steps: int, max_branching: int, 
                            diversity_threshold: float, seen_molecules: set) -> List[Dict]:
        """生成单条进化路径"""
        path = []
        current_mol = self.initial_mol
        current_smiles = self.initial_smiles
        
        # 记录初始状态
        if current_smiles not in seen_molecules:
            path.append({
                "step": 0,
                "smiles": current_smiles,
                "operation": "start",
                "details": {}
            })
            seen_molecules.add(current_smiles)
        
        for step in range(1, steps + 1):
            # 获取可能的操作
            possible_operations = self._get_possible_operations(current_mol)
            
            if not possible_operations:
                break
            
            # 根据多样性选择操作
            selected_operation = self._select_diverse_operation(
                possible_operations, current_mol, seen_molecules, diversity_threshold
            )
            
            if not selected_operation:
                # 如果没有找到合适的操作，尝试随机选择
                selected_operation = random.choice(possible_operations)
            
            # 应用操作
            operation_type = selected_operation["type"]
            operation_params = selected_operation.get("params", {})
            
            new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
            
            if new_mol and self.validate_molecule(new_mol):
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # 检查是否已经见过这个分子
                if new_smiles not in seen_molecules:
                    current_mol = new_mol
                    current_smiles = new_smiles
                    
                    path.append({
                        "step": step,
                        "smiles": current_smiles,
                        "operation": operation_type,
                        "details": operation_params
                    })
                    
                    seen_molecules.add(current_smiles)
                else:
                    # 如果分子已存在，跳过这一步
                    continue
            else:
                # 如果操作失败，跳过这一步
                continue
        
        return path
    
    # MARK
    def _get_possible_operations(self, mol: Chem.Mol) -> List[Dict]:
        """获取当前分子所有可能的操作（全量扩展，移除权重）
        
        支持的操作类型包括：
        - 添加操作：add_atom, replace_atom, form_double_bond, form_triple_bond, form_ring
        - 移除操作：remove_add_stereo, remove_form_double_bond, remove_form_double_ring, 
                   remove_form_ring, remove_form_triple_bond
        - 立体化学操作：add_stereo
        
        返回的操作列表与generate_path_dict方法生成的操作格式一致，支持配置文件中的所有操作类型。
        """
        operations = []
        
        # 1. 添加原子操作
        if "add_atom" in self.operation_type_keys:
            for atom_symbol in self.common_atoms:
                for atom_idx in range(mol.GetNumAtoms()):
                    operations.append({
                        "type": "add_atom",
                        "params": {"atom_symbol": atom_symbol, "atom_idx": atom_idx}
                    })
        
        # 2. 替换原子操作
        if "replace_atom" in self.operation_type_keys:
            for atom_idx in range(mol.GetNumAtoms()):
                current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                for atom_symbol in self.common_atoms:
                    if atom_symbol != current_symbol:
                        operations.append({
                            "type": "replace_atom",
                            "params": {"atom_idx": atom_idx, "atom_symbol": atom_symbol}
                        })
        
        # 3. 键操作管理 - 正向操作（形成键）
        bond_formation_map = {
            "form_double_bond": {"type": "form_double_bond", "bond_type": Chem.BondType.DOUBLE},
            "form_triple_bond": {"type": "form_triple_bond", "bond_type": Chem.BondType.TRIPLE},
            "form_ring": {"type": "form_ring", "bond_type": Chem.BondType.SINGLE},
            "form_double_ring": {"type": "form_double_ring", "bond_type": Chem.BondType.DOUBLE},
            "form_aromatic_ring": {"type": "form_aromatic_ring", "bond_type": Chem.BondType.AROMATIC}
        }
        
        # 3.1 生成形成键操作（针对非键合原子对）
        for op_key, op_info in bond_formation_map.items():
            if op_key in self.operation_type_keys:
                for i in range(mol.GetNumAtoms()):
                    for j in range(i + 1, mol.GetNumAtoms()):
                        if not mol.GetBondBetweenAtoms(i, j):
                            operations.append({
                                "type": op_info["type"],
                                "params": {"atom_idx": i, "atom2_idx": j}
                            })
        
        # 4. 键操作管理 - 反向操作（移除键）
        bond_removal_map = {
            "remove_form_double_bond": {"type": "remove_form_double_bond", "bond_type": Chem.BondType.DOUBLE},
            "remove_form_triple_bond": {"type": "remove_form_triple_bond", "bond_type": Chem.BondType.TRIPLE},
            "remove_form_ring": {"type": "remove_form_ring", "bond_type": Chem.BondType.SINGLE},
            "remove_form_double_ring": {"type": "remove_form_double_ring", "bond_type": Chem.BondType.DOUBLE}
        }
        
        # 4.1 生成移除键操作（针对已存在的特定类型键）
        for op_key, op_info in bond_removal_map.items():
            if op_key in self.operation_type_keys:
                for bond in mol.GetBonds():
                    if bond.GetBondType() == op_info["bond_type"]:
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        # 确保索引顺序一致
                        atom_idx, atom2_idx = sorted([i, j])
                        operations.append({
                            "type": op_info["type"],
                            "params": {"atom_idx": atom_idx, "atom2_idx": atom2_idx}
                        })
        
        # 5. 立体化学操作
        # 5.1 添加立体化学
        if "add_stereo" in self.operation_type_keys:
            operations.append({
                "type": "add_stereo",
                "params": {}
            })
        
        # 5.2 移除立体化学
        if "remove_add_stereo" in self.operation_type_keys:
            # 查找具有立体化学的中心
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            for center_idx, _ in chiral_centers:
                operations.append({
                    "type": "remove_add_stereo",
                    "params": {"atom_idx": center_idx}
                })
            
            # 查找具有立体化学的双键
            for bond in mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    atom_idx, atom2_idx = sorted([i, j])
                    operations.append({
                        "type": "remove_add_stereo",
                        "params": {"atom_idx": atom_idx, "atom2_idx": atom2_idx}
                    })
        
        return operations
    
    def _select_diverse_operation(self, operations: List[Dict], current_mol: Chem.Mol, 
                                seen_molecules: set, diversity_threshold: float) -> Optional[Dict]:
        """选择能够产生多样化分子的操作"""
        diverse_operations = []
        
        for operation in operations:
            # 尝试应用操作
            test_mol = self._apply_operation(current_mol, operation["type"], **operation["params"])
            if test_mol and self.validate_molecule(test_mol):
                test_smiles = Chem.MolToSmiles(test_mol)
                
                # 如果这个分子还没见过，优先选择
                if test_smiles not in seen_molecules:
                    diverse_operations.append(operation)
        
        if diverse_operations:
            # 优先选择产生新分子的操作
            return random.choice(diverse_operations)
        else:
            # 如果没有新分子，随机选择
            return random.choice(operations) if operations else None
    
    # TAG API for outside call
    def generate_multiple_paths(self, num_paths: int = 10, steps_per_path: int = 5, 
                               max_branching: int = 3, diversity_threshold: float = 0.7) -> List[Dict]:
        """
        生成多个不同的进化路径
        
        Args:
            num_paths: 要生成的路径数量
            steps_per_path: 每条路径的步数
            max_branching: 每个节点的最大分支数
            diversity_threshold: 多样性阈值，控制路径间的差异
        """
        all_paths = []
        seen_molecules = set()
        
        # 添加tqdm进度条以显示路径生成进度
        for path_idx in tqdm(range(num_paths), desc="生成进化路径", unit="条"):
            path = self._generate_single_path(
                steps_per_path, 
                max_branching, 
                diversity_threshold,
                seen_molecules
            )
            if path:
                all_paths.append({
                    "path_id": path_idx,
                    "steps": path,
                    "initial_smiles": self.initial_smiles,
                    "final_smiles": path[-1]["smiles"] if path else self.initial_smiles,
                    "path_length": len(path)
                })
        
        return all_paths
    
    def generate_evolution_tree(self, max_depth: int = 5, max_branching: int = 3) -> Dict:
        """
        生成进化树，展示所有可能的进化路径
        
        Args:
            max_depth: 最大进化深度
            max_branching: 每个节点的最大分支数
        """
        tree = {
            "root": self.initial_smiles,
            "nodes": {},
            "edges": [],
            "error_stats": {}
        }
        
        # 显式重置错误统计
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
        
        queue = deque([(self.initial_smiles, 0)])
        visited = {self.initial_smiles}
        
        while queue:
            current_smiles, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            current_mol = Chem.MolFromSmiles(current_smiles)
            if not current_mol:
                # 分子解析失败，计入其他错误
                self.error_stats["other_errors"] += 1
                self.error_stats["total_attempts"] += 1
                continue
            
            # 记录当前节点
            tree["nodes"][current_smiles] = {
                "depth": depth,
                "molecule": current_smiles
            }
            
            # 获取可能的操作
            operations = self._get_possible_operations(current_mol)
            random.shuffle(operations)
            
            branch_count = 0
            for operation in operations:
                if branch_count >= max_branching:
                    break
                
                # 应用操作并验证
                new_mol = self._apply_operation(current_mol, operation["type"], **operation["params"])
                if new_mol is not None and self.validate_molecule(new_mol):
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    if new_smiles not in visited:
                        visited.add(new_smiles)
                        queue.append((new_smiles, depth + 1))
                        
                        # 记录边（演化步骤）
                        tree["edges"].append({
                            "from": current_smiles,
                            "to": new_smiles,
                            "operation": operation["type"],
                            "details": operation["params"],
                            "depth": depth + 1
                        })
                        
                        branch_count += 1
        
        # 将完整的错误统计信息添加到树中
        tree["error_stats"] = self.error_stats.copy()
        return tree
    
    def analyze_path_diversity(self, paths: List[Dict]) -> Dict:
        """分析路径多样性"""
        if not paths:
            return {"diversity_score": 0, "unique_molecules": 0}
        
        all_molecules = set()
        for path in paths:
            for step in path["steps"]:
                all_molecules.add(step["smiles"])
        
        total_possible = len(paths) * (len(paths[0]["steps"]) if paths else 0)
        diversity_score = len(all_molecules) / total_possible if total_possible > 0 else 0
        
        return {
            "diversity_score": diversity_score,
            "unique_molecules": len(all_molecules),
            "total_paths": len(paths),
            "average_path_length": sum(len(p["steps"]) for p in paths) / len(paths) if paths else 0
        }

    def generate_accumulative_paths(self, num_paths: int = 10, max_steps: int = 3) -> List[Dict]:
        """
        生成从初始分子开始的连续演化路径
        
        Args:
            num_paths: 要生成的路径数量
            max_steps: 最大步骤数
            
        Returns:
            包含所有演化步骤的列表
        """
        all_results = []
        
        # 添加tqdm进度条以显示路径生成进度
        for path_idx in tqdm(range(num_paths), desc="生成进化路径", unit="条"):
            current_mol = self.initial_mol
            current_smiles = self.initial_smiles
            
            # 逐步执行操作，直到达到最大步骤数
            for step in range(max_steps):
                # 获取可能的操作
                possible_operations = self._get_possible_operations(current_mol)
                if not possible_operations:
                    break
                
                # 随机选择一个操作
                selected_operation = random.choice(possible_operations)
                
                # 应用操作
                operation_type = selected_operation["type"]
                operation_params = selected_operation.get("params", {})
                
                new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
                
                if new_mol and self.validate_molecule(new_mol):
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    # 创建符合示例格式的结果
                    result_entry = {
                        "smiles_from": current_smiles,
                        "smiles_to": new_smiles,
                        "operations": [{
                            "position": str(operation_params.get("atom_idx", "")) if operation_type == "add_atom" 
                                      else str(operation_params.get("atom_idx", "")) if operation_type == "replace_atom"
                                      else f"{operation_params.get('atom_idx', '')}-{operation_params.get('atom2_idx', '')}" if "bond" in operation_type
                                      else "",
                            "atom": operation_params.get("atom_symbol", operation_params.get("atom_symbol", "")),
                            "operation": operation_type,
                            "from_atom": current_smiles
                        }]
                    }
                    all_results.append(result_entry)
                    
                    # 更新当前分子
                    current_mol = new_mol
                    current_smiles = new_smiles
                else:
                    # 如果操作失败，跳过这一步
                    continue
        
        return all_results

    # MARK
    def generate_expansion_tree(self, max_depth: int = 3, max_branching: int = 5, 
                              predictor=None, optimization_direction='increase', 
                              pruning_patience=3, initial_property_value=None, 
                              optimization_mode=None, logp_range=(0, 5), logp_patience=3) -> Dict:
        """
        生成从初始分子开始的扩展树，用于分子优化迭代，支持生成过程中的预测和剪枝
        
        Args:
            max_depth: 最大扩展深度
            max_branching: 每个节点的最大分支数
            predictor: 预测器对象，用于预测属性变化
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            pruning_patience: 剪枝耐心值，连续多少代没有改善就剪枝
            initial_property_value: 初始分子的属性值
            optimization_mode: 优化模式
            logp_range: logP值的有效范围，默认(0, 5)
            logp_patience: logP剪枝耐心值，连续多少代logP超出范围就剪枝
            
        Returns:
            包含扩展树结构的字典
        """
        # 初始化扩展树
        expansion_tree = {
            "initial_smiles": self.initial_smiles,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "nodes": {},
            "edges": []
        }
        
        # 使用队列进行广度优先搜索，按层处理
        current_level = [(self.initial_smiles, 0, "0")]  # (smiles, depth, parent_id) 根节点的parent_id设为"0"
        node_counter = 0
        
        # 初始化根节点
        root_node = {
            "id": str(node_counter),
            "smiles": self.initial_smiles,
            "depth": 0,
            "parent_id": None,
            "operation": None,
            "details": {}
        }
        
        # 计算并添加根节点的logP值
        root_logp = self.calculate_logP(self.initial_smiles)
        root_node["logP"] = root_logp
        # 记录logP是否在有效范围内
        root_node["logP_in_range"] = root_logp is not None and logp_range[0] <= root_logp <= logp_range[1]
        
        # 如果提供了预测器，为根节点添加属性值
        if predictor and initial_property_value is not None:
            root_node["property_value"] = initial_property_value
            root_node["property_change"] = 0.0
            root_node["accumulated_change"] = 0.0
        
        expansion_tree["nodes"][str(node_counter)] = root_node
        node_counter += 1
        
        seen_molecules = {self.initial_smiles}  # 存储已处理的分子
        
        # 按层进行扩展
        for depth in range(max_depth):
            print(f"\n=== 开始处理第 {depth} 层，当前层节点数: {len(current_level)} ===")
            
            # 收集当前层的所有候选节点
            all_candidates = []
            
            # 处理当前层的所有节点
            for current_smiles, _, parent_id in current_level:
                # 获取当前节点
                current_node = expansion_tree["nodes"].get(parent_id)
                if not current_node:
                    continue
                
                # INFO 检查是否需要剪枝该分支
                if predictor and pruning_patience > 0:
                    # 从当前节点回溯到最近的属性改善点
                    should_prune = False
                    stagnation_count = 0
                    last_improvement_depth = depth
                    
                    # 计算该路径的连续未改善次数
                    current_backtrack_id = parent_id
                    while current_backtrack_id:
                        backtrack_node = expansion_tree["nodes"].get(current_backtrack_id)
                        if not backtrack_node:
                            break
                        
                        # 如果该节点的属性变化是改善的，重置停滞计数
                        if "property_change" in backtrack_node:
                            change = backtrack_node["property_change"]
                            if (optimization_direction == 'increase' and change > 0) or \
                               (optimization_direction == 'decrease' and change < 0):
                                last_improvement_depth = backtrack_node["depth"]
                                break
                        
                        # 向前回溯
                        current_backtrack_id = backtrack_node.get("parent_id")
                    
                    stagnation_count = depth - last_improvement_depth
                    if stagnation_count >= pruning_patience:
                        print(f"剪枝分支，起始节点: {current_smiles}，连续未改善次数: {stagnation_count}")
                        continue
                
                # INFO 检查是否需要根据logP剪枝该分支
                if logp_patience > 0:
                    # 从当前节点回溯到最近的logP在范围内的节点
                    logp_out_range_count = 0
                    last_valid_logp_depth = depth
                    
                    # 计算该路径的连续logP超出范围次数
                    current_backtrack_id = parent_id
                    while current_backtrack_id:
                        backtrack_node = expansion_tree["nodes"].get(current_backtrack_id)
                        if not backtrack_node:
                            break
                        
                        # 检查该节点的logP是否在范围内
                        if backtrack_node.get("logP_in_range", True):
                            last_valid_logp_depth = backtrack_node["depth"]
                            break
                        
                        # 向前回溯
                        current_backtrack_id = backtrack_node.get("parent_id")
                    
                    logp_out_range_count = depth - last_valid_logp_depth
                    if logp_out_range_count >= logp_patience:
                        print(f"根据logP剪枝分支，起始节点: {current_smiles}，连续超出范围次数: {logp_out_range_count}")
                        continue
                
                # 解析当前分子
                current_mol = Chem.MolFromSmiles(current_smiles)
                if not current_mol:
                    continue
                
                # 获取可能的操作
                possible_operations = self._get_possible_operations(current_mol)
                
                # 如果有预测器，先对每个可能的操作进行预测，然后根据预测结果排序
                if predictor:
                    # 收集所有需要预测的操作
                    # THINK 这里是否需要 batch-size 的设定来限制呢？ # 已放到 batch——predict 中分批 64 处理。
                    batch_from_smiles = []
                    batch_to_smiles = []
                    batch_operations = []
                    valid_operations = []
                    
                    for operation in possible_operations:
                        try:
                            # 应用操作生成新分子
                            operation_type = operation["type"]
                            operation_params = operation.get("params", {})
                            new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
                            
                            # 检查新分子是否有效
                            if new_mol and self.validate_molecule(new_mol):
                                new_smiles = Chem.MolToSmiles(new_mol)
                                
                                # 添加到批量预测列表
                                batch_from_smiles.append(current_smiles)
                                batch_to_smiles.append(new_smiles)
                                batch_operations.append(operation)
                                valid_operations.append((operation, new_smiles))
                        except Exception as e:
                            # 记录错误但继续处理其他操作
                            continue
                    # 进行批量预测
                    operations_with_predictions = []
                    if batch_from_smiles:
                        try:
                            # 调用批量预测方法
                            property_changes = predictor.predict_batch(batch_from_smiles, batch_to_smiles, batch_operations)
                            
                            # 过滤掉预测值为None的操作
                            for i, (operation, new_smiles) in enumerate(valid_operations):
                                if i < len(property_changes):
                                    property_change = property_changes[i]
                                    # 确保property_change是有效的数值类型
                                    if property_change is not None and not (isinstance(property_change, list) or isinstance(property_change, dict)):
                                        operations_with_predictions.append((operation, new_smiles, property_change, parent_id))
                        except Exception as e:
                            raise e
                            # print(f"批量预测出错: {e}")
                            # # 如果批量预测失败，回退到逐个预测
                            # for operation, new_smiles in valid_operations:
                            #     try:
                            #         property_change = predictor.predict_property_change(current_smiles, new_smiles, operation)
                            #         if property_change is not None:
                            #             operations_with_predictions.append((operation, new_smiles, property_change))
                            #     except Exception as e:
                            #         continue
                    
                    # 将所有候选添加到当前层的候选列表
                    all_candidates.extend(operations_with_predictions)
                else:
                   raise NotImplementedError("[Molecular Evolver] 预测器未定义")
            
            # 对当前层的所有候选进行统一的 top 5% 过滤
            if all_candidates:
                # 根据优化方向排序所有候选
                if optimization_direction == 'increase':
                    all_candidates.sort(key=lambda x: x[2], reverse=True)
                else:
                    all_candidates.sort(key=lambda x: x[2])
                
                # 计算前50%的候选数量，至少取1个，最多500个
                top_50_percent_count = max(1, min(500, int(len(all_candidates) * 0.5)))
                print(f"第 {depth} 层总候选数: {len(all_candidates)}，前50%候选数: {top_50_percent_count}")
                
                # 处理前50%的候选
                next_level = []
                for operation, new_smiles, property_change, parent_id in all_candidates[:top_50_percent_count]:
                    # 避免重复分子
                    if new_smiles not in seen_molecules:
                        seen_molecules.add(new_smiles)
                        
                        # 获取父节点
                        current_node = expansion_tree["nodes"].get(parent_id)
                        if not current_node:
                            continue
                        
                        # 计算累计变化和新属性值
                        current_accumulated = current_node.get("accumulated_change", 0)
                        new_accumulated = current_accumulated + property_change
                        current_value = current_node.get("property_value", 0)
                        new_value = current_value + property_change
                        
                        # 计算新分子的logP值
                        new_logp = self.calculate_logP(new_smiles)
                        # 检查logP是否在有效范围内
                        logp_in_range = new_logp is not None and logp_range[0] <= new_logp <= logp_range[1]
                        
                        # 添加新节点
                        node_id = str(node_counter)
                        expansion_tree["nodes"][node_id] = {
                            "id": node_id,
                            "smiles": new_smiles,
                            "depth": depth + 1,
                            "parent_id": parent_id,
                            "operation": operation["type"],
                            "details": operation.get("params", {}),
                            "property_change": property_change,
                            "accumulated_change": new_accumulated,
                            "property_value": new_value,
                            "logP": new_logp,
                            "logP_in_range": logp_in_range
                        }
                        
                        # 添加边
                        expansion_tree["edges"].append({
                            "from": parent_id,
                            "to": node_id,
                            "operation": operation["type"],
                            "details": operation.get("params", {})
                        })
                        
                        # 添加到下一层
                        next_level.append((new_smiles, depth + 1, node_id))
                        node_counter += 1
                
                # 更新当前层为下一层
                current_level = next_level
                print(f"第 {depth} 层扩展完成，下一层节点数: {len(current_level)}")
            else:
                print(f"第 {depth} 层没有有效候选，停止扩展")
                break
                        
        return expansion_tree

    # MARK: MCTS 搜索模式
    def generate_expansion_tree_mcts(
        self,
        max_depth: int = 5,
        max_branching: int = 8,
        predictor=None,
        optimization_direction: str = 'increase',
        pruning_patience: int = 3,
        initial_property_value=None,
        optimization_mode=None,
        logp_range=(0, 5),
        logp_patience: int = 3,
        # --- MCTS 专属参数 ---
        num_simulations: int = 200,
        step_budget: Optional[int] = None,
        exploration_weight: float = 1.4,
        prior_mode: str = 'softmax',
        value_mode: str = 'accumulated',
        expansion_mode: str = 'topk',
        random_seed: Optional[int] = None,
    ) -> Dict:
        """
        使用 MCTS (Monte Carlo Tree Search) + PUCT 生成分子进化树。

        与 BFS 版 generate_expansion_tree 共享同一输出结构 (nodes / edges)，
        以兼容下游的 save / print / topK 链路。

        Args:
            max_depth: 最大搜索深度
            max_branching: 每个状态最多展开的子节点数
            predictor: 预测器，需要提供 predict_batch()
            optimization_direction: 'increase' | 'decrease'
            pruning_patience: 连续无改善剪枝阈值
            initial_property_value: 根节点属性初始值
            optimization_mode: 优化模式
            logp_range: logP 有效范围
            logp_patience: logP 连续超出范围剪枝阈值
            num_simulations: MCTS 模拟总轮数
            step_budget: 总步数预算（节点展开次数上限）；None = 不限制
            exploration_weight: PUCT 探索系数 c
            prior_mode: prior 构造方式 (softmax | uniform)
            value_mode: 叶节点价值方式 (accumulated | zero | step)
            expansion_mode: 扩展方式 (topk | random_topk | full)
            random_seed: 随机种子（主要用于 random_topk 可复现）
        """
        if predictor is None:
            raise NotImplementedError("[MCTS] 必须提供 predictor")

        valid_prior_modes = {'softmax', 'uniform'}
        valid_value_modes = {'accumulated', 'zero', 'step'}
        valid_expansion_modes = {'topk', 'random_topk', 'full'}
        if prior_mode not in valid_prior_modes:
            raise ValueError(f"[MCTS] 不支持的 prior_mode: {prior_mode}")
        if value_mode not in valid_value_modes:
            raise ValueError(f"[MCTS] 不支持的 value_mode: {value_mode}")
        if expansion_mode not in valid_expansion_modes:
            raise ValueError(f"[MCTS] 不支持的 expansion_mode: {expansion_mode}")

        rng = random.Random(random_seed) if random_seed is not None else random

        # ---------- 内部 MCTS 节点 ----------
        class _MCTSNode:
            __slots__ = (
                'smiles', 'depth', 'parent', 'children',
                'visit_count', 'total_value', 'prior',
                'property_value', 'accumulated_change', 'logp',
                'logp_in_range', 'operation', 'operation_params',
                'is_expanded', 'is_terminal',
            )

            def __init__(self, smiles, depth, parent=None, prior=0.0,
                         property_value=None, accumulated_change=0.0,
                         operation=None, operation_params=None):
                self.smiles = smiles
                self.depth = depth
                self.parent = parent
                self.children: List['_MCTSNode'] = []
                self.visit_count = 0
                self.total_value = 0.0
                self.prior = prior
                self.property_value = property_value
                self.accumulated_change = accumulated_change
                self.logp = None
                self.logp_in_range = True
                self.operation = operation
                self.operation_params = operation_params or {}
                self.is_expanded = False
                self.is_terminal = False

            def q_value(self):
                """平均价值"""
                if self.visit_count == 0:
                    return 0.0
                return self.total_value / self.visit_count

            def ucb_score(self, c: float):
                """PUCT 得分"""
                if self.parent is None:
                    return 0.0
                exploration = c * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
                return self.q_value() + exploration

        # ---------- 扩展缓存 ----------
        # key=canonical SMILES, value=List of (operation, new_smiles, property_change)
        expansion_cache: Dict[str, List[Tuple]] = {}
        step_counter = [0]   # 用列表以便闭包内修改；每次 _expand_node 首次进入主逻辑时 +1

        def _expand_node(node: _MCTSNode):
            """首次展开一个节点：生成候选、批量预测、创建子节点。"""
            if node.is_expanded or node.is_terminal:
                return

            node.is_expanded = True
            step_counter[0] += 1   # 计步：每次节点首次展开均计入（含缓存命中）

            if node.depth >= max_depth:
                node.is_terminal = True
                return

            # --- 检查 pruning patience (property) ---
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

            # --- 检查 logP patience ---
            if logp_patience > 0 and node.parent is not None:
                logp_out = 0
                cur = node
                while cur is not None:
                    if cur.logp_in_range:
                        break
                    logp_out += 1
                    cur = cur.parent
                if logp_out >= logp_patience:
                    node.is_terminal = True
                    return

            # --- 使用缓存或新生成候选 ---
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(node.smiles))
            if canonical in expansion_cache:
                candidates = expansion_cache[canonical]
            else:
                mol = Chem.MolFromSmiles(node.smiles)
                if mol is None:
                    node.is_terminal = True
                    return

                possible_ops = self._get_possible_operations(mol)

                # 应用操作，收集有效候选
                batch_from, batch_to, batch_ops, valid_pairs = [], [], [], []
                for op in possible_ops:
                    try:
                        new_mol = self._apply_operation(mol, op["type"], **op.get("params", {}))
                        if new_mol and self.validate_molecule(new_mol):
                            new_smiles = Chem.MolToSmiles(new_mol)
                            batch_from.append(node.smiles)
                            batch_to.append(new_smiles)
                            batch_ops.append(op)
                            valid_pairs.append((op, new_smiles))
                    except Exception:
                        continue

                candidates = []
                if batch_from:
                    try:
                        predictions = predictor.predict_batch(batch_from, batch_to, batch_ops)
                        for idx, (op, new_smi) in enumerate(valid_pairs):
                            if idx < len(predictions) and predictions[idx] is not None:
                                candidates.append((op, new_smi, predictions[idx]))
                    except Exception as e:
                        print(f"[MCTS] 批量预测出错: {e}")

                expansion_cache[canonical] = candidates

            if not candidates:
                node.is_terminal = True
                return

            # --- 候选选择：排序截断 / 随机截断 / 全展开 ---
            if optimization_direction == 'increase':
                sorted_cands = sorted(candidates, key=lambda x: x[2], reverse=True)
            else:
                sorted_cands = sorted(candidates, key=lambda x: x[2])

            if expansion_mode == 'topk':
                selected_cands = sorted_cands[:max_branching]
            elif expansion_mode == 'random_topk':
                if len(candidates) <= max_branching:
                    selected_cands = list(candidates)
                else:
                    selected_cands = rng.sample(candidates, max_branching)
            else:  # full
                selected_cands = sorted_cands

            # --- 计算 prior（softmax / uniform） ---
            if prior_mode == 'uniform':
                denom = len(selected_cands) or 1
                priors = [1.0 / denom] * len(selected_cands)
            else:
                raw_scores = [c[2] for c in selected_cands]
                if optimization_direction == 'decrease':
                    raw_scores = [-s for s in raw_scores]
                max_score = max(raw_scores) if raw_scores else 0
                exp_scores = [math.exp(s - max_score) for s in raw_scores]
                sum_exp = sum(exp_scores) or 1.0
                priors = [e / sum_exp for e in exp_scores]

            # --- 创建子节点 ---
            seen_children_smiles = set()
            for (op, new_smi, prop_change), prior in zip(selected_cands, priors):
                c_smi = Chem.MolToSmiles(Chem.MolFromSmiles(new_smi))
                if c_smi in seen_children_smiles:
                    continue
                seen_children_smiles.add(c_smi)

                new_acc = node.accumulated_change + prop_change
                new_val = (node.property_value + prop_change) if node.property_value is not None else None
                new_logp = self.calculate_logP(new_smi)
                in_range = new_logp is not None and logp_range[0] <= new_logp <= logp_range[1]

                child = _MCTSNode(
                    smiles=new_smi,
                    depth=node.depth + 1,
                    parent=node,
                    prior=prior,
                    property_value=new_val,
                    accumulated_change=new_acc,
                    operation=op["type"],
                    operation_params=op.get("params", {}),
                )
                child.logp = new_logp
                child.logp_in_range = in_range
                node.children.append(child)

            if not node.children:
                node.is_terminal = True

        def _select_child(node: _MCTSNode) -> Optional[_MCTSNode]:
            """PUCT 选择最佳子节点；跳过异常分数并在必要时回退。"""
            best, best_score = None, -float('inf')
            fallback = node.children[0] if node.children else None
            for ch in node.children:
                sc = ch.ucb_score(exploration_weight)
                if not math.isfinite(sc):
                    continue
                if sc > best_score:
                    best_score = sc
                    best = ch
            return best if best is not None else fallback

        def _evaluate_leaf(node: _MCTSNode) -> float:
            """叶节点估值：支持累计值 / 零值 / 单步值三种模式。"""
            if value_mode == 'zero':
                val = 0.0
            elif value_mode == 'step':
                if node.parent is None:
                    val = 0.0
                else:
                    val = node.accumulated_change - node.parent.accumulated_change
            else:
                val = node.accumulated_change

            if optimization_direction == 'decrease':
                val = -val
            return val

        def _backpropagate(node: _MCTSNode, value: float):
            """回传价值"""
            cur = node
            while cur is not None:
                cur.visit_count += 1
                cur.total_value += value
                cur = cur.parent

        # ---------- 构造根节点 ----------
        root_logp = self.calculate_logP(self.initial_smiles)
        root = _MCTSNode(
            smiles=self.initial_smiles,
            depth=0,
            property_value=initial_property_value,
            accumulated_change=0.0,
        )
        root.logp = root_logp
        root.logp_in_range = root_logp is not None and logp_range[0] <= root_logp <= logp_range[1]

        # ---------- MCTS 主循环 ----------
        print(f"\n[MCTS] 开始搜索: simulations={num_simulations}, c={exploration_weight}, "
              f"max_depth={max_depth}, max_branching={max_branching}, "
              f"prior_mode={prior_mode}, value_mode={value_mode}, expansion_mode={expansion_mode}, random_seed={random_seed}"
              + (f", step_budget={step_budget}" if step_budget is not None else ""))

        for sim_idx in range(num_simulations):
            # 检查中断信号
            if getattr(self, 'interrupted', False):
                print(f"[MCTS] 收到中断信号，在第 {sim_idx+1} 轮停止")
                break

            # 检查步数预算
            if step_budget is not None and step_counter[0] >= step_budget:
                print(f"[MCTS] 达到 step_budget={step_budget}，在第 {sim_idx+1} 轮停止")
                break

            # 1. Selection: 从根沿 PUCT 向下
            node = root
            while node.is_expanded and node.children and not node.is_terminal:
                node = _select_child(node)

            # 2. Expansion
            if not node.is_terminal and not node.is_expanded:
                _expand_node(node)

            # 3. Evaluation + 4. Backpropagation
            # 若展开后有子节点，选一个最优子节点评估并从该子节点回传
            if node.children:
                eval_node = _select_child(node)
                if eval_node is not None:
                    value = _evaluate_leaf(eval_node)
                    _backpropagate(eval_node, value)
                else:
                    value = _evaluate_leaf(node)
                    _backpropagate(node, value)
            else:
                value = _evaluate_leaf(node)
                _backpropagate(node, value)

            # 定期日志
            if (sim_idx + 1) % max(1, num_simulations // 5) == 0:
                print(f"[MCTS] simulation {sim_idx+1}/{num_simulations}, "
                      f"steps={step_counter[0]}"
                      + (f"/{step_budget}" if step_budget is not None else "")
                      + f", root visits={root.visit_count}, "
                      f"unique states cached={len(expansion_cache)}")

        # ---------- 将 MCTS 树转换为兼容的 nodes/edges 结构 ----------
        actual_simulations = root.visit_count
        expansion_tree = {
            "initial_smiles": self.initial_smiles,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "search_mode": "mcts",
            "mcts_stats": {
                "num_simulations": num_simulations,
                "actual_simulations": actual_simulations,
                "step_budget": step_budget,
                "actual_steps": step_counter[0],
                "exploration_weight": exploration_weight,
                "prior_mode": prior_mode,
                "value_mode": value_mode,
                "expansion_mode": expansion_mode,
                "random_seed": random_seed,
                "unique_states_expanded": len(expansion_cache),
                "root_visits": root.visit_count,
            },
            "nodes": {},
            "edges": [],
        }

        node_counter = [0]
        seen_smiles_set: Set[str] = set()

        def _traverse(mcts_node: _MCTSNode, parent_tree_id: Optional[str]):
            nid = str(node_counter[0])
            node_counter[0] += 1

            tree_node = {
                "id": nid,
                "smiles": mcts_node.smiles,
                "depth": mcts_node.depth,
                "parent_id": parent_tree_id,
                "operation": mcts_node.operation,
                "details": mcts_node.operation_params,
                "logP": mcts_node.logp,
                "logP_in_range": mcts_node.logp_in_range,
                # MCTS 统计
                "mcts_visits": mcts_node.visit_count,
                "mcts_prior": round(mcts_node.prior, 6),
                "mcts_q_value": round(mcts_node.q_value(), 6),
            }
            if mcts_node.property_value is not None:
                tree_node["property_value"] = mcts_node.property_value
                tree_node["accumulated_change"] = mcts_node.accumulated_change
                if mcts_node.parent is not None:
                    tree_node["property_change"] = mcts_node.accumulated_change - mcts_node.parent.accumulated_change
                else:
                    tree_node["property_change"] = 0.0

            expansion_tree["nodes"][nid] = tree_node
            seen_smiles_set.add(mcts_node.smiles)

            if parent_tree_id is not None:
                expansion_tree["edges"].append({
                    "from": parent_tree_id,
                    "to": nid,
                    "operation": mcts_node.operation,
                    "details": mcts_node.operation_params,
                })

            # 只输出被访问过的子节点（按访问次数降序）
            visited_children = [ch for ch in mcts_node.children if ch.visit_count > 0]
            visited_children.sort(key=lambda c: c.visit_count, reverse=True)
            for child in visited_children:
                _traverse(child, nid)

        _traverse(root, None)

        total_nodes = len(expansion_tree["nodes"])
        total_edges = len(expansion_tree["edges"])
        print(f"[MCTS] 搜索完成: 树节点={total_nodes}, 边={total_edges}, "
              f"唯一状态={len(expansion_cache)}")

        return expansion_tree

    # MARK: A* RL Demo 搜索模式
    def generate_expansion_tree_astar_demo(
        self,
        max_depth: int = 4,
        max_branching: int = 8,
        predictor=None,
        optimization_direction: str = "decrease",
        pruning_patience: int = 3,
        initial_property_value=None,
        optimization_mode=None,
        logp_range=(0, 5),
        logp_patience: int = 3,
        # --- A* RL 专属参数 ---
        policy_net=None,
        value_net=None,
        rl_trainer=None,
        top_n_prefilter: int = 20,
        open_set_budget: int = 200,
    ) -> Dict:
        """使用 A* 启发式搜索 + PolicyNet/ValueNet 生成分子进化树（RL Demo 模式）。

        与 BFS/MCTS 版共享同一输出结构（nodes / edges），以兼容下游 save/print/topK 链路。
        额外在树 JSON 中追加 ``astar_stats`` 字段，用于评测对比。

        搜索流程：
          1. _get_possible_operations() → 全量合法候选操作
          2. PolicyNet.top_k_actions() → top_n_prefilter 预筛（可选）
          3. predict_batch() → OFO 批量打分（property_change）
          4. ValueNet.estimate() → 未来收益 h_score（可选）
          5. f_score = g_score + h_score + policy_bonus
          6. heapq 优先队列按 f_score 展开
          7. 收集 TrajectoryStep → RLTrainer（在线训练）或忽略（纯评估）

        Args:
            max_depth:          最大搜索深度
            max_branching:      每次从 open set 展开时最多加入树的子节点数
            predictor:          预测器，需提供 predict_batch()
            optimization_direction: 'increase' | 'decrease'
            pruning_patience:   路径连续无改善剪枝阈值（暂未使用，保留接口）
            initial_property_value: 根节点属性初始值
            optimization_mode:  优化模式（透传）
            logp_range:         logP 有效范围
            logp_patience:      logP 连续超出范围剪枝阈值（暂未使用，保留接口）
            policy_net:         PolicyNet 实例（可选，None 时跳过预筛）
            value_net:          ValueNet 实例（可选，None 时 h_score=0）
            rl_trainer:         RLTrainer 实例（可选，None 时不收集轨迹）
            top_n_prefilter:    PolicyNet 保留的候选数（policy_net 存在时生效）
            open_set_budget:    open set 总展开次数上限

        Returns:
            dict: 包含 nodes / edges / astar_stats 的扩展树字典
        """
        import heapq

        if predictor is None:
            raise NotImplementedError("[astar_demo] 必须提供 predictor")

        # ------------------------------------------------------------------
        # 懒加载 encode_state / encode_action（避免循环导入）
        # ------------------------------------------------------------------
        try:
            from mol_evo.core.data.rl_demo_processing import encode_state, encode_action
            from mol_evo.core.models.astar_rl.rl_trainer import TrajectoryStep
            _rl_imports_ok = True
        except Exception:
            _rl_imports_ok = False

        # ------------------------------------------------------------------
        # 初始化搜索树
        # ------------------------------------------------------------------
        expansion_tree = {
            "initial_smiles": self.initial_smiles,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "nodes": {},
            "edges": [],
        }

        logp_min, logp_max = logp_range
        node_counter = 0

        # 根节点
        root_logp = self.calculate_logP(self.initial_smiles)
        root_logp_in_range = (
            root_logp is not None and logp_min <= root_logp <= logp_max
        )
        root_node = {
            "id": "0",
            "smiles": self.initial_smiles,
            "depth": 0,
            "parent_id": None,
            "operation": None,
            "details": {},
            "property_change": 0.0,
            "accumulated_change": 0.0,
            "logP": root_logp,
            "logP_in_range": root_logp_in_range,
            # A* 额外字段
            "g_score": 0.0,
            "h_score": 0.0,
            "f_score": 0.0,
            "policy_score": 0.0,
            "stagnation_count": 0,
            "logp_violation_count": 0,
        }
        if initial_property_value is not None:
            root_node["property_value"] = float(initial_property_value)
        expansion_tree["nodes"]["0"] = root_node
        node_counter = 1

        # ------------------------------------------------------------------
        # open set: (neg_f_score, tie_breaker, node_id, g_score)
        # 使用负 f_score 使 heapq（最小堆）效果等同于最大优先队列
        # ------------------------------------------------------------------
        tie_counter = 0
        open_set = []
        heapq.heappush(open_set, (0.0, tie_counter, "0", 0.0))
        tie_counter += 1

        # 路径感知去重：记录 (smiles, parent_id) 对，允许同一分子在不同路径下被访问
        visited_path_keys: set = set()
        visited_path_keys.add((self.initial_smiles, "root"))

        # ------------------------------------------------------------------
        # 统计
        # ------------------------------------------------------------------
        expanded_nodes = 0
        open_set_peak = 1
        policy_prefilter_size = 0
        ofo_scored_candidates = 0
        actual_expansions = 0

        # rl_trainer episode 重置
        if rl_trainer is not None and _rl_imports_ok:
            rl_trainer.reset_episode()

        import torch

        def _get_module_device(module):
            if module is None:
                return None
            try:
                return next(module.parameters()).device
            except (StopIteration, AttributeError, TypeError):
                return torch.device("cpu")

        policy_device = _get_module_device(policy_net)
        value_device = _get_module_device(value_net)

        # ------------------------------------------------------------------
        # A* 主循环
        # ------------------------------------------------------------------
        while open_set and expanded_nodes < open_set_budget:
            neg_f, _, current_id, current_g = heapq.heappop(open_set)
            open_set_peak = max(open_set_peak, len(open_set) + 1)

            current_node = expansion_tree["nodes"].get(current_id)
            if current_node is None:
                continue

            current_depth = current_node["depth"]
            if current_depth >= max_depth:
                continue

            current_smiles = current_node["smiles"]
            current_mol = Chem.MolFromSmiles(current_smiles)
            if current_mol is None:
                continue

            expanded_nodes += 1

            # ---- 生成候选操作 ----
            possible_operations = self._get_possible_operations(current_mol)
            if not possible_operations:
                continue

            # ---- PolicyNet 预筛（可选）----
            if policy_net is not None and _rl_imports_ok:
                try:
                    state_vec = encode_state(
                        smiles=current_smiles,
                        accumulated_change=current_node.get("accumulated_change", 0.0),
                        remaining_depth=max_depth - current_depth,
                        max_depth=max_depth,
                        logp_value=current_node.get("logP") or 0.0,
                        logp_in_range=current_node.get("logP_in_range", True),
                        stagnation_count=current_node.get("stagnation_count", 0),
                        direction=optimization_direction,
                        target_property_value=initial_property_value or 0.0,
                    )
                    if policy_device is not None:
                        state_vec = state_vec.to(policy_device)
                    action_vecs = torch.stack([
                        encode_action(op, ofo_predicted_change=0.0)
                        for op in possible_operations
                    ])
                    if policy_device is not None:
                        action_vecs = action_vecs.to(policy_device)
                    k = min(top_n_prefilter, len(possible_operations))
                    top_indices, top_logits = policy_net.top_k_actions(state_vec, action_vecs, k=k)
                    top_indices = top_indices.detach().cpu().tolist()
                    top_logits = top_logits.detach().cpu().tolist()
                    prefiltered_ops = [(possible_operations[i], top_logits[j]) for j, i in enumerate(top_indices)]
                    policy_prefilter_size += len(prefiltered_ops)
                except Exception:
                    prefiltered_ops = [(op, 0.0) for op in possible_operations]
            else:
                prefiltered_ops = [(op, 0.0) for op in possible_operations]

            # ---- 生成有效 SMILES ----
            valid_candidates = []  # (operation, new_smiles, policy_logit)
            for operation, pol_logit in prefiltered_ops:
                try:
                    operation_type = operation["type"]
                    operation_params = operation.get("params", {})
                    new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
                    if new_mol and self.validate_molecule(new_mol):
                        new_smiles = Chem.MolToSmiles(new_mol)
                        valid_candidates.append((operation, new_smiles, pol_logit))
                except Exception:
                    continue

            if not valid_candidates:
                continue

            # ---- OFO 批量打分 ----
            batch_from = [current_smiles] * len(valid_candidates)
            batch_to = [c[1] for c in valid_candidates]
            batch_ops = [c[0] for c in valid_candidates]

            try:
                property_changes = predictor.predict_batch(batch_from, batch_to, batch_ops)
            except Exception:
                continue

            ofo_scored_candidates += len(valid_candidates)

            # ---- 计算 f_score 并筛选 top max_branching ----
            scored = []
            for i, (operation, new_smiles, pol_logit) in enumerate(valid_candidates):
                if i >= len(property_changes):
                    continue
                pc = property_changes[i]
                if pc is None or isinstance(pc, (list, dict)):
                    continue

                # g_score = 已实现的累计改善（方向归一化，越大越好）
                step_gain = -float(pc) if optimization_direction == "decrease" else float(pc)
                child_g = current_g + step_gain

                # h_score（ValueNet 估计的未来收益）
                h_score = 0.0
                if value_net is not None and _rl_imports_ok:
                    try:
                        child_logp = self.calculate_logP(new_smiles) or 0.0
                        child_logp_in_range = logp_min <= child_logp <= logp_max
                        child_state_vec = encode_state(
                            smiles=new_smiles,
                            accumulated_change=current_node.get("accumulated_change", 0.0) + float(pc),
                            remaining_depth=max_depth - current_depth - 1,
                            max_depth=max_depth,
                            logp_value=child_logp,
                            logp_in_range=child_logp_in_range,
                            stagnation_count=0,
                            direction=optimization_direction,
                            target_property_value=initial_property_value or 0.0,
                        )
                        if value_device is not None:
                            child_state_vec = child_state_vec.to(value_device)
                        h_score = value_net.estimate(child_state_vec)
                    except Exception:
                        h_score = 0.0

                # policy_bonus（对数 logit 缩放，鼓励高 policy 优先级节点）
                policy_bonus = float(pol_logit) * 0.1

                f_score = child_g + h_score + policy_bonus
                scored.append((operation, new_smiles, float(pc), child_g, h_score, f_score, pol_logit))

            if not scored:
                continue

            # 取 top max_branching
            scored.sort(key=lambda x: x[5], reverse=True)
            top_scored = scored[:max_branching]

            # ---- 收集 TrajectoryStep（在线 RL）----
            if rl_trainer is not None and policy_net is not None and _rl_imports_ok:
                try:
                    import torch
                    state_vec_traj = encode_state(
                        smiles=current_smiles,
                        accumulated_change=current_node.get("accumulated_change", 0.0),
                        remaining_depth=max_depth - current_depth,
                        max_depth=max_depth,
                        logp_value=current_node.get("logP") or 0.0,
                        logp_in_range=current_node.get("logP_in_range", True),
                        stagnation_count=current_node.get("stagnation_count", 0),
                        direction=optimization_direction,
                        target_property_value=initial_property_value or 0.0,
                    )
                    all_action_vecs = torch.stack([
                        encode_action(op, ofo_predicted_change=float(pc))
                        for op, _, pc, *_ in top_scored
                    ])
                    state_vec_policy = state_vec_traj.to(policy_device) if policy_device is not None else state_vec_traj
                    all_action_vecs_policy = all_action_vecs.to(policy_device) if policy_device is not None else all_action_vecs
                    selection_mode = getattr(rl_trainer, "action_selection_mode", "sample")
                    selection_temperature = getattr(rl_trainer, "action_selection_temperature", 1.0)
                    selection_epsilon = getattr(rl_trainer, "action_selection_epsilon", 0.0)
                    selection = policy_net.select_action(
                        state_vec_policy,
                        all_action_vecs_policy,
                        mode=selection_mode,
                        temperature=selection_temperature,
                        epsilon=selection_epsilon,
                    )
                    selected_idx = int(selection["selected_idx"])
                    greedy_action_idx = int(selection["greedy_idx"])
                    log_prob = selection["log_prob"]
                    selected_prob = selection.get("prob")
                    selected_op, selected_smiles, selected_pc, _, _, selected_f_score, selected_pol_logit = top_scored[selected_idx]
                    selected_logp = self.calculate_logP(selected_smiles) or 0.0
                    selected_logp_in_range = logp_min <= selected_logp <= logp_max

                    if value_net is not None:
                        value_state_vec = state_vec_traj.to(value_device) if value_device is not None else state_vec_traj
                        value_est = value_net.estimate(value_state_vec)
                    else:
                        value_est = 0.0

                    try:
                        from mol_evo.core.models.astar_rl.reward import compute_step_reward
                        from mol_evo.core.models.astar_rl.reward import RewardConfig
                        step_reward = compute_step_reward(
                            property_change=selected_pc,
                            logp_in_range=selected_logp_in_range,
                            stagnation_count=current_node.get("stagnation_count", 0),
                            config=RewardConfig(direction=optimization_direction),
                        )
                    except Exception:
                        step_reward = 0.0

                    next_state_vec = encode_state(
                        smiles=selected_smiles,
                        accumulated_change=current_node.get("accumulated_change", 0.0) + selected_pc,
                        remaining_depth=max_depth - current_depth - 1,
                        max_depth=max_depth,
                        logp_value=selected_logp,
                        logp_in_range=selected_logp_in_range,
                        stagnation_count=0,
                        direction=optimization_direction,
                        target_property_value=initial_property_value or 0.0,
                    )
                    is_done = (current_depth + 1 >= max_depth)
                    traj_step = TrajectoryStep(
                        state_tensor=state_vec_traj.detach().cpu(),
                        action_tensors=all_action_vecs.detach().cpu(),
                        selected_action_idx=selected_idx,
                        log_prob=log_prob.detach().cpu(),
                        step_reward=step_reward,
                        next_state_tensor=next_state_vec.detach().cpu(),
                        value_estimate=float(value_est),
                        done=is_done,
                        selection_mode=selection_mode,
                        selected_action_rank=selected_idx,
                        policy_entropy=float(selection["entropy"].detach().cpu().item()),
                        selected_action_prob=(
                            float(selected_prob.detach().cpu().item())
                            if selected_prob is not None else None
                        ),
                        greedy_action_idx=greedy_action_idx,
                        matches_policy_greedy=(selected_idx == greedy_action_idx),
                        metadata={
                            "selected_smiles": selected_smiles,
                            "selected_operation": selected_op.get("type"),
                            "selected_f_score": float(selected_f_score),
                            "selected_policy_logit": float(selected_pol_logit),
                            "matches_f_score_top1": (selected_idx == 0),
                        },
                    )
                    rl_trainer.collect_step(traj_step)
                except Exception:
                    pass

            # ---- 加入树与 open set ----
            for operation, new_smiles, pc, child_g, h_score, f_score, pol_logit in top_scored:
                path_key = (new_smiles, current_id)
                if path_key in visited_path_keys:
                    continue
                visited_path_keys.add(path_key)

                new_logp = self.calculate_logP(new_smiles)
                new_logp_in_range = (
                    new_logp is not None and logp_min <= new_logp <= logp_max
                )
                new_accumulated = current_node.get("accumulated_change", 0.0) + pc
                new_property_value = (
                    current_node.get("property_value", 0.0) + pc
                    if "property_value" in current_node else None
                )

                node_id = str(node_counter)
                new_node = {
                    "id": node_id,
                    "smiles": new_smiles,
                    "depth": current_depth + 1,
                    "parent_id": current_id,
                    "operation": operation["type"],
                    "details": operation.get("params", {}),
                    "property_change": pc,
                    "accumulated_change": new_accumulated,
                    "logP": new_logp,
                    "logP_in_range": new_logp_in_range,
                    # A* 额外字段
                    "g_score": child_g,
                    "h_score": h_score,
                    "f_score": f_score,
                    "policy_score": float(pol_logit),
                    "stagnation_count": 0,
                    "logp_violation_count": 0 if new_logp_in_range else 1,
                }
                if new_property_value is not None:
                    new_node["property_value"] = new_property_value

                expansion_tree["nodes"][node_id] = new_node
                expansion_tree["edges"].append({
                    "from": current_id,
                    "to": node_id,
                    "operation": operation["type"],
                    "details": operation.get("params", {}),
                })
                node_counter += 1
                actual_expansions += 1

                # 压入 open set（负 f_score 实现最大优先）
                heapq.heappush(
                    open_set,
                    (-f_score, tie_counter, node_id, child_g),
                )
                tie_counter += 1
                open_set_peak = max(open_set_peak, len(open_set))

        # ------------------------------------------------------------------
        # episode 结束，触发在线 RL 更新
        # ------------------------------------------------------------------
        if rl_trainer is not None and _rl_imports_ok:
            try:
                rl_trainer.end_episode()
            except Exception:
                pass

        # ------------------------------------------------------------------
        # astar_stats
        # ------------------------------------------------------------------
        expansion_tree["astar_stats"] = {
            "expanded_nodes": expanded_nodes,
            "open_set_peak": open_set_peak,
            "policy_prefilter_size": policy_prefilter_size,
            "ofo_scored_candidates": ofo_scored_candidates,
            "actual_expansions": actual_expansions,
        }

        total_nodes = len(expansion_tree["nodes"])
        total_edges = len(expansion_tree["edges"])
        print(
            f"[astar_demo] 搜索完成: 树节点={total_nodes}, 边={total_edges}, "
            f"展开次数={expanded_nodes}, OFO调用={ofo_scored_candidates}"
        )

        return expansion_tree

    def generate_sequential_expansions(self, num_steps: int = 5) -> List[Dict]:
        """
        生成从初始分子开始的顺序扩展路径，每一步都是在前一步基础上进行操作
        
        Args:
            num_steps: 要执行的扩展步骤数
            
        Returns:
            包含每一步扩展结果的列表
        """
        expansions = []
        current_mol = self.initial_mol
        current_smiles = self.initial_smiles
        
        # 添加初始状态
        expansions.append({
            "step": 0,
            "smiles_from": None,
            "smiles_to": current_smiles,
            "operation": "initial",
            "details": {}
        })
        
        # 逐步执行扩展
        for step in range(1, num_steps + 1):
            # 获取可能的操作
            possible_operations = self._get_possible_operations(current_mol)
            if not possible_operations:
                break
                
            # 随机选择一个操作
            selected_operation = random.choice(possible_operations)
            
            # 应用操作
            operation_type = selected_operation["type"]
            operation_params = selected_operation.get("params", {})
            
            new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
            
            if new_mol and self.validate_molecule(new_mol):
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # 记录扩展步骤
                expansions.append({
                    "step": step,
                    "smiles_from": current_smiles,
                    "smiles_to": new_smiles,
                    "operation": operation_type,
                    "details": operation_params
                })
                
                # 更新当前分子
                current_mol = new_mol
                current_smiles = new_smiles
            else:
                # 如果操作失败，跳过这一步
                continue
                
        return expansions

    def generate_paths_with_max_steps(self, num_paths: int = 10, max_steps: int = 3) -> List[Dict]:
        """
        生成多个进化路径，返回所有步骤数小于等于max_steps的中间结果
        
        Args:
            num_paths: 要生成的路径数量
            max_steps: 最大步骤数，返回所有步骤数小于等于此值的中间结果
            
        Returns:
            包含所有满足条件的步骤的列表，每个步骤包含SMILES和对应操作
        """
        all_results = []
        
        # 生成多条顺序扩展路径
        for path_idx in tqdm(range(num_paths), desc="生成扩展路径", unit="条"):
            expansions = self.generate_sequential_expansions(max_steps)
            
            # 转换为所需的输出格式
            for expansion in expansions[1:]:  # 跳过初始状态
                result_entry = {
                    "smiles_from": expansion["smiles_from"],
                    "smiles_to": expansion["smiles_to"],
                    "operations": [{
                        "position": str(expansion["details"].get("atom_idx", "")) if expansion["operation"] == "add_atom" 
                                  else str(expansion["details"].get("atom_idx", "")) if expansion["operation"] == "replace_atom"
                                  else f"{expansion['details'].get('atom_idx', '')}-{expansion['details'].get('atom2_idx', '')}" if "bond" in expansion["operation"]
                                  else "",
                        "atom": expansion["details"].get("atom_symbol", expansion["details"].get("atom_symbol", "")),
                        "operation": expansion["operation"],
                        "from_atom": expansion["smiles_from"]
                    }]
                }
                all_results.append(result_entry)
        
        return all_results

        # Error Handling
   
    # TAG Error Handling: 加载默认配置
    def _load_default_config(self):
        """加载默认的原子类型和操作类型配置"""
        # 操作类型定义 - 与数据集中操作类型保持一致，移除 add_fragment
        self.operation_types = {
            "add_atom": "添加原子",
            "replace_atom": "替换原子",
            "form_double_bond": "形成双键",
            "form_triple_bond": "形成三键",
            "break_bond": "断开键", # UPDATE 数据集中好像没有这种操作；
            "add_stereo": "添加立体化学",
            "form_ring": "成环",
            "form_double_ring": "形成双键环",
            "form_triple_ring": "形成三键环",
            "form_aromatic_ring": "形成芳香环",     # 10
            "remove_add_stereo": "移除并添加立体化学",
            "remove_form_double_bond": "移除双键",
            "remove_form_triple_bond": "移除三键",
            "remove_form_ring": "移除环",
            "remove_form_double_ring": "移除双键环",    # 15
            "remove_form_triple_ring": "移除三键环",
            "remove_form_aromatic_ring": "移除芳香环"
        }
        
        # 默认操作类型键列表
        self.operation_type_keys = [
            "add_atom", "replace_atom", "form_double_bond", "form_triple_bond", 
            "break_bond", "add_stereo", "form_ring", "form_double_ring", 
            "form_triple_ring", "form_aromatic_ring", "remove_add_stereo",
            "remove_form_double_bond", "remove_form_triple_bond", "remove_form_ring",
            "remove_form_double_ring", "remove_form_triple_ring", "remove_form_aromatic_ring"
        ]
        
        # 常见的原子类型和片段（移除了片段部分，因为我们不再使用 add_fragment）
        self.common_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        
        # 确保错误统计被正确初始化
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
    
    # TAG Debug
    def _update_error_stats(self, e: Exception):
        """更新错误统计"""
        # 确保 error_stats 属性存在
        if not hasattr(self, 'error_stats'):
            self.error_stats = {
                "valence_errors": 0,
                "kekulization_errors": 0,
                "other_errors": 0,
                "total_attempts": 0
            }
        
        self.error_stats["total_attempts"] += 1
        error_msg = str(e).lower()
        if "explicit valence" in error_msg:
            self.error_stats["valence_errors"] += 1
        elif "kekulize" in error_msg or "aromatic" in error_msg:
            self.error_stats["kekulization_errors"] += 1
        else:
            self.error_stats["other_errors"] += 1
    
    def _print_tree_recursive(self, nodes: Dict, children_map: Dict, node_id: str, prefix: str, is_last: bool) -> None:
        """
        递归打印树形结构
        
        Args:
            nodes: 节点字典
            children_map: 子节点映射
            node_id: 当前节点ID
            prefix: 前缀字符串
            is_last: 是否为最后一个子节点
        """
        # 获取当前节点的子节点
        children = children_map.get(node_id, [])
        
        # 打印子节点
        for i, child_id in enumerate(children):
            if child_id not in nodes:
                continue
                
            child_node = nodes[child_id]
            is_last_child = (i == len(children) - 1)
            
            # 构造操作信息
            operation_info = f" [{child_node['operation']}]" if child_node['operation'] else ""
            
            # 构造变化值信息
            change_info = ""  
            if 'property_change' in child_node:
                change_info = f" (变化: {child_node['property_change']:.4f})"
            
            # 构造属性值和深度信息
            property_info = f", 属性值: {child_node['property_value']:.6f}" if 'property_value' in child_node else ""
            depth_info = f" (深度: {child_node['depth']})"
            
            # 打印当前子节点
            if is_last_child:
                print(f"{prefix}└── {child_node['smiles']}{operation_info}{change_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}    "
            else:
                print(f"{prefix}├── {child_node['smiles']}{operation_info}{change_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}│   "
            
            
            # 递归打印孙节点
            self._print_tree_recursive(nodes, children_map, child_id, new_prefix, is_last_child)

    def print_expansion_tree(self, tree_data: Dict, format: str = 'text') -> None:
        """
        以文件系统树的形式打印扩展树
        
        Args:
            tree_data: generate_expansion_tree方法生成的树形数据
            format: 输出格式 ('text' 或 'json')
        """
        if format == 'json':
            print(json.dumps(tree_data, ensure_ascii=False, indent=2))
            return
            
        # 以树形结构打印
        nodes = tree_data["nodes"]
        edges = tree_data["edges"]
        initial_smiles = tree_data["initial_smiles"]
        
        print(f"初始分子: {initial_smiles}")
        print(f"最大深度: {tree_data['max_depth']}")
        print(f"最大分支数: {tree_data['max_branching']}")
        print("\n扩展树结构:")
        
        # 打印根节点
        root_node = nodes["0"]
        property_info = f", 属性值: {root_node['property_value']:.6f}" if 'property_value' in root_node else ""
        print(f"└── {root_node['smiles']} (深度: {root_node['depth']}{property_info})")
        
        # 构建父子关系映射（基于边数据）
        children_map = {}
        for edge in edges:
            parent_id = edge["from"] if edge["from"] is not None else "0"  # 根节点的from为None
            child_id = edge["to"]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
        
        # 递归打印树形结构，从根节点的子节点开始
        self._print_tree_recursive(nodes, children_map, "0", "    ", True)


def main():
    parser = argparse.ArgumentParser(description='分子进化路径生成器')
    
    # 直接添加所有参数，不再使用 subparser
    parser.add_argument('smiles', help='初始分子的SMILES字符串')
    parser.add_argument('--num-paths', type=int, default=5, help='要生成的路径数量')
    parser.add_argument('--steps-per-path', type=int, default=3, help='每条路径的步数')
    parser.add_argument('--max-steps', type=int, help='最大步骤数，返回所有中间步骤结果')
    parser.add_argument('--max-branching', type=int, default=3, help='每个节点的最大分支数')
    parser.add_argument('--diversity-threshold', type=float, default=0.7, help='多样性阈值')
    parser.add_argument('--config-file', type=str, help='配置文件路径')
    parser.add_argument('--mode', choices=['sequential', 'tree'], default='sequential', 
                        help='扩展模式: sequential(顺序) 或 tree(树形)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='输出格式 (默认: text)')
    
    args = parser.parse_args()
    
    if args.config_file:
        expander = MolecularEvolutionExpansion(args.smiles, config_file=args.config_file)
    else:
        expander = MolecularEvolutionExpansion(args.smiles)
    
    if args.mode == 'tree':
        # 使用树形扩展模式
        paths = expander.generate_expansion_tree(
            max_depth=args.max_steps or 3,
            max_branching=args.max_branching
        )
    elif args.max_steps:
        # 使用顺序扩展模式
        paths = expander.generate_paths_with_max_steps(
            num_paths=args.num_paths,
            max_steps=args.max_steps
        )
    else:
        # 使用原有功能
        paths = expander.generate_multiple_paths(
            num_paths=args.num_paths,
            steps_per_path=args.steps_per_path,
            max_branching=args.max_branching,
            diversity_threshold=args.diversity_threshold
        )
    
    if args.format == 'text':
        if args.mode == 'tree':
            # 树形模式的文本输出格式
            expander.print_expansion_tree(paths, 'text')
        elif args.max_steps:
            # 顺序扩展模式的文本输出格式
            print(f"初始分子: {args.smiles}")
            print(f"最大步骤数: {args.max_steps}")
            print("顺序扩展步骤:")
            for step in paths:
                print(f"  {step['smiles_from']} -> {step['smiles_to']}: {step['operations'][0]['operation']}")
        else:
            # 原有功能的文本输出格式
            for i, path in enumerate(paths):
                print(f"路径 {i+1}:")
                print(f"  初始分子: {path['initial_smiles']}")
                print(f"  最终分子: {path['final_smiles']}")
                print(f"  路径长度: {path['path_length']}")
                print("  步骤:")
                for step in path['steps']:
                    print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
                print()
    elif args.format == 'json':
        print(json.dumps(paths, ensure_ascii=False, indent=2))


# 当作为主模块运行时执行示例代码
if __name__ == "__main__":
    # 检查是否提供了命令行参数
    if len(sys.argv) > 1:
        # 如果有参数，使用CLI模式
        main()
    else:
        # 否则运行原有的示例代码
        # 创建多路径进化器
        multi_evolver = MolecularEvolutionExpansion("CC")
        
        print("初始分子: CC")
        print("生成多条进化路径...")
        
        # 生成10条不同的进化路径，每条5步
        paths = multi_evolver.generate_multiple_paths(
            num_paths=10, 
            steps_per_path=5,
            diversity_threshold=0.8
        )
        
        # 分析多样性
        diversity = multi_evolver.analyze_path_diversity(paths)
        print(f"\n多样性分析:")
        print(f"  多样性分数: {diversity['diversity_score']:.3f}")
        print(f"  唯一分子数: {diversity['unique_molecules']}")
        print(f"  总路径数: {diversity['total_paths']}")
        print(f"  平均路径长度: {diversity['average_path_length']:.2f}")
        
        # 显示前3条路径
        print(f"\n前3条进化路径:")
        for i, path in enumerate(paths[:3]):
            print(f"\n路径 {i + 1}:")
            print(f"  最终分子: {path['final_smiles']}")
            print(f"  路径长度: {path['path_length']} 步")
            
            for step in path["steps"][:3]:  # 显示前3步
                print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
            if len(path["steps"]) > 3:
                print(f"    ... 还有 {len(path['steps']) - 3} 步")
        
        # 生成进化树
        print(f"\n生成进化树...")
        evolution_tree = multi_evolver.generate_evolution_tree(max_depth=3, max_branching=2)
        print(f"进化树包含 {len(evolution_tree['nodes'])} 个节点和 {len(evolution_tree['edges'])} 条边")
