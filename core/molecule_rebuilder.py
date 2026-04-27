#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子重建器模块，用于根据操作路径逐步重建分子并可视化每一步的状态。
"""

import os
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem, rdchem
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

try:
    from .utils.molecule import smile_to_graph_xyz
except ImportError:
    from mol_evo.core.utils.molecule import smile_to_graph_xyz
    
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'Noto Mono', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MoleculeRebuilder:
    """
    分子重建器，能够根据操作路径逐步重建分子并可视化每一步的状态。
    """
    
    def __init__(self, path: list, types: dict = None, obverse: bool = True):
        self.path = path
        self.steps = []
        self.current_mol = None
        self.atom_map = {}
        self.atom_counter = 0
        self.types = types or self._get_default_types()
        self.obverse = obverse
    
    def _get_default_types(self):
        """获取默认的原子类型映射"""
        return {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7,
            'Br': 8, 'I': 9, 'B': 10, 'Si': 11, 'Se': 12, 'As': 13, 'Te': 14
        }
    
    def rebuild_step_by_step(self, analyze_xyz: bool = False):
        """
        逐步执行路径操作，记录每一步的分子状态。
        返回步骤列表，每个步骤包含操作信息和分子SMILES。
        
        Args:
            analyze_xyz: 是否使用 smile_to_graph_xyz 分析 z 和 pos 参数
        """        
        # 初始化空分子
        self.current_mol = rdchem.RWMol()
        self.atom_map = {}
        self.atom_counter = 0
        self.steps = []
        
        for step_idx, operation in enumerate(self.path):
            step_info = {
                "step": step_idx + 1,
                "operation": operation,
                "smiles_before": self._get_current_smiles(),
                "description": self._describe_operation(operation)
            }
            
            # 执行操作
            self._execute_operation(operation)
            
            # 记录操作后的状态
            step_info["smiles_after"] = self._get_current_smiles()
            
            # 如果需要，分析 z 和 pos 参数
            if analyze_xyz:
                step_info["xyz_analysis"] = self._analyze_xyz()
            
            self.steps.append(step_info)
        
        return self.steps
    
    def _analyze_xyz(self):
        """
        分析当前分子的 z 和 pos 参数。
        优先尝试通过 SMILES 解析，如果失败则直接从 RWMol 对象提取原子信息。
        
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            smiles = self._get_current_smiles()
            
            if smiles != "invalid":
                x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, self.types)
                
                if z is not None and pos is not None:
                    return {
                        "success": True,
                        "z": z.tolist() if hasattr(z, 'tolist') else z,
                        "pos": pos.tolist() if hasattr(pos, 'tolist') else pos,
                        "num_atoms": len(z),
                        "edge_index": edge_index.tolist() if edge_index is not None else None,
                        "edge_attr": edge_attr.tolist() if edge_attr is not None else None,
                        "method": "smiles_based"
                    }
            
            return self._analyze_from_rwmol()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "z": None,
                "pos": None
            }
    
    def _analyze_from_rwmol(self):
        """
        直接从 RWMol 对象提取原子信息和键信息，不依赖 SMILES。
        对于不完整的分子结构，可以提取基本的原子序数和连接关系。
        
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            mol = self.current_mol.GetMol()
            num_atoms = mol.GetNumAtoms()
            
            if num_atoms == 0:
                return {
                    "success": False,
                    "error": "No atoms in molecule",
                    "z": None,
                    "pos": None
                }
            
            z = []
            for atom in mol.GetAtoms():
                z.append(atom.GetAtomicNum())
            
            z = np.array(z, dtype=np.int64)
            
            edge_index = []
            edge_attr = []
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                edge_index.append([start_idx, end_idx])
                edge_index.append([end_idx, start_idx])
                
                bond_type = bond.GetBondType()
                bond_type_map = {
                    Chem.BondType.SINGLE: [1, 0, 0, 0],
                    Chem.BondType.DOUBLE: [0, 1, 0, 0],
                    Chem.BondType.TRIPLE: [0, 0, 1, 0],
                    Chem.BondType.AROMATIC: [0, 0, 0, 1],
                }
                bond_attr = bond_type_map.get(bond_type, [0, 0, 0, 0])
                edge_attr.append(bond_attr)
                edge_attr.append(bond_attr)
            
            if edge_index:
                edge_index = np.array(edge_index, dtype=np.int64).T
                edge_attr = np.array(edge_attr, dtype=np.float32)
            else:
                edge_index = None
                edge_attr = None
            
            try:
                mol_with_h = Chem.AddHs(mol)
                conf_id = AllChem.EmbedMolecule(mol_with_h, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
                
                if conf_id != -1:
                    conf = mol_with_h.GetConformer()
                    pos = conf.GetPositions()
                    pos = pos[:num_atoms]
                else:
                    pos = self._generate_simple_coordinates(mol)
            except:
                pos = self._generate_simple_coordinates(mol)
            
            x = self._generate_node_features(z, edge_index, edge_attr)
            
            return {
                "success": True,
                "z": z.tolist() if hasattr(z, 'tolist') else z,
                "pos": pos.tolist() if hasattr(pos, 'tolist') else pos,
                "num_atoms": len(z),
                "edge_index": edge_index.tolist() if edge_index is not None else None,
                "edge_attr": edge_attr.tolist() if edge_attr is not None else None,
                "method": "rwmol_based"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"RWMol analysis failed: {str(e)}",
                "z": None,
                "pos": None
            }
    
    def _generate_simple_coordinates(self, mol):
        """
        为分子生成简化的 2D 坐标。
        使用 RDKit 的 2D 坐标生成方法，对不完整的分子更宽容。
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            numpy.ndarray: 原子坐标数组
        """
        try:
            mol_copy = Chem.Mol(mol)
            
            try:
                rdDepictor.Compute2DCoords(mol_copy)
                conf = mol_copy.GetConformer()
                pos = conf.GetPositions()
                
                return np.array(pos, dtype=np.float32)
            except:
                num_atoms = mol_copy.GetNumAtoms()
                pos = np.zeros((num_atoms, 3), dtype=np.float32)
                
                for i in range(num_atoms):
                    pos[i] = [float(i * 1.5), 0.0, 0.0]
                
                return pos
        except:
            import numpy as np
            num_atoms = mol.GetNumAtoms()
            return np.zeros((num_atoms, 3), dtype=np.float32)
    
    def _generate_node_features(self, z, edge_index, edge_attr):
        """
        生成节点特征矩阵。
        
        Args:
            z: 原子序数数组
            edge_index: 边索引
            edge_attr: 边属性
            
        Returns:
            numpy.ndarray: 节点特征矩阵
        """
        try:
            import numpy as np
            
            num_atoms = len(z)
            max_atomic_num = max(z) if len(z) > 0 else 0
            num_types = len(self.types)
            
            x = np.zeros((num_atoms, num_types), dtype=np.float32)
            
            atomic_num_to_type = {}
            for symbol, type_idx in self.types.items():
                atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
                atomic_num_to_type[atomic_num] = type_idx
            
            for i, atomic_num in enumerate(z):
                type_idx = atomic_num_to_type.get(atomic_num, 0)
                x[i, type_idx] = 1.0
            
            return x
        except:
            import numpy as np
            return np.zeros((len(z), len(self.types)), dtype=np.float32)
    
    def _execute_operation(self, operation):
        """执行单个操作"""
        op_type = operation.get("operation")
        position = operation.get("position")
        atom = operation.get("atom")
        
        if op_type == "init_atom":
            self._init_atom(position, atom)
        elif op_type == "add_atom":
            self._add_atom(position, atom)
        elif op_type == "form_double_bond":
            self._form_bond(position, Chem.BondType.DOUBLE)
        elif op_type == "form_triple_bond":
            self._form_bond(position, Chem.BondType.TRIPLE)
        elif op_type == "form_aromatic_bond":
            self._form_bond(position, Chem.BondType.AROMATIC)
        elif op_type == "form_ring":
            self._form_bond(position, Chem.BondType.SINGLE)
        elif op_type == "form_double_ring":
            self._form_bond(position, Chem.BondType.DOUBLE)
        elif op_type == "form_aromatic_ring" or op_type == "成苯环":
            # 处理合并的芳香环操作，位置格式为 "0-1-2-3-4-5"
            pos_parts = position.split("-")
            if len(pos_parts) == 2:
                # 如果只有两个位置，按照普通键处理
                self._form_bond(position, Chem.BondType.AROMATIC)
            else:
                # 如果有多个位置，处理为环结构
                # 首先形成相邻原子之间的键
                for i in range(len(pos_parts) - 1):
                    self._form_bond(f"{pos_parts[i]}-{pos_parts[i+1]}", Chem.BondType.AROMATIC)
                # 最后形成首尾原子之间的键（闭环）
                self._form_bond(f"{pos_parts[-1]}-{pos_parts[0]}", Chem.BondType.AROMATIC)
        elif op_type == "add_stereo_ccw" or op_type == "add_stereo_cw":
            self._add_stereo(operation, ccw_flag=(op_type == "add_stereo_ccw"))  # 传递整个操作对象，包含立体构型信息
        elif op_type == "add_fragment":
            self._add_fragment(position, atom)
    
    def _init_atom(self, position, atom_symbol):
        """初始化第一个原子"""
        atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
        self.atom_map[position] = atom_idx
        self.atom_counter += 1
    
    def _add_atom(self, position, atom_symbol):
        """添加原子到指定位置"""
        parent_pos = position
        if parent_pos not in self.atom_map:
            parent_pos = str(self.atom_counter - 1)
        
        atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
        self.atom_map[str(self.atom_counter)] = atom_idx
        
        # 添加单键连接到父原子
        if parent_pos in self.atom_map:
            parent_idx = self.atom_map[parent_pos]
            self.current_mol.AddBond(parent_idx, atom_idx, Chem.BondType.SINGLE)
        
        self.atom_counter += 1
    
    def _form_bond(self, position, bond_type):
        """形成键"""
        if "-" in position:
            pos1, pos2 = position.split("-")
            if pos1 in self.atom_map and pos2 in self.atom_map:
                idx1 = self.atom_map[pos1]
                idx2 = self.atom_map[pos2]
                
                # 检查是否已存在键
                existing_bond = self.current_mol.GetBondBetweenAtoms(idx1, idx2)
                if existing_bond:
                    existing_bond.SetBondType(bond_type)
                else:
                    # 在添加键之前检查是否形成闭环
                    will_form_ring = False
                    if bond_type == Chem.BondType.AROMATIC:
                        will_form_ring = self._is_ring_closure(idx1, idx2)
                    
                    self.current_mol.AddBond(idx1, idx2, bond_type)
                    
                    # 如果形成的是芳香键，标记连接的原子为芳香性
                    if bond_type == Chem.BondType.AROMATIC:
                        # 标记连接的原子为芳香性
                        self.current_mol.GetAtomWithIdx(idx1).SetIsAromatic(True)
                        self.current_mol.GetAtomWithIdx(idx2).SetIsAromatic(True)
                        
                        # 如果形成了闭环，检查芳香环中的氮原子
                        if will_form_ring:
                            self._adjust_aromatic_nitrogen_hydrogens()
    
    def _is_ring_closure(self, idx1, idx2):
        """
        检查在两个原子之间形成键是否会形成闭环。
        
        Args:
            idx1: 第一个原子的索引
            idx2: 第二个原子的索引
            
        Returns:
            bool: 如果形成闭环则返回True，否则返回False
        """
        try:
            mol = self.current_mol.GetMol()
            
            # 检查是否已经存在直接的键
            existing_bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if existing_bond:
                return False
            
            # 使用深度优先搜索检查两个原子之间是否已经存在路径
            visited = set()
            stack = [(idx1, [idx1])]
            
            while stack:
                current, path = stack.pop()
                
                if current == idx2:
                    # 如果路径长度大于1，说明已经存在路径，形成闭环
                    if len(path) > 1:
                        return True
                    continue
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                # 获取当前原子的所有邻居
                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in path:
                        stack.append((neighbor_idx, path + [neighbor_idx]))
            
            return False
        except Exception as e:
            return False
    
    def _adjust_aromatic_nitrogen_hydrogens(self):
        """
        在形成芳香键后，检查芳香环中的氮原子类型，为吡咯型氮原子添加氢原子。
        
        判断标准：
        - 氮原子连接了两个芳香键（键类型为AROMATIC）
        - 键价总和为3.0
        - 氮原子属于5元芳香环（吡咯型）
        """
        try:
            # 直接使用 self.current_mol，因为它是 RWMol 对象，可以添加原子
            mol = self.current_mol
            
            # 清理分子以确保环信息正确
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                # 继续尝试获取环信息
                pass
            
            # 获取环信息
            ring_info = mol.GetRingInfo()
            
            # 遍历所有原子
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != "N":
                    continue
                
                # 计算芳香键数（直接检查键类型）
                aromatic_bond_count = sum(1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.BondType.AROMATIC)
                
                # 计算键价总和
                current_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                
                # 检查是否符合吡咯型氮原子的条件
                if aromatic_bond_count == 2 and current_valence == 3.0:
                    # 检查该氮原子是否属于5元芳香环
                    atom_idx = atom.GetIdx()
                    is_in_5_ring = False
                    
                    for ring in ring_info.AtomRings():
                        if atom_idx in ring and len(ring) == 5:
                            is_in_5_ring = True
                            break
                    
                    # 判断类型：吡咯型氮原子（属于5元芳香环）
                    if is_in_5_ring:
                        # 获取当前氢原子数
                        atom.UpdatePropertyCache()
                        current_h = atom.GetTotalNumHs()
                        
                        # 如果没有氢原子，添加一个
                        if current_h == 0:
                            # 直接添加氢原子
                            h_atom = Chem.Atom("H")
                            h_idx = mol.AddAtom(h_atom)
                            mol.AddBond(atom.GetIdx(), h_idx, Chem.BondType.SINGLE)
                            
                            # 添加氢原子后立即返回，避免迭代器失效
                            return
        except Exception as e:
            pass
    
    def _add_stereo(self, operation, ccw_flag):
        # """添加立体化学信息"""
        # tmp_map = { # TEST
        #     5: 2,
        #     8: 3,
        #     2: 21,
        #     4: 23
        # }
        position = operation.get("position")
        idx = self.atom_map[position]
        # tmp_current_smiles = Chem.MolToSmiles(self.current_mol) # TEST 调整分子顺序
        # self.current_mol = rdchem.RWMol(Chem.MolFromSmiles(tmp_current_smiles))
        atom = self.current_mol.GetAtomWithIdx(idx)  # INFO ori
        # atom = self.current_mol.GetAtomWithIdx(tmp_map[idx])  # TEST
        # 使用操作类型确定手性：ccw_flag=True
        target_tag = Chem.CHI_TETRAHEDRAL_CCW if not ccw_flag else Chem.CHI_TETRAHEDRAL_CW
        atom.SetChiralTag(target_tag)
    
    def _add_fragment(self, position, fragment_smiles):
        """添加片段"""
        try:
            frag_mol = Chem.MolFromSmiles(fragment_smiles)
            if frag_mol:
                parent_idx = self.atom_map.get(position)
                if parent_idx is not None:
                    offset = self.current_mol.GetNumAtoms()
                    
                    # 添加片段中的所有原子
                    for atom in frag_mol.GetAtoms():
                        new_atom = Chem.Atom(atom.GetAtomicNum())
                        self.current_mol.AddAtom(new_atom)
                    
                    # 添加片段中的所有键
                    for bond in frag_mol.GetBonds():
                        b_idx = bond.GetBeginAtomIdx() + offset
                        e_idx = bond.GetEndAtomIdx() + offset
                        self.current_mol.AddBond(b_idx, e_idx, bond.GetBondType())
                    
                    # 连接片段到父原子
                    if offset < self.current_mol.GetNumAtoms():
                        self.current_mol.AddBond(parent_idx, offset, Chem.BondType.SINGLE)
        except Exception as e:
            pass
    
    def _get_current_smiles(self):
        """获取当前分子的SMILES"""
        try:
            mol = self.current_mol.GetMol()
            Chem.SanitizeMol(mol)
            
            # 直接生成SMILES
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return "invalid"
    
    def _describe_operation(self, operation):
        """生成操作的描述"""
        op_type = operation.get("operation")
        position = operation.get("position")
        atom = operation.get("atom")
        
        descriptions = {
            "init_atom": f"初始化原子 {atom} (位置 {position})",
            "add_atom": f"添加原子 {atom} 连接到位置 {position}",
            "form_double_bond": f"在位置 {position} 形成双键",
            "form_triple_bond": f"在位置 {position} 形成三键",
            "form_aromatic_bond": f"在位置 {position} 形成芳香键",
            "form_ring": f"在位置 {position} 形成环",
            "form_aromatic_ring": f"在位置 {position} 形成芳香环",
            "add_stereo": f"在位置 {position} 添加立体化学信息",
            "add_stereo_ccw": f"在位置 {position} 添加CCW手性",
            "add_stereo_cw": f"在位置 {position} 添加CW手性",
            "add_fragment": f"添加片段 {atom} 到位置 {position}"
        }
        
        return descriptions.get(op_type, f"执行操作: {op_type}")
    
    def visualize_steps(self, output_file: str = None, analyze_xyz: bool = False):
        """
        可视化重建步骤，打印每一步的详细信息。
        如果指定了 output_file，将结果保存到文件。
        
        Args:
            output_file: 输出文件路径
            analyze_xyz: 是否分析 xyz 数据
        """
        if not self.steps:
            self.rebuild_step_by_step(analyze_xyz=analyze_xyz)
        
        lines = []
        lines.append("=" * 80)
        lines.append("分子逐步重建过程可视化")
        lines.append("=" * 80)
        lines.append("")
        
        for step in self.steps:
            lines.append(f"步骤 {step['step']}: {step['description']}")
            lines.append("-" * 80)
            lines.append(f"操作类型: {step['operation']['operation']}")
            lines.append(f"位置: {step['operation']['position']}")
            lines.append(f"原子: {step['operation']['atom']}")
            lines.append(f"操作前 SMILES: {step['smiles_before']}")
            lines.append(f"操作后 SMILES: {step['smiles_after']}")
            
            if analyze_xyz and "xyz_analysis" in step:
                xyz = step["xyz_analysis"]
                lines.append("")
                lines.append("XYZ 分析结果:")
                if xyz["success"]:
                    lines.append(f"  状态: 成功")
                    lines.append(f"  方法: {xyz.get('method', 'unknown')}")
                    lines.append(f"  原子数: {xyz['num_atoms']}")
                    lines.append(f"  z (原子序数): {xyz['z']}")
                    lines.append(f"  pos (坐标数): {xyz['pos']}")
                    if xyz['edge_index'] is not None:
                        lines.append(f"  edge_index: {xyz['edge_index']}")
                    if xyz['edge_attr'] is not None:
                        lines.append(f"  edge_attr: {xyz['edge_attr']}")
                else:
                    lines.append(f"  状态: 失败")
                    lines.append(f"  错误: {xyz['error']}")
            
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"重建完成，共 {len(self.steps)} 步")
        
        if analyze_xyz:
            xyz_success_count = sum(1 for s in self.steps if s.get("xyz_analysis", {}).get("success", False))
            invalid_smiles_count = sum(1 for s in self.steps if s.get("smiles_after") == "invalid")
            lines.append(f"XYZ 分析成功率: {xyz_success_count}/{len(self.steps)}")
            lines.append(f"SMILES 为 'invalid' 的步骤数: {invalid_smiles_count}")
        
        lines.append("=" * 80)
        
        output = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def visualize_with_mpl(self, output_file: str = None, cols: int = 4, figsize: tuple = (20, 15), hspace: float = 0.3):
        """
        使用 matplotlib 可视化每一步的分子结构。
        
        Args:
            output_file: 输出图像文件路径
            cols: 每行显示的分子数量
            figsize: 图像大小 (width, height)
            hspace: 子图行间距（归一化值，默认0.3，越小行间距越小）
        """        
        if not self.steps:
            self.rebuild_step_by_step()
        
        # 重新执行步骤以获取每一步的分子
        self.current_mol = None
        self.atom_map = {}
        self.atom_counter = 0
        
        mols = []
        legends = []
        
        self.current_mol = rdchem.RWMol()
        
        for step_idx, operation in enumerate(self.path):
            # 执行操作
            self._execute_operation(operation)
            
            # 获取当前分子
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                mols.append(mol)
                legends.append(f"Step {step_idx + 1}\n{operation['operation']}")
            except:
                # 如果分子无效，创建一个空白分子
                mols.append(None)
                legends.append(f"Step {step_idx + 1}\n{operation['operation']}\n(invalid)")
        
        # 计算网格布局
        num_steps = len(mols)
        rows = (num_steps + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个分子
        for idx, mol in enumerate(mols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            if mol is not None:
                # 使用 RDKit 绘制分子
                img = Draw.MolToImage(mol, size=(300, 300))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Invalid\nMolecule", 
                       ha='center', va='center', fontsize=12, color='red')
            
            ax.set_title(legends[idx], fontsize=18, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_steps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout(h_pad=hspace, w_pad=0.5)
        plt.subplots_adjust(hspace=hspace)
        
        if output_file:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            # 同时生成 PDF 文件
            pdf_file = os.path.splitext(output_file)[0] + '.pdf'
            plt.savefig(pdf_file, bbox_inches='tight')


class PairMoleculeRebuilder(MoleculeRebuilder):
    """
    分子对重建器，能够根据操作路径实现 mol1 -> mcs -> mol2 的转换过程。
    支持撤销 mol1 的 non-mcs 操作得到 MCS，然后应用 mol2 的 non-mcs 操作得到 mol2。
    """
    
    def __init__(self, analysis_result: dict, types: dict = None, obverse: bool = True):
        """
        初始化分子对重建器
        
        Args:
            analysis_result: 包含 mol1, mol2, MCS 信息的字典（从 analysis_result.json 读取）
            types: 原子类型映射字典
            obverse: 是否输出 debug 信息，默认为 True
        """
        self.analysis_result = analysis_result
        self.types = types or self._get_default_types()
        self.obverse = obverse
        
        # 提取各个部分的信息
        self.mol1_section = None
        self.mol2_section = None
        self.mcs_info = None
        
        for section in analysis_result.get("combined_path", []):
            if section.get("section") == "mol1":
                self.mol1_section = section
            elif section.get("section") == "mol2":
                self.mol2_section = section
            elif section.get("section") == "mcs_info":
                self.mcs_info = section
        
        if not self.mol1_section or not self.mol2_section:
            raise ValueError("分析结果中缺少 mol1 或 mol2 部分")
        
        # 初始化父类，使用 mol1 的完整路径
        super().__init__(self.mol1_section["path"], types, obverse)
    
    def rebuild_mol1_to_mcs(self, analyze_xyz: bool = False):
        """
        从 mol1 开始，撤销 non-mcs 操作，得到 MCS
        
        Args:
            analyze_xyz: 是否分析 xyz 数据
            
        Returns:
            list: 撤销步骤列表
        """
        # 首先重建 mol1
        mol1_steps = self.rebuild_step_by_step(analyze_xyz=analyze_xyz)
        
        # 获取 mol1 的 non-mcs 路径（需要撤销的操作）
        non_mcs_path = self.mol1_section.get("path_non_mcs", [])
        
        # 按照相反的顺序撤销操作
        undo_steps = []
        for i, operation in enumerate(reversed(non_mcs_path)):
            step_info = {
                "step": len(mol1_steps) + i + 1,
                "operation": operation,
                "description": f"撤销操作: {operation['operation']} @ {operation['position']}",
                "smiles_before": self._get_current_smiles(),
                "type": "undo"
            }
            
            # 执行撤销操作
            self._undo_operation(operation)
            
            step_info["smiles_after"] = self._get_current_smiles()
            
            if analyze_xyz:
                step_info["xyz_analysis"] = self._analyze_xyz()
            
            undo_steps.append(step_info)
        
        return mol1_steps + undo_steps
    
    def _undo_operation(self, operation):
        """
        撤销单个操作
        
        Args:
            operation: 要撤销的操作字典
        """
        op_type = operation.get("operation")
        rdkit_idx = operation.get("rdkit_idx")
        
        if op_type == "add_atom":
            # 撤销添加原子：删除原子及其连接的键
            try:
                idx = int(rdkit_idx)
                if idx < self.current_mol.GetNumAtoms():
                    self.current_mol.RemoveAtom(idx)
            except (ValueError, Exception):
                pass
        
        elif op_type == "add_fragment":
            # 撤销添加片段：删除片段中的所有原子
            try:
                conn_indices = [int(idx) for idx in rdkit_idx.split(",") if idx]
                for idx in sorted(conn_indices, reverse=True):
                    if idx < self.current_mol.GetNumAtoms():
                        self.current_mol.RemoveAtom(idx)
            except (ValueError, Exception):
                pass
        
        elif op_type.startswith("form_") or op_type.startswith("add_stereo"):
            # 撤销键形成或立体化学操作：将键改为单键或移除立体化学信息
            try:
                if "-" in rdkit_idx:
                    idx1, idx2 = map(int, rdkit_idx.split("-"))
                    bond = self.current_mol.GetBondBetweenAtoms(idx1, idx2)
                    if bond:
                        if op_type.startswith("form_"):
                            bond.SetBondType(Chem.BondType.SINGLE)
                        elif op_type.startswith("add_stereo"):
                            atom = self.current_mol.GetAtomWithIdx(idx1)
                            atom.SetChiralTag(Chem.CHI_UNSPECIFIED)
            except (ValueError, Exception):
                pass
    
    def rebuild_mcs_to_mol2(self, analyze_xyz: bool = False):
        """
        从 MCS 开始，应用 mol2 的 non-mcs 操作，得到 mol2
        
        Args:
            analyze_xyz: 是否分析 xyz 数据
            
        Returns:
            list: 应用步骤列表
        """
        # 获取 mol2 的 non-mcs 路径（position 已转换为 MCS position）
        non_mcs_path = self.mol2_section.get("path_non_mcs", [])
        
        apply_steps = []
        for i, operation in enumerate(non_mcs_path):
            step_info = {
                "step": i + 1,
                "operation": operation,
                "description": f"应用操作: {operation['operation']} @ {operation['position']}",
                "smiles_before": self._get_current_smiles(),
                "type": "apply"
            }
            
            # 执行操作
            self._execute_operation(operation)
            
            step_info["smiles_after"] = self._get_current_smiles()
            
            if analyze_xyz:
                step_info["xyz_analysis"] = self._analyze_xyz()
            
            apply_steps.append(step_info)
        
        return apply_steps
    
    def rebuild_full_path(self, analyze_xyz: bool = False):
        """
        完整的重建过程：mol1 -> mcs -> mol2
        
        Args:
            analyze_xyz: 是否分析 xyz 数据
            
        Returns:
            dict: 包含三个阶段的步骤信息
        """
        # 阶段1: mol1 -> mcs
        
        # 重置状态
        self.current_mol = rdchem.RWMol()
        self.atom_map = {}
        self.atom_counter = 0
        
        mol1_to_mcs_steps = self.rebuild_mol1_to_mcs(analyze_xyz=analyze_xyz)
        
        # 保存 MCS 状态
        mcs_mol = self._get_current_smiles()
        mcs_steps_count = len(mol1_to_mcs_steps)
        
        # 阶段2: mcs -> mol2
        
        mcs_to_mol2_steps = self.rebuild_mcs_to_mol2(analyze_xyz=analyze_xyz)
        
        return {
            "mol1_to_mcs": {
                "steps": mol1_to_mcs_steps,
                "final_smiles": mcs_mol,
                "num_steps": mcs_steps_count
            },
            "mcs_to_mol2": {
                "steps": mcs_to_mol2_steps,
                "final_smiles": self._get_current_smiles(),
                "num_steps": len(mcs_to_mol2_steps)
            }
        }
    
    def visualize_full_path(self, output_file: str = None, analyze_xyz: bool = False):
        """
        可视化完整的重建路径：mol1 -> mcs -> mol2
        
        Args:
            output_file: 输出文件路径
            analyze_xyz: 是否分析 xyz 数据
        """
        result = self.rebuild_full_path(analyze_xyz=analyze_xyz)
        
        lines = []
        lines.append("=" * 80)
        lines.append("分子对完整重建路径可视化")
        lines.append("=" * 80)
        lines.append("")
        
        # 阶段1: mol1 -> mcs
        lines.append("阶段1: mol1 -> mcs")
        lines.append("-" * 80)
        lines.append(f"mol1 SMILES: {self.mol1_section['smiles']}")
        lines.append(f"MCS SMILES: {result['mol1_to_mcs']['final_smiles']}")
        lines.append(f"操作数: {result['mol1_to_mcs']['num_steps']}")
        lines.append("")
        
        for step in result['mol1_to_mcs']['steps']:
            lines.append(f"步骤 {step['step']}: {step['description']}")
            lines.append(f"  操作类型: {step['operation']['operation']}")
            lines.append(f"  位置: {step['operation']['position']}")
            lines.append(f"  SMILES: {step['smiles_after']}")
            
            if analyze_xyz and "xyz_analysis" in step:
                xyz = step["xyz_analysis"]
                if xyz["success"]:
                    lines.append(f"  原子数: {xyz['num_atoms']}")
                    lines.append(f"  z: {xyz['z']}")
                    lines.append(f"  pos: {xyz['pos']}")
            
            lines.append("")
        
        # 阶段2: mcs -> mol2
        lines.append("阶段2: mcs -> mol2")
        lines.append("-" * 80)
        lines.append(f"MCS SMILES: {result['mol1_to_mcs']['final_smiles']}")
        lines.append(f"mol2 SMILES: {self.mol2_section['smiles']}")
        lines.append(f"操作数: {result['mcs_to_mol2']['num_steps']}")
        lines.append("")
        
        for step in result['mcs_to_mol2']['steps']:
            lines.append(f"步骤 {step['step']}: {step['description']}")
            lines.append(f"  操作类型: {step['operation']['operation']}")
            lines.append(f"  位置: {step['operation']['position']}")
            lines.append(f"  SMILES: {step['smiles_after']}")
            
            if analyze_xyz and "xyz_analysis" in step:
                xyz = step["xyz_analysis"]
                if xyz["success"]:
                    lines.append(f"  原子数: {xyz['num_atoms']}")
                    lines.append(f"  z: {xyz['z']}")
                    lines.append(f"  pos: {xyz['pos']}")
            
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("重建完成")
        lines.append("=" * 80)
        
        output = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def visualize_with_mpl_full_path(self, output_file: str = None, cols: int = 4, figsize: tuple = (20, 15)):
        """
        使用 matplotlib 可视化完整的重建路径：mol1 -> mcs -> mol2
        展示每一步的中间状态
        
        Args:
            output_file: 输出图像文件路径
            cols: 每行显示的分子数量
            figsize: 图像大小 (width, height)
        """
        import matplotlib.pyplot as plt
        
        # 收集所有步骤的分子
        all_mols = []
        all_legends = []
        all_steps = []
        
        # 阶段1: mol1 -> mcs
        
        # 重置状态
        self.current_mol = rdchem.RWMol()
        self.atom_map = {}
        self.atom_counter = 0
        
        # 重建 mol1
        mol1_path = self.mol1_section["path"]
        for i, operation in enumerate(mol1_path):
            self._execute_operation(operation)
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
                all_legends.append(f"mol1 构建步骤 {i+1}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
            except Exception as e:
                all_mols.append(None)
                all_legends.append(f"mol1 构建步骤 {i+1}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
        
        # 撤销 mol1 的 non-mcs 操作
        non_mcs_path = self.mol1_section.get("path_non_mcs", [])
        for i, operation in enumerate(reversed(non_mcs_path)):
            step_num = len(mol1_path) + i + 1
            self._undo_operation(operation)
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
                all_legends.append(f"撤销步骤 {i+1}")
                all_steps.append(f"撤销: {operation['operation']} @ {operation.get('position', 'N/A')}")
            except Exception as e:
                all_mols.append(None)
                all_legends.append(f"撤销步骤 {i+1}")
                all_steps.append(f"撤销: {operation['operation']} @ {operation.get('position', 'N/A')}")
        
        # 阶段2: mcs -> mol2
        
        non_mcs_path_mol2 = self.mol2_section.get("path_non_mcs", [])
        for i, operation in enumerate(non_mcs_path_mol2):
            self._execute_operation(operation)
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
                all_legends.append(f"mol2 应用步骤 {i+1}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
            except Exception as e:
                all_mols.append(None)
                all_legends.append(f"mol2 应用步骤 {i+1}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
        
        # 创建图像
        num_mols = len(all_mols)
        rows = (num_mols + cols - 1) // cols
        
        # 调整图像大小以适应更多步骤
        figsize = (cols * 5, rows * 4)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i in range(num_mols):
            row = i // cols
            col = i % cols
            ax = axes[row][col]
            
            mol = all_mols[i]
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(400, 400))
                ax.imshow(np.array(img))
                
                # 添加步骤标记
                ax.set_xlabel(all_steps[i], fontsize=12, wrap=True)
            else:
                ax.text(0.5, 0.5, "无效分子", ha='center', va='center', fontsize=12)
                ax.set_xlabel(all_steps[i], fontsize=12, wrap=True)
            
            ax.set_title(f"步骤 {i+1}: {all_legends[i]}", fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_mols, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].axis('off')
        
        # 添加总标题
        fig.suptitle(f"分子转换完整路径可视化 (共 {num_mols} 步)\n"
                    f"mol1 -> MCS -> mol2", 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def visualize_mol1_to_mol2_path(self, output_file: str = None, cols: int = 4, figsize: tuple = (20, 15), output_pt_file: str = None):
        """
        可视化从 mol1 到 mol2 的转换路径（不包含 mol1 的构建步骤）
        只包含：mol1 -> 撤销操作得到 MCS -> 应用操作得到 mol2
        
        Args:
            output_file: 输出图像文件路径
            cols: 每行显示的分子数量
            figsize: 图像大小 (width, height)
            output_pt_file: 输出 pt 文件路径，保存中间状态的 z/pos 信息
        """
        # 收集关键步骤的分子
        all_mols = []
        all_legends = []
        all_steps = []
        
        # 收集 z/pos 信息
        xyz_data = {}
        
        # 阶段1: 获取 mol1 的最终状态
        
        # 重置状态并构建 mol1
        self.current_mol = rdchem.RWMol()
        self.atom_map = {}
        self.atom_counter = 0
        
        mol1_path = self.mol1_section["path"]
        for operation in mol1_path:
            self._execute_operation(operation)
        
        # 保存 mol1 的最终状态
        try:
            mol1_final = self.current_mol.GetMol()
            Chem.SanitizeMol(mol1_final)
            all_mols.append(mol1_final)
            all_legends.append("mol1 初始状态")
            all_steps.append("完整 mol1 分子")
            
            # 收集 z/pos 信息
            smiles = Chem.MolToSmiles(mol1_final)
            if output_pt_file:
                x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, self.types)
                xyz_data[smiles] = {
                    'z': torch.tensor(z, dtype=torch.long),
                    'pos': torch.tensor(pos, dtype=torch.float32)
                }
        except Exception as e:
            all_mols.append(None)
            all_legends.append("mol1 初始状态")
            all_steps.append("完整 mol1 分子")
        
        # 阶段2: 撤销 mol1 的 non-mcs 操作，逐步得到 MCS
        non_mcs_path = self.mol1_section.get("path_non_mcs", [])
        for i, operation in enumerate(reversed(non_mcs_path)):
            self._undo_operation(operation)
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
                all_legends.append(f"撤销步骤: {operation['operation']} @ {operation['atom']} @ {operation['position']}")
                all_steps.append(f"撤销: {operation['operation']} @ {operation.get('position', 'N/A')}")
                
                # 收集 z/pos 信息
                smiles = Chem.MolToSmiles(mol)
                if output_pt_file:
                    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, self.types)
                    xyz_data[smiles] = {
                        'z': torch.tensor(z, dtype=torch.long),
                        'pos': torch.tensor(pos, dtype=torch.float32)
                    }
            except Exception as e:
                all_mols.append(None)
                all_legends.append(f"撤销步骤: {operation['operation']} @ {operation['atom']} @ {operation['position']}")
                all_steps.append(f"撤销: {operation['operation']} @ {operation.get('position', 'N/A')}")
        
        # 阶段3: 从 MCS 应用 mol2 的 non-mcs 操作，逐步得到 mol2
        
        non_mcs_path_mol2 = self.mol2_section.get("path_non_mcs", [])
        for i, operation in enumerate(non_mcs_path_mol2):
            self._execute_operation(operation)
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
                all_legends.append(f"应用步骤: {operation['operation']} @ {operation['atom']} @ {operation['position']}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
                
                # 收集 z/pos 信息
                smiles = Chem.MolToSmiles(mol)
                if output_pt_file:
                    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, self.types)
                    xyz_data[smiles] = {
                        'z': torch.tensor(z, dtype=torch.long),
                        'pos': torch.tensor(pos, dtype=torch.float32)
                    }
            except Exception as e:
                all_mols.append(None)
                all_legends.append(f"应用步骤: {operation['operation']} @ {operation['atom']} @ {operation['position']}")
                all_steps.append(f"{operation['operation']} @ {operation.get('position', 'N/A')}")
        
        # 保存 z/pos 信息到 pt 文件
        if output_pt_file and xyz_data:
            torch.save(xyz_data, output_pt_file)
        
        # 创建图像
        num_mols = len(all_mols)
        rows = (num_mols + cols - 1) // cols
        
        # 调整图像大小
        figsize = (cols * 5, rows * 4)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i in range(num_mols):
            row = i // cols
            col = i % cols
            ax = axes[row][col]
            
            mol = all_mols[i]
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(400, 400))
                ax.imshow(np.array(img))
                
                # 添加步骤标记
                ax.set_xlabel(all_steps[i], fontsize=12, wrap=True)
            else:
                ax.text(0.5, 0.5, "无效分子", ha='center', va='center', fontsize=12)
                ax.set_xlabel(all_steps[i], fontsize=12, wrap=True)
            
            ax.set_title(f"步骤 {i+1}: {all_legends[i]}", fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_mols, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].axis('off')
        
        # 添加总标题
        fig.suptitle(f"分子转换路径可视化 (mol1 -> MCS -> mol2)\n"
                    f"共 {num_mols} 个关键状态", 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
            
