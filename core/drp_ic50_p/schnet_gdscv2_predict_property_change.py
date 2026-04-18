import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('ℹ️ schnet seed 初始化完成')
    
set_seed()

# --- 1. 从 molecule_path_finder.py 复制的常量和解码逻辑 ---
OP_ADD_ATOM = "ADD_ATOM"
OP_REMOVE_ATOM = "REMOVE_ATOM"
OP_REPLACE_ATOM = "REPLACE_ATOM"
OP_ADD_BOND = "ADD_BOND"
OP_REMOVE_BOND = "REMOVE_BOND"
OP_CHANGE_BOND = "CHANGE_BOND"

ALL_OPS = [OP_ADD_ATOM, OP_REMOVE_ATOM, OP_REPLACE_ATOM, OP_ADD_BOND, OP_REMOVE_BOND, OP_CHANGE_BOND]
# 这些应该与训练时使用的 ATOM_TYPES 一致
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'B', 'Cl', 'Br', 'I', 'P', 'S', 'Si', 'Se']
BOND_TYPES = [1.0, 2.0, 3.0, 1.5]
MAX_ATOM_IDX = 50
OP_DIM = len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + MAX_ATOM_IDX * 2

# --- 2. 从 moleculenet_4_gdscv2.py 复制的 smiles_to_pyg 函数 ---
def smiles_to_pyg(smiles, max_tries=3):
    """
    将SMILES转换为带3D坐标的PyG图数据（z, pos, batch）
    增强容错处理，尝试多种方法生成3D坐标
    """
    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    
    # 添加氢原子
    mol = Chem.AddHs(mol)
    
    # 尝试多种方法生成3D构象
    conf_id = -1
    for attempt in range(max_tries):
        try:
            # 方法1: ETKDGv3
            if attempt == 0:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42 + attempt
                conf_id = AllChem.EmbedMolecule(mol, params)
                if conf_id >= 0:
                    break
            
            # 方法2: ETKDG
            elif attempt == 1:
                params = AllChem.ETKDG()
                params.randomSeed = 42 + attempt
                conf_id = AllChem.EmbedMolecule(mol, params)
                if conf_id >= 0:
                    break
            
            # 方法3: 使用随机初始化的坐标
            else:
                conf_id = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if conf_id >= 0:
                    break
                    
        except Exception:
            continue
    
    # 如果仍然失败，使用基本坐标
    if conf_id < 0:
        try:
            conf_id = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        except:
            # 如果完全失败，抛出异常
            raise ValueError(f"Failed to generate 3D coordinates for SMILES: {smiles}")
    
    # 尝试优化（可选，失败不影响）
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    except:
        # 如果优化失败，使用原始坐标
        pass

    # 提取坐标和原子序数
    conf = mol.GetConformer(conf_id)
        
    pos = []
    z = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        try:
            p = conf.GetAtomPosition(idx)
            pos.append([p.x, p.y, p.z])
            z.append(atom.GetAtomicNum())
        except:
            # 如果某个原子位置获取失败，抛出异常
            raise ValueError(f"Failed to get atom position for atom {idx} in SMILES: {smiles}")

    if len(pos) == 0 or len(z) == 0:
        raise ValueError(f"No atoms found in molecule: {smiles}")

    x_pos = torch.tensor(pos, dtype=torch.float32)
    x_z = torch.tensor(z, dtype=torch.long)

    return x_pos, x_z

# --- 3. 从 models_mol_v3_SchNet_4_gdscv2.py 复制并修改的模型定义 ---
try:
    from .models_lib import SchNet
except ImportError:
    from models_lib import SchNet

class GraphEditOp:
    def __init__(self, op_type: str, params):
        self.op_type = op_type
        self.params = params

    def __repr__(self):
        return f"GraphEditOp({self.op_type}, {self.params})"
    
class PropertyChangePredictor(nn.Module):
    """
    一个用于预测分子经过操作后属性变化的模型。
    这是 MainModel 的简化版本，专注于预测属性变化。
    """
    def __init__(self, node_feat_dim=11, op_dim=117, hidden_dim=256, num_gnn_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        # 1. 常量定义与映射
        self.ALL_OPS = ALL_OPS
        self.ATOM_TYPES = ATOM_TYPES
        
        # 建立 字符 -> 原子序数 的映射
        self.SYMBOL_TO_Z = {
            'H': 1,
            'B': 5,
            'C': 6,
            'N': 7,
            'O': 8,
            'F': 9,
            'P': 15,
            'S': 16,
            'Cl': 17,
            'Br': 35,
            'I': 53,
        }
        
        # 2. 核心组件
        
        # A. 状态编码器 (SchNet)
        self.state_encoder = SchNet(hidden_dim, 128, 6, out_channels=hidden_dim, cutoff=5.0)
        
        # B. 属性预测头 (Property Predictor)
        # 输入图级特征，输出标量属性 V(G)
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.delta_property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, initial_graph, ops: torch.Tensor):
        """
        预测初始分子经过操作序列后的属性变化。

        Args:
            initial_graph (torch_geometric.data.Data): 初始分子的 PyG Data 对象 (z, pos, batch)。
            ops (torch.Tensor): 操作序列，形状为 [Batch_Size, Path_Len, Op_Dim]。

        Returns:
            torch.Tensor: 预测的总属性变化量，形状为 [Batch_Size, 1]。
        """
        batch_size = ops.shape[0]
        path_len = ops.shape[1]
        ops = ops.to(torch.float32)
        device = initial_graph.pos.device

        # 1. 初始状态编码和预测
        current_node_feats = self.state_encoder(initial_graph.z, initial_graph.pos, initial_graph.batch)
        initial_graph_emb = global_mean_pool(current_node_feats[0], initial_graph.batch)
        V_prev = self.property_predictor(initial_graph_emb) # [Batch, 1]

        # 2. 准备数据结构用于逐步演化
        # 将稀疏表示转换为密集表示，便于修改
        curr_z, curr_mask = to_dense_batch(initial_graph.z, initial_graph.batch)
        curr_pos, _ = to_dense_batch(initial_graph.pos, initial_graph.batch)
        B, N_curr = curr_z.shape

        # 初始化累积变化量
        total_delta_p = torch.zeros_like(V_prev)

        # 3. 逐步应用操作并累加增量
        for t in range(path_len):
            current_op_vecs = ops[:, t, :] # [Batch, Op_Dim]

            # 解码操作 (这里需要 MoleculePathFinder 的解码逻辑)
            # 为了独立性，我们在这里实现一个简化的解码器
            batch_ops_decoded = self.decode_ops_from_onehot(current_op_vecs)

            # 物理扩充 (为 ADD_ATOM 预留空间)
            zero_z = torch.zeros(B, 1, dtype=curr_z.dtype, device=device)
            curr_z = torch.cat([curr_z, zero_z], dim=1)
            zero_pos = torch.zeros(B, 1, 3, dtype=curr_pos.dtype, device=device)
            curr_pos = torch.cat([curr_pos, zero_pos], dim=1)
            false_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            curr_mask = torch.cat([curr_mask, false_mask], dim=1)
            N_prev_idx = N_curr
            N_curr = N_curr + 1

            # 逐分子执行操作
            for b_idx in range(batch_size):
                op = batch_ops_decoded[b_idx]
                
                # --- ADD_ATOM ---
                if op.op_type == OP_ADD_ATOM:
                    symbol = op.params[0]
                    attach_to = op.params[1]
                    atomic_num = self.SYMBOL_TO_Z.get(symbol, 6)
                    
                    curr_z[b_idx, N_prev_idx] = atomic_num
                    curr_mask[b_idx, N_prev_idx] = True
                    
                    # 简单的坐标推断策略
                    if attach_to is not None and attach_to < N_prev_idx and curr_mask[b_idx, attach_to]:
                        ref_pos = curr_pos[b_idx, attach_to]
                        offset = torch.randn(3, device=device)
                        offset = F.normalize(offset, dim=0) * 1.5 
                        new_pos = ref_pos + offset
                    else:
                        valid_pos = curr_pos[b_idx][curr_mask[b_idx]]
                        if len(valid_pos) > 0:
                            new_pos = valid_pos.mean(dim=0) + torch.tensor([1.5, 0, 0], device=device)
                        else:
                            new_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
                    curr_pos[b_idx, N_prev_idx] = new_pos

                # --- REPLACE_ATOM ---
                elif op.op_type == OP_REPLACE_ATOM:
                    idx, symbol = op.params[0], op.params[1]
                    if idx < N_prev_idx and curr_mask[b_idx, idx]:
                        atomic_num = self.SYMBOL_TO_Z.get(symbol, 6)
                        curr_z[b_idx, idx] = atomic_num

                # --- REMOVE_ATOM ---
                elif op.op_type == OP_REMOVE_ATOM:
                    idx = op.params[0]
                    if idx < N_prev_idx:
                        curr_mask[b_idx, idx] = False
                        curr_z[b_idx, idx] = 0
                        curr_pos[b_idx, idx] = 0.0

            # 重新编码 (Dense -> Sparse)
            flat_z = curr_z[curr_mask]
            flat_pos = curr_pos[curr_mask]
            batch_idx_map = torch.arange(batch_size, device=device).unsqueeze(1).expand(B, N_curr)
            flat_batch = batch_idx_map[curr_mask]

            # 计算当前步的状态特征
            step_node_feats = self.state_encoder(flat_z, flat_pos, flat_batch)
            step_graph_emb = global_mean_pool(step_node_feats[0], flat_batch)
            V_curr = self.delta_property_predictor(step_graph_emb) # [Batch, 1]

            # 计算单步增量
            step_delta = V_curr - V_prev

            # 累加增量
            total_delta_p = total_delta_p + step_delta

            # 更新前一步状态
            V_prev = V_curr

        return total_delta_p

    def decode_ops_from_onehot(self, tensor):
        ops = []
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        elif hasattr(tensor, "numpy"):
            tensor = tensor.numpy()
        
        len_ops = len(ALL_OPS)
        len_atoms = len(ATOM_TYPES)
        len_bonds = len(BOND_TYPES)
        
        idx_atom_start = len_ops
        idx_bond_start = len_ops + len_atoms
        idx_pos_start = len_ops + len_atoms + len_bonds

        for vec in tensor:
            if vec.sum() < 0.5: 
                ops.append(GraphEditOp(None, None))
                continue

            op_idx = np.argmax(vec[:len_ops])
            op_type = ALL_OPS[op_idx]

            atom_vec = vec[idx_atom_start : idx_bond_start]
            atom_idx = np.argmax(atom_vec)
            atom_sym = ATOM_TYPES[atom_idx]

            bond_vec = vec[idx_bond_start : idx_pos_start]
            bond_idx = np.argmax(bond_vec)
            bond_val = BOND_TYPES[bond_idx]

            pos_vec = vec[idx_pos_start:]
            pos_indices = np.where(pos_vec > 0.5)[0].tolist()

            params = ()
            if op_type == OP_ADD_ATOM:
                attach_to = pos_indices[0] if len(pos_indices) > 0 else None
                params = (atom_sym, attach_to, float(bond_val))
            elif op_type == OP_REMOVE_ATOM:
                idx = pos_indices[0] if len(pos_indices) > 0 else 0
                params = (int(idx),)
            elif op_type == OP_REPLACE_ATOM:
                idx = pos_indices[0] if len(pos_indices) > 0 else 0
                params = (int(idx), atom_sym)
            elif op_type in [OP_ADD_BOND, OP_CHANGE_BOND]:
                if len(pos_indices) >= 2:
                    i, j = pos_indices[0], pos_indices[1]
                else:
                    i, j = 0, 1
                params = (int(i), int(j), float(bond_val))
            elif op_type == OP_REMOVE_BOND:
                if len(pos_indices) >= 2:
                    i, j = pos_indices[0], pos_indices[1]
                else:
                    i, j = 0, 1
                params = (int(i), int(j))

            ops.append(GraphEditOp(op_type, params))
        return ops


def load_model(model_path, device='cuda:0'):
    """
    加载训练好的 PropertyChangePredictor 模型。
    """
    model = PropertyChangePredictor()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def prepare_molecule_from_smiles(smiles, device='cuda:0'):
    """
    将 SMILES 转换为 PyG Data 对象。
    """
    pos, z = smiles_to_pyg(smiles)
    data = Data(z=z, pos=pos)
    # Add a dummy batch if not present (for single molecule)
    # if not hasattr(data, 'batch'):
    #     data.batch = torch.zeros(z.size(0), dtype=torch.long)
    return data.to(device)

def predict_change_for_smiles(model, initial_smiles, ops_tensor, device='cuda:0'):
    """
    为 SMILES 输入预测属性变化。
    """
    initial_molecule = prepare_molecule_from_smiles(initial_smiles, device)
    print(initial_molecule)
    return predict_change(model, initial_molecule, ops_tensor, device)

def predict_change(model, initial_molecule, ops_tensor, device='cuda:0'):
    """
    预测分子经过操作后的属性变化。
    """
    model.eval()
    
    with torch.no_grad():
        initial_molecule = initial_molecule.to(device)
        ops_tensor = ops_tensor.to(device)

        # Ensure batch dimension is correct for single molecule if needed
        # The model expects ops shape [Batch, Path_Len, Op_Dim]
        # If initial_molecule is a single graph, its batch is likely [0, 0, ...]
        # The ops_tensor should have batch_size matching initial_molecule
        # This logic assumes ops_tensor is already shaped for the correct batch
        # If ops_tensor is for 1 molecule [1, Path_Len, Op_Dim], it should match
        # a single molecule batched as [1, ...]
        predicted_change = model(initial_molecule, ops_tensor)
    return predicted_change

def encode_operation_to_tensor(operation_details):
    """
    将操作详情编码为 one-hot 张量
    
    Args:
        operation_details: 操作详情字典，包含操作类型和相关参数
        
    Returns:
        one-hot 编码的操作张量，形状为 [1, 1, OP_DIM]
    """
    op_type = operation_details.get("type", "unknown")
    op_params = operation_details.get("params", {})
    
    # 创建 one-hot 向量
    op_vec = torch.zeros(OP_DIM, dtype=torch.float32)
    
    # 编码操作类型
    if op_type in ALL_OPS:
        op_idx = ALL_OPS.index(op_type)
        op_vec[op_idx] = 1.0
    
    # 编码原子类型
    atom_symbol = op_params.get("atom_symbol", "")
    if atom_symbol in ATOM_TYPES:
        atom_idx = ATOM_TYPES.index(atom_symbol)
        op_vec[len(ALL_OPS) + atom_idx] = 1.0
    
    # 编码键类型
    bond_type = op_params.get("bond_type", 1.0)
    if bond_type in BOND_TYPES:
        bond_idx = BOND_TYPES.index(bond_type)
        op_vec[len(ALL_OPS) + len(ATOM_TYPES) + bond_idx] = 1.0
    
    # 编码原子索引
    atom_idx = op_params.get("atom_idx", -1)
    if atom_idx >= 0 and atom_idx < MAX_ATOM_IDX:
        op_vec[len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + atom_idx] = 1.0
    
    # 编码目标原子索引（用于 ADD_BOND, REMOVE_BOND, CHANGE_BOND）
    target_atom_idx = op_params.get("atom2_idx", op_params.get("target_atom_idx", -1))
    if target_atom_idx >= 0 and target_atom_idx < MAX_ATOM_IDX:
        op_vec[len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + MAX_ATOM_IDX + target_atom_idx] = 1.0
    
    # 返回形状为 [1, 1, OP_DIM] 的张量
    return op_vec.unsqueeze(0).unsqueeze(0)

def predict_batch(model, valid_operations, current_smiles, device='cuda:0'):
    """
    批量预测分子属性变化
    
    Args:
        model: PropertyChangePredictor 模型
        valid_operations: 有效操作列表，每个元素是 (operation, new_smiles) 元组
                            operation: 操作详情字典，包含操作类型和相关参数
                            new_smiles: 目标分子SMILES（当前版本未使用，保留以保持接口兼容）
        current_smiles: 当前分子SMILES（用于准备初始分子图数据）
        device: 设备
        
    Returns:
        批量预测的属性变化值列表
    """
    if not valid_operations:
        return []
        
    # 准备批量数据
    ops_tensors = []
    
    for operation, new_smiles in valid_operations:
        try:
            # 将操作详情转换为操作张量
            ops_tensor = encode_operation_to_tensor(operation)
            ops_tensors.append(ops_tensor)
            
        except Exception as e:
            print(f"处理操作时出错: operation={operation}, error={e}")
            continue
    
    # 如果没有有效的数据，返回空列表
    if not ops_tensors:
        return []
    
    # 准备初始分子图数据（所有操作基于同一个初始分子）
    initial_graph = prepare_molecule_from_smiles(current_smiles, device)
    
    # 批量处理操作张量
    ops_batch = torch.cat(ops_tensors, dim=0).to(device)
    
    # 复制初始分子图以匹配操作批量大小
    num_ops = len(ops_tensors)
    z_batch = initial_graph.z.repeat(num_ops)
    pos_batch = initial_graph.pos.repeat(num_ops, 1)
    batch_indices = torch.arange(num_ops, device=device).repeat_interleave(len(initial_graph.z))
    
    # 创建批量图数据
    batched_graph = Data(z=z_batch, pos=pos_batch, batch=batch_indices)
    
    # 执行批量预测
    # 模型现在支持真正的批量处理，可以一次性预测所有操作
    with torch.no_grad():
        predicted_changes = model(batched_graph, ops_batch).detach().cpu()

    if predicted_changes.ndim == 0:
        return [float(predicted_changes.item())]

    if predicted_changes.ndim == 1:
        return [float(x) for x in predicted_changes.tolist()]

    if predicted_changes.ndim == 2 and predicted_changes.shape[1] == 1:
        return [float(x) for x in predicted_changes[:, 0].tolist()]

    raise ValueError(f"不支持的预测张量形状: {tuple(predicted_changes.shape)}")

# --- 4. 主函数示例 ---
def main():
    # --- 用户配置 ---
    model_path = "./SchNet_909729_checkpoints/0.9_N4_20260106_191804/best_model.pth"  # 模型文件路径
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = 'cpu'

    initial_smiles = "CCO" # 乙醇
    path_len = 1
    # 创建一个示例操作向量 [1, path_len, OP_DIM]
    # 这需要根据实际操作来构造，这里只是一个形状正确的零向量
    ops_tensor = torch.zeros((1, path_len, OP_DIM), dtype=torch.float32)

    ops_tensor[0, 0, 0] = 1.0 # Op Type: ADD_ATOM
    ops_tensor[0, 0, len(ALL_OPS) + ATOM_TYPES.index('C')] = 1.0 # Atom Type: C
    ops_tensor[0, 0, len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + 0] = 1.0 # Attach To: atom 0 (if applicable)
    
    print(ops_tensor)

    # --- 执行预测 ---
    model = load_model(model_path, device=device)
    predicted_change = predict_change_for_smiles(model, initial_smiles, ops_tensor, device=device)

    print(f"Predicted property change for '{initial_smiles}' with given ops: {predicted_change.item():.6f}")
    
    # --- 批量预测示例 ---
    print("\n" + "="*50)
    print("批量预测示例")
    print("="*50)
    
    # 定义多个操作
    valid_operations = [
        (
            {"type": "ADD_ATOM", "params": {"atom_symbol": "C", "atom_idx": 0}},
            "CCCO"  # 添加碳原子后的SMILES（示例）
        ),
        (
            {"type": "ADD_ATOM", "params": {"atom_symbol": "N", "atom_idx": 1}},
            "CCN"  # 添加氮原子后的SMILES（示例）
        ),
        (
            {"type": "ADD_ATOM", "params": {"atom_symbol": "O", "atom_idx": 2}},
            "CCCO"  # 添加氧原子后的SMILES（示例）
        ),
    ]
    
    # 执行批量预测
    batch_predictions = predict_batch(model, valid_operations, initial_smiles, device=device)
    
    # 打印批量预测结果
    print(f"\n批量预测结果（共 {len(batch_predictions)} 个操作）:")
    for i, (operation, new_smiles) in enumerate(valid_operations):
        if i < len(batch_predictions):
            print(f"  操作 {i+1}: {operation['type']}")
            print(f"    参数: {operation['params']}")
            print(f"    预测属性变化: {batch_predictions[i]:.6f}")
            print(f"    目标SMILES: {new_smiles}")
            print()


if __name__ == "__main__":
    main()