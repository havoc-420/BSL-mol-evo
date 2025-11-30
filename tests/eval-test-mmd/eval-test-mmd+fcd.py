# -*- coding: utf-8 -*-
"""
MMD 与 FCD 计算示例
依赖：
    torch、rdkit、numpy、scipy、fcd（pip install fcd）
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from fcd import get_fcd, load_ref_model   # FCD 官方实现[[1]][[2]]

# 禁用：RDKit 输出警告信息
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ==============================
# 1. 数据预处理：SMILES → 向量
# ==============================
def smiles_to_fp(smiles_list, nBits=2048):
    """
    将 SMILES 转换为 RDKit 指纹（二进制向量），用于 MMD 计算。
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        arr = np.zeros((1,))
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)

# ==============================
# 2. MMD 计算（RBF 核）
# ==============================
def rbf_kernel(x, y, sigma=1.0):
    """RBF（高斯）核矩阵"""
    dist = cdist(x, y, 'euclidean')
    return np.exp(-dist ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    """
    计算两组样本的 MMD（无偏估计）。
    x, y: shape (n_samples, n_features)
    """
    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)

    m = x.shape[0]
    n = y.shape[0]

    # 去除对角线的偏置（无偏估计）
    sum_xx = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1))
    sum_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))
    sum_xy = np.sum(K_xy) / (m * n)

    mmd = sum_xx + sum_yy - 2 * sum_xy
    return np.sqrt(max(mmd, 0.0))

# ==============================
# 3. FCD 计算（ChemNet 特征）
# ==============================
# 预加载 ChemNet（一次性），后续直接调用
_chemnet_model = load_ref_model()   # 只下载一次，内部使用 PyTorch

def compute_fcd(smiles_gen, smiles_ref):
    """
    计算生成分子集合与参考集合的 Fréchet ChemNet Distance。
    参数均为 SMILES 列表。
    """
    # fcd 包内部会把 SMILES 转为 ChemNet 向量并计算均值/协方差
    # 返回的是平方的 FCD（与论文中的公式一致），这里直接返回该值
    score = get_fcd(smiles_gen, smiles_ref, model=_chemnet_model)
    return score

# ==============================
# 4. 示例调用
# ==============================
if __name__ == "__main__":
    # 示例 SMILES（请自行替换为真实数据）
    ref_smiles = [
        "CCO", "c1ccccc1", "CC(=O)O", "NCCO", "CCN(CC)CC"
    ]
    gen_smiles = [
        "CCO", "c1ccccc1O", "CC(=O)N", "NCCN", "CCOC"
    ]

    # 1) MMD（基于指纹）
    ref_fp = smiles_to_fp(ref_smiles)
    gen_fp = smiles_to_fp(gen_smiles)
    mmd_val = compute_mmd(ref_fp, gen_fp, sigma=1.0)
    print(f"MMD (RBF, σ=1.0) = {mmd_val:.6f}")

    # 2) FCD（基于 ChemNet）
    fcd_val = compute_fcd(gen_smiles, ref_smiles)
    print(f"FCD (平方值) = {fcd_val:.6f}")

