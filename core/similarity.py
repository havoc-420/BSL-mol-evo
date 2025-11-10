import numpy as np
from .evolver import MoleculeEvolverAnalysis


def calculate_path_edit_distance(path1, path2):
    # TODO 需要改进
    len1, len2 = len(path1), len(path2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1): dp[i][0] = i
    for j in range(len2 + 1): dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if path1[i-1] == path2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len1][len2]


def calculate_evolutionary_similarity(smiles1: str, smiles2: str, verbose=False):
    """最终的、对用户友好的顶层调用函数。"""
    try:
        path1 = MoleculeEvolverAnalysis(smiles1).generate_path()
        path2 = MoleculeEvolverAnalysis(smiles2).generate_path()
    except ValueError as e:
        print(f"错误: {e}")
        return 0.0, [], []

    distance = calculate_path_edit_distance(path1, path2)
    # similarity = 1.0 / (1.0 + distance)
    
    if verbose:
        print(f"--- 比较 '{smiles1}' vs '{smiles2}' ---")
        print(f"路径 1 (长度 {len(path1)}): {path1}")
        print(f"路径 2 (长度 {len(path2)}): {path2}")
        print(f"路径编辑距离: {distance}")
        # print(f"最终相似度 (1 / (1 + d)): {similarity:.3f}\n")
        
    return distance, path1, path2