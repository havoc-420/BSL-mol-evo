from collections import deque
from mol_evo.core.evolver import MoleculeEvolver
from mol_evo.core.similarity import calculate_path_edit_distance, calculate_evolutionary_similarity


# --- 3. 验证 ---
# if __name__ == '__main__':
    # from tqdm import tqdm
    # df = pd.read_csv("/home/data1/lk/project/mol_tree_v2/similarity/full_cleaned_pcqm4mv2.csv")
    # smiles_library = df['smiles']

    # print(f"正在为 {len(smiles_library)} 个分子生成进化路径 (CPU密集型)...")
    
    # t_start_path = time.time()
    # 这一步仍然是CPU密集型的，但在您的设想中已经完成
    
    # 真实场景中，可以用joblib或multiprocessing来并行化这一步
    # all_paths = [MoleculeEvolver(s).generate_path() for s in tqdm(smiles_library)]
    # print(f"路径生成总耗时: {time.time() - t_start_path:.4f} 秒")
    # print("\n--- 启动快速相似度批量计算 ---")
    
    # # 1. 初始化计算器
    # calculator = FastPathSimilarityCalculator(use_tfidf=True)
    
    # # 2. "训练"它，即让它学习并转换我们的路径数据
    # # --- 核心修复：使用正确的变量名 all_paths ---
    # calculator.fit(all_paths)

    # # 3. 一键计算最终的相似度矩阵
    # similarity_matrix = calculator.calculate_similarity_matrix()

    # print("\n生成的进化路径（用于参考）:")
    # for i, p in enumerate(all_paths):
    #     print(f"  分子 {i} ('{smiles_library[i]}'): {p}")
        
    # print("\n最终的相似度矩阵 (Cosine Similarity):")
    # np.set_printoptions(precision=3, suppress=True)
    # print(similarity_matrix)

    # print("\n--- 结果分析 ---")
    # print(f"相似度(CCC, CCCC) = {similarity_matrix[0, 1]:.3f} (预期高)")
    # print(f"相似度(CCC, C1CC1) = {similarity_matrix[0, 2]:.3f} (预期较高)")
    # print(f"相似度(CCC, CCO) = {similarity_matrix[0, 4]:.3f} (预期较高)")
    # print(f"相似度(CCC, c1ccccc1) = {similarity_matrix[0, 5]:.3f} (预期低)")

# 添加简单的测试函数
def test_molecule_evolver():
    """测试分子进化路径生成器"""
    # 测试简单的分子
    test_smiles = [
        "CCC",           # 丙烷
        "CCCC",          # 丁烷
        "CCO",           # 乙醇
        "c1ccccc1",      # 苯
        "C1CC1",         # 环丙烷
    ]
    
    print("=== 分子进化路径生成器测试 ===\n")
    
    # 生成每个分子的进化路径
    for smiles in test_smiles:
        try:
            print(f"测试 SMILES: {smiles}")
            evolver = MoleculeEvolver(smiles)
            path = evolver.generate_path()
            print(f"进化路径: {path}")
            print("-" * 50)
        except Exception as e:
            print(f"处理 {smiles} 时出错: {e}")
            print("-" * 50)
    
    # 测试相似度计算
    print("\n=== 相似度计算测试 ===\n")
    smiles_pairs = [
        ("CCC", "CCCC"),        # 丙烷 vs 丁烷
        ("CCC", "CCO"),         # 丙烷 vs 乙醇
        ("c1ccccc1", "CCC"),    # 苯 vs 丙烷
        ("C1CC1", "CCC"),       # 环丙烷 vs 丙烷
    ]
    
    for smiles1, smiles2 in smiles_pairs:
        try:
            print(f"比较: {smiles1} vs {smiles2}")
            distance, path1, path2 = calculate_evolutionary_similarity(smiles1, smiles2, verbose=True)
            print("-" * 70)
        except Exception as e:
            print(f"比较 {smiles1} vs {smiles2} 时出错: {e}")
            print("-" * 70)

if __name__ == "__main__":
    test_molecule_evolver()