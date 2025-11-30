"""
QM9OptimizationPairs 类
用于从 QM9 数据集提取优化对
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
import warnings
import json
warnings.filterwarnings('ignore')

class QM9OptimizationPairs:
    def __init__(self, data_path=None, qm9_root=None, csv_path=None, indices_path=None):
        """
        初始化 QM9 优化对提取器
        
        参数:
        - data_path: QM9 数据文件路径，如果为None则使用示例数据
        - qm9_root: QM9 数据集根目录，用于直接从 PyTorch Geometric 数据集加载
        - csv_path: CSV文件路径，包含SMILES和相关属性
        - indices_path: 索引文件路径，用于过滤测试集
        """
        if csv_path:
            self.df = pd.read_csv(csv_path)
        elif data_path:
            self.df = pd.read_csv(data_path)
        elif qm9_root:
            self.df = self._load_qm9_dataset(qm9_root)
        else:
            # 创建示例 QM9 数据（实际使用时请替换为真实的 QM9 数据）
            self.df = self._create_sample_data()
        
        # 如果提供了索引文件路径，则加载测试集索引并过滤数据
        if indices_path:
            self._filter_test_set(indices_path)
        
        # 预处理分子
        self._preprocess_molecules()
    
    def _filter_test_set(self, indices_path):
        """
        根据索引文件过滤测试集数据
        
        参数:
        - indices_path: 索引文件路径
        """
        try:
            with open(indices_path, 'r') as f:
                indices_data = json.load(f)
            
            test_indices = indices_data.get('test_indices', [])
            if not test_indices:
                print("警告: 索引文件中未找到测试集索引")
                return
            
            # 过滤数据框，只保留测试集索引对应的数据
            self.df = self.df.iloc[test_indices].reset_index(drop=True)
            print(f"已过滤数据集，仅保留测试集 {len(test_indices)} 个样本")
            
        except Exception as e:
            print(f"加载索引文件时出错: {e}")

    def _load_qm9_dataset(self, root_dir):
        """
        从 PyTorch Geometric QM9 数据集加载数据
        
        参数:
        - root_dir: QM9 数据集根目录路径
        
        返回:
        - 包含 SMILES 和相关属性的 DataFrame
        """
        try:
            from torch_geometric.datasets import QM9
            import torch
            
            print(f"正在加载 QM9 数据集从 {root_dir}...")
            dataset = QM9(root=root_dir)
            print(f"数据集加载完成，共 {len(dataset)} 个分子")
            
            # QM9 属性名称
            target_names = [
                'mu',           # 偶极矩
                'alpha',        # 各向同性极化率
                'homo',         # HOMO 能量
                'lumo',         # LUMO 能量
                'gap',          # HOMO-LUMO 间隙
                'r2',           # 电子空间范围
                'zpve',         # 零点振动能量
                'U0',           # 0K 内能
                'U',            # 298.15K 内能
                'H',            # 298.15K 焓
                'G',            # 298.15K 自由能
                'Cv',           # 298.15K 热容
                'U0_atom',      # 原子化能 (0K)
                'U_atom',       # 原子化能 (298.15K)
                'H_atom',       # 原子化焓 (298.15K)
                'G_atom',       # 原子化自由能 (298.15K)
                'A',            # 旋转常数 A
                'B',            # 旋转常数 B
                'C'             # 旋转常数 C
            ]
            
            # 尝试获取SMILES列表
            smiles_list = None
            try:
                # 新版本QM9数据集将SMILES存储在smiles属性中
                smiles_list = dataset.smiles
            except AttributeError:
                # 旧版本或其他情况需要手动转换
                print("无法直接获取SMILES，将尝试从分子图重构...")
                pass
            
            # 收集所有分子数据
            data_list = []
            success_count = 0
            fail_count = 0
            max_success = 1000  # 限制成功处理的分子数量
            
            for i, data in enumerate(dataset):
                # 获取SMILES
                smiles = None
                if smiles_list is not None and i < len(smiles_list):
                    smiles = smiles_list[i]
                else:
                    # 尝试从分子图重构SMILES
                    try:
                        smiles = self._mol_from_graph(data)
                    except Exception as e:
                        print(f"无法为分子 {i} 生成SMILES: {e}")
                        fail_count += 1
                        continue
                
                if smiles is None:
                    fail_count += 1
                    continue
                else:
                    success_count += 1
                
                # 提取所有 19 个目标属性
                targets = {}
                for j, name in enumerate(target_names):
                    value = 0.0
                    if hasattr(data, 'y') and data.y is not None and data.y.shape[1] > j:
                        value = data.y[0, j]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                    targets[name] = value
                
                molecule_data = {
                    'index': i,
                    'smiles': smiles,
                    **targets
                }
                
                data_list.append(molecule_data)
                
                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(dataset)} 个分子 (成功: {success_count}, 失败: {fail_count})")
                
                # 达到最大成功数后停止
                if success_count >= max_success:
                    print(f"达到最大成功处理数 {max_success}，停止处理")
                    break
            
            print(f"总共处理了 {len(data_list)} 个分子 (成功: {success_count}, 失败: {fail_count})")
            return pd.DataFrame(data_list)
            
        except ImportError as e:
            print(f"导入错误: {e}")
            print("请确保已安装 PyTorch Geometric:")
            print("pip install torch torch-geometric")
            raise e
        except Exception as e:
            print(f"加载 QM9 数据集时出错: {e}")
            raise e
    
    def _mol_from_graph(self, data):
        """
        从图数据重构分子SMILES（备用方法）
        直接使用QM9数据集提供的SMILES
        """
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            
            # 禁用RDKit警告
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
            
            # 直接使用QM9数据集提供的SMILES
            if hasattr(data, 'smiles') and data.smiles:
                mol = Chem.MolFromSmiles(data.smiles)
                if mol is not None:
                    # 验证分子并返回规范化的SMILES
                    try:
                        Chem.SanitizeMol(mol)
                        # 返回隐藏氢原子的SMILES
                        smiles = Chem.MolToSmiles(mol, allHsExplicit=False)
                        return smiles
                    except Exception:
                        # 如果规范化失败，返回原始SMILES
                        return data.smiles
            return None
        except Exception as e:
            print(f"从图重构分子时出错: {e}")
            return None
    
    def _create_sample_data(self):
        """创建示例 QM9 数据"""
        print("使用示例数据，实际使用时请提供真实的 QM9 数据集")
        
        # 示例分子 SMILES（来自类 QM9 的小分子）
        sample_smiles = [
            'CCO', 'CCOC', 'CC=O', 'CC#N', 'C1CC1', 'C1COC1', 'C1CCNC1',
            'CC(C)O', 'CC(=O)O', 'C1CCCC1', 'C1CCC1', 'C1C=CC=C1',
            'C1=CN=CN1', 'C1=COC=C1', 'CC(C)=O', 'CCOCC', 'CCNCC',
            'C1CCCCC1', 'C1CC2CCCC2C1', 'C1CC1C#N'
        ]
        
        # 生成随机的电子属性（单位：eV）
        np.random.seed(42)
        n_molecules = len(sample_smiles)
        
        data = {
            'smiles': sample_smiles,
            'homo': np.random.uniform(-8, -5, n_molecules),  # HOMO 能量
            'lumo': np.random.uniform(-3, 0, n_molecules),   # LUMO 能量
        }
        
        df = pd.DataFrame(data)
        df['gap'] = df['lumo'] - df['homo']  # HOMO-LUMO gap
        
        return df
    
    def _preprocess_molecules(self):
        """预处理分子，计算指纹"""
        self.valid_indices = []
        self.molecules = []
        self.fingerprints = []
        
        # 禁用RDKit警告
        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        
        # 添加进度条
        from tqdm import tqdm
        
        for idx, smiles in enumerate(tqdm(self.df['smiles'], desc="处理分子")):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 过滤过于简单的分子（例如只含有氢原子或原子数少于3的分子）
                if mol.GetNumAtoms() < 3:
                    continue
                    
                # 检查是否有重原子（非氢原子）
                heavy_atom_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() > 1:
                        heavy_atom_count += 1
                        
                if heavy_atom_count < 2:
                    continue
                
                self.valid_indices.append(idx)
                self.molecules.append(mol)
                # 计算 Morgan 指纹
                try:
                    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
                    morgan_generator = GetMorganGenerator(radius=2)
                    fp = morgan_generator.GetFingerprint(mol)
                except ImportError:
                    # 回退到旧方法
                    fp = AllChem.GetMorganFingerprint(mol, 2)
                self.fingerprints.append(fp)
        
        print(f"成功处理 {len(self.molecules)} 个有效分子")
    
    def _calculate_fingerprint(self, mol):
        """
        计算分子指纹
        """
        try:
            from rdkit import RDLogger
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            
            # 禁用RDKit警告
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
            
            # 使用MorganGenerator替代已弃用的方法
            morgan_generator = GetMorganGenerator(radius=2, fpSize=2048)
            fingerprint = morgan_generator.GetFingerprint(mol)
            return fingerprint
        except ImportError:
            # 回退到旧方法（如果有旧版本RDKit）
            from rdkit import RDLogger
            from rdkit.Chem import AllChem
            
            # 禁用RDKit警告
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
            
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    def calculate_similarity(self, idx1, idx2):
        """计算两个分子之间的相似度"""
        return DataStructs.TanimotoSimilarity(
            self.fingerprints[idx1], 
            self.fingerprints[idx2]
        )
    
    def create_optimization_pairs(self, target_attr, 
                                improvement_direction='higher',
                                similarity_threshold=0.6,
                                min_improvement=0.05,
                                max_pairs_per_seed=5,
                                total_max_pairs=5000):
        """
        构建分子优化对
        
        参数:
        - target_attr: 目标属性 ('homo', 'lumo', 'gap')
        - improvement_direction: 'higher' 或 'lower'
        - similarity_threshold: 分子相似度阈值
        - min_improvement: 最小改进幅度
        - max_pairs_per_seed: 每个种子分子的最大对数
        - total_max_pairs: 总最大对数
        """
        
        pairs = []
        n_valid = len(self.valid_indices)
        
        print(f"开始构建 {target_attr.upper()} 优化对...")
        print(f"优化方向: {improvement_direction}")
        
        # 添加进度条
        from tqdm import tqdm
        pbar = tqdm(total=min(n_valid, total_max_pairs//max_pairs_per_seed), desc=f"构建{target_attr.upper()}对")
        
        for i in range(n_valid):
            seed_idx = self.valid_indices[i]
            seed_mol = self.molecules[i]
            seed_value = self.df.iloc[seed_idx][target_attr]
            
            candidate_pairs = []
            
            for j in range(n_valid):
                if i == j:
                    continue
                    
                candidate_idx = self.valid_indices[j]
                candidate_value = self.df.iloc[candidate_idx][target_attr]
                
                # 检查属性改进
                if improvement_direction == 'higher':
                    improvement = candidate_value > seed_value
                    improvement_amount = candidate_value - seed_value
                else:  # 'lower'
                    improvement = candidate_value < seed_value
                    improvement_amount = seed_value - candidate_value
                
                if improvement and improvement_amount >= min_improvement:
                    # 计算相似度
                    similarity = self.calculate_similarity(i, j)
                    
                    if similarity >= similarity_threshold:
                        candidate_pairs.append({
                            'j_index': j,
                            'improvement_amount': improvement_amount,
                            'similarity': similarity
                        })
            
            # 为当前种子分子选择最佳候选
            if candidate_pairs:
                # 按改进幅度排序
                candidate_pairs.sort(key=lambda x: x['improvement_amount'], reverse=True)
                
                # 选择前几个候选
                selected = candidate_pairs[:max_pairs_per_seed]
                
                for candidate in selected:
                    j = candidate['j_index']
                    candidate_idx = self.valid_indices[j]
                    
                    pair_data = {
                        'A_smiles': self.df.iloc[seed_idx]['smiles'],
                        'B_smiles': self.df.iloc[candidate_idx]['smiles'],
                        f'A_{target_attr}': seed_value,
                        f'B_{target_attr}': self.df.iloc[candidate_idx][target_attr],
                        'improvement': self.df.iloc[candidate_idx][target_attr] - seed_value,
                        'improvement_abs': candidate['improvement_amount'],
                        'similarity': candidate['similarity'],
                        'A_homo': self.df.iloc[seed_idx]['homo'],
                        'A_lumo': self.df.iloc[seed_idx]['lumo'],
                        'A_gap': self.df.iloc[seed_idx]['gap'],
                        'B_homo': self.df.iloc[candidate_idx]['homo'],
                        'B_lumo': self.df.iloc[candidate_idx]['lumo'],
                        'B_gap': self.df.iloc[candidate_idx]['gap'],
                    }
                    
                    pairs.append(pair_data)
                    
                    if len(pairs) >= total_max_pairs:
                        break
            
            pbar.update(1)
            
            if len(pairs) >= total_max_pairs:
                break
        
        pbar.close()
        
        pairs_df = pd.DataFrame(pairs)
        
        if len(pairs_df) > 0:
            # 去重并排序
            pairs_df = pairs_df.drop_duplicates(subset=['A_smiles', 'B_smiles'])
            pairs_df = pairs_df.sort_values('improvement_abs', ascending=False)
        
        print(f"生成 {len(pairs_df)} 个 {target_attr.upper()} 优化对")
        return pairs_df
    
    def create_all_optimization_pairs(self, output_prefix='qm9_optimization'):
        """为所有三个属性创建优化对"""
        
        # HOMO 优化对：提高 HOMO 能量
        homo_pairs = self.create_optimization_pairs(
            'homo', 
            improvement_direction='higher',
            similarity_threshold=0.6,
            min_improvement=0.05
        )
        
        # LUMO 优化对：降低 LUMO 能量
        lumo_pairs = self.create_optimization_pairs(
            'lumo',
            improvement_direction='lower', 
            similarity_threshold=0.6,
            min_improvement=0.05
        )
        
        # Gap 优化对：减小 HOMO-LUMO gap
        gap_pairs = self.create_optimization_pairs(
            'gap',
            improvement_direction='lower',
            similarity_threshold=0.6, 
            min_improvement=0.05
        )
        
        # 保存结果
        if len(homo_pairs) > 0:
            homo_pairs.to_csv(f'{output_prefix}_homo_pairs.csv', index=False)
            print(f"保存 HOMO 优化对到: {output_prefix}_homo_pairs.csv")
        
        if len(lumo_pairs) > 0:
            lumo_pairs.to_csv(f'{output_prefix}_lumo_pairs.csv', index=False)
            print(f"保存 LUMO 优化对到: {output_prefix}_lumo_pairs.csv")
        
        if len(gap_pairs) > 0:
            gap_pairs.to_csv(f'{output_prefix}_gap_pairs.csv', index=False)
            print(f"保存 GAP 优化对到: {output_prefix}_gap_pairs.csv")
        
        return homo_pairs, lumo_pairs, gap_pairs
    
    def analyze_pairs(self, pairs_df, property_name):
        """分析优化对的质量"""
        if len(pairs_df) == 0:
            print(f"没有找到 {property_name} 优化对")
            return
        
        print(f"\n=== {property_name.upper()} 优化对分析 ===")
        print(f"总对数: {len(pairs_df)}")
        print(f"平均改进: {pairs_df['improvement_abs'].mean():.4f} eV")
        print(f"平均相似度: {pairs_df['similarity'].mean():.4f}")
        print(f"改进范围: [{pairs_df['improvement_abs'].min():.4f}, {pairs_df['improvement_abs'].max():.4f}] eV")
        print(f"相似度范围: [{pairs_df['similarity'].min():.4f}, {pairs_df['similarity'].max():.4f}]")
        
        # 显示前几个示例
        print("\n前5个优化对示例:")
        for i in range(min(5, len(pairs_df))):
            pair = pairs_df.iloc[i]
            print(f"  {i+1}. A: {pair['A_smiles']} -> B: {pair['B_smiles']}")
            print(f"     改进: {pair['improvement_abs']:.3f} eV, 相似度: {pair['similarity']:.3f}")

def main():
    """主函数"""
    # 初始化提取器
    # 如果有真实的 QM9 数据文件，请在这里指定路径
    # extractor = QM9OptimizationPairs(data_path='your_qm9_data.csv')
    
    # 从 PyTorch Geometric QM9 数据集加载数据
    # extractor = QM9OptimizationPairs(qm9_root='raw-data/QM9')
    
    # 从CSV文件加载数据，并使用索引文件过滤测试集
    extractor = QM9OptimizationPairs(
        csv_path='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json',
        indices_path='mol_evo/dataset/data/dataset_indices/indices_20251122_213847_seed42.json'
    )
    
    # 为所有属性创建优化对
    homo_pairs, lumo_pairs, gap_pairs = extractor.create_all_optimization_pairs()
    
    # 分析结果
    extractor.analyze_pairs(homo_pairs, 'HOMO')
    extractor.analyze_pairs(lumo_pairs, 'LUMO') 
    extractor.analyze_pairs(gap_pairs, 'GAP')

if __name__ == "__main__":
    main()