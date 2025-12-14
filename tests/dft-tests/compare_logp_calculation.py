from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Crippen

# 初始化TDC的Oracle
try:
    tdc_logp_oracle = Oracle(name='logp')
    print("TDC Oracle初始化成功")
except Exception as e:
    print(f"TDC Oracle初始化失败: {e}")
    tdc_logp_oracle = None

# 测试用的SMILES字符串
test_smiles = [
    'NC(=O)C#CC=O',  # 从结果文件中选择的一个分子
    'O=CC#CC1=NN1O',  # 从结果文件中选择的另一个分子
    'CCO',  # 乙醇，已知logP约为-0.77
    'C1=CC=CC=C1',  # 苯，已知logP约为2.13
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林，已知logP约为1.2
    'CC1=CC=C(C=C1)C(=O)O',  # 苯甲酸，已知logP约为2.1
]

print("\n比较TDC Oracle和RDKit Crippen.MolLogP的logP计算结果：")
print("=" * 80)
print(f"{'SMILES':<30} {'TDC Oracle logP':<20} {'RDKit logP':<20} {'差异':<10}")
print("=" * 80)

for smiles in test_smiles:
    # 使用TDC的Oracle计算logP
    tdc_logp = None
    if tdc_logp_oracle is not None:
        try:
            tdc_logp = tdc_logp_oracle(smiles)
        except Exception as e:
            print(f"TDC计算logP时出错 ({smiles}): {e}")
    
    # 使用RDKit的Crippen.MolLogP计算logP
    rdkit_logp = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rdkit_logp = Crippen.MolLogP(mol)
        else:
            print(f"RDKit无法解析SMILES字符串 ({smiles})")
    except Exception as e:
        print(f"RDKit计算logP时出错 ({smiles}): {e}")
    
    # 计算差异
    diff = None
    if tdc_logp is not None and rdkit_logp is not None:
        diff = abs(tdc_logp - rdkit_logp)
    
    # 输出结果
    tdc_str = f"{tdc_logp:.4f}" if tdc_logp is not None else "N/A"
    rdkit_str = f"{rdkit_logp:.4f}" if rdkit_logp is not None else "N/A"
    diff_str = f"{diff:.4f}" if diff is not None else "N/A"
    print(f"{smiles:<30} {tdc_str:<20} {rdkit_str:<20} {diff_str:<10}")

print("\n" + "=" * 80)
print("测试完成")
