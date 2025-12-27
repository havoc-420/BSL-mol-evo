import os
import pandas as pd
from rdkit import Chem

def process_smiles_file(file_path):
    """
    处理包含SMILES的CSV文件，将显式氢[H]转换为隐式氢
    
    参数:
    file_path (str): CSV文件路径
    """
    # 备份原始文件
    backup_path = file_path.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_path):
        os.rename(file_path, backup_path)
        print(f"已创建备份文件: {backup_path}")
    
    # 读取备份文件
    df = pd.read_csv(backup_path)
    
    # 处理每一行的SMILES
    def remove_explicit_hydrogens(smiles):
        try:
            # 使用RDKit解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 将显式氢转换为隐式氢
                mol = Chem.RemoveHs(mol)
                # 生成新的SMILES
                new_smiles = Chem.MolToSmiles(mol)
                return new_smiles
            else:
                print(f"警告: 无法解析SMILES: {smiles}")
                return smiles
        except Exception as e:
            print(f"处理SMILES时出错 {smiles}: {e}")
            return smiles
    
    # 应用处理函数到SMILES列
    df['smiles'] = df['smiles'].apply(remove_explicit_hydrogens)
    
    # 保存处理后的文件
    df.to_csv(file_path, index=False)
    print(f"已处理并保存文件: {file_path}")

def main():
    # 处理所有1-9的文件
    base_dir = "."
    for i in range(1, 10):
        file_name = f"qm9_smiles_heavy_{i}_atoms.csv"
        file_path = os.path.join(base_dir, file_name)
        
        if os.path.exists(file_path):
            print(f"正在处理文件: {file_path}")
            process_smiles_file(file_path)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    main()