import numpy as np
import time
from ase import Atoms
from pyscf import gto, dft, scf, lib
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import sys

# 设置PySCF的日志级别为ERROR，减少输出信息
lib.logger.level = lib.logger.ERROR

# 使用 tqdm 的 write 方法替代 logging
def log(message):
    tqdm.write(message)

# Hartree到eV的转换常数
HARTREE_TO_EV = 27.211324570273


def smiles_to_coordinates(smiles, quiet=False):
    """
    将SMILES转换为3D坐标
    
    Args:
        smiles (str): SMILES字符串
        quiet (bool): 是否静默模式，不输出日志
        
    Returns:
        tuple: (原子符号列表, 原子坐标数组, 时间统计字典) 或 None（如果转换失败）
    """
    time_stats = {}
    
    try:
        # 创建分子对象
        start_time = time.time()
        mol = Chem.MolFromSmiles(smiles)
        time_stats['mol_creation'] = time.time() - start_time
        
        if mol is None:
            if not quiet:
                log(f"Failed to create molecule from SMILES: {smiles}")
            return None
            
        # 添加氢原子
        start_time = time.time()
        mol = Chem.AddHs(mol)
        time_stats['add_hydrogens'] = time.time() - start_time
        
        # 生成3D构象
        start_time = time.time()
        AllChem.EmbedMolecule(mol, randomSeed=42)
        time_stats['embed_molecule'] = time.time() - start_time
        
        start_time = time.time()
        AllChem.MMFFOptimizeMolecule(mol)
        time_stats['optimize_molecule'] = time.time() - start_time
        
        # 获取原子坐标
        start_time = time.time()
        conformer = mol.GetConformer()
        atoms = mol.GetAtoms()
        
        atomic_symbols = []
        coordinates = []
        
        for i, atom in enumerate(atoms):
            pos = conformer.GetAtomPosition(i)
            atomic_symbols.append(atom.GetSymbol())
            coordinates.append([pos.x, pos.y, pos.z])
        
        time_stats['get_coordinates'] = time.time() - start_time
        time_stats['total'] = sum(time_stats.values())
        
        return atomic_symbols, np.array(coordinates), time_stats
        
    except Exception as e:
        if not quiet:
            log(f"Error converting SMILES to coordinates: {e}")
        return None


def calculate_homo_lumo_gap_qm9(smiles, quiet=False):
    """
    按照QM9标准计算给定SMILES的HOMO、LUMO、GAP值以及计算时间
    使用B3LYP/6-31G(2df,p)方法进行计算
    
    Args:
        smiles (str): SMILES字符串
        quiet (bool): 是否静默模式，不输出日志
        
    Returns:
        dict: 包含HOMO、LUMO、GAP值和详细计算时间分析的字典（Hartree和eV单位）
    """
    # 记录开始时间
    start_time = time.time()
    
    # 初始化时间统计字典
    time_stats = {}
    
    try:
        # 将SMILES转换为3D坐标
        smiles_start = time.time()
        result = smiles_to_coordinates(smiles, quiet=quiet)
        if result is None:
            return {"error": "Failed to convert SMILES to 3D coordinates"}
            
        atomic_symbols, coordinates, smiles_time_stats = result
        time_stats['smiles_to_coordinates'] = smiles_time_stats
        time_stats['smiles_to_coordinates_total'] = smiles_time_stats['total']
        
        # 构建原子字符串
        atom_str_start = time.time()
        atom_str = "; ".join(
            f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in zip(atomic_symbols, coordinates)
        )
        time_stats['build_atom_string'] = time.time() - atom_str_start
        
        # 计算总电子数和自旋值
        electron_start = time.time()
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
        }
        
        # 计算总电子数
        total_electrons = sum(atomic_numbers[sym] for sym in atomic_symbols)
        
        # 设置自旋值：偶数电子自旋为0，奇数电子自旋为1
        spin = total_electrons % 2
        time_stats['electron_and_spin_calculation'] = time.time() - electron_start
        
        # 创建PySCF分子对象
        pyscf_start = time.time()
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = '6-31g(2df,p)'  # QM9标准基组
        mol.charge = 0
        mol.spin = spin
        mol.build()
        time_stats['pyscf_molecule_creation'] = time.time() - pyscf_start
        
        # 根据自旋选择DFT方法
        dft_start = time.time()
        if spin == 0:
            # 偶数电子使用限制性Kohn-Sham方法
            mf = dft.RKS(mol)
        else:
            # 奇数电子使用非限制性Kohn-Sham方法
            mf = dft.UKS(mol)
        mf.xc = 'b3lyp'  # QM9标准泛函
        mf.conv_tol = 1e-10
        
        # 存储结果的变量
        homo_energy = None
        lumo_energy = None
        
        # 定义回调函数以获取轨道能量
        def scf_callback(envs):
            nonlocal homo_energy, lumo_energy
            mo_energies = envs['mo_energy']
            mo_occs = envs['mo_occ']
            
            # 按照QM9方法提取HOMO和LUMO
            # 排序轨道能量和占据数
            sorted_indices = np.argsort(mo_energies)
            sorted_energies = mo_energies[sorted_indices]
            sorted_occupations = mo_occs[sorted_indices]
            
            # 查找HOMO: 最高占据轨道 (占据数 > 0.5)
            homo_index = None
            for i in range(len(sorted_occupations)-1, -1, -1):
                if sorted_occupations[i] > 0.5:  # 占据轨道
                    homo_index = i
                    break
            
            # 查找LUMO: 最低未占轨道
            lumo_index = homo_index + 1 if homo_index is not None else 0
            
            if homo_index is not None and lumo_index < len(sorted_energies):
                homo_energy = sorted_energies[homo_index]
                lumo_energy = sorted_energies[lumo_index]
        
        mf.callback = scf_callback
        time_stats['dft_setup'] = time.time() - dft_start
        
        # 运行计算
        scf_start = time.time()
        # if not quiet:
            # log("Running DFT calculation...")
        mf.kernel()
        time_stats['scf_calculation'] = time.time() - scf_start
        
        # 计算结束时间
        end_time = time.time()
        calculation_time = end_time - start_time
        
        if homo_energy is not None and lumo_energy is not None:
            gap = lumo_energy - homo_energy
            
            # 转换为eV单位
            homo_ev = homo_energy * HARTREE_TO_EV
            lumo_ev = lumo_energy * HARTREE_TO_EV
            gap_ev = gap * HARTREE_TO_EV
            
            return {
                "homo": float(homo_energy),
                "lumo": float(lumo_energy),
                "gap": float(gap),
                "homo_ev": float(homo_ev),
                "lumo_ev": float(lumo_ev),
                "gap_ev": float(gap_ev),
                "calculation_time": calculation_time,
                "time_stats": time_stats,
                "success": True
            }
        else:
            return {"error": "Failed to extract HOMO/LUMO energies", "time_stats": time_stats}
            
    except Exception as e:
        end_time = time.time()
        calculation_time = end_time - start_time
        return {
            "error": str(e),
            "calculation_time": calculation_time,
            "time_stats": time_stats
        }


# 示例用法
if __name__ == "__main__":
    # 测试函数
    # test_smiles = "CC"
    test_smiles = "C[C@@H](CO)CC=O"
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quiet':
        quiet = True
    else:
        quiet = False
    
    if not quiet:
        log(f"Calculating HOMO/LUMO for SMILES: {test_smiles} (QM9 method)")
    result = calculate_homo_lumo_gap_qm9(test_smiles, quiet=quiet)
    
    if "error" in result:
        if not quiet:
            log(f"Error: {result['error']}")
    else:
        if not quiet:
            log(f"HOMO: {result['homo']:.6f} Hartree ({result['homo_ev']:.6f} eV)")
            log(f"LUMO: {result['lumo']:.6f} Hartree ({result['lumo_ev']:.6f} eV)")
            log(f"GAP:  {result['gap']:.6f} Hartree ({result['gap_ev']:.6f} eV)")
            log(f"Total calculation time: {result['calculation_time']:.2f} seconds")
            
            # 输出详细的时间分析
            log("\nDetailed time analysis:")
            log(f"  SMILES to 3D coordinates: {result['time_stats']['smiles_to_coordinates_total']:.4f} seconds")
            
            # 输出SMILES转换的子步骤时间
            smiles_time = result['time_stats']['smiles_to_coordinates']
            log(f"    - Molecule creation: {smiles_time['mol_creation']:.4f} seconds")
            log(f"    - Add hydrogens: {smiles_time['add_hydrogens']:.4f} seconds")
            log(f"    - Embed molecule: {smiles_time['embed_molecule']:.4f} seconds")
            log(f"    - Optimize molecule: {smiles_time['optimize_molecule']:.4f} seconds")
            log(f"    - Get coordinates: {smiles_time['get_coordinates']:.4f} seconds")
            
            log(f"  Build atom string: {result['time_stats']['build_atom_string']:.4f} seconds")
            log(f"  Electron and spin calculation: {result['time_stats']['electron_and_spin_calculation']:.4f} seconds")
            log(f"  PySCF molecule creation: {result['time_stats']['pyscf_molecule_creation']:.4f} seconds")
            log(f"  DFT setup: {result['time_stats']['dft_setup']:.4f} seconds")
            log(f"  SCF calculation: {result['time_stats']['scf_calculation']:.4f} seconds")
            
            # 计算各步骤的百分比
            scf_percent = (result['time_stats']['scf_calculation'] / result['calculation_time']) * 100
            smiles_percent = (result['time_stats']['smiles_to_coordinates_total'] / result['calculation_time']) * 100
            other_percent = 100 - scf_percent - smiles_percent
            
            log(f"\nTime distribution:")
            log(f"  SCF calculation: {scf_percent:.1f}%")
            log(f"  SMILES to 3D coordinates: {smiles_percent:.1f}%")
            log(f"  Other steps: {other_percent:.1f}%")