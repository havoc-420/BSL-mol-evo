import numpy as np
import time
from ase import Atoms
from pyscf import gto, dft
import os
import glob
import logging
import sys

# 配置 logging，日志输出到终端和文件，带时间戳
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pyscf_dft_md22.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info  # 简写

npz_dir = '/home/zw/zw/MD22'
save_dir = '/home/zw/zw/MD22/dft_outputs_npz'
os.makedirs(save_dir, exist_ok=True)

npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
log(f"Found {len(npz_files)} npz files.")

for npz_path in npz_files:
    filename = os.path.basename(npz_path)
    mol_name = os.path.splitext(filename)[0]
    log(f"\nProcessing molecule: {mol_name}")

    data = np.load(npz_path)
    coords_all = data['R']
    atomic_numbers = data['z']
    #n_frames = min(200, len(coords_all))  # 跑200

    for frame_idx in range(0, 100):   
        positions = coords_all[frame_idx]
        molecule_ase = Atoms(numbers=atomic_numbers, positions=positions)
        element_symbols = molecule_ase.get_chemical_symbols()
        atom_str = "; ".join(
            f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in zip(element_symbols, positions)
        )

        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = 'def2-svp'
        mol.charge = 0
        mol.spin = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = 'PBE'
        mf.conv_tol = 1e-10

        e_tot_all = []
        homo_all = []
        lumo_all = []
        dm_all = []
        fock_all = []

        def scf_callback(envs):
            e_current = envs['e_tot'] * 627.5094740631  # Hartree to kcal/mol
            dm_current = np.array(envs['dm'])
            fock_current = np.array(envs['fock'])
            mo_energies = envs['mo_energy']
            mo_occs = envs['mo_occ']

            homo_index = max(i for i, occ in enumerate(mo_occs) if occ > 0)
            lumo_index = min(i for i, occ in enumerate(mo_occs) if occ == 0)
            homo_energy = mo_energies[homo_index]
            lumo_energy = mo_energies[lumo_index]

            e_tot_all.append(e_current)
            homo_all.append(homo_energy)
            lumo_all.append(lumo_energy)
            dm_all.append(dm_current)
            fock_all.append(fock_current)

            #log(f"Frame {frame_idx} SCF Iter {envs['cycle']:2d}: E = {e_current:.12f} kcal/mol")

        mf.callback = scf_callback

        log(f"\nRunning DFT single-point calculation for frame {frame_idx}...")
        mf.kernel()

        out_file = os.path.join(save_dir, f"{mol_name}_frame{frame_idx}.npz")
        np.savez_compressed(out_file,
                            e_tot=np.array(e_tot_all),
                            homo=np.array(homo_all),
                            lumo=np.array(lumo_all),
                            dm=np.array(dm_all),
                            fock=np.array(fock_all),
                            atomic_charges=atomic_numbers,
                            atomic_positions=positions
                            )
        log(f"Saved structured data to {out_file}")

log("\nAll molecules processed.")
