import numpy as np
from rdkit.Chem import AllChem
import torch
import random
import os
from rdkit.Chem import rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem


def get_symbols(df):

    s = []
    for i in df.smiles:
        mol = Chem.MolFromSmiles(i)
        atoms = mol.GetAtoms()
        s += [a.GetSymbol() for a in atoms]
    return set(s)


symbols = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
symb_to_id = {v: k for k, v in enumerate(symbols)}
deg_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
valence_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4, 5, 6])}
charge_to_id = {v: k for k, v in enumerate([-3, -2, -1, 0, 1, 2, 3])}
nH_to_id = {v: k for k, v in enumerate([0, 1, 2, 3, 4])}
radical_e_to_id = {v: k for k, v in enumerate([0, 1, 2])}
arom_to_id = {v: k for k, v in enumerate([0, 1])}
inring_to_id = {v: k for k, v in enumerate([0, 1])}


hyb_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    )
}

btype_to_id = {
    v: k
    for k, v in enumerate(
        [
            "None",
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    )
}  # reserve 0 for self edge bondtype

stero_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    )
}

bonddir_to_id = {
    v: k
    for k, v in enumerate(
        [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.BEGINWEDGE,
            Chem.rdchem.BondDir.BEGINDASH,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            Chem.rdchem.BondDir.ENDUPRIGHT,
        ]
    )
}

conj_to_id = {v: k for k, v in enumerate([0, 1])}
inring_to_id = {v: k for k, v in enumerate([0, 1])}


def get_label(value, dict):
    if value in dict:
        return dict[value]
    else:
        return len(dict)


def get_atom_features(atom):

    atom_token = get_label(atom.GetSymbol(), symb_to_id)
    degree = get_label(atom.GetDegree(), deg_to_id)
    implvalence = get_label(atom.GetImplicitValence(), valence_to_id)
    nradelec = get_label(atom.GetNumRadicalElectrons(), radical_e_to_id)
    charge = get_label(atom.GetFormalCharge(), charge_to_id)
    nH = get_label(atom.GetTotalNumHs(), nH_to_id)
    hyb = get_label(atom.GetHybridization(), hyb_to_id)
    arom = get_label(atom.GetIsAromatic(), arom_to_id)
    inring = get_label(atom.IsInRing(), inring_to_id)

    return [
        atom_token,
        degree,
        implvalence,
        nradelec,
        charge,
        nH,
        hyb,
        arom,
        inring,
    ]


def get_bond_features(bond):

    bt = get_label(bond.GetBondType(), btype_to_id)
    st = get_label(bond.GetStereo(), stero_to_id)
    bd = get_label(bond.GetBondDir(), bonddir_to_id)
    conj = get_label(bond.GetIsConjugated(), conj_to_id)
    inring = get_label(bond.IsInRing(), inring_to_id)

    return [bt, st, bd, conj, inring]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set)
        )
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_pair(mol, add_self_loops=False):

    bonds = mol.GetBonds()
    if add_self_loops:
        edge_index = [[], []]
        edge_type = []
        for bond in bonds:
            edge_index[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            edge_index[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]

        for i in range(len(edge_index[0])):
            edge_type.append(mol.GetBondBetweenAtoms(edge_index[0][i], edge_index[1][i]).GetBondTypeAsDouble())
        edge_index[0] += list(range(mol.GetNumAtoms()))
        edge_index[1] += list(range(mol.GetNumAtoms()))
        edge_type += [0 for _ in range(mol.GetNumAtoms())]

    else:
        edge_index = [[], []]
        edge_type = []
        for bond in bonds:
            edge_index[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            edge_index[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]

        for i in range(len(edge_index[0])):
            edge_type.append(mol.GetBondBetweenAtoms(edge_index[0][i], edge_index[1][i]).GetBondTypeAsDouble())

    return edge_index


def set_seed(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_scaffold(mol):

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold)
    except:
        scaffold = ''
    return scaffold