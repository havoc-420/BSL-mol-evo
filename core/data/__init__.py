#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子数据处理模块
"""

from .processing import (
    smiles_to_fingerprint,
    atom_type_to_onehot,
    operation_type_to_onehot,
    calculate_molecular_similarity,
    prepare_edge_features,
    load_qm9_properties,
    build_molecule_graph_with_properties,
    build_molecule_graph_with_fingerprints,
    build_molecule_evolution_dataset,
    prepare_property_change_targets,
    load_operation_config
)

from .data_v0 import (
    smiles_to_graph_data,
    prepare_edge_features as prepare_edge_features_v0,
)

from .fragnet_data import (
    smile_to_fragnet_features,
    smile_to_fragnet_batch,
    fragnet_collate,
)

from .pair_data import (
    MoleculePairDataset,
    pair_collate,
)

from .unified_processing import (
    build_molecule_evolution_dataset_v0,
    build_molecule_evolution_dataset_unified,
)

from .path_processing import (
    build_molecule_path_dataset_v0_3,
    load_path_json,
)

from .path_data import (
    MoleculePathDataset,
    path_collate,
)

from .pair_to_path import (
    build_path_from_pair,
    convert_pairs_file_to_paths,
    convert_step_files_to_path_dataset,
)

__all__ = [
    'smiles_to_fingerprint',
    'atom_type_to_onehot',
    'operation_type_to_onehot',
    'calculate_molecular_similarity',
    'prepare_edge_features',
    'load_qm9_properties',
    'build_molecule_graph_with_properties',
    'build_molecule_graph_with_fingerprints',
    'build_molecule_evolution_dataset',
    'smiles_to_graph_data',
    'prepare_edge_features_v0',
    'smile_to_fragnet_features',
    'smile_to_fragnet_batch',
    'fragnet_collate',
    'MoleculePairDataset',
    'pair_collate',
    'build_molecule_evolution_dataset_v0',
    'build_molecule_evolution_dataset_unified',
    'build_molecule_path_dataset_v0_3',
    'load_path_json',
    'MoleculePathDataset',
    'path_collate',
    'build_path_from_pair',
    'convert_pairs_file_to_paths',
    'convert_step_files_to_path_dataset',
]