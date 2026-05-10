# Mol-Evo

Molecular evolution and generation using deep learning approaches.

## Modules

This project integrates several open-source modules for molecular representation learning and dynamics. 

**Important Note**: These modules are included directly in this repository (rather than as git submodules or separate dependencies) because they have been **locally modified to adapt to this framework's architecture and interfaces**. The modifications are made to ensure compatibility with our training pipeline and model integration requirements.

Original repositories are listed below for reference and attribution purposes:

### 1. Equiformer
- **Repository**: [atomicarchitects/equiformer](https://github.com/atomicarchitects/equiformer)
- **Remote**: `git@github.com:atomicarchitects/equiformer.git`
- **Description**: Equivariant Transformer for 3D molecular systems

### 2. Equiformer V2
- **Repository**: [atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2)
- **Remote**: `https://github.com/atomicarchitects/equiformer_v2.git`
- **Description**: Improved version of Equiformer with enhanced equivariant attention

### 3. TorchMD-Net
- **Repository**: [torchmd/torchmd-net](https://github.com/torchmd/torchmd-net)
- **Remote**: `https://github.com/torchmd/torchmd-net.git`
- **Description**: PyTorch implementation of molecular dynamics with neural networks

### 4. FragNet
- **Repository**: [pnnl/FragNet](https://github.com/pnnl/FragNet)
- **Remote**: `git@github.com:pnnl/FragNet.git`
- **Description**: Fragment-based molecular representation learning

## Acknowledgments

We gratefully acknowledge the original authors and contributors of the modules used in this project:

- **Equiformer & Equiformer V2**: Thanks to the [atomicarchitects](https://github.com/atomicarchitects) team for developing the equivariant transformer architectures for molecular systems.

- **TorchMD-Net**: Thanks to the [torchmd](https://github.com/torchmd) team for the molecular dynamics neural network implementation.

- **FragNet**: Thanks to [PNNL](https://www.pnnl.gov/) (Pacific Northwest National Laboratory) for the fragment-based molecular representation framework.

These open-source projects provide the foundation for molecular representation learning and dynamics modeling in our work.

## Usage

The modules are located in the `modules/` directory. Please refer to each module's original documentation for installation and usage instructions.

## License

Please respect the licenses of the individual modules listed above.