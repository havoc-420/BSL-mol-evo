## Equiformer

`Equiformer` 是一类面向三维原子/分子图的**等变图注意力 Transformer**，其核心贡献在于证明了：只要将标准 Transformer 中的关键算子替换为适配 `SE(3)/E(3)` 对称性的等变版本，并在特征通道中引入基于 irreps 的张量积交互，Transformer 架构同样可以有效处理分子这类三维几何对象。具体而言，`Equiformer` 使用等变线性层、等变归一化、基于张量积的特征交互，以及提出的 `equivariant graph attention` 来替代普通 dot-product attention，使模型能够在保持平移/旋转等变性的同时，更充分地建模原子间方向关系与高阶几何耦合。与传统等变 `MPNN` 相比，`Equiformer` 更突出全局注意力框架的表达优势；与早期等变 Transformer 相比，它又通过更适配三维图的注意力和消息传递设计取得了更强的经验性能。因此，从相关工作角度看，`Equiformer` 的意义不仅在于提出了一个高性能 backbone，更在于为后续工作提供了一个清晰范式：**将 Transformer 的结构优势与三维几何等变约束统一到同一分子表示学习框架中。**
