# 尝试过拟合 case - 验证架构

```bash
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0_model.py --max-pairs 50 --epochs 20
2025-10-10 14:00:25,752 - INFO - training_console - ============================================================
2025-10-10 14:00:25,752 - INFO - training_console - 基于NNConv的分子进化预测器模型训练开始
2025-10-10 14:00:25,752 - INFO - training_console - ============================================================
2025-10-10 14:00:25,752 - INFO - training_console - 数据文件: mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv
2025-10-10 14:00:25,752 - INFO - training_console - 最大对数: 50
2025-10-10 14:00:25,752 - INFO - training_console - 训练轮数: 20
2025-10-10 14:00:25,753 - INFO - training_console - 正在构建图数据: mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv
2025-10-10 14:00:26,455 - INFO - training_console - 加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/training_dataset.pt，包含 1377 个分子
2025-10-10 14:00:26,460 - INFO - training_console - 分子处理统计: 总数=100, 命中=100, 未命中=0, 命中率=100.00%
2025-10-10 14:00:26,464 - INFO - training_console - 训练集样本示例 (头部数据):
2025-10-10 14:00:26,464 - INFO - training_console -   样本 1:
2025-10-10 14:00:26,464 - INFO - training_console -     起始分子节点数: 5
2025-10-10 14:00:26,465 - INFO - training_console -     起始分子边数: 8
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子节点数: 8
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子边数: 14
2025-10-10 14:00:26,465 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:00:26,465 - INFO - training_console -     目标属性值: 0.46517816185951233
2025-10-10 14:00:26,465 - INFO - training_console -   样本 2:
2025-10-10 14:00:26,465 - INFO - training_console -     起始分子节点数: 3
2025-10-10 14:00:26,465 - INFO - training_console -     起始分子边数: 4
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子节点数: 6
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子边数: 10
2025-10-10 14:00:26,465 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:00:26,465 - INFO - training_console -     目标属性值: 0.013040220364928246
2025-10-10 14:00:26,465 - INFO - training_console -   样本 3:
2025-10-10 14:00:26,465 - INFO - training_console -     起始分子节点数: 4
2025-10-10 14:00:26,465 - INFO - training_console -     起始分子边数: 6
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子节点数: 11
2025-10-10 14:00:26,465 - INFO - training_console -     目标分子边数: 20
2025-10-10 14:00:26,466 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:00:26,466 - INFO - training_console -     目标属性值: 0.5481558442115784
2025-10-10 14:00:26,466 - INFO - training_console - 数据构建完成:
2025-10-10 14:00:26,466 - INFO - training_console -   - 样本数: 50
2025-10-10 14:00:26,466 - INFO - training_console -   - 节点特征维度: 11
2025-10-10 14:00:26,466 - INFO - training_console -   - 边特征维度: 11
2025-10-10 14:00:26,466 - INFO - training_console -   - 目标属性变化维度: 1
2025-10-10 14:00:26,467 - INFO - training_console - 数据集划分完成:
2025-10-10 14:00:26,467 - INFO - training_console -   - 训练集: 40 (80.0%)
2025-10-10 14:00:26,467 - INFO - training_console -   - 验证集: 5 (10.0%)
2025-10-10 14:00:26,467 - INFO - training_console -   - 测试集: 5 (10.0%)
2025-10-10 14:00:26,506 - INFO - training_console - 正在创建模型...
2025-10-10 14:00:26,507 - INFO - training_console - 模型参数数量: 980993
2025-10-10 14:00:26,507 - INFO - training_console - 使用设备: cuda
Epoch [10/20], Train Loss: 0.768312, Val Loss: 0.075120, R²: 0.6900, MAE: 0.2548                                                                          
Epoch [20/20], Train Loss: 0.418153, Val Loss: 0.193127, R²: 0.2031, MAE: 0.2527                                                                          
Training Epochs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:06<00:00,  3.17it/s]
2025-10-10 14:00:33,082 - INFO - training_console - 开始训练 (20 轮)...
损失趋势图已保存: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140025/loss_trends.png
指标趋势图已保存: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140025/metrics_trends.png
2025-10-10 14:00:34,063 - INFO - training_console - 训练完成:
2025-10-10 14:00:34,063 - INFO - training_console -   - 最终训练损失: 0.418153
2025-10-10 14:00:34,063 - INFO - training_console -   - 最佳验证损失: 0.001852
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试损失 (MSE): 0.008926
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试RMSE: 0.094475
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试MAE: 0.072970
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试R²: 0.971540
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试阈值准确率 (0.4): 1.0000
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试阈值准确率 (0.3): 1.0000
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试阈值准确率 (0.2): 1.0000
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试阈值准确率 (0.1): 0.8000
2025-10-10 14:00:34,064 - INFO - training_console -   - 测试阈值准确率 (0.05): 0.4000
2025-10-10 14:00:34,064 - INFO - training_console -   - 模型已保存到: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140025/molecule_evolution_gcn_v0_mu_predictor.pth
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$  python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 0
找到一个模型文件: training_20251010_140025
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 2 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C([H])([H])[H]
目标分子 SMILES: [H]C([H])([H])C([H])([H])[H]
变化原子类型: C
操作类型: add
使用模型: training_20251010_140025
数据集行号: 0
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              -0.0834         0.0000          -0.0834        
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0069          0.0834          0.0834         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$  python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 1
找到一个模型文件: training_20251010_140025
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 2 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]O[H]
目标分子 SMILES: [H]OC([H])([H])[H]
变化原子类型: C
操作类型: add
使用模型: training_20251010_140025
数据集行号: 1
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              0.2004          -0.3253         0.5257         
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.2763          0.5257          0.5257         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0_model.py --max-pairs 50 --epochs 1000
2025-10-10 14:01:12,446 - INFO - training_console - ============================================================
2025-10-10 14:01:12,446 - INFO - training_console - 基于NNConv的分子进化预测器模型训练开始
2025-10-10 14:01:12,447 - INFO - training_console - ============================================================
2025-10-10 14:01:12,447 - INFO - training_console - 数据文件: mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv
2025-10-10 14:01:12,447 - INFO - training_console - 最大对数: 50
2025-10-10 14:01:12,447 - INFO - training_console - 训练轮数: 1000
2025-10-10 14:01:12,447 - INFO - training_console - 正在构建图数据: mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv
2025-10-10 14:01:13,170 - INFO - training_console - 加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/training_dataset.pt，包含 1377 个分子
2025-10-10 14:01:13,176 - INFO - training_console - 分子处理统计: 总数=100, 命中=100, 未命中=0, 命中率=100.00%
2025-10-10 14:01:13,194 - INFO - training_console - 训练集样本示例 (头部数据):
2025-10-10 14:01:13,194 - INFO - training_console -   样本 1:
2025-10-10 14:01:13,194 - INFO - training_console -     起始分子节点数: 5
2025-10-10 14:01:13,194 - INFO - training_console -     起始分子边数: 8
2025-10-10 14:01:13,194 - INFO - training_console -     目标分子节点数: 8
2025-10-10 14:01:13,194 - INFO - training_console -     目标分子边数: 14
2025-10-10 14:01:13,194 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:01:13,194 - INFO - training_console -     目标属性值: 0.46517816185951233
2025-10-10 14:01:13,194 - INFO - training_console -   样本 2:
2025-10-10 14:01:13,194 - INFO - training_console -     起始分子节点数: 3
2025-10-10 14:01:13,194 - INFO - training_console -     起始分子边数: 4
2025-10-10 14:01:13,195 - INFO - training_console -     目标分子节点数: 6
2025-10-10 14:01:13,195 - INFO - training_console -     目标分子边数: 10
2025-10-10 14:01:13,195 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:01:13,195 - INFO - training_console -     目标属性值: 0.013040220364928246
2025-10-10 14:01:13,195 - INFO - training_console -   样本 3:
2025-10-10 14:01:13,195 - INFO - training_console -     起始分子节点数: 4
2025-10-10 14:01:13,195 - INFO - training_console -     起始分子边数: 6
2025-10-10 14:01:13,195 - INFO - training_console -     目标分子节点数: 11
2025-10-10 14:01:13,195 - INFO - training_console -     目标分子边数: 20
2025-10-10 14:01:13,195 - INFO - training_console -     边特征: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
2025-10-10 14:01:13,195 - INFO - training_console -     目标属性值: 0.5481558442115784
2025-10-10 14:01:13,195 - INFO - training_console - 数据构建完成:
2025-10-10 14:01:13,195 - INFO - training_console -   - 样本数: 50
2025-10-10 14:01:13,195 - INFO - training_console -   - 节点特征维度: 11
2025-10-10 14:01:13,195 - INFO - training_console -   - 边特征维度: 11
2025-10-10 14:01:13,195 - INFO - training_console -   - 目标属性变化维度: 1
2025-10-10 14:01:13,197 - INFO - training_console - 数据集划分完成:
2025-10-10 14:01:13,197 - INFO - training_console -   - 训练集: 40 (80.0%)
2025-10-10 14:01:13,197 - INFO - training_console -   - 验证集: 5 (10.0%)
2025-10-10 14:01:13,197 - INFO - training_console -   - 测试集: 5 (10.0%)
2025-10-10 14:01:13,247 - INFO - training_console - 正在创建模型...
2025-10-10 14:01:13,247 - INFO - training_console - 模型参数数量: 980993
2025-10-10 14:01:13,247 - INFO - training_console - 使用设备: cuda
Epoch [10/1000], Train Loss: 0.768312, Val Loss: 0.075120, R²: 0.6900, MAE: 0.2548                                                                        
Epoch [20/1000], Train Loss: 0.418153, Val Loss: 0.193127, R²: 0.2031, MAE: 0.2527                                                                        
Epoch [30/1000], Train Loss: 0.810375, Val Loss: 0.147881, R²: 0.3898, MAE: 0.3235                                                                        
Epoch [40/1000], Train Loss: 0.773631, Val Loss: 0.337551, R²: -0.3929, MAE: 0.5174                                                                       
Epoch [50/1000], Train Loss: 0.375064, Val Loss: 0.225317, R²: 0.0702, MAE: 0.3801                                                                        
Epoch [60/1000], Train Loss: 0.274041, Val Loss: 0.184050, R²: 0.2405, MAE: 0.3693                                                                        
Training Epochs:   6%|█████▉                                                                                            | 60/1000 [00:18<04:36,  3.40it/s]2025-10-10 14:01:31,875 - INFO - training_console - 早停机制触发，在第 61 轮停止训练
Training Epochs:   6%|█████▉                                                                                            | 60/1000 [00:18<04:47,  3.27it/s]
2025-10-10 14:01:31,876 - INFO - training_console - 开始训练 (1000 轮)...
损失趋势图已保存: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140112/loss_trends.png
指标趋势图已保存: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140112/metrics_trends.png
2025-10-10 14:01:32,758 - INFO - training_console - 训练完成:
2025-10-10 14:01:32,758 - INFO - training_console -   - 最终训练损失: 0.272976
2025-10-10 14:01:32,759 - INFO - training_console -   - 最佳验证损失: 0.001852
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试损失 (MSE): 0.108207
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试RMSE: 0.328949
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试MAE: 0.239593
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试R²: 0.654966
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试阈值准确率 (0.4): 0.8000
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试阈值准确率 (0.3): 0.8000
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试阈值准确率 (0.2): 0.6000
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试阈值准确率 (0.1): 0.4000
2025-10-10 14:01:32,759 - INFO - training_console -   - 测试阈值准确率 (0.05): 0.2000
2025-10-10 14:01:32,759 - INFO - training_console -   - 模型已保存到: /home/data2/rhj/project/mol_editor/mol_evo/../mol_evo/model-data/v0/training_20251010_140112/molecule_evolution_gcn_v0_mu_predictor.pth
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 0
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 4 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C([H])([H])[H]
目标分子 SMILES: [H]C([H])([H])C([H])([H])[H]
变化原子类型: C
操作类型: add
使用模型: training_20251010_140112
数据集行号: 0
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              -0.0789         0.0000          -0.0789        
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0062          0.0789          0.0789         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 1
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 4 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]O[H]
目标分子 SMILES: [H]OC([H])([H])[H]
变化原子类型: C
操作类型: add
使用模型: training_20251010_140112
数据集行号: 1
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              0.2073          -0.3253         0.5326         
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.2836          0.5326          0.5326         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 2
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 4 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C#C[H]
目标分子 SMILES: [H]C([H])([H])C([H])([H])C([H])([H])[H]
变化原子类型: C
操作类型: replace
使用模型: training_20251010_140112
数据集行号: 2
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              -0.1901         0.0597          -0.2498        
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0624          0.2498          0.2498         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 3
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 6 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C#C[H]
目标分子 SMILES: [H]C1([H])C([H])([H])C1([H])[H]
变化原子类型: C
操作类型: replace
使用模型: training_20251010_140112
数据集行号: 3
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              -0.1083         0.0005          -0.1088        
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0118          0.1088          0.1088         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 4
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 7 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C#N
目标分子 SMILES: [H]C([H])([H])C#N
变化原子类型: C
操作类型: add
使用模型: training_20251010_140112
数据集行号: 4
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              0.9228          0.9329          -0.0101        
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0001          0.0101          0.0101         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 5
找到一个模型文件: training_20251010_140112
加载缓存文件: /home/data2/rhj/project/mol_editor/mol_evo/core/utils/../../.cache/molecule_graphs/prediction_dataset.pt，包含 9 个分子

预测结果与真实值对比:
============================================================
起始分子 SMILES: [H]C([H])=O
目标分子 SMILES: [H]C(=O)C([H])([H])[H]
变化原子类型: C
操作类型: add
使用模型: training_20251010_140112
数据集行号: 5
============================================================
属性名称            预测值             真实值             差值             
------------------------------------------------------------
mu              0.5156          0.4593          0.0563         
------------------------------------------------------------
总体指标            MSE             RMSE            MAE            
                0.0032          0.0563          0.0563         
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor$ 
```