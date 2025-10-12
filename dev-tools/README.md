# Molecule Evolution Training Data API

这个API服务提供对`mol_evo`模块中训练数据的访问接口。

## 启动服务

```bash
cd /Users/havoc420/Documents/Projects/python/BSL-mol-editor/mol_evo/dev-tools
python api.py
```

服务将在 `http://localhost:6000` 上启动。

## API 文档

FastAPI自动生成交互式API文档：
- Swagger UI: http://localhost:6000/docs
- ReDoc: http://localhost:6000/redoc

## API 接口

### 1. 健康检查
- **URL**: `/health`
- **方法**: GET
- **描述**: 检查API服务是否正常运行
- **响应示例**:
  ```json
  {
    "status": "ok",
    "message": "Molecule Evolution Training Data API is running"
  }
  ```

### 2. 列出训练数据文件
- **URL**: `/training-data`
- **方法**: GET
- **参数**:
  - `limit` (可选, 默认: 10): 返回的项目数量
  - `offset` (可选, 默认: 0): 分页偏移量
- **描述**: 列出所有可用的训练数据文件及其基本信息
- **响应示例**:
  ```json
  [
    {
      "file_path": "/path/to/training_data.json",
      "training_params": {
        "data_file": "mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv",
        "max_pairs": 2000,
        "epochs": 50,
        "seed": 42,
        "target_property": "gap_change_pct",
        "batch_size": 1024,
        "device": "cuda",
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "weight_decay": 1e-05,
        "scheduler": "ReduceLROnPlateau",
        "loss_function": "L1Loss",
        "patience_limit": 50
      },
      "model_params": {
        "node_feature_dim": 11,
        "edge_feature_dim": 11,
        "hidden_dim": 128,
        "output_dim": 1,
        "num_layers": 2
      },
      "property_stats": {
        "gap_change_pct": [
          0.0726774729888522,
          0.199669904795772
        ]
      },
      "test_metrics": {
        "test_loss": 0.28654947876930237,
        "rmse": 0.4368484914302826,
        "mae": 0.28654947876930237,
        "r2": 0.794584333896637,
        "threshold_accs": {
          "0.4": 0.7699999809265137,
          "0.3": 0.7099999785423279,
          "0.2": 0.5349999666213989,
          "0.1": 0.32499998807907104,
          "0.05": 0.1550000011920929
        },
        "dataset_info": {
          "train_size": 1600,
          "val_size": 200,
          "test_size": 200
        }
      }
    }
  ]
  ```

### 3. 获取特定训练数据
- **URL**: `/training-data/{file_id}`
- **方法**: GET
- **参数**:
  - `file_id`: 文件索引 (从list_training_data获取)
- **描述**: 获取特定训练数据文件的完整内容
- **响应示例**:
  ```json
  {
    "training_params": {
      "data_file": "mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv",
      "max_pairs": 2000,
      "epochs": 50,
      "seed": 42,
      "target_property": "gap_change_pct",
      "batch_size": 1024,
      "device": "cuda",
      "optimizer": "Adam",
      "learning_rate": 0.01,
      "weight_decay": 1e-05,
      "scheduler": "ReduceLROnPlateau",
      "loss_function": "L1Loss",
      "patience_limit": 50
    },
    "model_params": {
      "node_feature_dim": 11,
      "edge_feature_dim": 11,
      "hidden_dim": 128,
      "output_dim": 1,
      "num_layers": 2
    },
    "property_stats": {
      "gap_change_pct": [
        0.0726774729888522,
        0.199669904795772
      ]
    },
    "test_metrics": {
      "test_loss": 0.28654947876930237,
      "rmse": 0.4368484914302826,
      "mae": 0.28654947876930237,
      "r2": 0.794584333896637,
      "threshold_accs": {
        "0.4": 0.7699999809265137,
        "0.3": 0.7099999785423279,
        "0.2": 0.5349999666213989,
        "0.1": 0.32499998807907104,
        "0.05": 0.1550000011920929
      },
      "dataset_info": {
        "train_size": 1600,
        "val_size": 200,
        "test_size": 200
      }
    },
    "losses": {
      "train_losses": [
        3.3320417022705078,
        1.004459524154663,
        0.8162130856513977
      ],
      "val_losses": [
        0.6542766690254211,
        0.5060027837753296,
        0.43504250049591064
      ]
    }
  }
  ```

### 4. 根据路径获取训练数据
- **URL**: `/training-data/path/{path}`
- **方法**: GET
- **参数**:
  - `path`: 文件路径 (相对于项目根目录，必须指向training_data.json文件)
- **描述**: 根据文件路径获取训练数据
- **示例**: `GET /training-data/path/mol_evo/output/v0/gcn/training_20251010_140112/training_data.json`

## 使用示例

### Python 示例

```python
import requests

# 列出训练数据文件
response = requests.get('http://localhost:6000/training-data?limit=5')
data_list = response.json()

# 获取第一个训练数据文件的详细内容
if data_list:
    file_id = 0
    response = requests.get(f'http://localhost:6000/training-data/{file_id}')
    training_data = response.json()
    print(training_data)
```

### curl 示例

```bash
# 健康检查
curl http://localhost:6000/health

# 列出训练数据文件
curl "http://localhost:6000/training-data?limit=5" | python -m json.tool

# 获取特定训练数据
curl http://localhost:6000/training-data/0 | python -m json.tool
```

## 数据结构

### TrainingDataResponse
- `training_params`: 训练参数
  - `data_file`: 数据文件路径
  - `max_pairs`: 最大训练对数
  - `epochs`: 训练轮数
  - `seed`: 随机种子
  - `target_property`: 目标属性
  - `batch_size`: 批处理大小
  - `device`: 训练设备 (cuda/cpu)
  - `optimizer`: 优化器
  - `learning_rate`: 学习率
  - `weight_decay`: 权重衰减
  - `scheduler`: 学习率调度器
  - `loss_function`: 损失函数
  - `patience_limit`: 早停耐心值
- `model_params`: 模型参数
  - `node_feature_dim`: 节点特征维度
  - `edge_feature_dim`: 边特征维度
  - `hidden_dim`: 隐藏层维度
  - `output_dim`: 输出维度
  - `num_layers`: 网络层数
- `property_stats`: 属性统计信息
- `test_metrics`: 测试指标
  - `test_loss`: 测试损失
  - `rmse`: 均方根误差
  - `mae`: 平均绝对误差
  - `r2`: 决定系数
  - `threshold_accs`: 阈值准确率
  - `dataset_info`: 数据集信息
- `losses`: 训练损失值
  - `train_losses`: 训练损失列表
  - `val_losses`: 验证损失列表

### TrainingDataListItem
- `file_path`: 文件路径
- `training_params`: 训练参数 (同上)
- `model_params`: 模型参数 (同上)
- `property_stats`: 属性统计信息
- `test_metrics`: 测试指标 (同上)

## 注意事项

1. 确保在启动API之前已经生成了训练数据文件。
2. API服务默认运行在6000端口。
3. 所有训练数据文件都存储在`../output`目录下（相对于API文件位置）。
4. Swagger UI文档依赖外部CDN资源，如果无法加载，请检查网络连接或尝试使用ReDoc界面。