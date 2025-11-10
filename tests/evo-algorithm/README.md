# MoleculeEvolverAnalysis 测试系统

## 概述

本目录包含了用于测试 MoleculeEvolverAnalysis 类的所有测试代码和相关资源。测试系统设计为模块化和可扩展的，支持多种测试类型和运行方式。

## 目录结构

```
evo-ag/
├── testcases/                  # 测试数据集目录
│   ├── basic/                  # 基础测试数据
│   │   └── test_data.json      # 基础测试数据
│   ├── extended/               # 扩展测试数据
│   │   └── extended_test_data.json # 扩展测试数据
│   └── __init__.py             # Python包初始化文件
├── tests/                      # 测试代码目录
│   ├── __init__.py             # Python包初始化文件
│   ├── test_basic.py           # 基础功能测试
│   ├── test_extended.py        # 扩展测试
│   └── test_consistency.py     # 一致性测试
├── runners/                    # 测试运行器目录
│   ├── __init__.py             # Python包初始化文件
│   ├── run_all_tests.py        # 统一测试入口
│   ├── run_extended_tests.py   # 扩展测试运行器
│   └── test_runner.sh          # Shell脚本测试运行器
├── config/                     # 配置文件目录
│   ├── __init__.py             # Python包初始化文件
│   └── test_config.yaml        # 测试配置文件
├── docs/                       # 文档目录
│   ├── test_plan.md            # 测试计划文档
│   ├── test_analysis.md        # 测试分析报告
│   └── test_report_template.md # 测试报告模板
├── bug-logs/                   # Bug日志目录
│   └── evo-bug-1/              # 特定bug测试用例
├── run_tests.sh                # 测试运行脚本
└── README.md                   # 本文件
```

## 测试类型

1. **基础功能测试** - 验证基本分子处理能力
2. **扩展测试** - 包含复杂分子和边界情况
3. **性能测试** - 测量执行时间和资源使用
4. **一致性测试** - 验证不同输出格式的一致性

## 测试策略

与传统的断言失败测试不同，本测试系统采用统计技术指标的方式：

- 不会在遇到预期与实际不匹配时立即失败
- 而是记录通过和失败的测试用例
- 最终输出总体通过率和详细错误信息
- 这种方式更适合评估模型在复杂任务中的整体表现

## 使用方法

### Shell脚本方式

```bash
# 给脚本添加执行权限
chmod +x run_tests.sh

# 运行所有测试
./run_tests.sh all

# 运行基础测试
./run_tests.sh basic

# 运行扩展测试
./run_tests.sh extended

# 运行一致性测试
./run_tests.sh consistency

# 显示帮助
./run_tests.sh help
```

## 测试配置

测试系统使用 `config/test_config.yaml` 文件进行配置，可以配置：

- 测试套件定义
- 运行选项
- 性能测试参数
- 测试分子集合

## 添加新测试

1. 在 `testcases/extended/extended_test_data.json` 中添加新的测试用例
2. 如果需要特殊的测试逻辑，可以在 `tests/` 目录中添加新的测试方法
3. 更新 `config/test_config.yaml` 以包含新的测试套件（如果需要）

## 测试数据格式

测试数据以JSON格式存储，每个测试用例包含：

- `name`: 测试用例名称
- `smiles`: SMILES表示
- `type`: 分子类型
- `expected_path_dict`: 期望的字典格式路径（可选）
- `expectations`: 期望的测试结果特征

## 生成测试报告

扩展测试运行器会自动生成测试报告，包含：

- 测试通过率统计
- 性能数据
- 失败用例详情

## 故障排除

如果遇到问题，请检查：

1. RDKit是否正确安装
2. Python路径是否包含项目根目录
3. 测试数据文件是否存在且格式正确
4. 依赖包是否完整安装