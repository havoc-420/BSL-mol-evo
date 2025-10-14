#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具函数封装
包含用于测试分子特征提取器的工具函数
"""

import torch
import sys
import os
from torch_geometric.data import Data

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    import inquirer
except ImportError as e:
    print(f"导入模块时出错: {e}")
    inquirer = None


def select_feature_extractor_interactively(extractors):
    """
    交互式选择特征提取器
    
    Args:
        extractors (dict): 可用的特征提取器字典，格式为 {"名称": 类}
        
    Returns:
        选择的特征提取器类
    """
    # 获取所有可用的特征提取器名称
    extractor_names = list(extractors.keys())
    
    if not extractor_names:
        print("未找到任何可用的特征提取器")
        return None

    if len(extractor_names) == 1:
        print(f"只有一个可用特征提取器: {extractor_names[0]}")
        return extractors[extractor_names[0]]

    # 如果没有安装inquirer库，使用基础交互模式
    try:
        if inquirer:
            inquirer.__version__
        else:
            raise AttributeError
    except AttributeError:
        print("警告: 未安装 inquirer 库，使用基础交互模式")
        
        # 基础模式：纯文本交互
        filtered_extractors = extractor_names[:]
        filter_text = ""
        
        while True:
            print(f"\n找到 {len(extractor_names)} 个特征提取器 (当前显示 {len(filtered_extractors)} 个):")
            if filter_text:
                print(f"当前过滤条件: '{filter_text}'")
            
            if not filtered_extractors:
                print("没有匹配的特征提取器")
            else:
                for i, extractor_name in enumerate(filtered_extractors):
                    print(f"{i+1}. {extractor_name}")
            
            print("\n操作说明:")
            print("  • 直接输入数字选择特征提取器 (如: 1)")
            print("  • 输入关键词进行过滤 (如: gcn)")
            print("  • 输入 'clear' 清除过滤条件")
            print("  • 输入 'quit' 退出选择")
            
            try:
                choice = input("\n请选择或输入过滤条件: ").strip()
                
                if choice.lower() == 'quit':
                    print("用户取消选择")
                    return None
                    
                if choice.lower() == 'clear':
                    filtered_extractors = extractor_names[:]
                    filter_text = ""
                    continue
                
                # 检查是否为数字选择
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(filtered_extractors):
                        selected_extractor = filtered_extractors[choice_idx]
                        print(f"选择特征提取器: {selected_extractor}")
                        return extractors[selected_extractor]
                    else:
                        print(f"无效选择: {choice_idx + 1}, 请输入 1 到 {len(filtered_extractors)} 之间的数字")
                    
                else:
                    # 视为过滤关键词
                    filter_text = choice
                    filtered_extractors = [e for e in extractor_names 
                                         if filter_text.lower() in e.lower()]
                    if not filtered_extractors:
                        print(f"没有找到包含 '{filter_text}' 的特征提取器")
                        # 保持当前过滤状态，让用户可以修改关键词
                    else:
                        print(f"已过滤，找到 {len(filtered_extractors)} 个匹配项")
                        
            except KeyboardInterrupt:
                print("\n用户取消选择")
                return None
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")
                continue
    
    # 高级模式：使用inquirer库
    try:
        # 创建选项列表
        all_choices = [(name, name) for name in extractor_names]
        
        # 动态过滤选择主循环
        filtered_choices = all_choices[:]
        filter_text = ""
        
        while True:
            # 构建可选择的选项列表
            selectable_options = []
            
            # 添加过滤后的模型选项
            for display_name, extractor_name in filtered_choices:
                selectable_options.append((display_name, extractor_name))
            
            # 添加分隔线和功能选项
            selectable_options.append(('────────────────', 'SEPARATOR'))
            selectable_options.append(('❌ 取消选择', 'ACTION_CANCEL'))
            selectable_options.append(('🔍 输入过滤关键词', 'ACTION_FILTER'))
            
            # 创建选择问题
            question = [
                inquirer.List(
                    'selection',
                    message=f'选择特征提取器 (显示 {len(filtered_choices)}/{len(all_choices)} 项)',
                    choices=selectable_options,
                    carousel=True
                )
            ]
            
            # 显示当前过滤状态
            status_msg = f"当前过滤: '{filter_text}'" if filter_text else "过滤: 无"
            print(f"\n📌 {status_msg}")
            
            # 获取用户选择
            answer = inquirer.prompt(question)
            if not answer:
                print("用户取消选择")
                return None
            
            selection = answer['selection']
            
            # 处理特殊操作
            if selection == 'ACTION_FILTER':
                filter_q = [
                    inquirer.Text(
                        'filter',
                        message='输入关键词过滤特征提取器',
                        default=filter_text
                    )
                ]
                filter_ans = inquirer.prompt(filter_q)
                if filter_ans:
                    new_filter = filter_ans['filter'].strip()
                    if new_filter != filter_text:
                        filter_text = new_filter
                        if filter_text:
                            filtered_choices = [
                                (dn, en) for dn, en in all_choices 
                                if filter_text.lower() in en.lower()
                            ]
                            print(f"🔍 过滤结果: {len(filtered_choices)} 个匹配项")
                        else:
                            filtered_choices = all_choices[:]
                            
            elif selection == 'ACTION_CANCEL':
                print("❌ 用户取消选择")
                return None
                
            elif selection == 'SEPARATOR':
                continue  # 忽略分隔线
                
            else:
                # 选择了具体特征提取器
                selected_display = next((dn for dn, en in filtered_choices if en == selection), "选中特征提取器")
                print(f"✅ 选择特征提取器: {selected_display}")
                return extractors[selection]
                
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消选择")
        return None
    except Exception as e:
        print(f"❌ 选择特征提取器时出错: {e}")
        return None


def get_test_smiles():
    """
    获取预定义的测试SMILES列表
    
    Returns:
        测试SMILES列表
    """
    # 预定义的一些测试SMILES
    test_smiles = [
        "C",       # 甲烷
        "CC",      # 乙烷
        "CCC",     # 丙烷
        "CCO",     # 乙醇
        "c1ccccc1" # 苯
    ]
    
    return test_smiles


def test_single_molecule_feature_extractor(feature_extractor_class, smile, types=None):
    """
    测试单个分子特征提取器
    
    Args:
        feature_extractor_class: 特征提取器类
        smile: SMILES字符串
        types: 原子类型映射字典
        
    Returns:
        提取的特征张量
    """
    if types is None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    
    try:
        from mol_evo.core.utils.molecule import smile_to_graph_xyz
        
        # 将SMILES转换为图结构
        x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smile, types)
        
        if x is None:
            print(f"错误: 无法生成分子 '{smile}' 的图结构")
            return None
        
        # 创建模型实例
        node_feature_dim = x.size(1)
        model = feature_extractor_class(node_feature_dim=node_feature_dim)
        
        # 创建PyG Data对象
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            features = model(graph_data)
        
        return {
            "smile": smile,
            "node_count": x.shape[0],
            "node_feature_dim": x.shape[1],
            "edge_count": edge_index.shape[1],
            "features": features,
            "features_shape": features.shape
        }
        
    except Exception as e:
        print(f"测试分子 '{smile}' 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_feature_extractor_info(feature_extractor_class):
    """
    获取特征提取器信息
    
    Args:
        feature_extractor_class: 特征提取器类
        
    Returns:
        包含特征提取器信息的字典
    """
    try:
        # 创建一个示例模型来获取信息
        dummy_model = feature_extractor_class(node_feature_dim=10)
        
        return {
            "class_name": feature_extractor_class.__name__,
            "module": feature_extractor_class.__module__,
            "parameter_count": sum(p.numel() for p in dummy_model.parameters())
        }
    except Exception as e:
        print(f"获取特征提取器信息时出错: {e}")
        import traceback
        traceback.print_exc()
        return None