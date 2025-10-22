"""
训练模型工具模块
包含与模型类型选择等相关的工具函数
"""

import sys
import os

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    from mol_evo.core.models.v0 import ModelFactory, get_model_display_name
    import inquirer
except ImportError:
    ModelFactory = None
    inquirer = None
    get_model_display_name = None


def select_model_type_interactively():
    """
    交互式选择模型类型（支持动态过滤）
    
    Returns:
        选择的模型类型字符串
    """
    if not ModelFactory:
        raise ImportError("无法导入ModelFactory")
    
    # 获取所有可用的模型类型
    model_types = ModelFactory.list_models()
    
    if not model_types:
        print("未找到任何可用的模型类型")
        return None

    if len(model_types) == 1:
        model_name = model_types[0]
        display_name = get_model_display_name(model_name) if get_model_display_name else model_name
        print(f"只有一个可用模型类型: {display_name} ({model_name})")
        return model_types[0]

    # 如果没有安装inquirer库，使用基础交互模式
    if not inquirer:
        print("警告: 未安装 inquirer 库，使用基础交互模式")
        
        # 基础模式：纯文本交互
        filtered_models = model_types[:]
        filter_text = ""
        
        while True:
            print(f"\n找到 {len(model_types)} 个模型类型 (当前显示 {len(filtered_models)} 个):")
            if filter_text:
                print(f"当前过滤条件: '{filter_text}'")
            
            if not filtered_models:
                print("没有匹配的模型类型")
            else:
                for i, model_type in enumerate(filtered_models):
                    display_name = get_model_display_name(model_type) if get_model_display_name else model_type
                    print(f"{i+1}. {display_name} ({model_type})")
            
            print("\n操作说明:")
            print("  • 直接输入数字选择模型类型 (如: 1)")
            print("  • 输入关键词进行过滤 (如: gcn)")
            print("  • 输入 'clear' 清除过滤条件")
            print("  • 输入 'quit' 退出选择")
            
            try:
                choice = input("\n请选择或输入过滤条件: ").strip()
                
                if choice.lower() == 'quit':
                    print("用户取消选择")
                    return None
                    
                if choice.lower() == 'clear':
                    filtered_models = model_types[:]
                    filter_text = ""
                    continue
                
                # 检查是否为数字选择
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(filtered_models):
                        selected_model = filtered_models[choice_idx]
                        display_name = get_model_display_name(selected_model) if get_model_display_name else selected_model
                        print(f"选择模型类型: {display_name} ({selected_model})")
                        return selected_model
                    else:
                        print(f"无效选择: {choice_idx + 1}, 请输入 1 到 {len(filtered_models)} 之间的数字")
                    
                else:
                    # 视为过滤关键词
                    filter_text = choice
                    filtered_models = [m for m in model_types 
                                     if filter_text.lower() in m.lower() or 
                                        (get_model_display_name and filter_text.lower() in get_model_display_name(m).lower())]
                    if not filtered_models:
                        print(f"没有找到包含 '{filter_text}' 的模型类型")
                        # 保持当前过滤状态，让用户可以修改关键词
                    else:
                        print(f"已过滤，找到 {len(filtered_models)} 个匹配项")
                        
            except KeyboardInterrupt:
                print("\n用户取消选择")
                return None
            except Exception as e:
                print(f"选择过程中发生错误: {e}")
                continue
    
    else:
        # 使用 inquirer 进行高级交互
        questions = [
            inquirer.List('model_type',
                         message="请选择要训练的模型类型",
                         choices=[(f"{get_model_display_name(m)} ({m})", m) for m in model_types] if get_model_display_name 
                                else model_types,
                         carousel=True)
        ]
        
        try:
            answers = inquirer.prompt(questions)
            if answers and 'model_type' in answers:
                selected_model = answers['model_type']
                display_name = get_model_display_name(selected_model) if get_model_display_name else selected_model
                print(f"选择模型类型: {display_name} ({selected_model})")
                return selected_model
            else:
                print("未选择模型类型")
                return None
        except KeyboardInterrupt:
            print("\n用户取消选择")
            return None
        except Exception as e:
            print(f"选择过程中发生错误: {e}")
            return None
