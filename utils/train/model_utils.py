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
    from mol_evo.core.models.v0 import ModelFactory
    import inquirer
except ImportError:
    ModelFactory = None
    inquirer = None


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
        print(f"只有一个可用模型类型: {model_types[0]}")
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
                    print(f"{i+1}. {model_type}")
            
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
                        print(f"选择模型类型: {selected_model}")
                        return selected_model
                    else:
                        print(f"无效选择: {choice_idx + 1}, 请输入 1 到 {len(filtered_models)} 之间的数字")
                    
                else:
                    # 视为过滤关键词
                    filter_text = choice
                    filtered_models = [m for m in model_types 
                                     if filter_text.lower() in m.lower()]
                    if not filtered_models:
                        print(f"没有找到包含 '{filter_text}' 的模型类型")
                        # 保持当前过滤状态，让用户可以修改关键词
                    else:
                        print(f"已过滤，找到 {len(filtered_models)} 个匹配项")
                        
            except KeyboardInterrupt:
                print("\n用户取消选择")
                return None
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")
                continue
    
    # 高级模式：使用inquirer库
    try:
        # 创建选项列表
        all_choices = [(model_type, model_type) for model_type in model_types]
        
        # 动态过滤选择主循环
        filtered_choices = all_choices[:]
        filter_text = ""
        
        while True:
            # 构建可选择的选项列表
            selectable_options = []
            
            # 添加过滤后的模型选项
            for display_name, model_type in filtered_choices:
                selectable_options.append((display_name, model_type))
            
            # 添加分隔线和功能选项
            selectable_options.append(('────────────────', 'SEPARATOR'))
            selectable_options.append(('❌ 取消选择', 'ACTION_CANCEL'))
            selectable_options.append(('🔍 输入过滤关键词', 'ACTION_FILTER'))
            
            # 创建选择问题
            question = [
                inquirer.List(
                    'selection',
                    message=f'选择模型类型 (显示 {len(filtered_choices)}/{len(all_choices)} 项)',
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
                        message='输入关键词过滤模型类型',
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
                                (dn, mt) for dn, mt in all_choices 
                                if filter_text.lower() in mt.lower()
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
                # 选择了具体模型类型
                selected_display = next((dn for dn, mt in filtered_choices if mt == selection), "选中模型类型")
                print(f"✅ 选择模型类型: {selected_display}")
                return selection
                
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消选择")
        return None
    except Exception as e:
        print(f"❌ 选择模型类型时出错: {e}")
        return None