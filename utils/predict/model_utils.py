"""
模型工具模块
包含与模型加载、查找、选择等相关的工具函数
"""

import os
import logging
import json
import sys
import inquirer


# 设置项目根目录路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

def setup_logger(log_level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        log_level: 日志级别
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger('prediction')
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建格式器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
    
    return logger


def find_model_files(base_path=None):
    """
    自动查找模型文件
    
    Args:
        base_path: 基础路径，如果为None则使用默认路径
        
    Returns:
        模型文件列表
    """
    # 递归搜索所有 .pth 文件
    if base_path is None:
        base_path = os.path.join(project_root, 'mol_evo', 'output', 'v0')
    model_files = []
    
    # 遍历所有子目录查找 .pth 文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth'):
                # 过滤掉以 checkpoint_epoch_ 开头的检查点文件
                if not file.startswith('checkpoint_epoch_'):
                    model_files.append(os.path.join(root, file))
    
    return sorted(model_files)


def select_models_interactively(model_files):
    """
    交互式选择模型（支持动态过滤）
    
    Args:
        model_files: 模型文件列表
        
    Returns:
        选择的模型文件路径列表
    """
    if not inquirer:
        print("警告: 未安装 inquirer 库，使用基础交互模式")
        if not model_files:
            print("未找到任何模型文件")
            return []
        
        if len(model_files) == 1:
            print(f"找到一个模型文件: {os.path.relpath(model_files[0], project_root)}")
            return [model_files[0]]
        
        # 基础模式：纯文本交互
        filtered_models = model_files[:]
        filter_text = ""
        base_display_path = os.path.join('mol_evo', 'output', 'v0')
        
        while True:
            # 更新显示名称
            display_models = []
            for model_file in filtered_models:
                relative_path = os.path.relpath(model_file, os.path.join(project_root, base_display_path))
                dir_name = os.path.dirname(relative_path)
                file_name = os.path.basename(model_file)
                display_name = os.path.join(dir_name, file_name) if dir_name else file_name
                display_models.append((model_file, display_name))
            
            print(f"\n找到 {len(model_files)} 个模型文件 (当前显示 {len(display_models)} 个):")
            if filter_text:
                print(f"当前过滤条件: '{filter_text}'")
            
            if not display_models:
                print("没有匹配的模型文件")
            else:
                for i, (model_file, display_name) in enumerate(display_models):
                    print(f"{i+1}. {display_name}")
            
            print("\n操作说明:")
            print("  • 直接输入数字选择模型 (如: 1 或 1,3,5)")
            print("  • 输入关键词进行过滤 (如: epoch100)")
            print("  • 按回车选择最新模型")
            print("  • 输入 'clear' 清除过滤条件")
            print("  • 输入 'quit' 退出选择")
            
            try:
                choice = input("\n请选择或输入过滤条件: ").strip()
                
                if not choice:
                    # 选择最新的模型
                    latest_model = max(model_files, key=os.path.getctime)
                    relative_path = os.path.relpath(latest_model, os.path.join(project_root, base_display_path))
                    dir_name = os.path.dirname(relative_path)
                    file_name = os.path.basename(latest_model)
                    display_name = os.path.join(dir_name, file_name) if dir_name else file_name
                    print(f"选择最新模型: {display_name}")
                    return [latest_model]
                
                if choice.lower() == 'quit':
                    print("用户取消选择")
                    return []
                    
                if choice.lower() == 'clear':
                    filtered_models = model_files[:]
                    filter_text = ""
                    continue
                
                # 检查是否为数字选择
                if all(c.isdigit() or c in ', ' for c in choice):
                    choices = [int(c.strip()) - 1 for c in choice.split(',') if c.strip().isdigit()]
                    selected_models = []
                    for choice_idx in choices:
                        if 0 <= choice_idx < len(display_models):
                            selected_models.append(display_models[choice_idx][0])
                        else:
                            print(f"无效选择: {choice_idx + 1}, 请输入 1 到 {len(display_models)} 之间的数字")
                            break
                    else:  # 只有当所有选择都有效时才返回
                        if selected_models:
                            return selected_models
                    
                else:
                    # 视为过滤关键词
                    filter_text = choice
                    filtered_models = [m for m in model_files 
                                     if filter_text.lower() in os.path.relpath(m, project_root).lower()]
                    if not filtered_models:
                        print(f"没有找到包含 '{filter_text}' 的模型文件")
                        # 保持当前过滤状态，让用户可以修改关键词
                    else:
                        print(f"已过滤，找到 {len(filtered_models)} 个匹配项")
                        
            except KeyboardInterrupt:
                print("\n用户取消选择")
                return []
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")
                continue
    
    # 高级模式：使用inquirer库
    if not model_files:
        print("未找到任何模型文件")
        return []

    if len(model_files) == 1:
        model_file = model_files[0]
        base_display_path = os.path.join('mol_evo', 'output', 'v0')
        relative_path = os.path.relpath(model_file, os.path.join(project_root, base_display_path))
        dir_name = os.path.dirname(relative_path)
        file_name = os.path.basename(model_file)
        display_name = os.path.join(dir_name, file_name) if dir_name else file_name
        print(f"找到一个模型文件: {display_name}")
        return [model_file]

    # 准备带时间戳的显示数据
    base_display_path = os.path.join('mol_evo', 'output', 'v0')
    display_items = []
    
    for model_file in model_files:
        relative_path = os.path.relpath(model_file, os.path.join(project_root, base_display_path))
        dir_name = os.path.dirname(relative_path)
        file_name = os.path.basename(model_file)
        display_name = os.path.join(dir_name, file_name) if dir_name else file_name
        mtime = os.path.getctime(model_file)
        display_items.append((model_file, display_name, mtime))
    
    # 按修改时间排序，最新的在前
    display_items.sort(key=lambda x: x[2], reverse=True)
    
    # 创建选项列表
    all_choices = [(display_name, model_file) for model_file, display_name, _ in display_items]
    
    # 动态过滤选择主循环
    try:
        filtered_choices = all_choices[:]
        filter_text = ""
        
        while True:
            # 构建可选择的选项列表
            selectable_options = [
                ('🔍 输入过滤关键词', 'ACTION_FILTER'),
                ('🆕 选择最新模型', 'ACTION_LATEST'),
                ('📋 显示全部模型', 'ACTION_SHOW_ALL'),
                ('❌ 取消选择', 'ACTION_CANCEL')
            ]
            
            # 添加分隔线
            selectable_options.append(('────────────────', 'SEPARATOR'))
            
            # 添加过滤后的模型选项
            for display_name, model_file in filtered_choices:
                selectable_options.append((display_name, model_file))
            
            # 创建选择问题
            question = [
                inquirer.List(
                    'selection',
                    message=f'选择模型 (显示 {len(filtered_choices)}/{len(all_choices)} 项)',
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
                return []
            
            selection = answer['selection']
            
            # 处理特殊操作
            if selection == 'ACTION_FILTER':
                filter_q = [
                    inquirer.Text(
                        'filter',
                        message='输入关键词过滤模型',
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
                                (dn, mf) for dn, mf in all_choices 
                                if filter_text.lower() in dn.lower()
                            ]
                            print(f"🔍 过滤结果: {len(filtered_choices)} 个匹配项")
                        else:
                            filtered_choices = all_choices[:]
                            print("📌 显示全部模型")
                            
            elif selection == 'ACTION_LATEST':
                # 选择最新模型
                latest_model = max(model_files, key=os.path.getctime)
                latest_display = next((dn for dn, mf in all_choices if mf == latest_model), "最新模型")
                print(f"✅ 选择最新模型: {latest_display}")
                return [latest_model]
                
            elif selection == 'ACTION_SHOW_ALL':
                filtered_choices = all_choices[:]
                filter_text = ""
                print("📌 显示全部模型")
                
            elif selection == 'ACTION_CANCEL':
                print("❌ 用户取消选择")
                return []
                
            elif selection == 'SEPARATOR':
                continue  # 忽略分隔线
                
            else:
                # 选择了具体模型
                selected_display = next((dn for dn, mf in filtered_choices if mf == selection), "选中模型")
                print(f"✅ 选择模型: {selected_display}")
                return [selection]
                
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消选择")
        return []
    except Exception as e:
        print(f"❌ 选择模型时出错: {e}")
        return []


def load_property_stats(model_dir, config_file_name="training_process.json"):
# def load_property_stats(model_dir, config_file_name="training_config.json"):
    """
    从模型目录加载属性统计信息
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        属性统计信息字典
    """
    stats_file = os.path.join(model_dir, config_file_name)
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            # 从训练数据中提取属性统计信息
            if 'property_stats' in data:
                return data['property_stats']
            else:
                print("警告: 训练数据中未找到属性统计信息，将无法进行反标准化")
                return None
    else:
        print(f"[load_property_stats]警告: 未找到训练数据文件 {stats_file}，将无法进行反标准化")
        import traceback
        traceback.print_exc()
        return None


def load_training_params(model_dir, config_file_name = "training_config.json"):
    """
    从模型目录加载训练参数
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        训练参数字典
    """
    stats_file = os.path.join(model_dir, config_file_name)
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            # 从训练数据中提取训练参数
            if 'training_params' in data:
                return data['training_params']
            else:
                print("警告: 训练数据中未找到训练参数")
                return None
    else:
        print(f"警告: 未找到训练数据文件 {stats_file}")
        return None


def is_model_normalized(model_dir):
    """
    检查模型是否使用了标准化数据进行训练
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        bool: 如果模型使用了标准化数据则返回True，否则返回False
    """
    # 首先尝试从training_process.json获取标准化信息
    process_file = os.path.join(model_dir, "training_process.json")
    if os.path.exists(process_file):
        with open(process_file, 'r') as f:
            try:
                data = json.load(f)
                # 检查是否有property_stats信息
                if 'property_stats' in data and data['property_stats']:
                    return True
            except Exception as e:
                print(f"读取training_process.json时出错: {e}")
    
    # 兼容旧版：从training_data.json获取标准化信息
    stats_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            try:
                data = json.load(f)
                # 检查训练参数中是否启用了标准化
                if 'training_params' in data and 'normalize' in data['training_params']:
                    return data['training_params']['normalize']
                # 检查是否有属性统计信息
                elif 'property_stats' in data and data['property_stats']:
                    return True
            except Exception as e:
                print(f"读取training_data.json时出错: {e}")
    
    return False