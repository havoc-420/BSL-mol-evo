#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具函数
"""

from .training_utils import setup_logger


class DualLogger:
    """
    双通道日志记录器，支持同时向文件和控制台输出日志
    
    Attributes:
        file_logger: 仅向文件输出日志的记录器
        console_logger: 同时向文件和控制台输出日志的记录器
    """
    
    def __init__(self, model_dir):
        """
        初始化DualLogger实例
        
        Args:
            model_dir: 模型目录路径
        """
        # 为确保两个logger独立，使用不同的logger名称
        self.file_logger = setup_logger(model_dir, logger_name='training_file', console_output=False)
        self.console_logger = setup_logger(model_dir, logger_name='training_console', console_output=True)
    
    def info(self, message, to_console=True):
        """
        记录信息日志，默认同时输出到文件和控制台
        
        Args:
            message (str): 日志消息
            to_console (bool): 是否同时输出到控制台，默认为True
        """
        # 始终记录到文件
        self.file_logger.info(message)
        
        # 根据参数决定是否输出到控制台（默认输出）
        if to_console:
            self.console_logger.info(message)
    
    def warning(self, message, to_console=True):
        """
        记录警告日志，默认同时输出到文件和控制台
        
        Args:
            message (str): 日志消息
            to_console (bool): 是否同时输出到控制台，默认为True
        """
        # 始终记录到文件
        self.file_logger.warning(message)
        
        # 根据参数决定是否输出到控制台（默认输出）
        if to_console:
            self.console_logger.warning(message)
    
    def error(self, message, to_console=True):
        """
        记录错误日志，默认同时输出到文件和控制台
        
        Args:
            message (str): 日志消息
            to_console (bool): 是否同时输出到控制台，默认为True
        """
        # 始终记录到文件
        self.file_logger.error(message)
        
        # 根据参数决定是否输出到控制台（默认输出）
        if to_console:
            self.console_logger.error(message)
    
    def debug(self, message, to_console=True):
        """
        记录调试日志，默认同时输出到文件和控制台
        
        Args:
            message (str): 日志消息
            to_console (bool): 是否同时输出到控制台，默认为True
        """
        # 始终记录到文件
        self.file_logger.debug(message)
        
        # 根据参数决定是否输出到控制台（默认输出）
        if to_console:
            self.console_logger.debug(message)