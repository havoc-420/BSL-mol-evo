"""_summary_
多色 log，暂时是准备弃用了。
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[37m',       # 白色（取消绿色）
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # 获取日志级别对应的颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        # 重置颜色
        reset = self.COLORS['RESET']
        
        # 应用颜色到整个日志消息
        record.levelname = f"{color}{record.levelname}{reset}"
        
        # 只对消息内容应用颜色，不包括时间戳等其他信息
        original_msg = record.getMessage()
        record.msg = f"{color}{original_msg}{reset}"
        record.message = record.msg
        
        return super().format(record)


class TrainingLogger:
    """训练过程专用日志记录器"""
    
    def __init__(self, name: str = "TrainingLogger", level: int = logging.INFO):
        """初始化训练日志记录器"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # 创建彩色格式化器
            formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # 添加处理器到日志记录器
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def log_progress(self, epoch: int, total_epochs: int, train_loss: float, 
                     val_loss: Optional[float] = None, metrics: Optional[dict] = None):
        """
        记录训练进度
        
        Args:
            epoch: 当前轮次
            total_epochs: 总轮次
            train_loss: 训练损失
            val_loss: 验证损失（可选）
            metrics: 额外的评估指标（可选）
        """
        # 根据损失值确定颜色
        train_loss_color = self._get_loss_color(train_loss)
        
        progress_msg = f"Epoch [{epoch}/{total_epochs}], Train Loss: {train_loss_color}{train_loss:.6f}\033[0m"
        
        if val_loss is not None:
            val_loss_color = self._get_loss_color(val_loss)
            progress_msg += f", Val Loss: {val_loss_color}{val_loss:.6f}\033[0m"
        
        self.logger.info(progress_msg)
        
        # 记录额外指标
        if metrics:
            metrics_msg = "  Metrics - "
            metric_parts = []
            
            # RMSE指标
            if 'rmse' in metrics:
                rmse_color = self._get_metric_color(metrics['rmse'], 1.0, 0.5)
                metric_parts.append(f"RMSE: {rmse_color}{metrics['rmse']:.6f}\033[0m")
            
            # MAE指标
            if 'mae' in metrics:
                mae_color = self._get_metric_color(metrics['mae'], 1.0, 0.5)
                metric_parts.append(f"MAE: {mae_color}{metrics['mae']:.6f}\033[0m")
            
            # R²指标
            if 'r2' in metrics:
                r2_color = self._get_r2_color(metrics['r2'])
                metric_parts.append(f"R²: {r2_color}{metrics['r2']:.6f}\033[0m")
            
            # 阈值准确率指标
            if 'threshold_accs' in metrics:
                threshold_accs = metrics['threshold_accs']
                if 'all_0.1' in threshold_accs:
                    acc_color = self._get_accuracy_color(threshold_accs['all_0.1'])
                    metric_parts.append(f"Acc(All@0.1): {acc_color}{threshold_accs['all_0.1']:.4f}\033[0m")
                if 'mean_0.1' in threshold_accs:
                    acc_color = self._get_accuracy_color(threshold_accs['mean_0.1'])
                    metric_parts.append(f"Acc(Mean@0.1): {acc_color}{threshold_accs['mean_0.1']:.4f}\033[0m")
            
            metrics_msg += ", ".join(metric_parts)
            self.logger.info(metrics_msg)
    
    def _get_loss_color(self, loss: float) -> str:
        """
        根据损失值返回相应的颜色代码
        
        Args:
            loss: 损失值
            
        Returns:
            ANSI颜色代码
        """
        if loss < 0.1:
            return '\033[32m'  # 绿色 - 很好
        elif loss < 0.5:
            return '\033[36m'  # 青色 - 好
        elif loss < 1.0:
            return '\033[33m'  # 黄色 - 一般
        else:
            return '\033[31m'  # 红色 - 较差
    
    def _get_metric_color(self, value: float, good_threshold: float, moderate_threshold: float) -> str:
        """
        根据指标值返回相应的颜色代码（适用于RMSE、MAE等越小越好的指标）
        
        Args:
            value: 指标值
            good_threshold: 好的阈值
            moderate_threshold: 一般的阈值
            
        Returns:
            ANSI颜色代码
        """
        if value < good_threshold:
            return '\033[32m'  # 绿色 - 很好
        elif value < moderate_threshold:
            return '\033[36m'  # 青色 - 好
        else:
            return '\033[31m'  # 红色 - 较差
    
    def _get_r2_color(self, r2: float) -> str:
        """
        根据R²值返回相应的颜色代码
        
        Args:
            r2: R²值
            
        Returns:
            ANSI颜色代码
        """
        if r2 > 0.8:
            return '\033[32m'  # 绿色 - 很好
        elif r2 > 0.6:
            return '\033[36m'  # 青色 - 好
        elif r2 > 0.4:
            return '\033[33m'  # 黄色 - 一般
        else:
            return '\033[31m'  # 红色 - 较差
    
    def _get_accuracy_color(self, accuracy: float) -> str:
        """
        根据准确率返回相应的颜色代码
        
        Args:
            accuracy: 准确率值
            
        Returns:
            ANSI颜色代码
        """
        if accuracy > 0.9:
            return '\033[32m'  # 绿色 - 很好
        elif accuracy > 0.7:
            return '\033[36m'  # 青色 - 好
        elif accuracy > 0.5:
            return '\033[33m'  # 黄色 - 一般
        else:
            return '\033[31m'  # 红色 - 较差