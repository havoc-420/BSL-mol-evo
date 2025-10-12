"""
训练指标记录和管理工具
"""

from typing import Dict, List, Optional, Any
import json
from tqdm import tqdm
import torch


class TrainingMetricsRecorder:
    """
    训练指标记录器，用于统一管理训练过程中的各种指标
    """
    
    def __init__(self):
        # 训练损失历史
        self.train_losses: List[float] = []
        
        # 验证损失历史
        self.val_losses: List[float] = []
        
        # 验证集评估指标历史 (每N个epoch记录一次)
        self.val_metrics_history: List[Dict[str, Any]] = []
        
        # 验证集RMSE历史
        self.val_rmses: List[float] = []
        
        # 验证集MAE历史
        self.val_maes: List[float] = []
        
        # 验证集R2历史
        self.val_r2s: List[float] = []
        
        # 测试集最终评估指标
        self.test_metrics: Optional[Dict[str, Any]] = None
        
        # 训练参数
        self.training_params: Optional[Dict[str, Any]] = None
        
        # 模型参数
        self.model_params: Optional[Dict[str, Any]] = None
        
        # 属性统计信息
        self.property_stats: Optional[Dict[str, Any]] = None
        
        # 早停相关信息
        self.early_stopping_info: Optional[Dict[str, Any]] = None
        
        # 学习率调度信息
        self.lr_scheduler_info: Optional[Dict[str, Any]] = None
    
    def record_train_loss(self, loss: float):
        """记录训练损失"""
        self.train_losses.append(loss)
    
    def record_val_loss(self, loss: float):
        """记录验证损失"""
        self.val_losses.append(loss)
    
    def record_val_metrics(self, epoch: int, loss: float, mse: float, rmse: float, 
                          mae: float, r2: float):
        """记录验证集评估指标"""
        metrics = {
            "epoch": epoch,
            "val_loss": loss,
            "val_mse": mse,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2": r2
        }
        self.val_metrics_history.append(metrics)
        
        # 同时记录到各自的列表中
        self.val_rmses.append(rmse)
        self.val_maes.append(mae)
        self.val_r2s.append(r2)
    
    def record_test_metrics(self, test_loss: float, rmse: float, mae: float, r2: float,
                           threshold_accs: Dict[float, float], dataset_info: Dict[str, int],
                           original_scale_metrics: Optional[Dict[str, float]] = None):
        """记录测试集最终评估指标"""
        self.test_metrics = {
            "test_loss": test_loss,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "threshold_accs": threshold_accs,
            "dataset_info": dataset_info
        }
        
        if original_scale_metrics:
            self.test_metrics["original_scale_metrics"] = original_scale_metrics
    
    def set_training_params(self, params: Dict[str, Any]):
        """设置训练参数"""
        self.training_params = params
    
    def set_model_params(self, params: Dict[str, Any]):
        """设置模型参数"""
        self.model_params = params
    
    def set_property_stats(self, stats: Dict[str, Any]):
        """设置属性统计信息"""
        self.property_stats = stats
    
    def record_early_stopping(self, best_epoch: int, best_loss: float, patience_counter: int):
        """记录早停信息"""
        self.early_stopping_info = {
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "patience_counter": patience_counter
        }
    
    def record_lr_scheduler(self, scheduler_type: str, **kwargs):
        """记录学习率调度器信息"""
        self.lr_scheduler_info = {
            "type": scheduler_type,
            **kwargs
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """将所有记录的指标转换为字典格式"""
        result = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics_history": self.val_metrics_history,
            "val_rmses": self.val_rmses,
            "val_maes": self.val_maes,
            "val_r2s": self.val_r2s
        }
        
        if self.test_metrics:
            result["test_metrics"] = self.test_metrics
            
        if self.training_params:
            result["training_params"] = self.training_params
            
        if self.model_params:
            result["model_params"] = self.model_params
            
        if self.property_stats:
            result["property_stats"] = self.property_stats
            
        if self.early_stopping_info:
            result["early_stopping_info"] = self.early_stopping_info
            
        if self.lr_scheduler_info:
            result["lr_scheduler_info"] = self.lr_scheduler_info
            
        return result
    
    def save_to_json(self, filepath: str):
        """将指标保存为JSON文件"""
        data = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_latest_val_metrics(self) -> Optional[Dict[str, Any]]:
        """获取最新的验证指标"""
        if self.val_metrics_history:
            return self.val_metrics_history[-1]
        return None
    
    def log_epoch_progress(self, logger, epoch: int, total_epochs: int, 
                          epoch_loss: float, val_loss: Optional[float] = None,
                          log_epoch_progress=None):
        """记录epoch进度信息到日志"""
        
        # 获取最新的验证指标
        latest_metrics = self.get_latest_val_metrics()
        if latest_metrics:
            val_r2 = latest_metrics.get("val_r2")
            val_mae = latest_metrics.get("val_mae")
        else:
            val_r2 = None
            val_mae = None
            
        # 将epoch_loss封装为Tensor以匹配log_epoch_progress函数期望的类型
        log_epoch_progress(logger, epoch, total_epochs, torch.tensor(epoch_loss), val_loss, val_r2, val_mae)
        
        # 控制台只显示基本进度信息
        message = f"Epoch [{epoch+1}/{total_epochs}], Train Loss: {epoch_loss:.6f}"
        if val_loss is not None:
            message += f", Val Loss: {val_loss:.6f}"
            if val_r2 is not None and val_mae is not None:
                message += f", R²: {val_r2:.4f}, MAE: {val_mae:.4f}"
        tqdm.write(message)
    
    def get_best_val_loss(self) -> Optional[float]:
        """获取最佳验证损失"""
        if self.val_losses:
            return min(self.val_losses)
        return None
    
    def get_best_val_metrics(self) -> Optional[Dict[str, Any]]:
        """获取最佳验证指标"""
        if not self.val_metrics_history:
            return None
        
        # 找到验证损失最小的记录
        best_record = min(self.val_metrics_history, key=lambda x: x["val_loss"])
        return best_record


def create_training_metrics_recorder() -> TrainingMetricsRecorder:
    """创建训练指标记录器实例"""
    return TrainingMetricsRecorder()