from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import glob
from pathlib import Path

app = FastAPI(
    title="Molecule Evolution Training Data API",
    description="API for accessing training data from molecule evolution models",
    version="1.0.0"
)

class TrainingDataResponse(BaseModel):
    training_params: Dict[str, Any]
    model_params: Dict[str, Any]
    property_stats: Dict[str, Any]
    test_metrics: Dict[str, Any]
    losses: Dict[str, Any]

class TrainingDataListItem(BaseModel):
    file_path: str
    training_params: Dict[str, Any]
    model_params: Dict[str, Any]
    property_stats: Dict[str, Any]
    test_metrics: Dict[str, Any]

class TrainingDataListResponse(BaseModel):
    total_num: int
    items: List[TrainingDataListItem]

def load_training_data(file_path: str) -> dict:
    """加载训练数据文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file {file_path}: {str(e)}")

def get_all_training_data_files() -> List[str]:
    """获取所有训练数据文件路径"""
    # 查找项目根目录下的所有训练数据文件
    project_root = Path(__file__).parent.parent.parent
    pattern = str(project_root) + "/mol_evo/output/**/training_data.json"
    return glob.glob(pattern, recursive=True)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Molecule Evolution Training Data API is running"}

@app.get("/training-data", response_model=TrainingDataListResponse)
async def list_training_data(
    limit: int = Query(10, description="Number of items to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    列出所有可用的训练数据文件
    
    参数:
    - limit: 返回的项目数量
    - offset: 分页偏移量
    """
    try:
        files = get_all_training_data_files()
        files.sort()  # 排序文件列表
        
        total_num = len(files)
        
        # 应用分页
        paginated_files = files[offset:offset+limit]
        
        result = []
        for file_path in paginated_files:
            data = load_training_data(file_path)
            result.append(TrainingDataListItem(
                file_path=file_path,
                training_params=data.get("training_params", {}),
                model_params=data.get("model_params", {}),
                property_stats=data.get("property_stats", {}),
                test_metrics=data.get("test_metrics", {})
            ))
        
        return TrainingDataListResponse(
            total_num=total_num,
            items=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing training data: {str(e)}")

@app.get("/training-data/{file_id}", response_model=TrainingDataResponse)
async def get_training_data(file_id: int):
    """
    获取特定训练数据文件的内容
    
    参数:
    - file_id: 文件索引 (从list_training_data获取)
    """
    try:
        files = get_all_training_data_files()
        files.sort()  # 确保排序一致
        
        if file_id < 0 or file_id >= len(files):
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = files[file_id]
        data = load_training_data(file_path)
        
        return TrainingDataResponse(
            training_params=data.get("training_params", {}),
            model_params=data.get("model_params", {}),
            property_stats=data.get("property_stats", {}),
            test_metrics=data.get("test_metrics", {}),
            losses=data.get("losses", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training data: {str(e)}")

@app.get("/training-data/path/{path:path}", response_model=TrainingDataResponse)
async def get_training_data_by_path(path: str):
    """
    根据文件路径获取训练数据
    
    参数:
    - path: 文件路径 (相对于项目根目录)
    """
    try:
        # 确保路径指向的是训练数据文件
        if not path.endswith("training_data.json"):
            raise HTTPException(status_code=400, detail="Path must point to a training_data.json file")
        
        project_root = Path(__file__).parent.parent.parent
        full_path = project_root / path
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        data = load_training_data(full_path)
        
        return TrainingDataResponse(
            training_params=data.get("training_params", {}),
            model_params=data.get("model_params", {}),
            property_stats=data.get("property_stats", {}),
            test_metrics=data.get("test_metrics", {}),
            losses=data.get("losses", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6000)