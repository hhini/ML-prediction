import os
import pickle

# 尝试导入joblib
try:
    import joblib
except ImportError:
    joblib = None
    print("Warning: joblib module not found. Some model formats may not be supported.")

import pandas as pd
from typing import Optional, Any

class ModelService:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model = None
        self.model_loaded = False
    
    def find_model_file(self) -> Optional[str]:
        """查找模型文件"""
        model_extensions = ['.pkl', '.pickle', '.joblib', '.model']
        
        for file in os.listdir(self.model_dir):
            if any(file.endswith(ext) for ext in model_extensions):
                return os.path.join(self.model_dir, file)
        
        return None
    
    def load_model(self) -> bool:
        """加载模型文件"""
        model_file = self.find_model_file()
        
        if not model_file:
            print(f"No model file found in {self.model_dir}")
            return False
        
        try:
            if model_file.endswith('.joblib'):
                if joblib is None:
                    print("Error: joblib module not found. Cannot load .joblib files.")
                    return False
                self.model = joblib.load(model_file)
            else:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
            
            self.model_loaded = True
            print(f"Model loaded successfully from {model_file}")
            print(f"最终模型类型: {type(self.model).__name__}")
            print(f"模型是否有predict方法: {hasattr(self.model, 'predict')}")
            print(f"模型是否有predict_proba方法: {hasattr(self.model, 'predict_proba')}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, features: list) -> Optional[Any]:
        """使用模型进行预测"""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        try:
            # 处理缺失值
            # 这里可以根据模型的要求进行处理
            # 例如，用0填充或使用其他策略
            processed_features = [0 if f is None else f for f in features]
            
            # 创建DataFrame，使用正确的列名
            feature_names = [
                '如果使用马桶，是否习惯盖马桶盖',
                '家庭厕所类型',
                '居住房屋所有权',
                '零食的食用频率',
                '家中蔬菜的购买方式'
            ]
            
            df = pd.DataFrame([processed_features], columns=feature_names)
            
            # 确保输入形状正确
            if hasattr(self.model, 'predict_proba'):
                # 对于分类模型，返回概率
                return self.model.predict_proba(df)[0]
            else:
                # 对于回归模型，返回预测值
                return self.model.predict(df)[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        model_file = self.find_model_file()
        
        return {
            'model_loaded': self.model_loaded,
            'model_file': model_file,
            'model_type': type(self.model).__name__ if self.model else 'None'
        }