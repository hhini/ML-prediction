import os
import joblib
import pandas as pd
from typing import Optional, Any

class ModelService:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model = None
        self.model_loaded = False
        self.load_error = None
        
        # 初始化时自动加载模型
        self.load_model()
    
    def find_model_file(self) -> Optional[str]:
        """查找模型文件"""
        model_extensions = ['.joblib', '.pkl', '.pickle', '.model']
        
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
            print(f"Attempting to load model from {model_file}")
            # 直接使用joblib加载模型
            self.model = joblib.load(model_file)
            print(f"Raw model loaded. Type: {type(self.model)}")
            
            # 处理模型数据，如果是元组，尝试找到模型对象
            if isinstance(self.model, tuple):
                print(f"Model is a tuple of length {len(self.model)}")
                found_predict = False
                for i, item in enumerate(self.model):
                    print(f"Checking item {i}: type={type(item)}")
                    try:
                        if hasattr(item, 'predict'):
                            print(f"Item {i} has 'predict' method.")
                            self.model = item
                            found_predict = True
                            break
                    except Exception as e:
                        print(f"Error checking item {i}: {e}")
                
                # 如果没有找到，使用第一个元素
                if not found_predict and isinstance(self.model, tuple):
                    print("No item with 'predict' found, using first element.")
                    self.model = self.model[0]
            
            self.model_loaded = True
            print(f"Model loaded successfully from {model_file}")
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
            'model_type': type(self.model).__name__ if self.model else 'None',
            'load_error': self.load_error
        }