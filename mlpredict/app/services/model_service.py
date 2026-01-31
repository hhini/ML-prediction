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
        
        print("========================================")
        print("        初始化ModelService")
        print("========================================")
        print(f"初始模型目录: {model_dir}")
        print(f"初始模型目录是否存在: {os.path.exists(model_dir)}")
        
        # 尝试查找模型目录
        self.find_valid_model_dir()
    
    def find_valid_model_dir(self):
        """尝试查找有效的模型目录"""
        # 尝试不同的模型路径
        possible_model_dirs = [
            self.model_dir,
            os.path.join(os.path.dirname(__file__), '../../..', 'models'),
            os.path.join(os.path.dirname(__file__), '../../..', 'mlpredict', 'models'),
            os.path.join(os.path.abspath('.'), 'models'),
            os.path.join(os.path.abspath('.'), 'mlpredict', 'models'),
            # 添加当前文件目录作为备选
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), '..', '..'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'ui')
        ]
        
        print("\n尝试不同的模型路径:")
        for i, path in enumerate(possible_model_dirs):
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            print(f"{i+1}. {abs_path} - {'存在' if exists else '不存在'}")
            
            if exists and os.path.isdir(abs_path):
                # 检查是否有模型文件
                files = os.listdir(abs_path)
                model_files = [f for f in files if any(f.endswith(ext) for ext in ['.pkl', '.pickle', '.joblib', '.model'])]
                if model_files:
                    print(f"   找到模型文件: {model_files}")
                    self.model_dir = abs_path
                    print(f"   选择此路径作为模型目录")
                    break
        
        # 最后尝试直接在当前工作目录查找模型文件
        print("\n尝试直接在当前工作目录查找模型文件:")
        current_work_dir = os.getcwd()
        print(f"当前工作目录: {current_work_dir}")
        
        if os.path.exists(current_work_dir):
            files = os.listdir(current_work_dir)
            model_files = [f for f in files if any(f.endswith(ext) for ext in ['.pkl', '.pickle', '.joblib', '.model'])]
            if model_files:
                print(f"找到模型文件: {model_files}")
                self.model_dir = current_work_dir
                print(f"选择当前工作目录作为模型目录")
    
    def find_model_file(self) -> Optional[str]:
        """查找模型文件"""
        print("========================================")
        print("        查找模型文件")
        print("========================================")
        print(f"当前模型目录: {self.model_dir}")
        print(f"目录是否存在: {os.path.exists(self.model_dir)}")
        
        if not os.path.exists(self.model_dir):
            print(f"Error: 模型目录不存在: {self.model_dir}")
            return None
        
        try:
            model_extensions = ['.pkl', '.pickle', '.joblib', '.model']
            files = os.listdir(self.model_dir)
            print(f"目录中的文件: {files}")
            
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    model_file = os.path.join(self.model_dir, file)
                    print(f"找到模型文件: {model_file}")
                    return model_file
            
            print("Error: 未找到模型文件")
            return None
        except Exception as e:
            print(f"Error listing directory: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_model(self) -> bool:
        """加载模型文件"""
        print("========================================")
        print("        加载模型")
        print("========================================")
        
        model_file = self.find_model_file()
        
        if not model_file:
            print(f"No model file found in {self.model_dir}")
            return False
        
        print(f"尝试加载模型文件: {model_file}")
        print(f"文件是否存在: {os.path.exists(model_file)}")
        print(f"文件大小: {os.path.getsize(model_file) if os.path.exists(model_file) else 'N/A'} bytes")
        
        try:
            if model_file.endswith('.joblib'):
                if joblib is None:
                    print("Error: joblib module not found. Cannot load .joblib files.")
                    return False
                print("使用joblib加载模型...")
                model_data = joblib.load(model_file)
                print("使用joblib加载模型成功")
            else:
                print("使用pickle加载模型...")
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                print("使用pickle加载模型成功")
            
            print(f"模型数据类型: {type(model_data).__name__}")
            
            # 处理模型数据，如果是元组
            if isinstance(model_data, tuple):
                print(f"模型数据是元组，长度: {len(model_data)}")
                # 尝试找到具有predict方法的对象
                for i, item in enumerate(model_data):
                    print(f"元组元素 {i} 类型: {type(item).__name__}")
                    print(f"是否有predict方法: {hasattr(item, 'predict')}")
                    if hasattr(item, 'predict'):
                        self.model = item
                        print(f"从元组中找到模型对象: {type(self.model).__name__}")
                        break
                # 如果没有找到，使用第一个元素
                if not self.model:
                    self.model = model_data[0]
                    print(f"使用元组的第一个元素作为模型: {type(self.model).__name__}")
            else:
                self.model = model_data
            
            print(f"最终模型类型: {type(self.model).__name__}")
            print(f"模型是否有predict方法: {hasattr(self.model, 'predict')}")
            print(f"模型是否有predict_proba方法: {hasattr(self.model, 'predict_proba')}")
            
            self.model_loaded = True
            print("模型加载成功！")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, features: list) -> Optional[Any]:
        """使用模型进行预测"""
        print("========================================")
        print("        进行预测")
        print("========================================")
        print(f"当前模型加载状态: {self.model_loaded}")
        print(f"当前模型: {self.model}")
        print(f"输入特征: {features}")
        
        if not self.model_loaded:
            print("模型未加载，尝试加载...")
            if not self.load_model():
                print("模型加载失败，无法进行预测")
                return None
        
        print(f"模型加载状态: {self.model_loaded}")
        print(f"模型类型: {type(self.model).__name__}")
        print(f"模型是否有predict方法: {hasattr(self.model, 'predict')}")
        
        try:
            # 处理缺失值
            print("处理输入特征...")
            processed_features = [0 if f is None else f for f in features]
            print(f"处理后的特征: {processed_features}")
            
            # 创建DataFrame，使用正确的列名
            feature_names = [
                '如果使用马桶，是否习惯盖马桶盖',
                '家庭厕所类型',
                '居住房屋所有权',
                '零食的食用频率',
                '家中蔬菜的购买方式'
            ]
            
            print(f"创建DataFrame，列名: {feature_names}")
            df = pd.DataFrame([processed_features], columns=feature_names)
            print(f"创建的DataFrame:\n{df}")
            
            # 确保输入形状正确
            print("使用模型进行预测...")
            if hasattr(self.model, 'predict_proba'):
                # 对于分类模型，返回概率
                print("使用predict_proba方法...")
                result = self.model.predict_proba(df)
                print(f"预测结果: {result}")
                print(f"预测结果类型: {type(result).__name__}")
                print(f"预测结果形状: {result.shape}")
                return result[0]
            else:
                # 对于回归模型，返回预测值
                print("使用predict方法...")
                result = self.model.predict(df)
                print(f"预测结果: {result}")
                print(f"预测结果类型: {type(result).__name__}")
                print(f"预测结果形状: {result.shape}")
                return result[0]
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