#!/usr/bin/env python3
"""
测试模型文件加载和预测功能
"""

import os
import sys
import joblib
import pickle

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_model_loading():
    """测试模型文件加载"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print(f"模型目录: {model_dir}")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return False
    
    # 列出模型目录中的文件
    model_files = os.listdir(model_dir)
    print(f"模型目录中的文件: {model_files}")
    
    # 找到模型文件
    model_file = None
    model_extensions = ['.pkl', '.pickle', '.joblib', '.model']
    
    for file in model_files:
        if any(file.endswith(ext) for ext in model_extensions):
            model_file = os.path.join(model_dir, file)
            break
    
    if not model_file:
        print("未找到模型文件")
        return False
    
    print(f"找到模型文件: {model_file}")
    
    # 尝试加载模型
    try:
        if model_file.endswith('.joblib'):
            model = joblib.load(model_file)
            print("使用joblib加载模型成功")
        else:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print("使用pickle加载模型成功")
        
        print(f"模型类型: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def test_model_prediction(model):
    """测试模型预测功能"""
    if not model:
        print("模型未加载，无法测试预测")
        return False
    
    # 测试输入特征
    test_features = [1, 2, 4, 3, 1]  # 示例特征值
    print(f"测试输入特征: {test_features}")
    
    try:
        # 测试predict方法
        if hasattr(model, 'predict'):
            print("尝试使用predict方法...")
            prediction = model.predict([test_features])
            print(f"predict方法测试成功: {prediction}")
        else:
            print("模型没有predict方法")
        
        # 测试predict_proba方法
        if hasattr(model, 'predict_proba'):
            print("尝试使用predict_proba方法...")
            proba = model.predict_proba([test_features])
            print(f"predict_proba方法测试成功: {proba}")
        else:
            print("模型没有predict_proba方法")
        
        return True
    except Exception as e:
        print(f"预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("========================================")
    print("        模型测试")
    print("========================================")
    
    # 测试模型加载
    model = test_model_loading()
    
    if model:
        # 测试模型预测
        test_model_prediction(model)
    
    print("========================================")
    print("        测试完成")
    print("========================================")

if __name__ == "__main__":
    main()