#!/usr/bin/env python3
"""
测试模型加载和预测功能
"""

import os
import sys
import joblib
import pickle
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_model_loading():
    """测试模型加载"""
    print("========================================")
    print("        模型加载测试")
    print("========================================")
    
    # 测试不同的模型路径
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'mlpredict', 'models'),
        os.path.join(os.path.dirname(__file__), 'models'),
        'models'
    ]
    
    for model_dir in model_paths:
        print(f"\n尝试模型路径: {model_dir}")
        print(f"路径是否存在: {os.path.exists(model_dir)}")
        
        if os.path.exists(model_dir):
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
            
            if model_file:
                print(f"找到模型文件: {model_file}")
                print(f"文件是否存在: {os.path.exists(model_file)}")
                
                # 尝试加载模型
                try:
                    if model_file.endswith('.joblib'):
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
                        model = None
                        for i, item in enumerate(model_data):
                            print(f"元组元素 {i} 类型: {type(item).__name__}")
                            print(f"是否有predict方法: {hasattr(item, 'predict')}")
                            if hasattr(item, 'predict'):
                                model = item
                                print(f"从元组中找到模型对象: {type(model).__name__}")
                                break
                        # 如果没有找到，使用第一个元素
                        if not model:
                            model = model_data[0]
                            print(f"使用元组的第一个元素作为模型: {type(model).__name__}")
                    else:
                        model = model_data
                    
                    print(f"最终模型类型: {type(model).__name__}")
                    print(f"模型是否有predict方法: {hasattr(model, 'predict')}")
                    print(f"模型是否有predict_proba方法: {hasattr(model, 'predict_proba')}")
                    
                    # 测试预测
                    test_features = [1, 2, 4, 3, 1]
                    print(f"\n测试输入特征: {test_features}")
                    
                    # 创建DataFrame，使用正确的列名
                    feature_names = [
                        '如果使用马桶，是否习惯盖马桶盖',
                        '家庭厕所类型',
                        '居住房屋所有权',
                        '零食的食用频率',
                        '家中蔬菜的购买方式'
                    ]
                    
                    df = pd.DataFrame([test_features], columns=feature_names)
                    print(f"创建的DataFrame:\n{df}")
                    
                    # 尝试预测
                    try:
                        print("尝试使用predict方法...")
                        prediction = model.predict(df)
                        print(f"predict方法测试成功: {prediction}")
                    except Exception as e:
                        print(f"predict方法测试失败: {e}")
                    
                    try:
                        print("尝试使用predict_proba方法...")
                        proba = model.predict_proba(df)
                        print(f"predict_proba方法测试成功: {proba}")
                    except Exception as e:
                        print(f"predict_proba方法测试失败: {e}")
                    
                    return True
                except Exception as e:
                    print(f"加载模型失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("未找到模型文件")
        else:
            print("路径不存在")
    
    return False

def main():
    """主函数"""
    test_model_loading()
    print("\n========================================")
    print("        测试完成")
    print("========================================")

if __name__ == "__main__":
    main()