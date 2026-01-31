#!/usr/bin/env python3
"""
启动脚本
用于启动Streamlit健康风险预测应用
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import streamlit
        import pandas
        import numpy
        print("依赖检查通过")
        return True
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_model_file():
    """检查模型文件是否存在"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        print("请创建models目录并放入模型文件")
        return False
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.pickle', '.joblib', '.model'))]
    
    if not model_files:
        print(f"models目录中未找到模型文件")
        print("请放入模型文件，支持的格式: .pkl, .pickle, .joblib, .model")
        return False
    
    print(f"找到模型文件: {model_files}")
    return True

def start_streamlit():
    """启动Streamlit应用"""
    ui_file = os.path.join(os.path.dirname(__file__), 'app', 'ui', 'main.py')
    
    if not os.path.exists(ui_file):
        print(f"UI文件不存在: {ui_file}")
        return False
    
    print("启动Streamlit应用...")
    print(f"UI文件路径: {ui_file}")
    
    # 构建启动命令
    command = [
        sys.executable,
        '-m', 'streamlit',
        'run', ui_file,
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ]
    
    try:
        # 启动Streamlit应用
        print(f"执行命令: {' '.join(command)}")
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        return False
    except KeyboardInterrupt:
        print("应用已停止")
        return True

def main():
    """主函数"""
    print("========================================")
    print("        健康风险预测系统启动")
    print("========================================")
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查模型文件
    if not check_model_file():
        return 1
    
    # 启动应用
    if not start_streamlit():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())