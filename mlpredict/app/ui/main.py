import streamlit as st
import os
import sys

# 尝试导入必要的模块
try:
    # 添加项目根目录到Python路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    
    from mlpredict.app.services.feature_processor import FeatureProcessor
    from mlpredict.app.services.model_service import ModelService
    modules_loaded = True
except ImportError as e:
    modules_loaded = False
    error_message = str(e)
    print(f"Error loading modules: {error_message}")

# 设置页面配置
st.set_page_config(
    page_title="幽门螺旋杆菌风险预测",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 主容器样式 */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* 标题样式 */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d5a8c;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* 副标题样式 */
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* 卡片样式 */
    .card {
        background-color: white;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    }
    
    /* 表单样式 */
    .form-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(52, 152, 219, 0.4);
    }
    
    /* 结果卡片样式 */
    .result-card {
        background-color: #f8f9fa;
        border-radius: 16px;
        padding: 2.5rem;
        margin-top: 2rem;
        border-left: 6px solid #3498db;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* 结果标题样式 */
    .result-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2d5a8c;
        margin-bottom: 1.5rem;
    }
    
    /* 结果值样式 */
    .result-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3498db;
        margin-bottom: 1.5rem;
    }
    
    /* 结果值样式 - 阳性 */
    .result-value-positive {
        color: #e74c3c;
    }
    
    /* 结果值样式 - 阴性 */
    .result-value-negative {
        color: #27ae60;
    }
    
    /* 说明文本样式 */
    .info-text {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    /* 侧边栏样式 */
    .sidebar {
        background-color: #f8f9fa;
        border-radius: 16px;
        padding: 2rem;
        margin-right: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* 侧边栏标题 */
    .sidebar h2 {
        color: #2d5a8c;
        margin-bottom: 1.5rem;
    }
    
    /* 侧边栏内容 */
    .sidebar-content {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #495057;
    }
    
    /* 预防建议样式 */
    .prevention-tips {
        background-color: #e3f2fd;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 4px solid #2196f3;
    }
    
    .prevention-tips h4 {
        color: #1976d2;
        margin-bottom: 1rem;
    }
    
    .prevention-tips ul {
        margin-left: 1.5rem;
        color: #37474f;
    }
    
    /* 加载动画样式 */
    .loader {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(52, 152, 219, 0.3);
        border-radius: 50%;
        border-top-color: #3498db;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* 特征标签样式 */
    .feature-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    /* 选择框样式 */
    .stSelectbox > div {
        border-radius: 8px;
    }
    
    /* 响应式调整 */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .card {
            padding: 1.5rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 主页面
def main():
    # 页面标题
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">幽门螺旋杆菌风险预测</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">基于个人生活习惯的幽门螺旋杆菌感染风险评估</p>', unsafe_allow_html=True)
    
    # 检查模块是否加载成功
    if not modules_loaded:
        st.error(f"应用启动失败，缺少必要的依赖模块：{error_message}")
        st.info("请确保已安装所有必要的依赖：pip install -r requirements.txt")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # 初始化服务
    print("========================================")
    print("        初始化服务")
    print("========================================")
    
    # 获取绝对路径
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    models_dir = os.path.join(project_root, 'mlpredict', 'models')
    
    print(f"当前文件目录: {current_dir}")
    print(f"项目根目录: {project_root}")
    print(f"模型目录: {models_dir}")
    print(f"模型目录是否存在: {os.path.exists(models_dir)}")
    
    if os.path.exists(models_dir):
        print(f"模型目录中的文件: {os.listdir(models_dir)}")
    else:
        print("警告: 模型目录不存在！")
    
    feature_processor = FeatureProcessor()
    model_service = ModelService(model_dir=models_dir)
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.header("关于系统")
        st.write("本系统基于个人生活习惯数据，预测幽门螺旋杆菌感染风险水平。")
        st.write("请填写以下特征信息，系统将为您提供预测结果。")
        
        # 模型信息
        model_info = model_service.get_model_info()
        st.subheader("模型信息")
        st.write(f"模型加载状态: {'已加载' if model_info['model_loaded'] else '未加载'}")
        if model_info['model_file']:
            st.write(f"模型文件: {os.path.basename(model_info['model_file'])}")
        st.write(f"模型类型: {model_info['model_type']}")
        
        # 幽门螺旋杆菌知识
        st.subheader("幽门螺旋杆菌知识")
        st.write("幽门螺旋杆菌是一种常见的胃肠道细菌，可引起胃炎、胃溃疡等疾病。")
        st.write("主要通过口-口、粪-口途径传播，与卫生习惯密切相关。")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 主内容区
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("特征信息填写")
        
        # 表单布局
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. 如果使用马桶，是否习惯盖马桶盖
            toilet_lid = st.selectbox(
                "如果使用马桶，是否习惯盖马桶盖",
                options=['请选择', '是', '否', '未填'],
                index=0
            )
            
            # 2. 家庭厕所类型
            toilet_type = st.selectbox(
                "家庭厕所类型",
                options=['请选择', '传统旱厕', '冲洗坑厕', '抽水马桶'],
                index=0
            )
            
            # 3. 居住房屋所有权
            house_ownership = st.selectbox(
                "居住房屋所有权",
                options=['请选择', '自己购买新房', '自己购买二手房', '自建房', '租房', '否'],
                index=0
            )
        
        with col2:
            # 4. 零食的食用频率
            snack_frequency = st.selectbox(
                "零食的食用频率",
                options=['请选择', '否', '1-2次/年', '1-2次/月', '1-2次/周', '3-5次/周', '＞5次/周', '未填'],
                index=0
            )
            
            # 5. 家中蔬菜的购买方式
            vegetable_purchase = st.selectbox(
                "家中蔬菜的购买方式",
                options=['请选择', '自家种植', '超市', '菜市场', '街头小贩', '都有'],
                index=0
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测按钮
    with st.container():
        if st.button("开始预测"):
            # 验证输入
            if any([toilet_lid == '请选择', toilet_type == '请选择', 
                    house_ownership == '请选择', snack_frequency == '请选择', 
                    vegetable_purchase == '请选择']):
                st.error("请填写所有特征信息")
            else:
                # 显示加载动画
                with st.spinner("正在分析..."):
                    # 准备特征数据
                    features = {
                        'toilet_lid': toilet_lid if toilet_lid != '请选择' else '',
                        'toilet_type': toilet_type if toilet_type != '请选择' else '',
                        'house_ownership': house_ownership if house_ownership != '请选择' else '',
                        'snack_frequency': snack_frequency if snack_frequency != '请选择' else '',
                        'vegetable_purchase': vegetable_purchase if vegetable_purchase != '请选择' else ''
                    }
                    
                    # 处理特征
                    processed_features = feature_processor.process_all_features(features)
                    
                    # 进行预测
                    prediction = model_service.predict(processed_features)
                    
                    # 展示结果
                    if prediction is not None:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.header("预测结果")
                        
                        # 根据预测结果类型展示
                        if isinstance(prediction, list) or (hasattr(prediction, '__len__') and len(prediction) > 1):
                            # 分类模型结果
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown('<h3 class="result-title">感染概率</h3>', unsafe_allow_html=True)
                            
                            # 假设第一个概率是阴性，第二个是阳性
                            if len(prediction) >= 2:
                                negative_prob = float(prediction[0])
                                positive_prob = float(prediction[1])
                                
                                # 显示阴性概率
                                st.markdown(f'<p class="result-value result-value-negative">阴性概率: {negative_prob:.4f}</p>', unsafe_allow_html=True)
                                # 显示阳性概率
                                st.markdown(f'<p class="result-value result-value-positive">阳性概率: {positive_prob:.4f}</p>', unsafe_allow_html=True)
                                
                                # 显示最终结果
                                if positive_prob > negative_prob:
                                    st.markdown('<p class="result-value result-value-positive">最终结果: 阳性</p>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<p class="result-value result-value-negative">最终结果: 阴性</p>', unsafe_allow_html=True)
                            else:
                                # 显示所有概率
                                for i, prob in enumerate(prediction):
                                    st.markdown(f'<p class="result-value">类别 {i}: {float(prob):.4f}</p>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            # 回归模型结果
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown('<h3 class="result-title">风险评分</h3>', unsafe_allow_html=True)
                            # 处理数组或标量
                            if hasattr(prediction, '__len__') and len(prediction) == 1:
                                prediction_value = float(prediction[0])
                            else:
                                prediction_value = float(prediction)
                            
                            # 根据评分显示结果
                            if prediction_value > 0.5:
                                st.markdown(f'<p class="result-value result-value-positive">{prediction_value:.4f} (阳性)</p>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<p class="result-value result-value-negative">{prediction_value:.4f} (阴性)</p>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 预防建议
                        st.markdown('<div class="prevention-tips">', unsafe_allow_html=True)
                        st.subheader("预防建议")
                        st.write("1. 保持良好的个人卫生习惯，勤洗手")
                        st.write("2. 使用马桶时，养成盖马桶盖的习惯")
                        st.write("3. 确保家庭厕所卫生，定期清洁")
                        st.write("4. 注意饮食卫生，蔬菜洗净后食用")
                        st.write("5. 减少零食摄入，保持健康饮食")
                        st.write("6. 分餐制，避免交叉感染")
                        st.write("7. 定期体检，早发现早治疗")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 结果说明
                        st.markdown('<p class="info-text">注: 预测结果仅供参考，不构成医疗建议。如有健康问题，请咨询专业医生。</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("预测失败，请检查模型是否正确加载")
    
    # 页脚
    st.markdown('<footer style="text-align: center; margin-top: 3rem; color: #7f8c8d;">', unsafe_allow_html=True)
    st.markdown('<p>© 2026 幽门螺旋杆菌风险预测系统 | 基于机器学习技术</p>', unsafe_allow_html=True)
    st.markdown('<p>仅供参考，不构成医疗建议</p>', unsafe_allow_html=True)
    st.markdown('</footer>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()