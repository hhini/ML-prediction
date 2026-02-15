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
    /* 引入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Noto+Sans+SC', sans-serif;
    }

    /* 主容器样式 */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* 标题样式 */
    .title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    /* 副标题样式 */
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* 卡片通用样式 */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #3b82f6;
    }
    
    /* 侧边栏样式定制 */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* 按钮美化 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
    }

    /* 风险仪表盘 */
    .risk-meter-container {
        position: relative;
        height: 24px;
        background: #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        margin: 2rem 0;
    }
    
    .risk-meter-fill {
        height: 100%;
        transition: width 1.5s cubic-bezier(0.1, 0, 0.1, 1);
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
    }

    .risk-label-container {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out forwards;
    }

    /* 响应式调整 */
    @media (max-width: 640px) {
        .title { font-size: 2rem; }
        .card { padding: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# 主页面
def main():
    # 页面标题
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 顶部 Hero 区域
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        # 这里可以放置 logo
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100px; font-size: 50px;">🦠</div>', unsafe_allow_html=True)
        # 如果有图片，可以使用 st.image("logo.png")
    with col_title:
        st.markdown('<h1 class="title">幽门螺旋杆菌风险预测</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">专业的 AI 辅助健康评估系统，关注您的胃部健康</p>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="sidebar-header">🦠 关于系统</div>', unsafe_allow_html=True)
        st.write("本系统基于个人生活习惯数据，通过机器学习算法预测**幽门螺旋杆菌**感染风险水平。")
        
        with st.expander("📌 系统说明", expanded=True):
            st.info("请如实填写右侧的特征信息，系统将为您提供实时的预测结果。")
        
        # 模型信息
        model_info = model_service.get_model_info()
        with st.expander("🤖 模型信息", expanded=False):
            st.write(f"**加载状态:** {'✅ 已就绪' if model_info['model_loaded'] else '❌ 未加载'}")
            if model_info['model_file']:
                st.write(f"**模型文件:** `{os.path.basename(model_info['model_file'])}`")
            st.write(f"**算法类型:** `{model_info['model_type']}`")
            if not model_info['model_loaded'] and model_info.get('load_error'):
                st.error(f"加载错误: {model_info['load_error']}")
        
        # 幽门螺旋杆菌知识
        with st.expander("📚 医学知识", expanded=True):
            st.markdown("""
            **幽门螺旋杆菌 (Hp)** 是一种常见的胃肠道细菌，与以下疾病密切相关：
            - 慢性胃炎
            - 胃溃疡 / 十二指肠溃疡
            - 胃癌风险增加
            
            **传播途径：**
            主要通过“口-口”或“粪-口”途径传播。
            """)
    
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
                        st.markdown('<div class="card animate-fade-in">', unsafe_allow_html=True)
                        st.markdown('<h2 style="text-align: center; color: #1e293b; margin-bottom: 1.5rem;">🎯 风险评估报告</h2>', unsafe_allow_html=True)
                        
                        # 获取阳性概率
                        if isinstance(prediction, list) or (hasattr(prediction, '__len__') and len(prediction) > 1):
                            risk_score = float(prediction[1])
                        else:
                            risk_score = float(prediction)
                        
                        # 风险等级判定
                        if risk_score < 0.3:
                            risk_level = "低风险"
                            risk_color = "#22c55e"
                            risk_desc = "您的生活习惯良好，感染风险较低。请继续保持！"
                        elif risk_score < 0.7:
                            risk_level = "中等风险"
                            risk_color = "#eab308"
                            risk_desc = "存在一定的感染风险。建议改善卫生习惯，并关注胃部状况。"
                        else:
                            risk_level = "高风险"
                            risk_color = "#ef4444"
                            risk_desc = "感染风险较高！建议及时去医院进行 C13/C14 呼气试验筛查。"

                        # 可视化仪表盘
                        st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.2rem; color: #64748b;">风险概率: </span>
                            <span style="font-size: 2.5rem; font-weight: 800; color: {risk_color};">{risk_score*100:.1f}%</span>
                        </div>
                        <div class="risk-meter-container">
                            <div class="risk-meter-fill" style="width: {risk_score*100}%; background: {risk_color};"></div>
                        </div>
                        <div class="risk-label-container">
                            <span>低风险</span>
                            <span>中等风险</span>
                            <span>高风险</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 风险结论卡片
                        st.markdown(f"""
                        <div style="background: {risk_color}15; border-radius: 16px; padding: 1.5rem; border: 1px solid {risk_color}30; margin-top: 1rem;">
                            <h3 style="color: {risk_color}; margin-top: 0;">评估结果：{risk_level}</h3>
                            <p style="color: #334155; margin-bottom: 0;">{risk_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # 预防建议
                        st.markdown('<div class="prevention-tips">', unsafe_allow_html=True)
                        st.markdown('<h4>💡 专家预防建议</h4>', unsafe_allow_html=True)
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("✅ **个人卫生**")
                            st.write("- 饭前便后勤洗手")
                            st.write("- 建议使用公筷公勺")
                            st.write("- 定期更换牙刷")
                        with col_b:
                            st.write("🥗 **饮食习惯**")
                            st.write("- 减少生食摄入")
                            st.write("- 蔬菜水果洗净削皮")
                            st.write("- 避免共用餐具")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<p class="info-text" style="text-align: center;">⚠️ 注: 本评估基于统计模型，结果仅供参考。如有不适请务必咨询专业医师。</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("预测失败，请检查模型是否正确加载")
                        model_info = model_service.get_model_info()
                        if model_info.get('load_error'):
                            st.error(f"详细错误: {model_info['load_error']}")
    
    # 页脚
    st.markdown('<footer style="text-align: center; margin-top: 3rem; color: #7f8c8d;">', unsafe_allow_html=True)
    st.markdown('<p>© 2026 幽门螺旋杆菌风险预测系统 | 基于机器学习技术</p>', unsafe_allow_html=True)
    st.markdown('<p>仅供参考，不构成医疗建议</p>', unsafe_allow_html=True)
    st.markdown('</footer>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()