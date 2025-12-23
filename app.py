import gradio as gr
import pandas as pd
import numpy as np
import joblib

# 1. 加载训练好的模型和特征信息
print("正在加载模型和特征信息...")
try:
    model = joblib.load('house_price_model.pkl')
    feature_info = joblib.load('feature_info.pkl')
    features = feature_info['features']
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']
    print("✅ 模型和特征信息加载成功！")
except FileNotFoundError as e:
    print(f"❌ 加载失败: {e}")
    print("请确保 ‘house_price_model.pkl‘ 和 ‘feature_info.pkl‘ 文件存在于当前目录。")
    exit()

# 2. 为分类特征准备选项（从训练数据中获取，这里提供示例，你可能需要调整）
# 注意：为了应用能运行，这里为每个分类特征硬编码了常见选项。
# 更严谨的做法是从原始训练数据文件中读取所有唯一值。
categorical_options = {
    'Neighborhood_Grouped': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Other'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'SaleCondition': ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']
}

# 3. 核心预测函数
def predict_price(*input_values):
    """
    根据输入预测房价
    """
    # 将输入值转换为字典，键为特征名
    input_dict = dict(zip(features, input_values))
    input_df = pd.DataFrame([input_dict])

    # 模型预测（预测结果是对数价格）
    prediction_log = model.predict(input_df)[0]
    # 将对数价格转换回实际价格（美元）
    prediction = np.expm1(prediction_log)

    # 格式化输出
    return f"**预测房价约为: ${prediction:,.2f}**\n\n(基于您提供的 {len(features)} 个房屋特征)"


# 5. 使用Blocks API创建更灵活的界面
print("正在启动Web应用界面...")

with gr.Blocks(title="🏠 房屋价格预测器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏠 房屋价格预测器
    请输入房屋的特征信息，模型将预测其市场售价。
    **注意**：此模型基于Ames Housing数据集训练，预测结果仅供参考。
    """)

    with gr.Row():  # 创建一个行，用于放置多列输入
        # === 第一列：核心结构与面积 ===
        with gr.Column():
            gr.Markdown("### 🏗️ 核心结构与面积")
            input_components_dict = {}  # 用字典存储组件，方便后续引用
            # 我们手动将特征分组，并为每个输入框创建变量
            # 请根据你的特征列表，将以下‘默认值’调整得更具代表性
            with gr.Group():
                input_components_dict['TotalSF'] = gr.Number(value=2500, label="总面积 (平方英尺)", info="TotalSF")            
        
                input_components_dict['GrLivArea'] = gr.Number(value=1700, label="地上居住面积", info="GrLivArea")
                input_components_dict['TotalPorchSF'] = gr.Number(value=500, label="门廊总面积")
                input_components_dict['OverallQual'] = gr.Slider(1, 10, step=1, value=7, label="整体质量 (1-10分)")
                input_components_dict['YearBuilt'] = gr.Slider(1900, 2020, step=1, value=1995, label="建造年份")
                input_components_dict['HouseAge'] = gr.Number(value=30, label="房屋年龄 (年)", interactive=False) # 可设为只读，由计算得出
                input_components_dict['RemodAge'] = gr.Number(value=25, label="重装修年龄 (年)")

        # === 第二列：房间与设施 ===
        with gr.Column():
            gr.Markdown("### 🛏️ 房间与设施")
            with gr.Group():
                input_components_dict['GarageCars'] = gr.Slider(0, 4, step=1, value=2, label="车库可容纳车辆数")
                input_components_dict['TotalBath'] = gr.Number(value=3.0, label="浴室总数")
                input_components_dict['TotalKitchen'] = gr.Number(value=1, label="厨房总数")
                input_components_dict['TotRmsAbvGrd'] = gr.Number(value=8, label="地上总房间数")
                input_components_dict['OverallGrade'] = gr.Number(value=65, label="综合质量分", info="(质量×条件)")
                input_components_dict['LivAreaRatio'] = gr.Number(value=0.35, label="居住面积占地比")
                input_components_dict['SpaceEfficiency'] = gr.Number(value=0.6, label="空间效率分数")          
    
                # 注意：你需要将这个特征加入你的特征列表并重新训练模型，或者删除此行

        # === 第三列：分类与其他特征 ===
        with gr.Column():
            gr.Markdown("### 📍 分类与其他特征")
            with gr.Group():
                # 分类特征的下拉菜单
                input_components_dict['Neighborhood_Grouped'] = gr.Dropdown(
                    choices=['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Other'],
                    value='CollgCr',
                    label="地段分组"
                )
                input_components_dict['KitchenQual'] = gr.Dropdown(
                    choices=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                    value='Gd',
                    label="厨房质量"
                )
                input_components_dict['SaleCondition'] = gr.Dropdown(
                    choices=['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'],
                    value='Normal',
                    label="销售条件"
                )
                
    # === 预测按钮和结果显示区域 ===
    with gr.Row():
        predict_btn = gr.Button("预测房价", variant="primary", size="lg")
    with gr.Row():
        output = gr.Markdown("## 预测结果将显示在这里。")

    # === 绑定点击事件 ===
    # 注意：根据上面 input_components_dict 的键顺序，确保与函数参数顺序一致
    predict_btn.click(
        fn=predict_price,
        # 这里按顺序列出所有输入组件的值
        inputs=[input_components_dict[feat] for feat in features],
        outputs=output
    )

    # === 添加示例 ===
    gr.Markdown("### 💡 快速尝试")
    gr.Examples(
        examples=[
            [2500, 1700, 7, 1995, 30, 3.0, 2, 8, 2, 1, 'CollgCr', 'Gd', 'Normal', 500, 65, 0.35],  # 示例1
            [1200, 1100, 5, 1970, 50, 1.5, 1, 5, 1, 0, 'Other', 'TA', 'Normal', 200, 25, 0.2]     # 示例2
        ],
        # 对应的输入组件列表，必须和上面inputs的顺序完全一致
        inputs=[input_components_dict[feat] for feat in features],
        outputs=output,
        fn=predict_price,
        cache_examples=True
    )

# 6. 启动应用
if __name__ == "__main__":
    demo.launch(share=False)