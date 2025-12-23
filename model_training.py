# model_training.py
"""
房屋价格预测 - 模型训练脚本
说明：此脚本执行完整的机器学习流程，包括数据加载、预处理、特征工程、模型训练、评估和保存。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 加载与准备数据 ====================
print("步骤1: 加载数据...")
# 加载你之前处理并保存的数据
train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')
print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")

# ==================== 2. 定义目标变量和特征 ====================
print("\n步骤2: 准备目标变量和特征...")

# 2.1 目标变量：对房价取对数（这是处理偏态分布、提升模型性能的标准做法）
y = np.log1p(train_df['SalePrice'])  # 使用 log1p 防止对0取对数
print(f"目标变量 ‘SalePrice' 已进行对数转换。")

# 2.2 特征选择：这是最关键的一步！你需要根据数据探索的结果确定最终的特征列表。
# 请确保选择的特征在训练集和测试集中都存在，且经过适当处理（如缺失值填充、编码等）。
# 对于分类特征，选择有意义的、且不是高基数的（类别不是特别多的）。

numerical_features = [
    'TotalSF',           # 我们新建的总面积
    'TotalPorchSF',      # 我们新建的门廊面积
    'OverallQual',       # 整体质量（通常与房价最相关）
    'GrLivArea',         # 地上居住面积

    'HouseAge',          # 房龄
    'YearBuilt',         # 建造年份
    'RemodAge',          # 重装修年龄

    'GarageCars',        # 车库容量
    'TotalBath',         # 我们新建的浴室总数
    'TotalKitchen',      # 我们新建的厨房总数

    'TotRmsAbvGrd',      # 地上总房间数
    'OverallGrade',      # 我们新建的质量×条件
    'LivAreaRatio',      # 新建的居住面积比例
    'SpaceEfficiency'    # 新建的空间效率分数
]

categorical_features = [
    'Neighborhood_Grouped',  # 我们分组后的地段
    'KitchenQual',      # 厨房质量
    'SaleCondition',    # 销售条件

]

# 合并特征列表
features = numerical_features + categorical_features
print(f"最终用于建模的特征数量: {len(features)}")
print("数值特征:", numerical_features)
print("分类特征:", categorical_features)

# 创建特征数据集
X = train_df[features]
X_test = test_df[features]  # 用于最终Kaggle提交的测试集

# ==================== 3. 划分训练集和验证集 ====================
print("\n步骤3: 划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练样本数: {X_train.shape[0]}, 验证样本数: {X_val.shape[0]}")

# ==================== 4. 构建预处理管道 ====================
print("\n步骤4: 构建数据预处理管道...")
# 分别对数值列和分类列进行处理，然后合并
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),          # 数值特征：标准化
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # 分类特征：独热编码
    ])

# ==================== 5. 创建并训练XGBoost模型管道 ====================
print("\n步骤5: 创建机器学习模型管道...")
# 将预处理器和模型串联成一个管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',  # 回归问题
        random_state=42,
        n_jobs=-1,                     # 使用所有CPU核心
        eval_metric='rmse'
    ))
])

# 5.1 基础训练
print("开始基础模型训练...")
model.fit(X_train, y_train)

# 5.2 在验证集上进行初步评估
print("\n步骤6: 模型初步评估...")
y_val_pred_log = model.predict(X_val)  # 预测的是对数房价

# 将预测值和对数真实值转换回原始房价尺度（美元）
y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val)

# 计算回归任务的关键指标
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_r2 = r2_score(y_val_true, y_val_pred)

print(f"验证集 RMSE (预测误差): ${val_rmse:,.2f}")
print(f"验证集 MAE (平均绝对误差): ${val_mae:,.2f}")
print(f"验证集 R² Score (解释方差): {val_r2:.4f}")
# R²越接近1越好，RMSE/MAE越小越好

# ==================== 6. (可选但推荐) 超参数调优 ====================
print("\n步骤7: 开始超参数调优 (Grid Search)...")
# 定义要搜索的参数网格
param_grid = {
    'regressor__n_estimators': [100, 200],      # 树的数量
    'regressor__max_depth': [3, 5, 7],          # 树的最大深度
    'regressor__learning_rate': [0.01, 0.05, 0.1], # 学习率
    'regressor__subsample': [0.8, 1.0],         # 样本采样比例
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,                           # 5折交叉验证
    scoring='neg_root_mean_squared_error', # 用负的RMSE评分，因为sklearn遵循“越大越好”
    n_jobs=-1,                      # 并行运行
    verbose=1                       # 打印详细过程
)

print("正在进行网格搜索，这可能需要几分钟...")
grid_search.fit(X_train, y_train)

print(f"\n调优完成！")
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证分数 (负RMSE): {grid_search.best_score_:.4f}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# ==================== 7. 用最佳模型重新评估 ====================
print("\n步骤8: 使用最佳模型进行最终评估...")
y_val_best_pred_log = best_model.predict(X_val)
y_val_best_pred = np.expm1(y_val_best_pred_log)

best_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_best_pred))
best_r2 = r2_score(y_val_true, y_val_best_pred)
print(f"调优后验证集 RMSE: ${best_rmse:,.2f}")
print(f"调优后验证集 R² Score: {best_r2:.4f}")

# ==================== 8. 在整个训练集上训练最终模型并进行预测 ====================
print("\n步骤9: 在整个训练集上训练最终模型...")
# 为了充分利用数据，我们用最佳参数在整个训练集上重新训练一个最终模型
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        **{k.replace('regressor__', ''): v for k, v in grid_search.best_params_.items()},
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ))
])
final_model.fit(X, y)  # 使用全部训练数据
print("最终模型训练完成。")

# 对比赛测试集进行预测
print("对Kaggle测试集进行预测...")
test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)  # 转换回实际房价

# ==================== 9. 保存预测结果和模型 ====================
print("\n步骤10: 保存结果和模型...")
# 9.1 保存Kaggle提交文件
if 'Id' in test_df.columns:
    submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred})
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存至 ‘submission.csv'，可用于Kaggle提交。")

# 9.2 保存训练好的模型，供Web应用使用
joblib.dump(final_model, 'house_price_model.pkl')
print("模型已保存至 ‘house_price_model.pkl'。")

# 9.3 (可选) 保存特征列表，确保Web应用使用相同的特征
feature_info = {
    'features': features,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features
}
joblib.dump(feature_info, 'feature_info.pkl')
print("特征信息已保存至 ‘feature_info.pkl'。")

print("\n" + "="*50)
print("模型训练流程全部完成！")
print("="*50)