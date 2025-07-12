# -*- coding: utf-8 -*-
"""
投篮机器人转速预测系统（篮筐原点优化版）
功能：以篮筐为原点，集成自动化调参、数据增强和异常值处理
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import random

# ==================== 1. 初始化设置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
np.random.seed(42)  # 固定随机种子

# ==================== 2. 数据准备 ====================
# 模拟数据生成函数
def generate_data(n_samples=200):
    """生成模拟数据（实际使用时替换为真实数据）"""
    # 场地尺寸：x[0.5, 3.5]m, y[0.5, 2.5]m
    X = np.random.uniform(low=[0.5, 0.5], high=[3.5, 2.5], size=(n_samples, 2))
    
    # 篮筐位置（场地右上角）
    HOOP_POS = np.array([3.0, 2.0])
    
    # 计算到篮筐的距离和角度
    dist = np.linalg.norm(X - HOOP_POS, axis=1)
    angles = np.arctan2(HOOP_POS[1]-X[:,1], HOOP_POS[0]-X[:,0])
    
    # 转速计算公式：基于距离和角度
    base_rpm = 1200 + 100 * dist**1.5
    angle_factor = np.cos(angles)**2  # 角度影响因子
    Y = base_rpm * angle_factor + np.random.normal(0, 50, n_samples)
    
    # 添加10%的异常值
    n_outliers = int(n_samples * 0.1)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    Y[outlier_indices] += np.random.uniform(300, 600, n_outliers)
    
    return X, Y, HOOP_POS

# 生成数据
X, Y, HOOP_POS = generate_data(n_samples=200)
print(f"篮筐位置: ({HOOP_POS[0]:.1f}, {HOOP_POS[1]:.1f}) m")

# ==================== 3. 异常值处理 ====================
def detect_outliers(X, Y, z_threshold=3):
    """使用Z-Score检测并处理异常值"""
    # 计算Z分数
    z_scores = np.abs(stats.zscore(Y))
    
    # 识别异常值
    outlier_mask = z_scores > z_threshold
    
    # 创建清洁数据集
    X_clean = X[~outlier_mask]
    Y_clean = Y[~outlier_mask]
    
    print(f"\n异常值处理:")
    print(f"  原始样本数: {len(X)}")
    print(f"  检测到异常值: {np.sum(outlier_mask)}个")
    print(f"  清洁数据集: {len(X_clean)}个样本")
    
    return X_clean, Y_clean

# 检测并移除异常值
X_clean, Y_clean = detect_outliers(X, Y)

# ==================== 4. 坐标转换（以篮筐为原点） ====================
def transform_to_hoop_coordinates(X, hoop_pos):
    """将坐标转换为以篮筐为原点"""
    return X - hoop_pos

# 坐标转换
X_hoop = transform_to_hoop_coordinates(X_clean, HOOP_POS)

# 可视化转换后的坐标
plt.figure(figsize=(8, 6))
plt.scatter(X_hoop[:, 0], X_hoop[:, 1], c=Y_clean, cmap='viridis')
plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # x轴
plt.axvline(0, color='r', linestyle='--', alpha=0.5)  # y轴
plt.colorbar(label='电机转速 (RPM)')
plt.xlabel('X方向 (篮筐前方为正)')
plt.ylabel('Y方向 (篮筐右侧为正)')
plt.title('以篮筐为原点的坐标分布')
plt.grid(True)
plt.show()

# ==================== 5. 数据增强 ====================
def augment_data(X, Y, noise_std=0.05, symmetry=True):
    """为数据集添加噪声和对称样本"""
    X_aug, Y_aug = [X], [Y]
    
    # 1. 添加高斯噪声
    for _ in range(2):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        Y_aug.append(Y)
    
    # 2. 对称增强（关于Y轴对称）
    if symmetry:
        X_flipped = X.copy()
        X_flipped[:, 0] = -X_flipped[:, 0]  # X方向镜像
        X_aug.append(X_flipped)
        Y_aug.append(Y)  # 假设对称位置转速相同
    
    return np.vstack(X_aug), np.concatenate(Y_aug)

# 应用数据增强
X_aug, Y_aug = augment_data(X_hoop, Y_clean)
print(f"\n数据增强后样本数: {len(X_aug)} (原始清洁样本: {len(X_hoop)})")

# ==================== 6. 数据预处理 ====================
# 对称归一化（保留负值）
def symmetric_normalization(X):
    """将数据归一化到[-1, 1]范围"""
    # 计算每个维度的最大绝对值
    max_abs = np.max(np.abs(X), axis=0)
    # 避免除零错误
    max_abs[max_abs == 0] = 1
    return X / max_abs

# 归一化坐标
X_norm = symmetric_normalization(X_aug)

# 转速归一化
Y_min, Y_max = np.min(Y_aug), np.max(Y_aug)
Y_norm = (Y_aug - Y_min) / (Y_max - Y_min)

print("\n归一化参数：")
print(f"X归一化范围: [-1, 1]")
print(f"Y_min: {Y_min:.1f}, Y_max: {Y_max:.1f}")

# ==================== 7. 特征工程 ====================
def add_polar_features(X):
    """添加极坐标特征"""
    r = np.linalg.norm(X, axis=1)  # 到篮筐的距离
    theta = np.arctan2(X[:, 1], X[:, 0])  # 角度
    return np.column_stack([X, r, theta, r**2, np.sin(theta), np.cos(theta)])

# 添加极坐标特征
X_extended = add_polar_features(X_norm)
print(f"特征维度: {X_extended.shape[1]} (原始: {X_norm.shape[1]})")

# ==================== 8. 自动化模型调参 ====================
def auto_tune_model(X_train, y_train):
    """自动化模型调参"""
    # 创建管道：多项式特征 + 岭回归
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('ridge', Ridge())
    ])
    
    # 参数网格
    param_grid = {
        'poly__degree': [2, 3],  # 多项式阶数
        'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]  # 正则化强度
    }
    
    # 网格搜索
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='neg_mean_absolute_error',
        verbose=1, n_jobs=-1
    )
    
    # 执行搜索
    grid_search.fit(X_train, y_train)
    
    print("\n自动化调参结果:")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳MAE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 划分训练测试集 (90%训练, 10%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_extended, Y_norm, test_size=0.1, random_state=42
)

# 自动调参
best_model = auto_tune_model(X_train, y_train)

# ==================== 9. 模型评估 ====================
def evaluate_model(model, X, y_true, Y_range, dataset_name):
    """评估模型性能"""
    y_pred_norm = model.predict(X)
    y_pred = y_pred_norm * (Y_range[1] - Y_range[0]) + Y_range[0]
    
    # 反归一化真实值
    y_true_denorm = y_true * (Y_range[1] - Y_range[0]) + Y_range[0]
    
    mae = mean_absolute_error(y_true_denorm, y_pred)
    max_error = np.max(np.abs(y_true_denorm - y_pred))
    r2 = r2_score(y_true_denorm, y_pred)
    
    print(f"\n{dataset_name}集性能：")
    print(f"  样本数: {len(y_true)}")
    print(f"  平均绝对误差(MAE): {mae:.1f} RPM")
    print(f"  最大绝对误差: {max_error:.1f} RPM")
    print(f"  决定系数(R²): {r2:.4f}")
    
    return y_pred, mae, r2

# 训练集评估
train_pred, train_mae, train_r2 = evaluate_model(
    best_model, X_train, y_train, (Y_min, Y_max), "训练"
)

# 测试集评估
test_pred, test_mae, test_r2 = evaluate_model(
    best_model, X_test, y_test, (Y_min, Y_max), "测试"
)

# ==================== 10. 可视化结果 ====================
def plot_hoop_coord_predictions(model, X_hoop, Y_true, hoop_pos, Y_range, title):
    """在篮筐坐标系中可视化预测结果"""
    # 创建预测网格
    x_range = np.linspace(-2.5, 1.0, 30)  # 篮筐前方2.5m，后方1.0m
    y_range = np.linspace(-1.5, 1.5, 30)  # 篮筐左右各1.5m
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 归一化网格点
    grid_norm = symmetric_normalization(grid_points)
    grid_extended = add_polar_features(grid_norm)
    
    # 预测
    pred_norm = model.predict(grid_extended)
    pred = pred_norm * (Y_range[1] - Y_range[0]) + Y_range[0]
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(
        xx + hoop_pos[0], yy + hoop_pos[1], 
        pred.reshape(xx.shape), 20, cmap='viridis'
    )
    
    # 添加篮筐位置
    plt.scatter(hoop_pos[0], hoop_pos[1], s=300, marker='o', 
                c='red', edgecolors='white', label='篮筐')
    
    # 添加原始数据点
    plt.scatter(
        X_clean[:, 0] + hoop_pos[0], 
        X_clean[:, 1] + hoop_pos[1], 
        c=Y_clean, cmap='viridis', s=50, alpha=0.7,
        edgecolors='k', label='数据点'
    )
    
    plt.colorbar(contour, label='预测转速 (RPM)')
    plt.xlabel('场地X坐标 (m)')
    plt.ylabel('场地Y坐标 (m)')
    plt.title(f'{title}\n测试集MAE: {test_mae:.1f} RPM | R²: {test_r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

# 可视化预测热力图
plot_hoop_coord_predictions(
    best_model, X_hoop, Y_clean, HOOP_POS, (Y_min, Y_max),
    "篮筐坐标系下的转速预测"
)

# ==================== 11. 嵌入式代码生成 ====================
def generate_embedded_code(model, hoop_pos, Y_range, test_mae, test_r2):
    """生成C语言预测代码（篮筐原点版）"""
    # 获取模型参数
    if hasattr(model.named_steps['poly'], 'powers_'):
        degree = model.named_steps['poly'].degree
        coef = model.named_steps['ridge'].coef_
        intercept = model.named_steps['ridge'].intercept_
    else:
        degree = 2
        coef = model.coef_
        intercept = model.intercept_
    
    # 生成C代码
    code = f"""// ==================== 投篮机器人转速预测代码 ====================
// 坐标系: 以篮筐为原点
// 篮筐位置: ({hoop_pos[0]:.2f}f, {hoop_pos[1]:.2f}f)
// 模型类型: {degree}阶多项式回归
// 测试集性能: 
//   MAE = {test_mae:.1f} RPM
//   R² = {test_r2:.4f}

#include <math.h>

// -------------------- 1. 归一化参数 --------------------
const float hoop_x = {hoop_pos[0]:.4f}f;
const float hoop_y = {hoop_pos[1]:.4f}f;

// 对称归一化范围 (根据训练数据计算)
const float norm_x_range = 2.5f;  // X方向归一化范围
const float norm_y_range = 1.5f;  // Y方向归一化范围

const float rpm_min = {Y_range[0]:.1f}f;
const float rpm_max = {Y_range[1]:.1f}f;

// -------------------- 2. 模型系数 --------------------
const float coef[{len(coef)}] = {{
    {', '.join([f'{c:.6f}f' for c in coef])}
}};
const float intercept = {intercept:.6f}f;

// -------------------- 3. 辅助函数 --------------------
// 极坐标转换
void to_polar(float x, float y, float* r, float* theta) {{
    *r = sqrtf(x*x + y*y);
    *theta = atan2f(y, x);
}}

// -------------------- 4. 预测函数 --------------------
float predict_rpm(float world_x, float world_y) {{
    // 1. 转换到篮筐坐标系
    float x = world_x - hoop_x;
    float y = world_y - hoop_y;
    
    // 2. 对称归一化
    float x_norm = x / norm_x_range;
    float y_norm = y / norm_y_range;
    
    // 3. 计算极坐标特征
    float r, theta;
    to_polar(x_norm, y_norm, &r, &theta);
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);
    
    // 4. 构建特征向量 [x, y, r, theta, r², sinθ, cosθ]
    float features[] = {{
        x_norm, y_norm, r, theta, r*r, sin_theta, cos_theta
    }};
    
    // 5. 计算预测值(归一化转速)
    float rpm_norm = intercept;
    for(int i=0; i<{len(coef)}; i++) {{
        rpm_norm += coef[i] * features[i];
    }}
    
    // 6. 反归一化得到实际转速
    float result = rpm_norm * (rpm_max - rpm_min) + rpm_min;
    
    // 7. 输出保护
    if (result < rpm_min) return rpm_min;
    if (result > rpm_max) return rpm_max;
    return result;
}}

// ==================== 使用示例 ====================
/*
#include <stdio.h>

int main() {{
    // 测试点: 篮筐正前方2米处
    float x = hoop_x - 2.0f, y = hoop_y;
    float rpm = predict_rpm(x, y);
    printf("(%.1f, %.1f) -> %.1f RPM\\n", x, y, rpm);
    return 0;
}}
*/"""
    return code

# 生成嵌入式代码
embedded_code = generate_embedded_code(
    best_model, HOOP_POS, (Y_min, Y_max), test_mae, test_r2
)

print("\n" + "="*80)
print("嵌入式C代码（篮筐坐标系版）：")
print(embedded_code)
print("="*80)

# 保存代码到文件
with open("hoop_origin_predictor.c", "w", encoding="utf-8") as f:
    f.write(embedded_code)
print("代码已保存到 hoop_origin_predictor.c")

# ==================== 12. 部署建议 ====================
print("\n" + "="*50 + " 部署建议 " + "="*50)
print("1. 篮筐坐标系使用说明：")
print("   - X正方向：篮筐前方")
print("   - Y正方向：篮筐右侧")
print("   - 原点(0,0)：篮筐中心")

print("\n2. 模型性能分析：")
print(f"   - 测试集MAE: {test_mae:.1f} RPM")
print(f"   - 测试集R²: {test_r2:.4f}")

if test_mae > 100:
    print("\n警告：MAE > 100 RPM，建议：")
    print("  a) 增加篮筐附近数据点采样密度")
    print("  b) 检查异常值处理是否充分")
    print("  c) 尝试SVR或神经网络模型")
else:
    print("\n模型精度满足要求，可直接部署")

print("\n3. 实际部署注意事项：")
print("  - 首次使用时校准篮筐位置")
print("  - 添加距离保护：if (r < 0.3) return 0; // 防止过于接近篮筐")
print("  - 定期用实际投篮数据更新模型")