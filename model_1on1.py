import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import time
import pandas as pd
import os

# ==================== 1. 初始化设置 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
np.random.seed(42)  # 固定随机种子

# ==================== 2. 真实数据加载 ====================
def load_real_data(file_path):
    """
    从CSV文件加载真实数据集
    数据格式要求: 
        - 至少包含 'distance', 'rpm' 两列
    """
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
        print(f"成功加载数据: {os.path.basename(file_path)}")
        print(f"样本数量: {len(df)}")
        
        # 检查缺失值
        missing_values = df[['distance', 'rpm']].isnull().sum()
        print(f"缺失值统计:\n{missing_values}")
        
        # 删除包含缺失值的行
        df_clean = df.dropna(subset=['distance', 'rpm'])
        if len(df) != len(df_clean):
            print(f"删除包含缺失值的行: {len(df) - len(df_clean)}行")
        
        # 提取距离和转速数据
        X = df_clean[['distance']].values
        Y = df_clean['rpm'].values
        
        # 检查数据范围
        min_dist, max_dist = np.min(X), np.max(X)
        min_rpm, max_rpm = np.min(Y), np.max(Y)
        print(f"距离范围: [{min_dist:.2f}-{max_dist:.2f}]米")
        print(f"转速范围: [{min_rpm:.0f}-{max_rpm:.0f}] RPM")
        
        return X, Y
    
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        print("请检查数据文件路径和格式")
        raise

# 真实数据文件路径
DATA_PATH = "basketball_shot_data.csv"  # 替换为你的数据文件路径

# 加载真实数据
X, Y = load_real_data(DATA_PATH)

# ==================== 3. 异常值处理 ====================
def detect_outliers(X, Y, z_threshold=3.5):
    """使用稳健的Z-Score检测并处理异常值"""
    # 检查NaN值
    if np.isnan(X).any() or np.isnan(Y).any():
        print("警告：异常值检测前发现NaN值")
        # 删除包含NaN的行
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y)
        X_clean = X[~nan_mask]
        Y_clean = Y[~nan_mask]
        print(f"删除包含NaN的行: {len(X) - len(X_clean)}行")
    else:
        X_clean, Y_clean = X.copy(), Y.copy()
    
    # 使用中位数和MAD（中位数绝对偏差）替代均值和标准差
    median = np.median(Y_clean)
    mad = stats.median_abs_deviation(Y_clean)
    
    # 计算稳健的Z分数
    modified_z_scores = 0.6745 * (Y_clean - median) / mad
    
    outlier_mask = np.abs(modified_z_scores) > z_threshold
    X_clean = X_clean[~outlier_mask]
    Y_clean = Y_clean[~outlier_mask]
    
    print(f"\n异常值处理:")
    print(f"  原始样本数: {len(X)}")
    print(f"  检测到异常值: {np.sum(outlier_mask)}个")
    print(f"  清洁数据集: {len(X_clean)}个样本")
    
    # 可视化异常值检测
    plt.figure(figsize=(10, 6))
    plt.scatter(X_clean, Y_clean, c='blue', alpha=0.6, label='正常数据')
    plt.title('异常值检测 (距离 vs RPM)')
    plt.xlabel('距离 (米)')
    plt.ylabel('RPM')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return X_clean, Y_clean

# 检测并移除异常值
X_clean, Y_clean = detect_outliers(X, Y)

# ==================== 4. 数据增强 ====================
def augment_data(X, Y, noise_std=0.05, scaling=True):
    """为数据集添加噪声和缩放样本"""
    X_aug, Y_aug = [X], [Y]
    
    # 1. 添加高斯噪声
    for _ in range(2):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        Y_aug.append(Y)
    
    # 2. 尺度变换增强
    if scaling:
        for scale in [0.95, 1.05]:
            X_scaled = X * scale
            X_aug.append(X_scaled)
            Y_aug.append(Y)
    
    # 合并增强后的数据
    X_aug = np.vstack(X_aug)
    Y_aug = np.concatenate(Y_aug)
    
    # 确保距离值非负（物理约束）
    X_aug[X_aug < 0] = 0
    
    return X_aug, Y_aug

# 应用数据增强
X_aug, Y_aug = augment_data(X_clean, Y_clean, noise_std=0.03)
print(f"\n数据增强后样本数: {len(X_aug)} (原始清洁样本: {len(X_clean)})")

# ==================== 5. 特征工程 ====================
def add_distance_features(X):
    """添加基于距离的衍生特征（确保非负运算）"""
    X = X.reshape(-1)  # 确保是一维数组
    X_non_neg = np.maximum(X, 0)  # 确保非负值用于运算
    
    # 添加更多衍生特征（使用安全运算）
    features = np.column_stack([
        X, 
        np.square(X),  # 平方
        1 / (X_non_neg + 1e-6),  # 倒数（避免除零）
        np.log(X_non_neg + 1e-6),  # 对数
        np.sqrt(X_non_neg),  # 平方根
        np.power(X_non_neg, 0.33)  # 立方根
    ])
    
    # 检查NaN或无穷大
    nan_mask = np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1)
    if np.any(nan_mask):
        print(f"警告：特征工程后检测到 {np.sum(nan_mask)} 个无效值（NaN或Inf）")
        print("删除无效样本...")
        features = features[~nan_mask]
        return features, nan_mask
    
    return features, None

# 添加距离特征
X_extended, nan_mask = add_distance_features(X_aug)
if nan_mask is not None:
    # 同步删除对应的Y值
    Y_aug = Y_aug[~nan_mask]
    print(f"删除无效样本后剩余: {len(Y_aug)}个样本")
    
print(f"特征维度: {X_extended.shape[1]}")

# ==================== 6. 数据预处理 ====================
# SVR对特征缩放敏感，必须进行归一化
scaler = MinMaxScaler(feature_range=(-1, 1))  # 归一化到[-1, 1]范围
X_norm = scaler.fit_transform(X_extended)

# 转速归一化
Y_min, Y_max = np.min(Y_aug), np.max(Y_aug)
Y_norm = (Y_aug - Y_min) / (Y_max - Y_min)

print("\n归一化参数：")
print(f"特征范围: [-1, 1]")
print(f"转速范围: [{Y_min:.1f}, {Y_max:.1f}] RPM")

# 最终检查NaN值
if np.isnan(X_norm).any() or np.isnan(Y_norm).any():
    print("警告：预处理后仍然存在NaN值")
    # 删除包含NaN的行
    nan_mask = np.isnan(X_norm).any(axis=1) | np.isnan(Y_norm)
    X_norm = X_norm[~nan_mask]
    Y_norm = Y_norm[~nan_mask]
    print(f"删除包含NaN的行: {np.sum(nan_mask)}行")

# ==================== 7. SVR模型训练与调参 ====================
def train_svr_model(X_train, y_train):
    """训练并优化SVR模型"""
    print("\n开始SVR模型训练与调参...")
    start_time = time.time()
    
    # 创建基础SVR模型
    svr = SVR(kernel='rbf')
    
    # 参数网格 - 为加速训练选择了合理范围
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],        # 正则化参数
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10], # 核函数系数
        'epsilon': [0.01, 0.05, 0.1, 0.2]     # 不敏感带宽度
    }
    
    # 网格搜索优化
    grid_search = GridSearchCV(
        svr, param_grid, cv=5, 
        scoring='neg_mean_absolute_error',
        verbose=1, n_jobs=-1
    )
    
    # 执行搜索 - 注意：SVR训练可能较慢
    grid_search.fit(X_train, y_train)
    
    # 输出结果
    print(f"调参完成! 耗时: {time.time()-start_time:.1f}秒")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳MAE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 划分训练测试集 (90%训练, 10%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, Y_norm, test_size=0.1, random_state=42
)

# 训练SVR模型
svr_model = train_svr_model(X_train, y_train)

# ==================== 8. 模型评估 ====================
def evaluate_model(model, X, y_true, Y_range, dataset_name):
    """评估模型性能"""
    y_pred_norm = model.predict(X)
    y_pred = y_pred_norm * (Y_range[1] - Y_range[0]) + Y_range[0]
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
    svr_model, X_train, y_train, (Y_min, Y_max), "训练"
)

# 测试集评估
test_pred, test_mae, test_r2 = evaluate_model(
    svr_model, X_test, y_test, (Y_min, Y_max), "测试"
)

# ==================== 9. 可视化结果 ====================
def plot_distance_predictions(model, scaler, Y_range, X_clean, Y_clean):
    """可视化距离与转速的关系"""
    plt.figure(figsize=(12, 8))
    
    # 原始数据点
    plt.scatter(X_clean, Y_clean, c='blue', alpha=0.7, label='原始数据')
    
    # 创建预测距离范围
    min_dist, max_dist = np.min(X_clean), np.max(X_clean)
    dist_range = np.linspace(min_dist-0.2, max_dist+0.2, 200).reshape(-1, 1)
    
    # 为预测准备特征
    dist_features, _ = add_distance_features(dist_range.reshape(-1))
    dist_features_norm = scaler.transform(dist_features)
    
    # 预测
    pred_norm = model.predict(dist_features_norm)
    pred = pred_norm * (Y_range[1] - Y_range[0]) + Y_range[0]
    
    # 绘制预测曲线
    plt.plot(dist_range, pred, 'r-', linewidth=2.5, label='预测曲线')
    
    plt.title(f'距离 vs 转速预测\n测试集MAE: {test_mae:.1f} RPM | R²: {test_r2:.4f}')
    plt.xlabel('到篮筐的距离 (米)')
    plt.ylabel('电机转速 (RPM)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 实际vs预测散点图
    plt.figure(figsize=(10, 8))
    y_test_denorm = y_test * (Y_range[1] - Y_range[0]) + Y_range[0]
    test_pred_denorm = test_pred * (Y_range[1] - Y_range[0]) + Y_range[0]
    
    plt.scatter(y_test_denorm, test_pred_denorm, alpha=0.6)
    plt.plot([Y_range[0], Y_range[1]], [Y_range[0], Y_range[1]], 'r--')
    plt.xlabel('实际转速 (RPM)')
    plt.ylabel('预测转速 (RPM)')
    plt.title(f'实际vs预测转速 (测试集)\nMAE: {test_mae:.1f} RPM | R²: {test_r2:.4f}')
    plt.grid(True)
    plt.show()

# 可视化距离-转速关系
plot_distance_predictions(svr_model, scaler, (Y_min, Y_max), X_clean, Y_clean)

# ==================== 10. 嵌入式代码生成 ====================
def generate_svr_embedded_code(model, scaler, Y_range, test_mae, test_r2):
    """生成SVR模型的C语言预测代码（基于距离）"""
    # 获取模型参数
    support_vectors = model.support_vectors_
    dual_coef = model.dual_coef_[0]
    intercept = model.intercept_[0]
    gamma = model.gamma
    
    # 格式化归一化参数
    feat_min_str = ', '.join([f'{x:.6f}f' for x in scaler.data_min_])
    feat_max_str = ', '.join([f'{x:.6f}f' for x in scaler.data_max_])
    
    # 格式化支持向量
    sv_lines = []
    for vec in support_vectors:
        vec_str = ', '.join([f'{val:.6f}f' for val in vec])
        sv_lines.append(f'    {{{vec_str}}}')
    sv_str = ',\n'.join(sv_lines)
    
    # 生成C代码
    code = f"""// ==================== 投篮机器人转速预测代码 (SVR模型) ====================
// 输入: 到篮筐的距离 (米)
// 输出: 电机转速 (RPM)
// 模型类型: RBF核支持向量回归
// 支持向量数: {len(support_vectors)}
// 测试集性能: 
//   MAE = {test_mae:.1f} RPM
//   R² = {test_r2:.4f}

#include <math.h>

// -------------------- 1. 归一化参数 --------------------
// 特征归一化范围 (训练时使用MinMaxScaler)
const float feat_min[{len(scaler.data_min_)}] = {{{feat_min_str}}};
const float feat_max[{len(scaler.data_max_)}] = {{{feat_max_str}}};

const float rpm_min = {Y_range[0]:.1f}f;
const float rpm_max = {Y_range[1]:.1f}f;

// -------------------- 2. SVR模型参数 --------------------
const float gamma = {gamma:.6f}f;  // RBF核参数
const float intercept = {intercept:.6f}f;

// 支持向量 (共{len(support_vectors)}个)
const float support_vectors[{len(support_vectors)}][{support_vectors.shape[1]}] = {{
{sv_str}
}};

// 对偶系数
const float dual_coefs[{len(dual_coef)}] = {{
    {', '.join([f'{coef:.6f}f' for coef in dual_coef])}
}};

// -------------------- 3. 辅助函数 --------------------
// RBF核函数
float rbf_kernel(const float* x1, const float* x2, int dim) {{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {{
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }}
    return expf(-gamma * sum);
}}

// 特征归一化
void normalize_features(const float* input, float* output, int dim) {{
    for (int i = 0; i < dim; i++) {{
        output[i] = 2.0f * (input[i] - feat_min[i]) / (feat_max[i] - feat_min[i]) - 1.0f;
    }}
}}

// 添加基于距离的特征
void add_distance_features(float distance, float* features) {{
    // 确保距离非负
    float safe_distance = fmaxf(distance, 0.0f);
    
    // 基本特征
    features[0] = distance;
    features[1] = distance * distance;          // 距离平方
    features[2] = 1.0f / (safe_distance + 1e-6f);   // 距离倒数 (避免除零)
    features[3] = logf(safe_distance + 1e-6f);       // 距离对数
    features[4] = sqrtf(safe_distance);              // 距离平方根
    features[5] = powf(safe_distance, 0.333f);       // 距离立方根
}}

// -------------------- 4. 预测函数 --------------------
float predict_rpm(float distance) {{
    // 1. 计算特征
    float raw_features[6];
    add_distance_features(distance, raw_features);
    
    // 2. 归一化特征
    float features[6];
    normalize_features(raw_features, features, 6);
    
    // 3. 计算SVR预测
    float sum = intercept;
    for (int i = 0; i < {len(support_vectors)}; i++) {{
        float k = rbf_kernel(features, support_vectors[i], 6);
        sum += dual_coefs[i] * k;
    }}
    
    // 4. 反归一化得到实际转速
    float rpm_norm = sum;
    float result = rpm_norm * (rpm_max - rpm_min) + rpm_min;
    
    // 5. 输出保护
    if (result < rpm_min) return rpm_min;
    if (result > rpm_max) return rpm_max;
    return result;
}}

// ==================== 使用示例 ====================
/*
#include <stdio.h>
#include <math.h>  // 对于powf, logf等函数

int main() {{
    // 测试点: 距离篮筐2米处
    float rpm = predict_rpm(2.0f);
    printf("距离: 2.0米 -> 预测转速: %.1f RPM\\n", rpm);
    
    // 测试点: 距离篮筐3.5米处
    rpm = predict_rpm(3.5f);
    printf("距离: 3.5米 -> 预测转速: %.1f RPM\\n", rpm);
    
    return 0;
}}
*/"""

    return code

# 生成嵌入式代码
embedded_code = generate_svr_embedded_code(
    svr_model, scaler, (Y_min, Y_max), test_mae, test_r2
)

print("\n" + "="*80)
print("嵌入式C代码（SVR模型版）：")
print(embedded_code)
print("="*80)

# 保存代码到文件
output_file = "distance_based_rpm_predictor.c"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(embedded_code)
print(f"代码已保存到 {output_file}")

# ==================== 11. 部署建议 ====================
print("\n" + "="*50 + " 部署建议 " + "="*50)
print("1. SVR模型特点：")
support_vectors = svr_model.support_vectors_
print(f"   - 支持向量数: {len(support_vectors)}")
print(f"   - 训练时间: 比岭回归长，但预测精度通常更高")

print("\n2. 模型性能分析：")
print(f"   - 测试集MAE: {test_mae:.1f} RPM")
print(f"   - 测试集R²: {test_r2:.4f}")

if test_mae > 100:
    print("\n警告：MAE > 100 RPM，建议：")
    print("  a) 增加训练样本量，特别是不同距离区域")
    print("  b) 调整SVR参数：增加C值或减小epsilon")
    print("  c) 检查数据质量，确保传感器校准正确")
else:
    print("\n模型精度满足要求，可直接部署")

print("\n3. 嵌入式部署注意事项：")
print("  - 确保MCU有足够内存存储支持向量")
memory_usage = len(support_vectors) * support_vectors.shape[1] * 4
print(f"    所需内存: {len(support_vectors)} 向量 × {support_vectors.shape[1]} 特征 × 4 字节 = {memory_usage} 字节")
print("  - 使用硬件浮点运算单元(FPU)加速计算")
print("  - 在真实环境中验证模型性能")