import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==================== 1. 参数配置 ====================
# 场地尺寸 (单位：米)
COURT_WIDTH = 3.5    # x方向 [0.5, 3.5]
COURT_HEIGHT = 2.5   # y方向 [0.5, 2.5]

# 篮筐位置 (右上角区域)
HOOP_X = 3.0
HOOP_Y = 2.0

# 数据生成参数
NUM_SAMPLES = 50    # 总样本数
OUTLIER_RATIO = 0.1  # 异常值比例
NOISE_LEVEL = 50     # 正常噪声水平 (RPM)
OUTLIER_NOISE = 400  # 异常值噪声水平 (RPM)

# 输出文件路径
OUTPUT_FILE = "basketball_shot_data.csv"

# ==================== 2. 数据生成函数 ====================
def generate_shot_data(n_samples, hoop_pos, noise_level, outlier_ratio, outlier_noise):
    """
    生成投篮模拟数据
    返回: DataFrame包含 x, y, rpm 列
    """
    # 生成均匀分布的坐标
    x_coords = np.random.uniform(0.5, COURT_WIDTH, n_samples)
    y_coords = np.random.uniform(0.5, COURT_HEIGHT, n_samples)
    
    # 计算到篮筐的距离和角度
    distances = np.sqrt((x_coords - hoop_pos[0])**2 + (y_coords - hoop_pos[1])**2)
    angles = np.arctan2(hoop_pos[1]-y_coords, hoop_pos[0]-x_coords)
    
    # 基础RPM计算公式
    base_rpm = 1200 + 100 * distances**1.5
    angle_factor = np.cos(angles)**2  # 角度影响因子
    
    # 添加正常噪声
    rpm_values = base_rpm * angle_factor + np.random.normal(0, noise_level, n_samples)
    
    # 添加异常值
    if outlier_ratio > 0:
        n_outliers = int(n_samples * outlier_ratio)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        rpm_values[outlier_indices] += np.random.uniform(outlier_noise*0.8, outlier_noise*1.2, n_outliers)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'rpm': rpm_values,
        'hoop_x': hoop_pos[0],  # 记录篮筐位置
        'hoop_y': hoop_pos[1]
    })
    
    return data

# ==================== 3. 生成并保存数据 ====================
# 生成数据
shot_data = generate_shot_data(
    n_samples=NUM_SAMPLES,
    hoop_pos=[HOOP_X, HOOP_Y],
    noise_level=NOISE_LEVEL,
    outlier_ratio=OUTLIER_RATIO,
    outlier_noise=OUTLIER_NOISE
)

# 保存到CSV
shot_data.to_csv(OUTPUT_FILE, index=False)
print(f"已生成 {NUM_SAMPLES} 条投篮数据并保存到 {OUTPUT_FILE}")

# ==================== 4. 数据可视化 ====================
def plot_shot_data(data, hoop_pos):
    """可视化生成的投篮数据"""
    plt.figure(figsize=(12, 8))
    
    # 绘制数据点
    scatter = plt.scatter(
        data['x'], data['y'], 
        c=data['rpm'], cmap='viridis',
        s=50, alpha=0.7, edgecolors='k'
    )
    
    # 标记篮筐位置
    plt.scatter(
        hoop_pos[0], hoop_pos[1],
        s=300, marker='o', c='red',
        edgecolors='white', label='篮筐'
    )
    
    # 添加颜色条
    plt.colorbar(scatter, label='转速 (RPM)')
    
    # 设置图形属性
    plt.xlim(0, COURT_WIDTH + 0.5)
    plt.ylim(0, COURT_HEIGHT + 0.5)
    plt.xlabel('场地X坐标 (m)')
    plt.ylabel('场地Y坐标 (m)')
    plt.title(f'投篮数据分布 (样本数: {len(data)})')
    plt.legend()
    plt.grid(True)
    
    # 显示场地边界
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(COURT_HEIGHT, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(COURT_WIDTH, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()

# 可视化数据
plot_shot_data(shot_data, [HOOP_X, HOOP_Y])

# ==================== 5. 数据质量检查 ====================
def check_data_quality(data):
    """检查生成数据的质量"""
    print("\n数据质量检查:")
    print("="*40)
    
    # 基本统计信息
    print("坐标范围:")
    print(f"  x: [{data['x'].min():.2f}, {data['x'].max():.2f}] m")
    print(f"  y: [{data['y'].min():.2f}, {data['y'].max():.2f}] m")
    print(f"  rpm: [{data['rpm'].min():.0f}, {data['rpm'].max():.0f}] RPM")
    
    # 异常值检测
    z_scores = np.abs(stats.zscore(data['rpm']))
    outliers = data[z_scores > 3]
    print(f"\n检测到潜在异常值: {len(outliers)} 个 ({len(outliers)/len(data)*100:.1f}%)")
    
    # 分布可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.hist(data['rpm'], bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('RPM')
    plt.ylabel('频数')
    plt.title('转速分布')
    
    plt.subplot(122)
    plt.scatter(data['x'], data['y'], c='blue', alpha=0.5, s=10)
    plt.scatter(outliers['x'], outliers['y'], c='red', s=30, label='异常值')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('异常值空间分布')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 执行数据质量检查
check_data_quality(shot_data)