// ==================== 投篮机器人转速预测代码 ====================
// 坐标系: 以篮筐为原点
// 篮筐位置: (3.00f, 2.00f)
// 模型类型: 3阶多项式回归
// 测试集性能: 
//   MAE = 99.2 RPM
//   R² = 0.9236

#include <math.h>

// -------------------- 1. 归一化参数 --------------------
const float hoop_x = 3.0000f;
const float hoop_y = 2.0000f;

// 对称归一化范围 (根据训练数据计算)
const float norm_x_range = 2.5f;  // X方向归一化范围
const float norm_y_range = 1.5f;  // Y方向归一化范围

const float rpm_min = -108.6f;
const float rpm_max = 2063.3f;

// -------------------- 2. 模型系数 --------------------
const float coef[119] = {
    -0.115809f, -0.024480f, 0.083104f, -0.814837f, 0.215194f, 0.169096f, 0.499164f, -0.122344f, -0.519252f, 0.363280f, 0.061535f, -0.327440f, -0.641477f, 0.015136f, 0.337538f, 1.018373f, 0.614312f, 0.044800f, 0.067968f, -0.641477f, 0.215194f, 0.413597f, -0.399201f, -0.024480f, -0.115809f, 0.889233f, -0.314511f, -1.444596f, 1.996856f, -0.103498f, 1.018373f, 0.363280f, -0.096976f, -1.179061f, 0.096976f, -1.166750f, 0.014869f, 0.101472f, 0.890110f, 0.132908f, -0.182941f, 1.065301f, 0.839309f, 0.040399f, 0.583115f, 0.768728f, -0.702021f, -0.182941f, -0.327440f, 0.611400f, 0.361044f, -0.519252f, -0.122344f, -0.917695f, -1.361105f, 0.044234f, 0.200431f, -0.015337f, 0.040399f, 0.101472f, 0.117341f, 0.186705f, -0.233150f, 0.029931f, -0.500673f, -1.204621f, -0.236406f, 1.201314f, -0.702021f, 0.044800f, -0.371655f, -0.114290f, 0.337538f, -0.519252f, -0.519468f, 0.699218f, 0.213166f, 0.044234f, 0.126836f, -0.500673f, 0.040399f, -0.211185f, 0.117341f, 0.186705f, -0.399201f, -0.314511f, -0.103498f, 1.018373f, 0.363280f, -0.889196f, -0.894557f, 0.614312f, 0.061535f, 0.888004f, 0.044800f, -0.327440f, 0.067968f, -0.641477f, 0.015136f, 0.280588f, 0.040156f, 0.439713f, 0.852554f, 0.033686f, -0.371655f, 0.611400f, -0.801467f, -1.199628f, -0.013370f, -0.439549f, -0.114290f, 0.361044f, 0.337538f, -0.519252f, -0.122344f, 0.302453f, 0.765101f, -0.133357f, -0.265937f
};
const float intercept = 0.288412f;

// -------------------- 3. 辅助函数 --------------------
// 极坐标转换
void to_polar(float x, float y, float* r, float* theta) {
    *r = sqrtf(x*x + y*y);
    *theta = atan2f(y, x);
}

// -------------------- 4. 预测函数 --------------------
float predict_rpm(float world_x, float world_y) {
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
    float features[] = {
        x_norm, y_norm, r, theta, r*r, sin_theta, cos_theta
    };
    
    // 5. 计算预测值(归一化转速)
    float rpm_norm = intercept;
    for(int i=0; i<119; i++) {
        rpm_norm += coef[i] * features[i];
    }
    
    // 6. 反归一化得到实际转速
    float result = rpm_norm * (rpm_max - rpm_min) + rpm_min;
    
    // 7. 输出保护
    if (result < rpm_min) return rpm_min;
    if (result > rpm_max) return rpm_max;
    return result;
}

// ==================== 使用示例 ====================
/*
#include <stdio.h>

int main() {
    // 测试点: 篮筐正前方2米处
    float x = hoop_x - 2.0f, y = hoop_y;
    float rpm = predict_rpm(x, y);
    printf("(%.1f, %.1f) -> %.1f RPM\n", x, y, rpm);
    return 0;
}
*/