# 代码说明
1. 包括两个模型 [model_based_svr.py](model_based_svr.py) 是二维映射 由x，y 至 电机转速
  [model_1on1.py](model_1on1.py) 是一维映射 由距离 至 电机转速
  模型会自动生成c_code，里面有使用说明
2. 需要提供篮球场的大小，以及篮筐在左下角为原点的坐标系的位置
3. 数据的示例在[basketball_shot_data.csv](basketball_shot_data.csv)当中
