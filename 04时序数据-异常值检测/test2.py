# coding=utf-8
__author__ = 'hasee'
# 导入其他模块
import paddle
import paddle.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings

# 异常数据预览
df_daily_jumpsup_path = './data_nab/data_error.csv'
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_path, parse_dates=True, index_col="timestamp"
)

print(df_daily_jumpsup)
# 异常的时序数据可视化
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
plt.show()


class AutoEncoder(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv0 = paddle.nn.Conv1D(in_channels=1, out_channels=32, kernel_size=7, stride=2)
        self.conv1 = paddle.nn.Conv1D(in_channels=32, out_channels=16, kernel_size=7, stride=2)
        self.convT0 = paddle.nn.Conv1DTranspose(in_channels=16, out_channels=32, kernel_size=7, stride=2)
        self.convT1 = paddle.nn.Conv1DTranspose(in_channels=32, out_channels=1, kernel_size=7, stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.convT0(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.convT1(x)
        return x

# 初始化并保存得到的均值和方差，用于初始化数据。
training_mean = 42.438353
training_std = 28.077122
df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()
# 这是测试集里面的异常数据，可以看到第11~~12天发生了异常

# 探测异常数据
threshold = 0.033  # 阀值设定，即刚才求得的值
param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
mse_loss = paddle.nn.loss.MSELoss()


def create_sequences(values, time_steps=288):
    '''
    探测数据预处理
    '''
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


x_test = create_sequences(df_test_value.values)
print(x_test)
print(x_test.shape)
x = paddle.to_tensor(x_test).astype('float32')
abnormal_index = []  # 记录检测到异常时数据的索引

indexs = range(len(x_test))
for i in indexs:
    input_x = paddle.reshape(x[i], (1, 1, 288))
    out = model(input_x)
    loss = mse_loss(input_x[:, :, :-1], out)
    if loss.numpy()[0] > threshold:
        # 开始检测到异常时序列末端靠近异常点，所以要加上序列长度，得到真实索引位置
        abnormal_index.append(i + 288)

# 不再检测异常时序列的前端靠近异常点，所以要减去索引长度得到异常点真实索引，为了结果明显，给异常位置加宽40单位
abnormal_index = abnormal_index[:(-288)]
print(len(abnormal_index))
print(abnormal_index)
# 异常检测结果可视化
df_subset = df_daily_jumpsup.iloc[abnormal_index]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()
