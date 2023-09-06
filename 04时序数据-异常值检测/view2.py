# coding=utf-8
__author__ = 'hasee'
# 导入其他模块
import paddle
import paddle.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

import warnings


class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, data, time_steps):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        注意：这个是不需要label
        """
        super().__init__()
        self.time_steps = time_steps
        self.data = paddle.to_tensor(self.transform(data), dtype='float32')

    def transform(self, data):
        '''
        构造时序数据
        '''
        output = []
        for i in range(len(data) - self.time_steps):
            output.append(np.reshape(data[i: (i + self.time_steps)], (1, self.time_steps)))
        return np.stack(output)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        data = self.data[index]
        label = self.data[index]
        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


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


df_daily_jumpsup_path = './data_nab/data_error.csv'
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_path, parse_dates=True, index_col="timestamp"
)

print(df_daily_jumpsup)
training_mean = 42.438353
training_std = 28.077122
df_test_value = (df_daily_jumpsup - training_mean) / training_std
train_dataset = MyDataset(df_test_value.values, 288)

param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
data_reader = paddle.io.DataLoader(train_dataset,
                                   places=[paddle.CPUPlace()],
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=0)
for batch_id, data in enumerate(data_reader()):
    x = data[0]
    out = model(x)
    step = np.arange(287)
    plt.plot(step, x[0, 0, :-1].numpy())
    plt.plot(step, out[0, 0].numpy())
    plt.show()
