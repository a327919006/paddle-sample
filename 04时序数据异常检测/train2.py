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

warnings.filterwarnings("ignore")
# 正常数据预览
df_small_noise_path = './artificialNoAnomaly/art_daily_small_noise.csv'
df_small_noise = pd.read_csv(
    df_small_noise_path, parse_dates=True, index_col="timestamp"
)

print(df_small_noise)

# 正常的时序数据可视化
fig, ax = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
plt.show()

# 初始化并保存得到的均值和方差，用于初始化数据。
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("training_mean:", training_mean)
print("training_std:", training_std)
print("训练数据总量:", len(df_training_value))

# 时序步长
TIME_STEPS = 288


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


# 实例化数据集
train_dataset = MyDataset(df_training_value.values, TIME_STEPS)
print(train_dataset)


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


# 参数设置
epoch_num = 200
batch_size = 128
learning_rate = 0.001


def train():
    print('训练开始')
    # 实例化模型
    model = AutoEncoder()
    # 将模型转换为训练模式
    model.train()
    # 设置优化器，学习率，并且把模型参数给优化器
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    # 设置损失函数
    mse_loss = paddle.nn.MSELoss()
    # 设置数据读取器
    data_reader = paddle.io.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True)
    history_loss = []
    iter_epoch = []
    for epoch in tqdm.tqdm(range(epoch_num)):
        for batch_id, data in enumerate(data_reader()):
            x = data[0]
            y = data[1]
            out = model(x)
            avg_loss = mse_loss(out, (y[:, :, :-1]))  # 输入的数据经过卷积会丢掉最后一个数据
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        iter_epoch.append(epoch)
        history_loss.append(avg_loss.numpy()[0])
    # 绘制loss
    plt.plot(iter_epoch, history_loss, label='loss')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('Loss')
    plt.show()
    # 保存模型参数
    paddle.save(model.state_dict(), 'model')


# train()

# 计算阀值
param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
total_loss = []
datas = []
# 预测所有正常时序
mse_loss = paddle.nn.loss.MSELoss()
# 这里设置batch_size为1，单独求得每个数据的loss
data_reader = paddle.io.DataLoader(train_dataset,
                                   places=[paddle.CPUPlace()],
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=0)
for batch_id, data in enumerate(data_reader()):
    x = data[0]
    y = data[1]
    out = model(x)
    avg_loss = mse_loss(out, (y[:, :, :-1]))
    total_loss.append(avg_loss.numpy()[0])
    datas.append(batch_id)

plt.bar(datas, total_loss)
plt.ylabel("reconstruction loss")
plt.xlabel("data samples")
plt.show()

# 获取重建loss的阀值
threshold = np.max(total_loss)
print("阀值:", threshold)
