# coding=utf-8
__author__ = 'hasee'
import paddlets
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets import TSDataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. 安装PaddleTS
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlets
print(paddlets.__version__)

# 2. 构建TSDataset
# 2.1. 内置TSDataset数据集
print(f"built-in datasets: {dataset_list()}")

dataset = get_dataset('UNI_WTH')
print(type(dataset))
dataset.plot()
plt.show()

# 2.2. 构建自定义数据集
x = np.linspace(-np.pi, np.pi, 200)
sinx = np.sin(x) * 4 + np.random.randn(200)

df = pd.DataFrame(
    {
        'time_col': pd.date_range('2022-01-01', periods=200, freq='1h'),
        'value': sinx
    }
)
custom_dataset = TSDataset.load_from_dataframe(
    df,  # Also can be path to the CSV file
    time_col='time_col',
    target_cols='value',
    freq='1h'
)
custom_dataset.plot()
plt.show()

# 3. 数据查看与分析
print(dataset.summary())
from paddlets.analysis import FFT

fft = FFT()
res = fft(dataset, columns='WetBulbCelsius')
fft.plot()
plt.show()

# 4. 模型训练及预测
# 4.1. 构建训练、验证以及测试数据集
train_dataset, val_test_dataset = dataset.split(0.7)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot(add_data=[val_dataset, test_dataset], labels=['Val', 'Test'])
plt.show()
# 4.2. 模型训练
from paddlets.models.forecasting import MLPRegressor

mlp = MLPRegressor(
    in_chunk_len=7 * 24,
    out_chunk_len=24,
    max_epochs=100
)
mlp.fit(train_dataset, val_dataset)
# 4.3. 模型预测
subset_test_pred_dataset = mlp.predict(val_dataset)
subset_test_pred_dataset.plot()
plt.show()
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
subset_test_pred_dataset = mlp.recursive_predict(val_dataset, 24 * 4)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
