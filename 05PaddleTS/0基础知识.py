# coding=utf-8
__author__ = 'hasee'
import warnings

import numpy as np
import paddlets
import pandas as pd
from matplotlib import pyplot as plt
from paddlets import TSDataset
from paddlets.datasets.repository import get_dataset, dataset_list

warnings.filterwarnings("ignore")

# 1. 安装PaddleTS
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlets
print(paddlets.__version__)

# 2. 构建TSDataset
# 2.1. 内置TSDataset数据集
# datasets: ['UNI_WTH', 'ETTh1', 'ETTm1', 'ECL', 'WTH', 'NAB_TEMP', 'psm_train', 'psm_test',
# 'BasicMotions_Train', 'BasicMotions_Test']
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
plt.show()
subset_test_pred_dataset = mlp.recursive_predict(val_dataset, 24 * 4)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

# 5. 评估和回测
from paddlets.metrics import MAE

mae = MAE()
mae(subset_test_dataset, subset_test_pred_dataset)
# {'WetBulbCelsius': 0.6734366664042076}
from paddlets.utils import backtest

metrics_score = backtest(
    data=val_test_dataset,
    model=mlp,
    start=0.5,
    predict_window=24,
    stride=24,
    metric=mae
)
print(f"mae: {metrics_score}")
# mae: 1.3767653357878213

# 6. 协变量

from paddlets.transform import TimeFeatureGenerator

time_feature_generator = TimeFeatureGenerator(feature_cols=['dayofyear', 'weekofyear', 'is_workday'])
dataset_gen_target_cov = time_feature_generator.fit_transform(dataset)
print(dataset_gen_target_cov)
print(dataset_gen_target_cov.known_cov)
# 6.1. 自动构建日期相关协变量
# 使用 paddlets.transform 中的 TimeFeatureGenerator 去自动生成日期与时间相关的协变量。如是否节假日，
# 当前是每年的第几周等信息，因为这些信息在预测未来数据的时候也是已知的，因此其属于 known_covariate(已知协变量)。
# 在以下示例中，我们会生成三个时间相关的协变量，分别代表 一年中的第几天 、一周中的第几天、 是否是工作日 。
from paddlets.transform import TimeFeatureGenerator

time_feature_generator = TimeFeatureGenerator(feature_cols=['dayofyear', 'weekofyear', 'is_workday'])
dataset_gen_target_cov = time_feature_generator.fit_transform(dataset)
print(dataset_gen_target_cov)
print(dataset_gen_target_cov.known_cov)

# 6.2. 自定义协变量
import pandas as pd
from paddlets import TSDataset

df = pd.DataFrame(
    {
        'time_col': pd.date_range(
            dataset.target.time_index[0],
            periods=len(dataset.target),
            freq=dataset.freq
        ),
        'cov1': [i for i in range(len(dataset.target))]
    }
)
dataset_cus_cov = TSDataset.load_from_dataframe(
    df,
    time_col='time_col',
    known_cov_cols='cov1',
    freq=dataset.freq
)
print(dataset_cus_cov)
dataset_cus_target_cov = TSDataset.concat([dataset, dataset_cus_cov])
print(dataset_cus_target_cov)
