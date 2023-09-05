# coding=utf-8
__author__ = 'hasee'
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.utils.utils import plot_anoms
from matplotlib import pyplot as plt
from paddlets import TSDataset
import paddle
import pandas
import numpy as np
from paddlets.transform import StandardScaler
from paddlets.models.anomaly import AutoEncoder
from paddlets.metrics import F1, ACC, Precision, Recall
import warnings

warnings.filterwarnings("ignore")

# 1. 数据准备
df = pandas.read_csv("NAB_TEMP_tmp.csv")
print(df)
ts_data = TSDataset.load_from_dataframe(
    df,  # Also can be path to the CSV file
    time_col='timestamp',
    label_col='label',
    feature_cols=['value'],
    freq='5T'
)
plot_anoms(origin_data=ts_data, feature_name='value')
plt.show()

# 3. 数据处理
# 设置全局默认 generator 的随机种子。
seed = 2022
paddle.seed(seed)
np.random.seed(seed)
train_tsdata, test_tsdata = ts_data.split(0.15)

# standardize
scaler = StandardScaler('value')
scaler.fit(train_tsdata)
train_tsdata_scaled = scaler.transform(train_tsdata)
test_tsdata_scaled = scaler.transform(test_tsdata)

# 4. 模型训练
model = AutoEncoder(in_chunk_len=2, max_epochs=100)
model.fit(train_tsdata_scaled)


# 5. 模型预测和评估


# 6. 预测结果可视化


# 7. 模型持久化
