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
from paddlets.models.model_loader import load

warnings.filterwarnings("ignore")

# 1. 数据准备
train_df = pandas.read_csv("data/train_hudi.csv")
train_tsdata = TSDataset.load_from_dataframe(
    train_df,  # Also can be path to the CSV file
    time_col='timestamp',
    label_col='label',
    feature_cols=['value'],
    freq='1T',
    fill_missing_dates=True,
    fillna_method='pre',
)

test_df = pandas.read_csv("data/test_hudi.csv")
print(test_df)
test_tsdata = TSDataset.load_from_dataframe(
    test_df,  # Also can be path to the CSV file
    time_col='timestamp',
    label_col='label',
    feature_cols=['value'],
    freq='1T',
    fill_missing_dates=True,
    fillna_method='pre',
)
plot_anoms(origin_data=test_tsdata, feature_name='value')
plt.show()

# 3. 数据处理
# standardize
scaler = StandardScaler('value')
scaler.fit(train_tsdata)
test_tsdata_scaled = scaler.transform(test_tsdata)

# 4. 模型加载
model = load('./model_hudi/ae')

# 5. 模型预测和评估
pred_label = model.predict(test_tsdata_scaled)
lable_name = pred_label.target.data.columns[0]
f1 = F1()(test_tsdata, pred_label)
precision = Precision()(test_tsdata, pred_label)
recall = Recall()(test_tsdata, pred_label)
print('f1: ', f1[lable_name])
print('precision: ', precision[lable_name])
print('recall: ', recall[lable_name])

# 6. 预测结果可视化
plot_anoms(origin_data=test_tsdata, predict_data=pred_label, feature_name="value")
plt.show()
pred_score = model.predict_score(test_tsdata_scaled)
plot_anoms(origin_data=test_tsdata, predict_data=pred_score, feature_name="value")
plt.show()

# 7. 预测值打印
pred_label = model.predict(test_tsdata_scaled)
pred_score = model.predict_score(test_tsdata_scaled)
print('pred_label=', pred_label)
print('pred_score=', pred_score)
