# coding=utf-8
__author__ = 'hasee'

import warnings

import pandas
from matplotlib import pyplot as plt
from paddlets import TSDataset
from paddlets.metrics import F1, Precision, Recall
from paddlets.models.model_loader import load
from paddlets.transform import StandardScaler
from paddlets.utils.utils import plot_anoms

warnings.filterwarnings("ignore")

model_type = "bms_temp"
train_file = "data/" + model_type + "_train.csv"
test_file = "data/" + model_type + "_test.csv"
model_path = 'model_' + model_type

# 1. 数据准备
train_df = pandas.read_csv(train_file)
train_tsdata = TSDataset.load_from_dataframe(
    train_df,  # Also can be path to the CSV file
    time_col='timestamp',
    label_col='label',
    feature_cols=['value'],
    freq='1T',
    fill_missing_dates=True,
    fillna_method='pre',
)

test_df = pandas.read_csv(test_file)
print('----------原始数据-起-----------')
print(test_df)
print('----------原始数据-止-----------')
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
model = load(model_path + '/' + model_type)

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
print('pred_label=', pred_label)
print('pred_score=', pred_score)
