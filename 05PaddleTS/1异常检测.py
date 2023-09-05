# coding=utf-8
__author__ = 'hasee'
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.utils.utils import plot_anoms
from matplotlib import pyplot as plt
import paddle
import numpy as np
from paddlets.transform import StandardScaler
from paddlets.models.anomaly import AutoEncoder
from paddlets.metrics import F1, ACC, Precision, Recall
import warnings

warnings.filterwarnings("ignore")

# 1. 数据准备
# datasets: ['UNI_WTH', 'ETTh1', 'ETTm1', 'ECL', 'WTH', 'NAB_TEMP', 'psm_train', 'psm_test',
# 'BasicMotions_Train', 'BasicMotions_Test']
print(f"built-in datasets: {dataset_list()}")
ts_data = get_dataset('NAB_TEMP')  # label_col: 'label', feature_cols: 'value'
print(ts_data)

# 2. 数据可视化
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

# 7. 模型持久化
model.save('./model/ae')
from paddlets.models.model_loader import load

loaded_model = load('./model/ae')
pred_label = loaded_model.predict(test_tsdata_scaled)
pred_score = loaded_model.predict_score(test_tsdata_scaled)
print('pred_label=', pred_label)
print('pred_score=', pred_score)
