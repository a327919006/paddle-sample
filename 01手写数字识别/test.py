# coding=utf-8
__author__ = 'hasee'
import paddle
import numpy as np
from paddle.vision.transforms import Normalize
# 可视化图片
from matplotlib import pyplot as plt

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')

test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

lenet = paddle.vision.models.LeNet(num_classes=10)
model = paddle.Model(lenet)

# 加载模型
model.load('output/mnist')

# 从测试集中取出一张图片
img, label = test_dataset[0]
# 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)

# 执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
out = model.predict_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))

img = np.array(img[0])  # [1,28,28]
# plt.imshow(img[0])
# 展示灰度图，去掉cmap展示原图
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
