# coding=utf-8
__author__ = 'hasee'
import paddle
import numpy as np

transform = paddle.vision.transforms.Normalize(mean=[127.5], std=[127.5], data_format='CHW')
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
# 每个数字的概率[[ -2.1510553   -3.985444    -0.26829323   1.8520837   -7.1530023    4.7442236  -14.263434    21.189537    -2.8002763    7.695877  ]]
print(out)
# ndarray
print(type(out))
# 取概率最大的下标
pred_label = out.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))
# 可视化图片
from matplotlib import pyplot as plt

plt.imshow(img[0])
plt.show()
