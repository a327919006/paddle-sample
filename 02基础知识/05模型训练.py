# coding=utf-8
__author__ = 'hasee'
import paddle
import numpy as np

# 一、训练前准备
# 1.1 （可选）指定训练的硬件
# 指定在 CPU 上训练
paddle.device.set_device('cpu')
# 指定在 GPU 第 0 号卡上训练
# paddle.device.set_device('gpu:0')

# 1.2 准备训练用的数据集和模型
from paddle.vision.transforms import Normalize

transform = Normalize(mean=[127.5], std=[127.5])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(1, -1),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 二、使用 paddle.Model 高层 API 训练、评估与推理
# 2.1 使用 paddle.Model 封装模型
# 使用高层 API 训练模型前，可使用 paddle.Model 将模型封装为一个实例，方便后续进行训练、评估与推理。代码如下：
model = paddle.Model(mnist)

# 2.2 使用 Model.prepare 配置训练准备参数
# 优化器（optimizer）：即寻找最优解的方法，可计算和更新梯度，并根据梯度更新模型参数。飞桨框架在 paddle.optimizer 下提供了优化器相关 API。并且需要为优化器设置合适的学习率，或者指定合适的学习率策略，飞桨框架在 paddle.optimizer.lr 下提供了学习率策略相关的 API。
# 损失函数（loss）：用于评估模型的预测值和真实值的差距，模型训练过程即取得尽可能小的 loss 的过程。飞桨框架在 paddle.nn Loss层 提供了适用不同深度学习任务的损失函数相关 API。
# 评价指标（metrics）：用于评估模型的好坏，不同的任务通常有不同的评价指标。飞桨框架在 paddle.metric 下提供了评价指标相关 API。
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

# 2.3 使用 Model.fit 训练模型
# 训练数据集：传入之前定义好的训练数据集。
# 训练轮次（epoch）：训练时遍历数据集的次数，即外循环轮次。
# 批次大小（batch_size）：内循环中每个批次的训练样本数。
model.fit(train_data=train_dataset, batch_size=64, epochs=5, verbose=1)

# 2.4 使用 Model.evaluate 评估模型
eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)

# 2.5 使用 Model.predict 执行推理
img, label = test_dataset[0]
# 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)
test_result = model.predict_batch(img_batch)[0]
print(test_result)
pred_label = test_result.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))
# 可视化图片
from matplotlib import pyplot as plt

plt.imshow(img[0])
plt.show()

# 三、使用基础 API 训练、评估与推理
# 3.1 模型训练（拆解 Model.prepare、Model.fit）
