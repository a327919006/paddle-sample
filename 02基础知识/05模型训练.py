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
#方式一：设置训练过程中保存模型，save_freq可以设置保存动态图模型的频率，即多少个 epoch 保存一次模型，默认值是 1。
model.load('output/mnist')
model.fit(train_data=train_dataset, batch_size=64, epochs=2, verbose=1)
#方式二：设置训练后保存模型
# training=False时表示静态模型用于预测
model.save('output/mnisttest', training=True)  # save for training

# 2.4 使用 Model.evaluate 评估模型
# eval_result = model.evaluate(test_dataset, verbose=1)
# print(eval_result)

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
# dataset与mnist的定义与使用高层API的内容一致
# 用 DataLoader 实现数据加载
# train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
# # 将mnist模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。
# mnist.train()
# # 设置迭代次数
# epochs = 1
# # 设置优化器
# optim = paddle.optimizer.Adam(parameters=mnist.parameters())
# # 设置损失函数
# loss_fn = paddle.nn.CrossEntropyLoss()
# for epoch in range(epochs):
#     for batch_id, data in enumerate(train_loader()):
#
#         x_data = data[0]  # 训练数据
#         y_data = data[1]  # 训练数据标签
#         predicts = mnist(x_data)  # 预测结果
#
#         # 计算损失 等价于 prepare 中loss的设置
#         loss = loss_fn(predicts, y_data)
#
#         # 计算准确率 等价于 prepare 中metrics的设置
#         acc = paddle.metric.accuracy(predicts, y_data)
#
#         # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
#         # 反向传播
#         loss.backward()
#
#         if (batch_id + 1) % 900 == 0:
#             print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id + 1, loss.numpy(),
#                                                                             acc.numpy()))
#         # 更新参数
#         optim.step()
#         # 梯度清零
#         optim.clear_grad()
#
# # 加载测试数据集
# test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
# # 设置损失函数
# loss_fn = paddle.nn.CrossEntropyLoss()
# # 将该模型及其所有子层设置为预测模式。这只会影响某些模块，如Dropout和BatchNorm
# mnist.eval()
# # 禁用动态图梯度计算
# for batch_id, data in enumerate(test_loader()):
#
#     x_data = data[0]            # 测试数据
#     y_data = data[1]            # 测试数据标签
#     predicts = mnist(x_data)    # 预测结果
#
#     # 计算损失与精度
#     loss = loss_fn(predicts, y_data)
#     acc = paddle.metric.accuracy(predicts, y_data)
#
#     # 打印信息
#     if (batch_id+1) % 30 == 0:
#         print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id+1, loss.numpy(), acc.numpy()))