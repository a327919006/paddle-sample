# coding=utf-8
__author__ = 'hasee'
import paddle

# 一、直接使用内置模型

# 飞桨框架内置模型：
# ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext50_64x4d',
# 'resnext101_32x4d', 'resnext101_64x4d', 'resnext152_32x4d', 'resnext152_64x4d', 'wide_resnet50_2',
# 'wide_resnet101_2', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2',
# 'mobilenet_v2', 'MobileNetV3Small', 'MobileNetV3Large', 'mobilenet_v3_small', 'mobilenet_v3_large', 'LeNet',
# 'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenet264', 'AlexNet', 'alexnet',
# 'InceptionV3', 'inception_v3', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'GoogLeNet', 'googlenet',
# 'ShuffleNetV2', 'shufflenet_v2_x0_25', 'shufflenet_v2_x0_33', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
# 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_swish']
print('飞桨框架内置模型：', paddle.vision.models.__all__)

# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)

# 可视化模型组网结构和参数
paddle.summary(lenet, (1, 1, 28, 28))

# 二、Paddle.nn 介绍
# 使用 paddle.nn.Sequential 组网：构建顺序的线性网络结构（如 LeNet、AlexNet 和 VGG）时，可以选择该方式。
# 相比于 Layer 方式 ，Sequential 方式可以用更少的代码完成线性网络的构建。
# 使用 paddle.nn.Layer 组网（推荐）：构建一些比较复杂的网络结构时，可以选择该方式。
# 相比于 Sequential 方式，Layer 方式可以更灵活地组建各种网络结构。Sequential 方式搭建的网络也可以作为子网加入 Layer 方式的组网中。

# 三、使用 paddle.nn.Sequential 组网

from paddle import nn

lenet_Sequential = nn.Sequential(
    nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=400, out_features=120),
    nn.Linear(in_features=120, out_features=84),
    nn.Linear(in_features=84, out_features=10)
)
# 可视化模型组网结构和参数
paddle.summary(lenet, (1, 1, 28, 28))


# 四、使用 paddle.nn.Layer 组网
# 构建一些比较复杂的网络结构时，可以选择该方式，组网包括三个步骤：
# 创建一个继承自 paddle.nn.Layer 的类；
# 在类的构造函数 __init__ 中定义组网用到的神经网络层（layer）；
# 在类的前向计算函数 forward 中使用定义好的 layer 执行前向计算。
# 使用 Subclass 方式构建 LeNet 模型
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 构建 features 子网，用于对输入图像进行特征提取
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))
        # 构建 linear 子网，用于分类
        if num_classes > 0:
            self.linear = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(84, num_classes)
            )

    # 执行前向计算
    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.linear(x)
        return x


lenet_SubClass = LeNet()

# 可视化模型组网结构和参数
params_info = paddle.summary(lenet_SubClass, (1, 1, 28, 28))
print(params_info)
