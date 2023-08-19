# coding=utf-8
__author__ = 'hasee'

import paddle
import matplotlib.pyplot as plt

# 一、定义数据集
# 1.1 直接加载内置数据集
# 计算机视觉（CV）相关数据集： ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
# 自然语言处理（NLP）相关数据集： ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16', 'ViterbiDecoder', 'viterbi_decode']
print('计算机视觉（CV）相关数据集：', paddle.vision.datasets.__all__)
print('自然语言处理（NLP）相关数据集：', paddle.text.__all__)

# transform = paddle.vision.transforms.Normalize([127.5], std=[127.5])
# # train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# # test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
# # print('train images: ', len(train_dataset), ', test images: ', len(test_dataset))
# #
# # for data in train_dataset:
# #     image, label = data
# #     print('shape of image', image.shape)
# #     plt.title(str(label))
# #     plt.imshow(image[0])
# #     plt.show()
# #     break

# 1.2 使用 paddle.io.Dataset 自定义数据集
# 下载数据集：https://paddle-imagenet-models-name.bj.bcebos.com/data/mnist.tar
# 可构建一个子类继承自 paddle.io.Dataset ，并且实现下面的三个函数：
# __init__：完成数据集初始化操作，将磁盘中的样本文件路径和对应标签映射到一个列表中。
# __getitem__：定义指定索引（index）时如何获取样本数据，最终返回对应 index 的单条数据（样本数据、对应的标签）。
# __len__：返回数据集的样本总数。
import os
import cv2
import numpy as np
from paddle.io import Dataset
from paddle.vision.transforms import Normalize


class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        with open(label_path, encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 读取灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 打印数据集样本数
train_custom_dataset = MyDataset('mnist/train', 'mnist/train/label.txt', transform)
test_custom_dataset = MyDataset('mnist/val', 'mnist/val/label.txt', transform)
print('train_custom_dataset images: ', len(train_custom_dataset), 'test_custom_dataset images: ',
      len(test_custom_dataset))

# 二、迭代读取数据集
# 2.1 使用 paddle.io.DataLoader 定义数据读取器
# 通过前面介绍的直接迭代读取 Dataset 的方式虽然可实现对数据集的访问，
# 但是这种访问方式只能单线程进行并且还需要手动分批次（batch）。
# 在飞桨框架中，推荐使用 paddle.io.DataLoader API 对数据集进行多进程的读取，并且可自动完成划分 batch 的工作。

# batch_size：每批次读取样本数，示例中 batch_size=64 表示每批次读取 64 个样本。
# shuffle：样本乱序，shuffle=True 表示在取数据时打乱样本顺序，以减少过拟合发生的可能。
# drop_last：丢弃不完整的批次样本，drop_last=True 表示丢弃因数据集样本数不能被 batch_size 整除而产生的最后一个不完整的 batch 样本。
# num_workers：同步/异步读取数据，通过 num_workers 来设置加载数据的子进程个数，num_workers的值设为大于0时，即开启多进程方式异步加载数据，可提升数据读取速度。
train_loader = paddle.io.DataLoader(train_custom_dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)
# 调用 DataLoader 迭代读取数据
for batch_id, data in enumerate(train_loader()):
    images, labels = data
    print("batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(batch_id, images.shape, labels.shape))
    break
