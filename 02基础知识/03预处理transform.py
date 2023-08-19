# coding=utf-8
__author__ = 'hasee'

import paddle

# 一、paddle.vision.transforms 介绍

# 图像数据处理方法：
# ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip',
# 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform',
# 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomAffine', 'RandomRotation', 'RandomPerspective',
# 'Grayscale', 'ToTensor', 'RandomErasing', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'affine', 'rotate',
# 'perspective', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue',
# 'normalize', 'erase']
print('图像数据处理方法：', paddle.vision.transforms.__all__)

# 单个使用
transform = paddle.vision.transforms.Resize(size=28)
# 多个组合使用
from paddle.vision.transforms import RandomRotation, Resize

transforms = paddle.vision.transforms.Compose([RandomRotation(10), Resize(size=28)])

# 二、在数据集中应用数据预处理操作
# 2.1 在框架内置数据集中应用
# 通过 transform 字段传递定义好的数据处理方法，即可完成对框架内置数据集的增强
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transforms)
# 2.2 在自定义的数据集中应用
# 对于自定义的数据集，可以在数据集中将定义好的数据处理方法传入 __init__ 函数，将其定义为自定义数据集类的一个属性，
# 然后在 __getitem__ 中将其应用到图像上，见02中自定义Dataset

# 三、数据预处理的几种方法介绍
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(filename='flower_demo.png')

# CenterCrop；从中线点向外裁剪，size为裁剪后图片的大小
# RandomHorizontalFlip：随机概率水平翻转，prob表示翻转概率在0~1之间，1=100%翻转，0.5=50%概率翻转
# ColorJitter：随机调整图像的亮度、对比度、饱和度和色调。
# transform = paddle.vision.transforms.CenterCrop(size=224)
# transform = paddle.vision.transforms.RandomHorizontalFlip(prob=1)
transform = paddle.vision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
transform_image = transform(image)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(transform_image)
plt.show()
