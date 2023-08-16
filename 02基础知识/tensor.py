# coding=utf-8
__author__ = 'hasee'
import paddle

# 飞桨使用张量（Tensor） 来表示神经网络中传递的数据，Tensor 可以理解为多维数组，类似于 Numpy 数组（ndarray） 的概念。
# https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_cn.html

# 2.1 指定数据创建
# 2.1.1 创建1 维 Tensor
ndim_1_Tensor = paddle.to_tensor([1.1, 2.2, 3.3])
print(ndim_1_Tensor)

print(paddle.to_tensor(1))
print(paddle.to_tensor([2]))

# 2.1.2 创建2 维 Tensor
ndim_2_Tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_Tensor)

# 2.1.3 创建3 维 Tensor
ndim_3_Tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_Tensor)

# 2.2 指定形状创建
print(paddle.zeros([2, 3]))
# dtype指定数据类型，默认'float32'
print(paddle.ones([2, 3], dtype='int64'))
print(paddle.full([2, 3], 127))
# 创建一个空 Tensor，即根据 shape 和 dtype 创建尚未初始化元素值的 Tensor，可通过 paddle.empty 实现。
print(paddle.empty([2, 4]))
# 创建一个与其他 Tensor 具有相同 shape 与 dtype 的 Tensor，可通过 paddle.ones_like 、 paddle.zeros_like 、 paddle.full_like 、paddle.empty_like 实现。
print(paddle.ones_like(ndim_1_Tensor))
# 拷贝并创建一个与其他 Tensor 完全相同的 Tensor，可通过 paddle.clone 实现。
print(paddle.clone(ndim_1_Tensor))

# 2.3 指定区间创建
# paddle.arange(start, end, step)  # 创建以步长 step 均匀分隔区间[start, end)的 Tensor
print(paddle.arange(0, 5, 1))
# paddle.linspace(start, stop, num) # 创建以元素个数 num 均匀分隔区间[start, stop)的 Tensor
print(paddle.linspace(0, 10, 5))
# 创建一个满足特定分布的 Tensor，如 paddle.rand, paddle.randn , paddle.randint 等。
print(paddle.rand([2, 3]))
print(paddle.randn([2, 3]))
print(paddle.randint(0, 10, [2, 3]))

# 2.4 指定图像、文本数据创建
import numpy as np
from PIL import Image

fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))  # 创建随机图片
transform = paddle.vision.transforms.ToTensor()
tensor = transform(fake_img)  # 使用 ToTensor()将图片转换为 Tensor
print(tensor)

# 2.5 自动创建 Tensor
from paddle.vision.transforms import Compose, Normalize

# Compose：将用于数据集预处理的接口以列表的方式进行组合
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print(test_dataset[0][1])  # 打印原始数据集的第一个数据的 label
loader = paddle.io.DataLoader(test_dataset)
for data in enumerate(loader):
    x, label = data[1]
    print(label)  # 打印由 DataLoader 返回的迭代器中的第一个数据的 label
    break

# 3.1 Tensor 的形状（shape）
ndim_4_Tensor = paddle.ones([2, 3, 4, 5])
# shape：描述了 Tensor 每个维度上元素的数量
print(ndim_4_Tensor.shape, ndim_4_Tensor.dtype, ndim_4_Tensor.place)
# ndim： Tensor 的维度数量，例如向量的维度为 1，矩阵的维度为 2，Tensor 可以有任意数量的维度。
print(ndim_4_Tensor.ndim)
# axis 或者 dimension：Tensor 的轴，即某个特定的维度。
print(ndim_4_Tensor[0][1])
# size：Tensor 中全部元素的个数。
print(ndim_4_Tensor.size)
