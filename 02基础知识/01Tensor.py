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
# print(tensor)

# 2.5 自动创建 Tensor
from paddle.vision.transforms import Compose, Normalize

# Compose：将用于数据集预处理的接口以列表的方式进行组合
# transform = Compose([Normalize(mean=[127.5],
#                                std=[127.5],
#                                data_format='CHW')])
#
# test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
# print(test_dataset[0][1])  # 打印原始数据集的第一个数据的 label
# loader = paddle.io.DataLoader(test_dataset)
# for data in enumerate(loader):
#     x, label = data[1]
#     print(label)  # 打印由 DataLoader 返回的迭代器中的第一个数据的 label
#     break

# 3.1 Tensor 的形状（shape）
# 3.1.1 形状的介绍
ndim_4_Tensor = paddle.ones([2, 3, 4, 5])
# shape：描述了 Tensor 每个维度上元素的数量
print(ndim_4_Tensor.shape, ndim_4_Tensor.dtype, ndim_4_Tensor.place)
# ndim： Tensor 的维度数量，例如向量的维度为 1，矩阵的维度为 2，Tensor 可以有任意数量的维度。
print(ndim_4_Tensor.ndim)
# axis 或者 dimension：Tensor 的轴，即某个特定的维度。
print(ndim_4_Tensor[0][1])
# size：Tensor 中全部元素的个数。
print(ndim_4_Tensor.size)

# 3.1.2 重置 Tensor 形状（Reshape） 的方法
print("the shape:", ndim_2_Tensor)

reshape_Tensor = paddle.reshape(ndim_2_Tensor, [3, 2])
print("After reshape:", reshape_Tensor)
# -1 表示这个维度的值是从 Tensor 的元素总数和剩余维度自动推断出来的。因此，有且只有一个维度可以被设置为 -1。
print("After reshape:", paddle.reshape(ndim_2_Tensor, [-1]))
print("After reshape:", paddle.reshape(ndim_2_Tensor, [-1, 2]))
# paddle.squeeze，可实现 Tensor 的降维操作，即把 Tensor 中尺寸为 1 的维度删除。
# paddle.unsqueeze，可实现 Tensor 的升维操作，即向 Tensor 中某个位置插入尺寸为 1 的维度。
# paddle.flatten，将 Tensor 的数据在指定的连续维度上展平。
# paddle.transpose，对 Tensor 的数据进行重排。
print("squeeze:", paddle.squeeze(paddle.ones([1, 3])))

# 3.1.3 原位（Inplace）操作和非原位操作的区别
origin_tensor = paddle.to_tensor([1, 2, 3])
new_tensor = paddle.reshape(origin_tensor, [1, 3])  # 非原位操作
same_tensor = paddle.reshape_(origin_tensor, [1, 3])  # 原位操作
print("origin_tensor name: ", origin_tensor.name)
print("new_tensor name: ", new_tensor.name)
print("same_tensor name: ", same_tensor.name)

# 3.2 Tensor 的数据类型（dtype）
# Tensor 的数据类型 dtype 可以通过 Tensor.dtype 查看，
# 支持类型包括：bool、float16、float32、float64、uint8、int8、int16、int32、int64、complex64、complex128。
# 3.2.1 指定数据类型
print(paddle.ones([2, 3], dtype='float32'))
# 3.2.2 修改数据类型的方法
float32_Tensor = paddle.to_tensor(1.0)
float64_Tensor = paddle.cast(float32_Tensor, dtype='float64')
print("Tensor after cast to float64:", float64_Tensor.dtype)
int64_Tensor = paddle.cast(float32_Tensor, dtype='int64')
print("Tensor after cast to int64:", int64_Tensor.dtype)

# 3.3 Tensor 的设备位置（place）
# 初始化 Tensor 时可以通过 Tensor.place 来指定其分配的设备位置，
# 可支持的设备位置有：CPU、GPU、固定内存、XPU（Baidu Kunlun）、NPU（Huawei）、MLU（寒武纪）、IPU（Graphcore）等。
# 当未指定 place 时，Tensor 默认设备位置和安装的飞桨框架版本一致。如安装了 GPU 版本的飞桨，则设备位置默认为 GPU，即 Tensor 的place 默认为 paddle.CUDAPlace。
# 使用 paddle.device.set_device 可设置全局默认的设备位置。Tensor.place 的指定值优先级高于全局默认值。
# 创建 CPU 上的 Tensor
cpu_Tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_Tensor.place)
# 创建 GPU 上的 Tensor
# gpu_Tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
# print(gpu_Tensor.place) # 显示 Tensor 位于 GPU 设备的第 0 张显卡上
# 创建固定内存上的 Tensor
# pin_memory_Tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
# print(pin_memory_Tensor.place)

# 3.4 Tensor 的名称（name）
# Tensor 的名称是其唯一的标识符，为 Python 字符串类型，查看一个 Tensor 的名称可以通过 Tensor.name 属性。
# 默认地，在每个 Tensor 创建时，会自定义一个独一无二的名称。
print(ndim_1_Tensor.name)

# 3.5 Tensor 的 stop_gradient 属性
# stop_gradient 表示是否停止计算梯度，默认值为 True，表示停止计算梯度，梯度不再回传。
# 在设计网络时，如不需要对某些参数进行训练更新，可以将参数的 stop_gradient 设置为 True。

eg = paddle.to_tensor(1)
print("Tensor stop_gradient:", eg.stop_gradient)
eg.stop_gradient = False
print("Tensor stop_gradient:", eg.stop_gradient)

# 4.1 索引和切片
# 4.1.1 访问 Tensor
# 基于 0-n 的下标进行索引，如果下标为负数，则从尾部开始计算。
# 通过冒号 : 分隔切片参数，start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省。
ndim_1_Tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", ndim_1_Tensor.numpy())  # 原始 1 维 Tensor
print("First element:", ndim_1_Tensor[0].numpy())  # 取 Tensor 第一个元素的值
print("Last element:", ndim_1_Tensor[-1].numpy())  # 取 Tensor 最后一个元素的值
print("All element:", ndim_1_Tensor[:].numpy())  # 取 Tensor 所有元素的值
print("Before 3:", ndim_1_Tensor[:3].numpy())  # 取 Tensor 前三个元素的值
print("From 6 to the end:", ndim_1_Tensor[6:].numpy())  # 取 Tensor 第六个以后的值
print("From 3 to 6:", ndim_1_Tensor[3:6].numpy())  # 取 Tensor 第三个至第六个之间的值
print("Interval of 3:", ndim_1_Tensor[::3].numpy())  # 取 Tensor 从第一个开始，间距为 3 的下标的值
print("Reverse:", ndim_1_Tensor[::-1].numpy())  # 取 Tensor 翻转后的值
ndim_2_Tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", ndim_2_Tensor.numpy())
print("First row:", ndim_2_Tensor[0].numpy())
print("First row:", ndim_2_Tensor[0, :].numpy())
print("First column:", ndim_2_Tensor[:, 0].numpy())
print("Last column:", ndim_2_Tensor[:, -1].numpy())
print("All element:", ndim_2_Tensor[:].numpy())
print("First row and second column:", ndim_2_Tensor[0, 1].numpy())

# 4.1.2 修改 Tensor
import numpy as np

x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32))  # [[1., 1., 1.], [1., 1., 1.]]
x[0] = 0  # x : [[0., 0., 0.], [1., 1., 1.]]
x[0:1] = 2.1  # x : [[2.09999990, 2.09999990, 2.09999990], [1., 1., 1.]]
x[...] = 3  # x : [[3., 3., 3.], [3., 3., 3.]]
x[0:1] = np.array([1, 2, 3])  # x : [[1., 2., 3.], [3., 3., 3.]]
x[1] = paddle.ones([3])  # x : [[1., 2., 3.], [1., 1., 1.]]
print("x:", x)

# 4.2 数学运算
x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")
# 可以看出，使用 Tensor 类成员函数 和 Paddle API 具有相同的效果，由于 类成员函数 操作更为方便
# 计算 API 也有原位（inplace）操作和非原位操作之分，如x.add(y)是非原位操作，x.add_(y)为原位操作。
print(paddle.add(x, y), "\n")  # 方法一：使用 Paddle 的 API
print(x.add(y), "\n")  # 方法二：使用 tensor 类成员函数
# x.abs()                       #逐元素取绝对值
# x.ceil()                      #逐元素向上取整
# x.floor()                     #逐元素向下取整
# x.round()                     #逐元素四舍五入
# x.exp()                       #逐元素计算自然常数为底的指数
# x.log()                       #逐元素计算 x 的自然对数
# x.reciprocal()                #逐元素求倒数
# x.square()                    #逐元素计算平方
# x.sqrt()                      #逐元素计算平方根
# x.sin()                       #逐元素计算正弦
# x.cos()                       #逐元素计算余弦
# x.add(y)                      #逐元素相加
# x.subtract(y)                 #逐元素相减
# x.multiply(y)                 #逐元素相乘
# x.divide(y)                   #逐元素相除
# x.mod(y)                      #逐元素相除并取余
# x.pow(y)                      #逐元素幂运算
# x.max()                       #指定维度上元素最大值，默认为全部维度
# x.min()                       #指定维度上元素最小值，默认为全部维度
# x.prod()                      #指定维度上元素累乘，默认为全部维度
# x.sum()                       #指定维度上元素的和，默认为全部维度
# x + y  -> x.add(y)            #逐元素相加
# x - y  -> x.subtract(y)       #逐元素相减
# x * y  -> x.multiply(y)       #逐元素相乘
# x / y  -> x.divide(y)         #逐元素相除
# x % y  -> x.mod(y)            #逐元素相除并取余
# x ** y -> x.pow(y)            #逐元素幂运算

# 4.3 逻辑运算
# x.isfinite()                  #判断 Tensor 中元素是否是有限的数字，即不包括 inf 与 nan
# x.equal_all(y)                #判断两个 Tensor 的全部元素是否相等，并返回形状为[]的布尔类 0-D Tensor
# x.equal(y)                    #判断两个 Tensor 的每个元素是否相等，并返回形状相同的布尔类 Tensor
# x.not_equal(y)                #判断两个 Tensor 的每个元素是否不相等
# x.less_than(y)                #判断 Tensor x 的元素是否小于 Tensor y 的对应元素
# x.less_equal(y)               #判断 Tensor x 的元素是否小于或等于 Tensor y 的对应元素
# x.greater_than(y)             #判断 Tensor x 的元素是否大于 Tensor y 的对应元素
# x.greater_equal(y)            #判断 Tensor x 的元素是否大于或等于 Tensor y 的对应元素
# x.allclose(y)                 #判断 Tensor x 的全部元素是否与 Tensor y 的全部元素接近，并返回形状为[]的布尔类 0-D Tensor
# x == y  -> x.equal(y)         #判断两个 Tensor 的每个元素是否相等
# x != y  -> x.not_equal(y)     #判断两个 Tensor 的每个元素是否不相等
# x < y   -> x.less_than(y)     #判断 Tensor x 的元素是否小于 Tensor y 的对应元素
# x <= y  -> x.less_equal(y)    #判断 Tensor x 的元素是否小于或等于 Tensor y 的对应元素
# x > y   -> x.greater_than(y)  #判断 Tensor x 的元素是否大于 Tensor y 的对应元素
# x >= y  -> x.greater_equal(y) #判断 Tensor x 的元素是否大于或等于 Tensor y 的对应元素
# x.logical_and(y)              #对两个布尔类型 Tensor 逐元素进行逻辑与操作
# x.logical_or(y)               #对两个布尔类型 Tensor 逐元素进行逻辑或操作
# x.logical_xor(y)              #对两个布尔类型 Tensor 逐元素进行逻辑亦或操作
# x.logical_not(y)              #对两个布尔类型 Tensor 逐元素进行逻辑非操作

# 4.4 线性代数
# x.t()                         #矩阵转置
# x.transpose([1, 0])           #交换第 0 维与第 1 维的顺序
# x.norm('fro')                 #矩阵的弗罗贝尼乌斯范数
# x.dist(y, p=2)                #矩阵（x-y）的 2 范数
# x.matmul(y)                   #矩阵乘法

# 五、Tensor 的广播机制
# 可以广播的例子 1
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# 两个 Tensor 形状一致，可以广播
z = x + y
print(z.shape)
# [2, 3, 4]

# 可以广播的例子 2
x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))
# 从最后一个维度向前依次比较：
# 第一次：y 的维度大小是 1
# 第二次：x 的维度大小是 1
# 第三次：x 和 y 的维度大小相等
# 第四次：y 的维度不存在
# 所以 x 和 y 是可以广播的
z = x + y
print(z.shape)
# [2, 3, 4, 5]

# 不可广播的例子
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 6))
# 此时 x 和 y 是不可广播的，因为第一次比较：4 不等于 6
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.

# 六、Tensor 与 Numpy 数组相互转换
# numpy转tensor
tensor_temp = paddle.to_tensor(np.array([1.0, 2.0]))
print(tensor_temp)
# tensor转numpy
tensor_to_convert = paddle.to_tensor([1., 2.])
print(tensor_to_convert.numpy())
