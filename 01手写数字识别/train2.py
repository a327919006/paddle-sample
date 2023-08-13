# coding=utf-8
__author__ = 'hasee'
import paddle
from paddle.vision.transforms import Normalize

# 归一化，由于图片的数据值都是0-255，所以mean和std都使用127.5
# 本次MNIST为灰度图，因此只要[127.5]，如果是彩色图使用[127.5,127.5,127.5]
transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')

# image_path (str，可选) - 图像文件路径，如果 download 参数设置为 True，image_path 参数可以设置为 None。默认值为 None，
# 默认存放在：~/.cache/paddle/dataset/mnist。
# label_path (str，可选) - 标签文件路径，如果 download 参数设置为 True，label_path 参数可以设置为 None。默认值为 None，
# 默认存放在：~/.cache/paddle/dataset/mnist。
# mode (str，可选) - 'train' 或 'test' 模式两者之一，默认值为 'train'。
# transform (Callable，可选) - 图片数据的预处理，若为 None 即为不做预处理。默认值为 None。
# download (bool，可选) - 当 data_file 是 None 时，该参数决定是否自动下载数据集文件。默认值为 True。
# backend (str，可选) - 指定要返回的图像类型：PIL.Image 或 numpy.ndarray。必须是 {'pil'，'cv2'} 中的值。
# 如果未设置此选项，将从 paddle.vision.get_image_backend 获得这个值。默认值为 None。
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform, download=True)
# 打印默认返回的图像类型：pil
print(paddle.vision.get_image_backend())
# 打印数据集里图片数量：60000 images in train_dataset, 10000 images in test_dataset
print('{} images in train_dataset, {} images in test_dataset'.format(len(train_dataset), len(test_dataset)))

# 模型组网并初始化网络，由于做的是0~9的数字识别，所以num_classes=10分类任务
lenet = paddle.vision.models.LeNet(num_classes=10)
# 可视化模型组网结构和参数
paddle.summary(lenet, (1, 1, 28, 28))
