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

# 模型组网并初始化网络
# 使用paddle提供的LeNet，由于做的是0~9的数字识别，所以num_classes=10分类任务，默认10
lenet = paddle.vision.models.LeNet(num_classes=10)
# 可视化模型组网结构和参数
paddle.summary(net=lenet, input_size=(1, 1, 28, 28))

# 封装模型，便于进行后续的训练、评估和推理
# Model相关方法文档： https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html
model = paddle.Model(lenet)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),  # 优化器
              loss=paddle.nn.CrossEntropyLoss(),  # 损失函数
              metrics=paddle.metric.Accuracy()  # 评价指标
              )

# 开始训练

# epochs (int，可选) - 训练的轮数。默认值：1。
# batch_size (int，可选) - 训练数据或评估数据的批大小，当 train_data 或 eval_data 为 DataLoader 的实例时，该参数会被忽略。默认值：1。
# verbose (int，可选) - 可视化的模型，必须为 0，1，2。当设定为 0 时，不打印日志，设定为 1 时，使用进度条的方式打印日志，设定为 2 时，一行一行地打印日志。默认值：2。
model.fit(train_data=train_dataset, epochs=5, batch_size=64, verbose=1)

# 进行模型评估
model.evaluate(eval_data=test_dataset, batch_size=64, verbose=1)

# 保存模型，文件夹会自动创建
# 代码执行后会在output目录下保存两个文件，mnist.pdopt为优化器的参数，mnist.pdparams为模型的参数。
model.save('./output/mnist')
