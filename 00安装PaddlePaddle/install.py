# coding=utf-8
__author__ = 'hasee'

# 安装CPU版本
# python -m pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装GPU版本，需根据CUDA版本决定下载的版本，详见官网
# python -m pip install paddlepaddle-gpu==2.5.1.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html


import paddle

# 安装完成后打印paddle版本
print(paddle.__version__)
# 打印PaddlePaddle is installed successfully!则表示安装成功
paddle.utils.run_check()
