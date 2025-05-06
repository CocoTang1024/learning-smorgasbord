# coding=utf-8
'''
FilePath     : /learning-smorgasbord/PyTorch/onnx/onnx.py
Author       : CocoTang1024 1972555958@qq.com
Date         : 2024-12-05 10:23:32
Version      : 0.0.1
LastEditTime : 2024-12-05 10:33:46
Email        : robotym@163.com
Description  : 
'''
import torch
import torchvision
import numpy as np

# 判断是否存在GPU，如果存在则使用GPU进行计算，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个简单的Pytorch模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # 定义第一个卷积层
        self.relu = torch.nn.ReLU() # 定义激活函数ReLU
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 定义最大池化层
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 定义第二个卷积层
        self.flatten = torch.nn.Flatten() # 定义展平层
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 10) # 定义全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 创建模型实例并移动到GPU或CPU
model = MyModel().to(device)

# 指定当前的模型的输入尺寸
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# 将PyTorch模型转换成onnx模型，不进行常量折叠
torch.onnx.export(model, dummy_input, "mymodel.onnx", do_constant_folding=False)