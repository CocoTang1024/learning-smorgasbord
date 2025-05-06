# coding=utf-8
'''
FilePath     : /learning-smorgasbord/PyTorch/onnx/resnet18_.py
Author       : CocoTang1024 1972555958@qq.com
Date         : 2024-12-05 10:39:39
Version      : 0.0.1
LastEditTime : 2024-12-05 10:39:44
Email        : robotym@163.com
Description  : 该脚本用于将PyTorch模型转换为ONNX模型，并测试转换后的模型。
'''
import torch
import torchvision.models as models
import onnx
import onnxruntime

# 导入必要的库

# 加载预训练模型
model = models.resnet18(pretrained=True)  # 加载ResNet-18预训练模型
model.eval()  # 设置模型为评估模式

# 定义输入和输出张量的名称和形状
input_names = ["input"]  # 输入张量的名称
output_names = ["output"]  # 输出张量的名称
batch_size = 1  # 批次大小为1
input_shape = (batch_size, 3, 224, 224)  # 输入张量的形状
output_shape = (batch_size, 1000)  # 输出张量的形状

# 将当前的PyTorch模型转化成onnx模型
torch.onnx.export(
    model, 
    torch.randn(input_shape),  # 生成一个随机张量作为输入
    "resnet18.onnx",  # 导出的ONNX模型文件名
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}  # 动态轴，即输入和输出张量可以具有不同的批次大小
)

# 加载onnx模型
onnx_model = onnx.load("resnet18.onnx")  # 加载导出的ONNX模型
onnx_model_graph = onnx_model.graph
onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())  # 创建ONNX运行时会话

# 使用随机张量测试ONNX模型
x = torch.randn(input_shape).numpy()  # 生成一个随机张量并转换为NumPy数组
onnx_output = onnx_session.run(output_names, {input_names[0]: x})[0]  # 在ONNX模型上运行输入张量并获取输出

print(f"ONNX output shape: {onnx_output[0, :5]}")  # 打印ONNX模型输出的前5个元素
print(f"PyTorch output shape: {model(torch.tensor(x)).detach().numpy()[0, :5]}")  # 使用PyTorch模型计算输出并打印前5个元素