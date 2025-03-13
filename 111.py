import torch
import torchvision
print("PyTorch版本: ", torch.version) # 打印PyTorch版本
print("torchvision版本 ", torchvision.version) # 打印torchvision版本
print("CUDA是否可用: ", torch.cuda.is_available()) # 检查CUDA是否可用