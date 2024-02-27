import torch
from torch import nn
#
# inputs = torch.rand(1, 1, 32, 32)
# outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=18, padding=3, stride=1)(inputs)
# print(outputs.size())
#
# inputs = torch.rand(1, 1, 20, 20)
# outputs = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=16, padding=4, stride=8)(inputs)
# print(outputs.size())

import numpy as np


matrix = np.array([[2, 2, 4],
                   [5, 5, 5],
                   [4, 8, 6]])

# 计算数组中非 NaN 值的平均值
arr = np.diag(matrix)/matrix.sum(axis=1)
arr = np.nanmean(arr)

print(arr)
