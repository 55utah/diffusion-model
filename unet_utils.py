import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import matplotlib.animation as animation
from matplotlib import pyplot as plt

'''
卷积核计算公式：
卷积后的W或H = (input_H - kernel_size + 2 * padding)/stride+1

参考：https://www.zhihu.com/tardis/zm/art/349683405?source_id=1003
学习卷积过程

反卷积（转置卷积）：
N = (Input - 1) * stride + Pout - 2 * Pin + kernel_size
'''


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.isSameChannel = in_channels == out_channels
        self.conv1 = nn.Sequential(
            # kernel_size: 3、stride: 1、padding: 1 时 input_height === out_height
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        # （N, C, H, W）通过conv1变成(N, out_channels, H, W)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if (self.isSameChannel):
            return x + x2
        else:
            shortcut = nn.Conv2d(
                x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
            out = shortcut(x) + x2
            # 1.414是2的平方根, 除以1.414对输出张量进行归一化，从而确保残差连接后的输出张量和原张量具有相似的范围和分布。
            return out / 1.414


class UnetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.model(x)


class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(
                out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)

        x = self.model(x)
        return x


# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio,

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.


class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform):
        self.data = np.load(sfilename)
        self.label = np.load(lfilename)
        print(
            f"data shape: {self.data.shape}, label shape: {self.label.shape}, ")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(self.data[index])
            label = torch.tensor(self.label[index]).to(torch.int64)
        return (image, label)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

# 将采样过程生成gif直观感受


def generateGIF(frames: list, file_name: str):
    print('generate gif begin.')

    def update(frame):
        im.set_array(frame)
        return [im]

    # 创建一个画布和图像对象
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], animated=True)
    # interval是两帧之间毫秒
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=20, blit=True)

    # 保存为GIF文件
    # fps为每秒播放的帧数
    ani.save(file_name, dpi=100, writer=animation.PillowWriter(fps=50))
    plt.close()
    # interval设20, fps设50，总共500帧，则计算需10s
    print('generate gif success.')
