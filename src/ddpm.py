

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from utils import *

'''
# 训练过程
repeat
  x_0 ~ q(x_0) # x_0是训练数据中获取的
  t = rand(1, T) # 范围内随机时间
  \epsilon ~ N(0, 1) # 随机噪声采样
  对如下loss函数进行梯度下降：
    MSE(\epsilon - \epsilon_theta)
  网络输入是 x_t 和 时间t

# 采样过程
for t = T,...,1 do
  z~N(0, 1) if t > 1，else z = 0
  x_{t-1} = \miu + \sigma * z
end
retun x_0


'''

device = ('cuda' if torch.cuda.is_available() else "cpu")
# 设置模型保存路径
model_path = './model/ddpm_model5.pth'

lr = 1e-3
batch_size = 512
in_channels = 3
n_feat = 64  # 64 hidden dimension feature
context_feat = 5  # 标签size
image_size = 16
timesteps = 500

beta1 = 1e-4
betaT = 0.02

# beta_t
b_t = torch.linspace(beta1, betaT, timesteps).sqrt()
# alpha_t
a_t = (1 - b_t ** 2).sqrt()
# alpha_t_bar 利用exp(log(x_1*x_2*...*x_n)) == xi累积求累积, 转换为 torch.cumsum 求累加
a_bar_t = torch.cumsum(a_t.log(), dim=0).exp().to(device)
a_bar_t[0] = 1

# a_bar_t: [timesteps], t: [timesteps]
transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])


def pre_input(x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    # 这里是pytorch高级索引功能，使用None扩充出一个维度
    # a_bar_t: [500], t: [64], a_bar_t[t, None, None, None] => [64, 1, 1, 1]

    return a_bar_t[t, None, None, None] * x + (1 - a_bar_t[t, None, None, None] ** 2).sqrt() * noise


class DDPM_UNet(nn.Module):
    def __init__(self, in_channels: int, n_feat: int, height: int, context_feat: int):
        super().__init__()
        self.n_feat = n_feat
        self.height = height
        self.in_channels = in_channels
        self.ctx_feat = context_feat
        # [batch_size, in_channels, H, W] -> [batch_size, n_feat, H, W]
        self.init_conv = ResidualConvBlock(in_channels, n_feat)
        self.down1 = UnetDownBlock(n_feat, n_feat)
        self.down2 = UnetDownBlock(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Initialize the up-sampling path of the U-Net with three levels
        # 15 // 2 == 7 向下取整数除法
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat,
                               self.height//4, self.height//4),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUpBlock(4 * n_feat, n_feat)
        self.up2 = UnetUpBlock(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(self.ctx_feat, 2*n_feat)
        self.contextembed2 = EmbedFC(self.ctx_feat, 1*n_feat)

    def forward(self, x, t, ctx):
        x = self.init_conv(x)  # [64, 64, 16, 16]
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [64, 64, 16, 16] -> [64, 64, 8, 8]
        down2 = self.down2(down1)  # [64, 64, 8, 8] -> [64, 128, 4, 4]

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)  # [64, 128, 4, 4] -> [64, 128, 1, 1]

        # (batch, 2*n_feat, 1,1)
        cemb1 = self.contextembed1(ctx).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(ctx).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)  # [64, 128, 4, 4]
        a = cemb1 * up1 + temb1
        up2 = self.up1(a, down2)  # [64, 128, 4, 4]
        b = cemb2 * up2 + temb2
        up3 = self.up2(b, down1)  # [64, 64, 16, 16]
        out = self.out(torch.cat((up3, x), 1))  # [64, 3, 16, 16]
        return out


def loss_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """用l2距离为损失函数，比mse更好
    """
    # return torch.sum((y_true - y_pred).pow(2), dim=[1, 2, 3]).sum()
    return nn.MSELoss()(y_true, y_pred)


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model.to(device)

        self.loss_fn = loss_fn

        dataset = CustomDataset('./custom_dataset/sprites_1788_16x16.npy',
                                './custom_dataset/sprite_labels_nc_1788_16x16.npy', transform)

        self.dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)

        # 动态调整学习率，每20个epoch学习率乘以0.9
        self.schedular = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.9)

        # 尝试加载模型
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=device))
            print("Model loaded.")

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        # 也可以用 next(iter()) 迭代器方式获取一个batch的数据
        # train_data, label = next(iter(self.dataloader))
        # 训练一般用tqdm进度条库，可以直观看到进度条

        pbar = tqdm(self.dataloader)
        # label: 类似 [0, 0, 0, 0, 1]，属于one hot编码的数据
        for x, label in pbar:
            loss_total = 0.0

            t = torch.randint(1, timesteps, (x.shape[0],)).to(device)
            noise = torch.randn_like(x).to(device)

            # 通过标签构建context控制输出
            # 参考官方代码，需要进行随机掩码策略，10%的概率输出全零结果[0,0,0,0,0]，90%输出原值
            # context = zero_out_tensor(label, x).to(device)

            c = label.to(device, dtype=torch.float32)
            # 加掩码之后训练效果差，去掉掩码，直接分类
            # context_mask = torch.bernoulli(
            #     torch.zeros(c.shape[0]) + 0.9).to(device)
            # c = c * context_mask.unsqueeze(-1)

            # Concatenate the time embedding with the data tensor
            x = x.to(device)
            pred_x = pre_input(x, t, noise)
            pred_noise = self.model(pred_x, t / timesteps, c)
            loss = self.loss_fn(pred_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            pbar.set_description("Loss: %f" % loss_total)

        pbar.close()
        self.schedular.step()
        return loss_total

    # 添加保存模型的方法
    def save_model(self):
        torch.save(self.model.state_dict(), model_path)
        print("Model saved.")


model = DDPM_UNet(in_channels, n_feat, image_size, context_feat)
trainer = Trainer(model)
