from ddpm import *
from matplotlib import pyplot as plt


class Sampler:
    def __init__(self, model: DDPM_UNet, total: int, rows: int, cols: int, context: torch.Tensor):
        # 尝试加载模型
        if os.path.exists(model_path):
            self.model = model.to(device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=device))
            self.model.eval()
            print("Model loaded.")
        else:
            raise Exception("Err: Model Not Found!")

        # 定义列表存储采样过程中的图像，用于可视化
        self.frames = []
        self.total = total
        self.rows = rows
        self.cols = cols
        self.context = context

    def sample(self, x_t, t: int):
        # 在使用模型预测时，需要使用torch.no_grad禁用梯度计算，大大降低显存占用，避免内存暴涨程序挂掉的问题
        with torch.no_grad():
            time_embedding = torch.full((self.total, ), t).to(device)
            pred_noise = self.model(
                x_t, time_embedding / timesteps, self.context)

            noise = torch.randn(self.total, 1, image_size,
                                image_size).to(device)
            sigma = b_t[t].pow(2) * ((1 - a_bar_t[t].pow(2)) / b_t[t]) / \
                (1 - a_bar_t[t].pow(2))

            mean = 1 / a_t[t] * (x_t - ((1 - a_t[t].pow(2)) /
                                        (1 - a_bar_t[t].pow(2)).sqrt()) * pred_noise)
            if (t % 50 == 0):
                print(f"已采样{timesteps - t}次")

            return mean + noise * sigma

    def tensorToFrame(self, tensor):
        # 需要先把张量移动到cpu，numpy才能处理
        frame = tensor.detach().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))
        frame = np.clip(frame, 0, 1)
        return frame

    def build_image(self, x_t):
        for i in range(self.total):
            img = self.tensorToFrame(x_t[i])
            plt.subplot(self.rows, self.cols, i + 1)
            plt.imshow(img)
        # 绘制
        plt.show()
        # 生成动图
        # generateGIF(self.frames, 'ddpm_steps.gif')

    def run(self):
        x_t = torch.randn(self.total, in_channels,
                          image_size, image_size).to(device)

        t = timesteps - 1
        for i in range(t):
            x_t = self.sample(x_t, t - i)
            # 只把第一张图采样过程动画处理
            frame = self.tensorToFrame(x_t[0])
            self.frames.append(frame)

        self.build_image(x_t)


# 一次采样几张图
total = 4
rows = 1
cols = 4
# 生成的类别
labels = [1, 0, 0, 0, 0]
context = torch.Tensor(labels).to(device, dtype=torch.float32)
# 生成 total 个一样的context
context = context.expand(total, 5)

# 初始一个model给Sampler
sampler = Sampler(model, total, rows, cols, context)

sampler.run()
