from ddpm import *

# 训练epoch数量
epochs = 600


def run(epochs: int):
    x = range(0, epochs)
    y = list(range(0, epochs))
    for epoch in range(epochs):
        loss = trainer.train()
        print(f"epoch: {epoch:>7d}, loss: {loss:>7f}")
        y[epoch] = loss
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            trainer.save_model()
    # 结束时保存一次
    trainer.save_model()

    # 绘制趋势图
    plt.subplot(2, 1, 1)
    plt.plot(x, y, linewidth=0.5)
    plt.show()
    # plt.savefig('趋势图.jpg')
    print("DONE!")


# 训练epoch数量
run(epochs)
