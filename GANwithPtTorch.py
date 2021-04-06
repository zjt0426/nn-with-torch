import numpy as np
import torch
import matplotlib.pylab as plt
import torch.nn as nn


LR = 0.0001
BATCH_SIZE = 64
DATA_SIZE = 16
IDEA = 5
X = np.linspace(0, 2 * np.pi, DATA_SIZE)  # DATA_SIZE *

def p_data(x):
    f = np.zeros((BATCH_SIZE, DATA_SIZE))
    for j in range(BATCH_SIZE):
        f[j] = np.sin(x)
    return f

# print(p_data(X))
# a = p_data(X)[:, 1]
# print(a)
# plt.plot(a)
# plt.show()


# the input of G is same to the second dim of the data
G = nn.Sequential(
    nn.Linear(IDEA, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, DATA_SIZE),
    # nn.Sigmoid()
)


# the input of the D is same to the out of the G
D = nn.Sequential(
    nn.Linear(DATA_SIZE, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
D_optimizer = torch.optim.Adam(D.parameters(), lr=LR)

torch.autograd.set_detect_anomaly(True)

plt.ion()
# start trainning
for i in range(10000):
    real = torch.tensor(p_data(X)).float()   # 这里给一个BATCH_SIZE是列；行表示数据的变化，和X的维度一样。
    # print(real.data.numpy())
    idea = torch.randn(BATCH_SIZE, IDEA)    # 噪声的第二维度和生成器G的输入相等
    fake = G(idea)

    prob_fake = D(fake)
    G_loss = torch.mean(torch.log(torch.tensor(1) - prob_fake))
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()



    prob_real = D(real)
    prob_fake = D(fake.detach())     # 由于1.4版本之前D_optimizer.step()原地修改了鉴别器的参数，这些参数是计算发生器的
                                     # 梯度所必需的。而1.5优化了

    D_loss = -torch.mean((torch.log(prob_real) + torch.log(torch.tensor(1) - prob_fake)))
    D_optimizer.zero_grad()
    D_loss.backward(retain_graph=True)
    D_optimizer.step()



    if i % 100 == 0:
        print(prob_real.mean())
        print(prob_fake.mean())
        print('---------------------------')

    if torch.abs(prob_real.mean() - 0.5) <= 1.e-6:
        break

    if i % 50 == 0:
        plt.cla()
        plt.plot(X, fake.data.numpy()[0], c='red', lw=3, label='Generate painting')
        plt.plot(X, real.data.numpy()[1, :], c='black', lw=1, label='real painting')
        plt.text(1, .5, 'the prob of Generated painting is real = %.2f' % prob_fake.data.numpy().mean())
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='best', fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()



