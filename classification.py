import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


trainset = torchvision.datasets.CIFAR10(root=r'D:\dessktop\Python\data\cifar-10-python',train=True, transform=transform, download=False)
trainloader = data.DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0)
# print(len(trainloader))   # 注意这时候trainloader长度变为总数量/batch_size大小
# for batch_idx, (inputs, targets) in enumerate(trainloader):
#     print(inputs)
#     print(targets.size(0))
#     print(targets)
# 这里inputs的是([N, C, H, W]), target维度是([N])
# 这里输入是一幅RGB通道图像(经过处理之后的矩阵）四维， 目标是一个batch_size长度的标签。
testset = torchvision.datasets.CIFAR10(root=r'D:\dessktop\Python\data\cifar-10-python', train=False, transform=transform, download=False)
testloader = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


class ZNet(nn.Module):
    def __init__(self):
        super(ZNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),    # 这里使用全连接的时候是先前特征图h*w*c的个数
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# in_size = torch.ones((64, 3, 32, 32))
net = ZNet()
# print(net)
# output = net(in_size)


# writer = SummaryWriter("logs")
# writer.add_graph(net, in_size)  # 计算图
# writer.close()

# from torchsummary import summary
# summary(net, input_size=(3, 32, 32))

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader): # 这里一次读取图片的数量和batch_size维度一样。
        outputs = net(torch.squeeze(inputs, 1))
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()   # epoch_total_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)    # 总数量
        correct += predicted.eq(targets).sum().item()

        print(batch_idx+1, '/', len(trainloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # 结构为一个epoch 里的批次，epoch序号，loss为每一个批次总的loss累加/训练的批次。Acc为累加批次中全部正确的数量/当前批次总的数量。
    writer = SummaryWriter('runs')
    writer.add_scalar('Train/Acc', 100.*correct/total, epoch)
    writer.add_scalar('Train/Loss', train_loss/(len(trainloader)+1), epoch)




def test(epoch):
    global best_acc
    net.eval()
    total = 0
    test_loss = 0
    correct = 0
    while torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx+1, '/', len(testloader), 'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writer = SummaryWriter('testruns')
        writer.add_scalar('Test/Acc', 100.*correct/total, epoch)
        writer.add_scalar('Test/loss', test_loss/(len(testloader)+1), epoch)


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)

    torch.save(net.state_dict(), 'znetcifar10.pkl')

    net.load_state_dict(torch.load('D:\dessktop\Python\daliy_demo\znetcifar10.pkl'))
    print('begin test')
    for epoch in range(5):
        test(epoch)

