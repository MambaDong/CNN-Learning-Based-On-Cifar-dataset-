import torch as t
import torchvision as tv
from torchvision import transforms as transforms
from torchvision.transforms import ToPILImage
show=ToPILImage()
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#实现数据的加载和预处理
# torchvision输出的是PILImage，值的范围是[0, 1].
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
transform = transforms.Compose([transforms.RandomCrop(32,padding=2),                                #将图像周围填充2圈0，然后随机剪裁为32*32
                                transforms.RandomHorizontalFlip(),                                  #图像一半的概率翻转
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

# trainset 将cifar-10-batches-py文件夹中的全部数据加载到内存中（50000张图片作为训练数据）
trainset = tv.datasets.CIFAR10(root=r'D:\PyCharm 2018.2.4\untitled4\cifar-dataset', train=True, download=False, transform=transform)

# 将训练集的50000张图片划分成10000份，每份5张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = t.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)

#loadset load data
testset=tv.datasets.CIFAR10(root=r'D:\PyCharm 2018.2.4\untitled4\cifar-dataset', train=False, download=False, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')           #classification label



device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

#定义我的卷积神经网络
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()  #继承

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)  # 卷积层的搭建
        nn.BatchNorm2d(32)                                                                          # 将卷积后的图像归一化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)                                                             # 池化层的搭建

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=5,stride=1,padding=1)
        nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)                                                             #池化层的搭建

        self.fc1=nn.Linear(256*5*5,120)                                                             #全连接层的搭建
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)


    def forward(self, x):
        x = F.relu(self.conv1(x))                                                                   #激活函数采用relu函数
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x =  self.pool2(F.relu(self.conv4(x)))
        x = x.view(-1, 256*5*5)                                                                     #reshape 张量 -1表示一个不确定的值
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

mynet=CNN()
print(mynet)
mynet.to(device)

#定义损失函数和优化器
from torch import optim
criterion=nn.CrossEntropyLoss()                                 #交叉熵损失函数
optimizer=optim.SGD(mynet.parameters(),lr=0.001,momentum=0.9)   #定义优化器

X=[]                                                            #X列表
Y=[]                                                            #Y列表
n=0
#训练卷积神经网络
import time
start_time=time.time()
for epoch in range(50):                                         #epoch for the amount of training
    loss = 0.0
    for i, data in enumerate(trainloader, 0):                  # 实现依次读取,取索引和对应的值
        inputs, labels = data

        # 数据输入
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()                                  # 梯度清零
        outputs = mynet(inputs)
        lossing = criterion(outputs, labels)
        lossing.backward()
        optimizer.step()                                       # 更新参数
        loss+=lossing.data[0]
        if i % 2000 == 1999:
            n+=1
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, loss / 1999))
            X.append(n)
            Y.append(loss/1999)
            loss = 0.00

    end_time = time.time()
    print("经历的时间:", end_time - start_time)
    print('第',epoch+1,'次训练结束！')

#将训练结果图形化展示出来
#绘制损失曲线和正确率曲线图像

plt.plot(X, Y, c='red')
plt.xlabel('Iterations')
plt.ylabel('cost')
plt.title('Loss By Iterations')
plt.show()



#测试模块
correct1 = 0  # 预测正确的图片数
total1 = 0  # 总共的图片数
for data in trainloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = mynet(images)
    _, predicted1 = t.max(outputs.data, 1)
    total1 += labels.size(0)
    correct1 += (predicted1 == labels).sum().item()

print("训练集中的准确率为：%d %%" % (100 * correct1 / total1))
#计算准确率
correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = mynet(images)
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("测试集中的准确率为：%d %%" % (100 * correct / total))

#卷积核实现可视化
def kel_vision(x,i,j):
    fig=plt.figure()
    for idx,filt in enumerate(x.weight.data):
        plt.subplot(i,j,idx + 1)
        plt.imshow(filt[0, :, : ],cmap=plt.cm.gray_r)
        plt.axis("off")
    fig.show()

#调用可视化函数
kel_vision(mynet.conv1,4,8)
kel_vision(mynet.conv2,8,8)
kel_vision(mynet.conv3,8,16)
kel_vision(mynet.conv4,16,16)