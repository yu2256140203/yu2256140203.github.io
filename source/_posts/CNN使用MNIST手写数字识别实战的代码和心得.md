---
title: CNN使用MNIST手写数字识别实战的代码和心得
date: 2021-03-04 14:33:46
categories:
- 学习
- NLP
thumbnail: https://i.loli.net/2020/07/31/rSYOE68HnqgJaU4.png#vwid=1409&vhei=821
---

> CNN(Convolutional Neural Network)卷积神经网络对于MNIST手写数字识别的实战代码和心得

首先是对代码结构思路进行思路图展示，如下：
![CNN进行MNIST手写数字识别][1]
参数和原理剖析：
因为MNIST图片为长和宽相同的28像素，为黑白两色，所以图片的高度为1，为灰度通道。
在传入的时候，我定义的BATCH_SIZE为512，所以具体的输入维度为(512,1,28,28)
我的CNN卷积神经网络的为两层卷积层，两次激活函数，两层池化层，和两层全连接层
卷积核设为5X5，步长Stride = 2（卷积核移动的步长）
填充padding = （kernal_size - stride） /2 (在图像张量周围加两圈0)
1.1经过卷积层 输入通道为1，输出通道为14，其他参数值不变(BATCH_SIZE,1,28,28)
1.2经过激活函数，只将张量中的为负数的值变为0，不改变shape，各维度不变(BATCH_SIZE,14,28,28)
1.3经过最大池化层，将图片缩小，降采样，只取图片的最大值细节，图片长宽维度变为原来的二分之一(BATCH_SIZE,14,14,14)
2.1经过卷积层 输入通道为14，输出通道为28，其他参数值不变(BATCH_SIZE,28,14,14)
2.2经过激活函数，只将张量中的为负数的值变为0，不改变shape，各维度不变(BATCH_SIZE,28,14,14)
2.3经过最大池化层，将图片缩小，降采样，只取图片的最大值细节，图片长宽维度变为原来的二分之一(BATCH_SIZE,28,7,7)
3.利用view函数，将张量拉平，shape变为(BATCH_SIZE,28*7*7)
4.1经过第一层全连接层，将(28*7*7)变为200，高度提纯，一个全连接层将卷积层提取的特征进行线性组合
4.2经过第二层全连接层，将200变为10，针对最后分类的10钟图片，进行十种维度的结果，实现了对输入的数据进行高度的非线性变换的目的
下面是对库的导入
```python
    # 1 加载必要的库
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # 优化器
    import torch.optim as optim
    from torchvision import datasets, transforms
```

对于超参数的定义

    # 2 定义超参数
    BATCH_SIZE = 512  # 每批处理的数据
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否使用GPU还是CPU
    EPOCHS = 10  # 训练数据集的伦次
BATCH_SIZE是每批处理数据的样本数量
对于DEVICE的定义是对于程序运行在CPU还是GPU进行识别，通过torch的CUDA函数
EPOCHS指训练和测试方法运行的次数，运行在一定范围内次数越多能提高正确率

对于图像进行处理
```pyhton
    # 3创建pipeline，对图像做处理（transforms变换）
    pipeline = transforms.Compose([
         transforms.ToTensor(),  # 将图片转换成tensor
         transforms.Normalize((0.1307,), (0.3081,))
    ])
ToTensor将本来图片像素点的形式，转化为张量的形式，利用于计算
normalize正则化，模型出现过拟合时，降低模型复杂度
```
进行数据集的下载和加载
```python
    # 4 下载，加载数据
    from torch.utils.data import DataLoader

    # 下载数据集
    train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)

    test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
    # 加载数据
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # 打乱图片顺序

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
```
构建网络模型，针对于网络模型的构建，我采用了Module和Sequential两种方式
1.Moudle方式
```python
    # 5 构建网络模型
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 14, 5,1,2)  # 卷积函数   1:灰度图片的通道 14：输出通达 5：kernel 
    1:stride 2:padding
            self.conv2 = nn.Conv2d(14, 28, 5,1,2)  # 14:输入通道 20 输出通道 3：kernel
            self.fc1 = nn.Linear(28 * 7 * 7, 200)  # 全连接层,28*7*7:输入通道 200输出通道
            self.fc2 = nn.Linear(200, 10)  # 200:输入通道，10：输出通道
       
       def forward(self, x):
            input_size = x.size(0)  # batch_size
            x = self.conv1(x)  # 卷积操作 输入batch_size*1*28*28，输出：batch_size*14*28*28
            x = F.relu(x)  # 输出：batch_size*14*28*28
            x = F.max_pool2d(x, 2, 2)  # 输入：batch*14*28*28,输出：batch*14*14*14，
            x = self.conv2(x)  # 输入：batch*14*14*14,输出：batch*28*14*14
            x = F.max_pool2d(x, 2, 2)
            x = x.view(input_size, -1)  #28*14*14=1372
            x = self.fc1(x)  # 输入batch*1372 输出batch*200
            x = F.relu(x)  # 保持shape不变
            x = self.fc2(x)  # 输入：batch*200 输出：batch*10
            output = F.log_softmax(x, dim=1)  #
            return output
```
2.Sequential方式
```python
    #  构建网络模型
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 14, 5, 1, 2),  # padding=(kernel_size - stride)/2
                nn.ReLU(),  # (512,4,28,28）
                nn.MaxPool2d(kernel_size=2),
            )  # (14,14,14,)
             self.conv2 = nn.Sequential(
                nn.Conv2d(14, 28, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
             self.fc1 = nn.Linear(28*7*7, 200)  # 全连接层,28*7*7:输入通道 200输出通道
            self.fc2 = nn.Linear(200, 10)  # 200:输入通道，10：输出通道

        def forward(self, x):
            input_size = x.size(0)  # batch_size
            x = self.conv1(x)  # 卷积操作 输入batch_size*1*28*28，输出：batch_size*14*14*14
            x = self.conv2(x)  # 输入：batch*14*14*14,输出：batch*28*7*7
            x = x.view(input_size, -1)  # 拉平，-1自动计算维度，28*7*7=1372
            x = self.fc1(x)  # 输入batch*1372 输出batch*200
            x = F.relu(x)  # 保持shape不变
            x = self.fc2(x)  # 输入：batch*200 输出：batch*10
            output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值
            return output
```
针对于卷积神经网络具体的层和函数的作用理解:
1.卷积层：对图片信息进行抽象化
2.激活函数：激活函数，非线性函数神经网络更好表达,保持shape不变，
3.池化层：池化降采样，将原图缩小，取最大或者取平均池化
4.全连接层：高度提纯，一个全连接层将卷积层提取的特征进行线性组合，第二个“实现了对输入的数据进行高度的非线性变换的目的”。

定义优化器
```python
    # 6 定义模型，优化器
    model = CNN().to(DEVICE)  # 创建模型，部署到设备上
    print(model)
    optimizer = optim.Adam(model.parameters())  # 创建优化器

定义训练方法

    # 7 定义训练方法
    def train_model(model, device, train_loader, optimizer, epoch):  # epoch就是循环的次数
    # 模型训练
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):  # target是标签，可以用label
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target) 
            loss.backward()
            optimizer.step()
            if batch_index % 3000 == 0:
                print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))
```
epoch为循环的次数
optimizer.zero_grad()是对梯度进行初始化
output预测，训练后的结果，data调用的方法是model.forword()函数
loss计算交叉熵损失
loss.backward()反向传播
optimizer.step()参数优化

定义测试方法，测试方法的代码与训练类似，直接在原代码进行标注
```python
    # 8 定义测试方法
    def test_model(model, device, test_loader):
        # 模型验证
        model.eval()
        # 正确率
        correct = 0.0
        # 测试损失
        test_loss = 0.0
        with torch.no_grad():  # 不会计算梯度，也不会反向传播
            for data, target in test_loader:
                # 部署到device上去
                data, target = data.to(device), target.to(device)
                # 测试数据
                output = model(data)
                # 计算测试损失
                test_loss += F.cross_entropy(output, target).item()
                # 找到概率值最大的下标
                pred = output.max(1, keepdim=True)[1]  # 值，索引
                # pred = torch.max(output,dim=1)
                # pred = output.argmax(dim=1)
                # 累计正确率
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            print("Test -- Average loss :{:.4f}, Accuracy : {:.3f}\n".
                  format(test_loss, 100.0 * correct / len(test_loader.dataset)))
```
对于方法的调用

```python
    # 9 调用方法
    for epoch in range(1, EPOCHS + 1):
        train_model(model, DEVICE, train_loader, optimizer, epoch)
        test_model(model, DEVICE, test_loader)
```
运行结果
```python
    CNN(
      (conv1): Sequential(
        (0): Conv2d(1, 14, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(14, 28, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc1): Linear(in_features=1372, out_features=200, bias=True)
      (fc2): Linear(in_features=200, out_features=10, bias=True)
    )
    Train Epoch : 10 	 Loss : 0.014751
    Test -- Average loss :0.0001, Accuracy : 99.040
```
[1]: https://i.loli.net/2020/07/31/rSYOE68HnqgJaU4.png#vwid=1409&vhei=821