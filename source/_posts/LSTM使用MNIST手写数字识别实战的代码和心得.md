---
title: LSTM使用MNIST手写数字识别实战的代码和心得
date: 2021-03-04 14:29:49
categories:
- 学习
- NLP
thumbnail: https://i.loli.net/2020/07/31/MvYaBCwx1Aont9F.png#vwid=1397&vhei=698
---

> RNN的架构除了RNN类中的模型不同，其他的构架与CNN类似，如果还没有阅读过CNN文章的可以点击下方链接进入：
> [CNN使用MNIST手写数字识别实战的代码和心得][1]
> LSTM(Long Short-Term Memory长短时记忆网络)虽然在MNIST手写数字识别方面不擅长，但是也可以进行使用，效果比CNN略显逊色

对LSTM使用MNIST手写数字识别的思路图
![LSTM进行MNIST手写数字识别][2]
LSTM是在RNN的主线基础上增加了支线，增加了三个门，输入门，输出门和忘记门。
避免了可能因为加权问题，使程序忘记之前的内容，梯度弥散或者梯度爆炸。
batch_size在这里选取的是100，选择了一个隐藏层和128的神经元，对LSTM结构进行部署，
MNIST长宽为28，选取一行28作为一份数据传入input_size,RNN是按照时间序列进行传值，batch_size为100，也就是在每次传入的数据为（128，28）
进入隐藏层后，out结果张量的shape为（[100, 28, 128]）
在out[:, -1, :]时间序列中取得最后一次的输出，得到（[100,  128]）
再进入全连接层后将hidden_size的128变为所需要的输出的10种图片的维度（[100,  10]）

对超参数的定义
    #定义超参数
    input_size = 28
    time_step = 28# 时间序列
    Layers = 1# 隐藏单元的个数
    hidden_size = 128# 每个隐藏单元中神经元个数
    classes = 10
    batch_size = 100
    EPOCHS = 10
    learning_rate = 0.01 #学习率
RNN对于数据的读取有别于CNN,按照时间来读取，在这里可以将input_size看作是图片的长，而time_step看作宽的长度。
```python
    #Long Short-Term Memory(长短时记忆网络)
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, Layers, classes):
            super(RNN, self).__init__()
            self.Layers = Layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, Layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, classes)
        def forward(self, x):
            # 设置初始隐藏状态和单元格状态
            h0 = torch.zeros(self.Layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.Layers, x.size(0), self.hidden_size).to(device)
            # out张量的shape(batch_size, time_step, hidden_size)
            out, _ = self.lstm(x, (h0, c0))#torch.Size([100, 28, 128])
            #out[:, -1, :].shape torch.Size([100, 128])
            # 只得到时间顺序点的最后一步
            out = self.fc(out[:, -1, :])#torch.Size([100, 10])
            return out
```
运行结果：
```python
    RNN(
      (lstm): LSTM(28, 128, batch_first=True)
      (fc): Linear(in_features=128, out_features=10, bias=True)
    )
    Epoch [10/10],  Loss: 0.0115
    Test Accuracy to test: 98.07 %
```
[1]: https://void.fly-kiss.xyz/index.php/archives/7/
[2]: https://i.loli.net/2020/07/31/MvYaBCwx1Aont9F.png#vwid=1397&vhei=698