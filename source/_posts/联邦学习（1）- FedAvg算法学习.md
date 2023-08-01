---
title: 联邦学习（1）- FedAvg算法学习
date: 2023-07-31 14:35:15
categories:
- 学习
- 博客
- VUE
---

# 一、何为FedAvg算法

> FedAvg是一种常用的联邦学习算法，它通过加权平均来聚合模型参数。FedAvg的基本思想是将本地模型的参数上传到服务器，服务器计算所有模型参数的平均值，然后将这个平均值广播回所有本地设备。这个过程可以迭代多次，直到收敛。


## FedAvg联邦平均算法的优势：
1.低通信开销：由于只需要上传本地模型参数，因此通信开销较低。
2.支持异质性数据：由于本地设备可以使用不同的数据集，因此FedAvg可以处理异质性数据。
3.泛化性强：FedAvg算法通过全局模型聚合，利用所有设备上的本地数据训练全局模型，从而提高了模型的精度和泛化性能。
## FedAvg联邦平均算法的不足：
1.需要协调：由于需要协调多个本地设备的计算，因此FedAvg需要一个中心化的协调器来执行此任务。这可能会导致性能瓶颈或单点故障。
2.数据不平衡问题：在FedAvg算法中，每个设备上传的模型参数的权重是根据设备上的本地数据量大小进行赋值的。这种方式可能会导致数据不平衡的问题，即数据量较小的设备对全局模型的贡献较小，从而影响模型的泛化性能。

## FedAvg 联邦平均算法：
1.服务器初始化全局模型参数 $w_0$；
2.所有本地设备随机选择一部分数据集，并在本地计算本地模型参数 $w_i$；
3.所有本地设备上传本地模型参数 $w_i$ 到服务器；
4.服务器计算所有本地模型参数的加权平均值 $\bar{w}$，并广播到所有本地设备；
5.所有本地设备采用 $\bar{w}$ 作为本地模型参数的初始值，重复步骤2~4，直到全局模型收敛。

# PaddlePaddle飞桨框架实现FedAvg样例

> 该学习项目所在地址：https://aistudio.baidu.com/aistudio/projectdetail/4167786

## 1.数据加载 使用Mnist数据集
```python
 mnist_data_train=np.load('data/data2489/train_mnist.npy')
mnist_data_test=np.load('data/data2489/test_mnist.npy')
print('There are {} images for training'.format(len(mnist_data_train)))
print('There are {} images for testing'.format(len(mnist_data_test)))
#数据和标签分离（便于后续处理）
Label=[int(i[0]) for i in mnist_data_train]
Data=[i[1:] for i in mnist_data_train]
```
加载mnist数据集，并打印出当前的测试集和训练集的量
## 2.模型构建
```python
class CNN(nn.Layer):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2D(1,32,5)
        self.relu = nn.ReLU()
        self.pool1=nn.MaxPool2D(kernel_size=2,stride=2)
        self.conv2=nn.Conv2D(32,64,5)
        self.pool2=nn.MaxPool2D(kernel_size=2,stride=2)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,10)
        # self.softmax = nn.Softmax()
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x=paddle.reshape(x,[-1,1024])
        x = self.relu(self.fc1(x))
        y = self.fc2(x)
        return y
```

模型构建，这是用的CNN模型，当然也可以用其他的方法。

## 数据采样
### 1.均匀采样

```python
# 均匀采样，分配到各个client的数据集都是IID且数量相等的
def IID(dataset, clients):
  num_items_per_client = int(len(dataset)/clients)
  client_dict = {}
  image_idxs = [i for i in range(len(dataset))]
  for i in range(clients):
    client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False)) # 为每个client随机选取数据
    image_idxs = list(set(image_idxs) - client_dict[i]) # 将已经选取过的数据去除
    client_dict[i] = list(client_dict[i])

  return client_dict
```

### 2.非均匀采样

```python
# 非均匀采样，同时各个client上的数据分布和数量都不同
def NonIID(dataset, clients, total_shards, shards_size, num_shards_per_client):
  shard_idxs = [i for i in range(total_shards)]
  client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
  idxs = np.arange(len(dataset))
  data_labels = Label

  label_idxs = np.vstack((idxs, data_labels)) # 将标签和数据ID堆叠
  label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
  idxs = label_idxs[0,:]

  for i in range(clients):
    rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False)) 
    shard_idxs = list(set(shard_idxs) - rand_set)

    for rand in rand_set:
      client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0) # 拼接
  
  return client_dict
```

```python
class MNISTDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        image=np.array(self.data[idx]).astype('float32')
        image=np.reshape(image,[1,28,28])
        label=np.array(self.label[idx]).astype('int64')
        return image, label

    def __len__(self):
        return len(self.label)
```
## 模型训练

```python
class ClientUpdate(object):
    def __init__(self, data, label, batch_size, learning_rate, epochs):
        dataset = MNISTDataset(data,label)
        self.train_loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True)
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def train(self, model):
        optimizer=paddle.optimizer.SGD(learning_rate=self.learning_rate,parameters=model.parameters())
        criterion = nn.CrossEntropyLoss(reduction='mean')
        model.train()
        e_loss = []
        for epoch in range(1,self.epochs+1):
            train_loss = []
            for image,label in self.train_loader:
                # image=paddle.to_tensor(image)
                # label=paddle.to_tensor(label.reshape([label.shape[0],1]))
                output=model(image)
                loss= criterion(output,label)
                # print(loss)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                train_loss.append(loss.numpy()[0])
            t_loss=sum(train_loss)/len(train_loss)
            e_loss.append(t_loss)
        total_loss=sum(e_loss)/len(e_loss)
        return model.state_dict(), total_loss
```

```python
train_x = np.array(Data)
train_y = np.array(Label)
```

```python
BATCH_SIZE = 32
# 通信轮数
rounds = 100
# client比例
C = 0.1
# clients数量
K = 100
# 每次通信在本地训练的epoch
E = 5
# batch size
batch_size = 10
# 学习率
lr=0.001
# 数据切分
iid_dict = IID(mnist_data_train, 100)
```

```python
def training(model, rounds, batch_size, lr, ds,L, data_dict, C, K, E, plt_title, plt_color):
    global_weights = model.state_dict()
    train_loss = []
    start = time.time()
    # clients与server之间通信
    for curr_round in range(1, rounds+1):
        w, local_loss = [], []
        m = max(int(C*K), 1) # 随机选取参与更新的clients
        S_t = np.random.choice(range(K), m, replace=False)
        for k in S_t:
            # print(data_dict[k])
            sub_data = ds[data_dict[k]]
            sub_y = L[data_dict[k]]
            local_update = ClientUpdate(sub_data,sub_y, batch_size=batch_size, learning_rate=lr, epochs=E)
            weights, loss = local_update.train(model)
            w.append(weights)
            local_loss.append(loss)

        # 更新global weights
        weights_avg = w[0]
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                # weights_avg[k] += (num[i]/sum(num))*w[i][k]
                weights_avg[k]=weights_avg[k]+w[i][k]   
            weights_avg[k]=weights_avg[k]/len(w)
            global_weights[k].set_value(weights_avg[k])
        # global_weights = weights_avg
        # print(global_weights)
    #模型加载最新的参数
        model.load_dict(global_weights)

        loss_avg = sum(local_loss) / len(local_loss)
        if curr_round % 10 == 0:
            print('Round: {}... \tAverage Loss: {}'.format(curr_round, np.round(loss_avg, 5)))
        train_loss.append(loss_avg)

    end = time.time()
    fig, ax = plt.subplots()
    x_axis = np.arange(1, rounds+1)
    y_axis = np.array(train_loss)
    ax.plot(x_axis, y_axis, 'tab:'+plt_color)

    ax.set(xlabel='Number of Rounds', ylabel='Train Loss',title=plt_title)
    ax.grid()
    fig.savefig(plt_title+'.jpg', format='jpg')
    print("Training Done!")
    print("Total time taken to Train: {}".format(end-start))
  
    return model.state_dict()

#导入模型
mnist_cnn = CNN()
mnist_cnn_iid_trained = training(mnist_cnn, rounds, batch_size, lr, train_x,train_y, iid_dict, C, K, E, "MNIST CNN on IID Dataset", "orange")
```