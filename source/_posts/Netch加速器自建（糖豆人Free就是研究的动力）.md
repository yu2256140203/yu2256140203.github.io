---
title: Netch游戏加速器自建（糖豆人Free就是研究的动力）
date: 2022-06-29 18:35:15
categories:
- 学习
- 博客
- Netch
thumbnail: https://img-blog.csdnimg.cn/1511544d17cc4239b7fd5da449708e9f.png
---

## 兄弟们 Epic糖豆人Free了
![在这里插入图片描述](https://img-blog.csdnimg.cn/1511544d17cc4239b7fd5da449708e9f.png)
那既然免费的话，我肯定是要尝试一下的哇，但没开加速器的情况下，总是右下角显示一个白色的WIFI标志，估计是小卡，还有的时候显示一个WIFI,一个红色的斜杠，不用多说了，卡爆啦。

## 自建Netch加速器 
身为计科人，遇到延迟高，真的难绷，于是乎，搜了下加速器，哇，价格高的离谱，没办法，坑的都是游戏党的钱，原理上讲，就是服务器转发流量实现的游戏加速，本身游戏加速就消耗流量很少，更何况，游戏加速只追求延迟，腾讯和阿里的香港轻量就很符合条件。
### 1.搭建节点,什么协议都是可以的

> 这里就不多介绍了，自行学习。

### 2.Netch
Netch算是sstap的高阶平替，比sstap的操作步骤更加简单，称得上是一键操作。
下面是Netch的下载网址（选择的是1.9.2 ， 最新版1.9.7-亲测最新版不可用）网址是github的，下载需要干啥，懂得都懂

```url
https://github.com/netchx/netch/releases/download/1.9.2/Netch.7z
```
### 3.将节点导出到Netch中
个人比较喜欢用url直接导入
![在这里插入图片描述](https://img-blog.csdnimg.cn/f6ba269aa77449f982c7eef4fbe66af4.png)
导入成功后，就可以看到对应的节点名称和延迟
![在这里插入图片描述](https://img-blog.csdnimg.cn/371c984c366f4f67b6c4629ee3571ab8.png)
例如上图为47ms延迟，糖豆人可以说的上是丝滑了。
然后模式选择 Bypass LAN 或者 Bypass LAN and China
第一个是全局模式，适合流量大亨
第二个是绕过国内ip，适合小流量服务器，避免快速用空

### 点击启动，剩下的就打开epic，打开糖豆人，快乐就完事了！