---
title: '学习Onindex的搭建[国际Onedrive]'
date: 2021-03-22 08:12:41
tags:
  - 网盘
author: YuJerry
top: false
categories:
  - 网盘
thumbnail: https://pic4.zhimg.com/80/v2-3de51a56e8c463a4fa496709987a2f72_720w.png
---

# 什么是网盘

网盘很常见，百度，腾讯，微软，亚马逊，甚至正在公测的阿里都是在为客户提供储存用户资料，提供较大空间来供客户保存生活中的一些生产和学习资料。

但由于提供资源的空间较小，也如某盘的带宽流氓限制，导致我们使用网盘的体验并不是很好，再往深处说，我们甚至可以使用网盘来当作我们图片的外链使用，使我们的网站可以利用网盘当作图床，音频床和视频床，然而这些资料都是存在云端，不会影响到我们的服务器内存，很Nice。✨🏆

# 搭建Oneindex的方法

Oneindex是什么？

Oneindex是利用Onedrive，Google Drive等云盘当作储存空间，利用微软和谷歌的强大带宽来实现搭建私人的网盘空间，可以实现文件预览和共享的一种个人网站。

当然随着Serverless Frame框架的不断发展，我们应用的创建和使用有了越来越多不同的方式，对于网盘搭建感兴趣的，可以去看一看Oneindex，OneManager，可道云等多种私人网盘的相关内容。

## 服务器方式搭建

利用服务器搭建网站，是个人站长最基本的技能之一，不需要过于阐述。

随着Oneindex开发者的删库跑路，我们在网上找到一个最新的Onendex

Github网站如下：

`https://github.com/mengxiangke/oneindex-3/releases`

具体的搭建方法如下:

`https://github.com/xinb/Oneindex`

`https://www.shanyemangfu.com/oneindex.html`



## 无服务器方式搭建

没有服务器，也就意味着不需要担心服务器没钱，然后一夜之间数据关停，高额的续费金额让人忘却止步（没错，俺懒得备份）🌦

那就不得不说腾讯云 SCF（云函数），身为Serverless的 一部分，曾经免费向客户使用，但随着时间的推移，到了2021年了，呜呜呜，SCF也开始收费了~~爷清洁~~🤣

还好云函数每个月有100万次的免费额度，仅仅是我们个人使用的话是绝对够用的。

下面是关于SCF的几位站长的搭建教程：

`https://www.nbmao.com/archives/4076`

`https://www.mad-coding.cn/2019/12/02/使用腾讯SCF-onedrive搭建5T个人网盘/`

`https://blog.csdn.net/weixin_42409476/article/details/106522893`

PS:这是OneManager的搭建方法，可以体验一下：

`https://github.com/qkqpttgf/OneManager-php`

