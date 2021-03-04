---
title: 学习vuepress搭建博客
date: 2021-03-04 14:35:15
categories:
- 学习
- 博客
- VUE
thumbnail: https://www.w3cplus.com/sites/default/files/blogs/2017/1709/vuejs-2.jpg
---

@[TOC](新建文件夹myblog)
ps：简单的方法在文件夹打开控制台---按住shift+右键,点击打开powershell窗口（这相当于cmd的高级版）
![如果在文件夹打开powershell](https://img-blog.csdnimg.cn/20210122155900575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODkzNTkx,size_16,color_FFFFFF,t_70)

在文件夹中，进行初始化

```
PS C:\Users\Lenovo\Desktop\myblog> npm init -y
```

![npm init -y](https://img-blog.csdnimg.cn/20210122155150678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODkzNTkx,size_16,color_FFFFFF,t_70)
@[TOC](打开vscode)
在powershell里使用命令来打开vscode

```
PS C:\Users\Lenovo\Desktop\myblog> code .
```
进行package.json文件内容进行编辑

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210122173343663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODkzNTkx,size_16,color_FFFFFF,t_70)
在输入完后，可以使用命令`npm run dev`来检验是否package.json编写成功。