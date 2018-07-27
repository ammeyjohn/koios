---
title: Visual Studio 2017 使用 Chrome 调试浏览器自动关闭的问题
date: 2018-07-27 11:27:53
tags: [issue,问题,visual studio 2017,chrome,闪退]
categories: ISSUE
---

最近使用Visual Studio 2017开发WebApi项目的时候碰到个奇怪的问题。在Visual Studio 2017中启动IIS Express调试后，自动打开了Chrome浏览器，并已经导航到了默认页面。但是无论是在弹出的浏览器地址栏还是之前开着的Chrome浏览器的地址栏中输入任何内容，自动打开的Chrome浏览器窗口自动关闭，多次尝试都是如此。
在Google中搜索后发现了解决方法，具体解决方法如下：
在Visual Studio 2017中，依次打开`工具=>选项=>调试=>常规`，找到“对ASP.NET启用Javascript调试(Chrome、Edge和IE)”项并勾选，如下图：
![issue-vs2017-chrome-debug](/images/issue-vs2017-chrome-debug.png)
根据上面步骤操作后，再次启动调试，在浏览器地址栏中输入地址正常，浏览器不会自动关闭。