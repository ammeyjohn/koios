---
title: Microservice - Consul
tags: [Microservice,Consul,微服务]
categories: 微服务
date: 2018-7-26 21:19:28
---

# Consul - [https://www.consul.io/](https://www.consul.io/)

## 运行Consul

### 运行开发环境

```sh
consul agent -dev
```

+ `-dev` - 该节点的启动不能用于生产环境，因为该模式下不会持久化任何状态，该启动模式仅仅是为了快速便捷的启动单节点consul

```sh
consul anget -dev -client 192.168.0.100
```
使用`-dev`参数启动的consul服务，默认已经开启了ui界面，但是只能使用 http://localhost:8500 访问，增加`-client`参数后，可以使用指定的IP地址访问到ui界面。

### 以Server方式运行

```sh
consul agent -server -bootstrap-expect 1 -data-dir ./data -node=n0 -bind=192.168.0.100 -datacenter=dc1 -ui -client=192.168.0.100
```

+ `-server` - 指定节点为server；
+ `-bootstrap-expect` - 该命令通知consul server我们现在准备加入的server节点个数，该参数是为了延迟日志复制的启动直到我们指定数量的server节点成功的加入后启动；
+ `-data-dir` - 指定agent储存状态的数据目录；
+ `-node` - 指定节点在集群中的名称，该名称在集群中必须是唯一的（默认采用机器的host）；
+ `-bind` - 指明节点的IP地址；
+ `-datacenter` - 指定机器加入到哪一个数据中心；
+ `-ui` - 启动内建界面，server节点启动后默认不显示内建界面；
+ `-client` - 指定节点为client，指定客户端接口的绑定地址，包括：HTTP、DNS、RP；

```sh
consul agent -config-dir D:\Microservice\consul\conf
```

+ `-config-dir` - 指定service的配置文件和检查定义所在的位置

```json
{
    "encrypt": "U+ff6ZZlI7Zm4w2oHWVCSg==",
    "bootstrap": true,
    "bootstrap_expect": 1,
    "server": true,
    "datacenter": "RD1",
    "data_dir": "D:\\Microservice\\consul\\data\\",
    "ui": true, 
    "bind_addr": "192.168.0.100",
    "client_addr": "192.168.0.100",
    "log_level": "Trace"
}
```
配置文件的详细说明可以参考[官方文档 - Configuration](https://www.consul.io/docs/agent/options.html)

## 将Consul安装成服务

在Windows Server的生产环境中，我们需要将Consul安装成服务

```sh
sc.exe create "Consul" binPath= "D:\Microservice\consul\consul.exe agent -config-dir \"D:\Microservice\consul\conf\"" start= auto
```

**这里有2个需要注意的地方:**
**1. `binPath=`，`start=`和参数之间有一个空格**
**2. `binPath`的参数中如果带有空格，需要用`"`将参数包括起来，如果参数中还有双引号，可以用`\`字符进行转义**

具体方法可以参考[官方文档 - Running Consul run as a service on Windows](https://www.consul.io/docs/guides/windows-guide.html)

> 参考网站
[Consul 简介、安装、常用命令的使用](https://blog.csdn.net/u010046908/article/details/61916389)
