# YOLOV7_for_PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLO算法作为one-stage目标检测算法最典型的代表，其基于深度神经网络进行对象的识别和定位，运行速度很快，可以用于实时系统。YOLOV7是目前YOLO系列最先进的算法，在准确率和速度上超越了以往的YOLO系列。

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/yolov7/tree/master
  commit_id=072f76c72c641c7a1ee482e39f604f6f8ef7ee92
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。



# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动  | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```
  pillow建议安装较新版本，与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision 
  建议：Pillow版本是9.1.0 torchvision版本是0.6.0。


## 准备数据集


   用户自行获取coco数据集，包含images图片和annotations文件。其中images图片和annotations文件从[coco官网](https://cocodataset.org/#download)获取，另外还需要labels图片，用户可以从[google drive](https://drive.google.com/uc?export=download&id=1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L)中获取。将获取后的数据集解压放置服务器的任意目录下(建议放到源码包根目录XXX/coco/下)。

  数据集目录结构如下所示：

```
    coco
       |-- annotations
       |-- images
          |-- train2017
          |-- val2017   
       |-- labels
          |-- train2017
          |-- val2017
```	  

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 1p精度    
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 1p性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8p精度    
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8p性能

     ```

   - NPU 多机多卡训练指令
   
     启动多机多卡训练。
     ```
     bash test/train_yolov7_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     ```
     ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。

   --data_path参数填写数据集路径，需写到数据集的一级目录。


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --workers                           //dataloader读数据线程数
   --batch-size                        //训练批次大小，默认32
   --data                              //训练所需的yaml文件，默认：coco.yaml                  
   --cfg                               //训练过程中涉及的参数配置文件
   --img                               //训练图像大小，默认640 640
   --batch-size                        //训练批次大小，默认：32
   --epochs                            //重复训练次数，默认：300
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAP |  FPS | AMP_Type |
| -------  | -----  | ---: | -------: |
| 1p-竞品A  | - | 53.76 |       O1 |
| 1p-NPU   | - | 55.36 |       O1 |
| 8p-竞品A  | 0.509 | 396.8 |       O1 |
| 8p-NPU   | 0.509 | 435 |       O1 |


# 版本说明

## 变更

2022.11.29：首次发布。

## 已知问题

无。
