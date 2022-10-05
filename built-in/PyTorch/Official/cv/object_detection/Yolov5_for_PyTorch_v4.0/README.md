# yolov5 for PyTorch

# 概述

## 简述

YOLOv5是一种单阶段目标检测算法，该算法在YOLOv4的基础上添加了一些新的改进思路，使其速度与精度都得到了极大的性能提升。具体包括：输入端的Mosaic数据增强、自适应锚框计算、自适应图片缩放操作；基准端的Focus结构与CSP结构；Neck端的SPP与FPN+PAN结构；输出端的损失函数GIOU_Loss以及预测框筛选的DIOU_nms。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5/tree/v4.0
  commit_id=69be8e7
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo-GPL.git
  code_path=built-in/PyTorch/Official/cv/object_detection/
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone -b v4.0 https://github.com/ultralytics/yolov5.git       # 克隆仓库的代码
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [22.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```shell script
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取coco2017数据集，并解压，解压后目录如下所示：。


   ```shell script
   ├── coco2017: #根目录
         ├──train2017 #训练集图片，约118287张
         ├──val2017 #验证集图片，约5000张
         │──annotations #标注目录             
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（按需处理所需要的数据集）。

（1）将代码仓中cocofile/coco2yolo.py和cocofile/coco_class.txt拷贝到coco2017根目录

（2）运行coco2yolo.py

（3）运行上述脚本后，将在coco_data根目录生成train2017.txt和val2017.txt


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录，将数据集进行软连接。
```shell script
ln -s /path/to/coco2017 ./coco
```
2. 运行训练脚本。

  该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡性能采集

     ```shell script
     bash test/train_yolov5s_performance_1p.sh    
     ```

   - 单机8卡训练

     ```shell script
     bash test/train_yolov5s_full_8p.sh    
     ```


   模型训练脚本参数说明如下。

   ```shell script
   公共参数：
   --data                              //数据集路径
   --weights                           //权重加载路径
   --local_rank                        //卡ID     
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --perf                              //是否使用性能采集
   多卡训练参数：
   --device                            //是否使用多卡训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| 名字    | 精度 |  性能 |
| ------- | ----- | ---: |
| 竞品V-1p |    | 80 |
| 竞品V-8p | 0.347 | 455 |
| NPU-1p  |    | 165 |
| NPU-8p  | 0.35 | 970 |


# 版本说明

## 变更

2022.8.23：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











