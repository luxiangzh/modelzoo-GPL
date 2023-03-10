# Yolov5_for_PyTorch_v6.0

- [Yolov5\_for\_PyTorch\_v6.0](#yolov5_for_pytorch_v60)
- [概述](#概述)
  - [简述](#简述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [开始训练](#开始训练)
  - [训练模型](#训练模型)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 概述

## 简述
YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。YOLOv5于2020.05.27首次发布，截至2020.12.01仍在更新，目前NPU适配的版本为Yolov5 Tag=v6.0。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5.git
  Tag=v6.0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo-GPL
  code_path=built-in/PyTorch/Official/cv/object_detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集coco2017，将数据集上传到服务器任意路径下并解压，数据集目录结构参考如下所示：

   ```
   ├── coco #根目录
        ├── train2017 #训练集图片，约118287张
        ├── val2017 #验证集图片，约5000张
        └── annotations #标注目录
        		  ├── instances_train2017.json #对应目标检测、分割任务的训练集标注文件
        		  ├── instances_val2017.json #对应目标检测、分割任务的验证集标注文件
        		  ├── captions_train2017.json
        		  ├── captions_val2017.json
        		  ├── person_keypoints_train2017.json
        		  └── person_keypoints_val2017.json
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。
2. 生成yolov5专用标注文件。

   （1）将代码仓中coco/coco2yolo.py和coco/coco_class.txt拷贝到coco数据集**根目录**。

   （2）运行coco2yolo.py。

   ```
   python3 coco2yolo.py
   ```

   （3）运行上述脚本后，将在coco数据集**根目录**生成train2017.txt和val2017.txt。
3. 在源码包根目录下配置数据集路径。

   ```
   mkdir datasets
   ln -s coco_path ./datasets/coco  # coco_path为数据集实际路径
   ```

  - 编译安装torchvision
  
    ***为了更快的推理性能，请编译安装而非直接安装torchvision***

     ```
      git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
      cd vision
      python setup.py bdist_wheel
      pip3 install dist/*.whl
     ```
4. 编译安装Opencv-python。

   为了获得最好的图像处理性能，**请编译安装opencv-python而非直接安装**。编译安装步骤如下：

   ```
   export GIT_SSL_NO_VERIFY=true
   git clone https://github.com/opencv/opencv.git
   cd opencv
   mkdir -p build
   cd build
   cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m -D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include -D   PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..
   make -j$nproc
   make install
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
     bash test/train_yolov5s_performance_1p.sh  # yolov5s 1p_performance
     bash test/train_yolov5m_performance_1p.sh  # yolov5m 1p_performance
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash test/train_yolov5s_performance_8p.sh  # yolov5s 8p_performance
     bash test/train_yolov5m_performance_8p.sh  # yolov5m 8p_performance
     bash test/train_yolov5s_full_8p.sh  # yolov5s 8p_accuracy
     bash test/train_yolov5m_full_8p.sh  # yolov5m 8p_accuracy
     ```

   - NPU 多机多卡训练指令
   
     启动多机多卡训练。
     ```
     bash test/train_yolov5s_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     bash test/train_yolov5m_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     ```
     ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --workers                           //加载数据进程数     
   --batch-size                        //训练批次大小
   --epochs                            //重复训练次数
   --weights                           //初始权重路径
   --rect                              //矩形训练
   --nosave                            //保存最后一个权重
   --noval                             //验证最后一个epoch
   --artifact_alias                    //数据集版本
   --save-period                       //权重保存
   --native_amp                        //使用torch amp进行混合精度训练，如不配置默认使用apex
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  yolov5m训练结果展示表

| NAME     | mAP50 |  FPS | AMP_Type | Torch_Version |
| :-----:  | :---:  | :--: | :------: | :------: |
| 1p-竞品A  | - | 181 |       O1 |    1.5 |
| 8p-竞品A | 0.641 |  1264  |    O1    |      1.5      |
| 1p-NPU | - | 176.3 | O1 | 1.8 |
|  8p-NPU  | 0.636 | 1432.2 |    O1    |      1.8      |

**表 3**  yolov5s训练结果展示表

| NAME     | mAP50 |  FPS | AMP_Type | Torch_Version |
| :-----:  | :---:  | :------: | -------  | -------  |
| 1p-NPU | - | 265 | O1 | 1.8 |
| 8p-NPU   | - | 2044.5 |       O1 |    1.8 |

# 版本说明

## 变更

2023.02.16：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

1. 训练过程中若遇到该问题`wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key])`，在不能获取到key的情况下，请卸载三方库`wandb`，再进行训练。
