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
  url=https://github.com/WongKinYiu/yolov7.git
  commit_id=072f76c72c641c7a1ee482e39f604f6f8ef7ee92
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo-GPL
  code_path=built-in/PyTorch/Official/cv/object_detection
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。



# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  | PyTorch 1.11 | pillow==9.1.0 |
  | PyTorch 2.1 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》，需要安装二进制算子包。

  > **说明：** 
  > 在X86_64架构下，官方torch包使用的是MKI加速库。如果需要使用blas和lapack加速库来提升性能，请使用源码编译安装方式安装官方torch包。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本

  pip install -r 1.11_requirements.txt  # PyTorch1.11版本

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

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
  > **说明：** 
  > 该数据集的训练过程脚本只作为一种参考示例。
2. 在源码包根目录下配置数据集路径。

  - 编译安装torchvision
  
    ***为了更快的推理性能，请编译安装而非直接安装torchvision***

     ```
      git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
      cd vision
      python setup.py bdist_wheel
      pip3 install dist/*.whl
     ```
3. 编译安装Opencv-python。

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
| 1p-NPU-ARM   | - | 55.36 |       O1 |
| 1p-NPU-非ARM   | - | 44.176 |       O1 |
| 8p-竞品A  | 0.509 | 396.8 |       O1 |
| 8p-NPU-ARM   | 0.509 | 435 |       O1 |
| 8p-NPU-非ARM   | - | 281.344 |       O1 |

> **说明：** 
> 表2中非ARM的性能值是直接使用官方torch包测试的结果。


# 版本说明

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

## 变更

2022.11.29：首次发布。

## 已知问题

无。
