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
     bash test/train_yolov5m_full_8p_high_preci.sh  # yolov5m 8p_high_precision 
     ```

   - NPU 多机多卡训练指令
   
     启动多机多卡训练。
     ```
     bash test/train_yolov5s_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     bash test/train_yolov5m_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     ```
     ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。
  
   - 纯FP32计算

     启动单卡训练
     ```
     bash test/train_yolov5s_fp32_performance_1p.sh  # yolov5s 1p_fp32_performance
     ```
     启动多卡训练
     ```
     bash test/train_yolov5s_fp32_performance_8p.sh  # yolov5s 8p_fp32_performance
     bash test/train_yolov5s_fp32_full_8p.sh  # yolov5s 8p_fp32_accuracy
     ```
     
   - 在线推理
     启动在线推理。
     ```
     bash ./test/train_yolov5s_eval.sh #在线推理
     ```
     

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   --hf32使能开启HF32模式

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
   --FP32                              //使用纯FP32方式计算，如不配置默认使用混合精度计算
   --half                              //eval执行脚本中参数，如配置默认使用混合精度计算
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  yolov5m训练结果展示表

| NAME     | mAP0.5 |  FPS | AMP_Type | Torch_Version |
| :-----:  | :---:  | :--: | :------: | :------: |
| 1p-竞品A  | - | 181 |       O1 |    1.8 |
| 8p-竞品A | 64.1 |  1264  |    O1    |      1.8      |
| 1p-NPU | - | 176.3 | O1 | 1.8 |
|  8p-NPU  | 63.6 | 1432.2 |    O1    |      1.8      |

**表 3**  yolov5m高精度8p训练结果展示表

| NAME     | mAP0.5 |  FPS | AMP_Type | Torch_Version |
| :-----:  | :---:  | :--: | :------: | :------: |
| 8p-竞品A | 64.1 |  1264  |    O1    |      1.8      |
|  8p-NPU  | 64.5 | 1432.2 |    O1    |      1.8      |

**表 4**  yolov5s训练结果展示表

| NAME     | mAP0.5~0.95 |  FPS | AMP_Type | Torch_Version | Architecture | Device_Type |
| :-----:  | :---:  | :------: | :-----:  | :-----:  | :-----: | :-----: |
| 1p-NPU | - | 265 | O1 | 1.8 | | - |
| 8p-NPU   | 35.4 | 2044.5 |       O1 |    1.8 | Arm | - |
| 1p-NPU  | - | 317 | O1 | 1.8 | 非Arm | - |
| 8p-NPU  | - | 1163.6 | O1 | 1.8 | 非Arm | - |
| 8p-NPU  | 35.5 | 1507.7 | FP32 | 1.8 | Arm | 910B1 |

# 版本说明

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

## 变更

2023.02.16：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

1. 训练过程中若遇到该问题`wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key])`，在不能获取到key的情况下，请卸载三方库`wandb`，再进行训练。

# YOLOv5-离线推理指导


- [YOLOv5-推理指导](#yolov5-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
        - [\* 如果有多卡推理的需求，请跳过该步骤，om\_val.py该脚本不支持多卡推理](#-如果有多卡推理的需求请跳过该步骤om_valpy该脚本不支持多卡推理)
  - [多卡推理](#多卡推理)
  - [aipp](#aipp)
- [模型推理性能\&精度](#模型推理性能精度)
- [FAQ](#faq)

******


# 概述
YOLO系列网络模型是最为经典的one-stage算法，也是目前工业领域使用最多的目标检测网络，YOLOv5网络模型是YOLO系列的最新版本，在继承了原有YOLO网络模型优点的基础上，具有更优的检测精度和更快的推理速度。  
YOLOv5版本不断迭代更新，不同版本的模型结构有所差异。比如Conv模块各版本差异示例如下:  
  
  | yolov5版本	 | Conv模块激活函数 |
  |:---------:|:----------:|
  | 6.0	      |    SiLU    |


YOLOv5每个版本主要有4个开源模型，分别为YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，四个模型的网络结构基本一致，只是其中的模块数量与卷积核个数不一致。YOLOv5s模型最小，其它的模型都在此基础上对网络进行加深与加宽。
- 版本说明（目前已适配以下版本）：
  ```
  url=https://github.com/ultralytics/yolov5
  tag=v6.0
  model_name=yolov5
  ```


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本     | 环境准备指导                                                                                                                                      |
| ------------------------------------------------------- |--------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 固件与驱动                                               | 22.0.4 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                                               |
| CANN                                                    | 6.0.0  | [推理应用开发学习文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/aclpythondevg/aclpythondevg_0000.html) |
| Python                                                  | 3.7.5  | -                                                                                                                                           |
| PyTorch                                                 | 1.11.0 | -                                                                                                                                           |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                                                                                                           |


# 快速上手

## 安装依赖

   ```
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   python3 -m pip install --upgrade pip
   python3 -m pip install wheel
   python3 -m pip install .
   cd ..
   pip3 install -r onnx_requirements.txt
   ```

## 准备数据集
- 该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`yolov5`源码根目录下新建`coco`文件夹，数据集放到`coco`里，文件结构如下：
   ```
   coco
   ├── val2017
      ├── 00000000139.jpg
      ├── 00000000285.jpg
      ……
      └── 00000581781.jpg
   ├── instances_val2017.json
   └── val2017.txt
   ```
   `val2017.txt`中保存`.jpg`的相对路径，请自行生成该`txt`文件，文件内容实例如下：
   ```
   ./val2017/00000000139.jpg
   ./val2017/00000000285.jpg
   ……
   ./val2017/00000581781.jpg
   ```
 
## 模型推理
模型推理提供后处理脚本方式：  
直接用官网`export.py`导出`onnx`模型，模型结构和官网一致，推理流程也和官方一致，NMS后处理采用脚本实现。

### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   使用训练保存的模型文件yolov5.pt

2. 导出`ONNX`模型  
   运行`bash pth2onnx.sh`导出动态shape的`ONNX`模型，模型参数在[model.yaml](model.yaml)中设置。
   ```
   bash pth2onnx.sh --tag 6.0 --model yolov5 --nms_mode nms_script  # nms_script

   ```
   - 命令参数说明：
     -   `--tag`：模型版本，可选`[6.0]`, 默认`6.0`。
     -   `--model`：模型文件名
     -   `--nms_mode`：模型推理方式，可选`[nms_script]`


3. 使用`ATC`工具将`ONNX`模型转`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
     该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（`${soc_version}`）
   ```
   npu-smi info
   # 该设备芯片名为Ascend910A （自行替换）
   回显如下：
   +-------------------+-----------------+------------------------------------------------------+
   | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
   | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
   +===================+=================+======================================================+
   | 0       910A     | OK              | 15.8         42                0    / 0              |
   | 0       0         | 0000:82:00.0    | 0            1074 / 32768                            |
   +===================+=================+======================================================+
   | 1       910A     | OK              | 15.4         43                0    / 0              |
   | 0       1         | 0000:89:00.0    | 0            1070 / 32768                            |
   +===================+=================+======================================================+
   ```

   3.3 导出`OM`模型  
   运行`onnx2om.sh`导出`OM`模型。
   ```
   bash onnx2om.sh --tag 6.0 --model yolov5 --nms_mode nms_script --bs 4 --soc Ascend910A  # nms_script
   ```
      - `atc`命令参数说明（参数见`onnx2om.sh`）：
        -   `--model`：ONNX模型文件
        -   `--output`：输出的OM模型
        -   `--framework`：5代表ONNX模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--soc_version`：处理器型号
        -   `--log`：日志级别
        -   `--compression_optimize_conf`：模型量化配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0084.html)
        -   `--enable_small_channel`：输入端aipp算子配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0081.html)
        -   `--insert_op_conf`：输入端aipp算子配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0072.html)


### 2 开始推理验证

1. 安装`ais-infer`推理工具  
   `ais-infer`工具获取及使用方式请点击查看 [[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

##### * 如果有多卡推理的需求，请跳过该步骤，om_val.py该脚本不支持多卡推理

2. 执行推理 & 精度验证  
   运行`om_val.py`推理OM模型，模型参数在[model.yaml](model.yaml)中设置，结果默认保存在`predictions.json`。
   ```
   python3 om_val.py --tag 6.0 --model=yolov5_bs4.om --nms_mode nms_script --batch_size=4  # nms_script
   ```
   - 命令参数说明：
     -   `--tag`：模型版本，可选`[6.0]`。
     -   `--model`：模型文件名。
     -   `--nms_mode`：模型推理方式，可选`[nms_script]`。
     -   `--batch_size`: 模型推理batch大小，默认`4`。
     -   `--cfg_file`：模型推理参数设置，默认读取文件[model.yaml](model.yaml)。

3. 性能验证  
   可使用`ais_infer`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 -m ais_bench --model=yolov5_bs4.om --loop=1000 --batchsize=4  # nms_script
   ```

# 模型推理性能&精度

调用ACL接口推理计算，yolov5m_6.0性能&精度参考下列数据。

    | 模型tag |   芯片型号   | 最优Batch |    数据集    |         阈值       | 精度 (mAP@0.5) | OM模型性能 (fps) |
    |:------:|:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
    | 6.0   | Ascend910A |     4      | coco val2017 |  conf=0.0005 iou=0.5  |     64.2     |   828.48    |
    
    
# FAQ
1、如遇到问题：    
   <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777)>
   解决方案：
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context