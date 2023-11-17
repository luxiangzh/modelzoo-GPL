# YOLOv5-推理指导


- [YOLOv5-推理指导](#yolov5-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
    - [1 模型编译](#1-模型编译)
    - [2 执行推理](#2-执行推理)
- [模型推理性能\&精度](#模型推理性能精度)
- [FAQ](#faq)

******


# 概述
YOLO系列网络模型是最为经典的one-stage算法，也是目前工业领域使用最多的目标检测网络，YOLOv5网络模型是YOLO系列的最新版本，在继承了原有YOLO网络模型优点的基础上，具有更优的检测精度和更快的推理速度。  
YOLOv5版本不断迭代更新，不同版本的模型结构有所差异。比如Conv模块各版本差异示例如下:  
  
  | yolov5版本	 | Conv模块激活函数 |
  |:---------:|:----------:|
  | 2.0	      | LeakyRelu  |
  | 3.0	      | LeakyRelu  |
  | 3.1	      |   hswish   |
  | 4.0	      |    SiLU    |
  | 5.0	      |    SiLU    |
  | 6.0	      |    SiLU    |
  | 6.1	      |    SiLU    |
  | 6.2	      |    SiLU    |
  | 7.0	      |    SiLU    |

YOLOv5每个版本主要有4个开源模型，分别为YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，四个模型的网络结构基本一致，只是其中的模块数量与卷积核个数不一致。YOLOv5s模型最小，其它的模型都在此基础上对网络进行加深与加宽。
- 版本说明（目前已适配以下版本）：
  ```
  url=https://github.com/ultralytics/yolov5
  tag=v2.0/v5.0/v6.0/v6.1
  model_name=yolov5
  ```


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本     | 环境准备指导                                                                                                                                      |
| ------------------------------------------------------- |--------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 固件与驱动                                               | 23.0.RC1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                                               |
| CANN                                                    | 7.0.RC1.alpha003  | |
| Python                                                  | 3.9.0  | -                                                                                                                                           |
| PyTorch                                                 | 2.0.1 | -                                                                                                                                           |
| TorchVision | 0.15.2      |            -                                                                                                                                 |


# 快速上手

## 获取源码

1. 获取ModelZoo-GPL代码  

   ```
   cd ..
   git clone https://gitee.com/ascend/modelzoo-GPL.git
   cd /modelzoo-GPL/built-in/AscendIE/TorchAIE/built-in/cv/detection/Yolov5
   ``` 
2. 获取`YoloV5`源码  
   ```
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   git checkout v2.0/v5.0/v6.0/v6.1  # 切换到所用版本
   # 将推理部署代码拷贝到yolov5源码相应目录下。
   cp -r ../common ./
   cp ../aie_compile.py ./
   cp ../aie_val.py ./
   cp ../model.ymal ./
   ``` 

3. 安装依赖  
   ```
   pip install numpy==1.23
   pip install tqdm
   pip install opencv-python
   pip install pandas==2.0.2
   pip install requests
   pip install pyyaml
   pip install Pillow==9.5
   pip install matplotlib
   pip install seaborn
   pip install pycocotools
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
   ├── annotations
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

### 1 模型编译  
将模型`.pth`文件首先导出为torchscript模型，然后通过torch_aie进行编译，使其可以运行在昇腾npu上。

1. 获取权重文件  
   在[链接](https://github.com/ultralytics/yolov5/tags)中找到所需版本下载，也可以使用下述命令下载。
   ```
   wget https://github.com/ultralytics/yolov5/releases/download/v${tag}/${model}.pt
   ```
- 命令参数说明：
   -   `${tag}`：模型版本，可选`[2.0/5.0/6.0/6.1]`
   -   `${model}`：模型大小，可选`yolov5[s/m]`

2. 导出`torchscript`模型  
   运行yolov5原本代码中的`export.py`脚本，导出`torchscript`模型。
   ```
   # 对于v2.0和v5.0，可参考如下命令
   python ./models/export.py --weights=yolov5s.pt
   # 对于v6.0和v6.1，可参考如下命令
   python export.py --weights=yolov5s.pt --include=torchscript
   # 为了方便区分不同版本，建议重命名模型，以v6.0为例
   mv yolov5s.torchscript.pt yolov5s_v6.torchscript.pt
   ```

3. 编译模型    
   运行`aie_compile.py`编译上一步导出的torchscript模型。模型参数在[model.yaml](model.yaml)中设置。
   ```
   python aie_compile.py --ts_model=yolov5s_v6.torchscript.pt --batch_size=4
   ```
- 命令参数说明（参数见`aie_compile.py`）：
   -   `--ts_model`：torchscrip模型的路径
   -   `--batch_size`：批大小

   以上述命令为例，编译完成后将会生成yolov5s_v6_bs4_aie.pt文件


### 2 执行推理
   1. 在coco2017数据集上验证  
   运行`aie_val.py`脚本，在coco2017上输出模型的mAP和吞吐量。
   ```
   # 以前述步骤所示命令为前提，推理执行命令参考如下
   python aie_val.py --data_path=./coco --ground_truth_json=./coco/annotations/instances_val2017.json --tag=6.0 --model=./yolov5s_v6_bs4_aie.pt --batch_size=4   
   ```
   - 命令参数说明（参数见`aie_val.py`）：
      -   `--data_path`：coco数据集所在路径
      -   `--ground_truth_json`：数据集的真实标注
      -   `--tag`：yolov5版本
      -   `--model`：编译好的yolov5模型路径
      -   `--batch_size`：批大小


# 模型推理性能&精度

结果torch_aie编译后的yolov5s在coco2017数据集上的性能&精度参考下列数据。
1. v2.0版本

    |   芯片型号   | Batch大小 |    数据集    |         阈值       | 精度 (mAP@0.5) | 吞吐量 |
    |:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
   | Ascend310P3 |     1      | coco val2017 |  conf=0.001 iou=0.6  |     54.7     |   756.98   |
    | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   764.13    |
   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   714.36    |
   | Ascend310P3 |     16      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   668.26    |
   | Ascend310P3 |     32      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   657.42    |
  

1. v5.0版本

    |   芯片型号   | Batch大小 |    数据集    |         阈值       | 精度 (mAP@0.5) | 吞吐量 |
    |:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
   | Ascend310P3 |     1      | coco val2017 |  conf=0.001 iou=0.6  |     55.5     |   663.33   |
    | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   721.00    |
   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   690.75    |
   | Ascend310P3 |     16      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   647.78    |
   | Ascend310P3 |     32      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   641.35    |

1. v6.0版本

    |   芯片型号   | Batch大小 |    数据集    |         阈值       | 精度 (mAP@0.5) | 吞吐量 |
    |:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
   | Ascend310P3 |     1      | coco val2017 |  conf=0.001 iou=0.6  |     55.9     |   673.61   |
    | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   837.38    |
   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   714.51    |
   | Ascend310P3 |     16      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   675.95    |
   | Ascend310P3 |     32      | coco val2017 |  conf=0.001 iou=0.6  |     -     |   648.12    |

# FAQ
常见问题1：AttributeError: ‘Upsample‘ object has no attribute ‘recompute_scale_factor‘
解决：这是由于旧版本的yolov5项目依赖于较低版本的pytorch，因此在较新的pytorch2.0.1中若遇到这个问题，可以参考进行 https://blog.csdn.net/weixin_44577224/article/details/130183964 修复。

常见问题2：_pickle.UnpicklingError: STACK_GLOBAL requires str
解决：删除coco数据集目录下的.cache文件再重新执行脚本即可。
