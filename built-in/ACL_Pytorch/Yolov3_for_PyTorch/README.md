# YOLOv3-推理指导


- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
YOLOv3是一种端到端的one-stage目标检测模型。相比YOLOv2，YOLOv3采用了一个新的backbone-Darknet-53来进行特征提取工作，
这个新网络比Darknet-19更加强大，也比ResNet-101或者ResNet-152更加高效。
同时，对于一张输入图片，YOLOv3可以在3个不同尺度预测物体框，每个尺度预测三种大小的边界框，通过多尺度联合预测的方式有效提升了小目标的检测精度。

- 版本说明：  
  本代码仓基于yolov3最新tag v9.6.0推理，其他tag可以参考该流程。
  ```
  url=https://github.com/ultralytics/yolov3/tree/v9.6.0
  tag=v9.6.0
  model_name=yolov3
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型  |            大小             | 数据排布格式 | 
  |:-------:|:-------:|:------:|:----------:| 
  | img     | FLOAT32 | batchsize x 3 x 640 x 640 |  NCHW  |
  | imginfo | FLOAT32 | batchsize x 4             |  ND    |


- 输出数据

  | 输出数据     |  数据类型   |        大小        | 数据排布格式 |
  |:-------:|:----------:|:----------------:|:----------:|
  | boxout     | FLOAT32 | batchsize x 6144 |   ND       |
  | boxout_num | INT32   | batchsize x 8    |   ND       |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------- |---------| ------------------------------------------------------------ |
| 固件与驱动                                                | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                    | 6.0.RC1 | -                                                            |
| Python                                                  | 3.7.5   | -                                                            |
| PyTorch                                                 | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/ultralytics/yolov3.git
   cd yolov3
   git checkout v9.6.0
   mkdir output     # 新建output文件夹，作为模型结果的默认保存路径
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```

3. 获取`OM`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
   YOLOv3_for_PyTorch
   ├── v9_6_0.patch   放到yolov3下
   ├── add_nms.py     放到yolov3下
   ├── atc.sh         放到yolov3下
   └── om_val.py      放到yolov3下
   ```   


## 准备数据集
- 该模型使用[coco2017 val数据集](https://cocodataset.org/#download)进行精度评估，在`Pytorch`源码根目录下新建`coco2017`文件夹，数据集放到`coco2017`里，文件结构如下：
   ```
   coco2017
   ├── val2017
      ├── 00000000139.jpg
      ├── 00000000285.jpg
      ……
      └── 00000581781.jpg
   └── instances_val2017.json
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载YOLOv3[权重文件](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt)或使用下述命令下载。
   ```
   wget https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt
   ```

2. 导出`ONNX`模型  
   运行`export.py`导出`ONNX`模型，`--dynamic`支持导出动态`batch`的`ONNX`，`--simplify`简化导出的`ONNX`。
   ```
   git apply v9_6_0.patch     # 导出ONNX模型需要修改开源代码仓代码
   python3 export.py --weights=yolov3.pt --imgsz=640 --batch=1 --include=onnx --opset=11 --dynamic --simplify
   ```
   - 若基于其他tag推理，此处有2点不同需注意：  
     1）`patch`不同。`patch`的目的是对源码做一些修改，已提供`v9.6.0`的`patch`，其他tag可参考`v9_6_0.patch`，切换到对应`tag`直接修改源码即可。  
     2）`export.py`的路径和参数可能不同。之前的版本`export.py`在`models`文件夹下，且不包含`--dynamic --simplify`等参数，去掉这些参数即可，不影响导出。
   
   运行`add_nms.py`为导出的`ONNX`添加后处理算子`NMS`，参数`--conf-thres`和`--iou-thres`分别设置后处理的置信度阈值和`iou`阈值，`--class`设置实际推理数据集的类别数。
   ```
   python3 add_nms.py --pt=yolov3.pt --onnx=yolov3.onnx --output=output --batch=1 --conf=0.4 --iou=0.5 --class=80
   ```

3. 使用`ATC`工具将`ONNX`模型转`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
     该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（${soc_version}）
   ```
   npu-smi info
   #该设备芯片名为Ascend310P3 （自行替换）
   回显如下：
   +-------------------+-----------------+------------------------------------------------------+
   | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
   | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
   +===================+=================+======================================================+
   | 0       310P3     | OK              | 15.8         42                0    / 0              |
   | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
   +===================+=================+======================================================+
   | 1       310P3     | OK              | 15.4         43                0    / 0              |
   | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
   +===================+=================+======================================================+
   ```

   3.3 执行ATC命令  
   运行`atc.sh`导出`OM`模型，默认保存在`output`文件夹下。
   ```
   # 导出不带nms后处理的模型，用于性能测试
   bash atc.sh --model yolov3 --bs 1 --img_h 640 --img_w 640 --output_dir output --soc Ascend310P3
   # 导出带nmx后处理的模型，用于精度测试
   bash atc.sh --model yolov3_nms --bs 1 --img_h 640 --img_w 640 --output_dir output --soc Ascend310P3
   ```
      - `atc`命令参数说明（参数见`atc.sh`）：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号

    
### 2 开始推理验证

1. 安装`ais-infer`推理工具  
   `ais-infer`工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

2. 执行推理  
   运行`om_val.py`推理OM模型，结果默认保存在`output/predictions.json`，可设置参数`--eval`计算`mAP`，`--visible`将检测结果显示到图片。
   ```
   python3 om_val.py --model=yolov3_nms_bs1.om --output=output --batch=1 --img-size 640 640 --eval
   ```

3. 性能验证  
   可使用`ais_infer`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 ${ais_infer_path}/ais_infer.py --model=yolov3_bs1.om --loop=20 --batchsize=1
   python3 ${ais_infer_path}/ais_infer.py --model=yolov3_nms_bs1.om --loop=20 --batchsize=1
   ```

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|     模型      |   芯片型号   |  Batch Size   |    数据集      |     阈值           | 精度 (mAP) |         性能 (fps)       |
|:-----------:|:-----------:|:-------------:|:------------:|:------------------:|:--------:|:------------------------:|
| yolov3_nms  | Ascend310P3 |   1 / 4 / 8   | coco2017 val |  conf=0.4 iou=0.5  |  0.399   | 128.05 / 164.56 / 165.78 |
| yolov3_nms | Ascend310P3 |   1 / 4 / 8   | coco2017 val | conf=0.05 iou=0.65 |  0.453   | 128.23 / 165.18 / 167.24 |
