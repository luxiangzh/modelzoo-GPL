# YOLOv7-推理指导


- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
YOLOv7是yolo系列目标检测网络，在5 FPS到160 FPS范围内的速度和精度达到了新的高度，并在GPU V100上具有30 FPS或更高的所有已知实时目标检测器中具有最高的精度56.8%AP。

- 版本说明：  
  本代码仓基于yolov7-main，其他tag可以参考该流程。
  ```
  url=https://github.com/WongKinYiu/yolov7
  tag=main
  model_name=yolov7
  ```


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本     | 环境准备指导                                                 |
| ------------------------------------------------------- |--------| ------------------------------------------------------------ |
| 固件与驱动                                                | 22.0.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                    | 6.0.0  | -                                                            |
| Python                                                  | 3.7.5  | -                                                            |
| PyTorch                                                 | 1.8.0  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/WongKinYiu/yolov7.git
   cd yolov7
   mkdir output     # 新建output文件夹，作为模型结果的默认保存路径
   ```
   
2. 安装依赖  
   ```
   pip3.7.5 install -r requirements.txt
   ```

3. 获取`OM`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
   YOLOv7_for_PyTorch
   ├── aipp.cfg   放到yolov7下
   ├── atc.sh     放到yolov7下
   └── om_nms_acc.py  放到yolov7下
   ```


## 准备数据集
- 该模型使用[coco2017 val数据集](https://cocodataset.org/#download)进行精度评估，在`Pytorch`源码根目录下新建`coco`文件夹，数据集放到`coco`里，文件结构如下：
   ```
   coco
   ├── images
   |    ├── val2017
   |        ├── 00000000139.jpg
   |        ├── 00000000285.jpg
   |         ……
   |        └── 00000581781.jpg
   |    
   └── annotations
        └── instances_val2017.json
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pt`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载YOLOv7[权重文件](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt)或使用下述命令下载。
   ```
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
   ```
   获得`yolov7x.pt`文件


2. 导出`ONNX`模型  
   运行`export.py`导出`ONNX`模型，`--dynamic-batch`支持导出动态`batch`的`ONNX`，`--simplify`简化导出的`ONNX`。
   ```
   python3.7.5 export.py --weights=yolov7x.pt --grid --img-size=640 --dynamic-batch --simplify
   ```
   获得`yolov7x.onnx`文件


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
   # 导出batchsize=8的om模型，若需导出其他batchsize的om模型，直接修改输入shape（第三个参数 8）即可。
   bash atc.sh yolov7x.onnx yolov7x_bs8 8 Ascend310P3
   ```
      - `atc`命令参数说明（参数见`atc.sh`）：
        -   `--framework`: 5代表ONNX模型  
        -   `--model`: ONNX模型文件
        -   `--output`: 输出的OM模型
        -   `--input_format`: 输入数据的格式
        -   `--input_shape`: 输入数据的shape
        -   `--log`: 日志级别
        -   `--soc_version`: 处理器型号
        -   `--insert_op_conf`: aipp配置文件
        -   `--optypelist_for_implmode`: 设置optype列表中算子的实现方式。该参数需要与--op_select_implmode参数配合使用。
        -   `--op_select_implmode`: 设置网络模型中所有算子是高精度实现还是高性能实现。高精度是指在fp16输入的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度；高性能是指在fp16输入的情况下，不影响网络精度前提的最优性能实现。


4. 使用`aipp`进行预处理
    `aipp`功能的开启需要在atc工具转换的过程中通过选项`--insert_op_conf=xxx.config`添加配置文件。AIPP配置可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191944?idPath=23710424%7C251366513%7C22892968%7C251168373)，本文案例配置文件示例`aipp.cfg`：
    ```sh
   aipp_op{
       aipp_mode : static
       input_format : RGB888_U8
       src_image_size_w : 640 
       src_image_size_h : 640
       
       csc_switch : false
       rbuv_swap_switch : false

       crop: false
       load_start_pos_h : 0
       load_start_pos_w : 0
       crop_size_w : 640
       crop_size_h : 640
       
       //均值 : 255x[0, 0, 0], 方差 : 1/(255x[1, 1, 1])
       min_chn_0 : 0
       min_chn_1 : 0
       min_chn_2 : 0
       var_reci_chn_0: 0.0039215686274509803921568627451
       var_reci_chn_1: 0.0039215686274509803921568627451
       var_reci_chn_2: 0.0039215686274509803921568627451
   
   }
   ```
   
### 3 开始推理验证

1. 安装ais_bench推理工具  
   `ais_bench`工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

2. 执行推理  
   运行`om_nms_acc.py`推理OM模型，结果默认保存在`output/predictions.json`，可设置参数`--eval`计算`mAP`，`--visible`将检测结果显示到图片。
   ```
   python3.7.5 om_nms_acc.py --model=yolov7x_bs8.om --output=output --batch=8 --conf-thres=0.001 --iou-thres=0.65 --device=0 --eval
   ```

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batchsize`的`OM`模型的性能，参考命令如下：
   ```
   python3.7.5 -m ais_bench --model=yolov7x_bs8.om --output=output --batchsize=8 --device=0 --loop=1000 
   ```

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|      | mAP   | 310P    | T4     | 310P/T4 |
|------|-------|---------|--------|---------|
| bs1  | 0.525 | 133.632 | 71.508 | 1.87    |
| bs4  | 0.525 | 144.010 | 92.083 | 1.56    |
| bs8  | 0.525 | 147.187 | 89.986 | 1.64    |
| bs16 | 0.525 | 145.478 | 88.409 | 1.65    |
| bs32 | 0.525 | 123.214 | 82.898 | 1.49    |
| bs64 | 0.525 | 115.592 | 73.922 | 1.56    |
| 最优bs | 0.525 | 147.187 | 92.083 | 1.60    |
