# YOLOv8-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)

  - [准备数据集](#section183221994411)

  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

******

# 概述
YOLO系列网络模型是最为经典的one-stage算法，也是目前工业领域使用最多的目标检测网络，YOLOv8网络模型是YOLO系列的最新版本，在继承了原有YOLO网络模型优点的基础上，具有更高的检测精度。

- 参考实现：

  ```
  url=https://github.com/ultralytics/ultralytics
  commit_id=7a7c8dc7b70cf4bc0be18763a6b66805974ecbe6
  model_name=yolov8
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型  | 大小                      | 数据排布格式  |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据  | 数据类型  | 大小                  | 数据排布格式  |
  | -------- | -------- | --------------------- | ------------ |
  | output0  | FLOAT32  | batchsize x 84 x 8400 | ND           |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本     | 环境准备指导                                                                                                                                      |
| ------------------------------------------------------- |--------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 固件与驱动                                               | 22.0.4 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                                               |
| CANN                                                    | 6.0.0  | [推理应用开发学习文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/aclpythondevg/aclpythondevg_0000.html) |
| Python                                                  | 3.7.5  | -                                                                                                                                           |
| PyTorch                                                 | 1.13.0 | -                                                                                                                                           |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                                                                                                           |


# 快速上手

## 获取源码

1. 获取源码  

   ```
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   git reset --hard 7a7c8dc7b70cf4bc0be18763a6b66805974ecbe6
   pip3 install -e .
   git apply ../dataloader.patch
   cd  ..
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```
   

## 准备数据集

1. 获取原始数据集。
   
   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，文件结构如下：

   ```
   coco
   ├── annotations
      └── instance_val2017.json
   ├── images
      ├── test2017
      ├── train2017
      ├── val2107
         ├── 00000000139.jpg
         ├── 00000000285.jpg
         ......
         └── 00000581781.jpg
      ├── test2017.zip
      ├── train2017.zip
      ├── val2017.zip
   ├── labels
   ├── LICENSE
   ├── README.txt
   ├── test-dev2017.txt
   ├── train2017.txt
   ├── val2017.txt
   ```


## 模型推理

1. 模型转换  

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件  

      在[链接](https://github.com/ultralytics/assets/releases/tag/v0.0.0)中找到所需版本下载，也可以使用下述命令下载：

      ```
      wget https://github.com/ultralytics/assets/releases/download/v0.0.0/${model}.pt
      ```

      - 参数说明：
         - `${model}`：模型大小，可选`yolov8[n/s/m/l/x]`

      > **说明**：后续以 yolov8n 模型作为示例进行指导说明，请根据实际情况进行对应修改。

   2. 导出ONNX模型  
   
      模型参数在 [model.yaml](model.yaml) 中设置，运行下述命令导出ONNX模型。

      ```
      python3 pth2onnx.py --pt=yolov8n.pt
      ```

      - 参数说明：
         - `--pt`：权重文件路径
      
      获得 yolov8n.pt 文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         `/usr/local/Ascend/ascend-toolkit/set_env.sh` 是CANN软件包安装在默认路径生成的环境变量文件，如自行安装在其他位置，需要改为自己的安装路径。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

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

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=yolov8n.onnx --input_format=NCHW --input_shape='images:{batchsize},3,640,640' --output_type=FP16 --output=yolov8n_bs{batchsize} --log=error --soc_version=Ascend{chip_name} --op_precision_mode=op_precision.ini
         ```

         - 参数说明：

           -   --model：为 ONNX 模型文件
           -   --framework：5 代表 ONNX 模型
           -   --input\_format：输入数据的格式
           -   --input\_shape：输入数据的形状
           -   --output\_type：输出数据的类型
           -   --output：输出的 OM 模型
           -   --log：日志级别
           -   --soc\_version：处理器型号

         - 自定义参数说明：
           - {batchsize} 需要指定为要生成的 om 模型的批处理大小， 如 1、4、8、16 等，此处以 8 为示例进行说明。

         运行成功后生成 yolov8n_bs8.om 模型文件。

    
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问 [ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench) 代码仓，根据readme文档进行工具安装。

   2. 参数设置

      在 `ultralytics/ultralytics/yolo/cfg` 文件夹的 `default.yaml` 与 `ultralytics/ultralytics/yolo/data/datasets` 文件夹的 `coco.yaml` 中填入相关参数。
      
      请根据实际情况修改相关参数。

      - default.yaml 参数说明：
           -   `model`：pt 权重文件，必须与 onnx 模型对应一致
           -   `data`: 数据配置文件，此处以 coco.yaml 为例进行说明
           -   `batch`：批处理数量大小，必须与 om 模型的 batchsize 相等
           -   `project`：推理结果的总保存路径
           -   `name`：每次推理结果的文件名称
           -   ......
      
      - coco.yaml 参数说明：
           -   `path`：coco 数据集存放路径
           -   `train`：train2017.txt， 训练数据集路径文本
           -   `val`：val2017.txt，验证数据集路径文本
           -   `test`：test-dev2017.txt，测试数据集路径文本

   3. 执行推理 & 精度验证  

      运行 `om_infer.py` 推理 OM 模型，结果默认保存在 `project/name` 文件夹下的 `predictions.json`，精度计算结果通过打屏显示。

      ```
      python3 om_infer.py --weight=yolov8n.pt --om=yolov8n_bs8.om --device_id=0
      ```

      - 命令参数说明：
      -   `--weight`：pt 权重文件所在路径
      -   `--om`：om 模型所在路径
      -   `--device_id`：使用芯片的序号

   4. 性能验证  

      可使用 ais_infer 推理工具的纯推理模式验证不同 batch_size 的 OM 模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=yolov8n_bs8.om --loop=1000 --batchsize=8
      ```

      - 参数说明：
        - `--model`：om 模型所在路径
        - `--loop`：循环执行的次数
        - `--batchsize`：批处理数量大小


# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

| 模型 |   芯片型号   | 最优Batch |    数据集    |         阈值       | 精度 (mAP@0.5) | OM模型性能 (fps) |
|:------:|:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
| yolov8n   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.7  |     52.5     |   753.23   |
| yolov8s   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.7  |     61.6     |   443.60   |
| yolov8m   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.7  |     67.0     |   238.28   |
| yolov8l   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.7  |     69.6     |   148.60   |
| yolov8x   | Ascend310P3 |     8      | coco val2017 |  conf=0.001 iou=0.7  |     70.6     |   100.32   |
