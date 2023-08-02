# YOLOv5-推理指导


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
  tag=v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0
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
| PyTorch                                                 | 1.10.1 | -                                                                                                                                           |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                                                                                                           |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   git checkout v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0  # 切换到所用版本
   ```

2. 获取`OM`推理代码  
   由于v7.0版本的开源yolov5模型，head层发生了变动，所以后处理也需要做相应修改
   ```
   # 返回yolov5_for_pytorch目录
   cd ..
   # tag v7.0需要执行以下操作
   git apply 7.0.patch
   ```
   将推理部署代码放到`yolov5`源码相应目录下。
   ```
    Yolov5_for_Pytorch
    └── common                        放到yolov5下
      ├── util                          模型/数据接口
      ├── quantify                      量化接口
      ├── atc_cfg                       atc转模型配置文件
      └── patch                         v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0 兼容性修改
    ├── model.yaml                    放到yolov5下 
    ├── pth2onnx.sh                   放到yolov5下
    ├── onnx2om.sh                    放到yolov5下
    ├── aipp.cfg                      放到yolov5下
    ├── om_val.py                     放到yolov5下
    ├── yolov5_preprocess_aipp.py     放到yolov5下
    ├── yolov5_preprocess.py          放到yolov5下
    ├── yolov5_postprocess.py         放到yolov5下
    └── requirements.txt              放到yolov5下
   ```   

3. 安装依赖  
   ```
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   python3 -m pip install --upgrade pip
   python3 -m pip install wheel
   python3 -m pip install .
   cd ../..
   pip3 install -r requirements.txt
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
模型推理提供两种方式，区别如下：  
1. `nms`后处理脚本（`nms_script`）   
    直接用官网`export.py`导出`onnx`模型，模型结构和官网一致，推理流程也和官方一致，NMS后处理采用脚本实现。
    * 注意：如果使用的是nms_script方式，需要修改model.yaml文件，将其中的配置conf_thres:0.4和iou_thres:0.5修改为conf_thres:0.001和iou_thres:0.6，后续该方式下精度测试也是采用修改后的配置。
 
2. `nms`后处理算子（`nms_op`）  
    * 注意：为提升模型端到端推理性能，我们对上一步导出的`onnx`模型做了修改，增加后处理算子，将`NMS`后处理的计算集成到模型中。后处理算子存在阈值约束，要求 
    `conf>0.1`，由于其硬性要求，所以model.yaml文件默认设置conf_thres:0.4。使用nms_op方式，不需要修改model.yaml文件。

### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   在[链接](https://github.com/ultralytics/yolov5/tags)中找到所需版本下载，也可以使用下述命令下载。
   ```
   wget https://github.com/ultralytics/yolov5/releases/download/v${tag}/${model}.pt
   ```
   - 命令参数说明：
     -   `${tag}`：模型版本，可选`[2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]`
     -   `${model}`：模型大小，可选`yolov5[n/s/m/l]`,当前未适配X

2. 导出`ONNX`模型  
   运行`bash pth2onnx.sh`导出动态shape的`ONNX`模型，模型参数在[model.yaml](model.yaml)中设置。
   ```
   bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_script  # nms_script
   bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_op  # nms_op
   ```
   - 命令参数说明：
     -   `--tag`：模型版本，可选`[2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]`。
     -   `--model`：模型大小，可选`yolov5[n/s/m/l]`。
     -   `--nms_mode`：模型推理方式，可选`[nms_op/nms_script]`。`nms_op`方式下，pth导出onnx模型过程中会增加NMS后处理算子，后处理算子的参数`class_num`、`conf_thres`和`iou_thres`在[model.yaml](model.yaml)中设置。


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
   # 该设备芯片名为Ascend310P3 （自行替换）
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

   3.3 导出非量化`OM`模型  
   运行`onnx2om.sh`导出`OM`模型。
   ```
   bash onnx2om.sh --tag 6.1 --model yolov5s --nms_mode nms_script --bs 4 --soc Ascend310P3  # nms_script
   bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend310P3  # nms_op
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


   3.4 导出量化`OM`模型（可选）  
   （1）量化存在精度损失，要使用实际数据集进行校准以减少精度损失。提供 [generate_data.py](common/quantify/generate_data.py) 生成校准数据，[calib_img_list.txt](common/quantify/calib_img_list.txt) 中提供默认的校准数据，根据实际数据路径修改。运行脚本会新建`calib_data`文件夹，将生成的数据bin文件放到该文件夹下。  
   ```
   python3 common/quantify/gen_calib_data.py  
   ```
   （2）导出`OM`模型时设置`--quantify`参数，使能模型量化，量化对性能的提升视模型而定，实际效果不同。 
   ```
   bash onnx2om.sh --tag 6.1 --model yolov5s --nms_mode nms_script --bs 4 --soc Ascend310P3 --quantify True  # nms_script
   bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend310P3 --quantify True  # nms_op
   ```
   （3）部分网络层量化后损失较大，可在 [simple_config.cfg](common/atc_cfg/simple_config.cfg) 中配置不需要量化的层名称，默认为空列表。[skip_layers.cfg](common/atc_cfg/skip_layers.cfg) 中提供了参考写法，通常网络的首尾卷积层量化损失大些，其他版本可以用[Netron](https://github.com/lutzroeder/netron)打开模型，查找不需要量化的层名称。


### 2 开始推理验证

1. 安装`ais-infer`推理工具  
   `ais-infer`工具获取及使用方式请点击查看 [[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

##### * 如果有多卡推理的需求，请跳过该步骤，om_val.py该脚本不支持多卡推理

2. 执行推理 & 精度验证  
   运行`om_val.py`推理OM模型，模型参数在[model.yaml](model.yaml)中设置，结果默认保存在`predictions.json`。
   ```
   python3 om_val.py --tag 6.1 --model=yolov5s_bs4.om --nms_mode nms_script --batch_size=4  # nms_script
   python3 om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4  # nms_op
   ```
   - 命令参数说明：
     -   `--tag`：模型版本，可选`[2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]`。
     -   `--model`：模型大小，可选`yolov5[n/s/m/l]`。
     -   `--nms_mode`：模型推理方式，可选`[nms_op/nms_script]`。
     -   `--batch_size`: 模型推理batch大小，默认`4`。
     -   `--cfg_file`：模型推理参数设置，默认读取文件[model.yaml](model.yaml)。

3. 性能验证  
   可使用`ais_infer`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 -m ais_bench --model=yolov5s_bs4.om --loop=1000 --batchsize=4  # nms_script
   python3 -m ais_bench --model=yolov5s_nms_bs4.om --loop=1000 --batchsize=4  # nms_op
   ```

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。
1. 方式一 nms后处理脚本（nms_script）

    | 模型tag |   芯片型号   | 最优Batch |    数据集    |         阈值       | 精度 (mAP@0.5) | OM模型性能 (fps) |
    |:------:|:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
    | 2.0   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     55.3     |   998.004   |
    | 3.1   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     56.5     |   772.670    |
    | 4.0   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     55.3     |   884.088    |
    | 5.0   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     55.5     |   881.139    |
    | 6.0   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     55.9     |   737.037    |
    | 6.1   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     56.9     |   739.736    |
    | 6.2   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     56.6     |   789.77     |
    | 7.0   | Ascend310P3 |     4      | coco val2017 |  conf=0.001 iou=0.6  |     56.9     |   1125.92     |

2. 方式二 nms后处理算子（nms_op）

    | 模型tag |   芯片型号   | 最优Batch |    数据集    |         阈值       | 精度 (mAP@0.5) | OM模型性能 (fps) |
    |:------:|:----------:|:-------------:|:------------------:|:------------:|:------------:|:--------------:|
    | 2.0   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     40.9     |   948.276    |
    | 3.1   | Ascend310P3 |     8      | coco val2017 | conf=0.4 iou=0.5   |     42.3     |   728.035    |
    | 4.0   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     40.5     |   862.770    |
    | 5.0   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     40.7     |   860.746    |
    | 6.0   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     41.2     |   876.578    |
    | 6.1   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     43.4     |   881.867    |
    | 6.2   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     43.4     |   780.09     |
    | 7.0   | Ascend310P3 |     8      | coco val2017 |  conf=0.4 iou=0.5  |     44.6     |   744.55     |


## duo卡推理

1. 数据预处理，将原始数据转换为模型输入的数据
   执行yolov5_preprocess.py脚本，完成预处理
   ```
   python3 yolov5_preprocess.py --data_path="./coco" --nms-mode nms_script
   ```
   - 命令参数说明：
     -   `--data_path`：coco数据集的路径
     -   `--nms_mode`：模型推理方式，可选`[nms_op/nms_script]`, 默认`nms_script`
    执行完后，会在当前目录下生成./prep_data文件夹用于储存预处理完的二进制数据，并且生成path_list.npy用于储存图片的路径，生成shapes_list.npy用于储存图片原始shape

2. 数据集推理
   目前ais_bench已经支持多卡推理，若执行下述命令报错，请重新安装最新ais_bench
   ```
   python3 -m ais_bench --m yolov5s_bs4.om --input ./prep_data --output ./results --device 0,1
   ```
   - 命令参数说明：
     -   `--m`：om模型的路径
     -   `--input`：预处理生成的./prep_data的路径
     -   `--output`：推理结果保存的地址，会在./results下生成以时间戳命名的文件夹
     -   `--device`：现支持多卡推理

3. 后处理和精度验证，将推理结果转换为字典并储存进json文件，用于计算精度
   ```
   python3 yolov5_postprocess.py --nms_mode nms_script --ground_truth_json "./coco/instances_val2017.json" --output "./results/2023_04_23-17_35_23" --onnx yolov5s.onnx
   ```
   - 命令参数说明：
     -   `--ground_truth_json`：om模型的路径
     -   `--output`：推理结果保存的路径，在./results下生成以时间戳命名的文件夹
     -   `--onnx`：为onnx模型路径
     -   `--nms_mode`：模型推理方式，可选`[nms_op/nms_script]`, 默认`nms_script` 


## aipp
* 说明：由于op方式受限较多，故下文插入aipp算子只考虑script方式
1. 在模型输入端插入aipp  
   运行`onnx2om.sh`导出`OM`模型。
   ```
   bash onnx2om.sh --tag 6.1 --model yolov5s --nms_mode nms_script --bs 4 --soc Ascend310P3 --with_aipp True # nms_script
   bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend310P3 --with_aipp True # nms_op
   ```

2. 由于插入aipp算子后，模型输入会发生改变，需要调用yolov5_preprocess_aipp.py生成预处理数据集prep_data_aipp
   ```
   python3 yolov5_preprocess_aipp.py --data_path "./coco"
   ```
   - 命令参数说明：
     -   `--data_path`：coco数据集所在路径。

3. 推理
   ```
   python3 -m ais_bench --m yolov5m_bs24_aipp.om --input ./prep_data_aipp --output ./results --device 0,1
   ```
   - 命令参数说明：
     -   `--input`：二进制数据集路径
     -   `--output`：推理结果保存目录
     -   `--output_dirname`：推理结果保存子目录
     -   `--device`：请下载最新ais_bech，目前已经支持多卡推理

4. 数据后处理
   ```
   python3 yolov5_postprocess.py --ground_truth_json "./coco/instances_val2017.json" --output "./results/2023_04_23-17_35_23" --onnx yolov5s.onnx
   ```
   - 命令参数说明：
     -   `--ground_truth_json`：om模型的路径
     -   `--output`：推理结果保存的路径，在./results下生成以时间戳命名的文件夹
     -   `--onnx`：为onnx模型路径

# FAQ
常见问题可参考 [FAQ](FAQ.md)