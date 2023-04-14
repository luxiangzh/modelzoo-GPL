# Yolov3_ultralytics for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLOv3是在 COCO 数据集上预训练的对象检测架构和模型系列，代表了 Ultralytics 对未来视觉 AI 方法的开源研究，融合了数千小时研究和开发过程中积累的经验教训和最佳实践，Tags=v9.6.0。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov3/tree/v9.6.0
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo-GPL.git
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

- 编译安装torchvision。

  ***为了更快的推理性能，请编译安装而非直接安装torchvision***

   ```
    git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
    cd vision
    python setup.py bdist_wheel
    pip3 install dist/*.whl
   ```

- 编译安装Opencv-python（可选）

  为了获得最好的图像处理性能，***请编译安装opencv-python而非直接安装***。编译安装步骤如下：

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

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括coco2017、voc，其中voc数据集可通过训练脚本自动获取，将数据集上传到服务器任意路径下并解压。

   （1）以coco2017数据集为例，数据集目录结构参考如下所示。

   ```
    ├── coco2017
    │   ├── annotations
    │          ├── captions_train2017.json
    │          ├── captions_val2017.json
    │          ├── instances_train2017.json
    │          ├── instances_val2017.json
    │          ├── person_keypoints_train2017.json
    │          ├── person_keypoints_val2017.json
    │   ├── train2017
    │          ├── 000000000009.jpg
    │          ├── 000000000025.jpg
    │          ├── ......
    │   ├── val2017
    │          ├── 000000000139.jpg
    │          ├── 000000000285.jpg
    │          ├── ......
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

   （2）以voc2012数据集为例，数据集目录结构参考如下所示。

   ```
    ├── VOC2012
    │   ├── Annotations
    │   ├── ImageSets
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationObject
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
   下载的原始coco数据集需要进行数据预处理，生成coco专用标注文件，操作方式如下：

   （1）将代码仓中cocofile/coco2yolo.py和cocofile/coco_class.txt拷贝到coco数据集的实际路径/data/to/coco。

   （2）运行coco2yolo.py。运行该脚本后，将在/data/to/coco生成train2017.txt和val2017.txt。
   
   ```
   python3 coco2yolo.py
   ```

# 开始训练

## 训练模型
  该模型可以在voc数据集和coco数据集上进行训练，请用户根据实际需要进行选择。
- VOC数据集（可选）

   脚本命令默认为VOC数据集，执行训练脚本会自动下载VOC2012和VOC2007数据集，请确保网络通畅。其他配置的训练可使用相同方式启动。

- COCO2017数据集（可选）

   若需要在coco数据集上进行训练，则需**在下述启动命令后加入'--datasets=coco --data_path=/data/to/coco'**
   例如： bash test/train_full_1p.sh  --model_name=yolov3 --batch_size=64 --img_size=320 --datasets=coco --data_path=/data/to/coco  #coco数据集路径。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   yolov3-320*320

   ```
    bash test/train_full_1p.sh --model_name=yolov3 --batch_size=64 --img_size=320  # 单卡精度训练
    bash test/train_full_8p.sh --model_name=yolov3 --batch_size=512 --img_size=320 # 8卡精度训练
   ```

   yolov3-608*608

   ```
    bash test/train_full_1p.sh --model_name=yolov3 --batch_size=32 --img_size=608  # 单卡精度训练
    bash test/train_full_8p.sh --model_name=yolov3 --batch_size=256 --img_size=608  # 8卡精度训练
   ```

   yolov3-640*640

   ```
    bash test/train_full_1p.sh --model_name=yolov3 --batch_size=32 --img_size=640  # 单卡精度训练
    bash test/train_full_8p.sh --model_name=yolov3 --batch_size=256 --img_size=640  # 8卡精度训练
   ```

   yolov3_spp-640*640

   ```
    bash test/train_full_1p.sh --model_name=yolov3-spp --batch_size=32 --img_size=640  # 单卡精度训练
    bash test/train_full_8p.sh --model_name=yolov3-spp --batch_size=256 --img_size=640  # 8卡精度训练
   ```

   yolov3_tiny-640*640

   ```
    bash test/train_full_1p.sh --model_name=yolov3-tiny --batch_size=64 --img_size=640  # 单卡精度训练
    bash test/train_full_8p.sh --model_name=yolov3-tiny --batch_size=512 --img_size=640  # 8卡精度训练
   ```

   --data_path参数（可选）填写数据集路径，需写到数据集的一级目录。
   
   --datasets参数（可选）填写数据集名称。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data                              //数据集路径
   --workers                           //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --name                              //保存的文件名
   --save-period                       //保存周期
   --noval                             //设置仅验证最后一个epoch
   --weight_decay                      //权重衰减，默认：0.0005
   --loss-scale                        //混合精度loss scale大小
   ```

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version | Dataset | Model_Name  | Img_Size | CPU  |
| :----: | :---: | :-----: | :----: | :------: | :-----------: | :-----: | :---------: | -------- | ---- |
| 1p-NPU |   -   | 151.68  |   2    |    O1    |      1.8      |   voc   | yolov3-tiny | 640*640  | ARM  |
| 8p-NPU | 0.248 | 1013.76 |  300   |    O1    |      1.8      |   voc   | yolov3-tiny | 640*640  | ARM  |


|  NAME  |  mAR  |   FPS   | Epochs | AMP_Type | Torch_Version | Dataset  | Model_Name  | Img_Size | CPU  |
| :----: | :---: | :-----: | :----: | :------: | :-----------: | :------: | :---------: | -------- | ---- |
| 1p-NPU |   -   | 154.24  |   2    |    O1    |      1.8      | coco2017 | yolov3-tiny | 640*640  | ARM  |
| 8p-NPU | 0.385 | 1264.64 |  300   |    O1    |      1.8      | coco2017 | yolov3-tiny | 640*640  | ARM  |


# 版本说明

## 变更

2023.04.14：更新内容，重新发布。

## FAQ

无。