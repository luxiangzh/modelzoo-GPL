# 1.版本说明
yolov5版本Tags=v5.0, python版本为3.7.5

# 2.准备数据集

## 2.1下载coco2017数据集，并解压，解压后目录如下所示：

```
├── coco_data: #根目录
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

## 2.2 生成yolov5专用标注文件

（1）将代码仓中cocofile/coco2yolo.py和cocofile/coco_class.txt拷贝到coco的实际路径/data/to/coco

（2）运行coco2yolo.py

```
python3 coco2yolo.py
```

（3）运行上述脚本后，将在/data/to/coco生成train2017.txt和val2017.txt

# 3.配置数据集路径

建立软连接：

```
ln -s /data/to/coco coco
```

# 4.GPU,CPU依赖
按照requirements-GPU.txt安装python依赖包  

# 5.NPU依赖
按照requirements.txt安装python依赖包，还需安装(NPU-driver.run, NPU-firmware.run, NPU-toolkit.run, torch-ascend.whl, apex.whl)

- 编译安装torchvision

  ***为了更快的推理性能，请编译安装而非直接安装torchvision***

   ```
    git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
    cd vision
    python setup.py bdist_wheel
    pip3 install dist/*.whl
   ```

# 6.编译安装Opencv-python

为了获得最好的图像处理性能，***请编译安装opencv-python而非直接安装***。编译安装步骤如下：

```
# 以python3.7.5版本为例
export GIT_SSL_NO_VERIFY=true
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build
cd build
cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m -D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..
make -j$nproc
make install
```

# 7.NPU 单机单卡训练指令  
yolov5s:

```
bash test/train_yolov5s_full_1p.sh  
```

# 8.NPU 单机八卡训练指令  
yolov5s:

```
bash test/train_yolov5s_full_8p.sh 
```

# 9.NPU evalution指令  

```
python3 test.py --data ./data/coco.yaml --img-size 640 --weight 'yolov5_0.pt' --batch-size 32 --device npu --local_rank 0
```

# 13.导出onnx指令
python export_onnx.py --weights ./xxx.pt --img-size 640 --batch-size 1

# 14.训练结果展示

**表 1**  训练结果展示表

|   NAME   | Accuracy | FPS  | Torch_Version | CPU |
| :------: | :---: | :--: | :-------: | :------: |
|  8p-NPU  |  34.1  | 1781.91 |      1.8      |      arm      |
|  1p-NPU  |  - | 219.36 |      1.8      |      非arm      |
|  8p-NPU  | -  | 1335.43 |      1.8      |      非arm      |