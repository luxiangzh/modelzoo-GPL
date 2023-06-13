# 1.版本说明
yolov3版本Tags=v7.0, python版本为3.7.5

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

## 2.2 生成yolov3专用标注文件

（1）将代码仓中coco/coco2yolo.py和coco/coco_class.txt拷贝到coco_data**根目录**

（2）运行coco2yolo.py

```
python3 coco2yolo.py
```

（3）运行上述脚本后，将在coco_data**根目录**生成train2017.txt和val2017.txt

上述步骤可以等价替换为下载且解压github的资源coco2017labels.zip，位置是https://github.com/ultralytics/yolov3/releases/tag/v1.0
# 3.配置数据集路径

记得修改data/coco2017.data文件中的train字段和val字段的路径，分别指向上一节生成的train2017.txt和val2017.txt。
保持最后的图片路径和代码路径为：
```
├── coco: #根目录
     ├── images #图片
              ├──train2017 #训练集图片约118287
              ├── val2017 #验证集图片，约5000张
     ├── labels #标签
              ├──train2017 #训练集标签，有117266个txt文件，有图片没打标
              ├── val2017 #验证集标签，有4952个txt文件，有图片没打标
     └── annotations #标注目录
              ├── instances_train2017.json #对应目标检测、分割任务的训练集标注文件
              ├── instances_val2017.json #对应目标检测、分割任务的验证集标注文件
              ├── captions_train2017.json 
              ├── captions_val2017.json 
              ├── person_keypoints_train2017.json 
              └── person_keypoints_val2017.json
     ├── train2017.txt #训练图片路径
     ├── val2017.txt #测试图片路径
├── code: #代码根目录
     ├── test #脚本目录
              ├── output #输出目录
              ├── train_eval_8p.sh # 输出目录
              ├── train_full_1p.sh # 1卡训练脚本
              ├── train_full_8p.sh # 8卡训练脚本
              ├── train_performance_1p.sh # 1卡性能脚本
              ├── train_performance_8p.sh # 8卡性能脚本
     ├── cfg # 配置目录
     ├── data # 数据集配置
```


# 4.依赖包

按照requirements.txt安装python依赖包 ，还需安装(NPU-driver.run, NPU-firmware.run, NPU-toolkit.run, torch-ascend.whl, apex.whl)

# 5.编译安装Opencv-python

为了获得最好的图像处理性能，***请编译安装opencv-python而非直接安装***（不熟悉者用pip或者conda安装也可以，只要import cv2 成功）。编译安装步骤如下：

```
export GIT_SSL_NO_VERIFY=true
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build
cd build
cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m -D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..
make -j$nproc
make install
```

# 6.NPU 单机单卡训练指令  
```
cd test
mkdir output # 如果没有这个文件夹
bash train_full_1p.sh   # test目录下 
```
# 7.NPU 单机八卡训练指令  
```
cd test
mkdir output # 如果没有这个文件夹
bash train_full_8p.sh   # test目录下 
```
# 8.NPU evalution指令  

```
cd test
bash test.sh   # test目录下 
```
# 9.公网地址说明
```
代码涉及公网地址参考 public_address_statement.md
```