# 1.版本说明
yolov3,原始链接https://github.com/ultralytics/yolov3/tree/v9.6.0，Tags=v9.6.0

# 2.安装依赖

按照requirements.txt安装python依赖包  

# 3.（可选）编译安装Opencv-python

为了获得最好的图像处理性能，***请编译安装opencv-python而非直接安装***。编译安装步骤如下：

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

# 4.VOC数据集训练方法

yolov3-320*320-单卡

```
bash train_yolov3_320x320_1p.sh
```

yolov3-320*320-8卡

```
bash train_yolov3_320x320_8p.sh
```

**脚本会自动下载VOC2012和VOC2007数据集，请确保网络通畅。其他配置的训练可使用相同方式启动。**