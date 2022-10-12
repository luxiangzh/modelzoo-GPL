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

# 4.训练方法

## 4.1 VOC数据集

**脚本命令默认为VOC数据集，执行脚本会自动下载VOC2012和VOC2007数据集，请确保网络通畅。其他配置的训练可使用相同方式启动。**

## 4.2 COCO2017数据集

**在下述命令后加入'--datasets=coco --data_path=coco数据集路径'**
例如： bash test/train_full_1p.sh  --model_name=yolov3 --batch_size=64 --img_size=320 --datasets=coco --data_path=/The/Real/Path/To/Coco #coco数据集路径

yolov3-320*320

```
    bash test/train_full_1p.sh  --model_name=yolov3 --batch_size=64 --img_size=320         #单卡精度训练
    bash test/train_full_8p.sh  --model_name=yolov3 --batch_size=512 --img_size=320        #八卡精度训练
```

yolov3-608*608

```
    bash test/train_full_1p.sh  --model_name=yolov3 --batch_size=32 --img_size=608         #单卡精度训练
    bash test/train_full_8p.sh  --model_name=yolov3 --batch_size=256 --img_size=608        #八卡精度训练
```

yolov3-640*640

```
    bash test/train_full_1p.sh  --model_name=yolov3 --batch_size=32 --img_size=640         #单卡精度训练
    bash test/train_full_8p.sh  --model_name=yolov3 --batch_size=256 --img_size=640        #八卡精度训练
```

yolov3_spp-640*640

```
    bash test/train_full_1p.sh  --model_name=yolov3-spp --batch_size=32 --img_size=640     #单卡精度训练
    bash test/train_full_8p.sh  --model_name=yolov3-spp --batch_size=256 --img_size=640    #八卡精度训练
```

yolov3_tiny-640*640

```
    bash test/train_full_1p.sh  --model_name=yolov3-tiny --batch_size=64 --img_size=640     #单卡精度训练
    bash test/train_full_8p.sh  --model_name=yolov3-tiny --batch_size=512 --img_size=640    #八卡精度训练
```

