# 1.版本说明
yolov5版本Tags=v2.0, python版本为3.7.5

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

（1）将代码仓中coco/coco2yolo.py和coco/coco_class.txt拷贝到coco_data**根目录**

（2）运行coco2yolo.py

```
python3 coco2yolo.py
```

（3）运行上述脚本后，将在coco_data**根目录**生成train2017.txt和val2017.txt


# 3.GPU,CPU依赖
按照requirements-GPU.txt安装python依赖包  

# 4.NPU依赖
按照requirements.txt安装python依赖包，还需安装(NPU-driver.run, NPU-firmware.run, NPU-toolkit.run, torch-ascend.whl, apex.whl)
pillow建议安装较新版本

- 编译安装torchvision

  ***为了更快的推理性能，请编译安装而非直接安装torchvision***

   ```
    git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
    cd vision
    python setup.py bdist_wheel
    pip3 install dist/*.whl
   ```

# 5.编译安装Opencv-python

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

# 6.NPU 单机单卡训练指令  
yolov5s:

```
bash test/train_yolov5s_full_1p.sh  --data_path=数据集路径  
```


yolov5x:

```
bash test/train_yolov5x_full_1p.sh  --data_path=数据集路径  
```


# 7.1 NPU 单机八卡训练指令  
yolov5s:

```
bash test/train_yolov5s_full_8p.sh  --data_path=数据集路径
```


yolov5x:

```
bash test/train_yolov5x_full_8p.sh  --data_path=数据集路径  
```

# 7.2 NPU 多机多卡训练指令  
yolov5s:

```
bash test/train_yolov5s_performance_cluster.sh  --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
```
ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练

# 7.3 NPU 单机单卡评测指令  
yolov5s:

```
bash test/train_eval_1p.sh  --data_path=数据集路径
```

# 8.GPU 单机单卡训练指令  
python train.py --data coco.yaml --cfg yolov5x.yaml --weights '' --batch-size 32 --device 0  

# 9.GPU 单机八卡训练指令  
python -m torch.distributed.launch --nproc_per_node 8 train.py --data coco.yaml --cfg yolov5x.yaml --weights '' --batch-size 256  

# 10.CPU指令  
python train.py --data coco.yaml --cfg yolov5x.yaml --weights '' --batch-size 32 --device cpu  

# 11.导出onnx指令
python export_onnx.py --weights ./xxx.pt --img-size 640 --batch-size 1

# 12.Inference指令
python detect.py --source file.jpg --weights 'yolov5l.pt' --device npu --data coco.yaml（可选）

注：若保存的模型文件中无`names`字段，须传入data参数

# 13. 训练结果展示

表1 yolov5s训练结果展示表

|  NAME  | Accuracy |  FPS   | Torch_Version | CPU  |
| :----: | :------: | :----: | :-----------: | :--: |
| 8p-NPU |  34.02   | 1686.7 |      1.8      | arm  |

表2 yolov5x训练结果展示表

|   NAME   | Accuracy |  FPS   | Torch_Version | CPU  |
| :------: | :---: | :----: | :-----------: | :--: |
|  8p-NPU  |   48.2   | 665.6 |      1.8      | arm  |