# Yolov5_6.1 模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch_1.8(NPU版本)
* apex(NPU版本)

## Dataset Prepare
1. 下载coco数据集，包含图片、annotations、labels
    图片、annotations: 从coco官方网站获取
    labels: https://drive.google.com/uc?export=download&id=1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L
2. 将coco数据集放于工程根目录下
    coco目录结构如下：
	```
    coco
       |-- annotations
       |-- images
          |-- train2017
          |-- val2017   
       |-- labels
          |-- train2017
          |-- val2017
	```	  
## 安装依赖
pip3.7 install -r requirements.txt


## Train Model
### 单卡
1. 运行 train_1p.sh
```
bash test/train_full_1p.sh --data_path=real_path
```
若需要指定训练使用的卡号, 可修改train_1p.sh文件 "--device 0"配置项,其中卡号为0-7

### 8卡
1. 运行 train_8p.sh
```
bash test/train_full_8p.s --data_path=real_path
```

## 验证

```
chmod +x test.sh
./test.sh
```