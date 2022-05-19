cur_path=`pwd`
batch_size=128
data_path=""
# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
source ${cur_path}/test/env_npu.sh

sed -i "s|../datasets/coco|$data_path|g" data/coco.yaml

python3.7 -u train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --device 0 > $cur_path/test/train_full_1p.log 2>&1 &