#!/bin/bash

cur_path=`pwd`

batch_size=512
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

source ${cur_path}/test/env_npu.sh

sed -i "s|../datasets/coco|$data_path|g" data/coco.yaml

if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3.7 -u train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --local_rank $i --device_num 8 --epochs 2 > $cur_path/test/train_performance_8p.log 2>&1 &
	done
else
    nohup python3.7 -u -m torch.distributed.launch --nproc_per_node=8 train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --device_num 8 --epochs 2 > $cur_path/test/train_performance_8p.log 2>&1 &
fi