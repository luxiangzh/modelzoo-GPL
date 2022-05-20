#!/bin/bash

cur_path=`pwd`
model_name=yolov5s
batch_size=512

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   fi
done

source ${cur_path}/test/env_npu.sh

if [ $(uname -m) = "aarch64" ]
then
	export MASTER_ADDR=127.0.0.1
	export MASTER_PORT=29500
	export WORLD_SIZE=8
	for i in $(seq 0 7)
	do 
	export RANK=$i
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3.7 train.py --data ./data/coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --local_rank $i --device npu --epochs 2 > $cur_path/test/train_perf_8p.log 2>&1 &
	done
else
    nohup python3.7 -m torch.distributed.launch --nproc_per_node=8 train.py --data ./data/coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --device npu --epochs 2 &
fi
