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

model_path="${cur_path}/models/${model_name}.yaml"

source ${cur_path}/test/env_npu.sh

export HCCL_WHITELIST_DISABLE=1

if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3.7 train_mp.py \
			--data coco.yaml \
		    --cfg $model_path \
		    --addr 127.0.0.1 \
		    --weights '' \
		    --batch-size $batch_size \
		    --local_rank $i \
		    --device npu \
		    --device-num 8 \
		    --epochs 2 > $cur_path/test/train_performance_8p.log 2>&1 &
	done
else
    python3.7 train.py \
		    --data coco.yaml \
		    --cfg $model_path \
		    --addr 127.0.0.1 \
		    --weights '' \
		    --batch-size $batch_size \
		    --local_rank 0 \
		    --device npu \
		    --device-num 8 \
		    --epochs 2 > $cur_path/test/train_performance_8p.log 2>&1 &
fi
