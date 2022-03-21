#!/bin/bash

cur_path=`pwd`
model_name=yolov5x
batch_size=32 

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

taskset -c 0-23 python3.7 train.py \
		--data data/coco.yaml \
		--cfg $model_path \
		--weights '' \
		--batch-size $batch_size \
		--device npu \
		--npu 0 > $cur_path/test/train_full_1p.log 2>&1 &