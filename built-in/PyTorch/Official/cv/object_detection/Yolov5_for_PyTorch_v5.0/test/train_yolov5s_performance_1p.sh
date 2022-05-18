#!/bin/bash

cur_path=`pwd`
model_name=yolov5s
batch_size=128 

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

python3.7 -u train.py --data ./data/coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --device npu --epochs 2 > $cur_path/test/train_perf_1p.log 2>&1 &