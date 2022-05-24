#!/bin/bash

cur_path=`pwd`
model_name=yolov5s
batch_size=128 

device_id=0
i=$device_id
if [ -d ${cur_path}/test/output/${i} ];
	then
	   rm -rf ${cur_path}/test/output/${i}
		mkdir -p ${cur_path}/test/output/${i}
	else
	   mkdir -p ${cur_path}/test/output/${i}
	fi

source ${cur_path}/test/env_npu.sh

python3.7 -u train.py --data ./data/coco.yaml \
                      --cfg yolov5s.yaml \
                     --weights '' \
                     --batch-size $batch_size \
                     --device $device_id \
                     --epochs 2 > $cur_path/test/output/${i}/train_perf_1p.log 2>&1 &