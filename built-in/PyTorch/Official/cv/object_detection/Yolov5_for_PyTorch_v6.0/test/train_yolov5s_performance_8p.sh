#!/bin/bash

cur_path=`pwd`
model_name=yolov5s
batch_size=512

source ${cur_path}/test/env_npu.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=8

for i in $(seq 0 7)
do
    if [ -d ${cur_path}/test/output/${i} ];
	then
	    rm -rf ${cur_path}/test/output/${i}
		mkdir -p ${cur_path}/test/output/${i}
	else
	    mkdir -p ${cur_path}/test/output/${i}
	fi

	export RANK=$i
	export LOCAL_RANK=$i

    if [ $(uname -m) = "aarch64" ]
	then
		let p_start=0+24*i
	    let p_end=23+24*i
	    taskset -c $p_start-$p_end python3.7 train.py --data ./data/coco.yaml \
		                                           --cfg yolov5s.yaml \
		                                           --weights '' \
		                                           --batch-size $batch_size \
		                                           --epochs 2 \
		                                           --local_rank $i > $cur_path/test/output/${i}/train_8p_${i}.log 2>&1 &
	else
	    python3.7 train.py --data ./data/coco.yaml \
		                --cfg yolov5s.yaml \
		                --weights '' \
		                --batch-size $batch_size \
		                --epochs 2 \
		                --local_rank $i > $cur_path/test/output/${i}/train_8p_${i}.log 2>&1 &
    fi                        
done