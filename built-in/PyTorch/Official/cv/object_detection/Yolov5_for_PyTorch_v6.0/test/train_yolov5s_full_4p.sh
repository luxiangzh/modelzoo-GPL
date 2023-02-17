#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov5s_v6.0"

cur_path=`pwd`
model_name=yolov5s
batch_size=256

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   fi
done

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
ASCEND_DEVICE_ID=0
echo "device id is ${ASCEND_DEVICE_ID}"

source ${cur_path}/test/env_npu.sh

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=4

for i in $(seq 0 3)
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
		                                           --local_rank $i > $cur_path/test/output/${i}/train_4p_${i}.log 2>&1 &
	else
	    python3.7 train.py --data ./data/coco.yaml \
		                --cfg yolov5s.yaml \
		                --weights '' \
		                --batch-size $batch_size \
		                --local_rank $i > $cur_path/test/output/${i}/train_4p_${i}.log 2>&1 &
	fi
done

wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#训练后进行eval显示精度
python3.7 val.py --data ./data/coco.yaml --img-size 640 --weight 'yolov5.pt' --batch-size 128 --device 0 --half > ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_4p.log 2>&1 &

wait

#最后一个迭代FPS值
FPS=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_4p_0.log|awk 'END {print}'| awk -F "[" '{print $5}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}'`

#取acc值
acc=`grep -a 'IoU=0.50:0.95' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_4p.log|grep 'Average Precision'|awk 'NR==1'| awk -F " " '{print $13}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : $acc"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
