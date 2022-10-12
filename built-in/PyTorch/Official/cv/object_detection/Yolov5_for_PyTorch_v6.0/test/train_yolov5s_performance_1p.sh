#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov5s_v6.0"

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

#创建DeviceID输出目录，不需要修改
ASCEND_DEVICE_ID=0
echo "device id is ${ASCEND_DEVICE_ID}"
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
	then
	   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
		mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	else
	   mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

source ${cur_path}/test/env_npu.sh

python3.7 -u train.py --data ./data/coco.yaml \
                      --cfg yolov5s.yaml \
                     --weights '' \
                     --batch-size $batch_size \
                     --device 0 \
                     --epochs 2 > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_perf_1p.log 2>&1 &

wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
FPS=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_perf_1p.log|awk 'END {print}'| awk -F "[" '{print $5}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}'`

#打印，不需要修改
echo "ActualFPS : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}

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