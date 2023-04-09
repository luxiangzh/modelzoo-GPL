#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov5m_v6.0"

cur_path=`pwd`
model_name=yolov5m
batch_size=384

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
		                                           --cfg yolov5m.yaml \
		                                           --weights '' \
		                       --workers 64 \
												   --hyp hyp.scratch-high.yaml \
		                                           --batch-size $batch_size \
												   --epochs 400 \
		                                           --local_rank $i > $cur_path/test/output/${i}/train_8p_${i}.log 2>&1 &
	else
	    python3.7 train.py --data ./data/coco.yaml \
		                --cfg yolov5m.yaml \
		                --weights '' \
						--hyp hyp.scratch-high.yaml \
		                --batch-size $batch_size \
						--epochs 400 \
		                --local_rank $i > $cur_path/test/output/${i}/train_8p_${i}.log 2>&1 &
	fi
done

wait



#训练后进行eval显示精度
python3.7 val.py --data ./data/coco.yaml --conf-thres 0.0005 --iou-thres 0.50 --img-size 640 --weight 'yolov5.pt' --batch-size 128 --device $ASCEND_DEVICE_ID > ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_8p.log 2>&1 &

wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))
#最后一个迭代FPS值
FPS=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_8p_0.log|awk 'END {print}'| awk -F "[" '{print $5}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}'`

#取acc值
acc=`grep -a 'IoU=0.50     ' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_8p.log|grep 'Average Precision'|awk 'NR==1'| awk -F " " '{print $13}'`

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


#训练总时长，训练中每个step所使用的时间之和
total_trainning_time=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_8p_0.log | awk -F "[" '{print $7}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}' | awk '{t+=$1} END {print t}'`

#验证总时长，整个验证过程使用的时间
eval_time_str=`grep "mAP@" ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_8p.log`
total_eval_time=`echo ${eval_time_str##*[} | awk -F '<' '{print $1}'`
time_array=(${total_eval_time//:/ })
time_array_count=`echo ${#time_array[@]}`
if [ x"${time_array_count}" == x"3" ];then
    let h_second=${time_array[0]}*3600
    min_0=${time_array[1]}
    if [ x"${min_0:0:1}" == x"0" ];then
        min=${min_0:1:1}
        let m_second=${min}*60
    else
        let m_second=${min}*60
    fi
    let total_eval_time=${h_second}+${m_second}+${time_array[2]}
elif [ x"${time_array_count}" == x"2" ];then
    min_0=${time_array[0]}
    if [ x"${min_0:0:1}" == x"0" ];then
        min=${min_0:1:1}
        let m_second=${min}*60
    else
        let m_second=${min}*60
    fi
    let total_eval_time=${m_second}+${time_array[1]}
fi
calculate_eval_time=`grep -i "DONE" ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_acc_8p.log | awk -F "=" '{print $2}' | awk -F "s" '{print $1}' | awk '{t+=$1} END {print t}'`
total_eval_time=`echo "scale=2;${total_eval_time}+${calculate_eval_time}"|bc`

#训练最大，训练过程中全部step中的最大性能值
training_maxmum_fps=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_8p_0.log | awk -F "[" '{print $5}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}' |awk '$0>fps {fps=$0}END{print fps}'`

#训练平均， 训练的总样本数除以训练时长
total_step=`grep -a 'FPS:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_8p_0.log | wc -l`
let total_sambles=${batch_size}*${total_step}
training_average=`awk 'BEGIN{printf "%.2f\n", '${total_sambles}'/'${total_trainning_time}'}'`

#端到端平均
end_to_end_verage=`awk 'BEGIN{printf "%.2f\n", '${total_sambles}'/'${e2e_time}'}'`
echo "total time: ${e2e_time}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log
echo "total_training_time: ${total_trainning_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log
echo "total_eval_time: ${total_eval_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log
echo "training maxmum images/sec: ${training_maxmum_fps}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log
echo "training average images/sec: ${training_average}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log
echo "end to end average images/sec: ${end_to_end_verage}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_report.log




