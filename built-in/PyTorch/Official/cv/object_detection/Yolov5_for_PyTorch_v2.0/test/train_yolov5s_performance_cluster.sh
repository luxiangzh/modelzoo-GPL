#!/bin/bash

# 集群训练命令样例
# bash test/train_yolov5s_performance_cluster.sh --data_path=dataset --nnodes=2 --node_rank=0 --master_addr=x.x.x.x --master_port=50000 

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
#集合通信参数,不需要修改
export RANK_SIZE=8
model_name=yolov5s
batch_size=512

# 数据集路径,保持为空,不需要修改
data_path=""
#网络名称
Network="Yolov5_for_PyTorch"
device_number=8
for para in $*
do
   if [[ $para == --model_name* ]];then
        model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
   elif [[ $para == --node_rank* ]];then
        node_rank=`echo ${para#*=}`
   elif [[ $para == --master_addr* ]];then
        master_addr=`echo ${para#*=}`
   elif [[ $para == --master_port* ]];then
        master_port=`echo ${para#*=}`
   elif [[ $para == --nnodes* ]];then
        nnodes=`echo ${para#*=}`
   fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#配置数据集路径
sed -i 's#train: .*#train: '${data_path}'/train2017.txt#' ${cur_path}/data/coco.yaml
sed -i 's#val: .*#val: '${data_path}'/val2017.txt#' ${cur_path}/data/coco.yaml
sed -i 's#test: .*#test: '${data_path}'/test2017.txt#' ${cur_path}/data/coco.yaml
sed -i 's#python3.7 test.py --data data/coco.yaml --coco_instance_path.*#python3.7 test.py --data data/coco.yaml --coco_instance_path  .'${data_path}'/annotations/instances_val2017.json --img-size 672 --weight 'yolov5_0.pt' --batch-size 32 --device npu --npu 0 #' ${test_path_dir}/train_eval_1p.sh


model_path="${cur_path}/models/${model_name}.yaml"


#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

ASCEND_DEVICE_ID=0
#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

#训练开始时间，不需要修改
start_time=$(date +%s)

if [ $(uname -m) = "aarch64" ]
then
    for i in $(seq 0 7)
    do 
    rank=$((i + node_rank * device_number))
    let p_start=0+24*i
    let p_end=23+24*i
    taskset -c $p_start-$p_end $CMD python3.7 train_mp.py \
            --data coco.yaml \
            --cfg $model_path \
            --addr ${master_addr} \
            --weights '' \
            --batch-size $batch_size \
            --local_rank ${rank} \
            --npu $i \
            --device npu \
            --world-size ${nnodes} \
            --device-num 8 \
            --epochs 2 > $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
else
    nohup python3.7 train.py \
            --data coco.yaml \
            --cfg $model_path \
            --addr ${master_addr} \
            --weights '' \
            --batch-size $batch_size \
            --local_rank ${rank} \
            --device npu \
            --npu $i \
            --device-num 8 \
            --world-size=${nnodes} \
            --epochs 2 > $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
fi
wait

total_device=$((nndodes * device_number))
if [[ x"${master_addr}" == x"${HCCL_IF_IP}" ]];then
    #训练结束时间，不需要修改
    end_time=$(date +%s)
    e2e_time=$(( $end_time - $start_time ))

    #结果打印，不需要修改
    echo "------------------ Final result ------------------"
    #输出性能FPS，需要模型审视修改
    FPS=`grep -a 'FPS'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS:" '{print $2}'|awk -F "]" '{print $1}'|tail -1`
    #打印，不需要修改
    echo "Final Performance images/sec : $FPS"

    #输出训练精度,需要模型审视修改
    #train_accuracy=`grep -a '* Prec@1' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk '{print $3}'`

    #打印，不需要修改
    #echo "Final Train Accuracy : ${train_accuracy}"
    echo "E2E Training Duration sec : $e2e_time"

    #性能看护结果汇总
    #训练用例信息，不需要修改
    BatchSize=${batch_size}
    DeviceType=`uname -m`
    CaseName=${Network}_bs${BatchSize}_${total_device}'p'_'perf'

    ##获取性能数据，不需要修改
    #吞吐量
    ActualFPS=${FPS}
    #单迭代训练时长
    TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

    #从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
    grep FPS $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $8}'|awk -F ":" '{print $2}' >> $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

    #最后一个迭代loss值，不需要修改
    ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

    #关键信息打印到${CaseName}.log中，不需要修改
    echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
fi