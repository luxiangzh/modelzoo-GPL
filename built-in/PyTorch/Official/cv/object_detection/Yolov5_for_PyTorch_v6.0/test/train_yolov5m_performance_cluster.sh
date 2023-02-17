#!/bin/bash

# 集群训练命令样例
# bash test/train_yolov5m_performance_cluster.sh --data_path=dataset --nnodes=2 --node_rank=0 --master_addr=x.x.x.x --master_port=50000 

#网络名称,同目录名称,需要模型审视修改
Network="yolov5m_for_PyTorch_v6.0"

cur_path=`pwd`
model_name=yolov5m
batch_size=384
device_number=8
head_rank=0
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
    elif [[ $para == --device_number* ]];then
        device_number=`echo ${para#*=}`
    elif [[ $para == --head_rank* ]];then
        head_rank=`echo ${para#*=}`
   fi
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
     test_path_dir=${cur_path}
     cd ..
     cur_path=$(pwd)
else
     test_path_dir=${cur_path}/test
fi

# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
     source ${test_path_dir}/env_npu.sh
fi

#配置数据集路径
sed -i 's#train: .*#train: '${data_path}'/train2017.txt#' ${cur_path}/data/coco.yaml
sed -i 's#val: .*#val: '${data_path}'/val2017.txt#' ${cur_path}/data/coco.yaml
sed -i 's#test: .*#test: '${data_path}'/test2017.txt#' ${cur_path}/data/coco.yaml

export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export WORLD_SIZE=$((nnodes * device_number))

for i in $(seq ${head_rank} $((head_rank + device_number - 1)))
do
    if [ -d ${cur_path}/test/output/${i} ];
    then
        rm -rf ${cur_path}/test/output/${i}
        mkdir -p ${cur_path}/test/output/${i}
    else
        mkdir -p ${cur_path}/test/output/${i}
    fi
    rank=$((i + node_rank * device_number))
    export RANK=${rank}
    export LOCAL_RANK=$i

    if [ $(uname -m) = "aarch64" ]
    then
        let p_start=0+24*i
        let p_end=23+24*i
        taskset -c $p_start-$p_end python3.7 train.py --data ./data/coco.yaml \
                                                   --cfg yolov5m.yaml \
                                                   --weights '' \
                                                   --batch-size $batch_size \
                                                   --epochs 2 \
                                                   --local_rank $i > $cur_path/test/output/${i}/train_${WORLD_SIZE}p_${i}.log 2>&1 &
    else
        python3.7 train.py --data ./data/coco.yaml \
                        --cfg yolov5m.yaml \
                        --weights '' \
                        --batch-size $batch_size \
                        --epochs 2 \
                        --local_rank $i > $cur_path/test/output/${i}/train_${WORLD_SIZE}p_${i}.log 2>&1 &
    fi                        
done

wait


if [[ x"${master_addr}" == x"${HCCL_IF_IP}" ]];then

    # #训练结束时间，不需要修改
    end_time=$(date +%s)
    echo "end_time: ${end_time}"
    e2e_time=$(( $end_time - $start_time ))

    #最后一个迭代FPS值
    FPS=`grep -a 'FPS:'  ${cur_path}/test/output/0/train_${WORLD_SIZE}p_0.log|awk 'END {print}'| awk -F "[" '{print $5}'| awk -F "]" '{print $1}'| awk -F ":" '{print $2}'`

    #打印，不需要修改
    echo "Final Performance images/sec : $FPS"
    echo "E2E Training Duration sec : $e2e_time"

    #稳定性精度看护结果汇总
    #训练用例信息，不需要修改
    BatchSize=${batch_size}
    DeviceType=`uname -m`
    CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'perf'

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
    rm -rf $data_path/labels/*.cache
fi
