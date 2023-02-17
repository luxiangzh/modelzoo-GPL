#网络名称,同目录名称,需要模型审视修改
Network="Complex_yolov4"
# 数据集路径,保持为空,不需要修改
data_path=""
#当前路径,不需要修改
cur_path=`pwd`
#训练batch_size,需要模型审视修改
batch_size=64
RANK_SIZE=8

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [ ! -d './dataset/kitti/training' ]
then
	ln -s ${data_path}/training/ ./dataset/kitti/training
	ln -s ${data_path}/testing/ ./dataset/kitti/testing
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#创建DeviceID输出目录，不需要修改
device_id=0
if [ ! -d ${cur_path}/test/output/${device_id} ];then
    mkdir -p ${cur_path}/test/output/$device_id/
fi

source test/env_npu.sh
cd src

python3 -m torch.distributed.launch --nproc_per_node=${RANK_SIZE} train.py \
     --dist-url 'tcp://127.0.0.1:29500' \
     --dist-backend 'hccl' \
     --multiprocessing-distributed \
     --batch_size ${batch_size} > ${cur_path}/test/output/$device_id/train_acc_8p.log 2>&1

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
step_time=`grep -a 'Epoch:'  ${cur_path}/test/output/$device_id/train_acc_8p.log|awk 'END {print}'| awk -F " " '{print $8}'| cut -d ')' -f1`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${step_time}'}'`

#最后一个迭代loss值
loss=`grep -a 'Epoch:'  ${cur_path}/test/output/$device_id/train_acc_8p.log|awk 'END {print}'| awk -F " " '{print $15}'` | cut -d '(' -f2 | cut -d ')' -f1

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "ActualLoss : ${loss}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'loss:'  ${cur_path}/test/output/$device_id/train_acc_8p.log | awk -F "Loss" '{print $NF}' | awk -F " " '{print $16}' | cut -d '(' -f2 | cut -d ')' -f1 >> $cur_path/test/output/$device_id/train_${CaseName}_loss.txt

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$device_id/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "ActualLoss = ${loss}" >> $cur_path/test/output/$device_id/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$device_id/${CaseName}.log
