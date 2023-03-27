#网络名称,同目录名称,需要模型审视修改
Network="Complex_yolov4"
# 数据集路径,保持为空,不需要修改
data_path=""
#当前路径,不需要修改
cur_path=`pwd`
#训练batch_size,需要模型审视修改
batch_size=64
RANK_SIZE=1

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

python3 evaluate.py --local_rank $device_id --pretrained_path ../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_best.pth > ${cur_path}/test/output/$device_id/train_eval_1p.log 2>&1

#评估结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#评估的精度值
map=`grep -a 'mAP:' ${cur_path}/test/output/$device_id/train_eval_1p.log | awk -F " " '{print $2}'`

echo "Final Train Accuracy : $map"
