#!/bin/bash

###############ָ��ѵ���ű�ִ��·��###############
# cd����test�ļ���ͬ�㼶Ŀ¼��ִ�нű�����߼����ԣ�test_path_dirΪ����test�ļ��е�·��
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
RANK_SIZE=1
batch_size=32
img_size=640
model_name=yolov3-spp
# ���ݼ�·��,����Ϊ��,����Ҫ�޸�
data_path=""
datasets="voc"
#ѵ��epochs
epochs=2
#�������� yolov3_spp-640*640
Network="Yolov3_ID4106_for_PyTorch"

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
   elif [[ $para == --datasets* ]];then
        datasets=`echo ${para#*=}`
   elif [[ $para == --img_size* ]];then
      	img_size=`echo ${para#*=}`
   elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source ${test_path_dir}/set_conda.sh --conda_name=$conda_name
        source activate $conda_name
   fi
done

# ���ݼ�����������
if [ ${datasets} == "coco" ];then
  echo "data_path is: ${data_path}"
  if [ ! -d './data/coco' ]
  then
    ln -s ${data_path} ./data/coco
  fi
fi

# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
    if [ ${datasets} == "voc" ];then
      ln -nsf ${data_path} .
    fi
else
  cp -r ${data_path}/VOC.tar.gz ${cur_path}
  tar -xavf VOC.tar.gz
fi

# ָ��ѵ����ʹ�õ�npu device��id
device_id=0

# У���Ƿ�ָ����device_id,�ֶ�̬����device_id���ֶ�ָ��device_id,�˴�����Ҫ�޸�
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################������־���Ŀ¼������Ҫ�޸�#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)

nohup taskset -c 0-23 python3.7 train.py --data ${datasets}.yaml --cfg ${model_name}.yaml --epochs ${epochs} --weights '' --batch-size ${batch_size} --noval --img-size ${img_size} --local_rank ${device_id} >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#�����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
#�������FPS����Ҫģ�������޸�
fps=`grep -a 'it/s'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk -F "i" '{print $1}'|tail -n 1`
FPS=`echo "${batch_size} * ${fps}" |bc`
#��ӡ������Ҫ�޸�
echo "Final Performance images/sec : $FPS"

#���ѵ������,��Ҫģ�������޸�
#train_accuracy=`grep -a 'all' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|tail -1|awk -F ' ' '{print $NF}'`

#��ӡ������Ҫ�޸�
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#���ܿ����������
#ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##��ȡ�������ݣ�����Ҫ�޸�
#������
ActualFPS=${FPS}
#������ѵ��ʱ��
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`


#�ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
#�������ݼ�
if [ x"${etp_flag}" == x"true" ];then
    rm -rf ${cur_path}/VOC.tar.gz ${cur_path}/VOC
fi