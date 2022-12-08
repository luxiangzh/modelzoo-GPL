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

# 指定训练所使用的npu device卡id
device_id=0

batch_size=32

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

for para in $*
do
   if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   fi
done

export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3 
export DYNAMIC_OP="ADD#MUL"

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

cd ${cur_path}
python3.7 test.py --data data/coco.yaml --coco_instance_path  ../coco/annotations/instances_val2017.json --img-size 672 --weight 'yolov5_0.pt' --batch-size ${batch_size} --device npu --npu $ASCEND_DEVICE_ID
