cur_path=`pwd`
batch_size=64
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
source ${cur_path}/test/env_npu.sh

sed -i "s|../datasets/coco|$data_path|g" data/coco.yaml

python3.7 -u train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size $batch_size --device 0 > $cur_path/test/train_full_1p.log 2>&1 &