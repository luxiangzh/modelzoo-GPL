cur_path=`pwd`
source ${cur_path}/test/env_npu.sh
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
sed -i "s|../datasets/coco|$data_path|g" data/coco.yaml

python3.7 val.py --weights runs/train/exp9/weights/last.pt --data coco.yaml --img 640 \
     --iou 0.65 --half --device npu --batch-size 32 --local_rank 0 > $cur_path/test/eval.log 2>&1 &