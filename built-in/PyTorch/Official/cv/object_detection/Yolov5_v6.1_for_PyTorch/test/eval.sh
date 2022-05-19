cur_path=`pwd`
source ${cur_path}/test/env_npu.sh
data_path=""
# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
sed -i "s|../datasets/coco|$data_path|g" data/coco.yaml

python3.7 val.py --weights runs/train/exp9/weights/last.pt --data coco.yaml --img 640 \
     --iou 0.65 --half --device npu --batch-size 32 --local_rank 0 > $cur_path/test/eval.log 2>&1 &