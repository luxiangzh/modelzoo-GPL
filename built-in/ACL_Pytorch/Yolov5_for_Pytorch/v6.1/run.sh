##### 参数设置
soc=${1}
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi
version=${2:-"6.1"}
model=${3:-"yolov5s"}
bs=${4:-"4"}
type=${5:-"fp16"}
mode=${6:-"infer"}
output_dir=${7:-"output"}
install_path=${8:-"/usr/local/Ascend/ascend-toolkit"}

## pt导出om模型
bash common/pth2om.sh --version $version \
                      --model $model \
                      --bs $bs \
                      --type $type \
                      --mode $mode \
                      --output_dir $output_dir \
                      --soc $soc

if [ $? -ne 0 ]; then
    echo -e "pth导出om模型 Failed \n"
    exit 1
fi

## 推理om模型
bash common/eval.sh --version $version \
                    --model $model \
                    --bs $bs \
                    --type $type \
                    --mode $mode \
                    --output_dir $output_dir \
                    --install_path $install_path