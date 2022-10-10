## 帮助信息
### === Model Options ===
###  --model        yolov3, default: yolov3
###  --bs           batch size, default: 1
### === Inference Options ===
###  --output_dir   output dir, default: output
### === Environment Options ===
###  --soc          soc version [Ascend310P?], default: Ascend310P3
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al model:,bs:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;;
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi

# atc转模型
atc --model=${output_dir}/${model}.onnx \
    --framework=5 \
    --output=${output_dir}/${model}_bs${bs} \
    --input_format=NCHW \
    --input_shape="images:${bs},3,640,640;img_info:${bs},4" \
    --log=error \
    --soc_version=${soc} \
    --optypelist_for_implmode="Sigmoid" \
    --op_select_implmode=high_performance


