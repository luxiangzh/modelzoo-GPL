source env_npu.sh
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export RANK_SIZE=8
rm -f nohup.out

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK_ID=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
    let a=0+RANK_ID*24
    let b=23+RANK_ID*24
    nohup taskset -c $a-$b python3 ../train.py \
          --data ../data/coco2017.data  \
          --cfg ../cfg/yolov3-tiny.cfg  \
          --weights '' \
          --device-type npu \
          --batch-size 190 \
          --device-count 8 &
    else
        python3 train.py \
            --weights '' \
            --device-type npu \
            --batch-size 190 \
            --device-count 8 &
    fi
done

