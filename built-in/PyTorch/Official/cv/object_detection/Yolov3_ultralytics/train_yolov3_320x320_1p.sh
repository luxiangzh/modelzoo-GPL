source env_npu.sh
python3 train.py --data voc.yaml --cfg yolov3.yaml --weights '' --batch-size 64 --noval --img-size 320