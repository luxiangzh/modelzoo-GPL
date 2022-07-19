source env_npu.sh
python3 train.py --data voc.yaml --cfg yolov3-spp.yaml --weights '' --batch-size 32 --noval