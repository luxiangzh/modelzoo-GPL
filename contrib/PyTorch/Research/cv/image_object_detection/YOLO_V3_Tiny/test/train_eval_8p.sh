source env_npu.sh
nohup python3.7 -u ../test.py --data ../data/coco2017.data --cfg ../cfg/yolov3-tiny.cfg --weights output/last.pt --device npu --batch-size 64  --augment
