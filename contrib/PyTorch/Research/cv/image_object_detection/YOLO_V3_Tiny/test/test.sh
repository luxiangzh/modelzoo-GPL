source env_npu.sh
python3 -u ../test.py --data ../data/coco2017.data --weights ./output/last.pt --device npu --batch-size 64 --cfg ../cfg/yolov3-tiny.cfg --augment
