source env_npu.sh
python3 -m torch.distributed.launch --nproc_per_node 8 train.py --data voc.yaml --cfg yolov3.yaml --weights '' --batch-size 256 --noval