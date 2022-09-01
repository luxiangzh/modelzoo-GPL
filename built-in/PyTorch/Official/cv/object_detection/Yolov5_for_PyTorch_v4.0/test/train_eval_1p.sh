source test/env_npu.sh
python3 test.py --data /data/coco.yaml --img-size 640 --weight 'yolov5_0.pt' --batch-size 32 --device 0 
