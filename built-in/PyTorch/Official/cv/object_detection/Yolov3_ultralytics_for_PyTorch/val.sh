source env_npu.sh
python3 val.py --data voc.yaml --img-size 640 --weights 'best.pt' --batch-size 32 --device npu