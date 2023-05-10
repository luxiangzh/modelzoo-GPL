source env_npu.sh
nohup python3 -u ../train.py --weights '' --device-type npu --batch-size 160 --device-count 1 &

