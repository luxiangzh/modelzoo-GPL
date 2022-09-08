source test/env_npu.sh
cd src

python3 evaluate.py --local_rank 0 --pretrained_path ../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_best.pth
