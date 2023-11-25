# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import json
import argparse
import torch
import torch_aie

try:
    from utils.datasets import create_dataloader
except:
    from utils.dataloaders import create_dataloader

from common.util.dataset import BatchDataLoader, evaluate
from common.util.model import forward_nms_script


def main(opt, cfg):
    # load model
    torch_aie.set_device(opt.device_id)
    model = torch.jit.load(opt.model)
    model.eval()

    # load dataset
    single_cls = False if opt.tag >= 6.0 else opt
    dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size, max(cfg["stride"]), single_cls, pad=0.5)[0]

    # inference & nms
    pred_results, infer_time = forward_nms_script(model, dataloader, cfg, opt)

    pred_json_file = f"{os.path.basename(opt.model).split('.')[0]}_{opt.tag}_predictions.json"
    print(f'saving results to {pred_json_file}')
    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)

    # calculate infer throughput
    avg_inf_time = sum(infer_time[3:]) / len(infer_time[3:])
    throughput = opt.batch_size / avg_inf_time
    print("The throughput of the torch_aie yolov5 is ", throughput)

    # evaluate mAP
    evaluate(opt.ground_truth_json, pred_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 model torch_aie inference.')
    parser.add_argument('--data_path', type=str, default="coco", help='root dir for val images and annotations')
    parser.add_argument('--ground_truth_json', type=str, default="coco/instances_val2017.json",
                        help='annotation file path')
    parser.add_argument('--tag', type=float, default=6.1, help='yolov5 tags')
    parser.add_argument('--model', type=str, default="yolov5s_v6_bs8_aie.pt.", help='aie model path')
    # parser.add_argument('--nms_mode', type=str, default="nms_op", help='nms compute mode [nms_op/nms_script]')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)
