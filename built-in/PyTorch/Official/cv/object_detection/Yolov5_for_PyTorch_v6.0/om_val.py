# Copyright 2022 Huawei Technologies Co., Ltd
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

import yaml
import json
import argparse
from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.summary import summary

from utils.datasets import create_dataloader
from common.util.dataset import BatchDataLoader, evaluate
from common.util.model import forward_nms_op, forward_nms_script


def main(opt, cfg):
    # load model
    model = InferSession(opt.device_id, opt.model)

    if opt.nms_mode == "nms_op":
        # load dataset
        dataloader = BatchDataLoader(opt.data_path, batch_size=opt.batch_size)

        # inference & nms
        pred_results = forward_nms_op(model, dataloader)

    elif opt.nms_mode == "nms_script":
        # load dataset
        single_cls = False if opt.tag >= 6.0 else opt
        print(opt.data_path, flush=True)
        dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size, max(cfg["stride"]), single_cls, pad=0.5, pin_memory=False)[0]

        # inference & nms
        pred_results = forward_nms_script(model, dataloader, cfg)

    s = model.sumary()
    summary.npu_compute_time_list = s.exec_time_list
    summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
    summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
    summary.report(opt.batch_size, output_prefix=None, display_all_summary=False)

    pred_json_file = f"{opt.model.split('.')[0]}_{opt.tag}_predictions.json"
    print(f'saving results to {pred_json_file}')
    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)

    # evaluate mAP
    evaluate(opt.ground_truth_json, pred_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 offline model inference.')
    parser.add_argument('--data_path', type=str, default="./datasets/coco", help='root dir for val images and annotations')
    parser.add_argument('--ground_truth_json', type=str, \
        default="./datasets/coco/annotations/instances_val2017.json", help='annotation file path')
    parser.add_argument('--tag', type=float, default=6.1, help='yolov5 tags')
    parser.add_argument('--model', type=str, default="yolov5s.om", help='om model path')
    parser.add_argument('--nms_mode', type=str, default="nms_op", help='nms compute mode [nms_op/nms_script]')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)
