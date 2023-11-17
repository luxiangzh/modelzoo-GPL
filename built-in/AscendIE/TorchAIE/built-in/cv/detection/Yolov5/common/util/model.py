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

import torch
import torch_aie
from tqdm import tqdm
import time

from pathlib import Path
from common.util.dataset import coco80_to_coco91_class, correct_bbox, save_coco_json

try:
    from utils.general import non_max_suppression, scale_coords  # tag > 2.0
except:
    from utils.utils import non_max_suppression, scale_coords  # tag = 2.0



def forward_nms_script(model, dataloader, cfg, opt):
    pred_results = []
    infer_time = []
    inf_stream = torch_aie.npu.Stream("npu:0")
    for (img, targets, paths, shapes) in tqdm(dataloader):
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width

        padding = False
        if nb != opt.batch_size:
            img = torch.nn.functional.pad(img, ((0, batch_size - nb), (0,0), (0,0),(0,0)), 'constant', constant_values=0)
            padding = True
        
        img_npu = img.to("npu:0")

        # torch_aie model infer
        inf_s = time.time()
        with torch_aie.npu.stream(inf_stream):
            result = model(img_npu)
        inf_stream.synchronize()
        inf_e = time.time()
        infer_time.append(inf_e - inf_s)

        if len(result) == 3:  # number of output nodes is 3, each shape is (bs, na, no, ny, nx)
            out = []
            for i in range(len(result)):
                anchors = torch.tensor(cfg['anchors'])
                stride = torch.tensor(cfg['stride'])
                cls_num = cfg['class_num']
                result[i] = result[i].to("cpu")
                if padding == True:
                    result[i] = result[i][:nb]
                correct_bbox(result[i], anchors[i], stride[i], cls_num, out)
            box_out = torch.cat(out, 1)
        else:  # only use the first output node, which shape is (bs, -1, no)
            result[0].to("cpu")
            if padding == True:
                result[0] = result[0][:nb]
            box_out = result[0].clone().detach()

        # non_max_suppression
        boxout = nms(box_out, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        for idx, pred in enumerate(boxout):
            try:
                scale_coords(img[idx].shape[1:], pred[:, :4], shapes[idx][0], shapes[idx][1])  # native-space pred
            except:
                pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # append to COCO-JSON dictionary
            path = Path(paths[idx])
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            save_coco_json(pred, pred_results, image_id, coco80_to_coco91_class())

    return pred_results, infer_time


def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        boxout = non_max_suppression(box_out.to("cpu"), conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        boxout = non_max_suppression(box_out.to("cpu"), conf_thres=conf_thres, iou_thres=iou_thres)

    return boxout
