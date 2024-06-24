# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse
import os
import stat
import glob
import time
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from ais_bench.infer.interface import InferSession

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, SETTINGS, RANK, TQDM_BAR_FORMAT, LOGGER, colorstr, ops
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.ops import Profile, xywh2xyxy, xyxy2xywh
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader


def _reset_ckpt_args(args):
    for arg in 'augment', 'verbose', 'project', 'name', 'exist_ok', 'resume', 'batch', 'epochs', 'cache', \
            'save_json', 'half', 'v5loader', 'device', 'cfg', 'save', 'rect', 'plots':
        args.pop(arg, None)


def get_dataloader(args, dataset_path, batch_size):
    # TODO: manage splits differently
    # calculate stride - check if model is initialized
    return create_dataloader(path=dataset_path,
                             imgsz=args.imgsz,
                             batch_size=batch_size,
                             stride=32,
                             hyp=vars(args),
                             cache=False,
                             pad=0.5,
                             rect=False,
                             workers=args.workers,
                             prefix=colorstr(f'{args.mode}: '),
                             shuffle=False,
                             seed=args.seed)[0] if args.v5loader else \
        build_dataloader(args, batch_size, img_path=dataset_path, stride=32, mode="val")[0]


def init_metrics(args, model):
    val = args.data.get('val', '')  # validation path
    args.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
    args.class_map = ops.coco80_to_coco91_class() if args.is_coco else list(range(1000))
    args.save_json |= args.is_coco and not args.training  # run on final val if training COCO
    args.names = model.names
    args.nc = len(model.names)
    args.metrics.names = args.names
    args.metrics.plot = args.plots
    args.confusion_matrix = ConfusionMatrix(nc=args.nc)
    args.seen = 0
    args.jdict = []
    args.stats = []
    args.logger = LOGGER


def preprocess(args, batch):
    batch["img"] = (batch["img"].half() if args.half else batch["img"].float()) / 255

    nb = len(batch["img"])
    args.lb = [torch.cat([batch["cls"], batch["bboxes"]], dim=-1)[batch["batch_idx"] == i]
                for i in range(nb)] if args.save_hybrid else []  # for autolabelling
    
    return batch


def postprocess(args, preds):
    preds = non_max_suppression(prediction=preds,
                                conf_thres=args.conf,
                                iou_thres=args.iou,
                                labels=args.lb,
                                multi_label=True,
                                agnostic=args.single_cls,
                                max_det=args.max_det)

    return preds


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm = 0,  # number of masks
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nm (int): The number of masks output by the model.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    if conf_thres < 0 or conf_thres > 1:
        raise ValueError(f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0')
    if iou_thres < 0 or iou_thres > 1:
        raise ValueError(f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0')
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    
    ### mod
    prediction = torch.tensor(prediction)
    device = 'cpu'
    prediction.to(device)
    ###

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def _process_batch(args, detections, labels):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], args.iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(args.iouv)):
        x = torch.where((iou >= args.iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def update_metrics(args, preds, batch):
    # Metrics
    for si, pred in enumerate(preds):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]
        bbox = batch["bboxes"][idx]
        nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
        shape = batch["ori_shape"][si]
        correct_bboxes = torch.zeros(npr, args.niou, dtype=torch.bool, device=args.device)  # init
        args.seen += 1

        if npr == 0:
            if nl:
                args.stats.append((correct_bboxes, *torch.zeros((2, 0), device=args.device), cls.squeeze(-1)))
                if args.plots:
                    args.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
            continue

        # Predictions
        if args.single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        ops.scale_boxes(batch["img"][si].shape[1:], predn[:, :4], shape,
                        ratio_pad=batch["ratio_pad"][si])  # native-space pred

        # Evaluate
        if nl:
            height, width = batch["img"].shape[2:]
            tbox = xywh2xyxy(bbox) * torch.tensor(
                (width, height, width, height), device=args.device)  # target boxes
            ops.scale_boxes(batch["img"][si].shape[1:], tbox, shape,
                            ratio_pad=batch["ratio_pad"][si])  # native-space labels
            labelsn = torch.cat((cls, tbox), 1)  # native-space labels
            correct_bboxes = _process_batch(args, predn, labelsn)
            if args.plots:
                args.confusion_matrix.process_batch(predn, labelsn)
        args.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

        # Save
        if args.save_json:
            pred_to_json(args, predn, batch["im_file"][si])
    

def plot_val_samples(args, batch, ni):
    plot_images(batch["img"],
                batch["batch_idx"],
                batch["cls"].squeeze(-1),
                batch["bboxes"],
                paths=batch["im_file"],
                fname=args.save_dir / f"val_batch{ni}_labels.jpg",
                names=args.names)


def plot_predictions(args, batch, preds, ni):
    plot_images(batch["img"],
                *output_to_target(preds, max_det=15),
                paths=batch["im_file"],
                fname=args.save_dir / f'val_batch{ni}_pred.jpg',
                names=args.names)  # pred


def pred_to_json(args, predn, filename):
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        args.jdict.append({
            'image_id': image_id,
            'category_id': args.class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def get_desc():
    return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)")


def get_stats(args):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*args.stats)]  # to numpy
    if len(stats) and stats[0].any():
        args.metrics.process(*stats)
    args.nt_per_class = np.bincount(stats[-1].astype(int), minlength=args.nc)  # number of targets per class
    
    return args.metrics.results_dict


def print_results(args):
    pf = '%22s' + '%11i' * 2 + '%11.3g' * len(args.metrics.keys)  # print format
    args.logger.info(pf % ("all", args.seen, args.nt_per_class.sum(), *args.metrics.mean_results()))
    if args.nt_per_class.sum() == 0:
        args.logger.warning(
            f'WARNING ⚠️ no labels found in {args.task} set, can not compute metrics without labels')

    # Print results per class
    if args.verbose and not args.training and args.nc > 1 and len(args.stats):
        for i, c in enumerate(args.metrics.ap_class_index):
            args.logger.info(pf % (args.names[c], args.seen, args.nt_per_class[c], *args.metrics.class_result(i)))

    if args.plots:
        args.confusion_matrix.plot(save_dir=args.save_dir, names=list(args.names.values()))
    

def eval_json(args, stats):
    if args.save_json and args.is_coco and len(args.jdict):
        anno_json = Path(args.data['path']) / "annotations/instances_val2017.json"  # annotations
        pred_json = Path(args.save_dir) / "predictions.json"  # predictions
        args.logger.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO  # noqa
            from pycocotools.cocoeval import COCOeval  # noqa

            for x in anno_json, pred_json:
                if not x.is_file():
                    raise FileNotFoundError(f"{x} file not found")
            anno = COCO(str(anno_json))  # init annotations api
            pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
            eval = COCOeval(anno, pred, 'bbox')
            if args.is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in args.dataloader.dataset.im_files]  # images to eval
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            stats[args.metrics.keys[-1]], stats[args.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
        except Exception as e:
            args.logger.warning(f'pycocotools unable to run: {e}')


def val(input_args):
    weights = input_args.weight
    pt_model, _ = attempt_load_one_weight(weights)
    task = pt_model.args["task"]
    overrides = pt_model.args
    overrides = overrides.copy()
    _reset_ckpt_args(overrides)

    overrides["mode"] = "val"
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    args.task = task

    project = args.project or Path(SETTINGS['runs_dir']) / args.task

    name = args.name or f"{args.mode}"
    args.save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    args.lb = []
    args.metrics = DetMetrics(save_dir=args.save_dir)
    args.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    args.niou = args.iouv.numel()
    args.save_json = True
    args.device = 'cpu'
    args.training = False

    if isinstance(args.data, str) and args.data.endswith(".yaml"):
        args.data = check_det_dataset(args.data)
    args.dataloader = get_dataloader(args, args.data.get("val"), args.batch)

    om_model_path = input_args.om
    device_id = input_args.device_id
    om_model = InferSession(int(device_id), om_model_path)

    dt = Profile(), Profile(), Profile()
    n_batches = len(args.dataloader)
    desc = get_desc()
    bar = tqdm(args.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
    init_metrics(args, pt_model)
    for batch_i, batch in enumerate(bar):
        # preprocess
        with dt[0]:
            batch = preprocess(args, batch)
        
        # inference
        with dt[1]:
            preds = om_model.infer([batch['img']])

        # pre-process predicitions
        with dt[2]:
            preds = postprocess(args, preds)
        
        update_metrics(args, preds, batch)
        if args.plots and batch_i < 3:
            plot_val_samples(args, batch, batch_i)
            plot_predictions(args, batch, preds, batch_i)
    
    stats = get_stats(args)
    print_results(args)
    speed = tuple(x.t / len(args.dataloader.dataset) * 1E3 for x in dt)  # speeds per image
    args.logger.info('Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image' % speed)
    
    if args.save_json and args.jdict:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(str(args.save_dir / "predictions.json"), flags, mode), 'w') as f:
            args.logger.info(f"Saving {f.name}...")
            json.dump(args.jdict, f)  # flatten and save
        eval_json(args, stats)  # update stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='./yolov8n.pt', help='pt file path')
    parser.add_argument('--om', default='./yolov8n_bs1.om', help='om model path')
    parser.add_argument('--device_id', default='0', help='device id')
    input_args = parser.parse_args()

    val(input_args)


if __name__ == '__main__':
    main()
