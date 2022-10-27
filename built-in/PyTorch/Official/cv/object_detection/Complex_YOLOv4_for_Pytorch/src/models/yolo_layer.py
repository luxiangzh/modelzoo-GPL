'''# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================
'''

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

from utils.torch_utils import to_cpu
from utils.iou_rotated_boxes_utils import iou_pred_vs_target_boxes, iou_rotated_boxes_targets_vs_anchors, \
    get_polygons_areas_fix_xy


class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, num_classes, anchors, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        # Update the attributions when parsing the cfg during create the darknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = 100
        self.obj_scale = 1
        self.lgiou_scale = 3.54
        self.leular_scale = 3.54
        self.lobj_scale = 64.3
        self.lcls_scale = 37.4

        self.seen = 0
        # Initialize dummy variables
        self.grid_size = 0
        self.img_size = 0
        self.metrics = {}

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).view([1, g, g, 1])
        self.grid_y = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).t().view([1, g, g, 1])
        self.scaled_anchors = torch.tensor(self.anchors, device=self.device, dtype=torch.float)
        self.scaled_anchors[:, :2] /= self.stride
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, 1, 1 , self.num_anchors))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, 1, 1 , self.num_anchors))

        # Pre compute polygons and areas of anchors
        self.scaled_anchors_polygons, self.scaled_anchors_areas = get_polygons_areas_fix_xy(self.scaled_anchors)

    def build_targets(self, pred_boxes, pred_cls, target_ori, anchors):
        """ Built yolo targets to compute loss
        :param out_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, num_classes]
        :param target: [num_boxes, 8]
        :param anchors: [num_anchors, 4]
        :return:
        """
        nB, nG, _, nA, nC = pred_cls.size()
        nt_boxes = target_ori.size(0)
        new_nt_boxes = math.ceil(nt_boxes / 32) * 32

        # Create output tensors on "device"
        obj_mask = torch.full(size=(nB, nG, nG, nA), fill_value=1, device=self.device, dtype=torch.float)
        noobj_mask = torch.full(size=(nB, nG, nG, nA), fill_value=1, device=self.device, dtype=torch.bool)
        tes = torch.full(size=(6, new_nt_boxes), fill_value=0, device=self.device, dtype=torch.float)
        res = torch.full(size=(6, nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        tcls = torch.full(size=(nB, nG, nG, nA, nC), fill_value=0, device=self.device, dtype=torch.float)
        giou_loss = torch.tensor([0.], device=self.device, dtype=torch.float)

        if nt_boxes > 0:  # Make sure that there is at least 1 box
            target = torch.zeros([new_nt_boxes, 8])
            target[:nt_boxes] = target_ori
            target = target.npu().t()

            b, target_labels = target[:2, :].long()
            target_boxes = torch.cat((target[2:6, :] * nG, target[6:8, :]), dim=0)  # scale up x, y, w, h

            gxy = target_boxes[:2, :]
            gwh = target_boxes[2:4, :]
            gimre = target_boxes[4:6, :]
            gi, gj = gxy.long()
            
            # Get anchors with best iou
            targets_polygons, targets_areas = get_polygons_areas_fix_xy(target_boxes.t()[:, 2:6])
            ious = iou_rotated_boxes_targets_vs_anchors(self.scaled_anchors_polygons, self.scaled_anchors_areas,
                                                        targets_polygons, targets_areas)
            _, best_n = ious.max(1)

            # Set masks
            noobj_mask[b, gj, gi, best_n] = 0
            noobj_mask[0, 0, 0, 0] = 1
            obj_mask -= noobj_mask
            
            # Set noobj mask to zero where iou exceeds ignore threshold
            noobj_mask[b, gj, gi, :] *= 1. * (ious < self.ignore_thresh)


            # Coordinates
            tes[0:2] = gxy - gxy.floor()
            # Width and height
            tes[2:4] = torch.log(gwh / anchors[best_n].t()[:2] + 1e-16)
            # Im and real part
            tes[4:6] = gimre
            res[:, b, gj, gi, best_n] = tes

            # One-hot encoding of label
            if self.use_giou_loss:
                _, giou_loss = iou_pred_vs_target_boxes(pred_boxes[b, best_n, gj, gi], target_boxes,
                                                       GIoU=self.use_giou_loss)
            tcls[b, gj, gi, best_n, target_labels] = 1

        return giou_loss, obj_mask.type(torch.bool), noobj_mask, res, tcls, obj_mask

    def forward(self, x, targets=None, img_size=608, use_giou_loss=False):
        """
        :param x: [num_samples or batch, num_anchors * (6 + 1 + num_classes), grid_size, grid_size]
        :param targets: [num boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        self.use_giou_loss = use_giou_loss
        self.device = x.device
        num_samples, _, _, grid_size = x.size()

        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
        prediction = prediction.permute(0, 3, 4, 1, 2).contiguous()
        # prediction size: [num_samples, grid_size, grid_size, num_anchors, num_classes + 7]

        # Get outputs
        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]  # Width
        pred_h = prediction[..., 3]  # Height
        pred_im = prediction[..., 4]  # angle imaginary part
        pred_re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        # pred_boxes size: [num_samples, num_anchors, grid_size, grid_size, 6]
        pred_boxes = torch.empty(prediction[..., :6].shape, device=self.device, dtype=torch.float)
        pred_boxes[..., 0] = pred_x + self.grid_x
        pred_boxes[..., 1] = pred_y + self.grid_y
        pred_boxes[..., 2] = torch.exp(pred_w).clamp(max=1E3) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h).clamp(max=1E3) * self.anchor_h
        pred_boxes[..., 4] = pred_im
        pred_boxes[..., 5] = pred_re
        
        output = torch.cat((
            pred_boxes[..., :4] * self.stride,
            pred_boxes[..., 4:6],
            pred_conf.unsqueeze(-1),
            pred_cls,
        ), dim=-1)
        output = output.permute(0, 3, 1, 2, 4).reshape(num_samples, -1, 7 + self.num_classes)
        # output size: [num_samples, num boxes, 7 + num_classes]
        if targets is None:
            return output, 0

        self.reduction = 'sum'
        giou_loss, obj_mask, noobj_mask, res, tcls, tconf = self.build_targets(
            pred_boxes=pred_boxes, pred_cls=pred_cls, target_ori=targets, anchors=self.scaled_anchors)

        self.nt = torch.sum(obj_mask.reshape(-1))
        self.nont = torch.sum(noobj_mask.reshape(-1))
        self.nt_new = math.ceil(int(self.nt) / 32) * 32

        masks = torch.zeros([12, self.nt_new])
        preds = torch.zeros([6, num_samples, grid_size, grid_size, self.num_anchors])
        preds[0] = pred_x
        preds[1] = pred_y
        preds[2:6] = prediction[..., 2:6].permute(4, 0, 1, 2, 3)
        masks[:6, :self.nt] = preds[:, obj_mask]
        masks[6:, :self.nt] = res[:, obj_mask]
        masks = masks.npu().contiguous()
        loss_sum = F.mse_loss(masks[:6], masks[6:], reduction=self.reduction)

        masks_im_re1 = torch.zeros([self.nt_new])
        masks_im_re2 = torch.zeros([self.nt_new])
        masks_im_re3 = torch.zeros([self.nt_new])
        masks_im_re1[:self.nt] = pred_im[obj_mask]
        masks_im_re2[:self.nt] = pred_re[obj_mask]
        masks_im_re3[:self.nt] = 1
        loss_im_re = (1. - torch.sqrt(masks_im_re1 ** 2 + masks_im_re2 ** 2)) ** 2  # as tim^2 + tre^2 = 1
        loss_im_re_red = torch.sum(loss_im_re * masks_im_re3).npu()

        masks1 = torch.zeros([self.nt_new])
        masks2 = torch.zeros([self.nt_new])
        masks1[:self.nt] = pred_conf[obj_mask]
        masks2[:self.nt] = tconf[obj_mask]
        loss_conf_obj = F.binary_cross_entropy(masks1.npu(), masks2.npu(), reduction=self.reduction)

        loss_conf_noobj = F.binary_cross_entropy(pred_conf * noobj_mask, tconf * noobj_mask, reduction=self.reduction) / self.nont

        masks_cls1 = torch.zeros([self.nt_new, 3])
        masks_cls2 = torch.zeros([self.nt_new, 3])
        masks_cls1[:self.nt] = pred_cls[obj_mask]
        masks_cls2[:self.nt] = tcls[obj_mask]
        loss_cls = F.binary_cross_entropy(masks_cls1.npu(), masks_cls2.npu(), reduction=self.reduction) / 3

        total_loss = (loss_sum + loss_im_re_red + loss_cls + self.obj_scale * loss_conf_obj) / self.nt + self.noobj_scale * loss_conf_noobj

        return output, total_loss
