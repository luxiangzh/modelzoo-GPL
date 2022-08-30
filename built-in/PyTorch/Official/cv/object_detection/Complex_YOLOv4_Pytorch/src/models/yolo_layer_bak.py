"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the yolo layer

# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
# Refer: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
"""

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
        self.scaled_anchors = torch.tensor(
            [(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors], device=self.device,
            dtype=torch.float)
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

        # Create output tensors on "device"
        obj_mask = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        noobj_mask = torch.full(size=(nB, nG, nG, nA), fill_value=1, device=self.device, dtype=torch.float)
        tx = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        ty = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        tw = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        th = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        tim = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        tre = torch.full(size=(nB, nG, nG, nA), fill_value=0, device=self.device, dtype=torch.float)
        tcls = torch.full(size=(nB, nG, nG, nA, nC), fill_value=0, device=self.device, dtype=torch.float)
        giou_loss = torch.tensor([0.], device=self.device, dtype=torch.float)

        if nt_boxes > 0:  # Make sure that there is at least 1 box
            target = torch.zeros([math.ceil(nt_boxes / 32) * 32, 8])
            target[:nt_boxes] = target_ori
            target = target.cuda()

            b, target_labels = target[:, :2].long().t()
            target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:8]), dim=-1)  # scale up x, y, w, h

            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:4]
            gimre = target_boxes[:, 4:6]

            targets_polygons, targets_areas = get_polygons_areas_fix_xy(target_boxes[:, 2:6])
            # Get anchors with best iou
            ious = iou_rotated_boxes_targets_vs_anchors(self.scaled_anchors_polygons, self.scaled_anchors_areas,
                                                        targets_polygons, targets_areas)
            _, best_n = ious.max(1)

            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gim, gre = gimre.t()
            gi, gj = gxy.long().t()
            
            # Set noobj mask to zero where iou exceeds ignore threshold
            ious_mask = 1. * (ious < self.ignore_thresh)
            for i in range(ious.size(0)):
                noobj_mask[b[i], gj[i], gi[i], :] = ious_mask[i]

            # Set masks
            obj_mask[b, gj, gi, best_n] = 1
            obj_mask[0, 0, 0, 0] = 0
            noobj_mask[b, gj, gi, best_n] = 0
            noobj_mask[0, 0, 0, 0] = 1

            # Coordinates
            tx[b, gj, gi, best_n] = gx - gx.floor()
            ty[b, gj, gi, best_n] = gy - gy.floor()
            # Width and height
            tw[b, gj, gi, best_n] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            th[b, gj, gi, best_n] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
            # Im and real part
            tim[b, gj, gi, best_n] = gim
            tre[b, gj, gi, best_n] = gre

            # One-hot encoding of label
            if self.use_giou_loss:
                _, giou_loss = iou_pred_vs_target_boxes(pred_boxes[b, best_n, gj, gi], target_boxes,
                                                       GIoU=self.use_giou_loss)
            tcls[b, gj, gi, best_n, target_labels] = 1
        tconf = obj_mask.float()

        return giou_loss, obj_mask.type(torch.bool), noobj_mask.type(torch.bool), tx, ty, tw, th, tim, tre, tcls, tconf

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

        if targets.size(0) == 0:
            return output, 0
        else:
            self.reduction = 'sum'
            giou_loss, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target_ori=targets, anchors=self.scaled_anchors)

            self.nt = torch.sum(obj_mask.reshape(-1))
            self.nont = torch.sum(noobj_mask.reshape(-1))
            self.nt_new = math.ceil(int(self.nt) / 32) * 32

            masks = torch.zeros([12, self.nt_new])
            masks[0, :self.nt] = pred_x[obj_mask]
            masks[1, :self.nt] = pred_y[obj_mask]
            masks[2, :self.nt] = pred_w[obj_mask]
            masks[3, :self.nt] = pred_h[obj_mask]
            masks[4, :self.nt] = pred_im[obj_mask]
            masks[5, :self.nt] = pred_re[obj_mask]
            masks[6, :self.nt] = tx[obj_mask]
            masks[7, :self.nt] = ty[obj_mask]
            masks[8, :self.nt] = tw[obj_mask]
            masks[9, :self.nt] = th[obj_mask]
            masks[10, :self.nt] = tim[obj_mask]
            masks[11, :self.nt] = tre[obj_mask]
            masks = masks.cuda()
            loss_sum = F.mse_loss(masks[:6], masks[6:], reduction=self.reduction)

            masks_im_re1 = torch.zeros([self.nt_new])
            masks_im_re2 = torch.zeros([self.nt_new])
            masks_im_re3 = torch.zeros([self.nt_new])
            masks_im_re1[:self.nt] = pred_im[obj_mask]
            masks_im_re2[:self.nt] = pred_re[obj_mask]
            masks_im_re3[:self.nt] = 1

            loss_im_re = (1. - torch.sqrt(masks_im_re1 ** 2 + masks_im_re2 ** 2)) ** 2  # as tim^2 + tre^2 = 1
            loss_im_re_red = torch.sum(loss_im_re * masks_im_re3).cuda()

            masks1 = torch.zeros([self.nt_new])
            masks2 = torch.zeros([self.nt_new])
            masks1[:self.nt] = pred_conf[obj_mask]
            masks2[:self.nt] = tconf[obj_mask]
            loss_conf_obj = F.binary_cross_entropy(masks1.cuda(), masks2.cuda(), reduction=self.reduction)

            loss_conf_noobj = F.binary_cross_entropy(pred_conf * noobj_mask, tconf * noobj_mask, reduction=self.reduction) / self.nont

            masks_cls1 = torch.zeros([self.nt_new, 3])
            masks_cls2 = torch.zeros([self.nt_new, 3])
            masks_cls1[:self.nt] = pred_cls[obj_mask]
            masks_cls2[:self.nt] = tcls[obj_mask]
            loss_cls = F.binary_cross_entropy(masks_cls1.cuda(), masks_cls2.cuda(), reduction=self.reduction) / 3

            total_loss = (loss_sum + loss_im_re_red + loss_cls + self.obj_scale * loss_conf_obj) / self.nt + self.noobj_scale * loss_conf_noobj
            
            return output, total_loss
