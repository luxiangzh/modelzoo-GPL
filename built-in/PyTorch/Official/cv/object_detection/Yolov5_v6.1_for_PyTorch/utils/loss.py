# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DeterministicIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices_list):
        ctx.x = x
        ctx.indices_list = indices_list
        return x[indices_list[0], indices_list[1], :, indices_list[2], indices_list[3]]

    @staticmethod
    def backward(ctx, grad_output):
        tmp = torch.zeros_like(ctx.x)
        ind0, ind1, ind2, ind3 = ctx.indices_list
        tmp[ind0, ind1, :, ind2, ind3] = grad_output
        return tmp, None

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        self.ft = torch.npu.FloatTensor
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=self.ft([h['cls_pw']]), reduction='sum').to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=self.ft([h['obj_pw']]), reduction='mean').to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=(0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        self.model = model
        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = self.ft([0]).to(device), self.ft([0]).to(device), self.ft([0]).to(device)
        tcls, tbox, indices, anchors, targets_mask, targets_sum_mask = self.build_targets(p, targets, self.model)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            allmask = targets_mask[i]
            sum_mask = targets_sum_mask[i]
            tobj = torch.zeros_like(pi[:, :, 0, :, :]).to(device)  # target obj

            n = b.shape[0]  # number of targets
            if sum_mask.item() > 0:
                ps = DeterministicIndex.apply(pi, (b, a, gj, gi)).permute(1, 0).contiguous()
                # GIoU
                pxy = ps.index_select(0, torch.tensor([0, 1], device=targets.device))
                pwh = ps.index_select(0, torch.tensor([2, 3], device=targets.device))

                pxy = pxy.sigmoid() * 2. - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * (anchors[i].T)
                pbox = torch.cat((pxy, pwh), 0)  # predicted box

                # Regression
                iou = bbox_iou(pbox, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # iou = torch.npu_ciou(pbox, tbox[i], trans=True, is_cross=False).squeeze()
                iou = iou * (allmask) + (1. - allmask)
                lbox += (1.0 - iou).sum() / (sum_mask)  # iou loss

                # Objectness
                iou = iou * (allmask)
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.model.nc > 1:  # cls loss (only if multiple classes)
                    tmp = ps[5:, :]
                    tmp = tmp * (allmask) - (1. - allmask) * 50.
                    t = torch.full_like(tmp, self.cn).to(device)  # targets
                    range_nb = torch.arange(n, device=device).long()
                    t[tcls[i], range_nb] = self.cp

                    t = t * (allmask)
                    lcls += (self.BCEcls(tmp, t) / (sum_mask * t.shape[0]).float())  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[:, :, 4, :, :], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets, model):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
            else model.model[-1]  # Detect() module
        na, nt = det.na, targets.shape[1]  # number of anchors, targets

        batch_size = p[0].shape[0]
        nt_max = 32 * batch_size
        while nt > nt_max:
            nt_max *= 2
            print("************* nt max=", nt_max)
        max_target = torch.zeros(6, nt_max, device=targets.device)  # (6, 1024)
        max_target[0, :nt] = targets[0, :]
        max_target[1, :nt] = targets[1, :]
        max_target[2, :nt] = targets[2, :]
        max_target[3, :nt] = targets[3, :]
        max_target[4, :nt] = targets[4, :]
        max_target[5, :nt] = targets[5, :]

        tcls, tbox, indices, anch, targets_mask, targets_sum_mask = [], [], [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off_list = [
            torch.tensor([[1.], [0.]], device=targets.device),
            torch.tensor([[0.], [1.]], device=targets.device),
            torch.tensor([[-1.], [0.]], device=targets.device),
            torch.tensor([[0.], [-1.]], device=targets.device)
        ]
        at = torch.arange(na).view(na, 1).repeat(1, nt_max).to(targets.device)  # anchor tensor, same as .repeat_interleave(nt)
        a = at.view(-1)
        a = torch.cat((a, a, a, a, a), 0)

        g = 0.5  # bias

        for i in range(det.nl):
            anchors = det.anchors[i].float()
            gain[2:] = torch.tensor(p[i].shape)[[4, 3, 4, 3]].float()  # xyxy gain

            # Match targets to anchors
            # t = targets * gain
            t, offsets = max_target * gain[:, None], 0
            allmask = torch.zeros((15 * nt_max)).to(targets.device)
            sum_mask = torch.zeros((1)).to(targets.device)

            if nt:
                # Matches
                r = t[None, 4:6, :] / anchors[..., None]  # wh ratio
                fmask = torch.max(r, 1. / r).max(1)[0] < model.hyp['anchor_t']  # compare
                fmask = fmask.view(1, -1)
                t = t.repeat(1, 1, na).view(6, -1)  # filter

                # overlaps
                gxy = t.index_select(0, torch.tensor([2, 3], device=targets.device))  # (3072, 2)
                z = torch.zeros_like(gxy)

                jk = (gxy % 1. < g) & (gxy > 1.)
                lm = (gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]][:, None] - 1.))
                jk, lm = jk & fmask, lm & fmask
                allmask = torch.cat((fmask, jk, lm), 0).view(1, -1).float()
                t = torch.cat((t, t, t, t, t), 1)
                offsets = torch.cat((z, z + off_list[0], z + off_list[1], z + off_list[2], z + off_list[3]), 1) * g

                sum_mask = allmask.sum()
                t = t * allmask


            # Define
            b = t.index_select(0, torch.tensor([0], device=targets.device)).long().view(-1)  # (3072 * 5)
            c = t.index_select(0, torch.tensor([1], device=targets.device)).long().view(-1)  # (3072 * 5)
            gxy = t.index_select(0, torch.tensor([2, 3], device=targets.device))  # (2, 3072 * 5)
            gwh = t.index_select(0, torch.tensor([4, 5], device=targets.device))  # (2, 3072 * 5)
            gij = gxy - offsets
            gij2 = gij.long()
            gi = gij2.index_select(0, torch.tensor([0], device=targets.device)).view(-1)  # (2, 3072 * 5)
            gj = gij2.index_select(0, torch.tensor([1], device=targets.device)).view(-1)  # (2, 3072 * 5)

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij2.float(), gwh), 0))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            targets_mask.append(allmask)
            targets_sum_mask.append(sum_mask)

        return tcls, tbox, indices, anch, targets_mask, targets_sum_mask
