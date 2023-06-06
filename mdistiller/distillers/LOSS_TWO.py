import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def dkd_loss(logits_student, logits_teacher, target, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return beta * nckd_loss

def tcce_loss(logits_student, logits_teacher, target, temperature):
        
        if len(target.size()) > 1:
            label = torch.max(target, dim=1, keepdim=True)[1]
        else:
            label = target.view(len(target), 1)

        # N*class
        s_i = F.softmax(logits_student, dim=1)
        t_i = F.softmax(logits_teacher, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        mask = torch.zeros_like(logits_student).scatter_(1, label, 1).bool()
        logits_student = logits_student - 1000 * mask
        logits_teacher = logits_teacher - 1000 * mask
        
        # N*class
        T_i = F.softmax(logits_teacher/temperature, dim=1)
        S_i = F.softmax(logits_student/temperature, dim=1)
        # N*1
        T_t = torch.gather(T_i, 1, label)
        S_t = torch.gather(S_i, 1, label)
        # N*class 
        nt_i = T_i/(1-T_t)
        ns_i = S_i/(1-S_t)
        nt_i[T_i==T_t] = 0
        ns_i[T_i==T_t] = 1

        loss_t = - (t_t * torch.log(s_t)).mean()
        loss_non =  (nt_i * torch.log(ns_i)).sum(dim=1).mean()
        #loss_non = - gamma * (temperature**2) * loss_non

        return loss_t


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def loss_func(logits_student, logits_teacher, target, beta, temperature):
    final_loss = dkd_loss(logits_student, logits_teacher, target, beta, temperature) + tcce_loss(logits_student, logits_teacher, target,temperature)
    return final_loss

class LOSS_TWO(Distiller):
    """NCKD with CE of Target class"""
    
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.LOSS_TWO.CE_WEIGHT
        self.beta = cfg.LOSS_TWO.BETA
        #self.gamma = cfg.LOSS_TWO.GAMMA
        self.temperature = cfg.LOSS_TWO.T
        self.warmup = cfg.LOSS_TWO.WARMUP
    
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            
        #losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_two = min(kwargs["epoch"] / self.warmup, 1.0) * loss_func(logits_student, logits_teacher, target, self.beta, self.temperature)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_nkd_dkd": loss_two
        }
        return logits_student, losses_dict