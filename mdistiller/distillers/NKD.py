import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller



def nkd_loss(logits_student, logits_teacher, target, alpha, temperature):
        
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
        loss_non = - alpha * (temperature**2) * loss_non

        return loss_t + loss_non



class NKD(Distiller):
    """TCKD with Distributed Non-Target Loss"""
    
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.NKD.CE_WEIGHT
        self.alpha = cfg.NKD.ALPHA
        self.temperature = cfg.NKD.T
        
    
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            
        #losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nkd = nkd_loss(logits_student, logits_teacher, target, self.alpha, self.temperature)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_ndkd": loss_nkd,
        }
        return logits_student, losses_dict