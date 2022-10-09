from typing import Tuple
import unittest as ut
#
import torch
from torch.optim.sgd import SGD
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR
#
import mini_lightning as ml


def allclose(lr: float, lr2: float, atol=1e-6) -> Tuple[bool, float]:
    """return bool, true_atol: float"""
    atol2 = abs(lr - lr2)
    res = False
    if atol2 < atol:
        res = True
    return res, atol2


class TestLrs(ut.TestCase):
    def test_calr(self):
        # test cosine_annealing_lr
        initial_lr = 1e-2
        T_max = 10
        eta_min = 1e-4
        max_epoch = 20
        #
        optim = SGD([Parameter(torch.randn(100,))], initial_lr)
        lrs = CosineAnnealingLR(optim, T_max, eta_min)
        for i in range(max_epoch):
            lr = ml.cosine_annealing_lr(i, T_max, eta_min, [initial_lr])[0]
            lr2 = lrs.get_last_lr()[0]
            b, atol = allclose(lr, lr2)
            self.assertTrue(b, msg=f"atol: {atol}")
            optim.step()
            lrs.step()
        #
        lr = ml.cosine_annealing_lr(max_epoch, T_max, eta_min, [initial_lr])[0]
        lr2 = lrs.get_last_lr()[0]
        b, atol = allclose(lr, lr2)
        self.assertTrue(b, msg=f"atol: {atol}")

    def test_warmup1(self):
        initial_lr = 1e-2
        T_max = 10
        eta_min = 1e-4
        max_epoch = 20
        #
        optim = SGD([Parameter(torch.randn(100,))], initial_lr)
        warmup = 3
        lrs = ml.warmup_decorator(CosineAnnealingLR, warmup)(optim, T_max, eta_min)
        for i in range(max_epoch):
            lr = lrs.get_last_lr()[0]
            lr2 = ml.cosine_annealing_lr(i, T_max, eta_min, [initial_lr])[0]
            b, atol = allclose(lr, lr2)
            if i == 0:
                self.assertTrue(lr > 0)  # !=0
            elif i == warmup - 1:
                self.assertTrue(not b)
            elif i >= warmup:
                self.assertTrue(b, msg=f"atol: {atol}")
                if i == T_max:
                    self.assertTrue(lr == eta_min)
            optim.step()
            lrs.step()


if __name__ == "__main__":
    ut.main()
