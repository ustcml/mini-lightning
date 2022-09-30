# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import math
from typing import List, Callable, Union, Dict, Optional
#
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler, CosineAnnealingLR
from torch.optim import Optimizer
__all__ = ["cosine_annealing_lr", "get_T_max", "warmup_decorator",
           "WarmupCosineAnnealingLR", "WarmupCosineAnnealingLR2"]


def _get_offset_func(fa: float, fb: float, ga: float, gb: float) -> Callable[[float], float]:
    # (a, fa) -> (a, ga); (b, fb) -> (b, gb)
    # Generate a linear function of curve translation and scaling. gx=s*fx+t
    if fa == fb:
        raise ValueError("fa == fb")
    s = (gb-ga) / (fb-fa)
    t = ga - fa * s

    def func(x: float) -> float:
        return s * x + t
    return func


def cosine_annealing_lr(epoch: int, T_max: int, eta_min: float, initial_lrs: List[float]) -> List[float]:
    """
    epoch=0: lr=initial_lr
    epoch=T_max: lr=eta_min
    T of sine curve = 2 * T_max
    """
    # Avoid floating point errors
    if epoch % (2 * T_max) == 0:
        return initial_lrs
    if (epoch + T_max) % (2 * T_max) == 0:
        return [eta_min] * len(initial_lrs)
    res = []
    # x axis
    x = math.cos(math.pi * epoch / T_max)  # (2 * pi * x) / (2 * T_max)
    # y axis
    for initial_lr in initial_lrs:
        func = _get_offset_func(-1, 1, eta_min, initial_lr)
        res.append(func(x))
    return res


def get_T_max(dataset_len: int, batch_size: int, max_epochs: int,
              n_accumulate_grad: Union[int, Dict[int, int]] = 1, drop_last: bool = True) -> int:
    """Calculate T_max(iteration step) in cosine_annealing_lr"""
    if isinstance(n_accumulate_grad, int):
        if drop_last:
            T_max = dataset_len // batch_size
        else:
            T_max = math.ceil(dataset_len / batch_size)
        T_max = math.ceil(T_max / n_accumulate_grad)
        T_max *= max_epochs
    elif isinstance(n_accumulate_grad, dict):
        nag_dict = n_accumulate_grad.copy()
        if 0 not in nag_dict.keys():
            nag_dict.update({0: 1})
        T_max = 0
        nag_list = sorted(list(nag_dict.keys())) + [int(1e8)]
        for i in range(len(nag_list) - 1):
            nag = nag_dict[nag_list[i]]  # n_accumulate_grad
            me: int = min(nag_list[i + 1], max_epochs) - nag_list[i]  # max_epochs
            if me <= 0:
                break
            if drop_last:
                Tm = dataset_len // batch_size
            else:
                Tm = math.ceil(dataset_len / batch_size)
            Tm = math.ceil(Tm / nag)
            T_max += Tm * me
    return T_max


class _CosineAnnealingLR(LRScheduler):
    """
    epoch=0: lr=initial_lr
    epoch=T_max: lr=eta_min
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0., last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self, last_epoch=None) -> List[float]:
        if last_epoch is None:
            last_epoch = self.last_epoch
        return cosine_annealing_lr(last_epoch, self.T_max, self.eta_min, self.base_lrs)


def warmup_decorator(lr_s: type, shift_one: bool = True) -> type:
    class WarmupLRScheduler(lr_s):
        def __init__(self, optimizer: Optimizer, warmup: int, *args, **kwargs) -> None:
            self.warmup = warmup
            self.lrs_ori: Optional[List[int]] = None
            self.shift_one = shift_one
            super().__init__(optimizer, *args, **kwargs)

        def get_lr(self) -> List[float]:
            #
            if self.lrs_ori is not None:
                for p, lr in zip(self.optimizer.param_groups, self.lrs_ori):
                    p["lr"] = lr
            elif self.shift_one:  # first
                self.last_epoch += 1
            last_epoch = self.last_epoch
            lrs = super().get_lr()
            self.lrs_ori = lrs
            #
            scale = 1
            if last_epoch < self.warmup:
                scale = last_epoch / self.warmup
            return [lr * scale for lr in lrs]
    return WarmupLRScheduler


"""
Note! In order to avoid LR to be 0 in the first step, we shifted one step to the left
iter_idx=-1: lr=0 or cosine_annealing_lr(0) * 0
iter_idx=warmup-1: lr=cosine_annealing_lr(warmup)
iter_idx=T_max-1: lr=eta_min or cosine_annealing_lr(T_max)
"""

WarmupCosineAnnealingLR = warmup_decorator(CosineAnnealingLR)
# WarmupCosineAnnealingLR = warmup_decorator(_CosineAnnealingLR)  # or


class WarmupCosineAnnealingLR2(_CosineAnnealingLR):
    """Note! In order to avoid LR to be 0 in the first step, we shifted one step to the left
    iter_idx=-1: lr=0
    iter_idx=warmup-1: lr=initial_lr or cosine_annealing_lr(0)
    iter_idx=warmup+T_max-1: lr=eta_min or cosine_annealing_lr(T_max)
    """

    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        self.warmup = warmup
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> List[float]:
        last_epoch = self.last_epoch + 1  # shifted one step to the left
        if last_epoch < self.warmup:
            # warmup
            scale = last_epoch / self.warmup
            return [lr * scale for lr in self.base_lrs]
        lrs = super().get_lr(last_epoch - self.warmup)
        return lrs
